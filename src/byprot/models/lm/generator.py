
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import itertools
import math
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Mapping

import torch
from torch import nn
from tqdm import tqdm
from pprint import pprint


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def exists(obj):
    return obj is not None


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def maybe_remove_batch_dim(tensor):
    if len(tensor.shape) > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor

def refinement_step(
        batch, step, 
        model, alphabet,
        prev_decoder_out, encoder_out, 
        strategy='denoise', max_iter=1, temperature=0
    ):

    def _initialize():
        # 0) encoding
        encoder_out = model.forward_encoder(batch)

        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = model.initialize_output_tokens(
            batch, encoder_out=encoder_out)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )
        if strategy == 'discrete_diffusion':
            prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['prev_tokens'])
        return prev_decoder_out, encoder_out

    if prev_decoder_out is None and encoder_out is None:
        prev_decoder_out, encoder_out = _initialize()

    # core 
    decoder_out = model.forward_decoder(
        prev_decoder_out=prev_decoder_out,
        encoder_out=encoder_out,
        need_attn_weights=False
    )
    output_tokens = decoder_out['output_tokens']
    output_scores = decoder_out['output_scores']

    # 2.2: re-mask skeptical parts of low confidence
    # skeptical decoding (depend on the maximum decoding steps.)
    if (
        strategy == 'mask_predict'
        and (step + 1) < max_iter
    ):
        skeptical_mask = _skeptical_unmasking(
            output_scores=output_scores,
            output_masks=output_tokens.ne(alphabet.padding_idx),  # & coord_mask,
            p=1 - (step + 1) / max_iter
        )

        output_tokens.masked_fill_(skeptical_mask, alphabet.mask_idx)
        output_scores.masked_fill_(skeptical_mask, 0.0)

    elif strategy == 'denoise' or strategy == 'no':
        pass
    elif strategy == 'discrete_diffusion':
        non_special_sym_mask = model.get_non_special_sym_mask(
            prev_decoder_out['output_tokens']
        )
        output_masks = model._reparam_decoding(
            output_tokens=output_tokens,
            output_scores=output_scores,
            cur_tokens=prev_decoder_out['output_tokens'],
            cur_scores=prev_decoder_out['output_scores'],
            decoding_strategy='reparam-uncond-deterministic-linear',
            xt_neq_x0=prev_decoder_out['output_masks'],
            non_special_sym_mask=non_special_sym_mask,
            t=step + 1,
            max_step=max_iter,
            noise=model.mask_id
        )
        prev_decoder_out.update(output_masks=output_masks)
    else:
        pass

    prev_decoder_out.update(
        output_tokens=output_tokens,
        output_scores=output_scores,
        step=step + 1,
        history=decoder_out['history']
    )
    return prev_decoder_out

class IterativeRefinementGenerator(object):
    def __init__(self,
                 alphabet=None,
                 max_iter=1,
                 strategy='denoise',
                 temperature=None,
                 **kwargs
                 ):

        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self.max_iter = max_iter
        self.strategy = strategy
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, model, batch, alphabet=None, 
                 max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
                 partial_masks=None,
                 need_attn_weights=False,
                 print_output=False,
                 use_draft_seq=False,
                 sampling_strategy='gumbel_argmax'):
        alphabet = alphabet or self.alphabet
        max_iter = max_iter or self.max_iter
        strategy = strategy or self.strategy
        temperature = temperature or self.temperature

        # 0) encoding
        encoder_out = model.forward_encoder(batch, use_draft_seq=use_draft_seq)
        print(use_draft_seq)
        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = model.initialize_output_tokens(
            batch, encoder_out=encoder_out, partial_masks=partial_masks, use_draft_seq=use_draft_seq)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        if need_attn_weights:
            attns = [] # list of {'in', 'out', 'attn'} for all iteration

        if strategy == 'discrete_diffusion' or strategy == 'rdm':
            # prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['prev_tokens'])
            # prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['input_mask'])
            prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(
                    prev_decoder_out['output_tokens'], partial_masks=partial_masks
                )#.fill_(False)

        #print(self.alphabet.decode(initial_output_tokens, remove_special=False))
        # iterative refinement
        # max_iter = int(initial_output_tokens.size(1) * 1.5)
        # print('max iter: ', max_iter)
        for step in tqdm(range(max_iter), desc='Decoding', disable=True):
            # 2.1: predict
            decoder_out = model.forward_decoder(
                prev_decoder_out=prev_decoder_out,
                encoder_out=encoder_out,
                need_attn_weights=need_attn_weights,
                partial_masks=partial_masks,
                sampling_strategy=sampling_strategy
            )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']
            # print('============x0============')
            # pprint(self.alphabet.decode(output_tokens[0].unsqueeze(0), remove_special=False))

            # 2.2: re-mask skeptical parts of low confidence
            # skeptical decoding (depend on the maximum decoding steps.)
            if (
                strategy == 'mask_predict' or strategy == 'cmlm'
                and (step + 1) < max_iter
            ):
                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
                    p=1 - (step + 1) / max_iter
                )

                output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
                output_scores.masked_fill_(skeptical_mask, 0.0)

            elif strategy == 'denoise' or strategy == 'no':
                pass
            elif strategy == 'discrete_diffusion' or strategy == 'rdm':
                non_special_sym_mask = model.get_non_special_sym_mask(
                    prev_decoder_out['output_tokens'], partial_masks=partial_masks
                )
                
                output_masks, result_tokens, result_scores = model._reparam_decoding(
                    output_tokens=prev_decoder_out['output_tokens'].clone(),#output_tokens,#
                    output_scores=prev_decoder_out['output_scores'].clone(),#output_scores,##
                    cur_tokens=output_tokens.clone(),#prev_decoder_out['output_tokens'],##
                    cur_scores=output_scores.clone(),#prev_decoder_out['output_scores'],##
                    decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear',#,##
                    # decoding_strategy='reparam-uncond-deterministic-cosine',
                    xt_neq_x0=prev_decoder_out['output_masks'],
                    non_special_sym_mask=non_special_sym_mask,
                    t=step + 1,
                    max_step=max_iter,
                    noise=model.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
                )
                # output_masks, result_tokens, result_scores = model._reparam_decoding(
                #     output_tokens=output_tokens.clone(),#output_tokens,#
                #     output_scores=output_scores.clone(),#output_scores,##
                #     cur_tokens=prev_decoder_out['output_tokens'].clone(),#prev_decoder_out['output_tokens'],##
                #     cur_scores=prev_decoder_out['output_scores'].clone(),#prev_decoder_out['output_scores'],##
                #     decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear',#,##
                #     # decoding_strategy='reparam-uncond-deterministic-cosine',
                #     xt_neq_x0=prev_decoder_out['output_masks'],
                #     non_special_sym_mask=non_special_sym_mask,
                #     t=step + 1,
                #     max_step=max_iter,
                #     noise=model.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
                #     mask_811=False
                # )
                prev_decoder_out.update(output_masks=output_masks)
                output_tokens = result_tokens
                output_scores = result_scores
            else:
                pass
            # print('============input with mask============')
            # pprint(self.alphabet.decode(output_tokens[0].unsqueeze(0), remove_special=False))
            if print_output:
                pprint(self.alphabet.decode(output_tokens, remove_special=False))

            if replace_visible_tokens:
                visible_token_mask = ~batch['prev_token_mask']
                visible_tokens = batch['prev_tokens']
                output_tokens = torch.where(
                    visible_token_mask, visible_tokens, output_tokens)

            if need_attn_weights:
                attns.append(
                    dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
                         output=maybe_remove_batch_dim(output_tokens),
                         attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
                )

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        # skeptical_mask = _skeptical_unmasking(
        #     output_scores=output_scores,
        #     output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
        #     p=0.08
        # )

        # output_tokens.masked_fill_(skeptical_mask, self.alphabet.unk_idx)
        # output_scores.masked_fill_(skeptical_mask, 0.0)
        decoder_out = prev_decoder_out

        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores']


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def stochastic_sample_from_categorical_old(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    #logprobs = logits.log_softmax(dim=-1).div(temperature) 
    logprobs = logits.div(temperature).log_softmax(dim=-1)
    logprobs = logprobs + noise_scale * gumbel_noise
    scores, tokens = logprobs.max(dim=-1)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores