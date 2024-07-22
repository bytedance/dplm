
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn
from byprot import utils
from byprot.models import register_model
import math
from byprot.models.lm.model_utils import topk_masking, stochastic_sample_from_categorical
from byprot.models.lm.modules.dplm_adapter import DPLMWithConditionalAdatper, DPLMWithAdapterConfig
import numpy as np
from tqdm import tqdm
    
@dataclass
class GVPTransEncoderConfig:
    output_logits: bool = False
    d_model: int = 512

@dataclass
class ConditionalDPLMConfig:
    encoder: GVPTransEncoderConfig = field(default=GVPTransEncoderConfig())
    decoder: DPLMWithAdapterConfig = field(default=DPLMWithAdapterConfig())
    init_pred_where: bool = True

@register_model('cond_dplm')
class ConditionalDPLM(nn.Module):
    _default_cfg = ConditionalDPLMConfig()

    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.encoder = utils.instantiate_from_config(cfg=cfg.encoder, group='model')
        self.decoder = DPLMWithConditionalAdatper.from_pretrained(cfg=cfg.decoder)

        self._update_cfg(cfg)
        self.pad_id = self.decoder.pad_id
        self.mask_id = self.decoder.mask_id
        self.bos_id = self.decoder.bos_id
        self.eos_id = self.decoder.eos_id
        self.x_id = self.decoder.x_id
        self.init_pred_where = self.cfg.init_pred_where

    def _update_cfg(self, cfg):
        if '_target_' in cfg.encoder:
            cfg.encoder.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    def forward(self, batch, weighting='linear', return_outputs=True, output_encoder_logits=False, **kwargs):
        encoder_logits = None
        if output_encoder_logits:
            encoder_logits, encoder_out = self.encoder(batch, output_logits=True, **kwargs)
        else:
            encoder_out = self.encoder(batch, output_logits=False, **kwargs)
        
        encoder_out['feats'] = encoder_out['feats'].repeat(2, 1, 1).detach()
        
        encoder_out['encoder_attention_mask'] = batch['tokens'].ne(self.pad_id)
        encoder_out['encoder_attention_mask'] = encoder_out['encoder_attention_mask'].repeat(2, 1)
        if encoder_logits is not None:
            init_pred = encoder_logits.argmax(-1)
            if self.init_pred_where:
                # init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])
                init_pred = torch.where(batch['coord_mask'], init_pred, batch['tokens'])

        logits, target, loss_mask, weight = self.decoder.compute_loss(
            batch=batch,
            weighting=weighting,
            tokens=init_pred if encoder_logits is not None else None,
            encoder_out=encoder_out,
            return_outputs=return_outputs,
        )

        return logits, target, loss_mask, weight, encoder_logits.repeat(2, 1, 1) if output_encoder_logits else None

    def forward_encoder(self, batch, use_draft_seq=False):
        encoder_logits = None
        encoder_out = None
        if use_draft_seq:
            encoder_logits, encoder_out = self.encoder(batch, return_feats=True, output_logits=True)
            init_pred = encoder_logits.argmax(-1)
            if self.init_pred_where:
                init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])
            encoder_out['logits'] = encoder_logits
            encoder_out['init_pred'] = init_pred
        else:
            encoder_out = self.encoder(batch, return_feats=True, output_logits=False)
            encoder_out['coord_mask'] = batch['coord_mask']
        #encoder_out['encoder_attention_mask'] = batch['prev_tokens'].ne(self.pad_id)
        encoder_out['encoder_attention_mask'] = batch['motif_mask'] if 'motif_mask' in batch else batch['prev_tokens'].ne(self.pad_id)
        return encoder_out

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
               
    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax'):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        # output_masks = output_tokens.eq(self.mask_id)  # & coord_mask
        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)

        esm_out = self.decoder(
            batch={
                'prev_tokens':output_tokens,
            },
            encoder_out=encoder_out,
            need_head_weights=need_attn_weights
        )
        esm_logits = esm_out['logits']
        attentions = esm_out['attentions'] if need_attn_weights else None

        logits = esm_logits  # + encoder_out['logits']

        logits[..., self.mask_id] = -math.inf
        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf
        
        if sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions, # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=esm_out['last_hidden_state']
        )

    def initialize_output_tokens(self, batch, encoder_out, partial_masks=None, use_draft_seq=False, length_beam=1, mbr=1):
        mask = encoder_out.get('coord_mask', None)

        if use_draft_seq:
            prev_tokens = batch['prev_tokens']
            prev_token_mask = batch['prev_token_mask']
            initial_output_tokens = torch.where(
                prev_token_mask, encoder_out['init_pred'], prev_tokens)
            initial_output_scores = torch.zeros(
                *initial_output_tokens.size(), device=initial_output_tokens.device
            )
        else:
            tokens = batch['prev_tokens']
            if tokens is None:
                raise NotImplementedError
            else:
                assert length_beam == 1 and mbr == 1
                output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

                output_tokens = tokens.masked_fill(output_mask, self.mask_id)
                output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

                # output_tokens = torch.where(output_mask, encoder_out['init_pred'], output_tokens)
                return output_tokens, output_scores

        return initial_output_tokens, initial_output_scores
    
    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        to_be_resample = []
        for i, seq in enumerate(cur_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token == self.pad_id:
                    continue
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * 0.25:
                to_be_resample.append(i)
                
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
            if len(to_be_resample) > 0:
                noise_scale = 1.5
                #print(lowest_k_mask[to_be_resample[0]])
                lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
                                                             stochastic=True, temp=noise_scale * rate)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        #
        # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores
    
    def generate(self, batch, tokenizer=None, 
                 max_iter=None, temperature=None, 
                 partial_masks=None,
                 sampling_strategy='argmax',
                 use_draft_seq=True):
        tokenizer = tokenizer 
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch, use_draft_seq=use_draft_seq)
        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
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

        prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )

        for step in tqdm(range(max_iter), desc='Decoding'):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy
                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=output_tokens.clone(),
                output_scores=output_scores.clone(),
                cur_tokens=prev_decoder_out['output_tokens'].clone(),
                cur_scores=prev_decoder_out['output_scores'].clone(),
                decoding_strategy='reparam-uncond-deterministic-linear',
                # decoding_strategy='reparam-uncond-deterministic-cosine',
                xt_neq_x0=prev_decoder_out['output_masks'],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
            )
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores']