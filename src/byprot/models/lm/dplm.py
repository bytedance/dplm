
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from byprot.models import register_model
from omegaconf import OmegaConf
from byprot.models.lm.model_utils import LoRAConfig, NetConfig, get_net, get_net_class, \
    sample_from_categorical, stochastic_sample_from_categorical, top_k_top_p_filtering, topk_masking
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import os
    
@dataclass
class DPLMConfig:
    num_diffusion_timesteps: int = field(
        default=500
    )
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(
        default=False
    )
    rdm_couple: bool = field(
        default=False
    )

@register_model('dplm')
class DiffusionProteinLanguageModel(nn.Module):
    _default_cfg = DPLMConfig()
    
    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        
        self.net = get_net(self.cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id
        
        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
    
    @classmethod
    def from_pretrained(cls, net_name, cfg_override={}, net_override={}, from_huggingface=True):
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            from byprot.utils.config import load_yaml_config
            from pathlib import Path
            from collections import OrderedDict
            
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, '.hydra', 'config.yaml')
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop('_target_')
            model = cls(cfg)
            
            pretrained_state_dict = torch.load(net_name, map_location=torch.device("cpu"))['state_dict']
            new_pretrained_state_dict = OrderedDict()
            
            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            model.load_state_dict(new_pretrained_state_dict, strict=False) 
            return model
        else:
            # Load DPLM model checkpoint from huggingface
            net_type = AutoConfig.from_pretrained(net_name).model_type
            net_class = get_net_class(net_type)
            net = net_class.from_pretrained(net_name, **net_override)
            return cls(cfg=cfg_override, net=net)
    
    def _update_cfg(self, cfg):
        # if '_target_' in cfg.net:
        #     cfg.net.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id) 

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id) 

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }

    def q_sample(self, x_0, t1, maskable_mask):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }
        
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        outputs = self.net(
            input_ids=input_ids,
        )
        logits = outputs['logits']
        if return_last_hidden_state:
            last_hidden_state = outputs['last_hidden_state']
            return logits, last_hidden_state
        else:
            return logits

    def compute_loss(self, batch, weighting='constant'):
        target = batch['targets']

        # couple
        t1, t2 = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0), ),
            device=target.device
        ).chunk(2)

        if self.cfg.rdm_couple:
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target, t1, t2,
                    maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )
            target = target.repeat(2, 1)
        else:
            x_t, t, loss_mask = list(
                self.q_sample(
                    target, t1,
                    maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )

        logits = self.forward(x_t)

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t)
        }[weighting][:, None].float() / num_timesteps
        
        return logits, target, loss_mask, weight

    def forward_encoder(self, batch, **kwargs):
        return {}

    def initialize_output_tokens(self, batch, partial_masks=None, **kwargs):
        tokens = batch['input_ids']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    def resample_conditional(self, _tokens, _scores, ratio, scale):
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))
            
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            resample_logits = self.net(
                input_ids=resample_input,
            )['logits']
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            resample_logits[..., self.mask_id] = -math.inf
            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf
            
            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores
            
    def forward_decoder(self, prev_decoder_out, encoder_out=None, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax'):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)

        net_out = self.net(
            input_ids=output_tokens,
        )
        
        logits = net_out['logits']
        attentions = net_out['attentions'] if need_attn_weights else None
        
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        logits[..., self.mask_id] = -math.inf
        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf
        
        #logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == 'vanilla':
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0)
        else:
            raise NotImplementedError
        
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
            hidden_states=net_out['last_hidden_state']
        )

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask

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
                 sampling_strategy='gumbel_argmax'):
        tokenizer = tokenizer 
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch, encoder_out=encoder_out, partial_masks=partial_masks)
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
                output_tokens=prev_decoder_out['output_tokens'].clone(),
                output_scores=prev_decoder_out['output_scores'].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear'
                xt_neq_x0=prev_decoder_out['output_masks'],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )
            # output_masks, result_tokens, result_scores = self._reparam_decoding(
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
            #     noise=self.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
            #     mask_811=False
            # )
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