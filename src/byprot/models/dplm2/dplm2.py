# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2.modules.dplm2_modeling_esm import *
from byprot.models.utils import *


def exists(obj):
    return obj is not None


@dataclass
class SelfMixupConfig:
    enable: bool = field(default=False)
    with_original_loss: bool = field(default=False)


@dataclass
class TokenizerConfig:
    vocab_file: str = field(default="airkingbd/dplm2_650m")
    # amino acid tokens (33) + struct tokens (8192) + 4 special struct tokens
    vocab_size: int = field(default=33 + 8192 + 4)


@dataclass
class StructTokenizerConfig:
    enable: bool = field(default=True)
    exp_path: str = field(default="airkingbd/struct_tokenizer")


@dataclass
class DPLM2Config:
    ## DPLM model
    num_diffusion_timesteps: int = field(default=500)
    tokenizer: TokenizerConfig = field(default=TokenizerConfig())
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(default=False)

    ## multi-modal training
    training_stage: str = field(default="train_from_dplm")
    self_mixup: SelfMixupConfig = field(
        default=SelfMixupConfig()
    )  # training strategy
    single_modality_ratio: float = field(default=0.25)
    folding_loss_ratio: float = field(default=0.25)
    inverse_folding_loss_ratio: float = field(default=0.25)
    joint_loss_ratio: float = field(default=0.25)
    independent_loss_ratio: float = field(default=0.0)

    ## struct tokenizer
    struct_tokenizer: StructTokenizerConfig = field(
        default=StructTokenizerConfig()
    )


@register_model("dplm2")
class MultimodalDiffusionProteinLanguageModel(nn.Module):
    _default_cfg = DPLM2Config()

    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._prepare_special_token()
        self.cfg.tokenizer.vocab_size = len(self.tokenizer)
        if net is None:
            self.net = get_net_dplm2(self.cfg)
        else:
            if "bit" in net.config.dplm_type:
                raise ValueError(
                    f"Bit model is not supported in this DPLM-2 class, please use DPLM-2 bit model instead."
                )
            self.net = net

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

        self._struct_tokenizer = None

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    @property
    def special_token_list(self):
        return [
            self.aa_bos_id,
            self.aa_eos_id,
            self.aa_mask_id,
            self.struct_bos_id,
            self.struct_eos_id,
            self.struct_mask_id,
            self.pad_id,
            self.aa_unk_id,
            self.struct_unk_id,
            self.aa_X_id,
            self.aa_B_id,
            self.aa_U_id,
            self.aa_Z_id,
            self.aa_O_id,
        ]

    @classmethod
    def from_pretrained(
        cls, net_name, cfg_override={}, net_override={}, from_huggingface=True
    ):
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            from collections import OrderedDict
            from pathlib import Path

            from byprot.utils.config import load_yaml_config

            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, ".hydra", "config.yaml")
            cfg = load_yaml_config(str(cfg_path))
            OmegaConf.resolve(cfg)
            cfg = cfg.model
            cfg.net.pretrain = False
            cfg.pop("_target_")

            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            missing, unexpected = model.load_state_dict(
                new_pretrained_state_dict, strict=False
            )
            print(
                f"Restored from {net_name} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")
            return model

        else:
            # Load DPLM-2 model checkpoint from huggingface
            dplm_type = AutoConfig.from_pretrained(net_name).dplm_type
            net_class = get_net_class(dplm_type)
            net = net_class.from_pretrained(net_name, **net_override)
            return cls(cfg=cfg_override, net=net)

    def _prepare_special_token(self):
        self.aa_bos_id = self.tokenizer._token_to_id["<cls_aa>"]
        self.aa_eos_id = self.tokenizer._token_to_id["<eos_aa>"]
        self.aa_mask_id = self.tokenizer._token_to_id["<mask_aa>"]
        self.struct_bos_id = self.tokenizer._token_to_id["<cls_struct>"]
        self.struct_eos_id = self.tokenizer._token_to_id["<eos_struct>"]
        self.struct_mask_id = self.tokenizer._token_to_id["<mask_struct>"]
        self.pad_id = self.tokenizer._token_to_id["<pad>"]
        self.aa_unk_id = self.tokenizer._token_to_id["<unk_aa>"]
        self.struct_unk_id = self.tokenizer._token_to_id["<unk_struct>"]

        self.aa_X_id = self.tokenizer._token_to_id["X"]
        self.aa_B_id = self.tokenizer._token_to_id["B"]
        self.aa_U_id = self.tokenizer._token_to_id["U"]
        self.aa_Z_id = self.tokenizer._token_to_id["Z"]
        self.aa_O_id = self.tokenizer._token_to_id["O"]

        self.aa_type = 1
        self.struct_type = 0
        self.pad_type = 2

    @property
    def device(self):
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device("cpu")
        return device

    @property
    def struct_tokenizer(self):
        if not exists(self._struct_tokenizer):
            print(f"Loading struct_tokenizer...")
            self._struct_tokenizer = get_struct_tokenizer(
                self.cfg.struct_tokenizer.exp_path
            ).to(self.device)
        return self._struct_tokenizer

    def q_sample(self, x_0, t, type_ids, maskable_mask):
        aa_position = type_ids == self.aa_type
        struct_position = type_ids == self.struct_type

        # sample x_t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (
            u < (t / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t = x_0.masked_fill(t_mask & aa_position, self.aa_mask_id)
        x_t = x_t.masked_fill(t_mask & struct_position, self.struct_mask_id)

        return x_t, t_mask

    def get_modality_type(self, input_ids):
        input_mask = input_ids.ne(self.pad_id)
        # HACK: all amino acid token id < 33, while all struct token id >= 33
        # 0 stands for struct, 1 stands for aa
        modality_type = ((input_ids < 33) & input_mask).int()
        # 2 stands for padding
        modality_type[~input_mask] = self.pad_type
        return modality_type

    def forward(self, input_ids, **kwargs):
        input_mask = input_ids.ne(self.pad_id)

        type_ids = self.get_modality_type(input_ids)

        L = input_ids.shape[1]
        num_heads = self.net.config.num_attention_heads
        # [B, num_heads, L+2, L+2]
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids.shape
            ).repeat(1, num_heads, L, 1)
        )  # -inf for padding positions, 0 otherwise

        if "single_modality" in kwargs:
            single_modality_index = kwargs["single_modality"]
            struct_attention_bias, aa_attention_bias = attention_bias.chunk(
                2, dim=-2
            )
            struct_attention_bias[
                single_modality_index, :, :, L // 2 :
            ] = -math.inf
            aa_attention_bias[
                single_modality_index, :, :, : L // 2
            ] = -math.inf
            attention_bias = torch.concat(
                [struct_attention_bias, aa_attention_bias], dim=-2
            )

        # [B, L, d_model]
        input_embeds = self.net.esm.embeddings(
            input_ids, attention_mask=input_mask
        )

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            type_ids=type_ids,
        )

        return outputs

    def self_mixup(self, x_t, single_modality_index):
        # 1. first part: masked prediction
        with torch.no_grad():
            model_outputs = self.forward(
                input_ids=x_t, single_modality=single_modality_index
            )
            lm_logits = model_outputs["logits"]
        # 2. mixup: alternate mask with model prediction and gt with masks
        prev_input_ids = x_t
        non_special_sym_mask = self.get_non_special_symbol_mask(prev_input_ids)
        model_pred = torch.where(
            non_special_sym_mask, lm_logits.argmax(dim=-1), prev_input_ids
        )
        mixup_xt, mixup_loss_mask = self.get_mixup_xt(
            input_ids=prev_input_ids,
            model_pred=model_pred,
            non_special_sym_mask=non_special_sym_mask,
        )

        # # 3. second part: denoising + masked prediction
        model_outputs = self.forward(
            input_ids=mixup_xt, single_modality=single_modality_index
        )
        return model_outputs, mixup_loss_mask

    def get_mixup_xt(self, input_ids, model_pred, non_special_sym_mask=None):
        gt_mask = (
            input_ids.ne(self.aa_mask_id)
            & input_ids.ne(self.struct_mask_id)
            & non_special_sym_mask
        )

        type_ids = self.get_modality_type(input_ids)

        mixup_input_ids = model_pred
        # replace gt positions with mask
        mixup_input_ids = mixup_input_ids.masked_fill(
            gt_mask & (type_ids == self.aa_type), self.aa_mask_id
        )
        mixup_input_ids = mixup_input_ids.masked_fill(
            gt_mask & (type_ids == self.struct_type), self.struct_mask_id
        )
        mixup_loss_mask = non_special_sym_mask
        return mixup_input_ids, mixup_loss_mask

    def construct_x_t(self, struct_target, aatype_target):
        bsz = struct_target.size(0)
        # seperately add noise to struct and aa
        struct_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=struct_target.device,
        )
        aatype_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=aatype_target.device,
        )

        assert (
            self.cfg.single_modality_ratio
            + self.cfg.folding_loss_ratio
            + self.cfg.inverse_folding_loss_ratio
            + self.cfg.joint_loss_ratio
            + self.cfg.independent_loss_ratio
            == 1.0
        )

        split_sizes = [
            int(bsz * self.cfg.single_modality_ratio),
            int(bsz * self.cfg.folding_loss_ratio),
            int(bsz * self.cfg.inverse_folding_loss_ratio),
            int(bsz * self.cfg.independent_loss_ratio),
            int(bsz * self.cfg.joint_loss_ratio),
        ]
        split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(struct_target)
        int_index_list = torch.split(rand_index, split_sizes)

        bool_index_list = []
        for int_index in int_index_list:
            bool_index = torch.zeros(bsz, dtype=torch.bool).to(
                struct_target.device
            )
            bool_index[int_index] = True
            bool_index_list.append(bool_index)

        (
            single_modality_index,
            folding_index,
            inverse_folding_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        struct_t = struct_t.masked_fill(inverse_folding_index, 0)
        struct_type_id = self.get_modality_type(struct_target)
        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])
        aa_type_id = self.get_modality_type(aatype_target)
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            aatype_t,
            aa_type_id,
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def compute_loss(self, batch, weighting="linear"):
        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target)
        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
        if self.cfg.self_mixup.enable:
            model_outputs, mixup_loss_mask = self.self_mixup(
                x_t=x_t,
                single_modality_index=single_modality_index,
            )
            (
                struct_noised["mask"],
                aatype_noised["mask"],
            ) = mixup_loss_mask.chunk(2, dim=1)
        else:
            model_outputs = self.forward(
                input_ids=x_t,
                single_modality=single_modality_index,
            )

        struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)
        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_weight = struct_weight.expand(struct_target.size())

        aatype_weight = {
            "linear": (
                num_timesteps - (aatype_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        return (
            {
                "aatype": aatype_logits,
                "struct": struct_logits,
            },  # model pred logits
            {
                "aatype": aatype_target,
                "struct": struct_target,
            },  # training targets
            {  # training loss mask
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {
                "aatype": aatype_weight,
                "struct": struct_weight,
            },  # training loss weight
        )

    def forward_encoder(self, input_tokens, **kwargs):
        return {}

    def initialize_output_tokens(
        self, input_tokens, partial_masks=None, **kwargs
    ):
        type_ids = self.get_modality_type(input_tokens)
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        # fill the aatype part and struct part with specialized mask token
        aa_position = type_ids.eq(self.aa_type) & output_mask
        struct_position = type_ids.eq(self.struct_type) & output_mask
        output_tokens = input_tokens.masked_fill(aa_position, self.aa_mask_id)
        output_tokens = output_tokens.masked_fill(
            struct_position, self.struct_mask_id
        )
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

        return output_tokens, output_scores

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@2.2:1.0",
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )
        net_out = self.forward(input_ids=output_tokens)

        logits = net_out["logits"].log_softmax(dim=-1)
        attentions = net_out["attentions"] if need_attn_weights else None

        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        type_ids = self.get_modality_type(output_tokens)
        aa_position = type_ids.eq(self.aa_type) & output_masks
        struct_position = type_ids.eq(self.struct_type) & output_masks
        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        # HACK: all amino acid token id < 33, while all struct token id >= 33
        logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        logits[indices_struct[0], indices_struct[1], :33] = -math.inf

        logits[..., self.special_token_list] = -math.inf

        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = temperature
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )
            _tokens.masked_scatter_(
                ~output_masks, output_tokens[~output_masks]
            )
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )
        else:
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
        )

    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_symbol_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.aa_bos_id)
            & output_tokens.ne(self.aa_eos_id)
            & output_tokens.ne(self.struct_bos_id)
            & output_tokens.ne(self.struct_eos_id)
        )
        if partial_masks is not None:
            non_special_symbol_mask &= ~partial_masks
        return non_special_symbol_mask

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        type_ids,
        non_special_sym_mask,
        t,
        max_step,
    ):
        def _reparam_process(
            output_tokens,
            output_scores,
            cur_tokens,
            cur_scores,
            xt_neq_x0,
            noise,
            non_special_sym_mask,
        ):
            """This function is used to perform reparameterized decoding.

            output_tokens: [B, N]
            output_scores: [B, N]
            cur_tokens: [B, N]
            cur_scores: [B, N]
            xt_neq_x0: equivalent to not_b_t [B, N]
            non_special_sym_mask: [B, N]
            noise: either [B, N] or scalar (if using the mask noise)
            """

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
                non_special_sym_mask.sum(1, keepdim=True).type_as(
                    output_scores
                )
                * rate
            ).long()
            # set the scores of special symbols to a large value so that they will never be selected
            _scores_for_topk = cur_scores.masked_fill(
                ~non_special_sym_mask, 1000.0
            )

            # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
            if topk_mode.startswith("stochastic"):
                noise_scale = float(topk_mode.replace("stochastic", ""))
                lowest_k_mask = topk_masking(
                    _scores_for_topk,
                    cutoff_len,
                    stochastic=True,
                    temp=noise_scale * rate,
                )
            elif topk_mode == "deterministic":
                lowest_k_mask = topk_masking(
                    _scores_for_topk, cutoff_len, stochastic=False
                )

            elif topk_mode == "positionprior":
                lowest_k_mask_1 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask_2 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask = lowest_k_mask_1 | lowest_k_mask_2
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
                not_v1_t = (
                    (cur_tokens == output_tokens)
                    & (cur_scores < output_scores)
                    & lowest_k_mask
                )
            elif condition == "uncond":
                not_v1_t = lowest_k_mask
            else:
                raise NotImplementedError

            # for b_t = 0, the token is set to noise if it is in the lowest k scores.
            not_v2_t = lowest_k_mask

            last_mask_position = xt_neq_x0

            masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
            if isinstance(noise, torch.Tensor):
                output_tokens.masked_scatter_(
                    masked_to_noise, noise[masked_to_noise]
                )
            elif isinstance(noise, (int, float)):
                output_tokens.masked_fill_(masked_to_noise, noise)
            else:
                raise NotImplementedError(
                    "noise should be either a tensor or a scalar"
                )
            output_scores.masked_fill_(masked_to_noise, -math.inf)

            masked_to_x0 = xt_neq_x0 & ~not_v2_t
            output_tokens.masked_scatter_(
                masked_to_x0, cur_tokens[masked_to_x0]
            )
            output_scores.masked_scatter_(
                masked_to_x0, cur_scores[masked_to_x0]
            )
            assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
            # b_{t} = (b_{t+1} & u_t) | v_t
            # For convenience, save the NOT of b_t for the next iteration
            # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
            #
            # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t (?)
            new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
            assert (new_xt_neq_x0 == not_v2_t).all()
            return new_xt_neq_x0, output_tokens, output_scores

        aa_position = type_ids.eq(self.aa_type) & non_special_sym_mask
        struct_position = type_ids.eq(self.struct_type) & non_special_sym_mask
        new_xt_neq_x0 = xt_neq_x0.clone()
        new_xt_neq_x0_aa = new_xt_neq_x0.fill_(False)
        new_xt_neq_x0_struct = new_xt_neq_x0.fill_(False)
        if aa_position.any():
            new_xt_neq_x0_aa, output_tokens, output_scores = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & aa_position,
                noise=self.aa_mask_id,
                non_special_sym_mask=aa_position,
            )
        if struct_position.any():
            (
                new_xt_neq_x0_struct,
                output_tokens,
                output_scores,
            ) = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & struct_position,
                noise=self.struct_mask_id,
                non_special_sym_mask=struct_position,
            )
        new_xt_neq_x0 = new_xt_neq_x0_aa | new_xt_neq_x0_struct
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=1.0,
        partial_masks=None,
        unmasking_strategy="stochastic1.0",  # [stochastic{temperature}, deterministic]
        sampling_strategy="annealing@2.2:1.0",
    ):
        self.eval()
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(input_tokens)
        # 1) initialized from all mask tokens
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )

        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        for step in tqdm(range(max_iter), desc="Decoding"):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        return {
            "output_tokens": decoder_out["output_tokens"],
        }
