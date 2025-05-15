# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import reduce

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2.dplm2 import DPLM2Config
from byprot.models.dplm2.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
from byprot.models.dplm2.modules.dplm2_modeling_esm import *
from byprot.models.utils import *


@dataclass
class BitConfig:
    load_from_pretrained: bool = field(default=False)
    load_path: str = field(default="")
    # quantized feature is a 13-dimensional vector, resulting in 2^13 struct tokens
    codebook_embed_dim: int = field(default=13)


@dataclass
class DPLM2BitConfig(DPLM2Config):
    ## bit dplm2 config
    bit: BitConfig = field(default=BitConfig())


@register_model("dplm2_bit")
class DPLM2Bit(DPLM2):
    _default_cfg = DPLM2BitConfig()

    def __init__(self, cfg, net=None):
        nn.Module.__init__(self)
        self._update_cfg(cfg)
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._struct_tokenizer = None
        # binary classification for each dimension of quant feature
        self.cfg.bit.codebook_embed_dim = (
            self.struct_tokenizer.codebook_embed_dim
        )
        if net is None:
            self.net = get_net_dplm2_bit(self.cfg)
        else:
            if "bit" not in net.config.dplm_type:
                raise ValueError(
                    f"The loaded net is not a bit model, which can not be loaded by DPLM2Bit."
                )
            self.net = net
        self._prepare_special_token()

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

        if self.cfg.bit.load_from_pretrained:
            pretrained_state_dict = torch.load(
                self.cfg.bit.load_path, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            self.load_state_dict(new_pretrained_state_dict, strict=True)
            print(
                f"Successfully load pretrained dplm2 bit model from {self.cfg.bit.load_path}!"
            )

    def _prepare_special_token(self):
        super()._prepare_special_token()
        # HACK: struct tokens and amino acid tokens are in the same vocabulary,
        # there are 33 amino acid tokens, 3 special struct tokens (bos, eos, unk)
        # so the first index of normal struct token is 36
        self.struct_vocab_offset = 36

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

        ######## construct the input embedding
        # [B, L, d_model]
        input_struct_ids, input_aatype_ids = input_ids.chunk(2, dim=1)
        input_struct_mask, input_aatype_mask = input_mask.chunk(2, dim=1)
        input_aatype_embeds = self.net.esm.embeddings(
            input_aatype_ids, attention_mask=input_aatype_mask
        )
        input_struct_embeds = torch.zeros_like(input_aatype_embeds)
        quant = self.struct_tokenizer.quantize.get_codebook_entry(
            input_struct_ids - self.struct_vocab_offset
        )

        input_struct_embeds = self.net.quant2emb(quant).float()
        input_struct_embeds[:, 0] = self.net.struct_bos_emb
        eos_position = input_struct_ids == self.struct_eos_id
        input_struct_embeds[eos_position] = self.net.struct_eos_emb
        mask_position = input_struct_ids == self.struct_mask_id
        input_struct_embeds[mask_position] = self.net.struct_mask_emb

        input_embeds = torch.concat(
            [input_struct_embeds, input_aatype_embeds], dim=1
        )

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            output_hidden_states=True,
            type_ids=type_ids,
        )

        return outputs

    def construct_x_t(self, batch, struct_target, aatype_target):
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
        struct_x_t, struct_t, struct_loss_mask = list(
            self.q_sample(
                struct_target,
                struct_t,
                struct_type_id,
                maskable_mask=self.get_non_special_symbol_mask(struct_target),
                plddt_mask=batch["plddt_mask"]
                if "plddt_mask" in batch
                else None,
            ).values()
        )
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])
        aa_type_id = self.get_modality_type(aatype_target)
        aatype_x_t, aatype_t, aa_loss_mask = list(
            self.q_sample(
                aatype_target,
                aatype_t,
                aa_type_id,
                maskable_mask=self.get_non_special_symbol_mask(aatype_target),
                plddt_mask=batch["plddt_mask"]
                if "plddt_mask" in batch
                else None,
            ).values()
        )

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def compute_loss(self, batch, weighting="linear"):
        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        bsz, seq_len = struct_target.shape

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(batch, struct_target, aatype_target)
        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
        if self.cfg.self_mixup.enable:
            # 1. first part: masked prediction
            with torch.no_grad():
                model_outputs = self.forward(
                    input_ids=x_t, single_modality=single_modality_index
                )
                lm_logits = model_outputs["logits"]
            # 2. mixup: alternate mask with model prediction and gt with masks
            prev_input_ids = x_t
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_input_ids
            )
            model_pred = torch.where(
                non_special_sym_mask, lm_logits.argmax(dim=-1), prev_input_ids
            )
            mixup_input_ids = self._self_mixup(
                input_ids=prev_input_ids,
                model_pred=model_pred,
                non_special_sym_mask=non_special_sym_mask,
            )

            # # 3. second part: denoising + masked prediction
            model_outputs = self.forward(
                input_ids=mixup_input_ids,
                single_modality=single_modality_index,
            )
            aatype_logits = model_outputs["aatype_logits"]
            struct_logits = model_outputs["struct_logits"].reshape(
                bsz, seq_len, -1, 2
            )
        else:
            model_outputs = self.forward(
                input_ids=x_t,
                single_modality=single_modality_index,
            )
            aatype_logits = model_outputs["aatype_logits"]
            struct_logits = model_outputs["struct_logits"].reshape(
                bsz, seq_len, -1, 2
            )

        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_target = (
            self.struct_tokenizer.quantize.get_codebook_entry(
                struct_target - self.struct_vocab_offset
            )
            > 0
        ).long()
        assert struct_target.shape == struct_logits.shape[:3]
        struct_weight = struct_weight[:, :, None].expand(struct_target.size())

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

    def _self_mixup(self, input_ids, model_pred, non_special_sym_mask=None):
        replace_mask = input_ids.eq(self.aa_mask_id) | input_ids.eq(
            self.struct_mask_id
        )

        mixup_input_ids = torch.where(replace_mask, model_pred, input_ids)
        return mixup_input_ids

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@1.1:0.1",
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

        aatype_logits = net_out["aatype_logits"]
        struct_logits = net_out["struct_logits"]
        attentions = net_out["attentions"] if need_attn_weights else None

        if aatype_logits.dtype != output_scores.dtype:
            aatype_logits = aatype_logits.type_as(output_scores)
            struct_logits = struct_logits.type_as(output_scores)

        bsz, seq_len = aatype_logits.shape[:2]
        aatype_logits[:, :, :4] = -math.inf
        aatype_logits[:, :, 24:] = -math.inf
        struct_logits = struct_logits.reshape(bsz, seq_len, -1, 2)

        aatype_logits = top_k_top_p_filtering(aatype_logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=0.0
            )
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=temperature
            )
        else:
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,  # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
            all_hidden_states=net_out["all_hidden_states"],
        )

    def sample_from_logits(
        self, aatype_logits, struct_logits, temperature=1.0
    ):
        _aatype_tokens, _aatype_scores = sample_from_categorical(
            aatype_logits, temperature=temperature
        )
        _struct_bits, _struct_scores = sample_from_categorical(
            struct_logits, temperature=temperature
        )
        _struct_tokens = reduce(
            _struct_bits * self.struct_tokenizer.quantize.mask.int(),
            "b n c -> b n",
            "sum",
        )
        _struct_scores = _struct_scores.sum(dim=-1)
        # IMPORTANT: add struct_vocab_offset to _struct_tokens
        _struct_tokens += self.struct_vocab_offset
        assert _struct_tokens.shape == _struct_scores.shape
        _tokens = torch.concat([_struct_tokens, _aatype_tokens], dim=1)
        _scores = torch.concat([_struct_scores, _aatype_scores], dim=1)
        return _tokens, _scores

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        unmasking_strategy="stochastic1.0",  # [stochastic{temperature}, deterministic]
        sampling_strategy="annealing@1.1:0.1",
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
                all_hidden_states=decoder_out["all_hidden_states"],
            )

        decoder_out = prev_decoder_out

        decoder_out = self.prepare_for_struct_tokenizer(
            decoder_out, non_special_sym_mask
        )
        return {
            "output_tokens": decoder_out["output_tokens"],
            "res_mask": decoder_out["res_mask"],
            "final_struct_feature": decoder_out["final_struct_feature"],
        }

    def prepare_for_struct_tokenizer(self, decoder_out, non_special_sym_mask):
        lm_output_struct_tokens = decoder_out["output_tokens"].chunk(2, dim=1)[
            0
        ]
        non_bos_eos_mask = lm_output_struct_tokens.ne(
            self.struct_eos_id
        ) & lm_output_struct_tokens.ne(self.struct_bos_id)
        bsz, max_len = non_bos_eos_mask.shape

        res_mask = (
            non_special_sym_mask.chunk(2, dim=1)[0][non_bos_eos_mask]
            .view(bsz, max_len - 2)
            .int()
        )
        struct_tokens = (
            lm_output_struct_tokens[non_bos_eos_mask].view(bsz, max_len - 2)
            - self.struct_vocab_offset
        )
        struct_tokens[~res_mask.bool()] = 0
        quant = self.struct_tokenizer.quantize.get_codebook_entry(
            struct_tokens
        )
        decoder_out["res_mask"] = res_mask
        decoder_out["final_struct_feature"] = quant

        return decoder_out
