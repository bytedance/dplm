# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import esm
import torch
from torch import nn


def exists(x):
    return x is not None


class GVPTransformerEncoderWrapper(nn.Module):
    def __init__(self, alphabet=None, freeze=True, return_logits=False):
        super().__init__()
        _model, _alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.alphabet = alphabet or _alphabet
        self.return_logits = return_logits
        self.encoder = _model.encoder

        self.freeze = freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        self.embed_dim = self.encoder.embed_tokens.embedding_dim
        if self.return_logits:
            self.out_proj = nn.Linear(self.embed_dim, len(self.alphabet))

    def forward(self, backb_positions, mask, padding_mask, **kwargs):
        return_all_hiddens = False
        coords = backb_positions[:, :, :3, :]
        # padding_mask = padding_mask.bool()
        # mask = mask.bool() & (~padding_mask)
        # coords = torch.masked_fill(coords, mask[..., None, None], torch.nan)
        # confidence = mask.float()
        padding_mask = ~mask.bool()
        confidence = torch.ones(coords.shape[0:2]).to(coords.device)
        with torch.set_grad_enabled(not self.freeze):
            encoder_out = self.encoder(
                coords, padding_mask, confidence, return_all_hiddens=return_all_hiddens
            )
        # encoder_out['encoder_out'][0] = torch.transpose(encoder_out['encoder_out'][0], 0, 1)
        encoder_out["out"] = encoder_out["encoder_out"][0].transpose(0, 1)

        if self.return_logits:
            logits = self.out_proj(encoder_out["feats"])
            encoder_out["logits"] = logits
        return encoder_out


class GVPTransformerEncoderWrapper2(nn.Module):
    def __init__(self, alphabet=None, freeze=True, return_logits=False):
        super().__init__()
        _model, _alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.alphabet = alphabet or _alphabet
        self.return_logits = return_logits
        self.encoder = _model.encoder

        self.freeze = freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        self.embed_dim = self.encoder.embed_tokens.embedding_dim
        if self.return_logits:
            self.out_proj = nn.Linear(self.embed_dim, len(self.alphabet))

    def forward(self, backb_positions, mask, padding_mask, **kwargs):
        return_all_hiddens = False
        coords = backb_positions[:, :, :3, :]
        padding_mask = padding_mask.bool()
        mask = mask.bool() & (~padding_mask)
        coords = torch.masked_fill(coords, ~mask[..., None, None], torch.nan)
        confidence = mask.float()
        # padding_mask = ~mask.bool()
        # confidence = torch.ones(coords.shape[0:2]).to(coords.device)
        with torch.set_grad_enabled(not self.freeze):
            encoder_out = self.encoder(
                coords, padding_mask, confidence, return_all_hiddens=return_all_hiddens
            )
        # encoder_out['encoder_out'][0] = torch.transpose(encoder_out['encoder_out'][0], 0, 1)
        encoder_out["out"] = encoder_out["encoder_out"][0].transpose(0, 1)

        if self.return_logits:
            logits = self.out_proj(encoder_out["feats"])
            encoder_out["logits"] = logits
        return encoder_out
