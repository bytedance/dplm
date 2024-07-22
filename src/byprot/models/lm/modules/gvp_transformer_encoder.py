
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn
import esm
from byprot.models import register_model


@register_model('gvp_trans_encoder')
class GVPTransformerEncoderWrapper(nn.Module):
    def __init__(self, freeze=True, output_logits=False, d_model=512):
        super().__init__()
        _model, _alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.encoder = _model.encoder
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        if output_logits:
            self.embed_dim = self.encoder.embed_tokens.embedding_dim
            self.out_proj = nn.Linear(self.embed_dim, len(alphabet))

    def forward(self, batch, output_logits=False, **kwargs):
        return_all_hiddens = False
        padding_mask = torch.isnan(batch['coords'][:, :, 0, 0])
        coords = batch['coords'][:, :, :3, :]
        confidence = torch.ones(batch['coords'].shape[0:2]).to(coords.device)
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        # encoder_out['encoder_out'][0] = torch.transpose(encoder_out['encoder_out'][0], 0, 1)
        encoder_out['feats'] = encoder_out['encoder_out'][0].transpose(0, 1)
        if output_logits:
            logits = self.out_proj(encoder_out['feats'])
            return logits, encoder_out
        else:
            return encoder_out
    
