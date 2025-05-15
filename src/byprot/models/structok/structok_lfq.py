# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import os

import torch
from torch import nn

from byprot.datamodules.pdb_dataset import protein
from byprot.datamodules.pdb_dataset.pdb_datamodule import (
    PdbDataset,
    aatype_to_seq,
    collate_fn,
    seq_to_aatype,
    struct_ids_to_seq,
    struct_seq_to_ids,
)
from byprot.models import register_model

from .modules.ema import LitEma
from .modules.folding_utils.decoder import ESMFoldStructureDecoder as Decoder
from .modules.gvp_encoder import GVPTransformerEncoderWrapper2 as Encoder
from .modules.lfq import LFQ
from .modules.nn import TransformerEncoder


def exists(o):
    return o is not None


@register_model("structok_lfq")
class VQModel(nn.Module):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        codebook_config,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        batch_resize_range=None,
        scheduler_config=None,
        lr_g_factor=1.0,
        remap=None,
        sane_index_shape=True,  # tell vector quantizer to return indices as bhw
        use_ema=False,
    ):
        super().__init__()
        self.codebook_embed_dim = codebook_config.embed_dim
        self.num_codebook = codebook_config.num_codes
        self.image_key = image_key
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.loss = None  # instantiate_from_config(lossconfig)
        self.quantize = LFQ(
            dim=self.codebook_embed_dim,
            codebook_size=self.num_codebook,
            entropy_loss_weight=codebook_config.entropy_loss_weight,
            commitment_loss_weight=codebook_config.commitment_loss_weight,
        )
        # self.pre_quant = torch.nn.Linear(self.encoder.embed_dim, self.codebook_embed_dim)
        self.pre_quant = nn.Sequential(
            nn.LayerNorm(self.encoder.embed_dim),
            nn.Linear(self.encoder.embed_dim, self.codebook_embed_dim),
            nn.ReLU(),
            nn.Linear(self.codebook_embed_dim, self.codebook_embed_dim),
        )
        if codebook_config.get("freeze"):
            self.quantize.requires_grad_(False)
            self.pre_quant.requires_grad_(False)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.post_quant = nn.ModuleDict(
            {
                "mlp": nn.Sequential(
                    nn.LayerNorm(self.codebook_embed_dim),
                    nn.Linear(self.codebook_embed_dim, self.decoder.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder.input_dim, self.decoder.input_dim),
                ),
                "transformer": TransformerEncoder(
                    self.decoder.input_dim, 8, 4
                ),
            }
        )
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(
                f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}."
            )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        self.struct_seq_to_ids = struct_seq_to_ids
        self.struct_ids_to_seq = struct_ids_to_seq
        self.aatype_to_seq = aatype_to_seq
        self.seq_to_aatype = seq_to_aatype

        self.process_chain = PdbDataset.process_chain

    def forward(self, batch, return_pred_indices=True, decoder_kwargs={}):
        pre_quant, encoder_feats = self.encode(
            atom_positions=batch["all_atom_positions"],
            mask=batch["res_mask"],
            seq_length=batch["seq_length"],
            gvp_feat=batch.get("gvp_feat", None),
        )

        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=batch["res_mask"].bool()
        )

        struct_feat = quant

        decoder_out = self.decode(
            quant=struct_feat,  # quant,
            aatype=batch["aatype"],
            mask=batch["res_mask"],
            decoder_kwargs=decoder_kwargs,
        )
        if return_pred_indices:
            return decoder_out, loss, struct_tokens  # , hh
        else:
            return decoder_out, loss

    def encode(self, atom_positions, mask, seq_length=None, gvp_feat=None):
        if exists(gvp_feat):
            encoder_feats = gvp_feat
        else:
            # seq_length = seq_length.squeeze(-1)
            padding_mask = ~(
                torch.arange(mask.shape[1], device=mask.device)[None, :]
                < seq_length[:, None]
            )
            encoder_feats = self.encoder(
                backb_positions=atom_positions,
                mask=mask,
                padding_mask=padding_mask,
            )["out"].detach()

        pre_quant = self.pre_quant(encoder_feats)
        # NOTE: we need to mask out missing positions
        # such that input feature of these position to
        # the quantizer will be guaranteed to be zero therefore
        # resulting in index == 0
        pre_quant = pre_quant * mask[..., None]  # [B, L, C]
        return pre_quant, encoder_feats

    def decode(self, quant, aatype, mask, decoder_kwargs={}):
        def _post_quant(x, mask):
            x = self.post_quant["mlp"](x)
            x = self.post_quant["transformer"](x, padding_mask=1 - mask)["out"]
            return x

        quant = _post_quant(quant, mask)
        decoder_out = self.decoder(
            emb_s=quant,
            emb_z=None,
            mask=mask,
            aa=aatype,
            esmaa=aatype,
            **decoder_kwargs,
        )
        return decoder_out

    def quantize_and_decode(
        self, pre_quant, mask=None, aatype=None, decoder_kwargs={}
    ):
        if not exists(mask):
            mask = torch.ones(
                *pre_quant.shape[:2],
                dtype=torch.float32,
                device=pre_quant.device,
            )
        aatype = torch.zeros_like(mask, dtype=torch.int64)

        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=mask.bool()
        )
        decoder_out = self.decode(quant, aatype, mask, decoder_kwargs)
        return decoder_out, struct_tokens

    def get_decoder_features(self, struct_tokens, res_mask, unk_mask):
        # use 0 as unk/mask id
        struct_tokens = struct_tokens.masked_fill(unk_mask, 0)
        quant = self.quantize.get_codebook_entry(struct_tokens)
        res_mask = res_mask.float()
        quant = self._post_quant(quant, res_mask)

        _aatypes = torch.zeros_like(struct_tokens, dtype=torch.int64)
        decoder_out = self.decoder(
            emb_s=quant,
            emb_z=None,
            mask=res_mask,
            aa=_aatypes,
            esmaa=_aatypes,
            return_features_only=True,
        )
        single_feats, pair_feats = decoder_out["s_s"], decoder_out["s_z"]
        return single_feats, pair_feats

    def tokenize(self, atom_positions, res_mask, seq_length=None):
        pre_quant, _ = self.encode(
            atom_positions=atom_positions,
            mask=res_mask,
            seq_length=seq_length,
        )
        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=res_mask.bool()
        )
        return struct_tokens

    def detokenize(self, struct_tokens, res_mask=None, **kwargs):
        if struct_tokens.ndim == 2:
            quant = self.quantize.get_codebook_entry(struct_tokens)
        elif struct_tokens.ndim == 3:
            quant = struct_tokens
        else:
            raise ValueError(
                f"Invalid struct_tokens shape: {struct_tokens.shape}"
            )

        device = struct_tokens.device

        if not exists(res_mask):
            res_mask = torch.ones(
                struct_tokens.shape[:2], dtype=torch.float32, device=device
            )
        _aatypes = torch.zeros(
            struct_tokens.shape[:2], dtype=torch.int64, device=device
        )

        decoder_out = self.decode(
            quant=quant, aatype=_aatypes, mask=res_mask, decoder_kwargs=kwargs
        )
        decoder_out = dict(
            atom37_positions=decoder_out["final_atom_positions"],
            atom37_mask=decoder_out["atom37_atom_exists"],
            aatype=decoder_out["lm_logits"].argmax(-1),
            residue_index=decoder_out["residue_index"],
            plddt=decoder_out["plddt"],
        )
        return decoder_out

    def string_to_tensor(self, aatype_str, struct_token_str):
        device = next(self.parameters()).device
        aatype = torch.tensor([self.seq_to_aatype(aatype_str)], device=device)
        struct_tokens = torch.tensor(
            [self.struct_seq_to_ids(struct_token_str)], device=device
        )
        return aatype, struct_tokens

    def init_data(self, raw_batch):
        return collate_fn(raw_batch)

    def output_to_pdb(self, decoder_out, output_dir, is_trajectory=False):
        decoder_out = {
            kk: vv for kk, vv in decoder_out.items() if not kk == "sm"
        }
        headers = decoder_out.pop("header")

        pdb_strings = self.decoder.output_to_pdb(decoder_out)

        if is_trajectory:
            header = headers[0][: headers[0].index("_t")]
            saveto = os.path.join(output_dir, f"{header}.pdb")
            with open(saveto, "w") as f:
                for t, pdb_string in enumerate(pdb_strings):
                    prot = protein.from_pdb_string(pdb_string)
                    pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                    f.write(pdb_prot)
        else:
            for header, pdb_string in zip(headers, pdb_strings):
                saveto = os.path.join(output_dir, f"{header}.pdb")
                with open(saveto, "w") as f:
                    f.write(pdb_string)
