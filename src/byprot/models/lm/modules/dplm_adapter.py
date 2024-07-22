
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from byprot import utils
from byprot.models.lm.model_utils import LoRAConfig, NetConfig, get_net
from byprot.models.lm.dplm import DiffusionProteinLanguageModel
from transformers.models.esm.modeling_esm import EsmAttention, EsmIntermediate, \
                        EsmOutput, EsmSelfOutput, EsmSelfAttention
from transformers import AutoConfig
from copy import deepcopy

logger = utils.get_logger(__name__)

@dataclass
class DPLMWithAdapterConfig:
    num_diffusion_timesteps: int = field(
        default=100
    )
    adapter_dropout: float = field(
        default=0.1
    )
    encoder_d_model: int = field(
        default=512
    )
    dplm_name: str = field(
        default=""
    )
    net: NetConfig = field(default=NetConfig())
    
    
class DPLMWithConditionalAdatper(nn.Module):
    _default_cfg = DPLMWithAdapterConfig()
    
    @classmethod
    def from_pretrained(cls, cfg):
        net = DiffusionProteinLanguageModel.from_pretrained(cfg.dplm_name).net
        
        # change net.last_layer to AdapterLayer
        # by default based on the esm model
        adapter = AdapterLayer(cfg, deepcopy(net.config))
        net_last_layer = net.esm.encoder.layer[-1]   
        adapter.load_state_dict(net_last_layer.state_dict(), strict=False)  
        net.esm.encoder.layer[-1] = adapter
        del net_last_layer
        
        dplm_adapter = cls(cfg, net) 
        
        for pname, param in dplm_adapter.named_parameters():
            if 'adapter' not in pname:
                param.requires_grad = False
        return dplm_adapter
    
    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        
        self.net = get_net(cfg) if net is None else net
        self.tokenizer = self.net.tokenizer
        
        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

    def forward(self, batch, encoder_out=None, tokens=None, 
                loss_mask=None, forward_diffusion=False, 
                **kwargs):
        encoder_hidden_states = encoder_out['feats']

        encoder_attention_mask = encoder_out['encoder_attention_mask'] if 'encoder_attention_mask' in encoder_out else batch['prev_tokens'].ne(self.pad_id)
        outputs = self.net(
            input_ids=batch['prev_tokens'],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs

    def compute_loss(self, batch, weighting='constant', encoder_out=None, tokens=None, label_smoothing=False, return_outputs=False,):
        target = batch['tokens'] if tokens is None else tokens
        partial_masks = torch.zeros_like(target).bool()

        # couple
        t1, t2 = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0), ),
            device=target.device
        ).chunk(2)

        x_t, t, loss_mask = list(
            self.q_sample_coupled(
                target, t1, t2,
                maskable_mask=self.get_non_special_sym_mask(target, partial_masks)
            ).values()
        )
        target = target.repeat(2, 1)

        batch['prev_tokens'] = x_t  
        logits = self.forward(batch, encoder_out=encoder_out,
                              loss_mask=loss_mask, forward_diffusion=True)['logits']

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t)
        }[weighting][:, None].float() / num_timesteps
        weight = weight.expand(loss_mask.size())

        return logits, batch['tokens'].repeat(2, 1), loss_mask, weight
    
    def _update_cfg(self, cfg):
        # if '_target_' in cfg.denoiser:
        #     cfg.denoiser.pop('_target_')
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
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask


class AdapterLayer(nn.Module):
    def __init__(self, cfg, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        
        kdim = vdim = getattr(cfg, 'encoder_d_model', 512)
        config.hidden_dropout_prob = getattr(cfg, 'adapter_dropout', 0.0)
        self.adapter_crossattention = ModifiedEsmAttention(config, kdim=kdim, vdim=vdim)
        # config.intermediate_size = config.hidden_size // 2 # Notes: bottleneck ffn
        self.adapter_intermediate = EsmIntermediate(config)
        self.adapter_output = EsmOutput(config)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.adapter_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)
        
        # Adapter
        residual = layer_output
        # match the dimension of layer_output 
        # encoder_hidden_states_proj = self.adapter_proj(encoder_hidden_states)
        # FIXME: compute encoder_attention_mask
        dtype = torch.float32
        extended_encoder_attention_mask = encoder_attention_mask[:, None, None, :]
        extended_encoder_attention_mask = extended_encoder_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_encoder_attention_mask = (1.0 - extended_encoder_attention_mask) * torch.finfo(dtype).min
        
        # print(extended_encoder_attention_mask)
        # print(attention_mask)
        # assert (extended_encoder_attention_mask == attention_mask).all()
        # extended_encoder_attention_mask = attention_mask
        cross_attention_outputs = self.adapter_crossattention(
            hidden_states=layer_output,
            encoder_hidden_states=encoder_hidden_states, #encoder_hidden_states_proj,
            #encoder_attention_mask=attention_mask #if not attention_mask.any() else None,#encoder_attention_mask,
            encoder_attention_mask=extended_encoder_attention_mask,# attention_mask, #
        )
        cross_attention_output = cross_attention_outputs[0]
        ffn_output = self.adapter_feed_forward_chunk(cross_attention_output)
        ffn_output += residual

        outputs = (ffn_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def adapter_feed_forward_chunk(self, attention_output):
        attention_output_ln = self.adapter_LayerNorm(attention_output)
        intermediate_output = self.adapter_intermediate(attention_output_ln)
        layer_output = self.adapter_output(intermediate_output, attention_output)
        return layer_output
    

class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None, 
                 kdim=None,
                 vdim=None):
        super().__init__(config, position_embedding_type)
        if kdim is not None:
            self.key = nn.Linear(kdim, self.all_head_size)
        if vdim is not None:
            self.value = nn.Linear(vdim, self.all_head_size)

class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config, kdim=None, vdim=None):
        super().__init__(config)
        self.self = ModifiedEsmSelfAttention(config, kdim=kdim, vdim=vdim)