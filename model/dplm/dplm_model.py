import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from model.dplm.model_utils import LoRAConfig, NetConfig, get_net, get_net_class
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

class DiffusionProteinLanguageModel(nn.Module):
    _default_cfg = DPLMConfig()
    
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
        
        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
    
    def _update_cfg(self, cfg):
        # if '_target_' in cfg.net:
        #     cfg.net.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    @classmethod
    def from_pretrained(cls, net_name, cfg_override={}, net_override={}):
        # if os.path.exists(net_name):
        #     # Load model checkpoint from local if you pretrain a DPLM by yourself
        #     # The net_name should be like:
        #     # ${name}/checkpoints/last.ckpt
        #     def load_yaml_config(fpath: str) -> OmegaConf:
        #         return OmegaConf.load(fpath)
            
        #     from pathlib import Path
        #     from collections import OrderedDict
            
        #     cfg_path = Path(net_name).parents[1]
        #     cfg_path = Path(cfg_path, '.hydra', 'config.yaml')
        #     cfg = load_yaml_config(str(cfg_path)).model
        #     cfg.pop('_target_')
        #     model = cls(cfg)
            
        #     pretrained_state_dict = torch.load(net_name, map_location=torch.device("cpu"))['module']
        #     new_pretrained_state_dict = OrderedDict()
            
        #     for k, v in pretrained_state_dict.items():
        #         new_pretrained_state_dict[k[6:]] = v
        #     model.load_state_dict(new_pretrained_state_dict, strict=False) 
        #     return model
        # else:
        net_type = AutoConfig.from_pretrained(net_name).model_type
        net_class = get_net_class(net_type)
        net = net_class.from_pretrained(net_name, **net_override)
        return cls(cfg=cfg_override, net=net)
        
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


class ClassificationHead(nn.Module):
    """Inspired by HuggingFace ESM implementation. Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class DPLMForSequenceClassification(nn.Module):
    def __init__(self, net_name, num_labels, **extra_config):
        super().__init__()
        self.dplm = DiffusionProteinLanguageModel.from_pretrained(net_name=net_name, net_override=extra_config)
        self.classifier = ClassificationHead(
            hidden_size=self.dplm.net.config.hidden_size,
            hidden_dropout_prob=self.dplm.net.config.hidden_dropout_prob,
            num_labels=num_labels
        )
                
    def forward(self, input_ids):
        dplm_last_hidden_state = self.dplm(input_ids, return_last_hidden_state=True)[1]
        logits = self.classifier(dplm_last_hidden_state)
        return logits
        
        
        
        