from model.dplm.esm_dplm import EsmForDPLM
from dataclasses import dataclass, field
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
import torch
try:
    from peft import get_peft_model, LoraConfig, TaskType
except:
    pass


@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    config_override: dict = field(default_factory=dict)

@dataclass
class LoRAConfig:
    lora: bool = field(
        default=False
    )
    lora_rank: int = field(
        default=16
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_target_module: str = field(
        default=""
    )
    modules_to_save: str = field(
        default=""
    )

def get_net_class(arch_type):
    if arch_type == 'esm':
        return EsmForDPLM
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError
    
def get_net(cfg):
    if cfg.net.arch_type == 'esm':
        config = AutoConfig.from_pretrained(f'{cfg.net.name}')
        net = EsmForDPLM(config, **cfg.net.config_override)
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError
    
    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(',')

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False, r=cfg.lora.lora_rank, lora_alpha=32, lora_dropout=cfg.lora.lora_dropout
        )
        net = get_peft_model(net, peft_config)
            
    return net