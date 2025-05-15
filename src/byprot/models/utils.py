# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
import importlib
from huggingface_hub import snapshot_download
from dataclasses import dataclass, field
from byprot.utils import load_yaml_config
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import os

try:
    from peft import get_peft_model, LoraConfig, TaskType
    from peft.peft_model import PeftModel
except:
    pass


@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""


@dataclass
class LoRAConfig:
    lora: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_target_module: str = field(default="")
    modules_to_save: str = field(default="")


def get_net_class(dplm_type):
    from byprot.models import MODEL_REGISTRY
    net_class = MODEL_REGISTRY.get(dplm_type, None)
    if net_class is None:
        raise ValueError(f"Invalid architecture: {dplm_type}.")
    return net_class


def get_net(cfg):
    if cfg.net.arch_type == "esm":
        from byprot.models.dplm.modules.dplm_modeling_esm import EsmForDPLM
        
        config = AutoConfig.from_pretrained(f"{cfg.net.name}")
        net = EsmForDPLM(config, dropout=cfg.net.dropout)
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError

    # 2-stage training (please refer to our paper for more details.)
    ## stage 1: pretrain a masked language model (MLM) from scratch
    ## stage 2: continue pretrain a diffusion language model based on the pretrained MLM
    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if is_local:
            # load your pretrained model from local
            # state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')['state_dict']
            # net.load_state_dict(state_dict, strict=True)
            pretrained_state_dict = torch.load(
                pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            from collections import OrderedDict

            new_pretrained_state_dict = OrderedDict()
            # remove the module prefix "model.net."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[10:]] = v
            net.load_state_dict(new_pretrained_state_dict, strict=True)
        else:
            # or you can load a pretrained model from huggingface
            ptrn_net = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            net.load_state_dict(ptrn_net.state_dict(), strict=True)
            del ptrn_net

    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=32,
            lora_dropout=cfg.lora.lora_dropout,
        )
        net = get_peft_model(net, peft_config)

    return net


def get_net_dplm2(cfg):
    training_stage = getattr(cfg, "training_stage", "train_from_dplm")

    # dplm2 initialize from a pretrained dplm model
    if cfg.net.arch_type == "esm":
        from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2
        
        config = AutoConfig.from_pretrained(f"{cfg.net.name}")

        # training_state == "train_from_dplm" means initializing from a pretrained sequence-based DPLM,
        # whose vocab_size is 33 containing the standerd amino acid and special tokens
        # (https://huggingface.co/airkingbd/dplm_650m/blob/main/vocab.txt).
        if training_stage == "train_from_dplm" and cfg.net.pretrain:
            net = EsmForDPLM2(config, dropout=cfg.net.dropout, vocab_size=33)

        # training_state == "continue_train_from_dplm2" means continue training from a pretrained DPLM-2,
        # whose vocabulary contains amino acid and struct tokens,
        # and the vocab_size should be 33 + number of struct tokens and special tokens (e.g., 33 + 8192 + 4)
        elif training_stage == "continue_train_from_dplm2" or not cfg.net.pretrain:
            net = EsmForDPLM2(
                config,
                dropout=cfg.net.dropout,
                vocab_size=getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4),
            )

        else:
            raise NotImplementedError
    # TODO: dplm2 will support more architectures, such as Llama
    else:
        raise NotImplementedError

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if training_stage == "train_from_dplm":
            from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

            pretrained_state_dict = DiffusionProteinLanguageModel.from_pretrained(
                pretrained_model_name_or_path
            ).net.state_dict()
            net.load_state_dict(pretrained_state_dict, strict=True)

            # expand the embedding weights
            # # initialize the new embedding with the mean and variance of pretrained embeddings.
            net.resize_token_embeddings(getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4))

            pretrained_bias = net.lm_head.bias
            net.lm_head.bias = nn.Parameter(torch.zeros(getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4)))
            net.lm_head.bias.data[:33] = pretrained_bias.data[:33]
        elif training_stage == "continue_train_from_dplm2":
            assert is_local
            from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel

            pretrained_net = MultimodalDiffusionProteinLanguageModel.from_pretrained(
                pretrained_model_name_or_path, from_huggingface=False
            ).net
            if issubclass(type(pretrained_net), PeftModel):
                pretrained_net = pretrained_net.merge_and_unload()
            pretrained_state_dict = pretrained_net.state_dict()
            net.load_state_dict(pretrained_state_dict, strict=True)
        else:
            raise ValueError(f"Invalid training stage {training_stage}.")

        del pretrained_state_dict

    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=32,
            lora_dropout=cfg.lora.lora_dropout,
        )
        net = get_peft_model(net, peft_config)

    return net


def get_net_dplm2_bit(cfg):
    # dplm2 initialize from a pretrained dplm model
    if cfg.net.arch_type == 'esm':
        from byprot.models.dplm2.modules.dplm2_bit_modeling_esm import EsmForDPLM2Bit
        
        config = AutoConfig.from_pretrained(f'{cfg.net.name}')
        net = EsmForDPLM2Bit(config, dropout=cfg.net.dropout, codebook_embed_dim=getattr(cfg.bit, "codebook_embed_dim", 13))
    # TODO: dplm2 will support more architectures, such as Llama
    else:
        raise NotImplementedError
    
    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        from byprot.models.dplm import DiffusionProteinLanguageModel
        pretrained_state_dict = DiffusionProteinLanguageModel.from_pretrained(pretrained_model_name_or_path).net.state_dict()
        net.load_state_dict(pretrained_state_dict, strict=False)   
              
        del pretrained_state_dict
        
    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module.split(',')
        modules_to_save = cfg.lora.modules_to_save.split(',')

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False, r=cfg.lora.lora_rank, lora_alpha=32, lora_dropout=cfg.lora.lora_dropout
        )
        net = get_peft_model(net, peft_config)
        
    return net


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_masking_prior(scores, cutoff_len, stochastic=False, temp=1.0, prior_mask=None):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)  # + torch.tensor(1e-10)
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def mask_fill_811(inputs, masked_indices, mask_id):
    prev_tokens = inputs.clone()
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.8)).bool()
        & masked_indices
    )
    prev_tokens[indices_replaced] = mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(4, 24, prev_tokens.shape).type_as(prev_tokens)
    prev_tokens[indices_random] = random_words[indices_random]

    return prev_tokens


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def top_k_top_p_filtering(logits, top_k=0, top_p=0.95, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape)
    return logits


def get_struct_tokenizer(
    model_name_or_path="airkingbd/struct_tokenizer", eval_mode=True
):
    from byprot.models.structok.structok_lfq import VQModel

    if os.path.exists(model_name_or_path):
        root_path = f"{model_name_or_path}/.hydra"
    else:
        root_path = Path(snapshot_download(repo_id=model_name_or_path))
    cfg = load_yaml_config(f"{root_path}/config.yaml")
    stok = VQModel(**cfg)
    pretrained_state_dict = torch.load(
        f"{root_path}/dplm2_struct_tokenizer.ckpt", map_location=torch.device("cpu")
    )
    missing, unexpected = stok.load_state_dict(pretrained_state_dict, strict=False)
    print(
        f'Restored from \"{model_name_or_path}\" with {len(missing)} missing and {len(unexpected)} unexpected keys'
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
    stok = stok.requires_grad_(False)
    return stok.train(not eval_mode)
