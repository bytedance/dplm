import torch

from typing import List, Dict
# from data.pdb2feature import batch_coords2feature
from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification
# from module.esm.structure_module import (
#     EsmForMaskedLMWithStructure as EsmForMaskedLM,
#     EsmForSequenceClassificationWithStructure as EsmForSequenceClassification,
# )
from ..abstract_model import AbstractModel
from model.dplm.dplm_model import DPLMForSequenceClassification, DiffusionProteinLanguageModel


class DPLMBaseModel(AbstractModel):
    """
    ESM base model. It cannot be used directly but provides model initialization for downstream tasks.
    """

    def __init__(self,
                 task: str,
                #  config_path: str,
                 net_name: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 use_lora: bool = False,
                 lora_config_path: str = None,
                 dropout: float = 0.0,
                 **kwargs):
        """
        Args:
            task: Task name. Must be one of ['classification', 'regression', 'lm', 'base']

            config_path: Path to the config file of huggingface esm model

            extra_config: Extra config for the model

            load_pretrained: Whether to load pretrained weights of base model

            freeze_backbone: Whether to freeze the backbone of the model

            use_lora: Whether to use LoRA on downstream tasks

            lora_config_path: Path to the config file of LoRA. If not None, LoRA model is for inference only.
            Otherwise, LoRA model is for training.

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ['classification', 'regression', 'lm', 'base']
        self.task = task
        # self.config_path = config_path
        self.net_name = net_name
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout
        if use_lora:
            self.dropout = 0.0
            self.lora_dropout = dropout
        super().__init__(**kwargs)

        # After all initialization done, lora technique is applied if needed
        self.use_lora = use_lora
        if use_lora:
            self._init_lora(lora_config_path)

    def _init_lora(self, lora_config_path):
        from peft import (
            PeftModelForSequenceClassification,
            get_peft_model,
            LoraConfig,
        )

        if lora_config_path:
            # Note that the model is for inference only
            self.model = PeftModelForSequenceClassification.from_pretrained(self.model, lora_config_path)
            self.model.merge_and_unload()
            print("LoRA model is initialized for inference.")

        else:
            lora_config = {
                "task_type": "SEQ_CLS",
                "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
                "modules_to_save": ["classifier"],
                "inference_mode": False,
                "lora_dropout": self.lora_dropout,
                "lora_alpha": 8,
                "r": 16,
            }

            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
            # original_module is not needed for training
            self.model.classifier.original_module = None

            print("LoRA model is initialized for training.")
            self.model.print_trainable_parameters()

        # After LoRA model is initialized, add trainable parameters to optimizer
        self.init_optimizers()

    def initialize_model(self):
        # Initialize tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(self.net_name)

        # Initialize different models according to task
        # Add extra config if needed
        if self.extra_config is None:
            self.extra_config = {}

        self.extra_config['hidden_dropout_prob'] = self.dropout
        if self.task == 'classification':
            self.model = DPLMForSequenceClassification(
                self.net_name, num_labels=self.num_labels, **self.extra_config)

        elif self.task == 'regression':
            self.model = DPLMForSequenceClassification(
                self.net_name, num_labels=1, **self.extra_config)

        elif self.task == 'lm':
            self.model = DiffusionProteinLanguageModel.from_pretrained(self.net_name, net_override=self.extra_config)

        elif self.task == 'base':
            self.model = DiffusionProteinLanguageModel.from_pretrained(self.net_name, net_override=self.extra_config)

        # Freeze the backbone of the model
        if self.freeze_backbone:
            for param in self.model.dplm.net.parameters() if hasattr(self.model, "dplm") else self.model.net.parameters():
                param.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}

    def get_hidden_states(self, inputs, reduction: str = None) -> list:
        """
        Get hidden representations of the model.

        Args:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        outputs = self.model(inputs['input_ids'], return_last_hidden_state=True)
        assert len(outputs) == 2
        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)

        repr_list = []
        hidden_states = outputs[1]
        for i, idx in enumerate(indices):
            if reduction == "mean":
                repr = hidden_states[i][1:idx].mean(dim=0)
            else:
                repr = hidden_states[i][1:idx]

            repr_list.append(repr)

        return repr_list

    def save_checkpoint(self, save_path, save_info: dict = None) -> None:
        """
        Rewrite this function for saving LoRA parameters
        """
        if not self.use_lora:
            return super().save_checkpoint(save_path=save_path, save_info=save_info)

        else:
            #FIXME: use huggingface.save_pretrained, self.save_path is dirname, not filename
            self.model.save_pretrained(self.save_path)


