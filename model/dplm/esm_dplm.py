from typing import Optional
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union

from transformers.models.esm.modeling_esm import *
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class EsmForDPLM(EsmForMaskedLM):
    def __init__(self, config, **config_override):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        # config.hidden_dropout_prob = dropout
        for k, v in config_override.items():
            setattr(config, k, v)
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = EsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):
        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result
    
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
    
    def initialize_output_tokens(self, batch, encoder_out, partial_masks=None, **kwargs):
        tokens = batch['prev_tokens']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores