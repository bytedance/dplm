import copy
import os
from typing import Any, Callable, List, Union

import numpy as np
import torch
from byprot import utils
from byprot.modules import metrics
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg, merge_config
from lightning.pytorch.utilities import grad_norm

from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
from byprot.models.lm.model_utils import get_net

log = utils.get_logger(__name__)

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('lm/mlm')
class MLMTrainingTask(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='random_mask',  # ['full_mask', 'random_mask']
            num_unroll=0,
            mlm_prob=0.15
        ),
        generator=Cfg(
            max_iter=1,
            temperature=0,
        )
    )
    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
    
        self.build_model() 
        self.tokenizer = self.model.tokenizer

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')
        elif self.stage == 'test':
            self.test_step_outputs = []

    def on_before_optimizer_step(self, optimizer):
        if self.global_rank == 0:
            grad_norm_dict = grad_norm(self.trainer.strategy.model, norm_type=2)
            self.log_dict(grad_norm_dict)

    def build_model(self):
        wxy=1
        self.model = get_net(cfg=self.hparams.model)

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()
        
        self.acc = MeanMetric()
        self.acc_best = MaxMetric()
    
    @torch.no_grad()
    def inject_noise(self, tokens):
        padding_idx = self.tokenizer.pad_token_id
        mask_idx = self.tokenizer.mask_token_id

        def _mlm_mask(inputs):
            prev_tokens = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.hparams.learning.mlm_prob).to(inputs.device)
            special_tokens_mask = (
                prev_tokens.eq(padding_idx)  # & mask
                & prev_tokens.eq(self.tokenizer.cls_token_id)
                & prev_tokens.eq(self.tokenizer.eos_token_id)
            )

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full_like(probability_matrix, 0.8)).bool() & masked_indices
            prev_tokens[indices_replaced] = mask_idx

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full_like(probability_matrix, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape).type_as(prev_tokens)
            prev_tokens[indices_random] = random_words[indices_random]
            
            return prev_tokens, masked_indices

        prev_tokens, prev_tokens_mask = _mlm_mask(tokens)

        return prev_tokens, prev_tokens_mask
    
    def step(self, batch):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        tokens = batch['input_ids']

        noised_tokens, noise_mask = self.inject_noise(
            tokens
        )

        results = self.model(input_ids=noised_tokens)
        logits = results['logits']
        loss, logging_output = self.criterion(logits, tokens, label_mask=noise_mask)

        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output['nll_loss'] = logging_output['nll_loss'] * 0
            logging_output['fullseq_loss'] = logging_output['fullseq_loss'] * 0
            logging_output['fullseq_nll_loss'] = logging_output['fullseq_nll_loss'] * 0
            logging_output['ppl'] = logging_output['ppl'] * 0

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log train metrics
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)
        
        return {"loss": loss}

    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        self.hparams.noise = 'full_mask'

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)
        
        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            self.predict_step(batch, batch_idx)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            self.on_predict_epoch_end()

        super().on_validation_epoch_end()
        
    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        tokens = batch.pop('input_ids')

        noised_tokens, noise_mask = self.inject_noise(
            tokens,
        )
        batch['input_ids'] = noised_tokens

        output_tokens, output_scores = self.model.generate(
            batch=batch,
            max_iter=self.hparams.generator.max_iter,
            temperature=self.hparams.generator.temperature,
            sampling_strategy='argmax',
            partial_masks=~noise_mask,
        )
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens, noise_mask

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        tokens = batch['input_ids'].clone()
        pred_tokens, noise_mask = self.forward(batch, return_ids=True)

        if log_metrics:
            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=noise_mask)
            self.acc.update(recovery_acc, weight=noise_mask.sum())

    def on_predict_epoch_end(self) -> None:
        log_key = 'test' if self.stage == 'test' else 'val'

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)