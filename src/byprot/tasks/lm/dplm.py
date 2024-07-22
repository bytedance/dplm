
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


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

log = utils.get_logger(__name__)

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('lm/dplm')
class DPLMTrainingTask(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='rdm',  # ['full_mask', 'random_mask']
            num_unroll=0,
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight='constant',
        ),
    )
    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)
    
        self.build_model() 
        self.tokenizer = self.model.tokenizer
        # self.model = None
        # self.build_generator()

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
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()
        
    def step(self, batch):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        weighting = self.hparams.learning.weight
        logits, target, loss_mask, weights = self.model.compute_loss(
            batch, weighting=weighting)

        loss, logging_output = self.criterion(
            logits, target,
            loss_mask,
            weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )
        
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

        super().on_validation_epoch_end()