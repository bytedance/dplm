# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, List, Union

import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric

from byprot import utils
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg
from byprot.utils.config import merge_config

log = utils.get_logger(__name__)


def cal_index_acc(logits, target, loss_mask, bit_level=False):
    if not bit_level:
        model_pred = logits.argmax(dim=-1)
        index_match = (model_pred == target) & loss_mask
        index_accuracy = index_match.sum() / loss_mask.sum()
        return index_accuracy
    else:
        model_pred = logits.argmax(dim=-1)
        label_mask_expand = loss_mask[..., None].expand(
            model_pred.shape
        )  # B x L x 13
        total_bits = label_mask_expand.sum()
        bitwise_match = (model_pred == target) & label_mask_expand
        bitwise_accuracy = bitwise_match.sum() / total_bits
        index_accuracy = (
            bitwise_match.sum(dim=-1) == bitwise_match.shape[-1]
        ).sum() / loss_mask.sum()
        return index_accuracy, bitwise_accuracy


@register_task("lm/dplm2")
class DPLM2TrainingTask(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight="constant",
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
        self.save_hyperparameters(logger=True)

        self.build_model()
        self.tokenizer = self.model.tokenizer

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == "fit":
            log.info(f"\n{self.model}")
        elif self.stage == "test":
            self.test_step_outputs = []

    def on_before_optimizer_step(self, optimizer):
        if self.global_rank == 0:
            grad_norm_dict = grad_norm(
                self.trainer.strategy.model, norm_type=2
            )
            self.log_dict(grad_norm_dict)

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(
            cfg=self.hparams.model, group="model"
        )

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(
            cfg=self.hparams.criterion
        )
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        # Multi-modal valid loss
        self.eval_struct_loss = MeanMetric()
        self.eval_aatype_loss = MeanMetric()
        self.eval_struct_acc = MeanMetric()
        self.eval_aatype_acc = MeanMetric()

    def load_from_ckpt(self, ckpt_path, not_load=False):
        # do not load state dict from ckpt, just use the initialized parameters.
        if not_load:
            return
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        del state_dict
        print(
            f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def step(self, batch):
        """batch is a Dict containing:

        - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
        - corrd_mask: BooltTensor [bsz, len], where valid coordinates
            are set True, otherwise False
        - lengths: int [bsz, len], protein sequence lengths
        - tokens: LongTensor [bsz, len], sequence of amino acids
        """
        weighting = self.hparams.learning.weight
        logits, targets, loss_masks, weights = self.model.compute_loss(
            batch, weighting=weighting
        )

        loss, logging_output = self.criterion(
            logits,
            targets,
            loss_masks,
            weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )

        # calculate index accuracy
        logging_output["aatype/index_accuracy"] = cal_index_acc(
            logits["aatype"], targets["aatype"], loss_masks["aatype"]
        )
        if len(loss_masks["struct"].shape) == (
            len(targets["struct"].shape) - 1
        ):
            # if bit-based modeling,
            # the loss is in B x L x 13 and label_mask is in B x L
            (
                logging_output["struct/index_accuracy"],
                logging_output["struct/bit_accuracy"],
            ) = cal_index_acc(
                logits["struct"],
                targets["struct"],
                loss_masks["struct"],
                bit_level=True,
            )
        else:
            logging_output["struct/index_accuracy"] = cal_index_acc(
                logits["struct"], targets["struct"], loss_masks["struct"]
            )

        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output["nll_loss"] = logging_output["nll_loss"] * 0
            logging_output["fullseq_loss"] = logging_output["fullseq_loss"] * 0
            logging_output["fullseq_nll_loss"] = (
                logging_output["fullseq_nll_loss"] * 0
            )
            logging_output["ppl"] = logging_output["ppl"] * 0

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log train metrics
        self.log(
            "global_step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(
                f"train/{log_key}",
                log_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log other metrics
        sample_size = logging_output["sample_size"]
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(
            logging_output["nll_loss"], weight=sample_size
        )

        for log_key in logging_output:
            if "constant_diff_loss" not in log_key:
                continue
            log_value = logging_output[log_key]
            eval_type = log_key.split("/")[0]
            if eval_type == "aatype":
                self.eval_aatype_loss.update(log_value, weight=sample_size)
            elif eval_type == "struct":
                self.eval_struct_loss.update(log_value, weight=sample_size)
            else:
                raise NotImplementedError
        self.eval_aatype_acc.update(
            logging_output["aatype/index_accuracy"], weight=sample_size
        )
        self.eval_struct_acc.update(
            logging_output["struct/index_accuracy"], weight=sample_size
        )

        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = "test" if self.stage == "test" else "val"

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        eval_aatype_loss = self.eval_aatype_loss.compute()
        self.eval_aatype_loss.reset()
        eval_struct_loss = self.eval_struct_loss.compute()
        self.eval_struct_loss.reset()
        eval_aatype_accuracy = self.eval_aatype_acc.compute()
        self.eval_aatype_acc.reset()
        eval_struct_accuracy = self.eval_struct_acc.compute()
        self.eval_struct_acc.reset()

        self.log(
            f"{log_key}/loss",
            eval_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/nll_loss",
            eval_nll_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/ppl",
            eval_ppl,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f"{log_key}/aatype_loss",
            eval_aatype_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/struct_loss",
            eval_struct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/aatype_index_accuracy",
            eval_aatype_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/struct_index_accuracy",
            eval_struct_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.stage == "fit":
            self.val_ppl_best.update(eval_ppl)
            self.log(
                "val/ppl_best",
                self.val_ppl_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )

        super().on_validation_epoch_end()
