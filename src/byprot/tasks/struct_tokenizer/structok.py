# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from openfold.np import residue_constants
from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import gdt_ha, gdt_ts
from torch import nn
from torch.nn import functional as F
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from byprot import utils
from byprot.models.structok.modules.loss import drmsd
from byprot.tasks import (
    TaskLitModule,
    get_optimizer,
    get_scheduler,
    register_task,
)
from byprot.utils.config import compose_config as Cfg
from byprot.utils.config import merge_config

# import esm

log = utils.get_logger(__name__)


def exists(o):
    return o is not None


@register_task("structok")
class StrucTok(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            pretrained_model_path=None,
            restore_optimizer=False,
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

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == "fit":
            log.info(f"\n{self.model}")

            if exists(self.hparams.learning.pretrained_model_path):
                log.info(
                    f"Initializing model from pretrained weights: {self.hparams.learning.pretrained_model_path}"
                )
                self.load_from_ckpt(
                    self.hparams.learning.pretrained_model_path
                )

    def load_from_ckpt(self, ckpt_path):
        # return
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        state_dict_without_decoder = {
            nn: pp
            for nn, pp in state_dict.items()
            if not nn.startswith("model.decoder")
        }
        if self.hparams.learning.get("no_pretrained_decoder"):
            state_dict_decoder = {}
        else:
            state_dict_decoder = {
                nn: pp
                for nn, pp in state_dict.items()
                if nn.startswith("model.decoder")
            }
        for sd in [state_dict_without_decoder, state_dict_decoder]:
            try:
                missing, unexpected = self.load_state_dict(sd, strict=False)
                print(
                    f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
                )
            except RuntimeError as e:
                print(e)
                continue

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(
            cfg=self.hparams.model, group="model"
        )

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(
            cfg=self.hparams.criterion
        )

    def build_torchmetric(self):
        self.metrics = nn.ModuleDict(
            {"eval_loss": MeanMetric(), "eval_loss_best": MinMetric()}
        )

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def step(self, batch):
        try:
            outputs, codebook_loss, predicted_indices = self.model(batch)
            loss, logging_outputs = self.criterion(
                outputs,
                batch,
                codebook_loss,
                self.global_step,
                predicted_indices,
            )
        # loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)
        except:
            loss = None
            outputs = None
            logging_outputs = None

        return loss, outputs, logging_outputs

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        loss, model_outputs, logging_output = self.step(batch)
        if loss is None:
            try:
                log.info(
                    f"Error in current training step! csv index: {batch['csv_idx'].tolist()}"
                )
            except:
                log.info(
                    f"Error in current training step! CAN NOT PRINT batch['csv_idx']!"
                )
            return

        # log train metrics
        self.log(
            "global_step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        # Log it
        self._log(logging_output, batch, model_outputs, train=True)

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        loss, model_outputs, logging_output = self.step(batch)

        # Log it
        self._log(logging_output, batch, model_outputs, train=False)

        # # log other metrics
        self.metrics["eval_loss"].update(
            loss, weight=logging_output["num_residue"]
        )
        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = "test" if self.stage == "test" else "val"

        # compute metrics averaged over the whole dataset
        eval_loss_agg = self.metrics["eval_loss"].compute()
        self.metrics["eval_loss"].reset()
        self.log(
            f"{log_key}/loss",
            eval_loss_agg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.stage == "fit":
            self.metrics["eval_loss_best"].update(eval_loss_agg)
            self.log(
                f"{log_key}/loss_best",
                self.metrics["eval_loss_best"].compute(),
                on_epoch=True,
                prog_bar=True,
            )

        super().on_validation_epoch_end()

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}",
                indiv_loss,
                on_step=train,
                on_epoch=(not train),
                prog_bar=True,
                logger=True,
            )

            if train:
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, outputs, superimposition_metrics=(not train)
            )

        for k, v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def _compute_validation_metrics(
        self, batch, outputs, superimposition_metrics=False
    ):
        metrics = {}

        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]

        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.criterion.config.rec_loss.eps,
            per_residue=False,
        )

        metrics["lddt_ca"] = lddt_ca_score

        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca,  # still required here to compute n
        )

        metrics["drmsd_ca"] = drmsd_ca_score

        if superimposition_metrics:

            @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
            def safe_superimpose(reference, coords, mask):
                return superimpose(
                    reference,
                    coords,
                    mask,
                )

            superimposed_pred, alignment_rmsd = safe_superimpose(
                gt_coords_masked_ca,
                pred_coords_masked_ca,
                all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

        return metrics

    # -------# Optimizers & Lr Schedulers #-------- #
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = get_optimizer(
            self.hparams.optimizer,
            [pp for pp in self.parameters() if pp.requires_grad],
        )
        if (
            self.training
            and self.hparams.learning.restore_optimizer
            and exists(self.hparams.learning.pretrained_model_path)
        ):
            log.info(
                f"Restoring optimizer states from: {self.hparams.learning.pretrained_model_path}"
            )
            loaded_state_dict = torch.load(
                self.hparams.learning.pretrained_model_path, map_location="cpu"
            )["optimizer_states"][0]
            # only restore optimizer state, keep other optmizer hparams set this time.
            state_dict = optimizer.state_dict()
            state_dict["state"] = loaded_state_dict["state"]
            optimizer.load_state_dict(state_dict)
        if (
            "lr_scheduler" in self.hparams
            and self.hparams.lr_scheduler is not None
        ):
            lr_scheduler, extra_kwargs = get_scheduler(
                self.hparams.lr_scheduler, optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, **extra_kwargs},
            }
        return optimizer
