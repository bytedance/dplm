
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import os
from typing import List
from byprot.tasks import on_prediction_mode

from torch import nn
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger as LightningLoggerBase

from byprot import utils

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing/prediction pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=config.ckpt_path)    
    
    # loading pipeline
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    mode = config.mode

    # Start prediction
    log.info(f"Starting on mode='{mode}'!")

    # (1) Specify test dataset by configuring datamodule.test_split
    data_split = config.get('data_split') or config.datamodule.get('test_split', 'test')
    datamodule.hparams.test_split = data_split
    log.info(f"Loading test data from '{data_split}' dataset...")

    # Pytorch Lightning treat predict differently compared to what we commonly think of.
    # Must use this context manager and trainer.test to run prediction as expected.
    with on_prediction_mode(pl_module, enable=mode == 'predict'):
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=config.ckpt_path)

    log.info(f"Finished mode='{mode}' on '{data_split}' dataset.")
