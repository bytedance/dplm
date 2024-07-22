
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import glob
import importlib
import logging
import os
import random
import subprocess
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.seed import isolate_rng

from . import strategies
from .config import instantiate_from_config, load_yaml_config, resolve_experiment_config


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def load_from_experiment(experiment_save_dir, ckpt="best.ckpt"):
    cfg_path = Path(experiment_save_dir, ".hydra", "config.yaml")
    cfg = load_yaml_config(str(cfg_path))
    cfg.ckpt_path = Path(experiment_save_dir, "checkpoints", ckpt)

    pl_module = instantiate_from_config(cfg=cfg.task, group="task", model=cfg.model)
    pl_module.load_from_ckpt(str(cfg.ckpt_path))

    return pl_module, cfg


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    OmegaConf.register_new_resolver("eval", eval)

    # print current git revision sh
    log.info(f"Current git revision hash: {get_git_revision_hash()}")

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)

    return config


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "task",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "training",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}
    config = OmegaConf.to_container(config, resolve=True)

    # choose which parts of hydra config will be saved to loggers
    hparams["task"] = config["task"]
    hparams["task"].pop("model", None)

    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def common_pipeline(config, training=False):
    torch.set_float32_matmul_precision("high")
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate_from_config(
        cfg=config.datamodule, group="datamodule"
    )

    # Init lightning model as task
    log.info(f"Instantiating task (pl_module) <{config.task._target_}>")
    # pl_module: LightningModule = hydra.utils.instantiate(config.task, model=model)
    pl_module: LightningModule = instantiate_from_config(
        cfg=config.task, group="task", model=config.model
    )

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                lg = hydra.utils.instantiate(lg_conf)
                logger.append(lg)

                # FIXME: a hack to avoid tensorboard saving hparams error at first run
                if isinstance(lg, TensorBoardLogger):
                    hparams_file = os.path.join(lg.log_dir, lg.NAME_HPARAMS_FILE)
                    os.makedirs(lg.log_dir, exist_ok=True)
                    open(hparams_file, "w").close()

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            # if cb_name == 'model_summary' and not training:
            #     continue
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
        if config.trainer.get("enable_progress_bar", False):
            from byprot.utils.callbacks import BetterRichProgressBar

            callbacks.append(BetterRichProgressBar(leave=False))

    return datamodule, pl_module, logger, callbacks


def resolve_ckpt_path(ckpt_dir, ckpt_path):
    # if not absolute path, it should be inferred from current working directory or ckeckpoint directory
    if not os.path.isabs(ckpt_path):
        # if ckpt_path is in cwd
        if os.path.exists(os.path.join(hydra.utils.get_original_cwd(), ckpt_path)):
            ckpt_path = os.path.abspath(
                os.path.join(hydra.utils.get_original_cwd(), ckpt_path)
            )

        # or if ckpt_path is in the predefined checkpoint directory
        elif os.path.exists(os.path.join(ckpt_dir, ckpt_path)):
            ckpt_path = os.path.abspath(os.path.join(ckpt_dir, ckpt_path))

    return ckpt_path


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == "cpu":
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def recursive_apply(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, list):
        return [recursive_to(o, fn=fn) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, fn=fn) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, fn=fn) for k, v in obj.items()}
    else:
        raise TypeError(type(obj))


def recursive_eval(obj):
    if isinstance(obj, list):
        return [recursive_eval(o) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_eval(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_eval(v) for k, v in obj.items()}
    else:
        try:
            _obj = eval(obj)
        except:
            pass
        return _obj


def import_modules(models_dir, namespace, excludes=[]):
    for path in glob.glob(models_dir + "/**", recursive=True)[1:]:
        if any(e in path for e in excludes):
            continue

        file = os.path.split(path)[1]
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file

            _namespace = path.replace("/", ".")
            _namespace = _namespace[
                _namespace.find(namespace) : _namespace.rfind("." + module_name)
            ]
            importlib.import_module(_namespace + "." + module_name)


def get_git_revision_hash() -> str:
    from pathlib import Path

    REPO_DIR = str(Path(__file__).resolve().parents[2])
    return (
        subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )


def seed_everything(seed, verbose=False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    if verbose:
        log.info(f"Random seed set to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


@contextmanager
def local_seed(seed, enable=True):
    if enable:
        with isolate_rng():
            seed_everything(seed)
            yield
    else:
        yield
