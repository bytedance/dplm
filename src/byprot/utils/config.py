
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import importlib
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Sequence
import logging
from pytorch_lightning.utilities import rank_zero_only

import hydra
from omegaconf import DictConfig, OmegaConf

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


def make_config(**kwargs):
    return OmegaConf.structured(kwargs)


def compose_config(**kwds):
    return OmegaConf.create(kwds)


def merge_config(default_cfg, override_cfg):
    if override_cfg is None:
        return default_cfg
    return OmegaConf.merge(default_cfg, override_cfg)


def load_yaml_config(fpath: str) -> OmegaConf:
    return OmegaConf.load(fpath)


def parse_cli_override_args():
    _overrides = OmegaConf.from_cli()
    print(_overrides)
    override_dict = {}
    for kk, vv in _overrides.items():
        key = kk if not kk.startswith('+') else kk[1:]
        if key not in override_dict:
            override_dict[key] = vv
        else:
            override_dict[key] = merge_config(override_dict[key], vv)
    overrides = compose_config(**override_dict)
    return overrides


def resolve_experiment_config(config: DictConfig):
    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        from omegaconf import open_dict
        with open_dict(config):
            config.datamodule = experiment_config.datamodule
            config.model = experiment_config.model
            config.task = experiment_config.task
            config.train = experiment_config.train
            config.paths = experiment_config.paths
            config.name = experiment_config.name
            config.trainer = experiment_config.trainer
            config.paths.log_dir = config.experiment_path

            # deal with override args
            cli_overrides = parse_cli_override_args()
            config = merge_config(config, cli_overrides)
            print(cli_overrides)
            # chagne current directory
            os.chdir(config.paths.log_dir)
    return config


def _convert_target_to_string(t: Any) -> Any:
    if callable(t):
        return f"{t.__module__}.{t.__qualname__}"
    else:
        return t


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(cfg: OmegaConf, group=None, **override_kwargs):
    if "_target_" not in cfg:
        raise KeyError("Expected key `_target_` to instantiate.")

    if group is None:
        return hydra.utils.instantiate(cfg, **override_kwargs)
    else:
        from . import registry
        _target_ = cfg.pop('_target_')
        target = registry.get_module(group_name=group, module_name=_target_)
        if target is None:
            raise KeyError(
                f'{_target_} is not a registered <{group}> class [{registry.get_registered_modules(group)}].')
        target = _convert_target_to_string(target)
        log.info(f"    Resolving {group} <{_target_}> -> <{target}>")

        target_cls = get_obj_from_str(target)
        try:
            return target_cls(**cfg, **override_kwargs)
        except:
            cfg = merge_config(cfg, override_kwargs)
            return target_cls(cfg)


def instantiate_from_config2(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))