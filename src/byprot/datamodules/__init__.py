# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import glob
import importlib
import os

from omegaconf import DictConfig

from byprot.utils import import_modules

DATAMODULE_REGISTRY = {}


def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls

    return decorator


import_modules(os.path.dirname(__file__), "byprot.datamodules")
