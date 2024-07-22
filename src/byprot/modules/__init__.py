
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn
from torch.nn import functional as F


_registry = {}
class _Criterion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.criterions = {}
        self.weights = {}

        self._build()

    def _build(self):
        for name, cfg in self.cfg.items():
            _target_ = cfg.pop('_target_')
            weight = cfg.pop('weight', 1.0)
            self.criterions[name] = _instantiate(_target_, cfg=cfg, registry=_registry)
            self.weights[name] = weight

    def forward(self, model_outs, targets):
        """

        Args:
            model_outs (dict): dict of loss_name: model_out
            targets (_type_): _description_
        """
        logging_outs = {}
        total_loss = 0.

        for name, model_out in model_outs.items():
            if name in self.criterions:
                loss, logging_out = self.criterions[name](model_out, targets[name])

                total_loss += self.weights[name] * loss
                logging_out = {f'{name}/{key}': val for key, val in logging_out.items()}

                logging_outs.update(logging_out)  

        return total_loss, logging_outs