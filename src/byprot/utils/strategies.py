
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Union

from pytorch_lightning.strategies import StrategyRegistry
# from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy
# from pytorch_lightning.strategies.sharded import DDPShardedStrategy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import FSDPStrategy
from typing_extensions import override
from torch.nn import Module
import torch
from lightning_fabric.strategies.fsdp import _has_meta_device_parameters, _move_torchmetrics_to_device, _setup_activation_checkpointing
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

# from pytorch_lightning.utilities.imports import _FAIRSCALE_AVAILABLE
from torch.optim import Optimizer
import logging

#from byprot.models.lm.esm_hf_flashattn import ModifiedEsmLayer

log = logging.getLogger(__name__)

class CPUInitFSDPStrategy(FSDPStrategy):
    @override
    def _setup_model(self, model: Module) -> Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module."""
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in model.modules()):
            if _has_meta_device_parameters(model):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self.kwargs:
                # The user has wrapped their submodules manually, don't apply the auto wrap policy.
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self.kwargs["auto_wrap_policy"]
        else:
            log.debug(f"setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")
            model = model.to(torch.device('cpu'))
            torch.cuda.set_device(self.root_device)
            model = FullyShardedDataParallel(
                module=model,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                # device_id=self.root_device.index,
                **self.kwargs,
            )
            model = model.to(torch.device(self.root_device))

        _move_torchmetrics_to_device(model, self.root_device)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)

        return model