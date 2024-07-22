
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.logger import Logger
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from pathlib import Path
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from wandb import Artifact
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run

from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import os


class ByProtWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # HIGHLIGHT: Remove this check below
        # if not _WANDB_AVAILABLE:
        #     raise ModuleNotFoundError(str(_WANDB_AVAILABLE))

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        # super().__init__()
        Logger.__init__(self)
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)
        elif dir is not None:
            dir = os.fspath(dir)

        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = {
            "name": name,
            "project": project,
            "dir": save_dir or dir,
            "id": version or id,
            "resume": "allow",
            "anonymous": ("allow" if anonymous else None),
        }
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._checkpoint_name = checkpoint_name