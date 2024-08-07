import torch
import numpy as np
import random
import time
import re


from tqdm import tqdm
from Bio import SeqIO
from pytorch_lightning import callbacks
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from typing import Dict


# compute running time by 'with' grammar
class TimeCounter:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.start = time.time()
        print(self.text, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        t = end - self.start
        print(f"\nFinished. The time is {t:.2f}s.\n", flush=True)


def progress_bar(now: int, total: int, desc: str = '', end='\n'):
    length = 50
    now = now if now <= total else total
    num = now * length // total
    progress_bar = '[' + '#' * num + '_' * (length - num) + ']'
    display = f'{desc:<10} {progress_bar} {int(now/total*100):02d}% {now}/{total}'

    print(f'\r\033[31m{display}\033[0m', end=end, flush=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_seed():
    torch.seed()
    torch.cuda.seed()
    np.random.seed()
    random.seed()
    torch.backends.cudnn.deterministic = False


def a3m_formalize(input, output, keep_gap=True):
    with open(output, 'w') as w:
        for record in SeqIO.parse(input, 'fasta'):
            desc = record.description
            if keep_gap:
                seq = re.sub(r"[a-z]", "", str(record.seq))
            else:
                seq = re.sub(r"[a-z-]", "", str(record.seq))
            w.write(f">{desc}\n{seq}\n")


def merge_file(file_list: list, save_path: str):
    with open(save_path, 'w') as w:
        for i, file in enumerate(file_list):
            with open(file, 'r') as r:
                for line in tqdm(r, f"Merging {file}... ({i+1}/{len(file_list)})"):
                    w.write(line)
                    
                    
class ModelCheckpoint(callbacks.ModelCheckpoint):

    CHECKPOINT_NAME_BEST = "best"


    # @classmethod
    def _format_checkpoint_name(
        self,
        filename,
        metrics,
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        filename = super()._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)
        filename = filename.replace('/', '_') # avoid '/' in filename unexpectedly creates folder
        return filename

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        trainer.callback_metrics[self.monitor] = self.best_model_score

    def _update_best_and_save(
        self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        # update best checkpoint
        if self.best_model_path == filepath:
            self._save_checkpoint(
                trainer, 
                self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST)
            )

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)


    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and previous != filepath:
            trainer.strategy.remove_checkpoint(previous)