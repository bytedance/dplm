from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from byprot import utils
from byprot.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from byprot.datamodules.dataset.uniref_hf import UniRefHFDataset, setup_dataloader, Subset, UniRefDatasetForTesting

log = utils.get_logger(__name__)

@register_datamodule('uniref50_hf')
class UniRefHFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/tape",
        max_tokens: int = 6000,
        max_len: int = 2048,
        num_workers: int = 0,
        num_seqs: int = 40, # used for testing
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if stage == 'fit':
            self.train_dataset = UniRefHFDataset(data_dir=self.hparams.data_dir, split='train', max_len=self.hparams.max_len)
            self.valid_dataset = UniRefHFDataset(data_dir=self.hparams.data_dir, split='valid', max_len=self.hparams.max_len)
        elif stage == 'test' or stage == 'predict':
            self.test_dataset = UniRefDatasetForTesting(max_len=self.hparams.max_len, num_seqs=self.hparams.num_seqs)
            self.train_dataset = UniRefDatasetForTesting(max_len=self.hparams.max_len, num_seqs=self.hparams.num_seqs) # used for deepspeed
        else:
            raise ValueError(f"Invalid stage: {stage}.")
        self.stage = stage

    def train_dataloader(self):
        return setup_dataloader(
            self.train_dataset, 
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            max_batch_size=1 if self.stage == 'test' or self.stage == 'predict' else 800,
        )

    def val_dataloader(self):
        return setup_dataloader(
            self.valid_dataset, 
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
        )

    def test_dataloader(self):
        return setup_dataloader(
            self.test_dataset, 
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            bucket_size=self.hparams.num_seqs,
            max_batch_size=self.hparams.num_seqs,
        )