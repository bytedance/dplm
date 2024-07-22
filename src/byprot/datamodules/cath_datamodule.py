
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from byprot import utils
from byprot.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from byprot.datamodules.dataset.cath import CATH
from byprot.datamodules.dataset.data_utils import Alphabet, MaxTokensBatchSampler

log = utils.get_logger(__name__)


# @register_datamodule('struct2seq')
@register_datamodule('cath')
class CATHDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        chain_set_jsonl: str = 'chain_set.jsonl',
        chain_set_splits_json: str = 'chain_set_splits.json',
        max_length: int = 500,
        atoms: List[str] = ('N', 'CA', 'C', 'O'),
        alphabet=None,
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = 'train',
        valid_split: str = 'valid',
        test_split: str = 'test',
        filter_nan: bool = False,
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

        # load datasets only if they're not loaded already
        if stage == 'fit':
            (train, valid), alphabet = CATH(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                max_length=self.hparams.max_length,
                split=(self.hparams.train_split, self.hparams.valid_split),
            )
            self.train_dataset = train
            self.valid_dataset = valid
        elif stage == 'test' or stage == 'predict':
            test, alphabet = CATH(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                split=(self.hparams.test_split, ),
                filter_nan=self.hparams.filter_nan,
            )
            self.test_dataset = test
        else:
            raise ValueError(f"Invalid stage: {stage}.")

        self.alphabet = Alphabet(**self.hparams.alphabet)

        self.collate_batch = self.alphabet.featurizer

    def _build_batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size,
            max_tokens=max_tokens,
            sort=self.hparams.sort,
            drop_last=False,
            sort_key=lambda i: len(dataset[i]['seq']))
        return batch_sampler

    def train_dataloader(self):
        if not hasattr(self, 'train_batch_sampler'):
            self.train_batch_sampler = self._build_batch_sampler(
                self.train_dataset,
                max_tokens=self.hparams.max_tokens,
                shuffle=True
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_sampler=self._build_batch_sampler(
                self.valid_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=self._build_batch_sampler(
                self.test_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )
