import os
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pytorch_lightning import LightningDataModule

from byprot import utils
from byprot.datamodules import register_datamodule
from byprot.datamodules.dataset.tokenized_protein import (
    DPLM2Tokenizer,
    Subset,
    TokenizedProteinDataset,
    setup_dataloader,
)

log = utils.get_logger(__name__)


@register_datamodule("tokenized_protein")
class TokenizedProteinDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/tape",
        max_tokens: int = 6000,
        max_len: int = 1024,
        num_workers: int = 0,
        length_crop: bool = False,
        cluster_training: bool = False,
        min_crop_length: int = 60,
        csv_file: str = "/root",
        struct_vocab_size: int = 8192,
        vocab_file: str = "",
        num_seqs: int = 40,  # used for testing
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.train_dl = None

    def setup(self, stage: Optional[str] = None, split: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and
        `trainer.test()`, so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == "fit":
            self.train_dataset = TokenizedProteinDataset(
                data_dir=self.hparams.data_dir,
                csv_file=self.hparams.csv_file,
                vocab_file=self.hparams.vocab_file,
                split="train",
                max_len=self.hparams.max_len,
                struct_vocab_size=self.hparams.struct_vocab_size,
            )
            self.valid_dataset = TokenizedProteinDataset(
                data_dir=self.hparams.data_dir,
                csv_file=self.hparams.csv_file,
                vocab_file=self.hparams.vocab_file,
                split="valid",
                max_len=self.hparams.max_len,
                struct_vocab_size=self.hparams.struct_vocab_size,
            )
            self.tokenizer = DPLM2Tokenizer.from_pretrained(
                self.hparams.vocab_file
            )
        elif stage == "test" or stage == "predict":
            self.test_dataset = TokenizedProteinDataset(
                data_dir=self.hparams.data_dir,
                vocab_file=self.hparams.vocab_file,
                split="test" if split is None else split,
                max_len=self.hparams.max_len,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}.")
        self.stage = stage

    def train_dataloader(self):
        if self.train_dl is not None:
            self.epoch = self.train_dl.batch_sampler.sampler.epoch + 1
        else:
            self.epoch = 0

        self.train_dataset = TokenizedProteinDataset(
            data_dir=self.hparams.data_dir,
            csv_file=self.hparams.csv_file,
            vocab_file=self.hparams.vocab_file,
            struct_vocab_size=self.hparams.struct_vocab_size,
            split="train",
            max_len=self.hparams.max_len,
        )
        dataset_pandas = self.train_dataset.data.to_pandas()
        if self.hparams.length_crop:
            dataset_pandas = length_cropping(dataset_pandas, self.epoch)
        if self.hparams.cluster_training:
            dataset_pandas = sample_cluster(dataset_pandas, self.epoch)
        self.train_dataset.data = Dataset.from_pandas(dataset_pandas)

        self.train_dl = setup_dataloader(
            self.train_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            max_batch_size=(
                1 if self.stage == "test" or self.stage == "predict" else 800
            ),
            tokenizer=self.tokenizer,
            epoch=self.epoch,
        )
        return self.train_dl

    def val_dataloader(self):
        return setup_dataloader(
            self.valid_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            tokenizer=self.tokenizer,
        )

    def test_dataloader(self):
        return setup_dataloader(
            self.test_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            bucket_size=self.hparams.num_seqs,
            tokenizer=self.tokenizer,
        )


def length_cropping(dataset_pandas, epoch, min_crop_length=60):
    np.random.seed(epoch)
    dataset_pandas["length"] = dataset_pandas["length"].apply(
        lambda l: (
            l
            if np.random.rand() > 0.5
            else (
                np.random.randint(min_crop_length, l)
                if l > min_crop_length
                else l
            )
        )
    )
    return dataset_pandas


def sample_cluster(dataset_pandas, epoch):
    sampled_cluster = (
        dataset_pandas.groupby("cluster")
        .sample(1, random_state=epoch)
        .sort_index()
    )
    sampled_cluster = sampled_cluster.drop(columns="__index_level_0__")
    return sampled_cluster
