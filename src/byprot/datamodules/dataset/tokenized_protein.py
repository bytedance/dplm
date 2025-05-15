import imp
import math
import os
from typing import Iterable, Sequence, TypeVar

import datasets
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from transformers import EsmTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken

from byprot import utils

log = utils.get_logger(__name__)
T_co = TypeVar("T_co", covariant=True)


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


def preprocess_dataset(csv_path, data_bin, split):
    def remove_lowconf_ends(row, threshold=50):
        aa_seq, ss_seq, plddt = (
            row["aa_seq"],
            row["struct_seq"],
            np.array(row["plddt"]),
        )
        ss_seq = ss_seq.split(",")
        modeled_idx = np.where(plddt > threshold)[0]
        min_modeled_idx = np.min(modeled_idx)
        max_modeled_idx = np.max(modeled_idx)
        aa_seq = aa_seq[min_modeled_idx : (max_modeled_idx + 1)]
        ss_seq = ss_seq[min_modeled_idx : (max_modeled_idx + 1)]
        plddt = plddt[min_modeled_idx : (max_modeled_idx + 1)]
        ss_seq = ",".join(ss_seq)
        row["aa_seq"], row["struct_seq"], row["plddt"] = aa_seq, ss_seq, plddt
        return row

    # preprocess dataset
    afdb_pdb = pd.read_csv(csv_path)

    afdb_pdb.dropna(subset=["aa_seq"], inplace=True)
    afdb = afdb_pdb[afdb_pdb["split"] == "afdb_swissprot"]
    pdb = afdb_pdb[afdb_pdb["split"] == "pdb"]

    afdb["plddt"] = afdb["plddt"].apply(
        lambda l: [float(a) for a in l.split(",") if len(a) > 0]
    )
    afdb = afdb.apply(
        lambda row: remove_lowconf_ends(row, threshold=70), axis=1
    )
    pdb["plddt"] = pdb["plddt"].apply(
        lambda l: [float(a) for a in l.split(",") if len(a) > 0]
    )
    pdb = pdb.apply(lambda row: remove_lowconf_ends(row, threshold=70), axis=1)

    afdb["plddt_std"] = afdb["plddt"].apply(lambda l: np.std(l))
    afdb = afdb[afdb["plddt_std"] < 15]
    remaining_set = pd.concat([afdb, pdb], axis=0)

    remaining_set = remaining_set[remaining_set["aa_seq"].str.len() <= 1024]
    remaining_set = remaining_set[
        (remaining_set["split"] == "pdb")
        | (
            (remaining_set["avg_plddt"].notna())
            & (remaining_set["avg_plddt"] > 85)
        )
    ]
    remaining_set["cluster"] = remaining_set["cluster"].apply(lambda x: str(x))

    # save to huggingface dataset
    valid_set = afdb_pdb[afdb_pdb["split"] == "cameo2022"]
    valid_set = datasets.Dataset.from_pandas(valid_set)
    training_set = datasets.Dataset.from_pandas(remaining_set)

    def add_seqlen(example):
        example["length"] = len(example["aa_seq"])
        return example

    training_set = training_set.map(add_seqlen)
    valid_set = valid_set.map(add_seqlen)

    os.makedirs(data_bin, exist_ok=True)
    training_set.save_to_disk(os.path.join(data_bin, "train"), num_proc=1)
    valid_set.save_to_disk(os.path.join(data_bin, "valid"), num_proc=1)

    log.info(f"Preprocessed dataset from {csv_path}.")
    return training_set if split == "train" else valid_set


class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close
    together."""

    def __init__(
        self,
        sequence_lengths: Iterable,
        bucket_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ):
        if dist.is_available():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(
            math.ceil(len(self.data) * 1.0 / self.num_replicas)
        )
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size]
            for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = epoch
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        msa_depth=None,
        drop_last=False,
        batch_size=None,
        max_len=512,
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if (
                linear <= self.max_tokens
                and quadratic < self.max_square_tokens
            ):
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print("Current batch is empty! idx is ", idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)

        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


class TokenizedProteinDataset(Dataset):
    """Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        csv_file: str,
        max_len=2048,
        vocab_file="airkingbd/dplm2_650m",
        struct_vocab_size=8192,
    ):
        self.data_dir = data_dir
        self.split = split
        csv_path = os.path.join(self.data_dir, csv_file)
        data_path = os.path.join(self.data_dir, csv_file.replace(".csv", ""))
        try:
            self.data = load_dataset_from_hf(data_path, split)
        except:
            self.data = preprocess_dataset(
                csv_path, data_path, self.split, struct_vocab_size
            )
        log.info(f"Dataset size: {len(self.data)}")

        self.max_len = max_len
        self.tokenizer = DPLM2Tokenizer.from_pretrained(vocab_file)

    def __len__(self):
        return len(self.data)

    def get_metadata_lens(self):
        return self.data["length"]

    def __getitem__(self, idx):
        row = self.data[int(idx)]
        max_len = min(self.max_len, row["length"])

        struct_tokens = row["struct_seq"]
        struct_tokens = struct_tokens.split(",")
        if len(struct_tokens) - max_len > 0:
            start = np.random.choice(len(struct_tokens) - max_len)
            stop = start + max_len
        else:
            start = 0
            stop = len(struct_tokens)
        struct_tokens = struct_tokens[start:stop]
        struct_tokens = "".join(struct_tokens)
        struct_tokens = (
            self.tokenizer.struct_cls_token
            + struct_tokens
            + self.tokenizer.struct_eos_token
        )

        aatype_tokens = row["aa_seq"]
        if len(aatype_tokens) - max_len > 0:
            # in order to keep the parallelism, start and end position should be same as struct seq
            aatype_tokens = aatype_tokens[start:stop]
        aatype_tokens = (
            self.tokenizer.aa_cls_token
            + aatype_tokens
            + self.tokenizer.aa_eos_token
        )

        return_dict = {
            "struct_tokens": struct_tokens,
            "aatype_tokens": aatype_tokens,
            "length": max_len + 2,
        }
        if "pdb_name" in row:
            return_dict["pdb_name"] = row["pdb_name"]

        return return_dict


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DPLM2Tokenizer(EsmTokenizer):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "aa_cls_token",
        "aa_eos_token",
        "aa_unk_token",
        "aa_mask_token",
        "struct_cls_token",
        "struct_eos_token",
        "struct_unk_token",
        "struct_mask_token",
        "pad_token",
    ]

    def __init__(
        self,
        vocab_file,
        aa_cls_token="<cls_aa>",
        aa_eos_token="<eos_aa>",
        aa_unk_token="<unk_aa>",
        aa_mask_token="<mask_aa>",
        struct_cls_token="<cls_struct>",
        struct_eos_token="<eos_struct>",
        struct_unk_token="<unk_struct>",
        struct_mask_token="<mask_struct>",
        pad_token="<pad>",
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {
            tok: ind for ind, tok in enumerate(self.all_tokens)
        }

        self._aa_cls_token = None
        self._aa_eos_token = None
        self._aa_unk_token = None
        self._aa_mask_token = None
        self._struct_cls_token = None
        self._struct_eos_token = None
        self._struct_unk_token = None
        self._struct_mask_token = None
        self._pad_token = None

        PreTrainedTokenizer.__init__(
            self,
            aa_cls_token=aa_cls_token,
            aa_eos_token=aa_eos_token,
            aa_unk_token=aa_unk_token,
            aa_mask_token=aa_mask_token,
            struct_cls_token=struct_cls_token,
            struct_eos_token=struct_eos_token,
            struct_unk_token=struct_unk_token,
            struct_mask_token=struct_mask_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    @property
    def aa_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_eos_token is None:
            if self.verbose:
                log.error("Using aa_eos_token, but it is not set yet.")
            return None
        return str(self._aa_eos_token)

    @property
    def aa_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_cls_token is None:
            if self.verbose:
                log.error("Using aa_cls_token, but it is not set yet.")
            return None
        return str(self._aa_cls_token)

    @property
    def aa_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_unk_token is None:
            if self.verbose:
                log.error("Using aa_unk_token, but it is not set yet.")
            return None
        return str(self._aa_unk_token)

    @property
    def aa_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_mask_token is None:
            if self.verbose:
                log.error("Using aa_mask_token, but it is not set yet.")
            return None
        return str(self._aa_mask_token)

    @property
    def struct_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_eos_token is None:
            if self.verbose:
                log.error("Using struct_eos_token, but it is not set yet.")
            return None
        return str(self._struct_eos_token)

    @property
    def struct_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_cls_token is None:
            if self.verbose:
                log.error("Using struct_cls_token, but it is not set yet.")
            return None
        return str(self._struct_cls_token)

    @property
    def struct_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_unk_token is None:
            if self.verbose:
                log.error("Using struct_unk_token, but it is not set yet.")
            return None
        return str(self._struct_unk_token)

    @property
    def struct_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_mask_token is None:
            if self.verbose:
                log.error("Using struct_mask_token, but it is not set yet.")
            return None
        return str(self._struct_mask_token)

    @aa_cls_token.setter
    def aa_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_cls_token"
            )
        self._aa_cls_token = value

    @aa_eos_token.setter
    def aa_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_eos_token"
            )
        self._aa_eos_token = value

    @aa_unk_token.setter
    def aa_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_unk_token"
            )
        self._aa_unk_token = value

    @aa_mask_token.setter
    def aa_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_mask_token"
            )
        self._aa_mask_token = value

    @struct_cls_token.setter
    def struct_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_cls_token"
            )
        self._struct_cls_token = value

    @struct_eos_token.setter
    def struct_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_eos_token"
            )
        self._struct_eos_token = value

    @struct_unk_token.setter
    def struct_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_unk_token"
            )
        self._struct_unk_token = value

    @struct_mask_token.setter
    def struct_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_mask_token"
            )
        self._struct_mask_token = value


class DPLM2Collater(object):
    def __init__(self, tokenizer):
        self.tokenizer = (
            tokenizer  # DPLM2Tokenizer.from_pretrained(vocab_file)
        )

    def __call__(self, raw_batch):
        if len(list(zip(*raw_batch))) == 0:
            print("list idx error!")
            print(raw_batch)

        struct_tokens_list = [sample["struct_tokens"] for sample in raw_batch]

        batch_struct = self.tokenizer.batch_encode_plus(
            struct_tokens_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_struct = {
            "targets": batch_struct["input_ids"],
            "attention_mask": batch_struct["attention_mask"].bool(),
        }

        aatype_list = [sample["aatype_tokens"] for sample in raw_batch]
        batch_aatype = self.tokenizer.batch_encode_plus(
            aatype_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        batch_aatype = {
            "targets": batch_aatype["input_ids"],
            "attention_mask": batch_aatype["attention_mask"].bool(),
        }

        batch = {
            "struct_tokens": batch_struct,
            "aatype_tokens": batch_aatype,
        }

        if "pdb_name" in raw_batch[0]:
            pdb_name_list = [sample["pdb_name"] for sample in raw_batch]
            batch["pdb_name"] = pdb_name_list

        return batch


def setup_dataloader(
    ds: TokenizedProteinDataset,
    max_tokens=6000,
    bucket_size=1000,
    max_batch_size=100,
    num_workers=8,
    rank=0,
    world_size=1,
    max_len=512,
    tokenizer=None,
    epoch=0,
) -> DataLoader:
    collater = DPLM2Collater(tokenizer)
    lens = ds.get_metadata_lens()
    train_sortish_sampler = SortishSampler(
        lens, bucket_size, num_replicas=world_size, rank=rank, epoch=epoch
    )
    train_sampler = ApproxBatchSampler(
        train_sortish_sampler,
        max_tokens,
        max_batch_size,
        lens,
        max_len=max_len,
    )
    dl = DataLoader(
        dataset=ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collater,
    )
    return dl


def load_dataset_from_hf(data_path, split):
    ds = load_dataset(data_path, name=split)["train"]
    return ds
