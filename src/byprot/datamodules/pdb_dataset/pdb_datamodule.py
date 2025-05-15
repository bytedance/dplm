# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PDB data loader."""

import functools as fn
import logging
import math
import random

import numpy as np
import pandas as pd
import torch
import tree
from openfold.config import config as OF_CONFIG
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data.distributed import DistributedSampler, dist

from byprot import utils
from byprot.datamodules import register_datamodule
from byprot.datamodules.dataset.data_utils import MaxTokensBatchSampler
from byprot.datamodules.pdb_dataset import utils as du

from .utils import aatype_to_seq, seq_to_aatype

log = utils.get_logger(__name__)


SHAPE_SCHEMA = dict(OF_CONFIG.data.common.feat)
SHAPE_SCHEMA["chain_index"] = SHAPE_SCHEMA["residue_index"]
SHAPE_SCHEMA["gvp_feat"] = SHAPE_SCHEMA["residue_index"]

from textwrap import wrap

struct_seq_to_ids = lambda struct_seq: [
    int(elem) for elem in struct_seq.strip().split(",")
]
struct_ids_to_seq = lambda ids: ",".join([f"{elem:04d}" for elem in ids])


def load_from_pdb(pdb_path, batch=False):
    raw_chain_feats, metadata = du.process_pdb_file(pdb_path)
    chain_feats = PdbDataset.process_chain(raw_chain_feats)
    chain_feats["pdb_name"] = metadata["pdb_name"]
    return chain_feats


def collate_fn(batch: list):
    new_batch = []
    max_len = max([len(elem["res_mask"]) for elem in batch])
    # max_len = 400
    for raw_feats in batch:
        padded_feats = du.pad_feats(raw_feats, max_len=max_len, use_torch=True)
        new_batch.append(padded_feats)
    return default_collate(new_batch)


def exists(o):
    return o is not None


@register_datamodule("pdb")
class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

        self._dist_sampler = None
        self.collate_batch = collate_fn

    def setup(self, stage: str):
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            split=self.dataset_cfg.train_split,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            split=self.dataset_cfg.valid_split,
            is_training=False,
        )

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        if self.loader_cfg.get("length_batch"):
            return DataLoader(
                self._train_dataset,
                batch_sampler=LengthBatcher(
                    sampler_cfg=self.sampler_cfg,
                    metadata_csv=self._train_dataset.csv,
                    rank=rank,
                    num_replicas=num_replicas,
                ),
                num_workers=num_workers,
                prefetch_factor=(
                    None
                    if num_workers == 0
                    else self.loader_cfg.prefetch_factor
                ),
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif self.loader_cfg.get("bucket_sort"):
            if not hasattr(self, "train_batch_sampler"):
                self.train_batch_sampler = self._build_batch_sampler(
                    self._train_dataset,
                    max_tokens=int(self.sampler_cfg.max_num_res_squared),
                    shuffle=True,
                )
            else:
                self._train_dataset._seed = (
                    self.dataset_cfg.seed + self.train_batch_sampler._epoch
                )
                self._train_dataset._init_metadata()
            return DataLoader(
                dataset=self._train_dataset,
                batch_sampler=self.train_batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=self.collate_batch,
            )
        else:
            if self._dist_sampler is None:
                self._dist_sampler = DistributedSampler(
                    self._train_dataset, shuffle=True
                )
            else:
                self._train_dataset._seed = (
                    self.dataset_cfg.seed + self._dist_sampler.epoch
                )
                self._train_dataset._init_metadata()
            return DataLoader(
                self._train_dataset,
                sampler=self._dist_sampler,
                batch_size=self.loader_cfg.batch_size,
                num_workers=num_workers,
                prefetch_factor=(
                    None
                    if num_workers == 0
                    else self.loader_cfg.prefetch_factor
                ),
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                collate_fn=self.collate_batch,
            )

    def _build_batch_sampler(
        self, dataset, max_tokens, shuffle=False, distributed=True
    ):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.loader_cfg.batch_size,
            max_tokens=max_tokens,
            sort=True,
            drop_last=True,
            sort_key=lambda i: min(
                dataset.csv.iloc[i]["modeled_seq_len"],
                (
                    self.dataset_cfg.crop_size
                    if self.dataset_cfg.crop_size > 0
                    else 1e5
                ),
            )
            ** 2,
        )
        return batch_sampler

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            # sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )


class PdbDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_cfg,
        split,
        is_training,
    ):
        self._log = log
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._seed = self._dataset_cfg.seed
        self.split = split
        self.crop_size = self.dataset_cfg.crop_size
        self._init_metadata()
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)
        pdb_csv = pdb_csv[pdb_csv.split.isin(self.split)]
        self.raw_csv = pdb_csv

        # Filtering
        filter_conf = self.dataset_cfg.filter

        # pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        # pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]

        # Training or validation specific logic.
        if not self._is_training:
            self.crop_size = 0
            # pdb_csv = pdb_csv[pdb_csv.split == self.split]

            pdb_csv = pdb_csv[
                pdb_csv.modeled_seq_len <= self.dataset_cfg.eval_max_len
            ]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]

            pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = pdb_csv
            self._log.info(
                f"Validation ({self.split}): {len(self.csv)} examples"
            )
            return

        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]

        if (
            filter_conf.allowed_oligomer is not None
            and len(filter_conf.allowed_oligomer) > 0
        ):
            pdb_csv = pdb_csv[
                pdb_csv.oligomeric_detail.isin(filter_conf.allowed_oligomer)
            ]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.helix_percent < filter_conf.max_helix_percent
            ]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.coil_percent < filter_conf.max_loop_percent
            ]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.strand_percent > filter_conf.min_beta_percent
            ]
        if (
            self._is_training
            and filter_conf.rog_quantile is not None
            and filter_conf.rog_quantile > 0.0
        ):
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv,
                filter_conf.rog_quantile,
                np.arange(filter_conf.max_len),
            )
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x - 1]
            )
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]

        if filter_conf.subset is not None:
            pdb_csv = pdb_csv.iloc[: filter_conf.subset]

        if filter_conf.get("min_lddt_ca") is not None:
            _num_examples = len(pdb_csv)
            pdb_csv = pdb_csv[
                (pdb_csv.lddt_ca.notnull())
                & (pdb_csv.lddt_ca > filter_conf.min_lddt_ca)
            ]
            self._log.info(
                f"Filtering by min_lddt_ca. before: {_num_examples} examples; after: {len(pdb_csv)} examples"
            )

        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)

        if self._dataset_cfg.get("load_gvp_feat"):
            pdb_csv = pdb_csv[pdb_csv.gvp_feat_path.notnull()]

        # if self._dataset_cfg.get("load_struct_seq"):
        #     pdb_csv = pdb_csv[pdb_csv.struct_seq.notnull()]

        if self.is_training and filter_conf.get("cluster_sample"):
            print(f"Sampling clusters with seed {self._seed}")
            pdb_csv = self.sample_cluster(pdb_csv, seed=self._seed)

        # Training or validation specific logic.
        self.csv = pdb_csv
        self._log.info(
            f"{'Training' if self._is_training else 'Validation'} ({self.split}): {len(self.csv)} examples"
        )
        # if self.is_training:
        #     self.csv = pdb_csv
        #     self._log.info(f"Training: {len(self.csv)} examples")
        # else:
        #     eval_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.eval_max_len]
        #     all_lengths = np.sort(eval_csv.modeled_seq_len.unique())
        #     length_indices = (len(all_lengths) - 1) * np.linspace(
        #         0.0, 1.0, self.dataset_cfg.eval_num_lengths
        #     )
        #     length_indices = length_indices.astype(int)
        #     eval_lengths = all_lengths[length_indices]
        #     eval_csv = eval_csv[eval_csv.modeled_seq_len.isin(eval_lengths)]

        #     # Fix a random seed to get the same split each time.
        #     eval_csv = eval_csv.groupby("modeled_seq_len").sample(
        #         self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123
        #     )
        #     eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
        #     self.csv = eval_csv
        #     self._log.info(f"Validation: {len(self.csv)} examples with lengths {eval_lengths}")
        # self.crop_size = -1

    def sample_cluster(self, pdb_csv, seed):
        return pdb_csv.groupby("cluster").sample(1, random_state=seed)

    def _process_csv_row2(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats["modeled_idx"]
        processed_feats = tree.map_structure(
            lambda x: x[min_idx : (max_idx + 1)], processed_feats
        )

        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": torch.tensor(processed_feats["aatype"]).long(),
            "all_atom_positions": torch.tensor(
                processed_feats["atom_positions"]
            ).double(),
            "all_atom_mask": torch.tensor(
                processed_feats["atom_mask"]
            ).double(),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats["rigidgroups_gt_frames"]
        )[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_idx = processed_feats["residue_index"]
        return {
            "aatype": chain_feats["aatype"],
            "res_idx": res_idx - np.min(res_idx) + 1,
            "rotmats_1": rotmats_1,
            "trans_1": trans_1,
            "res_mask": torch.tensor(processed_feats["bb_mask"]).int(),
        }

    # @fn.lru_cache(maxsize=100)
    # def _process_csv_row(self, processed_file_path):
    #     processed_feats = du.read_pkl(processed_file_path)
    #     final_feats = PdbDataset.process_chain(
    #         processed_feats,
    #         random_crop=self.crop_size > 0,
    #         crop_size=self.crop_size,
    #     )
    #     return final_feats

    @staticmethod
    def process_chain(chain_feats: dict, random_crop=False, crop_size=256):
        processed_feats = du.parse_chain_feats(chain_feats)

        gvp_feat = processed_feats.pop("gvp_feat", None)

        # Only take modeled residues.
        modeled_idx = processed_feats.pop("modeled_idx")
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        modeled_seq_len = max_idx - min_idx + 1
        # processed_feats = tree.map_structure(
        #     lambda x: x[min_idx : (max_idx + 1)], processed_feats
        # )
        processed_feats = tree.map_structure(
            lambda x: x[min_idx : (max_idx + 1)], processed_feats
        )

        if "plddt" in chain_feats:

            def crop_by_conf(processed_feats, plddt, threshold=70):
                modeled_mask = plddt > threshold
                modeled_idx = np.where(modeled_mask)[0]
                min_modeled_idx = np.min(modeled_idx)
                max_modeled_idx = np.max(modeled_idx)
                processed_feats = tree.map_structure(
                    lambda x: x[min_modeled_idx : (max_modeled_idx + 1)],
                    processed_feats,
                )
                return processed_feats

            processed_feats = crop_by_conf(
                processed_feats, chain_feats["plddt"]
            )

        processed_feats["modeled_idx"] = modeled_idx

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats["residue_index"]
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = (
            np.array(random.sample(all_chain_idx, len(all_chain_idx)))
            - np.min(all_chain_idx)
            + 1
        )
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(
                int
            )
            new_res_idx = (
                new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask
            )

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": torch.LongTensor(processed_feats["aatype"]),
            "seq_mask": torch.FloatTensor(processed_feats["bb_mask"]),
            # "seq_length": torch.LongTensor(processed_feats["bb_mask"]).sum(),
            "seq_length": torch.tensor(modeled_seq_len),
            "all_atom_positions": torch.FloatTensor(
                processed_feats["atom_positions"]
            ),
            "all_atom_mask": torch.FloatTensor(processed_feats["atom_mask"]),
            "residue_index": torch.LongTensor(new_res_idx),
            "chain_index": torch.LongTensor(chain_idx),
        }

        if gvp_feat is not None:
            chain_feats["gvp_feat"] = torch.FloatTensor(gvp_feat)

        if random_crop:
            chain_feats = data_transforms.random_crop_to_size(
                crop_size, 0, SHAPE_SCHEMA
            )(chain_feats)
            chain_feats = data_transforms.make_fixed_size(
                SHAPE_SCHEMA, 0, 0, crop_size, 0
            )(chain_feats)
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        chain_feats = data_transforms.make_pseudo_beta("")(chain_feats)
        chain_feats = data_transforms.get_backbone_frames(chain_feats)

        # only modeling backbone
        MODELED_ATOMS = 5
        chain_feats["all_atom_mask"][..., MODELED_ATOMS:] = 0.0
        chain_feats["atom14_atom_exists"][..., MODELED_ATOMS:] = 0.0
        chain_feats["atom37_atom_exists"][..., MODELED_ATOMS:] = 0.0

        # To speed up processing, only take necessary features
        final_feats = {
            "aatype": chain_feats["aatype"],
            "residue_index": chain_feats["residue_index"],
            "seq_length": chain_feats["seq_length"],
            "chain_index": chain_feats["chain_index"],
            "res_mask": chain_feats["seq_mask"],
            "all_atom_positions": chain_feats["all_atom_positions"],
            "all_atom_mask": chain_feats["all_atom_mask"],
            "atom14_gt_positions": chain_feats["atom14_gt_positions"],
            "atom14_atom_exists": chain_feats["atom14_atom_exists"],
            "pseudo_beta": chain_feats["pseudo_beta"],
            "pseudo_beta_mask": chain_feats["pseudo_beta_mask"],
            "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
            "rigidgroups_gt_frames": chain_feats["rigidgroups_gt_frames"],
            "rigidgroups_gt_exists": chain_feats["rigidgroups_gt_exists"],
            "backbone_rigid_tensor": chain_feats["backbone_rigid_tensor"],
            "backbone_rigid_mask": chain_feats["backbone_rigid_mask"],
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
        }

        if gvp_feat is not None:
            final_feats["gvp_feat"] = chain_feats["gvp_feat"]
        return final_feats

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        chain_feats = self._process_csv_row(csv_row)
        chain_feats["csv_idx"] = torch.ones(1, dtype=torch.long) * csv_row.name

        return chain_feats

    def _process_csv_row(self, csv_row):
        path = csv_row["processed_path"].format(
            data_dir=self.dataset_cfg.data_dir
        )
        seq_len = csv_row["modeled_seq_len"]

        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]

        processed_feats = du.read_pkl(path)

        if self._is_training and self._dataset_cfg.get("load_gvp_feat"):
            gvp_path = csv_row["gvp_feat_path"].format(
                data_dir=self.dataset_cfg.data_dir
            )
            processed_feats["gvp_feat"] = torch.load(
                gvp_path, map_location="cpu"
            )

        if (
            self._is_training
            and self._dataset_cfg.filter.get("conf_masking")
            and csv_row.split == "afdb_swissprot"
        ):
            try:
                _read_plddt = lambda _ss: np.array(
                    list(map(float, _ss.split(",")))
                )
                processed_feats["plddt"] = plddt = _read_plddt(
                    csv_row["plddt"]
                )
                modeled_mask = plddt > 70
                processed_feats["atom_positions"][~modeled_mask] = np.nan
            except Exception as e:
                print(csv_row.pdb_name, csv_row["plddt"])

        processed_feats = PdbDataset.process_chain(
            processed_feats,
            random_crop=self.crop_size > 0
            and processed_feats["aatype"].shape[0] > self.crop_size,
            crop_size=self.crop_size,
        )
        # processed_feats["pdb_name"] = csv_row["pdb_name"]

        if use_cache:
            self._cache[path] = processed_feats
        return processed_feats


class LengthBatcher:
    def __init__(
        self,
        *,
        sampler_cfg,
        metadata_csv,
        seed=123,
        shuffle=True,
        num_replicas=None,
        rank=None,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv["index"] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(
            f"Created dataloader rank {self.rank+1} out of {self.num_replicas}"
        )
        self._create_batches()

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(
                len(self._data_csv), generator=rng
            ).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank :: self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv

        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby("modeled_seq_len"):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            shuffled_len_df = len_df.sample(frac=1).reset_index(drop=True)
            for i in range(num_batches):
                batch_df = shuffled_len_df.iloc[
                    i * max_batch_size : (i + 1) * max_batch_size
                ]
                batch_indices = batch_df["index"].tolist()
                sample_order.append(batch_indices)

        # Remove any length bias.
        new_order = (
            torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        )
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError("Exceeded number of augmentations.")
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[: self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        yield from iter(self.sample_order)
        self.epoch += 1
        self._create_batches()

    def __len__(self):
        return len(self.sample_order)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y
    return pred_y
