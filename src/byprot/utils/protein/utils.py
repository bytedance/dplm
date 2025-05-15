"""Utility functions for experiments."""

import glob
import logging
import math
import os
import random
import re
import shutil
import subprocess

import GPUtil
import mdtraj as md
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from biotite.sequence.io import fasta
from openfold.np import residue_constants
from openfold.utils import rigid_utils
from openfold.utils import rigid_utils as ru
from openfold.utils.superimposition import superimpose
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.nn import functional as F

from byprot.datamodules.pdb_dataset import utils as du

CA_IDX = residue_constants.atom_order["CA"]


Rigid = rigid_utils.Rigid


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length + 1,
            self._samples_cfg.length_step,
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_subset]
        all_sample_ids = []
        num_batch = self._samples_cfg.num_batch
        if num_batch <= 0:
            num_batch = self._samples_cfg.samples_per_length
        assert self._samples_cfg.samples_per_length % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor(
                    [num_batch * sample_id + i for i in range(num_batch)]
                )
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            "sample_id": sample_id,
            "seq_length": torch.full((sample_id.shape[0],), num_res, dtype=torch.long),
            "seq": torch.zeros((sample_id.shape[0], num_res), dtype=torch.long),
        }
        return batch


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    )
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    )
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order="memory", limit=8)[:num_device]


def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        "foldseek",
        "easy-cluster",
        designable_dir,
        os.path.join(output_dir, "res"),
        output_dir,
        "--alignment-type",
        "1",
        "--cov-mode",
        "0",
        "--min-seq-id",
        "0",
        "--tmscore-threshold",
        "0.5",
    ]
    process = subprocess.Popen(
        easy_cluster_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    # print(stdout, stderr)
    del stdout  # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, "res_rep_seq.fasta"))
    return len(rep_seq_fasta)


def get_all_top_samples(output_dir, csv_fname="*/*/top_sample.csv"):
    all_csv_paths = glob.glob(os.path.join(output_dir, csv_fname), recursive=True)
    top_sample_csv = pd.concat([pd.read_csv(x) for x in all_csv_paths])
    top_sample_csv.to_csv(os.path.join(output_dir, "all_top_samples.csv"), index=False)
    return top_sample_csv


def calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path):
    designable_samples = top_sample_csv[top_sample_csv.designable]
    designable_dir = os.path.join(output_dir, "designable")
    os.makedirs(designable_dir, exist_ok=True)
    designable_txt = os.path.join(designable_dir, "designable.txt")
    if os.path.exists(designable_txt):
        os.remove(designable_txt)
    with open(designable_txt, "w") as f:
        for sample_id, (_, row) in enumerate(designable_samples.iterrows()):
            sample_path = row.sample_path
            sample_name = f"sample_id_{sample_id}_length_{row.length}.pdb"
            write_path = os.path.join(designable_dir, sample_name)
            shutil.copy(sample_path, write_path)
            f.write(write_path + "\n")
    if metrics_df["Total codesignable"].iloc[0] <= 1:
        metrics_df["Clusters"] = metrics_df["Total codesignable"].iloc[0]
    else:
        add_diversity_metrics(designable_dir, metrics_df, designable_csv_path)


def add_diversity_metrics(designable_dir, designable_csv, designable_csv_path):
    designable_txt = os.path.join(designable_dir, "designable.txt")
    clusters = run_easy_cluster(designable_dir, designable_dir)
    designable_csv["Clusters"] = clusters
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_consistency(output_dir, designable_csv, designable_csv_path):
    # output dir points to directory containing length_60, length_61, ... etc folders
    sample_dirs = glob.glob(os.path.join(output_dir, "length_*/sample_*"))
    average_accs = []
    max_accs = []
    for sample_dir in sample_dirs:
        pmpnn_fasta_path = os.path.join(
            sample_dir, "self_consistency", "seqs", "sample_modified.fasta"
        )
        codesign_fasta_path = os.path.join(
            sample_dir, "self_consistency", "codesign_seqs", "codesign.fa"
        )
        pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
        codesign_fasta = fasta.FastaFile.read(codesign_fasta_path)
        codesign_seq = codesign_fasta["codesign_seq_1"]
        accs = []
        for seq in pmpnn_fasta:
            num_matches = sum(
                [
                    1 if pmpnn_fasta[seq][i] == codesign_seq[i] else 0
                    for i in range(len(pmpnn_fasta[seq]))
                ]
            )
            total_length = len(pmpnn_fasta[seq])
            accs.append(num_matches / total_length)
        average_accs.append(np.mean(accs))
        max_accs.append(np.max(accs))
    designable_csv["Average PMPNN Consistency"] = np.mean(average_accs)
    designable_csv["Average Max PMPNN Consistency"] = np.mean(max_accs)
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_designability(
    output_dir, designable_csv, designable_csv_path, all_mpnn_folds_df_path="pmpnn_results.csv"
):
    sample_dirs = glob.glob(os.path.join(output_dir, "length_*/sample_*"))
    try:
        single_pmpnn_results = []
        top_pmpnn_results = []
        for sample_dir in sample_dirs:
            all_pmpnn_folds_df = pd.read_csv(os.path.join(sample_dir, all_mpnn_folds_df_path))
            single_pmpnn_fold_df = all_pmpnn_folds_df.iloc[[0]]
            single_pmpnn_results.append(single_pmpnn_fold_df)
            min_index = all_pmpnn_folds_df["bb_rmsd"].idxmin()
            top_pmpnn_df = all_pmpnn_folds_df.loc[[min_index]]
            top_pmpnn_results.append(top_pmpnn_df)
        single_pmpnn_results_df = pd.concat(single_pmpnn_results, ignore_index=True)
        top_pmpnn_results_df = pd.concat(top_pmpnn_results, ignore_index=True)
        designable_csv["Single seq PMPNN Designability"] = np.mean(
            # single_pmpnn_results_df["bb_rmsd"].to_numpy() < 2.0
            single_pmpnn_results_df["bb_tmscore"].to_numpy()
            >= 0.5
        )
        designable_csv["Top seq PMPNN Designability"] = np.mean(
            # top_pmpnn_results_df["bb_rmsd"].to_numpy() < 2.0
            top_pmpnn_results_df["bb_tmscore"].to_numpy()
            >= 0.5
        )
        designable_csv.to_csv(designable_csv_path, index=False)
    except Exception as e:
        # TODO i think it breaks when one process gets here first
        print(f"calculate pmpnn designability didnt work: {e}")


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {
        "node_id": node_id,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
    }


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([(f"{k}:{i}", j) for i, j in flatten_dict(v)])
        else:
            flattened.append((k, v))
    return flattened


def save_traj(
    sample: np.ndarray,
    bb_prot_traj: np.ndarray,
    x0_traj: np.ndarray,
    diffuse_mask: np.ndarray,
    output_dir: str,
    aa_traj=None,
    clean_aa_traj=None,
    write_trajectories=True,
    omit_missing_residue=False,
):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, "sample.pdb")
    prot_traj_path = os.path.join(output_dir, "bb_traj.pdb")
    x0_traj_path = os.path.join(output_dir, "x0_traj.pdb")

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    clean_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (clean_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert clean_aa_traj is not None
        assert clean_aa_traj.shape == (clean_traj_length, num_res)

    sample_path = write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
        omit_missing_residue=omit_missing_residue
    )
    if write_trajectories:
        prot_traj_path = write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj,
            omit_missing_residue=omit_missing_residue
        )
        x0_traj_path = write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=clean_aa_traj,
            omit_missing_residue=omit_missing_residue
        )
    return {
        "sample_path": sample_path,
        "traj_path": prot_traj_path,
        "x0_traj_path": x0_traj_path,
    }


def get_dataset_cfg(cfg):
    if cfg.data.dataset == "pdb":
        return cfg.pdb_dataset
    else:
        raise ValueError(f"Unrecognized dataset {cfg.data.dataset}")


import os
import re

import numpy as np

from byprot.datamodules.pdb_dataset import protein


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
    omit_missing_residue=True,
    atom37_mask=None,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(pos37, atom37_mask, aatype=aatype, b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            if atom37_mask is None:
                atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            if not omit_missing_residue:
                prot_pos[~atom37_mask] = np.nan
                atom37_mask[..., :3] = True
                atom37_mask[..., 4] = True
            prot = create_full_prot(prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path


# def rigids_to_se3_vec(frame, scale_factor=1.0):
#     trans = frame[:, 4:] * scale_factor
#     rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
#     se3_vec = np.concatenate([rotvec, trans], axis=-1)
#     return se3_vec


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def sinusoidal_encoding(v, N, D):
    """Taken from GENIE.

    Args:

    """
    # v: [*]

    # [D]
    k = torch.arange(1, D + 1).to(v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D]
    enc = torch.zeros_like(sin_enc).to(v.device)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc.to(v.dtype)


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5


def dist_from_ca(trans):

    # [b, n_res, n_res, 1]
    d = distance(
        torch.stack(
            [
                trans.unsqueeze(2).repeat(1, 1, trans.shape[1], 1),  # Ca_1
                trans.unsqueeze(1).repeat(1, trans.shape[1], 1, 1),  # Ca_2
            ],
            dim=-2,
        )
    ).unsqueeze(-1)

    return d


def calc_rbf(ca_dists, num_rbf, D_min=1e-3, D_max=22.0):
    # Distance radial basis function
    device = ca_dists.device
    D_mu = torch.linspace(D_min, D_max, num_rbf).to(device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-(((ca_dists - D_mu) / D_sigma) ** 2))


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def process_folded_outputs(sample_path, folded_output, true_bb_pos=None):
    mpnn_results = {
        "header": [],
        "sequence": [],
        "ca_rmsd": [],
        "bb_rmsd": [],
        "bb_tmscore": [],
        "mean_plddt": [],
        "folded_path": [],
    }

    if true_bb_pos is not None:
        true_ca_pos = true_bb_pos.reshape(-1, 3, 3)[..., CA_IDX, :]
        mpnn_results["ca_rmsd_to_gt"] = []

        mpnn_results["bb_rmsd_to_gt"] = []
        mpnn_results["bb_tmscore_to_gt"] = []

        mpnn_results["fold_model_bb_rmsd_to_gt"] = []

    sample_feats = du.parse_pdb_feats("sample", sample_path)
    sample_seq = du.aatype_to_seq(sample_feats["aatype"])
    sample_ca_pos = sample_feats["bb_positions"]
    sample_bb_pos = sample_feats["atom_positions"][:, :3].reshape(-1, 3)

    def _calc_ca_rmsd(mask, sample_ca_pos, folded_ca_pos):
        if '7W2P' not in sample_path:
            rmsd = superimpose(
            torch.tensor(sample_ca_pos)[None], torch.tensor(folded_ca_pos[None]), mask
        )[1].item()
        else:
            print("There is a superimpose error!")
            rmsd = 100.0
        return rmsd

    def _calc_bb_rmsd(mask, sample_bb_pos, folded_bb_pos):
        if '7W2P' not in sample_path and "2o0a_A" not in sample_path \
            and "4bg7_A" not in sample_path \
            and "4l8o_A" not in sample_path \
            and "5k29_A" not in sample_path:
            rmsd = superimpose(
                torch.tensor(sample_bb_pos)[None],
                torch.tensor(folded_bb_pos)[None],
                mask[:, None].repeat(1, 3).reshape(-1),
            )[1].item()
        else:
            print("There is a superimpose error!")
            rmsd = 100.0
        return rmsd

    def _calc_bb_tmscore(mask, sample_bb_pos, folded_bb_pos, sample_seq):
        if '7W2P' not in sample_path and "2o0a_A" not in sample_path \
            and "4bg7_A" not in sample_path \
            and "4l8o_A" not in sample_path \
            and "5k29_A" not in sample_path:
            bb_mask = mask[:, None].repeat(1, 3).bool()
            _sample_seq = "A" * mask.long().sum().item()
            _, tmscore = calc_tm_score(
                # torch.tensor(sample_bb_pos)[mask],
                # torch.tensor(folded_bb_pos)[mask],
                sample_bb_pos[bb_mask.reshape(-1)].reshape(-1, 3, 3),
                folded_bb_pos[bb_mask.reshape(-1)].reshape(-1, 3, 3),
                _sample_seq,
                _sample_seq,
            )
        else:
            print("There is a superimpose error!")
            tmscore = 0.0
        return tmscore

    if folded_output is None:
        folded_output = {
            "folded_path": [sample_path],
            "header": ["placeholder"],
            "plddt": [1.0],
            "seq": [sample_seq],
        }
        folded_output = pd.DataFrame(folded_output)

    for _, row in folded_output.iterrows():
        folded_feats = du.parse_pdb_feats("folded", row.folded_path)
        seq = du.aatype_to_seq(folded_feats["aatype"])
        folded_ca_pos = folded_feats["bb_positions"]
        folded_bb_pos = folded_feats["atom_positions"][:, :3].reshape(-1, 3)

        res_mask = torch.ones(folded_ca_pos.shape[0])

        if true_bb_pos is not None:
            res_mask = torch.tensor(true_ca_pos).abs().sum(-1) > 1e-7
            bb_rmsd_to_gt = _calc_bb_rmsd(res_mask, sample_bb_pos, true_bb_pos)
            ca_rmsd_to_gt = _calc_ca_rmsd(res_mask, sample_ca_pos, true_ca_pos)
            mpnn_results["bb_rmsd_to_gt"].append(bb_rmsd_to_gt)
            mpnn_results["ca_rmsd_to_gt"].append(ca_rmsd_to_gt)

            bb_tmscore_to_gt = _calc_bb_tmscore(
                res_mask, sample_bb_pos, true_bb_pos, sample_seq
            )
            mpnn_results["bb_tmscore_to_gt"].append(bb_tmscore_to_gt)

            fold_model_bb_rmsd_to_gt = _calc_bb_rmsd(res_mask, folded_bb_pos, true_bb_pos)
            mpnn_results["fold_model_bb_rmsd_to_gt"].append(fold_model_bb_rmsd_to_gt)

            # fold_model_bb_tmscore_to_gt = _calc_bb_tmscore(res_mask, folded_bb_pos, true_bb_pos, seq)
            # mpnn_results["fold_model_bb_tmscore_to_gt"].append(fold_model_bb_tmscore_to_gt)
        bb_rmsd = _calc_bb_rmsd(res_mask, sample_bb_pos, folded_bb_pos)
        ca_rmsd = _calc_ca_rmsd(res_mask, sample_ca_pos, folded_ca_pos)
        bb_tmscore = _calc_bb_tmscore(res_mask, sample_bb_pos, folded_bb_pos, seq)
        mpnn_results["bb_rmsd"].append(bb_rmsd)
        mpnn_results["ca_rmsd"].append(ca_rmsd)
        mpnn_results["bb_tmscore"].append(bb_tmscore)

        mpnn_results["folded_path"].append(row.folded_path)
        mpnn_results["header"].append(row.header)
        mpnn_results["sequence"].append(seq)
        mpnn_results["mean_plddt"].append(row.plddt)
    mpnn_results = pd.DataFrame(mpnn_results)
    mpnn_results["sample_path"] = sample_path
    return mpnn_results


def extract_clusters_from_maxcluster_out(file_path):
    # Extracts cluster information from the stdout of a maxcluster run
    cluster_to_paths = {}
    paths_to_cluster = {}
    read_mode = False
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line == "INFO  : Item     Cluster\n":
                read_mode = True
                continue

            if line == "INFO  : ======================================\n":
                read_mode = False

            if read_mode:
                # Define a regex pattern to match the second number and the path
                pattern = r"INFO\s+:\s+\d+\s:\s+(\d+)\s+(\S+)"

                # Use re.search to find the first match in the string
                match = re.search(pattern, line)

                # Check if a match is found
                if match:
                    # Extract the second number and the path
                    cluster_id = match.group(1)
                    path = match.group(2)
                    if cluster_id not in cluster_to_paths:
                        cluster_to_paths[cluster_id] = [path]
                    else:
                        cluster_to_paths[cluster_id].append(path)
                    paths_to_cluster[path] = cluster_id

                else:
                    raise ValueError(f"Could not parse line: {line}")

    return cluster_to_paths, paths_to_cluster


def calc_mdtraj_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == "C")
        pdb_helix_percent = np.mean(pdb_ss == "H")
        pdb_strand_percent = np.mean(pdb_ss == "E")
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
        pdb_rg = md.compute_rg(traj)[0]
    except IndexError as e:
        print("Error in calc_mdtraj_metrics: {}".format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        "non_coil_percent": pdb_ss_percent,
        "coil_percent": pdb_coil_percent,
        "helix_percent": pdb_helix_percent,
        "strand_percent": pdb_strand_percent,
        "radius_of_gyration": pdb_rg,
    }


def calc_aatype_metrics(generated_aatypes):
    # generated_aatypes (B, N)
    unique_aatypes, raw_counts = np.unique(generated_aatypes, return_counts=True)

    # pad with 0's in case it didn't generate any of a certain type
    clean_counts = []
    for i in range(20):
        if i in unique_aatypes:
            clean_counts.append(raw_counts[np.where(unique_aatypes == i)[0][0]])
        else:
            clean_counts.append(0)

    # from the scope128 dataset
    reference_normalized_counts = [
        0.0739,
        0.05378621,
        0.0410424,
        0.05732177,
        0.01418736,
        0.03995128,
        0.07562267,
        0.06695857,
        0.02163064,
        0.0580802,
        0.09333149,
        0.06777057,
        0.02034217,
        0.03673995,
        0.04428474,
        0.05987899,
        0.05502958,
        0.01228988,
        0.03233601,
        0.07551553,
    ]

    reference_normalized_counts = np.array(reference_normalized_counts)

    normalized_counts = clean_counts / np.sum(clean_counts)

    # compute the hellinger distance between the normalized counts
    # and the reference normalized counts

    hellinger_distance = np.sqrt(
        np.sum(np.square(np.sqrt(normalized_counts) - np.sqrt(reference_normalized_counts)))
    )

    return {"aatype_histogram_dist": hellinger_distance}


def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    ca_bond_dists = np.linalg.norm(ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))

    ca_ca_dists2d = np.linalg.norm(ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol
    return {
        "ca_ca_deviation": ca_ca_dev,
        "ca_ca_valid_percent": ca_ca_valid,
        "num_ca_ca_clashes": np.sum(clashes),
    }


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    from tmtools import tm_align

    tm_results = tm_align(np.float64(pos_1), np.float64(pos_2), seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2
