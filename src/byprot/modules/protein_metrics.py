
# Copyright (c) 2022 Jason Yim, Brian L Trippe, Valentin De Bortoli, Emile Mathieu
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Xinyou Wang on Jul 21, 2024
#
# Original file was released under MIT, with the full license text
# available at https://github.com/jasonkyuyim/se3_diffusion/blob/master/LICENSE
#
# This modified file is released under the same license.


""" Metrics. """
import mdtraj as md
import numpy as np

# from openfold.np.relax import amber_minimize
import tree

# from analysis import utils as au
# from data import utils as du
from openfold.np import residue_constants
from tmtools import tm_align

CA_IDX = residue_constants.atom_order['CA']

INTER_VIOLATION_METRICS = [
    'bonds_c_n_loss_mean',
    'angles_ca_c_n_loss_mean',
    'clashes_mean_loss',
]

SHAPE_METRICS = [
    'coil_percent',
    'helix_percent',
    'strand_percent',
    'radius_of_gyration'
]

CA_VIOLATION_METRICS = [
    'ca_ca_bond_dev',
    'ca_ca_valid_percent',
    'ca_steric_clash_percent',
    'num_ca_steric_clashes',
]

EVAL_METRICS = [
    'tm_score', 
]

ALL_METRICS = (
    INTER_VIOLATION_METRICS
    + SHAPE_METRICS
    + CA_VIOLATION_METRICS
    + EVAL_METRICS
)

def calc_tm_score(pos_1, pos_2, seq_1, seq_2, mask):
    # nan_mask = np.isnan(pos_1)
    pos_1 = pos_1[mask]
    pos_2 = pos_2[mask]

    seq_1 = seq_1[:pos_1.shape[0]]
    seq_2 = seq_2[:pos_1.shape[0]]

    tm_results = tm_align(np.float64(pos_1), np.float64(pos_2), seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

def calc_perplexity(pred, labels, mask):
    one_hot_labels = np.eye(pred.shape[-1])[labels]
    true_probs = np.sum(pred * one_hot_labels, axis=-1)
    ce = -np.log(true_probs + 1e-5)
    per_res_perplexity = np.exp(ce)
    return np.sum(per_res_perplexity * mask) / np.sum(mask)

def calc_mdtraj_metrics(pdb_path):
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == 'C')
    pdb_helix_percent = np.mean(pdb_ss == 'H')
    pdb_strand_percent = np.mean(pdb_ss == 'E')
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
    pdb_rg = md.compute_rg(traj)[0]
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected


def calc_aligned_rmsd(pos_1, pos_2):
    nan_mask = np.isnan(pos_1)
    pos_1 = pos_1[~nan_mask].reshape(-1, 3)
    pos_2 = pos_2[~nan_mask].reshape(-1, 3)
    aligned_pos_1 = rigid_transform_3D(np.float64(pos_1), np.float64(pos_2))[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

def protein_metrics(
        *,
        pdb_path,
        atom37_pos,
        gt_atom37_pos,
        gt_aatype,
        diffuse_mask,
    ):

    # SS percentage
    mdtraj_metrics = calc_mdtraj_metrics(pdb_path)
    atom37_mask = np.any(atom37_pos, axis=-1)
    atom37_diffuse_mask = diffuse_mask[..., None] * atom37_mask
    prot = au.create_full_prot(atom37_pos, atom37_diffuse_mask)
    violation_metrics = amber_minimize.get_violation_metrics(prot)
    struct_violations = violation_metrics['structural_violations']
    inter_violations = struct_violations['between_residues']

    # Geometry
    bb_mask = np.any(atom37_mask, axis=-1)
    ca_pos = atom37_pos[..., CA_IDX, :][bb_mask.astype(bool)]
    ca_ca_bond_dev, ca_ca_valid_percent = ca_ca_distance(ca_pos)
    num_ca_steric_clashes, ca_steric_clash_percent = ca_ca_clashes(ca_pos)

    # Eval
    bb_diffuse_mask = (diffuse_mask * bb_mask).astype(bool)
    unpad_gt_scaffold_pos = gt_atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    unpad_pred_scaffold_pos = atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    seq = du.aatype_to_seq(gt_aatype[bb_diffuse_mask])
    _, tm_score = calc_tm_score(
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, seq, seq)

    metrics_dict = {
        'ca_ca_bond_dev': ca_ca_bond_dev,
        'ca_ca_valid_percent': ca_ca_valid_percent,
        'ca_steric_clash_percent': ca_steric_clash_percent,
        'num_ca_steric_clashes': num_ca_steric_clashes,
        'tm_score': tm_score,
        **mdtraj_metrics,
    }
    for k in INTER_VIOLATION_METRICS: 
        metrics_dict[k] = inter_violations[k]
    metrics_dict = tree.map_structure(lambda x: np.mean(x).item(), metrics_dict)
    return metrics_dict 

def ca_ca_distance(ca_pos, tol=0.1):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + tol))
    return ca_ca_dev, ca_ca_valid

def ca_ca_clashes(ca_pos, tol=1.5):
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < tol
    return np.sum(clashes), np.mean(clashes)
