# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Xinyou Wang on May 15, 2025
#
# Original file was released under MIT, with the full license text
# available at https://github.com/jasonkyuyim/multiflow/blob/main/LICENSE
#
# This modified file is released under the same license.


import collections
import dataclasses
import os
import pickle
import string
from typing import Any, Dict, List

import mdtraj as md
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBIO, MMCIFParser
from Bio.PDB.Chain import Chain
from openfold.utils import rigid_utils as ru
from torch_scatter import scatter, scatter_add

from byprot.datamodules.pdb_dataset import protein, residue_constants

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "b_factors",
]

NUM_TOKENS = residue_constants.restype_num
MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index("X")
CA_IDX = residue_constants.atom_order["CA"]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: "".join(
    [residue_constants.restypes_with_x[x] for x in aatype]
)
seq_to_aatype = lambda seq: [
    residue_constants.restypes_with_x.index(x) for x in seq
]

CHAIN_FEATS = [
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "b_factors",
]
UNPADDED_FEATS = [
    "t",
    "rot_score_scaling",
    "trans_score_scaling",
    "t_seq",
    "t_struct",
    "csv_idx",
    "seq_length",
    "pdb_name",
    "pdb_path",
]
RIGID_FEATS = [
    "rigids_0",
    "rigids_t",
    # 'rigidgroups_gt_frames', 'backbone_rigid_tensor'
]
PAIR_FEATS = ["rel_rots"]


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
        and hasattr(feat, "shape")
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(
                padded_feats[feat_name], max_len, pad_idx=1
            )
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    for feat_name in raw_feats:
        if feat_name in UNPADDED_FEATS or isinstance(
            raw_feats[feat_name], str
        ):
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats


def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = ru.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False
    )
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)


def pad(x: np.ndarray, max_len: int, dim=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[dim]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[dim] = (pad_amt, 0)
    else:
        pad_widths[dim] = (0, pad_amt)
    if use_torch:
        pad_widths = reversed(pad_widths)
        pad_widths = tuple(
            [item for pad_width in pad_widths for item in pad_width]
        )
        # print(pad_widths)
        return torch.nn.functional.pad(x, pad_widths)
    return np.pad(x, pad_widths)


class DataError(Exception):
    """Data exception."""

    pass


class FileExistsError(DataError):
    """Raised when file already exists."""

    pass


class MmcifParsingError(DataError):
    """Raised when mmcif parsing fails."""

    pass


class ResolutionError(DataError):
    """Raised when resolution isn't acceptable."""

    pass


class LengthError(DataError):
    """Raised when length isn't acceptable."""

    pass


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError("pos_1 and pos_2 must have the same shape.")
    if pos_1.ndim != 3:
        raise ValueError(f"Expected inputs to have shape [B, N, 3]")
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64)
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2
        )
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask],
    )
    aligned_pos_1 = torch.bmm(pos_1, align_rots)
    return aligned_pos_1, pos_2, align_rots


def adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known=None
) -> torch.Tensor:
    """Imputes the position of the oxygen atom on the backbone by using
    adjacent frame information. Specifically, we say that the oxygen atom is in
    the plane created by the Calpha and C from the current frame and the
    nitrogen of the next frame. The oxygen is then placed c_o_bond_length
    Angstrom away from the C in the current frame in the direction away from
    the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (
        atom_37[:-1, 2, :] - atom_37[:-1, 1, :]
    ) / (
        torch.norm(
            atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1
        )
        + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (
        atom_37[:-1, 2, :] - atom_37[1:, 0, :]
    ) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1)
        + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = (
        calpha_to_carbonyl + nitrogen_to_carbonyl
    )  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (
        atom_37[:, 2, :] - atom_37[:, 1, :]
    ) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1)
        + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (
        atom_37[:, 0, :] - atom_37[:, 1, :]
    ) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1)
        + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones(
            (atom_37.shape[0],), dtype=torch.int64, device=atom_37.device
        )

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()],
        dim=0,
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :]
        + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
    )

    return atom_37


def write_pkl(
    save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False
):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(
            pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL
        )
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(
                    f"Failed to read {read_path}. First error: {e}\n Second error: {e2}"
                )
            raise (e)


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def parse_chain_feats(chain_feats, scale_factor=1.0):
    nan_atom_mask = np.isnan(chain_feats["atom_positions"].sum(-1))
    chain_feats["atom_mask"][nan_atom_mask] = 0
    chain_feats["atom_positions"] = np.nan_to_num(
        chain_feats["atom_positions"], nan=0.0
    )

    ca_idx = residue_constants.atom_order["CA"]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, ca_idx]
    bb_pos = chain_feats["atom_positions"][:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (
        np.sum(chain_feats["bb_mask"]) + 1e-5
    )
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = (
        scaled_pos * chain_feats["atom_mask"][..., None]
    )
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, ca_idx]
    return chain_feats


def concat_np_features(
    np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool
):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def center_zero(
    pos: torch.Tensor, batch_indexes: torch.LongTensor
) -> torch.Tensor:
    """Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert (
        len(pos.shape) == 2 and pos.shape[-1] == 3
    ), "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """Align structures in a ChemGraph batch to a reference, e.g. for RMSD
    computation. This uses the sparse formulation of pytorch geometric. If the
    ChemGraph is composed of a single system, then the reference can be given
    as a single structure and broadcasted. Returns the structure coordinates
    shifted to the geometric center and the batch structures rotated to match
    the reference structures. Uses the Kabsch algorithm (see e.g.

    [kabsch_align1]_). No permutation of atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None],
        batch_indices,
        dim=0,
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices


def process_mmcif(mmcif_path: str, max_resolution: int, max_len: int):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    mmcif_name = os.path.basename(mmcif_path).replace(".cif", "")
    metadata["pdb_name"] = mmcif_name
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f"{mmcif_name}.pkl")
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    metadata["processed_path"] = processed_mmcif_path
    with open(mmcif_path, "r") as f:
        parsed_mmcif = mmcif_parsing.parse(
            file_id=mmcif_name, mmcif_string=f.read()
        )
    metadata["raw_path"] = mmcif_path
    if parsed_mmcif.errors:
        raise MmcifParsingError(f"Encountered errors {parsed_mmcif.errors}")
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if "_pdbx_struct_assembly.oligomeric_count" in raw_mmcif:
        raw_olig_count = raw_mmcif["_pdbx_struct_assembly.oligomeric_count"]
        oligomeric_count = ",".join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if "_pdbx_struct_assembly.oligomeric_details" in raw_mmcif:
        raw_olig_detail = raw_mmcif["_pdbx_struct_assembly.oligomeric_details"]
        oligomeric_detail = ",".join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata["oligomeric_count"] = oligomeric_count
    metadata["oligomeric_detail"] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header["resolution"]
    metadata["resolution"] = mmcif_resolution
    metadata["structure_method"] = mmcif_header["structure_method"]
    if mmcif_resolution >= max_resolution:
        raise ResolutionError(f"Too high resolution {mmcif_resolution}")
    if mmcif_resolution == 0.0:
        raise ResolutionError(f"Invalid resolution {mmcif_resolution}")

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in parsed_mmcif.structure.get_chains()
    }
    metadata["num_chains"] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = chain_str_to_int(chain_id)
        chain_prot = process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"
    complex_feats = concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise LengthError("No modeled residues")
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata["seq_len"] = len(complex_aatype)
    metadata["modeled_seq_len"] = max_modeled_idx - min_modeled_idx + 1
    complex_feats["modeled_idx"] = modeled_idx
    if complex_aatype.shape[0] > max_len:
        raise LengthError(f"Too long {complex_aatype.shape[0]}")

    chain_dict["ss"] = pdb_ss[0]
    metadata["coil_percent"] = (
        np.sum(pdb_ss == "C") / metadata["modeled_seq_len"]
    )
    metadata["helix_percent"] = (
        np.sum(pdb_ss == "H") / metadata["modeled_seq_len"]
    )
    metadata["strand_percent"] = (
        np.sum(pdb_ss == "E") / metadata["modeled_seq_len"]
    )

    # Radius of gyration
    metadata["radius_gyration"] = pdb_dg[0]

    # Return metadata
    return metadata


def process_pdb_file(file_path: str):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace(".pdb", "")
    metadata["pdb_name"] = pdb_name

    metadata["raw_path"] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain for chain in structure.get_chains()
    }
    metadata["num_chains"] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = chain_str_to_int(chain_id)
        chain_prot = process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"
    complex_feats = concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    metadata["seq_len"] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise LengthError("No modeled residues")
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata["modeled_seq_len"] = max_modeled_idx - min_modeled_idx + 1
    complex_feats["modeled_idx"] = modeled_idx

    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
    except Exception as e:
        raise DataError(f"Mdtraj failed with error {e}")

    chain_dict["ss"] = pdb_ss[0]
    metadata["coil_percent"] = (
        np.sum(pdb_ss == "C") / metadata["modeled_seq_len"]
    )
    metadata["helix_percent"] = (
        np.sum(pdb_ss == "H") / metadata["modeled_seq_len"]
    )
    metadata["strand_percent"] = (
        np.sum(pdb_ss == "E") / metadata["modeled_seq_len"]
    )

    # Radius of gyration
    metadata["radius_gyration"] = pdb_dg[0]

    # Return metadata
    return complex_feats, metadata


def parse_pdb_feats(
    pdb_name: str,
    pdb_path: str,
    scale_factor=1.0,
    # TODO: Make the default behaviour read all chains.
    chain_id="A",
):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {chain.id: chain for chain in structure.get_chains()}

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {x: _process_chain_id(x) for x in chain_id}
    elif chain_id is None:
        return {x: _process_chain_id(x) for x in struct_chains}
    else:
        raise ValueError(f"Unrecognized chain list {chain_id}")


def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[
                residue_constants.atom_order[atom.name]
            ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )
