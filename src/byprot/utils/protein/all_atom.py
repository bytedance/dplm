# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for calculating all atom representations.

Code adapted from OpenFold.
"""

import torch
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils as ru

from byprot.datamodules.pdb_dataset import utils as du

Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.


IDEALIZED_POS = torch.tensor(
    residue_constants.restype_atom14_rigid_group_positions
)
DEFAULT_FRAMES = torch.tensor(
    residue_constants.restype_rigid_group_default_frame
)
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)


def to_atom37(trans, rots):
    num_batch, num_res, _ = trans.shape
    final_atom37 = compute_backbone(
        du.create_rigid(rots, trans),
        torch.zeros(num_batch, num_res, 2, device=trans.device),
    )[0]
    return final_atom37


def torsion_angles_to_frames(
    r: Rigid,  # type: ignore [valid-type]
    alpha: torch.Tensor,
    aatype: torch.Tensor,
):
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.
    """
    # [*, N, 8, 4, 4]
    with torch.no_grad():
        default_4x4 = DEFAULT_FRAMES.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)  # type: ignore [attr-defined]

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)  # type: ignore [index]

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(
        prot_feats
    )
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


def frames_to_atom14_pos(
    r: Rigid,  # type: ignore [valid-type]
    aatype: torch.Tensor,
):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:
    """
    with torch.no_grad():
        group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]
        group_mask = torch.nn.functional.one_hot(
            group_mask,
            num_classes=DEFAULT_FRAMES.shape[-3],
        )
        frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1)  # type: ignore [attr-defined]
        frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask  # type: ignore [index]

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions):
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1),
    )
    aatype = torch.zeros(bb_rigids.shape, device=bb_rigids.device).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
    )
    atom14_pos = frames_to_atom14_pos(all_frames, aatype)
    atom37_bb_pos = torch.zeros(
        bb_rigids.shape + (37, 3), device=bb_rigids.device
    )
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle


def vector_projection(R_ab, P_n):
    """Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def transrot_to_atom37(transrot_traj, res_mask):
    atom37_traj = []
    res_mask = res_mask.detach().cpu()
    num_batch = res_mask.shape[0]
    for trans, rots in transrot_traj:
        rigids = du.create_rigid(rots, trans)
        atom37 = compute_backbone(
            rigids,
            torch.zeros(
                trans.shape[0], trans.shape[1], 2, device=trans.device
            ),
        )[0]
        atom37 = atom37.detach().cpu()
        batch_atom37 = []
        for i in range(num_batch):
            batch_atom37.append(du.adjust_oxygen_pos(atom37[i], res_mask[i]))
        atom37_traj.append(torch.stack(batch_atom37))
    return atom37_traj


def atom37_from_trans_rot(trans, rots, res_mask):
    rigids = du.create_rigid(rots, trans)
    atom37 = compute_backbone(
        rigids,
        torch.zeros(trans.shape[0], trans.shape[1], 2, device=trans.device),
    )[0]
    atom37 = atom37.detach().cpu()
    batch_atom37 = []
    num_batch = res_mask.shape[0]
    for i in range(num_batch):
        batch_atom37.append(adjust_oxygen_pos(atom37[i], res_mask[i]))
    return torch.stack(batch_atom37)


def process_trans_rot_traj(trans_traj, rots_traj, res_mask):
    res_mask = res_mask.detach().cpu()
    atom37_traj = [
        atom37_from_trans_rot(trans, rots, res_mask)
        for trans, rots in zip(trans_traj, rots_traj)
    ]
    atom37_traj = torch.stack(atom37_traj).swapaxes(0, 1)
    return atom37_traj


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
