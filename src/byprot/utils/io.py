
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import numpy as np
from scipy.spatial import transform
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List


from biotite.structure import filter_amino_acids


def filter_backbone2(array):
    """
    Filter all peptide backbone atoms of one array.

    This includes the "N", "CA" and "C" atoms of amino acids.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        as an backbone atom.
    """
    return ( ((array.atom_name == "N") |
              (array.atom_name == "CA") |
              (array.atom_name == "C") |
              (array.atom_name == "O")) &
              filter_amino_acids(array) )

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    # bbmask = filter_backbone(structure)
    bbmask = filter_backbone2(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray, atoms=["N", "CA", "C"]):
    """
    Args:
        structure: An instance of biotite AtomArray
        atoms: default ["N", "CA", "C"]
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    # coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    coords = get_atom_coords_residuewise(atoms, structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain, atoms=["N", "CA", "C", "O"]):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure, atoms=atoms)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def save_pdb(path, coords, seq):
    pass