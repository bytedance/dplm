
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
import esm
from pytorch_lightning import seed_everything
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataChunk, Dataset
import heapq
from torch import distributed as dist
import torch


class Alphabet(object):
    def __init__(self, name='esm', featurizer='cath', alphabet_cfg={}, featurizer_cfg={}):
        self.name = name
        self._alphabet = None

        if name == 'esm':
            self._alphabet = esm.Alphabet.from_architecture('ESM-1b')
            self.add_special_tokens = True
        elif name == 'protein_mpnn':
            self._alphabet = esm.Alphabet(
                standard_toks=['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
                prepend_toks=["<pad>", "<unk>"],
                append_toks=[],
                prepend_bos=False,
                append_eos=False
            )
            self.add_special_tokens = False
        else:
            self._alphabet = esm.Alphabet(**alphabet_cfg)
            self.add_special_tokens = self._alphabet.prepend_bos and self._alphabet.append_eos

        self._featurizer = self.get_featurizer(featurizer, **featurizer_cfg)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._alphabet, name)
        except:
            raise AttributeError(f"{self.__class__} has no attribute `{name}`.")

    def __len__(self):
        return len(self._alphabet)

    def get_featurizer(self, name='cath', **kwds):
        if name == 'cath':
            from .cath import Featurizer
            return Featurizer(alphabet=self, 
                              to_pifold_format=kwds.get('to_pifold_format', False),
                              coord_nan_to_zero=kwds.get('coord_nan_to_zero', True))
        elif name == 'multichain':
            from .multichain import Featurizer
            return Featurizer(self, **kwds)

    @property
    def featurizer(self):
        return self._featurizer

    def featurize(self, raw_batch, **kwds):
        return self._featurizer(raw_batch, **kwds)

    def decode(self, batch_ids, return_as='str', remove_special=False):
        ret = []
        for ids in batch_ids.cpu():
            if return_as == 'str':
                line = ''.join([self.get_tok(id) for id in ids])
                if remove_special:
                    line = line.replace(self.get_tok(self.mask_idx), '_') \
                        .replace(self.get_tok(self.eos_idx), '') \
                        .replace(self.get_tok(self.cls_idx), '') \
                        .replace(self.get_tok(self.padding_idx), '') \
                        .replace(self.get_tok(self.unk_idx), '-')
            elif return_as == 'list':
                line = [self.get_tok(id) for id in ids]
            ret.append(line)
        return ret


class PDBDataProcessor(object):
    def parse_PDB(self, path_to_pdb, input_chain_list=None, masked_chain_list=None, ca_only=False):
        c=0
        pdb_dict_list = []
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet + extra_alphabet
        
        if input_chain_list:
            chain_alphabet = input_chain_list  
        if not masked_chain_list: # mask all chains to design the entire complex
            masked_chain_list = chain_alphabet
        masked_list, visible_list = [], []

        biounit_names = [path_to_pdb]
        for biounit in biounit_names:
            my_dict = {}
            s = 0
            concat_seq = ''
            concat_coords = []
            concat_N = []
            concat_CA = []
            concat_C = []
            concat_O = []
            concat_mask = []
            coords_dict = {}
            for letter in chain_alphabet:
                if ca_only:
                    sidechain_atoms = ['CA']
                else:
                    sidechain_atoms = ['N', 'CA', 'C', 'O']
                xyz, seq = self.parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
                if type(xyz) != str:
                    concat_seq += seq[0]
                    my_dict['seq_chain_'+letter]=seq[0]
                    coords_dict_chain = {}
                    if ca_only:
                        coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                    else:
                        coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                        coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                        coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                        coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                    my_dict['coords_chain_'+letter] = coords_dict_chain
                    concat_coords.append(xyz.astype(np.float32))
                    if letter in masked_chain_list:
                        masked_list.append(letter)
                    else:
                        visible_list.append(letter)
                    s += 1
            fi = biounit.rfind("/")
            my_dict['name']=biounit[(fi+1):-4]
            my_dict['num_of_chains'] = s
            my_dict['seq'] = concat_seq
            my_dict['coords'] = np.concatenate(concat_coords, axis=0)
            my_dict['masked_list'] = masked_list
            my_dict['visible_list'] = visible_list
            if s <= len(chain_alphabet):
                pdb_dict_list.append(my_dict)
                c+=1
        return pdb_dict_list[0]

    def parse_PDB_biounits(self, x, atoms=['N','CA','C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''

        alpha_1 = list("ARNDCQEGHILKMFPSTWYVX")
        states = len(alpha_1)
        alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
        
        aa_1_N = {a:n for n,a in enumerate(alpha_1)}
        aa_3_N = {a:n for n,a in enumerate(alpha_3)}
        aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
        aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
        aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
        
        def AA_to_N(x):
            # ["ARND"] -> [[0,1,2,3]]
            x = np.array(x);
            if x.ndim == 0: x = x[None]
            return [[aa_1_N.get(a, states-1) for a in y] for y in x]
        
        def N_to_AA(x):
            # [[0,1,2,3]] -> ["ARND"]
            x = np.array(x);
            if x.ndim == 1: x = x[None]
            return ["".join([aa_N_1.get(a,"X") for a in y]) for y in x]

        xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
        for line in open(x,"rb"):
            line = line.decode("utf-8","ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17+3] == "MSE":
                line = line.replace("HETATM","ATOM  ")
                line = line.replace("MSE","MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:12+4].strip()
                    resi = line[17:17+3]
                    resn = line[22:22+5].strip()
                    x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

                    if resn[-1].isalpha(): 
                        resa,resn = resn[-1],int(resn[:-1])-1
                    else: 
                        resa,resn = "",int(resn)-1
            #         resn = int(resn)
                    if resn < min_resn: 
                        min_resn = resn
                    if resn > max_resn: 
                        max_resn = resn
                    if resn not in xyz: 
                        xyz[resn] = {}
                    if resa not in xyz[resn]: 
                        xyz[resn][resa] = {}
                    if resn not in seq: 
                        seq[resn] = {}
                    if resa not in seq[resn]: 
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x,y,z])

        # convert to numpy arrays, fill in missing values
        seq_,xyz_ = [],[]
        try:
            for resn in range(min_resn,max_resn+1):
                if resn in seq:
                    for k in sorted(seq[resn]): 
                        seq_.append(aa_3_N.get(seq[resn][k],20))
                else: 
                    seq_.append(20)
                if resn in xyz:
                    for k in sorted(xyz[resn]):
                        for atom in atoms:
                            if atom in xyz[resn][k]: 
                                xyz_.append(xyz[resn][k][atom])
                            else: 
                                xyz_.append(np.full(3,np.nan))
                else:
                    for atom in atoms: xyz_.append(np.full(3,np.nan))
            return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
        except TypeError:
            return 'no_chain', 'no_chain'
        

T_co = TypeVar("T_co", covariant=True)


def identity(example):
    return example

class MaxTokensBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        max_tokens=6000,
        drop_last=False,
        distributed=False,
        sort_key: Callable = None,
        sort=False,
        buffer_size_multiplier=100,
        seed=42,
        shuffle=True
    ):
        self.distibuted = distributed
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)

        self.max_tokens = max_tokens
        self.sort_key = sort_key
        self.sort = sort
        self.max_buffer_size = max_tokens * buffer_size_multiplier

        self._epoch = 0
        self._seed = seed
        self.shuffle = shuffle

        self.bucket_batches = []
        self._num_samples = None
        self._build_batches()

    def __len__(self):
        return len(self.bucket_batches)

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        for batch, batch_size in self.bucket_batches:
            yield batch

    def _build_batches(self):
        buffer: List = []
        buffer_size: int = 0

        batch: List = []
        batch_size: int = 0

        bucket_batches = []

        if self.sort:
            indices = sorted(list(self.sampler), key=self.sort_key)
        else:
            indices = self.sampler

        for index in indices:
            # 1) add to buffer
            length = self.sort_key(index)
            heapq.heappush(buffer, (length, index))
            buffer_size += length

            # 2) create batches in the sorted buffer once buffer is full
            if buffer_size > self.max_buffer_size:
                length, index = heapq.heappop(buffer)
                buffer_size -= length
                if batch_size + length > self.max_tokens:
                    bucket_batches.append((batch, batch_size))
                    batch, batch_size = [], 0
                batch.append(index)
                batch_size += length

        # 3) create batches from the rest data in the buffer
        while buffer:
            length, index = heapq.heappop(buffer)
            # print(length, index)
            if batch_size + length > self.max_tokens:
                bucket_batches.append((batch, batch_size))
                batch, batch_size = [], 0
            batch.append(index)
            batch_size += length

        # 4) the last batch
        if batch:
            bucket_batches.append((batch, batch_size))

        # 5) maybe shuffle
        if self.shuffle:
            seed_everything(self._seed + self._epoch)
            np.random.shuffle(bucket_batches)

        # 6) carefully deal with DDP, ensuring that every worker
        #      has the same number of batches
        if self.distibuted:
            num_samples = torch.tensor(len(bucket_batches)).to(self.sampler.rank)
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            num_samples = num_samples.item()

            if len(bucket_batches) < num_samples:
                padding_size = num_samples - len(bucket_batches)
                bucket_batches += bucket_batches[:padding_size]

        self.bucket_batches = bucket_batches

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distibuted:
            self.sampler.set_epoch(epoch)

