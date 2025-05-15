import os
import random
from copy import deepcopy
from pprint import pprint

import esm
import esm.inverse_folding
import torch

from byprot import utils
from byprot.datamodules.dataset.data_utils import PDBDataProcessor

STRUCT_TYPE = 0
AA_TYPE = 1

single_res = ["1qjg"]

scaffold_left = {
    "1PRW": [5, 20],
    "1BCF": [8, 15],
    "5TPN": [10, 40],
    "5IUS": [0, 30],
    "3IXT": [10, 40],
    "5YUI": [5, 30],
    "1QJG": [10, 20],
    "1YCR": [10, 40],
    "2KL8": [0, 0],
    "7MRX_60": [0, 38],
    "7MRX_85": [0, 63],
    "7MRX_128": [0, 122],
    "4JHW": [10, 25],
    "4ZYP": [10, 40],
    "5WN9": [10, 40],
    "5TRV_short": [0, 35],
    "5TRV_med": [0, 65],
    "5TRV_long": [0, 95],
    "6E6R_short": [0, 35],
    "6E6R_med": [0, 65],
    "6E6R_long": [0, 95],
    "6EXZ_short": [0, 35],
    "6EXZ_med": [0, 65],
    "6EXZ_long": [0, 95],
}
scaffold_right = {
    "1PRW": [5, 20],
    "1BCF": [8, 15],
    "5TPN": [10, 40],
    "5IUS": [0, 30],
    "3IXT": [10, 40],
    "5YUI": [10, 30],
    "1QJG": [10, 20],
    "1YCR": [10, 40],
    "2KL8": [0, 0],
    "7MRX_60": [0, 38],
    "7MRX_85": [0, 63],
    "7MRX_128": [0, 122],
    "4JHW": [10, 25],
    "4ZYP": [10, 40],
    "5WN9": [10, 40],
    "5TRV_short": [0, 35],
    "5TRV_med": [0, 65],
    "5TRV_long": [0, 95],
    "6E6R_short": [0, 35],
    "6E6R_med": [0, 65],
    "6E6R_long": [0, 95],
    "6EXZ_short": [0, 35],
    "6EXZ_med": [0, 65],
    "6EXZ_long": [0, 95],
}
motif_name_mapping = {
    "1PRW": "1prw",
    "1BCF": "1bcf",
    "5TPN": "5tpn",
    "5IUS": "5ius",
    "3IXT": "3ixt",
    "5YUI": "5yui",
    "1QJG": "1qjg",
    "1YCR": "1ycr",
    "2KL8": "2kl8",
    "7MRX_60": "7mrx",
    "7MRX_85": "7mrx",
    "7MRX_128": "7mrx",
    "4JHW": "4jhw",
    "4ZYP": "4zyp",
    "5WN9": "5wn9",
    "5TRV_short": "5trv",
    "5TRV_med": "5trv",
    "5TRV_long": "5trv",
    "6E6R_short": "6e6r",
    "6E6R_med": "6e6r",
    "6E6R_long": "6e6r",
    "6EXZ_short": "6exz",
    "6EXZ_med": "6exz",
    "6EXZ_long": "6exz",
}
scaffold_interval = {
    "1PRW": [[10, 25]],
    "1BCF": [[16, 30], [16, 30], [16, 30]],
    "5IUS": [[15, 40]],
    "5YUI": [[5, 20], [10, 35]],
    "1QJG": [[15, 30], [15, 30]],
    "2KL8": [[20, 20]],
    "4JHW": [[15, 30]],
}
total_length = {
    "1PRW": -1,
    "1BCF": -1,
    "5TPN": [50, 75],
    "5IUS": -1,
    "3IXT": [50, 75],
    "5YUI": [50, 100],
    "1QJG": -1,
    "1YCR": [40, 100],
    "2KL8": -1,
    "7MRX_60": [60, 61],
    "7MRX_85": [85, 86],
    "7MRX_128": [128, 129],
    "4JHW": [60, 90],
    "4ZYP": [30, 50],
    "5WN9": [35, 50],
    "5TRV_short": [56, 57],
    "5TRV_med": [86, 87],
    "5TRV_long": [116, 117],
    "6E6R_short": [48, 49],
    "6E6R_med": [78, 79],
    "6E6R_long": [108, 109],
    "6EXZ_short": [50, 51],
    "6EXZ_med": [80, 81],
    "6EXZ_long": [110, 111],
}

start_idx_dict = {
    "1prw": [15, 51],
    "1bcf": [90, 122, 46, 17],
    "5tpn": [108],
    "3ixt": [0],
    "4jhw": [144, 37],
    "4zyp": [357],
    "5wn9": [1],
    "5ius": [88, 34],
    "5yui": [89, 114, 194],
    "6vw1": [5, 45],
    "1qjg": [37, 13, 98],
    "1ycr": [2],
    "2kl8": [0, 27],
    "7mrx": [25],
    "5trv": [45],
    "6e6r": [22],
    "6exz": [25],
}
end_idx_dict = {
    "1prw": [34, 70],
    "1bcf": [98, 129, 53, 24],
    "5tpn": [126],
    "3ixt": [23],
    "4jhw": [159, 43],
    "4zyp": [371],
    "5wn9": [20],
    "5ius": [109, 53],
    "5yui": [93, 116, 196],
    "6vw1": [23, 63],
    "1qjg": [37, 13, 98],
    "1ycr": [10],
    "2kl8": [6, 78],
    "7mrx": [46],
    "5trv": [69],
    "6e6r": [34],
    "6exz": [39],
}

chain_dict = {
    "1prw": "A",
    "1bcf": "A",
    "5tpn": "A",
    "3ixt": "P",
    "4jhw": "F",
    "4zyp": "A",
    "5wn9": "A",
    "5ius": "A",
    "5yui": "A",
    "6vw1": "A",
    "1qjg": "A",
    "1ycr": "B",
    "2kl8": "A",
    "7mrx": "B",
    "5trv": "A",
    "6e6r": "A",
    "6exz": "A",
}


def get_intervals(list, single_res_domain=False):
    """Given a list (Tensor) of non-masked residues get new start and end index
    for motif placed in scaffold."""
    if single_res_domain:
        start = [l.item() for l in list]
        stop = start
    else:
        start = []
        stop = []
        for i, item in enumerate(list):
            if i == 0:
                start.append(item.item())
            elif i == (len(list) - 1):
                stop.append(item.item())
            elif i != len(list) and (item + 1) != list[i + 1]:
                stop.append(item.item())
                start.append(list[i + 1].item())
    return start, stop


def get_motif(pdb_name, ori_pdb_name, mask_token, spacer_list=None):
    # Get motif of sequence from PDB file
    start_idxs = start_idx_dict[pdb_name]
    end_idxs = end_idx_dict[pdb_name]

    pdb_clean_path = os.path.join(
        "data-bin/scaffolding-pdbs/" + str(pdb_name) + "_clean.pdb"
    )
    chain = chain_dict[pdb_name]
    chain_ids = [chain]
    print("WARNING: USING CHAIN", chain, "FROM PDB FILE")
    structure = esm.inverse_folding.util.load_structure(
        pdb_clean_path, chain_ids
    )
    native_seqs = (
        esm.inverse_folding.multichain_util.extract_coords_from_complex(
            structure
        )[1]
    )
    sequence = native_seqs[chain_ids[0]]
    print("sequence extracted from pdb", sequence)
    print("sequence length", len(sequence))
    assert len(start_idxs) == len(end_idxs)
    sequence = list(sequence)

    if spacer_list is None:
        spacer_list = []
    end_idxs = [i + 1 for i in end_idxs]  # inclusive of final residue
    if len(start_idxs) > 1:
        motif = []
        for i in range(len(start_idxs)):
            motif += sequence[start_idxs[i] : end_idxs[i]]
            if i < (len(start_idxs) - 1):
                if spacer_list is None:
                    interval_start = scaffold_interval[ori_pdb_name][i][0]
                    interval_end = scaffold_interval[ori_pdb_name][i][1]
                    spacer = random.randint(interval_start, interval_end)
                    spacer_list.append(spacer)
                else:
                    spacer = spacer_list[i]
                motif += [mask_token] * spacer
    else:
        motif = sequence[start_idxs[0] : end_idxs[0]]
    print("motif extracted from indexes supplied:", "".join(motif))

    return motif, spacer_list


# ====================================================================
# ==================== For DPLM motif-scaffolding ====================
# ====================================================================


def prepare_data(pdb_path, alphabet, collator, num_seqs, device):
    def _full_mask(target_tokens, coord_mask, alphabet):
        target_mask = (
            target_tokens.ne(alphabet.padding_idx)  # & mask
            & target_tokens.ne(alphabet.cls_idx)
            & target_tokens.ne(alphabet.eos_idx)
        )
        _tokens = target_tokens.masked_fill(target_mask, alphabet.mask_idx)
        _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
        return _tokens, _mask

    structure = PDBDataProcessor().parse_PDB(pdb_path)
    batch = collator([deepcopy(structure) for idx in range(num_seqs)])
    prev_tokens, prev_token_mask = _full_mask(
        batch["tokens"], batch["coord_mask"], alphabet
    )
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_tokens.eq(alphabet.mask_idx)
    batch = utils.recursive_to(batch, device=device)
    return batch, structure["seq"]


def get_initial_dplm(args, tokenizer, pdb, ori_pdb, device):
    num = args.num_seqs
    motif, _ = get_motif(pdb, ori_pdb, mask_token=tokenizer.mask_token)

    mask = tokenizer.mask_token_id
    bos = tokenizer.cls_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    init_seq = []
    scaffold_length_list = []
    for i in range(num):
        ## Process length
        length_compatible = False
        while length_compatible is False:
            scaffold_left_length = random.randint(
                scaffold_left[ori_pdb][0], scaffold_left[ori_pdb][1]
            )

            motif = get_motif(
                pdb_name=pdb,
                ori_pdb_name=ori_pdb,
                mask_token=tokenizer.mask_token,
            )
            motif_overall_length = len(motif)

            if total_length[ori_pdb] != -1:
                current_length_range = [
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[ori_pdb][0],
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[ori_pdb][1],
                ]
                total_length_range = [
                    total_length[ori_pdb][0],
                    total_length[ori_pdb][1],
                ]
                length_range = [
                    max(current_length_range[0], total_length_range[0]),
                    min(current_length_range[1], total_length_range[1]),
                ]
                # NOT compatible
                if length_range[0] > length_range[1]:
                    continue
                length_compatible = True
                scaffold_right_length = random.randint(
                    length_range[0], length_range[1]
                ) - (scaffold_left_length + motif_overall_length)
            else:
                length_compatible = True
                scaffold_right_length = random.randint(
                    scaffold_right[ori_pdb][0], scaffold_right[ori_pdb][1]
                )

            overall_length = (
                scaffold_left_length
                + motif_overall_length
                + scaffold_right_length
            )
            scaffold_length_list.append(
                scaffold_left_length + scaffold_right_length
            )

        seq = (
            [tokenizer.mask_token] * scaffold_left_length
            + motif
            + [tokenizer.mask_token] * scaffold_right_length
        )
        assert len(seq) == overall_length
        seq = "".join(seq)
        init_seq.append(seq)

    batch = tokenizer.batch_encode_plus(
        init_seq,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    batch = {
        "input_ids": batch["input_ids"],
        "input_mask": batch["attention_mask"].bool(),
    }
    batch = utils.recursive_to(batch, device)

    single_res_domain = pdb in single_res

    start_idxs_list = []
    end_idxs_list = []
    for seq in batch["input_ids"]:
        nonmask_locations = (
            (seq != mask) & (seq != bos) & (seq != eos) & (seq != pad)
        ).nonzero().flatten() - 1
        new_start_idxs, new_end_idxs = get_intervals(
            nonmask_locations, single_res_domain=single_res_domain
        )
        start_idxs_list.append(new_start_idxs)
        end_idxs_list.append(new_end_idxs)
    pprint(batch)
    return batch, start_idxs_list, end_idxs_list, scaffold_length_list


# ====================================================================
# =================== For DPLM-2 motif-scaffolding ===================
# ====================================================================


def get_initial_dplm2(
    args, aa_seq, struct_seq, tokenizer, pdb, ori_pdb, device
):
    init_aa_seq, init_struct_seq, scaffold_length_list = create_init_seq(
        pdb, ori_pdb, aa_seq, struct_seq, tokenizer, args
    )

    batch = collate(tokenizer, init_aa_seq, init_struct_seq, args, device)
    pprint(batch)

    start_idxs_list, end_idxs_list = create_idxs_list(
        pdb, tokenizer, batch, args
    )

    batches = create_batches(batch, args)

    return batches, start_idxs_list, end_idxs_list, scaffold_length_list


def create_init_seq(pdb, ori_pdb, aa_seq, struct_seq, tokenizer, args):
    num = args.num_seqs
    aa_mask_token = tokenizer.aa_mask_token
    aa_bos_token = tokenizer.aa_cls_token
    aa_eos_token = tokenizer.aa_eos_token
    struct_mask_token = tokenizer.struct_mask_token
    struct_bos_token = tokenizer.struct_cls_token
    struct_eos_token = tokenizer.struct_eos_token

    init_aa_seq = []
    init_struct_seq = []
    scaffold_length_list = []
    for i in range(num):
        ## Process length
        length_compatible = False
        while length_compatible is False:
            scaffold_left_length = random.randint(
                scaffold_left[ori_pdb][0], scaffold_left[ori_pdb][1]
            )

            motif_aa_seq, spacer_list = get_motif(
                pdb_name=pdb,
                ori_pdb_name=ori_pdb,
                mask_token=aa_mask_token,
            )
            motif_overall_length = len(motif_aa_seq)

            if total_length[ori_pdb] != -1:
                current_length_range = [
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[ori_pdb][0],
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[ori_pdb][1],
                ]
                total_length_range = [
                    total_length[ori_pdb][0],
                    total_length[ori_pdb][1],
                ]
                length_range = [
                    max(current_length_range[0], total_length_range[0]),
                    min(current_length_range[1], total_length_range[1]),
                ]
                # NOT compatible
                if length_range[0] > length_range[1]:
                    continue
                length_compatible = True
                scaffold_right_length = random.randint(
                    length_range[0], length_range[1]
                ) - (scaffold_left_length + motif_overall_length)
            else:
                length_compatible = True
                scaffold_right_length = random.randint(
                    scaffold_right[ori_pdb][0], scaffold_right[ori_pdb][1]
                )

            overall_length = (
                scaffold_left_length
                + motif_overall_length
                + scaffold_right_length
            )
            scaffold_length_list.append(
                scaffold_left_length + scaffold_right_length
            )

        ## motif aa seq initialization
        seq = (
            [aa_bos_token]
            + [aa_mask_token] * scaffold_left_length
            + motif_aa_seq
            + [aa_mask_token] * scaffold_right_length
            + [aa_eos_token]
        )
        seq = "".join(seq)
        assert len(
            tokenizer(seq, add_special_tokens=False, padding=False)[
                "input_ids"
            ]
        ) == (overall_length + 2)
        init_aa_seq.append(seq)

        ## motif struct seq initialization
        motif_struct_seq, _ = get_motif(
            pdb_name=pdb,
            ori_pdb_name=ori_pdb,
            mask_token=struct_mask_token,
            spacer_list=spacer_list,
        )
        seq = (
            [struct_bos_token]
            + [struct_mask_token] * scaffold_left_length
            + motif_struct_seq
            + [struct_mask_token] * scaffold_right_length
            + [struct_eos_token]
        )
        seq = "".join(seq)
        assert len(
            tokenizer(seq, add_special_tokens=False, padding=False)[
                "input_ids"
            ]
        ) == (overall_length + 2)
        init_struct_seq.append(seq)

    return init_aa_seq, init_struct_seq, scaffold_length_list


def collate(tokenizer, init_aa_seq, init_struct_seq, args, device):
    batch_aa = tokenizer.batch_encode_plus(
        init_aa_seq,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_aa = {
        "aa_ids": batch_aa["input_ids"],
        "aa_mask": batch_aa["attention_mask"].bool(),
        "aa_targets": batch_aa["input_ids"].clone(),
    }

    batch_struct = tokenizer.batch_encode_plus(
        init_struct_seq,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_struct = {
        "struct_ids": batch_struct["input_ids"],
        "struct_mask": batch_struct["attention_mask"].bool(),
        "struct_targets": batch_struct["input_ids"].clone(),
    }
    batch = {
        "input_ids": torch.cat(
            (batch_struct["struct_ids"], batch_aa["aa_ids"]), dim=-1
        ),
        "input_mask": torch.cat(
            (batch_struct["struct_mask"], batch_aa["aa_mask"]), dim=-1
        ),
        "targets": torch.cat(
            (batch_struct["struct_targets"], batch_aa["aa_targets"]), dim=-1
        ),
    }
    batch.update(batch_struct)
    batch.update(batch_aa)

    # HACK: all amino acid token id < 33, while all struct token id >= 33
    # 0 stands for struct, 1 stands for aa
    batch["type_ids"] = ((batch["input_ids"] < 33) & batch["input_mask"]).int()
    # 2 stands for padding
    batch["type_ids"].masked_fill_(~batch["input_mask"], 2)
    batch = utils.recursive_to(batch, device)

    # create partial mask
    aa_mask_idx = tokenizer.added_tokens_encoder[tokenizer.aa_mask_token]
    struct_mask_idx = tokenizer.added_tokens_encoder[
        tokenizer.struct_mask_token
    ]
    partial_mask = (
        batch["input_ids"].ne(aa_mask_idx)
        & batch["input_ids"].ne(struct_mask_idx)
        & batch["input_ids"].ne(tokenizer.pad_token_id)
    ).type_as(batch["input_mask"])

    batch["partial_mask"] = partial_mask

    return batch


def create_idxs_list(pdb, tokenizer, batch, args):
    # special tokens
    aa_mask_token = tokenizer.aa_mask_token
    aa_bos_token = tokenizer.aa_cls_token
    aa_eos_token = tokenizer.aa_eos_token

    single_res_domain = pdb in single_res

    start_idxs_list = []
    end_idxs_list = []
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.added_tokens_encoder[aa_mask_token]
    bos_id = tokenizer.added_tokens_encoder[aa_bos_token]
    eos_id = tokenizer.added_tokens_encoder[aa_eos_token]
    get_intervals_seqs = batch["aa_ids"]

    for seq in get_intervals_seqs:
        nonmask_locations = (
            (seq != mask_id)
            & (seq != bos_id)
            & (seq != eos_id)
            & (seq != pad_id)
        ).nonzero().flatten() - 1
        new_start_idxs, new_end_idxs = get_intervals(
            nonmask_locations, single_res_domain=single_res_domain
        )
        start_idxs_list.append(new_start_idxs)
        end_idxs_list.append(new_end_idxs)

    return start_idxs_list, end_idxs_list


def create_batches(batch, args):
    num = args.num_seqs
    batches = []
    start = 0
    end = start + args.batch_size
    while end < num:
        new_batch = {}
        for k, v in batch.items():
            new_batch[k] = v[start:end]
        batches.append(new_batch)
        start += args.batch_size
        end += args.batch_size
    assert end >= num
    # last batch if necessary
    if start < num:
        last_batch = {}
        for k, v in batch.items():
            last_batch[k] = v[start:end]
        batches.append(last_batch)
    return batches
