import os
import warnings
from glob import glob
import argparse

import torch
import torch.utils
from biotite.sequence.io import fasta

from tqdm.auto import tqdm

from byprot.datamodules.pdb_dataset import utils as du
from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset, collate_fn
from byprot.utils import recursive_to, get_logger
from byprot.models.utils import get_struct_tokenizer

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")
log = get_logger(__name__)


def load_from_pdb(pdb_path, process_chain=PdbDataset.process_chain):
    raw_chain_feats, metadata = du.process_pdb_file(pdb_path)
    chain_feats = process_chain(raw_chain_feats)
    chain_feats["pdb_name"] = metadata["pdb_name"]
    return chain_feats


@torch.no_grad()
def run_tokenize(struct_tokenizer, input_pdb_folder, output_dir):
    all_data = []

    for pdb_path in glob(os.path.join(input_pdb_folder, "*.pdb")):
        log.info(f"Processing {pdb_path}")

        # predicted structures
        feats = load_from_pdb(pdb_path, process_chain=struct_tokenizer.process_chain)
        feats["pdb_path"] = pdb_path
        feats["header"] = feats["pdb_name"]

        all_data.append(feats)

    dataloader = torch.utils.data.DataLoader(
        all_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    all_header_struct_seq = []
    all_header_aa_seq = []
    pbar = tqdm(dataloader)
    device = next(struct_tokenizer.parameters()).device
    for batch in pbar:
        pdb_name = batch["pdb_name"][0]
        pbar.set_description(f"Tokenize: {pdb_name} (L={batch['seq_length'][0]})")
        batch = recursive_to(batch, device)

        struct_ids = struct_tokenizer.tokenize(batch['all_atom_positions'], batch['res_mask'], batch['seq_length'])
        struct_seq = struct_tokenizer.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
        all_header_struct_seq.append((pdb_name, struct_seq))

        aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
        all_header_aa_seq.append((pdb_name, aa_seq))

    output_struct_fasta_path = os.path.join(output_dir, "struct_seq.fasta")
    fasta.FastaFile.write_iter(output_struct_fasta_path, all_header_struct_seq)

    output_aa_fasta_path = os.path.join(output_dir, "aa_seq.fasta")
    fasta.FastaFile.write_iter(output_aa_fasta_path, all_header_aa_seq)
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pdb_folder", type=str, default="/path/to/input/pdb/folder")
    parser.add_argument("--output_dir", type=str, default="./generation-results/tokenized_protein")
    args = parser.parse_args()
    
    struct_tokenizer = get_struct_tokenizer()
    struct_tokenizer = struct_tokenizer.cuda()
    run_tokenize(struct_tokenizer, args.input_pdb_folder, args.output_dir)


if __name__ == "__main__":
    main()