import argparse
import os
from pprint import pprint

import biotite.sequence.io.fasta as fasta
import numpy as np
import pandas as pd
import torch
from peft.peft_model import PeftModel

from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel
from byprot.utils.scaffold_utils import *
from generate_dplm2 import save_fasta


@torch.no_grad()
def generate(args, saveto):
    model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
        args.model_name
    )
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    # Read motif fasta file
    with open(args.motif_aa, "r") as f:
        fasta_file = fasta.FastaFile.read(f)
        motif_aa_seq = dict(fasta_file.items())
    with open(args.motif_struct, "r") as f:
        fasta_file = fasta.FastaFile.read(f)
        motif_struct_seq = dict(fasta_file.items())

    for ori_pdb_name, pdb_name in motif_name_mapping.items():
        struct_seq = motif_struct_seq[pdb_name]
        aa_seq = motif_aa_seq[pdb_name]
        max_iter = args.max_iter
        (
            batches,
            start_idxs_list,
            end_idxs_list,
            scaffold_lengths_list,
        ) = get_initial_dplm2(
            args,
            list(aa_seq),
            struct_seq.split(","),
            tokenizer,
            pdb_name,
            ori_pdb_name,
            device,
        )

        output_tokens = torch.tensor([], device=device)
        for batch in batches:
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_tokens=batch["input_ids"],
                    max_iter=max_iter,
                    sampling_strategy=args.sampling_strategy,
                    partial_masks=batch["partial_mask"],
                )["output_tokens"]
            output_tokens = torch.concat([output_tokens, outputs])
        assert output_tokens.shape[0] == len(start_idxs_list)
        print("final:")
        pprint(
            [
                ",".join(seq.split(" "))
                for seq in tokenizer.batch_decode(
                    output_tokens, skip_special_tokens=False
                )
            ]
        )

        # save output
        scaffold_fasta_path = os.path.join(saveto, "scaffold_fasta")
        os.makedirs(scaffold_fasta_path, exist_ok=True)
        scaffold_info_path = os.path.join(saveto, "scaffold_info")
        os.makedirs(scaffold_info_path, exist_ok=True)

        # save scaffold fasta
        save_results(
            output_tokens=output_tokens,
            save_dir=os.path.join(scaffold_fasta_path, ori_pdb_name),
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=True,
        )

        # save scaffold info
        struct_tokens, aa_tokens = output_tokens.chunk(2, dim=-1)
        aa_strings = [
            "".join(seq.split(" "))
            for seq in tokenizer.batch_decode(
                aa_tokens, skip_special_tokens=True
            )
        ]
        struct_strings = [
            ",".join(seq.split(" "))
            for seq in tokenizer.batch_decode(
                struct_tokens, skip_special_tokens=True
            )
        ]
        save_df = pd.DataFrame(
            list(
                zip(
                    aa_strings,
                    struct_strings,
                    start_idxs_list,
                    end_idxs_list,
                    scaffold_lengths_list,
                )
            ),
            columns=[
                "aa_seqs",
                "struct_seqs",
                "start_idxs",
                "end_idxs",
                "scaffold_lengths",
            ],
        )
        save_df.to_csv(
            os.path.join(scaffold_info_path, f"{ori_pdb_name}.csv"),
            index=False,
        )


def save_results(
    tokenizer,
    struct_tokenizer,
    save_dir,
    output_tokens,
    headers=None,
    save_pdb=False,
    continue_write=False,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}...")
    if headers is None:
        headers = [f"sample_{i}" for i in range(len(output_tokens))]

    struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
    aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
    struct_tokens_strings = list(
        map(
            lambda s: ",".join(s.split()),
            tokenizer.batch_decode(struct_tokens, skip_special_tokens=True),
        )
    )
    aatype_strings = list(
        map(
            lambda s: "".join(s.split()),
            tokenizer.batch_decode(aatype_tokens, skip_special_tokens=True),
        )
    )
    save_fasta(
        save_name=aatype_fasta_path,
        output_results=aatype_strings,
        headers=headers,
        continue_write=continue_write,
    )
    if save_pdb:
        pdb_save_dir = os.path.join(save_dir, "pdb")
        os.makedirs(pdb_save_dir, exist_ok=True)
        for header, aatype_str, struct_tokens_str in zip(
            headers, aatype_strings, struct_tokens_strings
        ):
            (
                aatype_tensor,
                struct_tokens_tensor,
            ) = struct_tokenizer.string_to_tensor(
                aatype_str, struct_tokens_str
            )
            decoder_out = struct_tokenizer.detokenize(struct_tokens_tensor)
            decoder_out["aatype"] = aatype_tensor
            decoder_out["header"] = [header]

            struct_tokenizer.output_to_pdb(
                decoder_out, output_dir=pdb_save_dir
            )

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seqs", type=int, default=20)
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="annealing@2.0:1.0"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm2_650m"
    )
    parser.add_argument(
        "--motif_aa",
        type=str,
        default="./data-bin/scaffolding-pdbs/aa_seq.fasta",
    )
    parser.add_argument(
        "--motif_struct",
        type=str,
        default="./data-bin/scaffolding-pdbs/struct_seq.fasta",
    )

    args = parser.parse_args()
    pprint(args)

    generate(args, args.saveto)


if __name__ == "__main__":
    main()
