import argparse
import os

import torch
import tree
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

from byprot.models.dplm2 import DPLM2Bit
from byprot.models.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)


def initialize_conditional_generation(
    fasta_path, tokenizer, device, args, model=None
):
    input_data_aatype = []
    input_data_struct_tokens = []
    input_data_name = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        input_data_name.append(record.name)
        if args.task == "folding":
            aatype = str(record.seq)
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = tokenizer.struct_mask_token * len(record.seq)
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        elif args.task == "inverse_folding":
            aatype = tokenizer.aa_mask_token * len(record.seq.split(","))
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = "".join(str(record.seq).split(","))
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        else:
            raise NotImplementedError
        input_data_aatype.append(aatype)
        input_data_struct_tokens.append(struct_tokens)

    # sorted by length
    len_input = [len(seq) for seq in input_data_aatype]
    sorted_batch = sorted(
        zip(
            len_input,
            input_data_aatype,
            input_data_struct_tokens,
            input_data_name,
        )
    )
    _, aa, struct, name = zip(*sorted_batch)
    input_data_aatype = list(aa)
    input_data_struct_tokens = list(struct)
    input_data_name = list(name)

    def build_batch(input_data_aatype, input_data_struct_tokens):
        batch_struct = tokenizer.batch_encode_plus(
            input_data_struct_tokens,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_aa = tokenizer.batch_encode_plus(
            input_data_aatype,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(device)

        aa_type = 1
        struct_type = 0
        non_special = model.get_non_special_symbol_mask(input_tokens)
        type_ids = model.get_modality_type(input_tokens)

        # folding
        if args.task == "folding":
            # mask struct token
            input_tokens.masked_fill_(
                (type_ids == struct_type) & non_special,
                tokenizer._token_to_id[tokenizer.struct_mask_token],
            )
            mask_type = aa_type
        # inverse folding
        elif args.task == "inverse_folding":
            # mask aa token
            input_tokens.masked_fill_(
                (type_ids == aa_type) & non_special,
                tokenizer._token_to_id[tokenizer.aa_mask_token],
            )
            mask_type = struct_type

        # construct batch
        batch = {}
        batch["input_tokens"] = input_tokens
        batch["partial_mask"] = type_ids == mask_type

        return batch

    batches = []
    input_data_name_list = []
    # split batch according to args.batch_size
    if args.batch_size > 0:
        num = len(input_data_aatype)
        start = 0
        end = start + args.batch_size
        while end < num + args.batch_size:
            input_data_aa_batch = input_data_aatype[start:end]
            input_data_struct_batch = input_data_struct_tokens[start:end]
            new_batch = build_batch(
                input_data_aa_batch, input_data_struct_batch
            )
            batches.append(new_batch)
            input_data_name_list.append(input_data_name[start:end])
            start += args.batch_size
            end += args.batch_size
    else:
        batches = [build_batch(input_data_aatype, input_data_struct_tokens)]
        input_data_name_list = [input_data_name]

    return batches, input_data_name_list


def initialize_generation(
    task, num_seqs, length, tokenizer, device, batch_size=50
):
    def create_init_seq(length):
        if task == "sequence_generation":
            seq = tokenizer.aa_mask_token * length
            seq = tokenizer.aa_cls_token + seq + tokenizer.aa_eos_token
        elif task in ["co_generation", "backbone_generation"]:
            seq_struct = tokenizer.all_tokens[50] * length
            seq_aa = "A" * length
            seq_struct = (
                tokenizer.struct_cls_token
                + seq_struct
                + tokenizer.struct_eos_token
            )
            seq_aa = tokenizer.aa_cls_token + seq_aa + tokenizer.aa_eos_token
            seq = (seq_struct, seq_aa)
        else:
            raise NotImplementedError

        return seq

    init_struct_list = []
    init_aa_list = []
    for _ in range(num_seqs):
        seq = create_init_seq(length)
        if type(seq) == tuple:
            seq_struct, seq_aa = seq
            seq = seq_struct + seq_aa
        init_struct_list.append(seq_struct)
        init_aa_list.append(seq_aa)

    input_tokens_batch = []
    start = 0
    end = start + batch_size
    while end < num_seqs + batch_size:
        input_data_struct_tokens = init_struct_list[start:end]
        input_data_aatype = init_aa_list[start:end]
        batch_struct = tokenizer.batch_encode_plus(
            input_data_struct_tokens,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_aatype = tokenizer.batch_encode_plus(
            input_data_aatype,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(device)
        input_tokens_batch.append(input_tokens)
        start += batch_size
        end += batch_size

    return input_tokens_batch


def unconditional_generate(args):
    if args.bit_model:
        model = DPLM2Bit.from_pretrained(args.model_name)
    else:
        model = DPLM2.from_pretrained(args.model_name)

    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    for seq_len in args.seq_lens:
        max_iter = args.max_iter
        input_tokens_batch = initialize_generation(
            task=args.task,
            num_seqs=args.num_seqs,
            length=seq_len,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            all_outputs = {}
            for input_tokens in input_tokens_batch:
                _struct_tokens, _aatype_tokens = input_tokens.chunk(2, dim=1)
                if args.task == "backbone_generation":
                    input_tokens = _struct_tokens
                if args.task == "sequence_generation":
                    input_tokens = _aatype_tokens
                outputs = model.generate(
                    input_tokens=input_tokens,
                    max_iter=max_iter,
                    temperature=args.temperature,
                    unmasking_strategy=args.unmasking_strategy,
                    sampling_strategy=args.sampling_strategy,
                )
                if args.task == "backbone_generation":
                    _struct_tokens = outputs["output_tokens"].chunk(2, dim=1)[
                        0
                    ]
                    outputs["output_tokens"] = torch.cat(
                        [_struct_tokens, _aatype_tokens], dim=1
                    )
                for k, v in outputs.items():
                    if k in all_outputs:
                        all_outputs[k] = torch.concat(
                            [all_outputs[k], v], dim=0
                        )
                    else:
                        all_outputs[k] = v

        print("final:")
        if args.task == "backbone_generation":
            print(
                [
                    ",".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        elif args.task == "sequence_generation":
            print(
                [
                    "".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        elif args.task == "co_generation":
            print(
                [
                    ",".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        else:
            raise NotImplementedError

        # save
        save_results(
            outputs=all_outputs,
            task=args.task,
            save_dir=os.path.join(args.saveto, args.task, f"length_{seq_len}"),
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
        )


def conditional_generate_from_fasta(args):
    if args.bit_model:
        model = DPLM2Bit.from_pretrained(args.model_name)
    else:
        model = DPLM2.from_pretrained(args.model_name)

    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    batches, name_lists = initialize_conditional_generation(
        args.input_fasta_path, tokenizer, device, args=args, model=model
    )

    for i, batch in enumerate(tqdm(batches, desc=f"{args.task}")):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                temperature=args.temperature,
                unmasking_strategy=args.unmasking_strategy,
                sampling_strategy=args.sampling_strategy,
                partial_masks=batch["partial_mask"],
            )

        save_results(
            outputs=outputs,
            task=args.task,
            save_dir=os.path.join(args.saveto, args.task),
            headers=name_lists[i],
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
            continue_write=True,
        )


def save_fasta(
    save_name,
    output_results,
    struct_tokens=False,
    headers=None,
    continue_write=False,
):
    fp_save = (
        open(save_name, "w") if not continue_write else open(save_name, "a")
    )
    for idx, seq in enumerate(output_results):
        if headers is not None:
            fp_save.write(f">{headers[idx]}\n")
        else:
            fp_save.write(f">SEQUENCE_{idx}\n")
        seq = seq.split(" ")
        if struct_tokens:
            fp_save.write(f"{','.join(seq)}\n")
        else:
            fp_save.write(f"{''.join(seq)}\n")
    fp_save.close()


def save_results(
    tokenizer,
    struct_tokenizer,
    save_dir,
    task,
    outputs,
    headers=None,
    save_pdb=True,
    continue_write=False,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}...")
    if headers is None:
        headers = [f"sample_{i}" for i in range(len(outputs["output_tokens"]))]

    if task in ["sequence_generation"]:
        aatype_tokens = outputs["output_tokens"]
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=aatype_fasta_path,
            output_results=aatype_strings,
            headers=headers,
            continue_write=continue_write,
        )

    elif task in [
        "backbone_generation",
        "co_generation",
        "folding",
        "inverse_folding",
    ]:
        output_tokens = outputs["output_tokens"]
        struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
        struct_token_fasta_path = os.path.join(save_dir, "struct_token.fasta")
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        struct_tokens_strings = list(
            map(
                lambda s: ",".join(s.split()),
                tokenizer.batch_decode(
                    struct_tokens, skip_special_tokens=True
                ),
            )
        )
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=struct_token_fasta_path,
            output_results=struct_tokens_strings,
            headers=headers,
            continue_write=continue_write,
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
            for idx, (header, aatype_str, struct_tokens_str) in enumerate(
                zip(headers, aatype_strings, struct_tokens_strings)
            ):
                (
                    aatype_tensor,
                    struct_tokens_tensor,
                ) = struct_tokenizer.string_to_tensor(
                    aatype_str, struct_tokens_str
                )
                if "final_struct_feature" in outputs:
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens=outputs["final_struct_feature"][idx][
                            None
                        ],
                        res_mask=outputs["res_mask"][idx][None],
                    )
                else:
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens_tensor
                    )

                decoder_out["aatype"] = aatype_tensor
                decoder_out["header"] = [header]

                struct_tokenizer.output_to_pdb(
                    decoder_out, output_dir=pdb_save_dir
                )
    else:
        raise NotImplementedError

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm_150m"
    )
    parser.add_argument("--num_seqs", type=int, default=40)
    parser.add_argument("--seq_lens", nargs="*", type=int)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="annealing@2.2:0.1"
    )
    parser.add_argument(
        "--unmasking_strategy", type=str, default="stochastic1.0"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_pdb", type=bool, default=True)
    parser.add_argument("--bit_model", action="store_true")

    # generation options
    ## task option
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "backbone_generation",
            "sequence_generation",
            "co_generation",
            "folding",
            "inverse_folding",
        ],
        default="co_generation",
    )
    ## conditional testset
    parser.add_argument("--input_fasta_path", type=str, default="")

    args = parser.parse_args()

    if args.task in [
        "backbone_generation",
        "sequence_generation",
        "co_generation",
    ]:
        unconditional_generate(args)
    elif args.task in ["folding", "inverse_folding"]:
        conditional_generate_from_fasta(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
