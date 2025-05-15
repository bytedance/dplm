# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Xinyou Wang on May 15, 2025
#
# Original file was released under MIT, with the full license text
# available at https://github.com/jasonkyuyim/multiflow/blob/main/LICENSE
#
# This modified file is released under the same license.


import glob
import json
import logging
import os
import subprocess

import esm
import numpy as np
import pandas as pd
import torch
from biotite.sequence.io import fasta


class FoldingModel:
    def __init__(self, cfg, device_id=None):
        self._print_logger = logging.getLogger(__name__)
        self._cfg = cfg
        self._esmf = None
        self._device_id = device_id
        self._device = None

    @property
    def device_id(self):
        if self._device_id is None:
            self._device_id = torch.cuda.current_device()
        return self._device_id

    @property
    def device(self):
        if self._device is None:
            self._device = f"cuda:{self.device_id}"
        return self._device

    def fold_fasta(self, fasta_path, output_dir):
        if self._cfg.folding_model == "esmf":
            folded_output = self._esmf_model(fasta_path, output_dir)
        elif self._cfg.folding_model == "af2":
            folded_output = self._af2_model(fasta_path, output_dir)
        else:
            raise ValueError(
                f"Unknown folding model: {self._cfg.folding_model}"
            )
        return folded_output

    @torch.no_grad()
    def _esmf_model(self, fasta_path, output_dir):
        if self._esmf is None:
            self._print_logger.info(f"Loading ESMFold on device {self.device}")
            # torch.hub.set_dir(self._cfg.pt_hub_dir)
            self._esmf = esm.pretrained.esmfold_v1().eval().to(self.device)
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            "folded_path": [],
            "header": [],
            "plddt": [],
            "seq": [],
        }
        for header, string in fasta_seqs.items():
            # Run ESMFold
            # Need to convert unknown amino acids to alanine since ESMFold
            # doesn't like them and will remove them...
            string = string.replace("X", "A")
            esmf_sample_path = os.path.join(output_dir, f"folded_{header}.pdb")
            esmf_outputs = self._esmf.infer(string)
            pdb_output = self._esmf.output_to_pdb(esmf_outputs)[0]
            with open(esmf_sample_path, "w") as f:
                f.write(pdb_output)
            mean_plddt = esmf_outputs["mean_plddt"][0].item()
            folded_outputs["folded_path"].append(esmf_sample_path)
            folded_outputs["header"].append(header)
            folded_outputs["plddt"].append(mean_plddt)
            folded_outputs["seq"].append(string)
        return pd.DataFrame(folded_outputs)

    def _af2_model(self, fasta_path, output_dir):
        af2_args = [
            self._cfg.colabfold_path,
            fasta_path,
            output_dir,
            "--msa-mode",
            "single_sequence",
            "--num-models",
            "1",
            "--random-seed",
            "123",
            "--device",
            f"{self.device_id}",
            "--model-order",
            "4",
            "--num-recycle",
            "3",
            "--model-type",
            "alphafold2_ptm",
        ]
        process = subprocess.Popen(
            af2_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        _ = process.wait()
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            "folded_path": [],
            "header": [],
            "plddt": [],
        }
        all_af2_files = glob.glob(os.path.join(output_dir, "*"))
        af2_model_4_pdbs = {}
        af2_model_4_jsons = {}
        for x in all_af2_files:
            if "model_4" in x:
                seq_name = os.path.basename(x)
                if x.endswith(".json"):
                    seq_name = seq_name.split("_scores")[0]
                    af2_model_4_jsons[seq_name] = x
                if x.endswith(".pdb"):
                    seq_name = seq_name.split("_unrelaxed")[0]
                    af2_model_4_pdbs[seq_name] = x
            else:
                os.remove(x)
        for header, _ in fasta_seqs.items():
            af2_folded_path = af2_model_4_pdbs[header]
            af2_json_path = af2_model_4_jsons[header]
            with open(af2_json_path, "r") as f:
                folded_confidence = json.load(f)
            mean_plddt = np.mean(folded_confidence["plddt"])
            folded_outputs["folded_path"].append(af2_folded_path)
            folded_outputs["header"].append(header)
            folded_outputs["plddt"].append(mean_plddt)
        return pd.DataFrame(folded_outputs)

    def run_pmpnn(self, input_dir, output_path):

        os.makedirs(os.path.join(input_dir, "seqs"), exist_ok=True)
        process = subprocess.Popen(
            [
                "python",
                os.path.join(
                    self._cfg.pmpnn_path,
                    "helper_scripts/parse_multiple_chains.py",
                ),
                f"--input_path={input_dir}",
                f"--output_path={output_path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        _ = process.wait()
        # stdout_data, stderr_data = process.communicate()
        # print(stdout_data, stderr_data)

        pmpnn_args = [
            "python",
            os.path.join(self._cfg.pmpnn_path, "protein_mpnn_run.py"),
            "--out_folder",
            input_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            str(self._cfg.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            "38",
            "--batch_size",
            "1",
            "--device",
            str(self.device_id),
        ]
        process = subprocess.Popen(
            pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        _ = process.wait()

    # def run_pmpnn(self, input_dir, output_path):

    #     os.makedirs(os.path.join(input_dir, "seqs"), exist_ok=True)
    #     process = subprocess.Popen(
    #         [
    #             "python",
    #             os.path.join(self._cfg.pmpnn_path, "helper_scripts/parse_multiple_chains.py"),
    #             f"--input_path={input_dir}",
    #             f"--output_path={output_path}",
    #         ],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.STDOUT,
    #     )
    #     _ = process.wait()
    #     # stdout_data, stderr_data = process.communicate()
    #     # print(stdout_data, stderr_data)

    #     pdb_name = input_dir.split('/')[-5]
    #     df = pd.read_csv('/'.join(input_dir.split('/')[:-8]) + f"/scaffold_info/{pdb_name}.csv")
    #     index = int(input_dir.split('/')[-2].split('_')[1].split('.')[0])
    #     start_idxs = eval(df.iloc[index]['start_idxs'])
    #     end_idxs = eval(df.iloc[index]['end_idxs'])
    #     position_list = []
    #     for i, start_idx in enumerate(start_idxs):
    #         end_idx = end_idxs[i]
    #         position_list += list(np.arange(start_idx, end_idx + 1) + 1)
    #     position_list = ' '.join([str(a) for a in position_list])

    #     fixed_pos_output_path = output_path.replace("sample.pdb", "sample_fixed_pos.pdb")
    #     process = subprocess.Popen(
    #         [
    #             "python",
    #             os.path.join(self._cfg.pmpnn_path, "helper_scripts/make_fixed_positions_dict.py"),
    #             f"--input_path={output_path}",
    #             f"--output_path={fixed_pos_output_path}",
    #             f"--chain_list=A",
    #             f"--position_list={position_list}"
    #             # --chain_list "$chains_to_design" --position_list "$fixed_positions"
    #         ],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.STDOUT,
    #     )
    #     _ = process.wait()

    #     pmpnn_args = [
    #         "python",
    #         os.path.join(self._cfg.pmpnn_path, "protein_mpnn_run.py"),
    #         "--out_folder",
    #         input_dir,
    #         "--jsonl_path",
    #         output_path,
    #         "--fixed_positions_jsonl",
    #         fixed_pos_output_path,
    #         "--num_seq_per_target",
    #         str(self._cfg.seq_per_sample),
    #         "--sampling_temp",
    #         "0.1",
    #         "--seed",
    #         "38",
    #         "--batch_size",
    #         "1",
    #         "--device",
    #         str(self.device_id),
    #     ]
    #     process = subprocess.Popen(
    #         pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    #     )
    #     _ = process.wait()
