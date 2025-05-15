"""
This script is highly inspired by MultiFlow.
"""

import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from glob import glob

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils
import tree
from biotite.sequence.io import fasta

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from byprot.datamodules.pdb_dataset import protein as protein_utils
from byprot.datamodules.pdb_dataset import utils as du
from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset, collate_fn
from byprot.utils import load_from_experiment, recursive_to, seed_everything
from byprot.utils.protein import folding_model
from byprot.utils.protein import utils as eu
from byprot.models.utils import get_struct_tokenizer
from byprot.utils.protein.residue_constants import restypes, restypes_with_x

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


def load_from_pdb(pdb_path, process_chain=PdbDataset.process_chain):
    raw_chain_feats, metadata = du.process_pdb_file(pdb_path)
    chain_feats = process_chain(raw_chain_feats)
    chain_feats["pdb_name"] = metadata["pdb_name"]
    return chain_feats


def load_pdb_by_name(pdb_name, metadata_df):
    row = metadata_df[metadata_df.pdb_name == pdb_name].iloc[0]
    try:
        raw_chain_feats = du.read_pkl(row.processed_path)
        chain_feats = PdbDataset.process_chain(raw_chain_feats)
    except:
        chain_feats = load_from_pdb(row.pdb_path)
    return chain_feats


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        if cfg.inference.task.startswith("unconditional"):
            ckpt_path = cfg.inference.input_fasta_dir
        elif cfg.inference.task == "forward_folding":
            ckpt_path = cfg.inference.input_fasta_dir
        elif cfg.inference.task == "inverse_folding":
            ckpt_path = cfg.inference.input_fasta_dir
        elif cfg.inference.task == "reconstruction" or cfg.inference.task == "reconstruction_continuous":
            ckpt_path = cfg.inference.input_pdb_folder
        else:
            raise ValueError(f"Unknown task {cfg.inference.task}")

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        self._inference_dir = None
        local_rank = os.environ.get("LOCAL_RANK", 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self.__inference_dir = inference_dir
            # self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, "config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f"Saving inference config to {config_path}")

        self._device_id = 0
        self._device = None

        self._folding_model = None
        self._folding_cfg = self._infer_cfg.folding

        self._struct_tokenizer = None

        self.aatype_pred_num_tokens = 21  # cfg.model.aatype_pred_num_tokens
        self.aatype_corrupt = False

        self.metadata = self.load_metadata(self._infer_cfg.metadata)

    def load_metadata(self, cfg):
        if os.path.exists(cfg.csv_path):
            df = pd.read_csv(cfg.csv_path)
            for column in ["processed_path", "raw_path", "pdb_path"]:
                if column in df:
                    df[column] = df[column].map(lambda x: os.path.join(cfg.data_dir, x))
            return df
        else:
            print(f"Metadata file not found in the {cfg.csv_path}")
            return None

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

    @property
    def folding_model(self):
        if self._folding_model is None:
            self._folding_model = folding_model.FoldingModel(
                self._folding_cfg, device_id=self.device_id
            )
        return self._folding_model

    @property
    def struct_tokenizer(self):
        if self._struct_tokenizer is None:
            print(
                f"Loading struct_tokenizer..."
            )
            self._struct_tokenizer = get_struct_tokenizer(self._infer_cfg.struct_tokenizer.exp_path).to(
                self.device
            )

        return self._struct_tokenizer

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self.__inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self.__inference_dir
            self._inference_dir = inference_dir
        return self._inference_dir

    def setup_inference_dir(self, ckpt_path):
        self._ckpt_name = "/".join(ckpt_path.replace(".ckpt", "").split("/")[-3:])
        output_dir = os.path.join(
            ckpt_path,
            self._infer_cfg.task,
            self._infer_cfg.inference_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving results to {output_dir}")
        return output_dir

    def run_detokenize_from_fasta(self, fasta_path):
        # read fasta file into sequences
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        all_header_seqs = [
            # (f"struct_seq_{i:03d}", struct_seq.lower().replace("x", "#"))
            (f"{header}", struct_seq.lower().replace("x", "#"))
            for i, (header, struct_seq) in enumerate(fasta_seqs.items())
        ]

        def _featurize(struct_seq):
            feats = {}
            feats["structok"] = torch.LongTensor(
                np.array(self.struct_tokenizer.struct_seq_to_ids(struct_seq))
            )
            feats["res_mask"] = torch.ones_like(feats["structok"], dtype=torch.float)
            return feats

        all_data = []
        for header, struct_seq in all_header_seqs:
            feats = _featurize(struct_seq)
            if self._infer_cfg.task == "reconstruction":
                pdb_name = header
            elif self._infer_cfg.task != "unconditional":
                header = pdb_name = header[
                    header.find("PDB_name_") + len("PDB_name_") : header.find("_L=")
                ]
                feats["aatype"] = torch.ones_like(feats["structok"])

            feats["header"] = f"{header}"
            all_data.append(feats)
        log.info(f"Loaded {len(all_data)} struct_seq from {fasta_path}")

        fasta_name = os.path.basename(fasta_path).replace(".fasta", "")
        output_dir = os.path.join(self.inference_dir, fasta_name, "struct_pred")
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Predicting strctures from {fasta_path}")

        cutoff = 450
        dataloader = torch.utils.data.DataLoader(
            all_data,
            batch_size=1,  # if not self._infer_cfg.get("is_trajectory") else len(all_data),
            shuffle=False,
            drop_last=False,
            collate_fn=self.struct_tokenizer.init_data,
        )
        pbar = tqdm(dataloader, desc="struct tokenizer")
        for batch in pbar:
            pbar.set_description(f"struct tokenizer ({batch['header']})")
            self._run_struct_tokenizer(batch, output_dir)

        if self._infer_cfg.get("is_trajectory"):
            self.write_trajectory(output_dir)
        return output_dir

    def get_pdb_from_struct_fasta(self, struct_fasta_path):
        if "codesign" in self._infer_cfg.task:
            self.aatype_corrupt = True
        directory_path = os.path.dirname(struct_fasta_path)
        origin_dir = os.path.join(directory_path, "pdb")
        directory_basename = os.path.basename(directory_path)
        output_dir = os.path.join(self.inference_dir, directory_basename)
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"ln -s {origin_dir} {output_dir}/pdb")
        return f"{output_dir}/pdb"

    def write_trajectory(self, pdb_folder):
        natsort = lambda s: [
            int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)
        ]
        pdb_paths = sorted(list(glob(os.path.join(pdb_folder, "*.pdb"))), key=natsort)

        saveto = os.path.join(pdb_folder, "_traj.pdb")
        f = open(saveto, "w")
        for pdb_path in pdb_paths:
            pdb_string = open(pdb_path, "r").read()
            os.remove(pdb_path)

            pdb_path = os.path.basename(pdb_path)
            t = int(pdb_path[pdb_path.index("t_") + 2 : pdb_path.index(".pdb")])

            prot = protein_utils.from_pdb_string(pdb_string)
            pdb_prot = protein_utils.to_pdb(prot, model=t + 1, add_end=False)
            f.write(pdb_prot)
        f.close()

    @torch.no_grad()
    def _run_struct_tokenizer(self, batch, output_dir, is_trajectory=False):
        batch = recursive_to(batch, device=self.device)
        decoder_out = self.struct_tokenizer.detokenize(batch['structok'], batch['res_mask'])

        save_with_aatype = self.aatype_corrupt or self._infer_cfg.task == "forward_folding"
        if save_with_aatype:
            decoder_out["aatype"] = batch["aatype"]
        decoder_out["header"] = batch["header"]

        self.struct_tokenizer.output_to_pdb(
            decoder_out, output_dir, is_trajectory=is_trajectory
        )
        log.info(f"Saved predicted structures to {output_dir}")

    @torch.no_grad()
    def run_tokenize(self, pdb_folder, output_dir):

        all_data = []

        for pdb_path in glob(os.path.join(pdb_folder, "*.pdb")):
            log.info(f"Processing {pdb_path}")

            # predicted structures
            feats = load_from_pdb(pdb_path, process_chain=self.struct_tokenizer.process_chain)
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
        for batch in pbar:
            pdb_name = batch["pdb_name"][0]
            pbar.set_description(f"Tokenize: {pdb_name} (L={batch['seq_length'][0]})")
            batch = recursive_to(batch, device=self.device)

            struct_ids = self.struct_tokenizer.tokenize(batch['all_atom_positions'], batch['res_mask'], batch['seq_length'])
            struct_seq = self.struct_tokenizer.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
            all_header_struct_seq.append((pdb_name, struct_seq))

            aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
            all_header_aa_seq.append((pdb_name, aa_seq))

        output_struct_fasta_path = os.path.join(output_dir, "struct_seq.fasta")
        fasta.FastaFile.write_iter(output_struct_fasta_path, all_header_struct_seq)

        output_aa_fasta_path = os.path.join(output_dir, "aa_seq.fasta")
        fasta.FastaFile.write_iter(output_aa_fasta_path, all_header_aa_seq)

        return output_struct_fasta_path, all_data
    
    def evaluate_reconstruction(self, pdb_folder, inplace_save=False):
        # 1. run tokenization
        pred_fasta_path, all_feats_gt = self.run_tokenize(pdb_folder, self.inference_dir)

        # 2. run detokenization
        pred_pdb_folder = self.run_detokenize_from_fasta(pred_fasta_path)

        # 3. run evaluation
        all_data = []
        # read predicted structure pdbs
        for feats_gt in all_feats_gt:
            # load gt structure
            pdb_name = feats_gt["pdb_name"]
            pdb_path = feats_gt["pdb_path"]
            log.info(f"Processing gt & pred of {pdb_path}")

            # load pred structure
            pred_pdb_path = os.path.join(pred_pdb_folder, pdb_name + ".pdb")
            feats = load_from_pdb(pred_pdb_path, process_chain=self.struct_tokenizer.process_chain)
            feats["pdb_path"] = pdb_path
            feats["header"] = pdb_name

            feats["all_atom_positions_gt"] = feats_gt["all_atom_positions"]
            feats["all_atom_mask_gt"] = feats_gt["all_atom_mask"]
            feats["aatype_gt"] = feats_gt["aatype"]

            # feats["all_atom_positions"][~feats["all_atom_mask_gt"].bool()] = 0.
            if 60 <= len(feats["aatype"]) <= 320:
                all_data.append(feats)

        log.info(f"Processed {len(all_data)} samples")

        dataloader = torch.utils.data.DataLoader(
            all_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        eval_dir = os.path.join(self.inference_dir, "eval")
        pbar = tqdm(dataloader)
        for batch in pbar:
            pbar.set_description(
                f"Eval Reconstruction: {batch['pdb_name'][0]} (L={batch['seq_length'][0]})"
            )
            self.run_evaluation(batch, eval_dir)

        return eval_dir

    def evaluate_unconditional(self, pdb_folder, inplace_save=False):
        if inplace_save:
            eval_dir = os.path.join(os.path.dirname(pdb_folder), "eval")
            # self._inference_dir = os.path.join(self.inference_dir, "eval")
        all_data = []
        for pdb_path in glob(os.path.join(pdb_folder, "*.pdb")):
            log.info(f"Processing {pdb_path}")
            feats = load_from_pdb(pdb_path)
            feats["pdb_path"] = pdb_path
            all_data.append(feats)
        log.info(f"Processed {len(all_data)} samples")

        dataloader = torch.utils.data.DataLoader(
            all_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        pbar = tqdm(dataloader)
        for batch in pbar:
            pbar.set_description(f"{batch['pdb_name'][0]} (L={batch['seq_length'][0]})")
            self.run_evaluation(batch, eval_dir)

        return eval_dir

    def evaluate_forward_folding(self, pdb_folder, inplace_save=False):
        if inplace_save:
            eval_dir = os.path.join(os.path.dirname(pdb_folder), "eval")
        all_data = []

        # read predicted structure pdbs
        for pdb_path in glob(os.path.join(pdb_folder, "*.pdb")):
            log.info(f"Processing {pdb_path}")

            # predicted structures
            feats = load_from_pdb(pdb_path)
            feats["pdb_path"] = pdb_path
            feats["header"] = feats["pdb_name"]

            feats_gt = load_pdb_by_name(feats["pdb_name"], self.metadata)
            feats["all_atom_positions_gt"] = feats_gt["all_atom_positions"]
            feats["all_atom_mask_gt"] = feats_gt["all_atom_mask"]
            feats["aatype_gt"] = feats_gt["aatype"]

            # feats["all_atom_positions"][~feats["all_atom_mask_gt"].bool()] = 0.
            if 60 <= len(feats["aatype"]) <= 512:
                all_data.append(feats)

        log.info(f"Processed {len(all_data)} samples")

        dataloader = torch.utils.data.DataLoader(
            all_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        pbar = tqdm(dataloader)
        for batch in pbar:
            pbar.set_description(f"{batch['pdb_name'][0]} (L={batch['seq_length'][0]})")
            self.run_evaluation(batch, eval_dir)

        return eval_dir

    def evaluate_inverse_folding(self, fasta_path, inplace_save=False):
        if inplace_save:
            eval_dir = fasta_path.replace(".fasta", "/eval")
        all_data = []

        fasta_seqs = fasta.FastaFile.read(fasta_path)
        all_header_seqs = [
            # (f"struct_seq_{i:03d}", struct_seq.lower().replace("x", "#"))
            (f"{header}", aa_seq.replace("#", "X"))
            for i, (header, aa_seq) in enumerate(fasta_seqs.items())
        ]

        output_dir = fasta_path.replace(".fasta", "/seq_pred")
        os.makedirs(output_dir, exist_ok=True)

        # read predicted structure pdbs
        for header, aa_seq in all_header_seqs:
            # pdb_name = pdb_name.replace("PDB_name_", "")[:4]
            # pdb_name = header[header.find("PDB_name_") + len("PDB_name_") : header.find("_L=")]
            pdb_name = header

            feats = load_pdb_by_name(pdb_name, self.metadata)
            feats["pdb_name"] = pdb_name
            feats["header"] = pdb_name
            feats["all_atom_positions_gt"] = deepcopy(feats["all_atom_positions"])
            # feats["all_atom_mask_gt"] = deepcopy(feats["all_atom_mask"])
            feats["aatype_gt"] = deepcopy(feats["aatype"])
            # predicted amino acid sequence
            feats["aatype"] = torch.LongTensor(np.array(du.seq_to_aatype(aa_seq)))

            saveto = os.path.join(output_dir, f"{pdb_name}.pdb")
            log.info(f"Saving {pdb_name} to {saveto}")
            eu.write_prot_to_pdb(
                prot_pos=feats["all_atom_positions_gt"].cpu().detach().numpy(),
                file_path=saveto,
                aatype=feats["aatype"].cpu().detach().numpy(),
            )

            all_data.append(feats)
            # batch_true.append(true_feats)

        log.info(f"Processed {len(all_data)} samples")

        dataloader = torch.utils.data.DataLoader(
            all_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        pbar = tqdm(dataloader)
        for batch in pbar:
            # if batch['pdb_name'][0] != '7fh0_B':
            #     continue
            pbar.set_description(f"{batch['pdb_name'][0]} (L={batch['seq_length'][0]})")
            self.run_evaluation(batch, eval_dir)
        
        return eval_dir

    def run_evaluation(self, batch, eval_dir):
        if "sample_id" in batch:
            sample_ids = batch["sample_id"].squeeze().tolist()
        else:
            sample_ids = list(range(batch["aatype"].shape[0]))
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        sample_lengths = batch["seq_length"].reshape(-1).tolist()
        num_batch = len(sample_ids)

        if self._infer_cfg.task.startswith("unconditional"):
            true_bb_pos = None
            sample_dirs = [
                os.path.join(
                    eval_dir,
                    f"length_{sample_length:03d}",
                    f"{os.path.basename(sample_path)}",
                )
                for sample_id, sample_length, sample_path in zip(
                    sample_ids, sample_lengths, batch["pdb_path"]
                )
            ]
            diffuse_mask = true_aatype = None
            sample_length = max(sample_lengths)

        elif self._infer_cfg.task.startswith("reconstruction"):
            sample_length = batch["res_mask"].shape[1]
            sample_dirs = [
                os.path.join(
                    # self.inference_dir,
                    eval_dir,
                    f"length_{sample_length}",
                    batch["pdb_name"][0],
                )
            ]

            true_bb_pos = batch["all_atom_positions_gt"]
            assert true_bb_pos.shape == (1, sample_length, 37, 3)

            for i, sample_dir in enumerate(sample_dirs):
                os.makedirs(sample_dir, exist_ok=True)
                # save the ground truth as a pdb
                eu.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + "_gt.pdb"),
                    aatype=batch["aatype_gt"][i].cpu().detach().numpy(),
                    no_indexing=True,
                    omit_missing_residue=False,
                )
                eu.write_prot_to_pdb(
                    prot_pos=batch["all_atom_positions"][i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + ".pdb"),
                    aatype=batch["aatype_gt"][i].cpu().detach().numpy(),
                    no_indexing=True,
                    omit_missing_residue=False,
                    # atom37_mask=batch["all_atom_mask_gt"][i].cpu().detach().numpy(),
                )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            assert true_bb_pos.shape == (sample_length * 3, 3)
            diffuse_mask = true_aatype = None

        elif self._infer_cfg.task == "forward_folding":
            sample_length = batch["res_mask"].shape[1]
            sample_dirs = [
                os.path.join(
                    # self.inference_dir,
                    eval_dir,
                    f"length_{sample_length}",
                    batch["pdb_name"][0],
                )
            ]

            true_bb_pos = batch["all_atom_positions_gt"]
            assert true_bb_pos.shape == (1, sample_length, 37, 3)

            for i, sample_dir in enumerate(sample_dirs):
                os.makedirs(sample_dir, exist_ok=True)
                # true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
                # save the ground truth as a pdb
                eu.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + "_gt.pdb"),
                    aatype=batch["aatype_gt"][i].cpu().detach().numpy(),
                    no_indexing=True,
                    omit_missing_residue=False,
                )
                eu.write_prot_to_pdb(
                    prot_pos=batch["all_atom_positions"][i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + ".pdb"),
                    aatype=batch["aatype_gt"][i].cpu().detach().numpy(),
                    no_indexing=True,
                    omit_missing_residue=False,
                    # atom37_mask=batch["all_atom_mask_gt"][i].cpu().detach().numpy(),
                )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            assert true_bb_pos.shape == (sample_length * 3, 3)
            diffuse_mask = true_aatype = None

        elif self._infer_cfg.task == "inverse_folding":
            # sample_length = batch['trans_1'].shape[1]
            sample_length = batch["res_mask"].shape[1]
            sample_dirs = [
                os.path.join(
                    # self.inference_dir,
                    eval_dir,
                    f"length_{sample_length}",
                    batch["pdb_name"][0],
                )
            ]

            true_bb_pos = batch["all_atom_positions_gt"]
            assert true_bb_pos.shape == (1, sample_length, 37, 3)

            for i, sample_dir in enumerate(sample_dirs):
                os.makedirs(sample_dir, exist_ok=True)
                # save the ground truth as a pdb
                eu.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + "_gt.pdb"),
                    aatype=batch["aatype_gt"][i].cpu().detach().numpy(),
                )
                # save predicted sequence with gt backbone as a pdb
                eu.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], batch["pdb_name"][i] + ".pdb"),
                    aatype=batch["aatype"][i].cpu().detach().numpy(),
                )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            assert true_bb_pos.shape == (sample_length * 3, 3)
            true_aatype = batch["aatype_gt"]
            diffuse_mask = true_bb_pos = None
            self.aatype_corrupt = True
        else:
            raise ValueError(f"Unknown task {self._infer_cfg.task}")

        # Skip runs if already exist
        top_sample_csv_paths = [
            os.path.join(sample_dir, "top_sample.csv") for sample_dir in sample_dirs
        ]
        if all(
            [
                os.path.exists(top_sample_csv_path)
                for top_sample_csv_path in top_sample_csv_paths
            ]
        ):
            log.info(f"Skipping instance {sample_ids} length {sample_length}")
            return

        prot_traj = [(batch["all_atom_positions"], batch["aatype"])]
        model_traj = deepcopy(prot_traj)

        diffuse_mask = (
            diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        )

        # backbone trajectories
        atom37_traj = [x[0] for x in prot_traj]
        atom37_model_traj = [x[0] for x in model_traj]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        # aa trajectories
        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        for i, sample_id in tqdm(
            zip(range(num_batch), sample_ids), total=num_batch, desc=f"{sample_length}"
        ):
            sample_dir = sample_dirs[i]
            top_sample_df = self.compute_sample_metrics(
                batch,
                model_trajs[i],
                bb_trajs[i],
                aa_trajs[i],
                clean_aa_trajs[i],
                true_bb_pos,
                true_aatype,
                diffuse_mask,
                sample_id,
                sample_lengths[i],
                sample_dir,
                aatypes_corrupt=self.aatype_corrupt,
                also_fold_pmpnn_seq=self._infer_cfg.also_fold_pmpnn_seq,
                write_sample_trajectories=self._infer_cfg.write_sample_trajectories,
            )
            top_sample_csv_path = os.path.join(sample_dir, "top_sample.csv")
            top_sample_df.to_csv(top_sample_csv_path)

    def run_pmpnn(
        self,
        write_dir,
        pdb_input_path,
    ):
        self.folding_model.run_pmpnn(
            write_dir,
            pdb_input_path,
        )
        mpnn_fasta_path = os.path.join(
            write_dir, "seqs", os.path.basename(pdb_input_path).replace(".pdb", ".fa")
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        all_header_seqs = [
            (f"pmpnn_seq_{i}", seq) for i, (_, seq) in enumerate(fasta_seqs.items()) if i > 0
        ]
        modified_fasta_path = mpnn_fasta_path.replace(".fa", "_modified.fasta")
        fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
        return modified_fasta_path

    def compute_sample_metrics(
        self,
        batch,
        model_traj,
        bb_traj,
        aa_traj,
        clean_aa_traj,
        true_bb_pos,
        true_aa,
        diffuse_mask,
        sample_id,
        sample_length,
        sample_dir,
        aatypes_corrupt,
        also_fold_pmpnn_seq,
        write_sample_trajectories,
    ):

        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        clean_traj_length = model_traj.shape[0]
        assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
        assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)
        assert aa_traj.shape == (noisy_traj_length, sample_length)
        assert clean_aa_traj.shape == (clean_traj_length, sample_length)

        os.makedirs(sample_dir, exist_ok=True)

        traj_paths = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(model_traj, axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj,
            clean_aa_traj=clean_aa_traj,
            write_trajectories=write_sample_trajectories,
            omit_missing_residue=False,
        )

        pdb_path = traj_paths["sample_path"]

        # Run PMPNN to get sequences
        sc_output_dir = os.path.join(sample_dir, "self_consistency")
        os.makedirs(sc_output_dir, exist_ok=True)
        pmpnn_pdb_path = os.path.join(sc_output_dir, os.path.basename(pdb_path))
        shutil.copy(pdb_path, pmpnn_pdb_path)
        assert (diffuse_mask == 1.0).all()
        if not self._infer_cfg.no_self_consistency:
            pmpnn_fasta_path = self.run_pmpnn(
                sc_output_dir,
                pmpnn_pdb_path,
            )
        else:
            pmpnn_fasta_path = None

        os.makedirs(os.path.join(sc_output_dir, "codesign_seqs"), exist_ok=True)
        codesign_fasta = fasta.FastaFile()
        codesign_fasta["codesign_seq_1"] = "".join([restypes[x] for x in aa_traj[-1]])
        codesign_fasta_path = os.path.join(sc_output_dir, "codesign_seqs", "codesign.fa")
        codesign_fasta.write(codesign_fasta_path)

        folded_dir = os.path.join(sc_output_dir, "folded")
        if os.path.exists(folded_dir):
            shutil.rmtree(folded_dir)
        os.makedirs(folded_dir, exist_ok=False)
        if aatypes_corrupt:
            # codesign metrics
            folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            mpnn_results = eu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)

            if also_fold_pmpnn_seq:
                pmpnn_folded_output = self.folding_model.fold_fasta(
                    pmpnn_fasta_path, folded_dir
                )
                pmpnn_results = eu.process_folded_outputs(
                    pdb_path, pmpnn_folded_output, true_bb_pos
                )
                pmpnn_results.to_csv(os.path.join(sample_dir, "pmpnn_results.csv"))

        else:
            # non-codesign metrics (unconditional, inverse folding)
            if pmpnn_fasta_path is not None:
                folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
            else:
                folded_output = None  # do not perform self-consistency evaluation
            mpnn_results = eu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)

        # mpnn_results = eu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)

        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)

            true_aa_fasta = fasta.FastaFile()
            true_aa_fasta["seq_1"] = "".join([restypes_with_x[i] for i in true_aa[0]])
            true_aa_fasta.write(os.path.join(sample_dir, "true_aa.fa"))

            sample_aa_fasta = fasta.FastaFile()
            sample_aa_fasta["seq_1"] = "".join([restypes_with_x[i] for i in aa_traj[-1]])
            sample_aa_fasta.write(os.path.join(sample_dir, "sample_aa.fa"))

            seq_recovery = (
                (torch.from_numpy(aa_traj[-1]).to(true_aa[0].device) == true_aa[0])
                .float()
                .mean()
            )
            mpnn_results["inv_fold_seq_recovery"] = seq_recovery.item()

            # get seq recovery for PMPNN as well
            if also_fold_pmpnn_seq:
                pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
                pmpnn_fasta_str = pmpnn_fasta["pmpnn_seq_1"]
                pmpnn_fasta_idx = torch.tensor(
                    [restypes_with_x.index(x) for x in pmpnn_fasta_str]
                ).to(true_aa[0].device)
                pmpnn_seq_recovery = (pmpnn_fasta_idx == true_aa[0]).float().mean()
                pmpnn_results["pmpnn_seq_recovery"] = pmpnn_seq_recovery.item()
                pmpnn_results.to_csv(os.path.join(sample_dir, "pmpnn_results.csv"))
                mpnn_results["pmpnn_seq_recovery"] = pmpnn_seq_recovery.item()
                mpnn_results["pmpnn_bb_rmsd"] = pmpnn_results["bb_rmsd"]
            else:
                mpnn_results["pmpnn_seq_recovery"] = 0.0
                mpnn_results["pmpnn_bb_rmsd"] = 0.0

        # Save results to CSV
        mpnn_results.to_csv(os.path.join(sample_dir, "sc_results.csv"))
        mpnn_results["length"] = sample_length
        mpnn_results["sample_id"] = sample_id
        del mpnn_results["header"]
        del mpnn_results["sequence"]

        # Select the top sample
        if self._infer_cfg.task.startswith("unconditional"):
            top_sample = mpnn_results.sort_values("bb_tmscore", ascending=False).iloc[:1]
        elif self._infer_cfg.task.startswith("reconstruction"):
            top_sample = mpnn_results.sort_values("bb_tmscore_to_gt", ascending=False).iloc[:1]
        elif self._infer_cfg.task == "forward_folding":
            top_sample = mpnn_results.sort_values("bb_tmscore_to_gt", ascending=False).iloc[:1]
        elif self._infer_cfg.task == "inverse_folding":
            top_sample = mpnn_results.sort_values("bb_rmsd", ascending=True).iloc[:1]

        # Compute secondary structure metrics
        sample_dict = top_sample.iloc[0].to_dict()
        ss_metrics = eu.calc_mdtraj_metrics(sample_dict["sample_path"])
        top_sample["helix_percent"] = ss_metrics["helix_percent"]
        top_sample["strand_percent"] = ss_metrics["strand_percent"]
        return top_sample

    def compute_unconditional_metrics(self, output_dir):
        log.info(f"Calculating metrics for {output_dir}")
        top_sample_csv = eu.get_all_top_samples(output_dir)
        # top_sample_csv["designable"] = top_sample_csv.bb_rmsd <= 2.0
        top_sample_csv["designable"] = top_sample_csv.bb_tmscore >= 0.5
        metrics_df = pd.DataFrame(
            data={
                "Total codesignable": top_sample_csv.designable.sum(),
                "Designable": top_sample_csv.designable.mean(),
                "Total samples": len(top_sample_csv),
            },
            index=[0],
        )
        designable_csv_path = os.path.join(output_dir, "designable.csv")
        metrics_df.to_csv(designable_csv_path, index=False)
        if self._infer_cfg.calculate_diversity:
            eu.calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path)
        if self.aatype_corrupt and self._infer_cfg.also_fold_pmpnn_seq:
            # co-design metrics
            eu.calculate_pmpnn_consistency(output_dir, metrics_df, designable_csv_path)
            eu.calculate_pmpnn_designability(output_dir, metrics_df, designable_csv_path)
        # elif self._infer_cfg.also_fold_pmpnn_seq:
        #     eu.calculate_pmpnn_designability(
        #         output_dir,
        #         metrics_df,
        #         designable_csv_path,
        #         # "sc_results.csv"
        #     )

    def compute_reconstruction_metrics(self, output_dir):
        log.info(f"Calculating metrics for {output_dir}")
        top_sample_csv = eu.get_all_top_samples(output_dir)
        # top_sample_csv["fold_match_seq"] = top_sample_csv.bb_rmsd_to_gt <= 2.0
        top_sample_csv["fold_match_seq"] = top_sample_csv.bb_tmscore_to_gt >= 0.8
        metrics_df = pd.DataFrame(
            data={
                "Total Match Seq": top_sample_csv.fold_match_seq.sum(),
                "Prop Match Seq": top_sample_csv.fold_match_seq.mean(),
                "Average bb_rmsd_to_gt": top_sample_csv.bb_rmsd_to_gt.mean(),
                "Average fold model bb_rmsd_to_gt": top_sample_csv.fold_model_bb_rmsd_to_gt.mean(),
                "Average bb_tmscore_to_gt": top_sample_csv.bb_tmscore_to_gt.mean(),
                "Total samples": len(top_sample_csv),
            },
            index=[0],
        )
        metrics_csv_path = os.path.join(output_dir, "reconstruction_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

    def compute_forward_folding_metrics(self, output_dir):
        log.info(f"Calculating metrics for {output_dir}")
        top_sample_csv = eu.get_all_top_samples(output_dir)
        # top_sample_csv["fold_match_seq"] = top_sample_csv.bb_rmsd_to_gt <= 2.0
        top_sample_csv["fold_match_seq"] = top_sample_csv.bb_tmscore_to_gt >= 0.8
        metrics_df = pd.DataFrame(
            data={
                "Total Match Seq": top_sample_csv.fold_match_seq.sum(),
                "Prop Match Seq": top_sample_csv.fold_match_seq.mean(),
                "Average bb_rmsd_to_gt": top_sample_csv.bb_rmsd_to_gt.mean(),
                "Average fold model bb_rmsd_to_gt": top_sample_csv.fold_model_bb_rmsd_to_gt.mean(),
                "Average bb_tmscore_to_gt": top_sample_csv.bb_tmscore_to_gt.mean(),
                "Total samples": len(top_sample_csv),
            },
            index=[0],
        )
        metrics_csv_path = os.path.join(output_dir, "forward_fold_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

    def compute_inverse_folding_metrics(self, output_dir):
        log.info(f"Calculating metrics for {output_dir}")
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv["designable"] = top_sample_csv.bb_rmsd <= 2.0
        metrics_df = pd.DataFrame(
            data={
                "Total designable": top_sample_csv.designable.sum(),
                "Designable": top_sample_csv.designable.mean(),
                "Total samples": len(top_sample_csv),
                "Average_bb_rmsd": top_sample_csv.bb_rmsd.mean(),
                "Average_seq_recovery": top_sample_csv.inv_fold_seq_recovery.mean(),
                "Average_pmpnn_bb_rmsd": top_sample_csv.pmpnn_bb_rmsd.mean(),
                "Average_pmpnn_seq_recovery": top_sample_csv.pmpnn_seq_recovery.mean(),
            },
            index=[0],
        )
        metrics_csv_path = os.path.join(output_dir, "inverse_fold_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)


config_path = "../../../../configs/experiment/structok/inference"


@hydra.main(version_base=None, config_path=config_path, config_name="inference_unconditional")
def run(cfg: DictConfig) -> None:
    os.environ["PROJECT_ROOT"] = cfg.env.PROJECT_ROOT
    # Read model checkpoint.
    log.info(f"Starting inference with {cfg.inference.num_gpus} GPUs")
    start_time = time.time()
    sampler = EvalRunner(cfg)
    processed_path = []

    def compute_metrics(inference_dir):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if cfg.inference.task.startswith("unconditional"):
            sampler.compute_unconditional_metrics(inference_dir)
        elif cfg.inference.task.startswith("reconstruction"):
            sampler.compute_reconstruction_metrics(inference_dir)
        elif cfg.inference.task == "forward_folding":
            sampler.compute_forward_folding_metrics(inference_dir)
        elif cfg.inference.task == "inverse_folding":
            sampler.compute_inverse_folding_metrics(inference_dir)
        else:
            raise ValueError(f"Unknown task {cfg.inference.task}")

    if cfg.inference.task == "reconstruction":
        eval_folder = sampler.evaluate_reconstruction(
            cfg.inference.input_pdb_folder, inplace_save=True
        )
        compute_metrics(eval_folder)
    else:
        natsort = lambda s: [
            int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)
        ]
        input_fasta_dir = cfg.inference.input_fasta_dir
        for fasta_path in sorted(
            glob(input_fasta_dir + ("/**/*.fasta" if cfg.inference.task == "unconditional" else "/*.fasta"), recursive=False), key=natsort
        ):
            if "struct_token" not in fasta_path and "aatype" not in fasta_path:
                continue
            if os.path.dirname(fasta_path) in processed_path:
                continue
            else:
                processed_path.append(os.path.dirname(fasta_path))
                
            print(fasta_path)
            if cfg.inference.task.startswith("unconditional"):
                pdb_folder = sampler.get_pdb_from_struct_fasta(fasta_path)
                if cfg.inference.compute_metrics:
                    eval_folder = sampler.evaluate_unconditional(pdb_folder, inplace_save=True)
                    compute_metrics(eval_folder)
            elif cfg.inference.task == "forward_folding":
                pdb_folder = sampler.get_pdb_from_struct_fasta(fasta_path)
                eval_folder = sampler.evaluate_forward_folding(pdb_folder, inplace_save=True)
                compute_metrics(eval_folder)
            elif cfg.inference.task == "inverse_folding":
                fasta_path = fasta_path.replace("struct_token", "aatype")
                pdb_folder = sampler.get_pdb_from_struct_fasta(fasta_path)
                eval_folder = sampler.evaluate_inverse_folding(fasta_path, inplace_save=True)
                compute_metrics(eval_folder)

    elapsed_time = time.time() - start_time
    log.info(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    run()
