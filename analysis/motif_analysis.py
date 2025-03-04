import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from ast import literal_eval
import subprocess
from Bio import PDB
import numpy as np
import argparse


def analysis(args):
    start_idx_dict = {
        "1prw": [15, 51],
        "1bcf": [17, 46, 90, 122],
        "5tpn": [108],
        "3ixt": [0],
        "4jhw": [37, 144],
        "4zyp": [357],
        "5wn9": [1],
        "5ius": [34, 88],
        "5yui": [89, 114, 194],
        "6vw1": [5, 45],
        "1qjg": [13, 37, 98],
        "1ycr": [2],
        "2kl8": [0, 27],
        "7mrx": [25],
        "5trv": [45],
        "6e6r": [22],
        "6exz": [25],
    }
    end_idx_dict = {
        "1prw": [34, 70],
        "1bcf": [24, 53, 98, 129],
        "5tpn": [126],
        "3ixt": [23],
        "4jhw": [43, 159],
        "4zyp": [371],
        "5wn9": [20],
        "5ius": [53, 109],
        "5yui": [93, 116, 196],
        "6vw1": [23, 63],
        "1qjg": [13, 37, 98],
        "1ycr": [10],
        "2kl8": [6, 78],
        "7mrx": [46],
        "5trv": [69],
        "6e6r": [34],
        "6exz": [39],
    }

    def calculate_avg_plddt(pdb_file):
        # 创建PDB解析器
        parser = PDB.PDBParser(QUIET=True)

        # 解析PDB文件
        structure = parser.get_structure("protein", pdb_file)

        # 获取所有的plDDT值
        plddt_values = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        # 获取 CA 原子的 B-factor，并假设它存储了 plDDT 值
                        ca_atom = residue["CA"]
                        plddt = ca_atom.get_bfactor()
                        plddt_values.append(plddt)

        # 计算平均plDDT值
        if plddt_values:
            avg_plddt = np.mean(plddt_values)
            return avg_plddt
        else:
            raise NotImplementedError

    def calc_rmsd_tmscore(
        pdb_name,
        reference_PDB,
        scaffold_pdb_path=None,
        scaffold_info_path=None,
        ref_motif_starts=[30],
        ref_motif_ends=[44],
        output_path=None,
    ):
        "Calculate RMSD between reference structure and generated structure over the defined motif regions"

        motif_df = pd.read_csv(
            os.path.join(scaffold_info_path, f"{pdb_name}.csv"), index_col=0
        )  # , nrows=num_structures)
        results = []
        for pdb in os.listdir(
            os.path.join(scaffold_pdb_path, f"{pdb_name}")
        ):  # This needs to be in numerical order to match new_starts file
            if not pdb.endswith(".pdb"):
                continue
            ref = mda.Universe(reference_PDB)
            predict_PDB = os.path.join(
                os.path.join(scaffold_pdb_path, f"{pdb_name}"), pdb
            )
            u = mda.Universe(predict_PDB)

            ref_selection = "name CA and resnum "
            u_selection = "name CA and resnum "
            i = int(pdb.split("_")[1].split(".")[0])
            new_motif_starts = literal_eval(motif_df["start_idxs"].iloc[i])
            new_motif_ends = literal_eval(motif_df["end_idxs"].iloc[i])

            for j in range(len(ref_motif_starts)):
                ref_selection += (
                    str(ref_motif_starts[j]) + ":" + str(ref_motif_ends[j]) + " "
                )
                u_selection += (
                    str(new_motif_starts[j] + 1)
                    + ":"
                    + str(new_motif_ends[j] + 1)
                    + " "
                )
            print("U SELECTION", u_selection)
            print("SEQUENCE", i)
            print("ref", ref.select_atoms(ref_selection).resnames)
            print("gen", u.select_atoms(u_selection).resnames)
            # This asserts that the motif sequences are the same - if you get this error something about your indices are incorrect - check chain/numbering
            assert len(ref.select_atoms(ref_selection).resnames) == len(
                u.select_atoms(u_selection).resnames
            ), "Motif lengths do not match, check PDB preprocessing \
                for extra residues"

            assert (
                ref.select_atoms(ref_selection).resnames
                == u.select_atoms(u_selection).resnames
            ).all(), "Resnames for motifRMSD do not match, check indexing"
            rmsd = rms.rmsd(
                u.select_atoms(u_selection).positions,
                # coordinates to align
                ref.select_atoms(ref_selection).positions,
                # reference coordinates
                center=True,  # subtract the center of geometry
                superposition=True,
            )  # superimpose coordinates

            temp_file = open(os.path.join(output_path, "temp_tmscores.txt"), "w")

            subprocess.call(
                ["./analysis/TMscore", reference_PDB, predict_PDB, "-seq"],
                stdout=temp_file,
            )
            with open(os.path.join(output_path, "temp_tmscores.txt"), "r") as f:
                for line in f:
                    if len(line.split()) > 1 and "TM-score" == line.split()[0]:
                        tm_score = line.split()[2]
                        break

            # plddt = float(predict_PDB.split('_')[-1][:-4])
            # 计算平均plDDT值
            plddt = calculate_avg_plddt(predict_PDB)
            results.append((pdb_name, i, rmsd, plddt, tm_score))
        return results

    scaffold_dir = args.scaffold_dir
    output_dir = os.path.join(scaffold_dir, "scaffold_results")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for pdb in start_idx_dict.keys():
        print(pdb)
        ref_motif_starts = start_idx_dict[pdb]
        ref_motif_ends = end_idx_dict[pdb]
        reference_PDB = os.path.join(
            "./data-bin/scaffolding-pdbs", pdb + "_reference.pdb"
        )
        with open(reference_PDB) as f:
            line = f.readline()
            ref_basenum = int(line.split()[5])
        ref_motif_starts = [num + ref_basenum for num in ref_motif_starts]
        ref_motif_ends = [num + ref_basenum for num in ref_motif_ends]
        results += calc_rmsd_tmscore(
            pdb_name=pdb,
            reference_PDB=reference_PDB,
            scaffold_pdb_path=f"{scaffold_dir}/scaffold_fasta/esmfold_pdb",
            scaffold_info_path=f"{scaffold_dir}/scaffold_info",
            ref_motif_starts=ref_motif_starts,
            ref_motif_ends=ref_motif_ends,
            output_path=output_dir,
        )

    results = pd.DataFrame(
        results, columns=["pdb_name", "index", "rmsd", "plddt", "tmscore"]
    )
    results.to_csv(os.path.join(output_dir, "rmsd_tmscore.csv"), index=False)


def cal_success_scaffold(pdb):
    total = len(pdb)
    pdb["total"] = total
    pdb = pdb[(pdb["rmsd"] < 1.0) & (pdb["plddt"] > 70)]
    return pdb


def motif_evaluation(args):
    analysis(args)

    output_dir = os.path.join(args.scaffold_dir, "scaffold_results")
    rmsd_tmscore = pd.read_csv(os.path.join(output_dir, "rmsd_tmscore.csv"))
    success_scaffold = rmsd_tmscore.groupby("pdb_name", as_index=False).apply(
        cal_success_scaffold
    )
    success_scaffold_count = success_scaffold.groupby("pdb_name").size()
    success_scaffold_count = success_scaffold_count.reset_index(name="success_count")

    all_pdb = list(rmsd_tmscore["pdb_name"].unique())
    success_pdb = list(success_scaffold_count["pdb_name"])
    failed_pdb = list(set(all_pdb) - set(success_pdb))
    failed_scaffold_count = {
        "pdb_name": failed_pdb,
        "success_count": [0] * len(failed_pdb),
    }
    results = pd.concat(
        [success_scaffold_count, pd.DataFrame(failed_scaffold_count)]
    ).sort_values("pdb_name")
    results.to_csv(os.path.join(output_dir, "result.csv"))
    print(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scaffold_dir", type=str, default="./generation-results")

    args = parser.parse_args()

    motif_evaluation(args)


if __name__ == "__main__":
    main()
