from glob import glob
import os
from tqdm.auto import tqdm
import multiprocessing as mp
import itertools
import numpy as np
import subprocess
import re
import pandas
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def run_tmalign(query, reference, fast=True):
    # --> one to one
    exec = "./TMalign"
    cmd = f"{exec} {query} {reference}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return np.nan

    score_lines = []
    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            score_lines.append(line)

    key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    try:
        a = results_dict["Chain_1"]
    except:
        wxy = 1
    return results_dict["Chain_1"]


def tm_one2refs(
    query,
    references,
    n_threads,
    fast = True,
    chunksize = 10,
    parallel = True
):
    args = [(query, ref, fast) for ref in references]
    if parallel:
        n_threads = min(n_threads, len(references))
        pool = mp.Pool(n_threads)
        values = list(pool.starmap(run_tmalign, args, chunksize=chunksize))
        pool.close()
        pool.join()
    else:
        values = list(itertools.starmap(run_tmalign, args))
    
    return values


def tm_set2set(
    querys,
    targets,
    save_path,
    n_threads=mp.cpu_count()
    ):
    save_dict = {
        'query': [],
        'tm_scores': [],
    }

    print("using {:d} threads", n_threads)
    cnt = 0
    for pdb in tqdm(querys[:]):
        tm_scores = tm_one2refs(
            pdb,
            targets,
            n_threads=n_threads
        )
        save_dict["query"].append(pdb)
        save_dict["tm_scores"].append(tm_scores)
        cnt += 1
    
    save_df = pandas.DataFrame.from_dict(save_dict)
    save_df.to_csv(os.path.join(save_path, "inter_tmscore.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', type=str, default=None,
                        help="the dirname of query pdb files")
    # if calculate diversity, ref_dir should be same to query_dir
    # if calculate novelty, ref_dir should be specified to the path of all pdb files
    parser.add_argument('--ref_dir', type=str, default=None,
                        help="the dirname of ref pdb files")
    parser.add_argument(
        '--cal_type',
        choices=['diversity', 'novelty'],
        required=True,
        help='Specify the type. Options are "diversity" or "novelty".'
    )
    args = parser.parse_args()
    
    query_dir = args.query_dir
    reference_dir = args.ref_dir
    
    if args.cal_type == 'diversity':
        cal_diversity(query_dir, reference_dir)
    elif args.cal_type == 'novelty':
        cal_novelty(query_dir, reference_dir)
    
  
def cal_novelty(query_dir, reference_dir):
    output_dir = os.path.join(query_dir, '../novelty')
    os.makedirs(output_dir, exist_ok=True)
    
    for dirname in os.listdir(query_dir):
        if not os.path.isdir(os.path.join(query_dir, dirname)):
            continue
        query_pdb_dir = os.path.join(query_dir, dirname)
        query_paths = [os.path.join(query_pdb_dir, pdb_name) for pdb_name in os.listdir(query_pdb_dir)]
        reference_paths = [os.path.join(reference_dir, pdb_name) for pdb_name in os.listdir(reference_dir)]
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
        tm_set2set(query_paths, reference_paths, os.path.join(output_dir, dirname)) #, mp.cpu_count())
    
def cal_diversity(query_dir, reference_dir):
    output_dir = os.path.join(query_dir, '../diversity')
    os.makedirs(output_dir, exist_ok=True)
    
    for dirname in os.listdir(query_dir):
        if not os.path.isdir(os.path.join(query_dir, dirname)):
            continue
        query_pdb_dir = os.path.join(query_dir, dirname)
        reference_pdb_dir = os.path.join(reference_dir, dirname)
        query_paths = [os.path.join(query_pdb_dir, pdb_name) for pdb_name in os.listdir(query_pdb_dir)]
        reference_paths = [os.path.join(reference_pdb_dir, pdb_name) for pdb_name in os.listdir(reference_pdb_dir)]
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
        tm_set2set(query_paths, reference_paths, os.path.join(output_dir, dirname)) #, mp.cpu_count())
    
if __name__ == "__main__":
    main()
