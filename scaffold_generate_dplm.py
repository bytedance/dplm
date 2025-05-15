# The get_motif function of this code is highly motivated by EvoDiff:
# https://github.com/microsoft/evodiff

import argparse
import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch

from byprot import utils
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.utils.scaffold_utils import *


def generate(args, saveto):
    model = DiffusionProteinLanguageModel.from_pretrained(args.model_name)
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda(); 
    device = next(model.parameters()).device

    # Generate
    for ori_pdb, pdb in motif_name_mapping.items():
        max_iter = args.max_iter
        batch, start_idxs_list, end_idxs_list, scaffold_lengths_list = get_initial_dplm(args, tokenizer, pdb, ori_pdb, device)
        partial_mask = (batch['input_ids'].ne(tokenizer.mask_token_id) & \
                    batch['input_ids'].ne(tokenizer.pad_token_id)).type_as(batch['input_mask'])

        if args.structure_enc:
            pdb_path = os.path.join('data-bin/scaffolding-pdbs/' + pdb + '_motif.pdb')
            from byprot.datamodules.dataset.data_utils import Alphabet
            alphabet = Alphabet()
            batch_struct, _ = prepare_data(
                pdb_path, alphabet, alphabet.featurize, 
                num_seqs=args.num_seqs, device=device
            )
            
            motif_coords = batch_struct['coords']
            extend_size = list(motif_coords.size())
            extend_size[1] = partial_mask.size(1)
            extend_coords = torch.zeros(extend_size).fill_(float('nan')).type_as(motif_coords)
            coords_extent_mask = partial_mask.unsqueeze(-1).unsqueeze(-1)
            extend_coords.masked_scatter_(coords_extent_mask, motif_coords)
            batch_struct['coords'] = extend_coords
            batch_struct['motif_mask'] = partial_mask
            
            batch.update(batch_struct)
        
        with torch.cuda.amp.autocast():
            outputs = model.generate(input_tokens=batch, temperature=args.temperature,
                                            max_iter=max_iter,
                                            sampling_strategy=args.sampling_strategy,
                                            partial_masks=partial_mask)
        output_tokens = outputs[0]
        
        print('final:')
        output_results = [''.join(seq.split(' ')) for seq in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)]
        pprint(output_results)
        
        # save output
        scaffold_fasta_path = os.path.join(saveto, 'scaffold_fasta')
        os.makedirs(scaffold_fasta_path, exist_ok=True)
        saveto_name = os.path.join(scaffold_fasta_path, f'{ori_pdb}.fasta')
        fp_save = open(saveto_name, 'w')
        for idx, seq in enumerate( 
            output_results
        ):
            fp_save.write(f">SEQUENCE_{idx}_PDB_{ori_pdb}\n")
            fp_save.write(f"{seq}\n")
        fp_save.close()
        
        scaffold_info_path = os.path.join(saveto, 'scaffold_info')
        os.makedirs(scaffold_info_path, exist_ok=True)
        strings = output_results
        save_df = pd.DataFrame(list(zip(strings, start_idxs_list, end_idxs_list, scaffold_lengths_list)), columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
        save_df.to_csv(os.path.join(scaffold_info_path, f'{ori_pdb}.csv'), index=True)

    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='airkingbd/dplm_150m')
    parser.add_argument('--num_seqs', type=int, default=40)
    parser.add_argument('--saveto', type=str, default='gen.fasta')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=500)
    
    parser.add_argument('--start-idxs', type=int, action='append',
                        help="If using cond-task=scaffold, provide start and end indexes for motif being scaffolded\
                             If defining multiple motifs, supply the start and end -idx motif as a new argument\
                              ex: --start-idx 3 --end-idx 10 --start-idx 20 --end-idx 25\
                              indexes are inclusive of both start and end values.\
                              WARNING: PDBs are OFTEN indexed at a number that is not 0. If your PDB file begins at 4\
                              and the motif you want to query is residues 5 to 10, as defined by the PDB, your inputs to\
                              this code should be --start-idx 1 and --end-idx 6")
    parser.add_argument('--end-idxs', type=int, action='append')
    parser.add_argument('--scaffold-min', type=int, default=50,
                        help="Min scaffold len ")
    parser.add_argument('--scaffold-max', type=int, default=100,
                        help="Max scaffold len, will randomly choose a value between min/max")
    parser.add_argument('--structure-enc', type=bool, default=False)
    
    args = parser.parse_args()
    pprint(args)

    generate(args, args.saveto)
    
    

if __name__ == '__main__':
    main()
