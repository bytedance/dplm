import argparse
import glob
import logging
import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from byprot import utils
from byprot.models.lm.generator import IterativeRefinementGenerator
from byprot.utils import io
from byprot.utils.config import compose_config as Cfg
import tqdm

from biotite.database.rcsb import fetch
from biotite.structure import AtomArray

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple
from byprot.datamodules.dataset.uniref import get_collater

GenOut = namedtuple(
    'GenOut', 
    ['output_tokens', 'output_scores', 'attentions']
)

ss3_to_id = {
    "H": 0, # helix
    "E": 1, # beta-strand
    "T": 2, # other
    "C": 2,
    "L": 2
}

def exists(t): return t is not None

def download_pdb(pdb_id, return_path=True):
    path = f'{pdb_id}.pdb'
    if not os.path.exists(path):
        ff = fetch(pdb_id, format="pdb")
        with open(path, 'w') as f:
            f.write(ff.getvalue())
    if return_path: return path

def get_ss3_from_pdb(pdb_path):
    from Bio.PDB import PDBParser, DSSP

    # Create a parser
    p = PDBParser(QUIET=True)

    # Parse the structure from a PDB file 
    structure = p.get_structure("your_protein", pdb_path)

    # DSSP output typically needs the PDB file and the PDB id
    model = structure[0]
    dssp = DSSP(model, pdb_path, acc_array='Wilke')

    # We only need 3 states (H, E, C). With DSSP we get H, B, E, G, I, T, S by default.
    # Let's change everything to the 3 classes
    ss3 = []

    for aa in list(dssp.keys()):
        if dssp[aa][2] == 'H' or dssp[aa][2] == 'G' or dssp[aa][2] == 'I':
            ss3.append("H")
        elif dssp[aa][2] == 'E' or dssp[aa][2] == 'B':
            ss3.append("E")
        else:
            ss3.append('C')
    return ss3

def ss3_to_label_ids(ss3_list):
    return torch.tensor(
        [-1] + list(map(lambda s: ss3_to_id[s], ss3_list)) + [-1],
        dtype=torch.int64
    )

template_pdb = "/root/research/projects/ByProt/examples/protein-programming-language/2l4v.pdb"
template_pdb = "/root/research/projects/ByProt/data/pdb_samples/5cw9.pdb"
template_pdb = "/root/research/projects/ByProt/data/denovo_pdb/4DB6.pdb"
template_pdb = "/root/research/projects/ByProt/data/denovo_pdb/4PWW.pdb"

# template_pdb = download_pdb('TOP7')
guidance_ss3 = get_ss3_from_pdb(template_pdb) 
guidance_ss3_labels = ss3_to_label_ids(guidance_ss3) 

def pplm_step(
    # last_hiddens,
    # last_logits,
    lm_head,
    guidance_model,
    hidden_states,
    selection_mask=None,
    input_mask=None,
    targets=None,
    guidance_step_size=1e-4,
    guidance_num_steps=1,
    guidance_stability_coef=0.1,
    rtol=1.0
):
    kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
    old_logits = lm_head(hidden_states[..., :, :]).log_softmax(dim=-1).detach()

    delta = torch.nn.Parameter(
        torch.zeros_like(hidden_states[..., :, :])
    )
    optimizer = torch.optim.Adagrad([delta], lr=guidance_step_size)

    # targets = torch.full((hidden_states.shape[0], ), fill_value=1., device=hidden_states.device)
    losses = [1e4]
    with torch.enable_grad():
        for step in range(guidance_num_steps):
            new_hidden_states = hidden_states[..., :, :] + delta

            # all_h = torch.cat([
            #     hidden_states[..., :-1, :], last_h
            # ], dim=-2)

            guidance_loss = guidance_model(
                new_hidden_states, input_mask=input_mask, targets=targets)['loss'].sum()

            new_logits = lm_head(new_hidden_states).log_softmax(dim=-1)
            kl = kl_loss(new_logits, old_logits)

            loss = guidance_loss + guidance_stability_coef * kl
            # print(f"{step}: guidance_loss: {guidance_loss}, kl: {kl}")

            if loss - losses[-1] > rtol:
                print(f"break")
                break
            # print(kl)
            # print(guide_loss)
            # print(loss)
            # print("********")

            optimizer.zero_grad()
            loss.backward()
            # delta_grad_norm = delta.grad.norm(keepdim=True)
            # delta.grad /= delta_grad_norm.clamp_min(1e-7)
            # delta.grad *= math.sqrt(h.size(-2))
            if exists(selection_mask):
                delta.grad *= selection_mask[:, :, None].float()
            optimizer.step()
            losses.append(loss)

    # print("\n")

    hidden_states[..., :, :] += delta.data

    return hidden_states

def guidance_step(model, esm_out, selection_mask, cur_iter, max_iter):
    hidden_states = esm_out['representations'][model.esm.num_layers]
    targets = torch.full(
        (hidden_states.shape[0], ), 
        fill_value=1., 
        device=hidden_states.device
    )
    targets = guidance_ss3_labels[None, :]\
        .repeat(hidden_states.shape[0], 1)\
        .to(hidden_states.device)

    lm_head = model.esm.lm_head
    new_hidden_states = pplm_step(
        lm_head=lm_head,
        guidance_model=model.guidance_model.forward_with_hiddens,
        hidden_states=hidden_states,
        selection_mask=selection_mask,
        input_mask=None,
        targets=targets,
        guidance_step_size=5e-3,
        guidance_num_steps=20 if 0 < cur_iter <= (max_iter - 0) else 0,
        guidance_stability_coef=0,
    )
    new_logits = lm_head(new_hidden_states)
    esm_out['representations'][model.esm.num_layers] = new_hidden_states
    esm_out['logits'] = new_logits

    guidance_out = model.guidance_model.forward_with_hiddens(
        new_hidden_states, input_mask=None, targets=targets)
    loss, metric = guidance_out['loss'], guidance_out['metrics']
    pred = guidance_out['prediction']
    print(f"[{cur_iter}/{max_iter}] loss: {loss}, metric: {metric}\n\tpred: {pred},") 
    print((pred == targets).sum(1)/pred.shape[1])
    print(f"{model.alphabet.decode(new_logits.argmax(-1)[:, 1:-1], remove_special=False)}")
    return esm_out

def load_discr_model(args):
    pl_module, exp_cfg = utils.load_from_experiment(
        args.discr_model_path, ckpt='last.ckpt')
    discr_model = pl_module.model
    return discr_model.eval()

def setup_guided_model(model, guidance_model):
    model.guidance_model = guidance_model
    model.guidance_step = lambda esm_out, selection_mask, cur_iter, max_iter: guidance_step(
        model, esm_out, selection_mask, cur_iter, max_iter)
    return model 

def setup_generation(args, ckpt):
    pl.seed_everything(args.seed)

    pl_module, exp_cfg = utils.load_from_experiment(
        args.experiment_path, ckpt=ckpt)
    model = pl_module.model
    discr_model = load_discr_model(args)
    model = setup_guided_model(model, discr_model)

    alphabet = pl_module.alphabet
    model.alphabet = alphabet
    collater = get_collater(name='esm')

    generator = IterativeRefinementGenerator(
        alphabet=alphabet, 
        max_iter=args.max_iter,
        strategy=args.strategy,
        temperature=args.temperature
    )
    return model.eval(), alphabet, collater, generator


def _full_mask(target_tokens, alphabet):
    target_mask = (
        target_tokens.ne(alphabet.padding_idx)  # & mask
        & target_tokens.ne(alphabet.cls_idx)
        & target_tokens.ne(alphabet.eos_idx)
    )
    _tokens = target_tokens.masked_fill(
        target_mask, alphabet.mask_idx
    )
    _mask = _tokens.eq(alphabet.mask_idx) 
    return _tokens, _mask


def get_initial(num, length, collater, device, cond_seq=None):
    if cond_seq is None:
        init_seq = [('A' * length, )] * num
    else:
        init_seq = [(cond_seq,)]
    batch = collater(init_seq)
    batch['input_ids'], _ = _full_mask(batch['input_ids'].clone(), collater.alphabet)
    batch = utils.recursive_to(batch, device)
    # pprint(batch)
    return batch

@torch.autocast(device_type="cuda")
def generate(args, ckpt, saveto, cond_seq=None):
    print(ckpt)
    model, alphabet, collater, generator = setup_generation(args, ckpt)
    model = model.cuda(); 
    device = next(model.parameters()).device

    for seq_len in args.seq_lens:
        saveto_seq_len = saveto + "_iter_" + str(args.max_iter) + "_L_" + str(seq_len) + ".fasta"
        fp_save = open(saveto_seq_len, 'w')
        batch = get_initial(args.num_seqs, seq_len, collater, device, cond_seq=cond_seq)
        partial_mask = None
        if cond_seq is not None:
            partial_mask = batch['input_ids'].ne(model.alphabet.mask_idx).type_as(batch['input_mask'])
        outputs = generator.generate(model=model, batch=batch, print_output=True, 
                                     max_iter=args.max_iter,
                                     sampling_strategy=args.sampling_strategy,
                                     partial_masks=partial_mask)
        output_tokens = outputs[0]

        print('final:')
        pprint(alphabet.decode(output_tokens, remove_special=False))

        for idx, seq in enumerate( 
            alphabet.decode(output_tokens, remove_special=True)
        ):
            fp_save.write(f">SEQUENCE_{idx}_L={seq_len}\n")
            fp_save.write(f"{seq}\n")
        fp_save.close()
        # print(guidance_ss3)
        print(guidance_ss3_labels)

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seqs', type=int, default=20)
    parser.add_argument('--seq_lens', nargs='*', type=int)
    parser.add_argument('--experiment_path', type=str)
    parser.add_argument('--discr_model_path', type=str)
    parser.add_argument('--saveto', type=str, default='gen.fasta')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--strategy', type=str, default='rdm')
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--ckpt', type=str, default='last.ckpt')
    # you can directly specify a directory of ckpt by ${ckpt_dir}
    # and test all of ckpt in this folder. The outputs will be saved to ${saveto_dir},
    # the output filename will be 'prefix of ckpt' + '.fasta'
    parser.add_argument('--ckpt_dir', type=str, default='not specified')
    parser.add_argument('--saveto_dir', type=str, default='not specified')
    args = parser.parse_args()
    pprint(args)

    # cond_seq = 'MFQWQADCLCTGNVLQGGNLVYSAPTSAGKTMVAELLMLKRVLETKR<mask><mask><mask><mask><mask><mask>' + \
    #     'PFVSVAREKMFYLQRLFQEAGVR<mask><mask><mask><mask>' + \
    #     'MGSHSPAGGFAATD<mask><mask><mask><mask><mask>' + \
    #     'IEKGNSLLNRLMEEGKVS<mask><mask><mask><mask><mask><mask><mask>DELHMVGDPNRGYLLELFLTKIRYLS'
    cond_seq = None
    # assert not ((args.ckpt_dir is "not specified") ^ (args.saveto_dir is "not specified"))
    # if args.ckpt_dir is not "not specified":
    #     ckpt_list = os.listdir(args.ckpt_dir)
    #     for ckpt in ckpt_list:
    #         prefix = ckpt[:ckpt.find(".ckpt")]
    #         saveto = os.path.join(args.saveto_dir, prefix + "_" + args.strategy + "_iter_" + 
    #                               str(args.max_iter))
    #         generate(args, ckpt, saveto)
    # else:
    generate(args, args.ckpt, args.saveto, cond_seq=cond_seq)
    

if __name__ == '__main__':
    main()
