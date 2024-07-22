
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import copy
import os
from typing import Any, Callable, List, Union

import numpy as np
import torch
from byprot import utils
from byprot.modules import metrics
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from byprot.datamodules.dataset.data_utils import Alphabet
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
import random
from byprot.modules.cross_entropy import label_smoothed_nll_loss
from copy import deepcopy

# from byprot.task.fix.cmlm import CMLM

log = utils.get_logger(__name__)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('lm/cond_dplm')
class ConditionalDPLMTrainingTask(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='rdm',  # ['full_mask', 'random_mask']
            num_unroll=0,
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight='constant',
            output_encoder_logits=False,
        ),
        generator=Cfg(
            max_iter=1,
            strategy='discrete_diffusion',  # ['denoise' | 'mask_predict']
            noise='full_mask',  # ['full_mask' | 'selected mask']
            replace_visible_tokens=False,
            temperature=0,
            eval_plddt=False,
            eval_sc=False,
            sampling_strategy='argmax',
            use_draft_seq=False,
        )
    )
    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)

        self.alphabet = Alphabet(**alphabet)
        self.build_model() 
        self.build_generator()

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')
        elif self.stage == 'test':
            self.test_step_outputs = []

    def on_test_epoch_start(self) -> None:
        if self.hparams.generator.eval_sc:
            import esm
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)
    
    def on_predict_epoch_start(self) -> None:
        if self.hparams.generator.eval_sc:
            import esm
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)
            
    def load_from_ckpt(self, ckpt_path, not_load=False):
        # do not load state dict from ckpt, just use the initialized parameters.
        if not_load:
            return
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_generator(self):
        self.hparams.generator = merge_config(
            default_cfg=self._DEFAULT_CFG.generator,
            override_cfg=self.hparams.generator
        )
        # from byprot.models.lm.generator import IterativeRefinementGenerator

        # self.generator = IterativeRefinementGenerator(
        #     alphabet=self.alphabet,
        #     **self.hparams.generator
        # )
        log.info(f"Generator config: {self.hparams.generator}")

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.alphabet.padding_idx

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()
        
        self.plddt = MeanMetric()
        self.plddt_best = MaxMetric()
        
        self.acc = MeanMetric()
        self.acc_best = MaxMetric()

        self.acc_median = CatMetric()
        self.acc_median_best = MaxMetric()
        
    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, noise=None, sel_mask=None, mask_by_unk=False):
        padding_idx = self.alphabet.padding_idx
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx
        else:
            mask_idx = self.alphabet.mask_idx

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.alphabet.cls_idx)
                & target_tokens.ne(self.alphabet.eos_idx)
            )
            # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            raise NotImplementedError

        noise = noise or self.hparams.noise

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx) & coord_mask
        # target_mask = prev_token_mask & coord_mask

        return prev_tokens, prev_token_mask  # , target_mask
    
    def step(self, batch):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        model_output = self.model(batch, output_encoder_logits=self.hparams.learning.output_encoder_logits,
                                  weighting=self.hparams.learning.weight)
        diff_logits, diff_target, diff_loss_mask, diff_weights, encoder_logits = model_output

        diff_loss, logging_output = self.criterion(
            diff_logits, diff_target,#[loss_mask],
            # hack to calculate ppl over coord_mask in test as same other methods
            label_mask=diff_loss_mask,
            weights=diff_weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )
        
        # Compute encoder loss
        if encoder_logits is not None:
            encoder_loss, encoder_logging_output = self.criterion(encoder_logits, diff_target, label_mask=diff_loss_mask, weights=diff_weights) #label_mask=label_mask)
            logging_output['encoder/nll_loss'] = encoder_logging_output['nll_loss']
            logging_output['encoder/ppl'] = encoder_logging_output['ppl']
            
        loss = diff_loss + encoder_loss if encoder_logits is not None else diff_loss
        
        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output['nll_loss'] = logging_output['nll_loss'] * 0
            logging_output['fullseq_loss'] = logging_output['fullseq_loss'] * 0
            logging_output['fullseq_nll_loss'] = logging_output['fullseq_nll_loss'] * 0
            logging_output['ppl'] = logging_output['ppl'] * 0

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log train metrics
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)
        
        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(deepcopy(batch))
        
        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            self.predict_step(batch, batch_idx)

        return {"loss": loss}

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def eval_self_consistency(self, pred_ids, positions, mask=None):
        from esm.esmfold.v1.misc import output_to_pdb

        import byprot.modules.protein_metrics as pmetrics

        # positions, pred_ids = tensor_to_list(positions, pred_ids, mask)
        pred_seqs = decode(pred_ids, self.alphabet, remove_special=True)

        # run_folding:
        sc_tmscores = []
        sc_rmsds = []
        plddts = []
        with torch.no_grad():
            # with torch.autocast()
            output = self._folding_model.infer(sequences=pred_seqs)
            # pred_seqs = decode(output["aatype"], self.alphabet, remove_special=True)

            # positions = positions.to(torch.float64)
            # output["positions"] = output["positions"].clone().to(torch.float64)
            # pred_pdbs = output_to_pdb(output)

            # output["positions"] = output["positions"].clone()
            # for i in range(positions.shape[0]):
            #     pred_seq = pred_seqs[i]
            #     seqlen = len(pred_seq)
            #     output["positions"][-1, i, :seqlen, :3, :] = positions[
            #         i, 1 : seqlen + 1, :3, :
            #     ]
            # true_pdbs = output_to_pdb(output)

            # for pred_pdb_str, true_pdb_str in zip(pred_pdbs, true_pdbs):
            #     sc_tmscore = calc_tm_score(pred_pdb_str, true_pdb_str)
            #     sc_tmscores.append(sc_tmscore)

            positions = positions.cpu()
            nan_mask = positions.isnan()
            # positions[nan_mask] = 0.0

            folded_positions = output["positions"][-1].cpu()
            CA_idx = 1
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                _, sc_tmscore = pmetrics.calc_tm_score(
                    positions[i, 1 : seqlen + 1, :3, :],
                    folded_positions[i, :seqlen, :3, :],
                    pred_seq,
                    pred_seq,
                    # ~nan_mask[i, 1 : seqlen + 1, CA_idx, 0] 
                    mask[i, 1 : seqlen + 1].cpu()
                )
                sc_tmscores.append(sc_tmscore)

                # sc_rmsd = pmetrics.calc_aligned_rmsd(
                #     positions[i, 1 : seqlen + 1, 1, :],
                #     folded_positions[i, :seqlen, 1, :],
                # )
                from openfold.utils.superimposition import superimpose
                _, sc_rmsd = superimpose(
                    positions[i, 1 : seqlen + 1, CA_idx, :][None],
                    folded_positions[i, :seqlen, CA_idx, :][None],
                    # ~nan_mask[i, 1 : seqlen + 1, CA_idx, 0][None],
                    mask[i, 1 : seqlen + 1].cpu()
                )
                sc_rmsds.append(sc_rmsd[0].item())

                plddt = output["mean_plddt"][i].item()
                plddts.append(plddt)

                print(f"{i+1}/{positions.shape[0]}: {sc_tmscore}, {sc_rmsd}, {plddt}")
        return sc_tmscores, (sc_rmsds, plddts)
    
    def on_validation_epoch_end(self):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            self.on_predict_epoch_end()
            self.valid_uncon_count = 0

        super().on_validation_epoch_end()

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        output_tokens, output_scores = self.model.generate(
            batch=batch,
            max_iter=self.hparams.generator.max_iter,
            sampling_strategy=self.hparams.generator.sampling_strategy,
            temperature=self.hparams.generator.temperature,
            use_draft_seq=self.hparams.generator.use_draft_seq,
        )
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True
    ) -> Any:
        tokens = batch.pop('tokens')
        coord_mask = batch['coord_mask']
        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask,
            noise=self.hparams.generator.noise,  # NOTE: 'full_mask' by default. Set to 'selected_mask' when doing inpainting.
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        pred_tokens = self.forward(batch, return_ids=True)
        
        special_sym_mask = (
                tokens.eq(self.alphabet.padding_idx) |
                tokens.eq(self.alphabet.cls_idx) |
                tokens.eq(self.alphabet.eos_idx)
            )
        pred_tokens.masked_scatter_(special_sym_mask, tokens[special_sym_mask])
        
        if log_metrics:
            # per-sample accuracy
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=coord_mask)
            self.acc_median.update(recovery_acc_per_sample)

            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=coord_mask)
            self.acc.update(recovery_acc, weight=coord_mask.sum())

        results = {
            "pred_tokens": pred_tokens,
            "names": batch["names"],
            "native": batch["seqs"],
            "recovery": recovery_acc_per_sample,
            "sc_tmscores": np.zeros(pred_tokens.shape[0]),
            "sc_rmsds": np.zeros(pred_tokens.shape[0]),
            "plddts": np.zeros(pred_tokens.shape[0]),
        }

        if self.hparams.generator.eval_sc:
            torch.cuda.empty_cache()
            sc_tmscores, (sc_rmsds, plddts) = self.eval_self_consistency(
                pred_tokens, batch["coords"], mask=coord_mask
            )
            results["sc_tmscores"] = sc_tmscores
            results["sc_rmsds"] = sc_rmsds
            results["plddts"] = plddts
            
        if self.stage == "test":
            self.test_step_outputs.append(results)
            
        # return results

    def on_predict_epoch_end(self) -> None:
        log_key = "test" if self.stage == "test" else "val"

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        print('ACC: ', acc)
        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        self.log(
            f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True
        )
        print('ACC_MEDIAN: ', acc_median)
        if self.stage == "fit":
            self.acc_best.update(acc)
            self.log(
                f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True
            )

            self.acc_median_best.update(acc_median)
            self.log(
                f"{log_key}/acc_median_best",
                self.acc_median_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )
        else:
            if self.hparams.generator.eval_sc:
                import itertools

                def _merge_and_log(name):
                    metrics_list = list(itertools.chain(*[result[name] for result in self.test_step_outputs]))
                    self.log(
                        f"{log_key}/{name}",
                        np.mean(metrics_list),
                        on_epoch=True,
                        prog_bar=True,
                    )

                _merge_and_log("sc_tmscores")
                _merge_and_log("sc_rmsds")
                _merge_and_log("plddts")

            self.save_prediction(
                self.test_step_outputs, saveto=f"./test_tau{self.hparams.generator.temperature}.fasta"
            )
            with open('./result.txt', 'w') as f:
                f.write(f'acc: {acc}')
                f.write(f'acc_median: {acc_median}')
                # if self.hparams.generator.eval_sc:
                #     f.write(f'sc_tmscores: {sc_tmscores}')
        
    def save_prediction(self, results, saveto=None):
        save_dict = {}
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, "w")
            fp_native = open("./native.fasta", "w")

        for entry in results:
            for name, prediction, native, recovery, scTM, scRMSD, plddt in zip(
                entry["names"],
                decode(entry["pred_tokens"], self.alphabet, remove_special=True),
                entry["native"],
                entry["recovery"],
                entry["sc_tmscores"],
                entry["sc_rmsds"],
                entry["plddts"],
            ):
                save_dict[name] = {
                    "prediction": prediction,
                    "native": native,
                    "recovery": recovery,
                }
                if saveto:
                    fp.write(
                        f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f} | scRMSD={scRMSD:.2f} | plddt={plddt:.2f} \n"
                    )
                    fp.write(f"{prediction}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")

        if saveto:
            fp.close()
            fp_native.close()
        return save_dict
            
def decode(batch_ids, alphabet, remove_special=False, replace_X=True):
    ret = []
    for ids in batch_ids:
        line = ''.join([alphabet.get_tok(id) for id in ids])
        if remove_special:
            line = line.replace(alphabet.get_tok(alphabet.mask_idx), '_') \
                .replace(alphabet.get_tok(alphabet.eos_idx), '') \
                .replace(alphabet.get_tok(alphabet.cls_idx), '') \
                .replace(alphabet.get_tok(alphabet.padding_idx), '') \
                .replace(alphabet.get_tok(alphabet.unk_idx), '-')
        if replace_X:
            line = line.replace('X', 'G')
        ret.append(line)
    return ret
