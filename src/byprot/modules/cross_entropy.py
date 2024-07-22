
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import Tensor, nn
from torch.nn import functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, mask=None) -> Tensor:
        """
          scores: [N, ..., C], unnormalized scores
          target: [N, ...]
          mask: [N, ...], where elements with `True` are allowed and `False` are masked-out
        """
        n_tokens = target.numel()
        n_nonpad_tokens = target.ne(self.ignore_index).long().sum()

        bsz, num_classes = scores.shape[0], scores.shape[-1]

        if mask is not None:
            scores = scores[mask]  # [N * len, C]
            target = target[mask]  # [N]
        scores = scores.reshape(-1, num_classes)
        target = target.reshape(-1)

        if self.ignore_index is not None:
            sample_size = target.ne(self.ignore_index).long().sum()
        else:
            sample_size = torch.tensor(target.numel(), device=target.device)

        # smooth_loss = F.cross_entropy(
        #     scores.transpose(1, -1), target,
        #     weight=self.weight,
        #     ignore_index=self.ignore_index, reduction=self.reduction,
        #     label_smoothing=self.label_smoothing)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=True,
        )
        loss_avg = loss / sample_size
        ppl = torch.exp(nll_loss / sample_size)

        logging_output = {
            'nll_loss_sum': nll_loss.data,
            'loss_sum': loss.data,
            'ppl': ppl.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        return loss_avg, logging_output


class Coord2SeqCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, coord_mask=None, weights=None) -> Tensor:
        """
          scores: [N, L, C], unnormalized scores
          target: [N, L]
          coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        if label_mask is None:
            label_mask = coord_mask

        bsz, num_classes = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        # [N, L]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss
        # nll_loss = nll_loss[label_mask] # calculate pesudo-ppl
        ppl = torch.exp(nll_loss)

        logging_output = {
            'nll_loss': nll_loss.data,
            'ppl': ppl.data, # torch.mean(ppl).data,
            'fullseq_loss': fullseq_loss.data,
            'fullseq_nll_loss': fullseq_nll_loss.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        return loss, logging_output


class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, weights=None,
                cal_constant_loss=False,
                watch_t1_t2_loss=False,
                ) -> Tensor:
        """
          scores: [N, L, C], unnormalized scores
          target: [N, L]
          coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        bsz, num_classes = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        # [N, L]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        t1_loss, t2_loss = None, None
        if watch_t1_t2_loss:
            t1_loss, t2_loss = loss.chunk(2)
            t1_mask, t2_mask = label_mask.chunk(2)
            t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
            t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())
            
        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        ppl = torch.exp(nll_loss)
        
        logging_output = {
            'nll_loss': nll_loss.data,
            'ppl': ppl.data,
            'fullseq_loss': fullseq_loss.data,
            'fullseq_nll_loss': fullseq_nll_loss.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens,
            'weight_diff_loss': loss.data
        }
        
        if cal_constant_loss:
            constant_weights = weights.new_ones(size=weights.size())
            constant_loss, _ = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            constant_loss = constant_loss * constant_weights
            constant_loss = (constant_loss * label_mask).sum() / sample_size
            logging_output['constant_diff_loss'] = constant_loss.data

        if watch_t1_t2_loss:
            logging_output['weight_diff_t1_loss'] = t1_loss.data
            logging_output['weight_diff_t2_loss'] = t2_loss.data
        
        return loss, logging_output
