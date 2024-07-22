
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr,
            ),
            {'monitor': "val/loss", 'interval': 'epoch'}
        )
    elif cfg.type == 'noam':
        return (
            NoamScheduler(
                optimizer,
                lr=cfg.lr,
                warmup_steps=cfg.warmup_steps,
                model_size=cfg.model_size,
                warmup_init_lr=cfg.get('warmup_init_lr')
            ),
            {'frequency': 1, 'interval': 'step'}
        )
    elif cfg.type == 'polynomial':
        return (
            PolyNomialLRScheduler(
                optimizer,
                total_steps=cfg.total_steps,
                warmup_steps=cfg.warmup_steps,
                lr=cfg.lr,
                lr_end=cfg.lr_end,
                warmup_init_lr=cfg.warmup_init_lr,
                power=cfg.power
            ),
            {'frequency': 1, 'interval': 'step'}
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def inverse_sqrt_lr_schedule(step, warmup_steps, warmup_init_lr, lr_step, decay_step):
    if step == 0:
        step = 1
    if step < warmup_steps:
        return warmup_init_lr + lr_step * step
    else:
        return decay_step * step ** -0.5


class InverseSqrtLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        lr: float = 5e-04,
        warmup_init_lr: float = 1e-07,
    ) -> None:

        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps
        self.lr_step = (lr - warmup_init_lr) / warmup_steps
        self.decay_step = lr * warmup_steps ** 0.5

        def lr_lambda(step):
            return inverse_sqrt_lr_schedule(
                step, warmup_steps, warmup_init_lr, self.lr_step, self.decay_step
            ) / lr

        super().__init__(optimizer, lr_lambda=lr_lambda)


def noam_lr_schedule(step, warmup_steps, factor, model_size):
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))


class NoamScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr,
        warmup_init_lr,
        model_size: int = 128,
        warmup_steps: int = 0,
        factor: int = 2,
    ) -> None:

        # dummy_lr = self.base_lrs[0]
        def lr_lambda(step):
            return noam_lr_schedule(step, warmup_steps, factor, model_size) / lr

        super().__init__(optimizer, lr_lambda=lr_lambda)


def polynomial_lr_schedule(step, total_steps, warmup_steps, warmup_init_lr, lr, lr_end, power):
    if step < warmup_steps:
        return warmup_init_lr + (lr - warmup_init_lr) * step / warmup_steps
    elif step > total_steps:
        return lr_end
    else:
        return lr_end + (lr - lr_end) * (1 - (step - warmup_steps) / (total_steps - warmup_steps)) ** power


class PolyNomialLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int = 1000,
        warmup_steps: int = 0,
        lr: float = 5e-04,
        lr_end: float = 1e-07,
        warmup_init_lr: float = 1e-07,
        power: float = 1.0,
    ) -> None:

        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            return polynomial_lr_schedule(
                step, total_steps, warmup_steps, warmup_init_lr, lr, lr_end, power
            ) / lr

        super().__init__(optimizer, lr_lambda=lr_lambda)