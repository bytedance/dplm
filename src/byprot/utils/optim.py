
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import torch
from torch.optim.adamw import adamw
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except:
    pass

def get_optimizer(cfg, params):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    elif cfg.type == 'adamw':
        return AdamW(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    elif cfg.type == 'fusedadam':
        return FusedAdam(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        # deepspeed.ops.op_builder.CPUAdamBuilder().load()
        # return DeepSpeedCPUAdam(
        #     model_params=params,
        #     lr=cfg.lr,
        #     weight_decay=cfg.weight_decay,
        #     betas=cfg.betas,
        # )
        
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


class AdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'].cpu())

            adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'])

        return loss
