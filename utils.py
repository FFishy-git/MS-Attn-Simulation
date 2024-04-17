import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from lightning.pytorch.utilities.parsing import AttributeDict
import torch

def generate_attributedict(d: dict):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = generate_attributedict(v)
    return AttributeDict(d)

class AdaptiveLRScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            learning_rate: float,
            grad_elbow: float,
            max_learning_rate: float,
            min_learning_rate: float,
            method: str = "elbow_decay", # "elbow_decay" or "constant"
    ):
        self.learning_rate = learning_rate
        self.last_lr = learning_rate
        self.grad_elbow = grad_elbow
        self.cumulative_time = 0
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.method = method
        super().__init__(optimizer=optimizer, last_epoch=-1)
    
    def get_lr(self):
        # get the largest gradient
        # max_grad_ls = [p.grad.data.abs().max().item() for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
        # max_weight_ls = [p.data.abs().max().item() for p in self.optimizer.param_groups[0]['params']]
        # if len(max_grad_ls) == 0:
        #     return [self.last_lr for group in self.optimizer.param_groups]
        # max_grad = max(max_grad_ls)
        # max_weight = max(max_weight_ls)

        # compute the learning rate based on the largest gradient
        # lr = min([max_weight * .1 / max_grad, self.learning_rate / max_grad])
        # lr = self.learning_rate / (self.grad_elbow + max_grad)

        # ratio = [(p.data.abs() / (p.grad.data.abs())).max().item() for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
        # if len(ratio) == 0:
        #     return [self.last_lr for group in self.optimizer.param_groups]
        # lr = max(ratio) if max(ratio)
        # self.cumulative_time += self.last_lr
        # self.last_lr = lr
        # return [lr for group in self.optimizer.param_groups]
        grad_max = [ p.grad.data.abs().max().item() for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
        if len(grad_max) == 0:
            return [self.last_lr for group in self.optimizer.param_groups]
        if self.method == "elbow_decay":
            lr = min([0.05 / max(grad_max), self.max_learning_rate]) if max(grad_max) > self.grad_elbow else self.max_learning_rate
        elif self.method == "constant":
            lr = self.learning_rate
        self.cumulative_time += self.last_lr
        self.last_lr = lr
        return [lr for group in self.optimizer.param_groups]
    
        


class CosineAnnealingWarmup(_LRScheduler):

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            learning_rate: float,
            min_lr: float,
            lr_decay_steps: int,
            verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, last_epoch=-1, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self.learning_rate * self._step_count / self.warmup_steps
                    for group in self.optimizer.param_groups]
        if self._step_count > self.lr_decay_steps:
            return [self.min_lr for group in self.optimizer.param_groups]
        
        decay_ratio = (
            (self._step_count - self.warmup_steps)
            / (self.lr_decay_steps - self.warmup_steps)
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.learning_rate - self.min_lr)
                for group in self.optimizer.param_groups]

def entropy(p, axis=-1):
    # Assuming p is a 2D tensor where each row is a probability distribution
    log_p = torch.log(p)
    entropy = -torch.sum(p * log_p, dim=axis)
    return entropy