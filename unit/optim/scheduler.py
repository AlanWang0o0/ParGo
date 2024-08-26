import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr   


# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        NOTE: this scheduler should be called per iteration/step.
        
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
        ):
        assert warmup_steps <= first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            # return self.base_lrs
            return 0.
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class StepLRWarmupPerIter(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps: int,
                 init_lr: float = 0.1,
                 min_lr: float = 0.001,
                 gamma: float = 0.9,
                 last_epoch: int = -1
        ):

        self.cycle_steps = cycle_steps  # first cycle step size
        self.max_lr = init_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.gamma = gamma  # decrease rate of max learning rate by cycle
        self.run_epochs = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            # return self.base_lrs
            return 0.
        elif self.run_epochs == 0:
            # assert self.step_in_cycle < self.cycle_steps
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.cycle_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (self.gamma ** (self.run_epochs - 1 + self.step_in_cycle / self.cycle_steps)) for base_lr in self.base_lrs]

    def step(self):
        self.step_in_cycle = self.step_in_cycle + 1
        if self.step_in_cycle >= self.cycle_steps:
            self.run_epochs += 1
            self.step_in_cycle = self.step_in_cycle - self.cycle_steps

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class StepLRWarmupPerIterStages(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps: int,
                 init_lr: float = 0.1,
                 min_lr: float = 0.001,
                 gamma: float = 0.9,
                 last_epoch: int = -1,
                 epoch_stages: list = [30],
        ):

        self.cycle_steps = cycle_steps  # first cycle step size
        self.max_lr = init_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.gamma = gamma  # decrease rate of max learning rate by cycle
        self.run_epochs = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle
        self.epoch_stages = epoch_stages    # example: [30, 10]
        self.stage_index = 0

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            # return self.base_lrs
            return 0.
        elif self.run_epochs == 0:
            # assert self.step_in_cycle < self.cycle_steps
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.cycle_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (self.gamma ** (self.run_epochs - 1 + self.step_in_cycle / self.cycle_steps)) for base_lr in self.base_lrs]

    def step(self):
        self.step_in_cycle = self.step_in_cycle + 1
        if self.step_in_cycle >= self.cycle_steps:
            self.run_epochs += 1
            self.step_in_cycle = self.step_in_cycle - self.cycle_steps
        
        if self.run_epochs == self.epoch_stages[self.stage_index]:
            self.run_epochs = 0
            self.stage_index += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class LinearWarmupCosineLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.init_lr = init_lr

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else min_lr

        self.total_cur_step = -2

        # init base_lrs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer) # 会默认调用一次step()
        
    def get_lr(self, base_lr):
        if self.total_cur_step < self.warmup_steps:
            return min(base_lr,  self.min_lr + (base_lr - self.min_lr) * max(self.total_cur_step, 0) / max(1, self.warmup_steps))
        else:
            return (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * max(self.total_cur_step, 0) / self.total_steps)) + self.min_lr

    def step(self):
        self.total_cur_step += 1
        # lr = self.get_lr()
        # print(self.total_cur_step, lr)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.get_lr(self.base_lrs[i])
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
