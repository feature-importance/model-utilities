import functools
import json
from typing import List, Callable

import torch.optim.lr_scheduler
import torchbearer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR
from torchbearer.callbacks import TorchScheduler


class StaticLR(LRScheduler):
    """Purely static learning rate.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, rate: float, last_epoch: int = -1):
        self.rate = rate
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.rate for _ in self.optimizer.param_groups]


class LazySequentialLR(SequentialLR):
    """
    Extension to SequentialLR that allows for schedulers to be specified in the form of functions that will return
    the scheduler from the respective optimiser. This allows lazy instantiation of individual schedulers in the
    case that you don't immediately have access to the optimiser.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 schedulers: List[Callable[[Optimizer], LRScheduler]],
                 milestones: List[int],
                 last_epoch: int = -1):
        schedulers = [s(optimizer) for s in schedulers]
        super().__init__(optimizer, schedulers, milestones, last_epoch)


class ManualLR(TorchScheduler):
    """
    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `ManualLRSchedule`_
    """

    def __init__(self,
                 schedulers: List[Callable[[Optimizer], LRScheduler]],
                 milestones: List[int],
                 step_on_batch=False):
        super().__init__(functools.partial(LazySequentialLR, schedulers=schedulers, milestones=milestones),
                         step_on_batch=step_on_batch)

    @classmethod
    def parse(cls, sched):
        scheds = sched.split(";")
        milestones = []
        schedulers = []
        for s in scheds:
            if "<@" in s:
                r, m = s.split("<@")
                milestones.append(int(m))

                try:
                    schedulers.append(functools.partial(StaticLR, rate=float(r)))
                except ValueError:
                    schedulers.append(_parse_schedule(r)._scheduler_builder)
            else:
                try:
                    schedulers.append(functools.partial(StaticLR, rate=float(s)))
                except ValueError:
                    schedulers.append(_parse_schedule(s)._scheduler_builder)

        if len(schedulers) == len(milestones):
            schedulers = schedulers.append(schedulers[-1])

        return cls(milestones=milestones, schedulers=schedulers)


def _parse_schedule(sched):
    if "<@" in sched:
        return ManualLR.parse(sched)

    if "*" in sched:
        print(sched)
        sched = sched[sched.index("*") + 1:]
        print(sched)

    if '@' in sched:
        factor, schtype = sched.split('@')
        factor = float(factor)
    else:
        factor, schtype = None, sched

    step_on_batch = False
    if schtype.endswith('B') or schtype.endswith('b'):
        step_on_batch = True
        schtype = schtype[:-1]

    if schtype.startswith('every'):
        step = int(schtype[5:])
        if not step_on_batch:
            return torchbearer.callbacks.StepLR(step, gamma=factor, step_on_batch=step_on_batch)
        else:
            # the step on batch code applies the update before the first batch (compared to the end of the epoch)
            # so for consistency in making the init_lr behave sanely, we add an extra constant lr step
            s1 = functools.partial(torch.optim.lr_scheduler.MultiplicativeLR, lr_lambda=lambda epoch: 1.0)
            s2 = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=step, gamma=factor)
            return ManualLR([s1, s2], [step], step_on_batch=True)
    if schtype.startswith('inv'):
        gamma, power = (float(i) for i in schtype[3:].split(","))
        if not step_on_batch:
            return torchbearer.callbacks.LambdaLR(lambda i: (1 + gamma * i) ** (- power), step_on_batch=False)
        else:
            # the step on batch code applies the update before the first batch (compared to the end of the epoch)
            # so for consistency in making the init_lr behave sanely, we add an extra constant lr step
            s1 = functools.partial(torch.optim.lr_scheduler.MultiplicativeLR, lr_lambda=lambda epoch: 1.0)
            s2 = functools.partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda i: (1 + gamma * i) ** (- power))
            return ManualLR([s1, s2], [1], step_on_batch=True)
    if schtype.startswith('linearWarmup'):
        start_factor, total_iters = (float(i) for i in schtype[12:].split(","))
        total_iters = int(total_iters)

        return torchbearer.callbacks.TorchScheduler(
            functools.partial(torch.optim.lr_scheduler.LinearLR, start_factor=start_factor, total_iters=total_iters),
            step_on_batch=step_on_batch)
    if schtype.startswith('cosineAnnealing'):
        min_lr, t_max = (float(i) for i in schtype[15:].split(","))
        t_max = int(t_max)

        return torchbearer.callbacks.CosineAnnealingLR(t_max, min_lr)
    elif schtype.startswith('['):
        milestones = json.loads(schtype)
        return torchbearer.callbacks.MultiStepLR(milestones, gamma=factor, step_on_batch=step_on_batch)
    elif schtype == 'plateau':
        return torchbearer.callbacks.ReduceLROnPlateau(factor=factor, step_on_batch=step_on_batch)

    assert False


def parse_learning_rate_arg(learning_rate: str):
    """
    Parse a learning rate argument into an initial rate and an optional
    scheduler callback.

    Examples:
        0.1                         --  fixed lr
        0.1*0.2@every10             --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@every10B            --  decrease by factor=0.2 every 10 batches
        0.1*0.2@[10,30,80]          --  decrease by factor=0.2 at epochs 10, 30, and 80
        0.1*0.2@[10,30,80]B         --  decrease by factor=0.2 at batches 10, 30, and 80
        0.1*0.2@plateau             --  decrease by factor=0.2 on validation plateau
        0.1*inv0.0001,0.75B         --  decrease by using the old caffe inv rule each batch
        0.01<@10;0.1<@20;0.01       --  manual lr
        0.01<@10;0.1@every1<@20;0.1 -- 0.01 static for 10 epochs, decay by *0.1 every 1 epoch for 10 epochs,
                                        then 0.1 static
        0.1*linearWarmup0.5,10      --  linear warmup from 0.05 to 0.1 for the first 10 epochs of training
        0.1*cosineAnnealing0.001,10 --  cosine annealing with min lr 0.001 and 10 epochs


    Args:
        learning_rate: lr string

    Returns:
        tuple of init_lr, callback
    """

    if "<@" in learning_rate:
        sch = _parse_schedule(learning_rate)
        initial = learning_rate.split("<@")[0]
        if "*" in initial:
            initial = initial.split("*")[0]
        return float(initial), sch

    sp = str(learning_rate).split('*')
    initial = float(sp[0])

    if len(sp) == 1:
        return initial, None
    elif len(sp) == 2:
        sched = _parse_schedule(sp[1])
        return initial, sched
    raise ValueError("Invalid learning rate string: " + learning_rate)
