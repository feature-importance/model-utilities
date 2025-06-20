import functools
import json
from collections import Counter
from typing import Iterable, List

import torchbearer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR
from torchbearer import Trial
from torchbearer.callbacks import TorchScheduler

class StaticLR(LRScheduler):
    """Purely static learning rate.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        rate: float,
        last_epoch: int = -1,
    ):
        self.rate = rate
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.rate for _ in self.optimizer.param_groups]


class ManualLR(TorchScheduler):
    """
    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `ManualLRSchedule`_
    """
    def __init__(self,
                 schedulers: Iterable[LRScheduler],
                 milestones: List[int],
                 step_on_batch=False):
        super().__init__(functools.partial(SequentialLR,
                                           schedulers=schedulers,
                                           milestones=milestones),
                                            step_on_batch=step_on_batch)

    @classmethod
    def parse(cls, sched):
        scheds = sched.split(",")
        milestones = []
        schedulers = []
        for s in scheds:
            if "<@" in s:
                r, m = s.split("<@")
                milestones.append(int(m))
                schedulers.append(StaticLR(float(r)))
            else:
                schedulers.append(StaticLR(float(s)))

        if len(rates) == len(milestones):
            rates = rates.append(rates[-1])

        return cls(milestones=milestones, rates=rates)


def _parse_schedule(sched):
    if "<@" in sched:
        return ManualLR.parse(sched)

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
        return torchbearer.callbacks.StepLR(step, gamma=factor,
                                            step_on_batch=step_on_batch)
    if schtype.startswith('inv'):
        gamma, power = (float(i) for i in schtype[3:].split(","))
        return torchbearer.callbacks.LambdaLR(
            lambda i: (1 + gamma * i) ** (- power), step_on_batch=step_on_batch)
    elif schtype.startswith('['):
        milestones = json.loads(schtype)
        return torchbearer.callbacks.MultiStepLR(milestones, gamma=factor,
                                                 step_on_batch=step_on_batch)
    elif schtype == 'plateau':
        return torchbearer.callbacks.ReduceLROnPlateau(factor=factor,
                                                       step_on_batch=step_on_batch)

    assert False


def parse_learning_rate_arg(learning_rate: str):
    """
    Parse a learning rate argument into an initial rate and an optional
    scheduler callback.

    Examples:
        0.1                        --  fixed lr
        0.1*0.2@every10            --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@every10B           --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@[10,30,80]         --  decrease by factor=0.2 at epochs 10,
        30, and 80
        0.1*0.2@[10,30,80]B        --  decrease by factor=0.2 at epochs 10,
        30, and 80
        0.1*0.2@plateau            --  decrease by factor=0.2 on validation
        plateau
        0.1*inv0.0001,0.75B        --  decrease by using the old caffe inv
        rule each batch
        0.01<@10,0.1<@20,0.01      --  manual lr


    Args:
        learning_rate: lr string

    Returns:
        tuple of init_lr, callback
    """

    if "<@" in learning_rate:
        sch = _parse_schedule(learning_rate)
        return float(learning_rate.split("<@")[0]), sch

    sp = str(learning_rate).split('*')
    initial = float(sp[0])

    if len(sp) == 1:
        return initial, None
    elif len(sp) == 2:
        return initial, _parse_schedule(sp[1])
