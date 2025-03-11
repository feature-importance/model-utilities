import functools
from collections import Counter
from typing import Iterable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchbearer import Trial
from torchbearer.callbacks import TorchScheduler


class ManualLRScheduler(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        rates (list): List of learning rates. Must have one more element than
        milestones.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        rates: Iterable[float],
        last_epoch: int = -1,
    ):
        assert len(milestones) + 1 == len(rates)

        self.milestones = milestones
        self.rates = rates
        self.current_index = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        while (self.current_index < len(self.milestones) and
               self.last_epoch >= self.milestones[self.current_index]):
            self.current_index += 1
        print(self.last_epoch, self.rates[self.current_index])
        return [self.rates[self.current_index]
                for _ in self.optimizer.param_groups]


class ManualLR(TorchScheduler):
    """
    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `ManualLRSchedule`_
    """
    def __init__(self,
                 milestones: Iterable[int],
                 rates: Iterable[float],
                 step_on_batch=False):
        super().__init__(functools.partial(ManualLRScheduler,
                                           milestones=milestones, rates=rates),
                                            step_on_batch=step_on_batch)

    @classmethod
    def parse(cls, sched):
        scheds = sched.split(",")
        milestones = []
        rates = []
        for s in scheds:
            if "<@" in s:
                r, m = s.split("<@")
                milestones.append(int(m))
                rates.append(float(r))
            else:
                rates.append(float(s))

        if len(rates) == len(milestones):
            rates = rates.append(rates[-1])

        return cls(milestones=milestones, rates=rates)


