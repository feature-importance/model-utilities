import torch.nn.functional
import torchbearer

from torchbearer.callbacks import Callback
from torchbearer.bases import get_metric


class StopOnMaxAcc(Callback):
    """Callback to stop training when training acc is 1.0
    """

    def __init__(self, epsilon=0.0001):
        super().__init__()

        self.epsilon = epsilon

    def on_end_epoch(self, state):
        if 'binary_acc' in state[torchbearer.METRICS]:
            acc = state[torchbearer.METRICS]['binary_acc']
        elif 'cat_acc' in state[torchbearer.METRICS]:
            acc = state[torchbearer.METRICS]['cat_acc']
        elif 'acc' in state[torchbearer.METRICS]:
            acc = state[torchbearer.METRICS]['acc']
        else:
            return

        if acc >= 1 - self.epsilon:
            state[torchbearer.STOP_TRAINING] = True


@torchbearer.callbacks.on_sample
@torchbearer.callbacks.on_sample_validation
def reformat_ytrue_bce(state):
    state[torchbearer.Y_TRUE] = state[torchbearer.Y_TRUE].float().unsqueeze(1)
