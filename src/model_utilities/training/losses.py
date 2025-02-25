import torch
import torch.nn as nn


def hinge_loss(out, y):
    y_ = (2 * y.float() - 1)
    if y_.shape != out.shape:
        y_ = y_.unsqueeze(1)
    assert (y_.shape == out.shape)
    return torch.mean(nn.functional.relu(1 - out * y_))
