"""
CIFAR VGG models pretrained weights.
"""

from typing import Optional, Any, Mapping

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import vgg16, vgg19
from torchvision.models._utils import _ovewrite_named_param

from ..transforms._cifar_presets import ImageClassificationEval
from .cifar_resnet import WeightsEnum


__all__ = [
    "vgg16", "vgg19"
]

