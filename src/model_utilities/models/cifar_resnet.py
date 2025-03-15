from torch import nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152, WeightsEnum, Weights)

from ..transforms._cifar_presets import ImageClassificationEval

__all__ = [
    "resnet18_3x3", #"ResNet18_3x3_Weights",
    "resnet34_3x3", #"ResNet34_3x3_Weights",
    "resnet50_3x3", #"ResNet50_3x3_Weights",
    "resnet101_3x3", #"ResNet101_3x3_Weights",
    "resnet152_3x3", #"ResNet152_3x3_Weights",
    #"ResNet18_CIFAR_Weights",
    #"ResNet34_CIFAR_Weights",
    #"ResNet50_CIFAR_Weights",
    #"ResNet101_CIFAR_Weights",
    #"ResNet152_CIFAR_Weights"
]

def _make_resnet_3x3(net):
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), bias=False)
    return net

def resnet18_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet18(*args, **kwargs))


def resnet34_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet34(*args, **kwargs))


def resnet50_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet50(*args, **kwargs))


def resnet101_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet101(*args, **kwargs))


def resnet152_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet152(*args, **kwargs))



