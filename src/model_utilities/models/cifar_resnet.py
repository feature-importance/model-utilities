from torch import nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152, WeightsEnum, Weights)

from ..transforms._cifar_presets import ImageClassificationEval

__all__ = [
    "resnet18_3x3", "ResNet18_3x3_Weights",
    "resnet34_3x3", "ResNet34_3x3_Weights",
    "resnet50_3x3", "ResNet50_3x3_Weights",
    "resnet101_3x3", "ResNet101_3x3_Weights",
    "resnet152_3x3", "ResNet152_3x3_Weights",
    "ResNet18_CIFAR_Weights",
    "ResNet34_CIFAR_Weights",
    "ResNet50_CIFAR_Weights",
    "ResNet101_CIFAR_Weights",
    "ResNet152_CIFAR_Weights"
]


def _make_resnet_3x3(net):
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), bias=False)
    return net


_COMMON_META_CIFAR10 = {
    "min_size": (1, 1),
    "categories": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck']
}
_COMMON_META_CIFAR100 = {
    "min_size": (1, 1),
    "categories": []
}


class ResNet18_3x3_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the  
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


def resnet18_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet18(*args, **kwargs))


class ResNet34_3x3_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


def resnet34_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet34(*args, **kwargs))


class ResNet50_3x3_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


def resnet50_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet50(*args, **kwargs))


class ResNet101_3x3_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


def resnet101_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet101(*args, **kwargs))


class ResNet152_3x3_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


def resnet152_3x3(*args, **kwargs):
    return _make_resnet_3x3(resnet152(*args, **kwargs))


class ResNet18_CIFAR_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


class ResNet34_CIFAR_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


class ResNet50_CIFAR_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


class ResNet101_CIFAR_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1


class ResNet152_CIFAR_Weights(WeightsEnum):
    CIFAR10_V1 = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        },
    )
    DEFAULT = CIFAR10_V1
