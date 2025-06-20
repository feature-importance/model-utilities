"""
CIFAR ConvNext models pretrained weights.
"""

from torchvision.models import (Weights, convnext_base, convnext_tiny,
                                convnext_large, convnext_small)

from ..models import _COMMON_META_CIFAR10, _COMMON_META_CIFAR100, WeightsEnum
from ..transforms._cifar_presets import ImageClassificationEval

__all__ = [
    "convnext_tiny", "convnext_small", "convnext_large", "convnext_base"
]


class ConvNext_Base_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 134670244,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.654,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 134670244,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.656,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 134670244,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.662,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 134301514,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.917,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 134301514,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.916,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg16-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 134301514,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.917,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0



