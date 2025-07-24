"""
CIFAR VGG models pretrained weights.
"""

from torchvision.models import vgg16, vgg19, Weights

from ..models import _COMMON_META_CIFAR10, _COMMON_META_CIFAR100, WeightsEnum
from ..transforms.cifar_presets import ImageClassificationEval

__all__ = [
    "vgg16", "vgg19", "VGG16_Weights", "VGG19_Weights"
]


class VGG16_Weights(WeightsEnum):
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


class VGG19_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 139979940,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.663,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 139979940,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.658,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 139979940,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.660,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 139611210,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.913,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 139611210,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.914,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/vgg19-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 139611210,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.912,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0
