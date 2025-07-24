"""
CIFAR ConvNeXt models pretrained weights.
"""

from torchvision.models import (Weights, convnext_base, convnext_tiny,
                                convnext_large, convnext_small)

from ..models import _COMMON_META_CIFAR10, _COMMON_META_CIFAR100, WeightsEnum
from ..transforms._cifar_presets import ImageClassificationEval

__all__ = [
    "convnext_tiny", "convnext_small", "convnext_large", "convnext_base",
    "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Base_Weights", "ConvNeXt_Large_Weights",
]


class ConvNeXt_Base_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 87668964,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.491,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 87668964,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.488,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 87668964,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.493,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 87576714,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.781,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 87576714,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.785,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_base-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 87576714,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.810,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ConvNeXt_Large_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 196245706,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.792,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 196245706,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.782,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 196245706,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.785,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 196384036,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.500,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 196384036,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.495,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_large-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 196384036,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.498,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ConvNeXt_Tiny_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 27827818,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.794,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 27827818,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.785,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 27827818,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.797,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 27897028,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.466,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 27897028,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.469,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_tiny-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 27897028,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.463,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ConvNeXt_Small_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 49462378,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.784,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 49462378,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.782,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 49462378,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.792,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 49531588,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.472,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 49531588,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.462,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities/convnext_small-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 49531588,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.467,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0
