from _testcapi import instancemethod
from types import MethodType
from typing import Type, List, Union, Optional, Any, Mapping

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152, WeightsEnum, Weights)
from torchvision.models._utils import _ovewrite_named_param
from ..transforms._cifar_presets import ImageClassificationEval

__all__ = [
    "resnet18_3x3", "ResNet18_3x3_Weights",
    "resnet34_3x3", "ResNet34_3x3_Weights",
    "resnet50_3x3", "ResNet50_3x3_Weights",
    "resnet101_3x3", "ResNet101_3x3_Weights",
    "resnet152_3x3", "ResNet152_3x3_Weights",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights"
]


def verify(cls, obj: Any) -> Any:
    if obj is not None:
        if type(obj) is str:
            obj = cls[obj.replace(cls.__name__ + ".", "")]
        elif cls.__name__ == obj.__class__.__name__:
            return obj
        elif not isinstance(obj, cls):
            raise TypeError(
                f"Invalid Weight class provided; expected {cls.__name__} but "
                f"received {obj.__class__.__name__}."
            )
    return obj


def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
    name = None
    if "https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities" in self.url:
        name = self.url.replace(
            "https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities",
            "").replace("/", "-")

    return load_state_dict_from_url(self.url,
                                    file_name=name,
                                    map_location=torch.device("cpu"),
                                    *args,
                                    **kwargs)


WeightsEnum.verify = classmethod(verify)
WeightsEnum.get_state_dict = get_state_dict


def _make_resnet_3x3(net, weights: Optional[WeightsEnum] = None,
                     progress: bool = True):
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), bias=False)

    if weights is not None:
        net.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True))

    return net


def resnet18_3x3(*args, weights: Optional[WeightsEnum] = None,
                 progress: bool = True, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    return _make_resnet_3x3(resnet18(*args, **kwargs), weights=weights,
                            progress=progress)


def resnet34_3x3(*args, weights: Optional[WeightsEnum] = None,
                 progress: bool = True, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    return _make_resnet_3x3(resnet34(*args, **kwargs), weights=weights,
                            progress=progress)


def resnet50_3x3(*args, weights: Optional[WeightsEnum] = None,
                 progress: bool = True, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    return _make_resnet_3x3(resnet50(*args, **kwargs), weights=weights,
                            progress=progress)


def resnet101_3x3(*args, weights: Optional[WeightsEnum] = None,
                  progress: bool = True, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    return _make_resnet_3x3(resnet101(*args, **kwargs), weights=weights,
                            progress=progress)


def resnet152_3x3(*args, weights: Optional[WeightsEnum] = None,
                  progress: bool = True, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    return _make_resnet_3x3(resnet152(*args, **kwargs), weights=weights,
                            progress=progress)


_COMMON_META_CIFAR10 = {
    "min_size": (1, 1),
    "categories": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck']
}

_COMMON_META_CIFAR100 = {
    "min_size": (1, 1),
    "categories": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                   'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
                   'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                   'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                   'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
                   'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                   'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
                   'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                   'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                   'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
                   'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
                   'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                   'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger',
                   'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                   'whale', 'willow_tree', 'wolf', 'woman', 'worm']
}


class ResNet152_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.920,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.922,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.928,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
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
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.647,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152_3x3-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.620,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet18_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.872,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.867,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.869,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11227812,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11227812,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11227812,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet50_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.547,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.546,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.526,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.872,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.879,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.880,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet101_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.932,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.920,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.928,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.649,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.613,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101_3x3-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.666,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet18_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
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
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.923,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.922,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.657,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.665,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18_3x3-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
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
    DEFAULT = CIFAR10_s0


class ResNet50_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.926,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.933,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.923,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.642,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.671,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet50_3x3-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.652,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet34_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.927,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.924,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.925,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
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
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.638,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34_3x3-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.653,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet34_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.558,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.569,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.568,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.867,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.871,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet34-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.870,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet152_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.496,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.507,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.553,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.865,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.868,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet152-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.875,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet101_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar10/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.879,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar10/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.871,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar10/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.872,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.502,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s1 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.540,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s2 = Weights(
        url="https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet101-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR100": {
                    "acc@1": 0.543,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0
