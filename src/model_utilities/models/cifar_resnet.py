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
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights"
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58156618,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58341028,
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


class ResNet18_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11181642,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23712932,
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
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23528522,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet101_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42512970,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42697380,
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


class ResNet18_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 11173962,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 11220132,
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


class ResNet50_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 23520842,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 23705252,
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


class ResNet34_3x3_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21282122,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21328292,
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


class ResNet34_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 21335972,
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
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 21289802,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet152_Weights(WeightsEnum):
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 58348708,
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
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 58164298,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    DEFAULT = CIFAR10_s0


class ResNet101_Weights(WeightsEnum):
    CIFAR10_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.556,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s1 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.552,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR10_s2 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR10,
            "num_params": 42520650,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                "CIFAR10": {
                    "acc@1": 0.559,
                }
            },
            "_docs": """These weights reproduce closely the results of the 
            paper using a simple training recipe.""",
        }
    )
    CIFAR100_s0 = Weights(
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_0.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_1.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
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
        url="http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
            "/resnet18-cifar100/model_2.pt",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_CIFAR100,
            "num_params": 42705060,
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
