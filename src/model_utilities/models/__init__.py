from typing import Any, Mapping

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import WeightsEnum

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


# We patch the enum so that it allows us to use the original model and
# correctly handles our models.
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
    if ("https://marc.ecs.soton.ac.uk/pytorch-models/model-utilities" in
            self.url):
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
