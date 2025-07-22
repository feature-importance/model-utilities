from typing import Tuple

import torch
from babel.messages.pofile import denormalize
from model_utilities.transforms.unnormalize import Denormalize
from torch import nn, Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from .unnormalize import Denormalize as _Denormalize

__all__ = ["ImageClassificationTraining", "ImageClassificationEval"]


class _ImageClassificationBase(nn.Module):
    def __init__(
        self,
        *,
        training: bool,
        crop_size: int = 32,
        mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
        std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010),
        padding: int = 4,
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.training = training
        self.crop_size = [crop_size]
        self.mean = list(mean)
        self.std = list(std)
        self.random_crop = RandomCrop(crop_size, padding=padding)
        self.random_horizontal_flip = RandomHorizontalFlip(flip_prob)

    def forward(self, img: Tensor) -> Tensor:
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)

        if self.training:
            img = self.random_crop(img)
            img = self.random_horizontal_flip(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        if self.training:
            return (
                "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single "
                "``(C, H, W)`` image ``torch.Tensor`` objects. "
                f"The images are cropped with ``crop_size={self.crop_size}``, "
                f"padded by {self.padding} pixels, randomly cropped to "
                f"{self.crop_size} pixels and then randomly"
                f"flipped horizontally. "
                f"Finally the values are first rescaled to "
                f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}``"
                f"and ``std={self.std}``."
            )

        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single "
            "``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are cropped with ``crop_size={self.crop_size}``. "
            f"Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` "
            f"and ``std={self.std}``."
        )


class ImageClassificationTraining(_ImageClassificationBase):
    def __init__(self):
        super().__init__(training=True)


class ImageClassificationEval(_ImageClassificationBase):
    def __init__(self):
        super().__init__(training=False)


class Denormalize(_Denormalize):
    def __init__(self):
        super().__init__(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
