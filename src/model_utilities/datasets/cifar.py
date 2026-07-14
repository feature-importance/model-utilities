from collections import defaultdict
from os import PathLike
from typing import Callable, Optional, Type, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from model_utilities.transforms.cifar_presets import (
    ImageClassificationEval,
    ImageClassificationTraining,
)

Path = Union[str, PathLike[str]]

__all__ = [
    "cifar10_loaders",
    "cifar100_loaders",
    "class_balanced_subset",
]


def class_balanced_subset(
    dataset: Dataset,
    sample_cap: Optional[int],
    *,
    num_classes: int,
    generator: Optional[torch.Generator] = None,
) -> Dataset:
    """Return a subset capped as evenly as possible across classes."""
    if sample_cap is None:
        return dataset
    if sample_cap < 0:
        raise ValueError("sample_cap must be non-negative or None")
    if sample_cap >= len(dataset):
        return dataset

    targets = _get_targets(dataset)
    if len(targets) != len(dataset):
        raise ValueError("dataset targets must have the same length as dataset")

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)

    per_class = sample_cap // num_classes
    remainder = sample_cap % num_classes
    selected_indices = []

    for class_index in range(num_classes):
        class_indices = indices_by_class[class_index]
        class_cap = per_class + int(class_index < remainder)
        if class_cap > len(class_indices):
            raise ValueError(
                f"sample_cap={sample_cap} requires {class_cap} samples from "
                f"class {class_index}, but only {len(class_indices)} are available"
            )
        selected_indices.extend(
            _sample_indices(class_indices, class_cap, generator=generator)
        )

    return Subset(dataset, selected_indices)


def cifar10_loaders(
    root: Path,
    batch_size: int,
    *,
    train_sample_cap: Optional[int] = None,
    eval_sample_cap: Optional[int] = None,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool = True,
    train_shuffle: bool = True,
    eval_shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    **loader_kwargs,
) -> tuple[DataLoader, DataLoader]:
    return _cifar_loaders(
        CIFAR10,
        root,
        batch_size,
        num_classes=10,
        train_sample_cap=train_sample_cap,
        eval_sample_cap=eval_sample_cap,
        num_workers=num_workers,
        download=download,
        pin_memory=pin_memory,
        train_shuffle=train_shuffle,
        eval_shuffle=eval_shuffle,
        generator=generator,
        train_transform=train_transform,
        eval_transform=eval_transform,
        **loader_kwargs,
    )


def cifar100_loaders(
    root: Path,
    batch_size: int,
    *,
    train_sample_cap: Optional[int] = None,
    eval_sample_cap: Optional[int] = None,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool = True,
    train_shuffle: bool = True,
    eval_shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    **loader_kwargs,
) -> tuple[DataLoader, DataLoader]:
    return _cifar_loaders(
        CIFAR100,
        root,
        batch_size,
        num_classes=100,
        train_sample_cap=train_sample_cap,
        eval_sample_cap=eval_sample_cap,
        num_workers=num_workers,
        download=download,
        pin_memory=pin_memory,
        train_shuffle=train_shuffle,
        eval_shuffle=eval_shuffle,
        generator=generator,
        train_transform=train_transform,
        eval_transform=eval_transform,
        **loader_kwargs,
    )


def _cifar_loaders(
    dataset_class: Type[Dataset],
    root: Path,
    batch_size: int,
    *,
    num_classes: int,
    train_sample_cap: Optional[int],
    eval_sample_cap: Optional[int],
    num_workers: int,
    download: bool,
    pin_memory: bool,
    train_shuffle: bool,
    eval_shuffle: bool,
    generator: Optional[torch.Generator],
    train_transform: Optional[Callable],
    eval_transform: Optional[Callable],
    **loader_kwargs,
) -> tuple[DataLoader, DataLoader]:
    train_data = dataset_class(
        root=root,
        train=True,
        download=download,
        transform=train_transform or ImageClassificationTraining(),
    )
    eval_data = dataset_class(
        root=root,
        train=False,
        download=download,
        transform=eval_transform or ImageClassificationEval(),
    )

    train_data = class_balanced_subset(
        train_data,
        train_sample_cap,
        num_classes=num_classes,
        generator=generator,
    )
    eval_data = class_balanced_subset(
        eval_data,
        eval_sample_cap,
        num_classes=num_classes,
        generator=generator,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=train_shuffle and len(train_data) > 0,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=eval_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **loader_kwargs,
    )
    return train_loader, eval_loader


def _get_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    raise ValueError("dataset must expose class targets via targets or labels")


def _sample_indices(
    indices: list[int],
    sample_count: int,
    *,
    generator: Optional[torch.Generator],
) -> list[int]:
    if sample_count == 0:
        return []
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[i] for i in permutation[:sample_count]]
