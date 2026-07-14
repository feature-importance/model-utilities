import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

from model_utilities.datasets.cifar import (
    cifar10_loaders,
    class_balanced_subset,
)


class DummyCIFAR(Dataset):
    def __init__(self, root=None, train=True, download=False, transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        samples_per_class = 10 if train else 5
        self.targets = [
            class_index
            for class_index in range(10)
            for _ in range(samples_per_class)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([index]), self.targets[index]


class TestCIFARLoaders(unittest.TestCase):
    def test_class_balanced_subset_distributes_remainder(self):
        dataset = DummyCIFAR(train=True)
        subset = class_balanced_subset(
            dataset,
            23,
            num_classes=10,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertIsInstance(subset, Subset)
        self.assertEqual(23, len(subset))
        counts = self._class_counts(dataset, subset.indices)
        self.assertEqual([3, 3, 3, 2, 2, 2, 2, 2, 2, 2], counts)

    def test_class_balanced_subset_returns_dataset_when_uncapped(self):
        dataset = DummyCIFAR(train=True)

        self.assertIs(dataset, class_balanced_subset(dataset, None, num_classes=10))
        self.assertIs(dataset, class_balanced_subset(dataset, len(dataset), num_classes=10))

    def test_cifar10_loaders_apply_balanced_caps(self):
        with patch("model_utilities.datasets.cifar.CIFAR10", DummyCIFAR):
            train_loader, eval_loader = cifar10_loaders(
                root="~/data",
                batch_size=4,
                train_sample_cap=20,
                eval_sample_cap=10,
                num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )

        self.assertEqual(20, len(train_loader.dataset))
        self.assertEqual(10, len(eval_loader.dataset))

        train_dataset = train_loader.dataset.dataset
        eval_dataset = eval_loader.dataset.dataset
        self.assertEqual(
            [2] * 10,
            self._class_counts(train_dataset, train_loader.dataset.indices),
        )
        self.assertEqual(
            [1] * 10,
            self._class_counts(eval_dataset, eval_loader.dataset.indices),
        )

    def test_negative_sample_cap_raises(self):
        with self.assertRaises(ValueError):
            class_balanced_subset(DummyCIFAR(), -1, num_classes=10)

    def _class_counts(self, dataset, indices):
        return [
            sum(1 for index in indices if dataset.targets[index] == class_index)
            for class_index in range(10)
        ]


if __name__ == "__main__":
    unittest.main()
