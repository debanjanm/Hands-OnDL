"""Data loading utilities."""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_cifar10(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 train/val/test loaders with ImageNet normalization."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    test_ds    = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


def get_image_folder(
    root: str,
    batch_size: int = 64,
    val_split: float = 0.15,
    img_size: int = 224,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, list]:
    """Generic ImageFolder loader (train/val split from one folder)."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(root, transform=transform)
    n_val   = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        ds.classes,
    )
