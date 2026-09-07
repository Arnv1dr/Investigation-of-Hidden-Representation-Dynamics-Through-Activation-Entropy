from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.utils import seed_worker


def _get_dataset_class(dataset_name: str):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return datasets.MNIST

    if dataset_name in ["fashionmnist", "fashion-mnist"]:
        return datasets.FashionMNIST

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Use 'MNIST' or 'FashionMNIST'."
    )


def _get_targets(dataset) -> np.ndarray:
    """Returns class labels from a torchvision dataset as a NumPy array."""
    targets = dataset.targets

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    else:
        targets = np.array(targets)

    return targets


def create_or_load_stratified_split(
    dataset,
    dataset_name: str,
    split_dir: str,
    val_per_class: int,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates or loads a fixed stratified train/validation split.
    """
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)

    dataset_key = dataset_name.lower().replace("-", "")
    train_idx_path = split_path / f"{dataset_key}_train_indices_splitseed{split_seed}_val{val_per_class}.npy"
    val_idx_path = split_path / f"{dataset_key}_val_indices_splitseed{split_seed}_val{val_per_class}.npy"

    if train_idx_path.exists() and val_idx_path.exists():
        train_indices = np.load(train_idx_path)
        val_indices = np.load(val_idx_path)
        return train_indices, val_indices

    targets = _get_targets(dataset)
    rng = np.random.default_rng(split_seed)

    train_indices = []
    val_indices = []

    classes = np.unique(targets)

    for class_label in classes:
        class_indices = np.where(targets == class_label)[0]
        rng.shuffle(class_indices)

        if len(class_indices) < val_per_class:
            raise ValueError(
                f"Class {class_label} has only {len(class_indices)} samples, "
                f"but val_per_class={val_per_class} was requested."
            )

        val_class_indices = class_indices[:val_per_class]
        train_class_indices = class_indices[val_per_class:]

        val_indices.extend(val_class_indices.tolist())
        train_indices.extend(train_class_indices.tolist())

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    np.save(train_idx_path, train_indices)
    np.save(val_idx_path, val_indices)

    return train_indices, val_indices


def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    split_dir: str,
    val_per_class: int,
    split_seed: int,
    train_seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation and test dataloaders.

    The official training set is split into train and validation subsets.
    The official test set is kept separate.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_class = _get_dataset_class(dataset_name)

    full_train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    official_test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_indices, val_indices = create_or_load_stratified_split(
        dataset=full_train_dataset,
        dataset_name=dataset_name,
        split_dir=split_dir,
        val_per_class=val_per_class,
        split_seed=split_seed,
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    generator = torch.Generator()
    generator.manual_seed(train_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        official_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
