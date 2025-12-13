"""
Robust CIFAR-10N loader that keeps noisy labels on the training split only.
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.serialization
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Allow PyTorch to safely load numpy arrays contained in CIFAR-10N .pt files
torch.serialization.add_safe_globals([np.ndarray])


class CIFAR10N(CIFAR10):
    """
    CIFAR-10 with optional noisy labels applied to the training split.
    Test split always uses the clean labels provided by torchvision.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
        noise_type: str = "aggre",
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if train:
            noise_path = os.path.join(root, "CIFAR-10_human.pt")
            if not os.path.isfile(noise_path):
                raise FileNotFoundError(
                    f"Could not find noisy labels file at {noise_path}"
                )

            noise_file = torch.load(
                noise_path, map_location="cpu", weights_only=False
            )

            # Map noise types to possible keys, trying them in order.
            key_map = {
                "aggre": ["aggre_label", "aggre_label1", "aggre_label2"],
                "worse": ["worse_label", "worst_label"],
                "worst": ["worse_label", "worst_label"],
                "random1": ["random_label1"],
                "random2": ["random_label2"],
                "clean": ["clean_label"],
            }

            requested_type = (noise_type or "clean").lower()
            key_candidates = key_map.get(requested_type, [])

            selected = None
            for key in key_candidates:
                if key in noise_file:
                    selected = noise_file[key]
                    break

            if selected is None:
                # Gracefully fall back to clean labels if the requested noise is missing.
                selected = noise_file.get("clean_label")
                if selected is None:
                    print(
                        f"[cifar10n] Noise type '{requested_type}' not found; "
                        "falling back to clean labels."
                    )

            # If still missing or size mismatch, keep original clean targets.
            if selected is None or len(selected) != len(self.targets):
                print(
                    f"[cifar10n] Using clean labels (missing/noisy labels size={len(selected) if selected is not None else 'None'})"
                )
                selected = self.targets

            self.targets = selected
        else:
            # Test split should never attempt to use noisy labels.
            pass


def _default_transforms() -> Tuple[T.Compose, T.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


def get_cifar10n_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    noise_type: str = "aggre",
    root: str = "data",
    **kwargs,
):
    """
    Returns train/val/test loaders for CIFAR-10N.

    Noise is applied only to the training set. Validation uses the clean test split
    (no validation split is carved out).
    """

    # Backward compatibility: allow callers to pass subset or data_root.
    if "subset" in kwargs and kwargs["subset"] is not None:
        noise_type = kwargs["subset"]
    if "data_root" in kwargs and kwargs["data_root"] is not None:
        root = kwargs["data_root"]

    train_tf, test_tf = _default_transforms()

    train_data = CIFAR10N(
        root=root,
        train=True,
        download=True,
        transform=train_tf,
        noise_type=noise_type,
    )

    test_data = CIFAR10N(
        root=root,
        train=False,
        download=True,
        transform=test_tf,
        noise_type="clean",
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
