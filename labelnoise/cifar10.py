"""
CIFAR-10 loader with optional synthetic noise.
This is separate from CIFAR-10N (the human-noisy version).
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from labelnoise.noise import (
    inject_symmetric_noise,
    inject_asymmetric_noise,
    default_cifar10_asym_mapping,
    save_noisy_mask,
)
from utils import log, set_seed


# used by both train/test transforms
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)


# ------------------------------------------
# Simple wrapper for noisy CIFAR-10 samples
# ------------------------------------------
class CIFAR10Noisy(Dataset):
    def __init__(self, images, labels, transform=None, return_index=False):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform:
            img = self.transform(img)

        if self.return_index:
            return img, self.labels[i], i
        return img, self.labels[i]


# ------------------------------------------
# Synthetic-noise CIFAR-10 loader
# ------------------------------------------
def get_cifar10_loaders(
    data_root="./data",
    batch_size=128,
    noise_type="clean",      # clean | symmetric | asymmetric
    noise_rate=0.0,
    seed=42,
    num_workers=4,
):
    set_seed(seed)

    # basic augs for train / normal preprocess for test
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # load full CIFAR-10
    train_raw = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=None
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    imgs = train_raw.data
    labels = np.array(train_raw.targets)

    # simple deterministic 45k train / 5k val split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)

    val_n = 5000
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    tr_imgs, tr_labels = imgs[train_idx], labels[train_idx]
    val_imgs, val_labels = imgs[val_idx], labels[val_idx]

    # ------------------------------------------
    # noise injection (only train split)
    # ------------------------------------------
    if noise_type == "clean":
        log("CIFAR-10 (clean) selected")
        noisy = tr_labels.copy()
        mask = np.zeros_like(noisy, dtype=bool)
        observed = 0.0

    elif noise_type == "symmetric":
        log(f"Symmetric noise @ {noise_rate}")
        noisy, mask, observed = inject_symmetric_noise(
            labels=tr_labels,
            noise_rate=noise_rate,
            num_classes=10,
            rng=rng,
        )

    elif noise_type == "asymmetric":
        log(f"Asymmetric noise @ {noise_rate}")
        mapping = default_cifar10_asym_mapping()
        noisy, mask, observed = inject_asymmetric_noise(
            labels=tr_labels,
            noise_rate=noise_rate,
            mapping=mapping,
            rng=rng,
        )

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    save_noisy_mask(
        mask,
        out_dir=os.path.join(data_root, "cifar10_noisy_info"),
        filename=f"mask_{noise_type}_{noise_rate}.npy",
    )

    # ------------------------------------------
    # wrap into datasets
    # ------------------------------------------
    train_set = CIFAR10Noisy(
        images=tr_imgs,
        labels=noisy,
        transform=transform_train,
        return_index=True,
    )

    val_set = CIFAR10Noisy(
        images=val_imgs,
        labels=val_labels,
        transform=transform_test,
    )

    # test_set already has transforms
    # ------------------------------------------

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    log(f"Train={len(train_set)}  Val={len(val_set)}  Test={len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader


__all__ = ["get_cifar10_loaders", "CIFAR10Noisy"]