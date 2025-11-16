import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils import log, set_seed
from labelnoise.noise import save_noisy_mask


# Normalization values for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)


class CIFAR10N_Dataset(Dataset):
    """
    Basic wrapper around CIFAR-10 images but with the noisy labels
    provided by the CIFAR-10N dataset.
    """
    def __init__(self, images, noisy_labels, transform=None, return_index=False):
        self.images = images
        self.labels = noisy_labels.astype(np.int64)
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.return_index:
            return img, label, idx

        return img, label


def get_cifar10n_loaders(
        data_root="./data",
        batch_size=128,
        subset="aggre",
        seed=42,
        num_workers=4,
):
    """
    Loads CIFAR-10 normally but replaces the original labels with
    the human-annotated noisy labels from CIFAR-10N.
    """
    set_seed(seed)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Base CIFAR-10 images
    cifar_train = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=None
    )
    cifar_test = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf
    )

    imgs = cifar_train.data
    clean_labels = np.array(cifar_train.targets)

    # Load the human labels
    cifar10n_dir = os.path.join(data_root, "CIFAR-10N")
    noisy_path = os.path.join(cifar10n_dir, f"{subset}.npy")

    if not os.path.isfile(noisy_path):
        raise FileNotFoundError(
            f"Missing noisy label file: {noisy_path}. "
            "Place the CIFAR-10N .npy files inside data/CIFAR-10N/."
        )

    noisy_labels = np.load(noisy_path)

    if len(noisy_labels) != len(clean_labels):
        raise ValueError("Noisy labels and CIFAR-10 samples do not match in length.")

    # Identify which samples were changed by annotators
    noisy_mask = (noisy_labels != clean_labels)
    save_noisy_mask(
        noisy_mask,
        out_dir=os.path.join(data_root, "cifar10n_noisy_info"),
        filename=f"mask_{subset}.npy"
    )

    noise_fraction = noisy_mask.mean()
    log(f"CIFAR-10N subset='{subset}', noise level ≈ {noise_fraction:.3f}")

    # Build final loaders
    train_set = CIFAR10N_Dataset(
        images=imgs,
        noisy_labels=noisy_labels,
        transform=train_tf,
        return_index=True
    )

    val_loader = None  # CIFAR-10N doesn't include a validation split

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        cifar_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    log(f"Train samples: {len(train_set)}, Test samples: {len(cifar_test)}")

    return train_loader, val_loader, test_loader


__all__ = ["get_cifar10n_loaders", "CIFAR10N_Dataset"]