import os
import numpy as np
from typing import Tuple, Dict

from utils import log


# ------------------------------------------------------
# Helper: save noisy mask
# ------------------------------------------------------
def save_noisy_mask(mask: np.ndarray, out_dir: str, filename: str = "noisy_mask.npy"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    np.save(path, mask.astype(bool))
    log(f"Saved noisy mask to {path}")


# ------------------------------------------------------
# Symmetric noise
# ------------------------------------------------------
def inject_symmetric_noise(
    labels: np.ndarray,
    noise_rate: float,
    num_classes: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Flip p% of labels uniformly to a WRONG class.
    Returns:
        noisy_labels: new label array
        noisy_mask: boolean mask where labels were changed
        observed_rate: fraction actually flipped
    """
    assert 0.0 <= noise_rate <= 1.0
    n = len(labels)
    noisy_labels = labels.copy()

    num_noisy = int(noise_rate * n)
    if num_noisy == 0:
        noisy_mask = np.zeros_like(labels, dtype=bool)
        return noisy_labels, noisy_mask, 0.0

    noisy_indices = rng.choice(n, size=num_noisy, replace=False)

    for idx in noisy_indices:
        true_label = labels[idx]
        # choose a random label different from the true one
        candidates = list(range(num_classes))
        candidates.remove(true_label)
        noisy_labels[idx] = rng.choice(candidates)

    noisy_mask = noisy_labels != labels
    observed_rate = noisy_mask.mean().item()

    log(
        f"[Symmetric noise] target rate={noise_rate:.3f}, "
        f"observed={observed_rate:.3f}, num_noisy={noisy_mask.sum()}/{n}"
    )

    return noisy_labels, noisy_mask, observed_rate


# ------------------------------------------------------
# Asymmetric noise
# ------------------------------------------------------
def default_cifar10_asym_mapping() -> Dict[int, int]:
    """
    Default asymmetric mapping for CIFAR-10:
      automobile <-> truck
      cat <-> dog
      deer <-> horse
      bird -> airplane
    Classes are indexed 0..9 using standard CIFAR-10 order:
      0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
      5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    """
    mapping = {
        1: 9,  # automobile -> truck
        9: 1,  # truck -> automobile
        3: 5,  # cat -> dog
        5: 3,  # dog -> cat
        4: 7,  # deer -> horse
        7: 4,  # horse -> deer
        2: 0,  # bird -> airplane
        # others (0, 6, 8) stay unchanged
    }
    return mapping


def inject_asymmetric_noise(
    labels: np.ndarray,
    noise_rate: float,
    mapping: Dict[int, int],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Flip p% of labels according to a class-conditional mapping.
    Only classes in `mapping` are candidates for flipping.
    """
    assert 0.0 <= noise_rate <= 1.0
    noisy_labels = labels.copy()
    n = len(labels)

    # only indices where mapping is defined
    candidate_indices = np.where(np.isin(labels, list(mapping.keys())))[0]
    num_candidates = len(candidate_indices)
    num_noisy = int(noise_rate * num_candidates)

    if num_noisy == 0 or num_candidates == 0:
        noisy_mask = noisy_labels != labels
        observed_rate = noisy_mask.mean().item()
        log(
            f"[Asymmetric noise] target rate={noise_rate:.3f}, "
            f"observed={observed_rate:.3f} (no candidates or zero rate)"
        )
        return noisy_labels, noisy_mask, observed_rate

    noisy_indices = rng.choice(candidate_indices, size=num_noisy, replace=False)

    for idx in noisy_indices:
        original = labels[idx]
        noisy_labels[idx] = mapping.get(original, original)

    noisy_mask = noisy_labels != labels
    observed_rate = noisy_mask.mean().item()

    log(
        f"[Asymmetric noise] target rate={noise_rate:.3f}, "
        f"observed={observed_rate:.3f}, num_noisy={noisy_mask.sum()}/{n}"
    )

    return noisy_labels, noisy_mask, observed_rate
