import os
import yaml
import random
import numpy as np
import torch


# ------------------------------------------------------
# 1. Set random seeds for reproducibility
# ------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------
# 2. Load YAML config
# ------------------------------------------------------
def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------
# 3. Accuracy (top-1)
# ------------------------------------------------------
@torch.no_grad()
def accuracy(output, target):
    preds = torch.argmax(output, dim=1)
    correct = (preds == target).sum().item()
    total = target.size(0)
    return correct / total


# ------------------------------------------------------
# 4. Simple logger
# ------------------------------------------------------
def log(msg: str):
    print(f"[LOG] {msg}")


# ------------------------------------------------------
# 5. Save model checkpoint
# ------------------------------------------------------
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)



# 6. Average meter (for losses, accuracies, etc.)
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)