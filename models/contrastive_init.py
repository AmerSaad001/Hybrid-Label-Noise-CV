"""
Contrastive init (SimCLR) + CIFAR tweaks for ResNet18.
Mostly just tries to load whatever SimCLR weights I give it,
and falls back to ImageNet if nothing works.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

from utils import log


# ------------------------------------------------
# Try reading SimCLR checkpoint
# ------------------------------------------------
def load_simclr_encoder(ckpt_path: str):
    # If no path or file missing → nothing to load
    if not ckpt_path or not os.path.isfile(ckpt_path):
        log("[ContrastiveInit] SimCLR checkpoint not found.")
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        enc = {}
        for k, v in state.items():
            if "encoder" in k:
                enc[k.replace("encoder.", "")] = v

        if len(enc) == 0:
            log("[ContrastiveInit] No encoder keys in checkpoint.")
            return None

        log("[ContrastiveInit] Loaded SimCLR encoder.")
        return enc

    except Exception as e:
        log(f"[ContrastiveInit] Failed to load SimCLR: {e}")
        return None


# ------------------------------------------------
# Build contrastive-initialized ResNet18
# ------------------------------------------------
def build_contrastive_resnet(
    num_classes=10,
    simclr_ckpt=None,
    freeze_backbone=False,
):
    # See if SimCLR weights exist
    simclr_w = load_simclr_encoder(simclr_ckpt)

    if simclr_w is not None:
        # No torchvision weights here
        model = resnet18(weights=None)
        log("[ContrastiveInit] Using SimCLR weights.")
        strict_load = False
        # CIFAR adjustments
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        missing, unexpected = model.load_state_dict(simclr_w, strict=strict_load)
        log(f"[ContrastiveInit] Missing: {missing}")
        log(f"[ContrastiveInit] Extra: {unexpected}")

    else:
        # Simple fallback
        log("[ContrastiveInit] Using ImageNet weights.")
        model = resnet18(weights="IMAGENET1K_V1")
        # CIFAR adjustments
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()

    # Replace FC
    model.fc = nn.Linear(512, num_classes)

    # Optional freeze (used in warm-start schedule)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
        log("[ContrastiveInit] Backbone frozen.")

    return model
