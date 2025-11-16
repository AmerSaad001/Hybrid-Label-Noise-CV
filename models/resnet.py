"""
ResNet-18 backbone + small classifier head.

I kept this file pretty simple because I didn't want to fight with
Torchvision's internal layers. The idea here is:
- use a slightly modified ResNet-18 for CIFAR (smaller images)
- optionally load ImageNet weights (if available)
- expose 'extract_features()' so I can plug in the structural/hybrid stuff later
"""

import warnings
import torch.nn as nn
from torchvision.models import resnet18

# Torchvision renamed weight enums in newer versions, so I'm doing a small try/except
# because my laptop has an older version and I got annoyed by the errors.
try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


# ----------------------------------------------------
# Figure out which weights to use (or none)
# ----------------------------------------------------
def _resolve_weights(pretrained: bool):
    # If I don't want pretrained, just bail out
    if not pretrained:
        return None

    # If the weight enum doesn't exist, I'm stuck – warn and continue
    if ResNet18_Weights is None:
        warnings.warn(
            "Tried to use pretrained ResNet18 but torchvision version "
            "doesn't expose ResNet18_Weights. Using random init instead."
        )
        return None

    # Otherwise just grab the standard ImageNet weights
    return ResNet18_Weights.IMAGENET1K_V1


# ----------------------------------------------------
# Build the ResNet18 backbone
# ----------------------------------------------------
def build_resnet18_backbone(pretrained: bool = False) -> nn.Module:
    # Figure out what weights to load (if any)
    weights = _resolve_weights(pretrained)

    # Build normal ResNet18 from torchvision
    backbone = resnet18(weights=weights)

    # CIFAR images are 32x32 so the original ResNet stem (7x7 conv + maxpool)
    # is a bit overkill. I swap them for a simpler conv and remove the pool.
    # Pretty common trick in CIFAR papers.
    backbone.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    backbone.maxpool = nn.Identity()

    # The fc layer isn't needed here because I want features only.
    backbone.fc = nn.Identity()

    return backbone


# ----------------------------------------------------
# Simple classifier wrapper
# ----------------------------------------------------
class ResNet18Classifier(nn.Module):
    """
    Tiny wrapper around the ResNet-18 backbone.
    I expose feature extraction explicitly because the structural/hybrid
    logic needs those embeddings to build the KNN graph.
    """

    def __init__(self, num_classes=10, pretrained=False, freeze_backbone=False):
        super().__init__()

        # Build the feature extractor
        self.backbone = build_resnet18_backbone(pretrained)

        # Just a basic linear head to turn 512-dim features into logits
        self.head = nn.Linear(512, num_classes)

        # Optionally freeze backbone params (used in contrastive warm-start mode)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # The backbone already returns the 512-d vector
        features = self.backbone(x)

        # Feed into the classifier head
        logits = self.head(features)
        return logits

    def extract_features(self, x):
        # Expose this because the hybrid/structural part needs embeddings
        return self.backbone(x)

    def get_backbone_state(self):
        return self.backbone.state_dict()

    def load_backbone_state(self, state_dict, strict=False):
        # I made strict=False the default because SimCLR checkpoints
        # sometimes miss keys inside torchvision models.
        return self.backbone.load_state_dict(state_dict, strict=strict)


__all__ = [
    "ResNet18Classifier",
    "build_resnet18_backbone",
]