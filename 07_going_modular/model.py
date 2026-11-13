"""Model definitions."""

import torch.nn as nn
from torchvision import models


def tiny_cnn(num_classes: int = 10) -> nn.Module:
    """Small CNN for 32x32 RGB images."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def resnet18_transfer(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    """ResNet-18 with replaced head. freeze_backbone=True for feature extraction."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone
