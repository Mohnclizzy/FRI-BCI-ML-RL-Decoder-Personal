"""Compact EEGNet-style network with true depthwise/separable convolutions.

Architecture reference: Lawhern et al. (2018), doi:10.1088/1741-2552/aace8c.
This PyTorch implementation is a project variant, not an exact reproduction.
"""
import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels=9, samples=320, f1=8, depth=2, f2=16, dropout=0.25):
        super().__init__()
        if channels < 2 or samples < 32:
            raise ValueError('Insufficient channels/samples.')
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 15), padding=(0, 7), groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (samples // 4 // 8), 2)

    def forward(self, x):
        return self.classifier(self.features(x[:, None]).flatten(1))

    @torch.no_grad()
    def constrain_weights(self):
        """Bound spatial-filter norms and output-row norms after each update."""
        for weight, limit in ((self.features[2].weight, 1.0), (self.classifier.weight, 0.25)):
            norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            factor = (limit / norms).clamp(max=1)
            weight.mul_(factor.reshape((-1,) + (1,) * (weight.ndim - 1)))
