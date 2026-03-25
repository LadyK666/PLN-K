from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision import models


def _resnet18(pretrained: bool) -> nn.Module:
    """
    Create torchvision resnet18 with backward/forward compatible weights API.
    """
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except Exception:
        # Older torchvision
        return models.resnet18(pretrained=pretrained)


class BackBone(nn.Module):
    """
    Backbone that returns ResNet18's `layer4` feature map.

    Input:  (B, 3, H, W)
    Output: (B, 512, H/32, W/32)  (for standard ResNet18 strides)
    """

    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()

        self.resnet = _resnet18(pretrained=pretrained)

        # ResNet18 layer4 output channels.
        self.out_channels: int = 512

        if not requires_grad:
            for p in self.resnet.parameters():
                p.requires_grad_(False)

        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self) -> None:
        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,3,H,W), got shape={tuple(x.shape)}")
        if x.size(1) != 3:
            raise ValueError(f"Expected input with 3 channels, got C={x.size(1)}")

        # Mirror torchvision ResNet forward up to layer4.
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # <-- BackBone output

        return x

