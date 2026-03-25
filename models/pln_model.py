from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .backbone import BackBone


class PLNModel(nn.Module):
    """
    PLN model skeleton.

    Currently implemented:
    - BackBone: ResNet18 -> layer4 features
    """

    def __init__(
        self,
        backbone_pretrained: bool = True,
        backbone_trainable: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()

        self.backbone = BackBone(
            pretrained=backbone_pretrained,
            requires_grad=backbone_trainable,
            freeze_bn=freeze_bn,
        )

        # Shared adapter: 512 -> 256 -> 256 -> 256
        self.adapter = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Four parallel branches (left-top, right-top, left-bottom, right-bottom)
        self.branch_keys: List[str] = ["left_top", "right_top", "left_bottom", "right_bottom"]
        self.branches = nn.ModuleList([_BranchModule() for _ in range(4)])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: Tensor of shape (B, 3, H, W)
        returns:
            A dict of 4 feature maps after branch dilation:
            {
              "left_top":     (B, 204, 14, 14),
              "right_top":    (B, 204, 14, 14),
              "left_bottom":  (B, 204, 14, 14),
              "right_bottom": (B, 204, 14, 14),
            }
        """
        feats = self.backbone(images)
        shared = self.adapter(feats)  # (B,256,14,14)

        outs: Dict[str, torch.Tensor] = {}
        for key, branch in zip(self.branch_keys, self.branches):
            outs[key] = branch(shared)  # (B,204,14,14)
        return outs


class _BranchModule(nn.Module):
    """
    One branch:
    - two convs: 256 -> 256 -> 204 (keeps spatial size 14x14)
    - dilation module: a stack of 3x3 dilated convs, output channels kept at 204
      with rate = [2,2,4,8,16,1,1]
    """

    def __init__(self):
        super().__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 204, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(204),
            nn.ReLU(inplace=True),
        )

        rates = [2, 2, 4, 8, 16, 1, 1]
        layers: List[nn.Module] = []
        for idx, r in enumerate(rates):
            layers.append(
                nn.Conv2d(
                    204,
                    204,
                    kernel_size=3,
                    stride=1,
                    padding=r,  # keep H/W
                    dilation=r,
                    bias=False,
                )
            )
            # Do NOT apply BN/ReLU after the last dilation conv.
            if idx != len(rates) - 1:
                layers.append(nn.BatchNorm2d(204))
                layers.append(nn.ReLU(inplace=True))
        self.dilation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.dilation(x)
        return x

