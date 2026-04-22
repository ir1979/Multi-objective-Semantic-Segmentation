"""PyTorch U-Net implementation."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from frameworks.pytorch.models.blocks import ConvBlock


class UNet(nn.Module):
    """Standard U-Net for binary segmentation."""

    def __init__(
        self,
        encoder_filters: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        filters = encoder_filters or [64, 128, 256, 512, 1024]
        if len(filters) != 5:
            raise ValueError("UNet expects exactly 5 encoder filter values.")

        self.enc1 = ConvBlock(3, filters[0], dropout_rate=0.0, use_batch_norm=batch_norm)
        self.enc2 = ConvBlock(filters[0], filters[1], dropout_rate=0.0, use_batch_norm=batch_norm)
        self.enc3 = ConvBlock(filters[1], filters[2], dropout_rate=dropout_rate, use_batch_norm=batch_norm)
        self.enc4 = ConvBlock(filters[2], filters[3], dropout_rate=dropout_rate, use_batch_norm=batch_norm)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(filters[3], filters[4], dropout_rate=dropout_rate, use_batch_norm=batch_norm)

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(filters[4], filters[3], dropout_rate=dropout_rate, use_batch_norm=batch_norm)
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(filters[3], filters[2], dropout_rate=dropout_rate, use_batch_norm=batch_norm)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(filters[2], filters[1], dropout_rate=0.0, use_batch_norm=batch_norm)
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(filters[1], filters[0], dropout_rate=0.0, use_batch_norm=batch_norm)

        self.out_conv = nn.Conv2d(filters[0], 1, kernel_size=1)
        self._encoder_features: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        self._encoder_features = [e1, e2, e3, e4]
        return torch.sigmoid(self.out_conv(d1))

    def get_encoder_features(self) -> List[torch.Tensor]:
        """Return encoder feature maps from the most recent forward pass."""
        return list(self._encoder_features)
