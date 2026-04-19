"""PyTorch UNet++ implementation with optional deep supervision."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from frameworks.pytorch.models.blocks import ConvBlock


class UNetPlusPlus(nn.Module):
    """Nested U-Net (UNet++) architecture."""

    def __init__(
        self,
        encoder_filters: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.filters = encoder_filters or [64, 128, 256, 512, 1024]
        if len(self.filters) != 5:
            raise ValueError("UNetPlusPlus expects exactly 5 encoder filters.")
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2)

        self.encoder_blocks = nn.ModuleList(
            [
                ConvBlock(3, self.filters[0], dropout_rate=0.0, use_batch_norm=batch_norm),
                ConvBlock(self.filters[0], self.filters[1], dropout_rate=0.0, use_batch_norm=batch_norm),
                ConvBlock(self.filters[1], self.filters[2], dropout_rate=dropout_rate, use_batch_norm=batch_norm),
                ConvBlock(self.filters[2], self.filters[3], dropout_rate=dropout_rate, use_batch_norm=batch_norm),
                ConvBlock(self.filters[3], self.filters[4], dropout_rate=dropout_rate, use_batch_norm=batch_norm),
            ]
        )

        self.decoder_blocks = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                in_channels = self.filters[i] * j + self.filters[i + 1]
                self.decoder_blocks[f"x_{i}_{j}"] = ConvBlock(
                    in_channels,
                    self.filters[i],
                    dropout_rate=dropout_rate if i < 2 else 0.0,
                    use_batch_norm=batch_norm,
                )

        self.heads = nn.ModuleList([nn.Conv2d(self.filters[0], 1, kernel_size=1) for _ in range(4)])

    def forward(self, x: torch.Tensor):
        nodes: Dict[Tuple[int, int], torch.Tensor] = {}

        current = x
        for i in range(5):
            current = self.encoder_blocks[i](current)
            nodes[(i, 0)] = current
            if i < 4:
                current = self.pool(current)

        for j in range(1, 5):
            for i in range(0, 5 - j):
                features = [nodes[(i, k)] for k in range(j)]
                up = F.interpolate(
                    nodes[(i + 1, j - 1)],
                    size=nodes[(i, 0)].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features.append(up)
                nodes[(i, j)] = self.decoder_blocks[f"x_{i}_{j}"](torch.cat(features, dim=1))

        outputs = [torch.sigmoid(self.heads[idx](nodes[(0, idx + 1)])) for idx in range(4)]
        if self.deep_supervision:
            return outputs
        return outputs[-1]
