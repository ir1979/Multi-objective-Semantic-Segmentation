"""Reusable PyTorch building blocks."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Double conv block with optional batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm)]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for skip-connection modulation."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate_up = nn.functional.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.relu(self.theta(skip) + self.phi(gate_up))
        coeff = self.sigmoid(self.psi(attn))
        return skip * coeff
