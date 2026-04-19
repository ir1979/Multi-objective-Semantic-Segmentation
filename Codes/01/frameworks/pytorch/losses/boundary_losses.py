"""Boundary-aware PyTorch losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _morph_gradient(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    dilation = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    erosion = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.relu(dilation - erosion)


class ApproxHausdorffLoss(nn.Module):
    """Differentiable approximate Hausdorff boundary loss."""

    def __init__(self, percentile: float = 95.0) -> None:
        super().__init__()
        self.percentile = percentile

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        true_boundary = _morph_gradient(y_true)
        pred_boundary = _morph_gradient(y_pred)
        dt = 1.0 - true_boundary
        for _ in range(4):
            dt = F.avg_pool2d(dt, kernel_size=3, stride=1, padding=1)
        weighted = (pred_boundary * dt).reshape(-1)
        k = max(1, int(round(weighted.numel() * (self.percentile / 100.0))))
        values, _ = torch.topk(weighted, k=k)
        return torch.mean(values)

