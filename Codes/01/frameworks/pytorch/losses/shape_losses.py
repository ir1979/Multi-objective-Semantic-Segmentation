"""Shape-related PyTorch regularization losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _morph_close(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    dilation = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    erosion = -F.max_pool2d(-dilation, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.clamp(erosion, 0.0, 1.0)


class ConvexityLoss(nn.Module):
    """Penalize non-convex segmentation regions."""

    def __init__(self, kernel_size: int = 7, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del y_true
        hull = _morph_close(y_pred, kernel_size=self.kernel_size)
        pred_area = torch.sum(y_pred, dim=(1, 2, 3))
        hull_area = torch.sum(hull, dim=(1, 2, 3))
        convexity = (pred_area + self.epsilon) / (hull_area + self.epsilon)
        return 1.0 - torch.mean(convexity)


class RegularityLoss(nn.Module):
    """Penalize jagged boundaries with Laplacian response."""

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del y_true
        boundary = torch.relu(
            F.max_pool2d(y_pred, kernel_size=3, stride=1, padding=1)
            - (-F.max_pool2d(-y_pred, kernel_size=3, stride=1, padding=1))
        )
        lap = F.conv2d(y_pred, self.kernel, padding=1)
        return torch.mean(torch.abs(lap) * boundary)
