"""Pixel-level PyTorch losses."""

from __future__ import annotations

import torch
from torch import nn


class BCELoss(nn.Module):
    """Binary cross entropy loss."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss = nn.BCELoss()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
        return self.loss(y_pred, y_true)


class IoULoss(nn.Module):
    """Differentiable IoU loss."""

    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        inter = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f) + torch.sum(y_pred_f) - inter
        iou = (inter + self.epsilon) / (union + self.epsilon)
        return 1.0 - iou


class DiceLoss(nn.Module):
    """Soft Dice loss."""

    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        inter = torch.sum(y_true_f * y_pred_f)
        dice = (2.0 * inter + self.epsilon) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + self.epsilon
        )
        return 1.0 - dice


class BCEIoULoss(nn.Module):
    """Weighted combination of BCE and IoU."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.bce = BCELoss()
        self.iou = IoULoss()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.bce(y_true, y_pred) + (1.0 - self.alpha) * self.iou(y_true, y_pred)


class FocalLoss(nn.Module):
    """Binary focal loss."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t)
        return torch.mean(loss)
