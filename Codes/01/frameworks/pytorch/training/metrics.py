"""PyTorch metric utilities for binary segmentation."""

from __future__ import annotations

from typing import Dict

import torch


def _binarize(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (pred > threshold).float()


def compute_batch_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute common segmentation metrics for one batch."""
    y_pred_bin = _binarize(y_pred, threshold=threshold)
    tp = torch.sum((y_true == 1) & (y_pred_bin == 1)).float()
    tn = torch.sum((y_true == 0) & (y_pred_bin == 0)).float()
    fp = torch.sum((y_true == 0) & (y_pred_bin == 1)).float()
    fn = torch.sum((y_true == 1) & (y_pred_bin == 0)).float()

    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    pixel_acc = (tp + tn + 1e-7) / (tp + tn + fp + fn + 1e-7)

    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(dice.item()),
        "pixel_accuracy": float(pixel_acc.item()),
    }
