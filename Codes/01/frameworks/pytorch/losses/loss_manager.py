"""Loss manager for PyTorch training modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import torch
from torch import nn

from frameworks.pytorch.losses.boundary_losses import ApproxHausdorffLoss
from frameworks.pytorch.losses.pixel_losses import BCELoss, BCEIoULoss, DiceLoss, FocalLoss, IoULoss
from frameworks.pytorch.losses.shape_losses import ConvexityLoss, RegularityLoss


def _pixel_loss(pixel_cfg: Mapping[str, object]) -> nn.Module:
    loss_type = str(pixel_cfg.get("type", "bce_iou")).lower()
    if loss_type == "bce":
        return BCELoss()
    if loss_type == "iou":
        return IoULoss()
    if loss_type == "dice":
        return DiceLoss()
    if loss_type == "bce_iou":
        return BCEIoULoss(alpha=float(pixel_cfg.get("bce_iou_alpha", 0.5)))
    if loss_type == "focal":
        return FocalLoss()
    raise ValueError(f"Unknown pixel loss type: {loss_type}")


@dataclass
class LossManager:
    """Central loss orchestrator for PyTorch training."""

    config: Mapping[str, object]

    def __post_init__(self) -> None:
        loss_cfg = dict(self.config.get("loss", {}))
        self.strategy = str(loss_cfg.get("strategy", "single")).lower()
        self.pixel_cfg = dict(loss_cfg.get("pixel", {}))
        self.pixel_loss = _pixel_loss(self.pixel_cfg)
        self.boundary_enabled = bool(loss_cfg.get("boundary", {}).get("enabled", False))
        self.shape_enabled = bool(loss_cfg.get("shape", {}).get("enabled", False))
        self.boundary_loss = ApproxHausdorffLoss()
        self.shape_loss = ConvexityLoss()
        self.regularity_loss = RegularityLoss()
        self.weights = {
            "pixel": float(self.pixel_cfg.get("weight", 1.0)),
            "boundary": float(loss_cfg.get("boundary", {}).get("weight", 0.0)),
            "shape": float(loss_cfg.get("shape", {}).get("weight", 0.0)),
        }

    def compute_losses(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {"pixel": self.pixel_loss(y_true, y_pred)}
        if self.boundary_enabled:
            losses["boundary"] = self.boundary_loss(y_true, y_pred)
        if self.shape_enabled:
            losses["shape"] = 0.5 * self.shape_loss(y_true, y_pred) + 0.5 * self.regularity_loss(
                y_true, y_pred
            )
        return losses

    def compute_weighted_total(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weighted = []
        weights = []
        for name, value in losses.items():
            w = self.weights.get(name, 0.0)
            if w <= 0.0:
                continue
            weighted.append(value * w)
            weights.append(w)
        if not weighted:
            if losses:
                first = next(iter(losses.values()))
                return torch.tensor(0.0, device=first.device, dtype=first.dtype)
            return torch.tensor(0.0)
        denom = max(1e-8, float(sum(weights)))
        return torch.stack(weighted).sum() / denom

    def get_loss_names(self) -> List[str]:
        names = ["pixel"]
        if self.boundary_enabled:
            names.append("boundary")
        if self.shape_enabled:
            names.append("shape")
        return names
