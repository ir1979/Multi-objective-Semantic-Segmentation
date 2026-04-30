"""Central loss manager for single/weighted/MGDA strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

import tensorflow as tf

from losses.boundary_losses import ApproxHausdorffLoss
from losses.deep_supervision_loss import DeepSupervisionLoss
from losses.pixel_losses import BCELoss, DiceLoss, FocalLoss, IoULoss
from losses.shape_losses import ConvexityLoss, RegularityLoss





@dataclass
class LossManager:
    """Compute segmentation losses under different composition strategies."""

    config: Mapping[str, object]

    def __post_init__(self) -> None:
        self.strategy = str(self.config.get("loss_strategy", "single")).lower()
        
        pixel_type = str(self.config.get("loss_pixel_type", "bce"))
        if pixel_type == "bce":
            self.pixel_loss = BCELoss()
        elif pixel_type == "iou":
            self.pixel_loss = IoULoss()
        elif pixel_type == "dice":
            self.pixel_loss = DiceLoss()
        elif pixel_type == "focal":
            self.pixel_loss = FocalLoss()
        else:
            raise ValueError(f"Unknown pixel loss type '{pixel_type}'.")

        self.boundary_enabled = bool(self.config.get("loss_boundary_enabled", False))
        self.shape_enabled = bool(self.config.get("loss_shape_enabled", False))
        self.boundary_loss = ApproxHausdorffLoss()
        self.shape_loss = ConvexityLoss()
        self.shape_reg_loss = RegularityLoss()
        self.weights = {
            "pixel": float(self.config.get("loss_pixel_weight", 1.0)),
            "boundary": float(self.config.get("loss_boundary_weight", 0.0)),
            "shape": float(self.config.get("loss_shape_weight", 0.0)),
        }
        self.deep_supervision_enabled = bool(self.config.get("model_deep_supervision", False))
        self.deep_supervision_weights = self.config.get("model_deep_supervision_weights", [0.5, 0.3, 0.2, 0.1])

    def _apply_loss(
        self,
        loss_fn: tf.keras.losses.Loss | callable,
        y_true: tf.Tensor,
        y_pred: tf.Tensor | Sequence[tf.Tensor],
    ) -> tf.Tensor:
        if isinstance(y_pred, (list, tuple)):
            if self.deep_supervision_enabled:
                weights = self.deep_supervision_weights[: len(y_pred)]
                if len(weights) < len(y_pred):
                    weights = weights + [weights[-1]] * (len(y_pred) - len(weights))
            else:
                weights = [1.0 / float(len(y_pred))] * len(y_pred)
            return DeepSupervisionLoss(loss_fn, weights)(y_true, y_pred)
        return loss_fn(y_true, y_pred)

    def compute_losses(self, y_true: tf.Tensor, y_pred: tf.Tensor | Sequence[tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Return individual loss values."""
        losses: Dict[str, tf.Tensor] = {
            "pixel": self._apply_loss(self.pixel_loss, y_true, y_pred),
        }
        if self.boundary_enabled:
            losses["boundary"] = self._apply_loss(self.boundary_loss, y_true, y_pred)
        if self.shape_enabled:
            shape_primary = self._apply_loss(self.shape_loss, y_true, y_pred)
            shape_regularizer = self._apply_loss(self.shape_reg_loss, y_true, y_pred)
            losses["shape"] = 0.5 * shape_primary + 0.5 * shape_regularizer
        return losses

    def compute_weighted_total(self, losses_dict: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Return weighted scalar loss sum."""
        total = tf.constant(0.0, dtype=tf.float32)
        weight_sum = tf.constant(0.0, dtype=tf.float32)
        for name, loss_value in losses_dict.items():
            weight = tf.constant(self.weights.get(name, 0.0), dtype=tf.float32)
            total += weight * tf.cast(loss_value, tf.float32)
            weight_sum += weight
        return tf.math.divide_no_nan(total, weight_sum)

    def get_loss_names(self) -> List[str]:
        """Return active loss component names."""
        names = ["pixel"]
        if self.boundary_enabled:
            names.append("boundary")
        if self.shape_enabled:
            names.append("shape")
        return names

    def get_loss_functions(self) -> Dict[str, Callable[[tf.Tensor, tf.Tensor | Sequence[tf.Tensor]], tf.Tensor]]:
        """Return active per-objective loss callables with deep-supervision support."""

        losses: Dict[str, Callable[[tf.Tensor, tf.Tensor | Sequence[tf.Tensor]], tf.Tensor]] = {
            "pixel": lambda y_true, y_pred: self._apply_loss(self.pixel_loss, y_true, y_pred),
        }
        if self.boundary_enabled:
            losses["boundary"] = lambda y_true, y_pred: self._apply_loss(self.boundary_loss, y_true, y_pred)
        if self.shape_enabled:
            losses["shape"] = lambda y_true, y_pred: 0.5 * self._apply_loss(
                self.shape_loss, y_true, y_pred
            ) + 0.5 * self._apply_loss(self.shape_reg_loss, y_true, y_pred)
        return losses


# Legacy helper expected by existing training code.
def build_losses(config: Mapping[str, object]) -> tuple[list, list, list]:
    """Return callable loss list, weights, and names (legacy API)."""
    pixel_type = str(config.get("pixel_loss", "bce")).lower()
    manager = LossManager(
        {"loss": config}
        if "strategy" in config
        else {
            "loss": {
                "strategy": "weighted",
                "pixel": {
                    "type": pixel_type,
                    "weight": config.get("pixel_weight", 1.0),
                },
                "boundary": {
                    "enabled": config.get("boundary_loss", "none") != "none",
                    "weight": config.get("boundary_weight", 0.3),
                },
                "shape": {
                    "enabled": config.get("shape_loss", "none") != "none",
                    "weight": config.get("shape_weight", 0.1),
                },
            }
        },
    )
    names = manager.get_loss_names()
    losses = []
    for name in names:
        if name == "pixel":
            losses.append(manager.pixel_loss)
        elif name == "boundary":
            losses.append(manager.boundary_loss)
        elif name == "shape":
            losses.append(lambda y_true, y_pred, _m=manager: 0.5 * _m.shape_loss(y_true, y_pred) + 0.5 * _m.shape_reg_loss(y_true, y_pred))
    weights = [manager.weights[name] for name in names]
    return losses, weights, names


def build_single_loss(config: Mapping[str, object]):
    """Legacy adapter returning a combined scalar loss function."""
    losses, weights, _ = build_losses(config)

    def combined(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        weighted = [w * loss(y_true, y_pred) for loss, w in zip(losses, weights)]
        denominator = tf.reduce_sum(tf.constant(weights, dtype=tf.float32))
        return tf.math.divide_no_nan(tf.add_n(weighted), denominator)

    return combined
