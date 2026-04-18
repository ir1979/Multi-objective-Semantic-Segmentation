"""Weighted-sum multi-objective strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import tensorflow as tf


@dataclass
class WeightedSumStrategy:
    """Combine objective losses using static or scheduled weights."""

    base_weights: Mapping[str, float]
    schedule_type: str = "static"
    ramp_epochs: int = 20
    milestones: Mapping[int, Mapping[str, float]] | None = None

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Return objective weights for the provided epoch."""
        weights = dict(self.base_weights)
        if self.schedule_type == "static":
            return weights

        if self.schedule_type == "linear_ramp":
            factor = min(1.0, max(0.0, float(epoch) / float(max(1, self.ramp_epochs))))
            return {name: float(weight) * factor for name, weight in weights.items()}

        if self.schedule_type == "step":
            if not self.milestones:
                return weights
            updated = dict(weights)
            for step_epoch in sorted(self.milestones):
                if epoch >= step_epoch:
                    updated.update({k: float(v) for k, v in self.milestones[step_epoch].items()})
            return updated

        if self.schedule_type == "cosine":
            factor = 0.5 * (1.0 + np.cos(np.pi * min(1.0, float(epoch) / float(max(1, self.ramp_epochs)))))
            return {name: float(weight) * factor for name, weight in weights.items()}

        raise ValueError(f"Unknown schedule_type '{self.schedule_type}'.")

    def combine(self, losses: Mapping[str, tf.Tensor], epoch: int) -> tf.Tensor:
        """Return weighted scalar loss."""
        weights = self.get_weights(epoch)
        weighted_terms = []
        weight_values = []
        for name, loss_value in losses.items():
            weight = tf.constant(weights.get(name, 0.0), dtype=tf.float32)
            weighted_terms.append(weight * tf.cast(loss_value, tf.float32))
            weight_values.append(weight)
        return tf.math.divide_no_nan(
            tf.add_n(weighted_terms),
            tf.add_n(weight_values) if weight_values else tf.constant(1.0, dtype=tf.float32),
        )
