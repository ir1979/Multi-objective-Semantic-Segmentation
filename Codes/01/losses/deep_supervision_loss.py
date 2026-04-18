"""Deep supervision loss utility for UNet++ outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass
class DeepSupervisionLoss:
    """Compute weighted loss across multiple decoder outputs."""

    base_loss: tf.keras.losses.Loss
    weights: Sequence[float]

    def __call__(self, y_true: tf.Tensor, outputs: Sequence[tf.Tensor]) -> tf.Tensor:
        if not outputs:
            raise ValueError("Deep supervision outputs cannot be empty.")
        if len(outputs) != len(self.weights):
            raise ValueError("Number of outputs must match number of weights.")

        weighted_losses = []
        for prediction, weight in zip(outputs, self.weights):
            resized = tf.image.resize(prediction, tf.shape(y_true)[1:3], method="bilinear")
            weighted_losses.append(tf.cast(weight, tf.float32) * self.base_loss(y_true, resized))
        return tf.math.divide_no_nan(
            tf.add_n(weighted_losses),
            tf.reduce_sum(tf.constant(self.weights, dtype=tf.float32)),
        )
