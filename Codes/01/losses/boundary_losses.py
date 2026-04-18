"""Boundary-aware losses implemented with differentiable TensorFlow ops."""

from __future__ import annotations

import tensorflow as tf


def _morphological_gradient(x: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
    dilation = tf.nn.max_pool2d(x, ksize=kernel_size, strides=1, padding="SAME")
    erosion = -tf.nn.max_pool2d(-x, ksize=kernel_size, strides=1, padding="SAME")
    return tf.nn.relu(dilation - erosion)


class ApproxHausdorffLoss(tf.keras.losses.Loss):
    """Differentiable approximate Hausdorff boundary loss."""

    def __init__(self, percentile: float = 95.0, name: str = "approx_hausdorff_loss") -> None:
        super().__init__(name=name)
        self.percentile = percentile

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_boundary = _morphological_gradient(y_true, kernel_size=3)
        pred_boundary = _morphological_gradient(y_pred, kernel_size=3)

        # Approximate distance transform with repeated box blur to stay differentiable.
        dt = 1.0 - true_boundary
        for _ in range(4):
            dt = tf.nn.avg_pool2d(dt, ksize=3, strides=1, padding="SAME")

        weighted_distance = pred_boundary * dt
        flattened = tf.reshape(weighted_distance, [-1])
        k = tf.cast(
            tf.maximum(1.0, tf.round(tf.cast(tf.size(flattened), tf.float32) * (self.percentile / 100.0))),
            tf.int32,
        )
        values, _ = tf.math.top_k(flattened, k=k, sorted=False)
        return tf.reduce_mean(values)
