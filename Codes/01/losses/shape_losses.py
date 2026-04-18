"""Shape-oriented losses for segmentation regularization."""

from __future__ import annotations

import tensorflow as tf


def _morph_close(x: tf.Tensor, kernel_size: int) -> tf.Tensor:
    dilation = tf.nn.max_pool2d(x, ksize=kernel_size, strides=1, padding="SAME")
    erosion = -tf.nn.max_pool2d(-dilation, ksize=kernel_size, strides=1, padding="SAME")
    return tf.clip_by_value(erosion, 0.0, 1.0)


class ConvexityLoss(tf.keras.losses.Loss):
    """Penalize non-convex predicted regions via morphological hull approximation."""

    def __init__(self, kernel_size: int = 7, epsilon: float = 1e-7, name: str = "convexity_loss") -> None:
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        del y_true
        y_pred = tf.cast(y_pred, tf.float32)
        hull = _morph_close(y_pred, kernel_size=self.kernel_size)
        pred_area = tf.reduce_sum(y_pred, axis=[1, 2, 3])
        hull_area = tf.reduce_sum(hull, axis=[1, 2, 3])
        convexity = (pred_area + self.epsilon) / (hull_area + self.epsilon)
        return 1.0 - tf.reduce_mean(convexity)


class RegularityLoss(tf.keras.losses.Loss):
    """Boundary smoothness loss using Laplacian response on boundary pixels."""

    def __init__(self, name: str = "regularity_loss") -> None:
        super().__init__(name=name)
        laplacian = tf.constant([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        self.kernel = tf.reshape(laplacian, (3, 3, 1, 1))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        del y_true
        y_pred = tf.cast(y_pred, tf.float32)
        dilation = tf.nn.max_pool2d(y_pred, ksize=3, strides=1, padding="SAME")
        erosion = -tf.nn.max_pool2d(-y_pred, ksize=3, strides=1, padding="SAME")
        boundary = tf.nn.relu(dilation - erosion)
        response = tf.nn.conv2d(y_pred, self.kernel, strides=1, padding="SAME")
        return tf.reduce_mean(tf.abs(response) * boundary)
