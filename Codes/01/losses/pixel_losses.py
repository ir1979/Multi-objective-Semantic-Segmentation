"""Pixel-level loss implementations for binary segmentation."""

from __future__ import annotations

import tensorflow as tf


class BCELoss(tf.keras.losses.Loss):
    """Binary cross-entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0, name: str = "bce_loss") -> None:
        super().__init__(name=name)
        self.label_smoothing = label_smoothing
        self._bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        return tf.reduce_mean(self._bce(y_true, y_pred))


class IoULoss(tf.keras.losses.Loss):
    """Differentiable IoU loss, defined as ``1 - IoU``."""

    def __init__(self, epsilon: float = 1e-7, name: str = "iou_loss") -> None:
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - iou


class DiceLoss(tf.keras.losses.Loss):
    """Soft Dice loss."""

    def __init__(self, epsilon: float = 1e-7, name: str = "dice_loss") -> None:
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2.0 * intersection + self.epsilon) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.epsilon
        )
        return 1.0 - score


class FocalLoss(tf.keras.losses.Loss):
    """Binary focal loss for class-imbalanced segmentation."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        epsilon: float = 1e-7,
        name: str = "focal_loss",
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        loss = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

