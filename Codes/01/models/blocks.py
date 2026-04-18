"""Reusable neural network blocks for segmentation models."""

from __future__ import annotations

import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    """Double convolution block with optional normalization and dropout.

    Parameters
    ----------
    filters:
        Number of output filters.
    kernel_size:
        Convolution kernel size.
    dropout_rate:
        Dropout probability applied after the second convolution.
    use_batch_norm:
        Whether to apply BatchNorm after each convolution.
    activation:
        Keras activation name.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
        activation: str = "relu",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.bn1 = tf.keras.layers.BatchNormalization() if use_batch_norm else None
        self.act1 = tf.keras.layers.Activation(activation)
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.bn2 = tf.keras.layers.BatchNormalization() if use_batch_norm else None
        self.act2 = tf.keras.layers.Activation(activation)
        self.dropout = (
            tf.keras.layers.Dropout(dropout_rate) if dropout_rate and dropout_rate > 0.0 else None
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply the block to an input tensor."""
        x = self.conv1(inputs)
        if self.bn1 is not None:
            x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x, training=training)
        x = self.act2(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x


class AttentionGate(tf.keras.layers.Layer):
    """Attention gate for decoder skip filtering.

    The gate takes a skip tensor and a gating tensor from a coarser decoder stage.
    It learns coefficients that suppress irrelevant skip activations.
    """

    def __init__(self, inter_channels: int, name: str | None = None) -> None:
        super().__init__(name=name)
        self.theta = tf.keras.layers.Conv2D(inter_channels, kernel_size=1, padding="same")
        self.phi = tf.keras.layers.Conv2D(inter_channels, kernel_size=1, padding="same")
        self.psi = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same")
        self.relu = tf.keras.layers.ReLU()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, skip: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
        """Return attention-weighted skip features."""
        gate_up = tf.image.resize(gate, tf.shape(skip)[1:3], method="bilinear")
        theta_x = self.theta(skip)
        phi_g = self.phi(gate_up)
        logits = self.relu(theta_x + phi_g)
        coefficients = self.sigmoid(self.psi(logits))
        return skip * coefficients
