"""UNet++ (Nested U-Net) implementation with optional deep supervision."""

from __future__ import annotations

from typing import Dict, List

import tensorflow as tf

from models.blocks import ConvBlock


class UNetPlusPlus(tf.keras.Model):
    """UNet++ segmentation model.

    Parameters
    ----------
    encoder_filters:
        Channel sizes for encoder levels from shallow to deep.
    deep_supervision:
        Whether to return intermediate decoder outputs.
    """

    def __init__(
        self,
        encoder_filters: List[int] | None = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        deep_supervision: bool = False,
        name: str = "UNetPlusPlus",
    ) -> None:
        super().__init__(name=name)
        self.filters = encoder_filters or [64, 128, 256, 512, 1024]
        if len(self.filters) != 5:
            raise ValueError("UNetPlusPlus expects exactly 5 filter values.")
        self.deep_supervision = deep_supervision
        self.output_activation = output_activation

        # Encoder nodes X_{i,0}
        self.encoder_blocks = [
            ConvBlock(
                filters=f,
                dropout_rate=dropout_rate if idx > 1 else 0.0,
                use_batch_norm=batch_norm,
                activation=activation,
                name=f"x_{idx}_0",
            )
            for idx, f in enumerate(self.filters)
        ]
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

        # Nested decoder conv blocks X_{i,j}, j > 0
        self.decoder_blocks: Dict[tuple[int, int], ConvBlock] = {}
        for j in range(1, 5):
            for i in range(0, 5 - j):
                block = ConvBlock(
                    filters=self.filters[i],
                    dropout_rate=dropout_rate if i < 2 else 0.0,
                    use_batch_norm=batch_norm,
                    activation=activation,
                    name=f"x_{i}_{j}",
                )
                self.decoder_blocks[(i, j)] = block
                setattr(self, f"decoder_x_{i}_{j}", block)

        self.output_heads = [
            tf.keras.layers.Conv2D(1, kernel_size=1, activation=output_activation, name=f"head_{idx}")
            for idx in range(1, 5)
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor | List[tf.Tensor]:
        """Forward pass for UNet++ graph."""
        x_nodes: Dict[tuple[int, int], tf.Tensor] = {}

        # Build encoder chain X_{i,0}
        current = inputs
        for i in range(5):
            current = self.encoder_blocks[i](current, training=training)
            x_nodes[(i, 0)] = current
            if i < 4:
                current = self.pool(current)

        # Build dense nested decoder nodes
        for j in range(1, 5):
            for i in range(0, 5 - j):
                concat_tensors = [x_nodes[(i, k)] for k in range(j)]
                upsampled = self.up(x_nodes[(i + 1, j - 1)])
                upsampled = tf.image.resize(
                    upsampled,
                    tf.shape(x_nodes[(i, 0)])[1:3],
                    method="bilinear",
                )
                concat_tensors.append(upsampled)
                merged = tf.concat(concat_tensors, axis=-1)
                x_nodes[(i, j)] = self.decoder_blocks[(i, j)](merged, training=training)

        if self.deep_supervision:
            return [head(x_nodes[(0, idx)]) for idx, head in enumerate(self.output_heads, start=1)]
        return self.output_heads[-1](x_nodes[(0, 4)])
