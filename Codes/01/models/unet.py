"""U-Net model implementation for building segmentation."""

from __future__ import annotations

from typing import List

import tensorflow as tf

from models.blocks import ConvBlock


class UNet(tf.keras.Model):
    """Standard U-Net architecture."""

    def __init__(
        self,
        encoder_filters: List[int] | None = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        name: str = "UNet",
    ) -> None:
        super().__init__(name=name)
        self.encoder_filters = encoder_filters or [64, 128, 256, 512, 1024]
        if len(self.encoder_filters) != 5:
            raise ValueError("UNet expects exactly 5 filter values.")

        self.down_blocks = [
            ConvBlock(
                filters=f,
                dropout_rate=dropout_rate if i > 1 else 0.0,
                use_batch_norm=batch_norm,
                activation=activation,
                name=f"encoder_block_{i}",
            )
            for i, f in enumerate(self.encoder_filters[:-1])
        ]
        self.pools = [tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) for _ in range(4)]
        self.bottleneck = ConvBlock(
            filters=self.encoder_filters[-1],
            dropout_rate=dropout_rate,
            use_batch_norm=batch_norm,
            activation=activation,
            name="bottleneck",
        )

        decoder_filters = list(reversed(self.encoder_filters[:-1]))
        self.up_samples = [tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear") for _ in range(4)]
        self.decoder_blocks = [
            ConvBlock(
                filters=f,
                dropout_rate=dropout_rate if i < 2 else 0.0,
                use_batch_norm=batch_norm,
                activation=activation,
                name=f"decoder_block_{i}",
            )
            for i, f in enumerate(decoder_filters)
        ]

        self.output_head = tf.keras.layers.Conv2D(1, kernel_size=1, activation=output_activation)
        self._encoder_features: List[tf.Tensor] = []

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = inputs
        self._encoder_features = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x, training=training)
            self._encoder_features.append(x)
            x = pool(x)

        x = self.bottleneck(x, training=training)

        for upsample, block, skip in zip(
            self.up_samples,
            self.decoder_blocks,
            reversed(self._encoder_features),
        ):
            x = upsample(x)
            x = tf.concat([x, skip], axis=-1)
            x = block(x, training=training)

        return self.output_head(x)

    def get_encoder_features(self) -> List[tf.Tensor]:
        """Return the intermediate encoder feature maps from last forward pass."""
        return list(self._encoder_features)
