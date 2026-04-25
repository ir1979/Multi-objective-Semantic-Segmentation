"""TensorFlow augmentation pipeline for image-mask pairs.

This module keeps compatibility wrappers consumed by the legacy dataset class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf


@dataclass
class AugmentationConfig:
    """Configuration subset required by the tf.data augmentation pipeline."""

    horizontal_flip: bool = True
    vertical_flip: bool = True
    random_rotation: bool = True
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    seed: int = 42


def _maybe_flip(
    image: tf.Tensor,
    mask: tf.Tensor,
    prob: tf.Tensor,
    horizontal: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    def flipped() -> tuple[tf.Tensor, tf.Tensor]:
        if horizontal:
            return tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
        return tf.image.flip_up_down(image), tf.image.flip_up_down(mask)

    return tf.cond(prob < 0.5, flipped, lambda: (image, mask))


def build_augmentation_pipeline(
    config: AugmentationConfig | dict,
) -> Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Return a map-compatible augmentation callable.

    Spatial transforms are applied identically to image and mask.
    """
    if isinstance(config, dict):
        config = AugmentationConfig(
            horizontal_flip=bool(config.get("horizontal_flip", True)),
            vertical_flip=bool(config.get("vertical_flip", True)),
            random_rotation=bool(config.get("random_rotation", True)),
            brightness_range=float(config.get("brightness_range", 0.1)),
            contrast_range=float(config.get("contrast_range", 0.1)),
            seed=int(config.get("seed", 42)),
        )

    def _make_seed(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Create a stateless RNG seed of shape ``[2]`` for TF 2.10+."""
        image_token = tf.cast(tf.reduce_sum(tf.cast(image * 255.0, tf.int64)), tf.int32)
        mask_token = tf.cast(tf.reduce_sum(tf.cast(mask * 255.0, tf.int64)), tf.int32)
        mixed = tf.math.floormod(image_token ^ mask_token, tf.constant(2**31 - 1, dtype=tf.int32))
        return tf.stack([tf.constant(config.seed, dtype=tf.int32), mixed], axis=0)

    def augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        random_values = tf.random.stateless_uniform(
            shape=[5],
            seed=_make_seed(image, mask),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )

        if config.horizontal_flip:
            image, mask = _maybe_flip(image, mask, random_values[0], horizontal=True)

        if config.vertical_flip:
            image, mask = _maybe_flip(image, mask, random_values[1], horizontal=False)

        if config.random_rotation:
            k = tf.cast(tf.floor(random_values[2] * 4.0), tf.int32)
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)

        brightness_delta = (random_values[3] * 2.0 - 1.0) * tf.constant(
            config.brightness_range,
            dtype=tf.float32,
        )
        contrast_factor = 1.0 + (random_values[4] * 2.0 - 1.0) * tf.constant(
            config.contrast_range,
            dtype=tf.float32,
        )

        image = tf.image.adjust_brightness(image, brightness_delta)
        image = tf.image.adjust_contrast(image, contrast_factor)
        image = tf.clip_by_value(image, 0.0, 1.0)
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        return image, mask

    return augment


class AugmentationPipeline:
    """Backward-compatible augmentation wrapper for NumPy arrays."""

    def __init__(
        self,
        augmentation_config: Optional[list[str] | str] = None,
        input_shape: tuple[int, int] = (256, 256),
        seed: int = 42,
    ) -> None:
        del input_shape  # Kept for compatibility with previous signature.
        if isinstance(augmentation_config, str):
            requested = {item.strip().lower() for item in augmentation_config.split(",")}
        elif isinstance(augmentation_config, (list, tuple)):
            requested = {str(item).strip().lower() for item in augmentation_config}
        else:
            requested = {"flip", "rotate"}
        self._augment = build_augmentation_pipeline(
            AugmentationConfig(
                horizontal_flip=("flip" in requested or "horizontal_flip" in requested),
                vertical_flip=("vertical_flip" in requested or "flip" in requested),
                random_rotation=("rotate" in requested or "rotation" in requested),
                brightness_range=0.1 if "brightness" in requested else 0.0,
                contrast_range=0.1 if "contrast" in requested else 0.0,
                seed=seed,
            )
        )

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **_: Any,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if mask is None:
            return image.astype(np.float32), None
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
        aug_image, aug_mask = self._augment(image_tensor, mask_tensor)
        return aug_image.numpy().astype(np.float32), aug_mask.numpy().astype(np.float32)


def build_augmentation_from_config(config: dict[str, Any]) -> Optional[AugmentationPipeline]:
    """Backward-compatible config factory used by legacy dataset code."""
    augmentations = config.get("augmentations")
    if augmentations in (None, [], "none", False):
        return None
    seed = int(config.get("seed", 42))
    tile_size = int(config.get("tile_size", 256))
    return AugmentationPipeline(augmentations, input_shape=(tile_size, tile_size), seed=seed)


def create_augmentation_pipeline(
    augmentation_config: Optional[list[str] | str] = None,
    input_shape: tuple[int, int] = (256, 256),
    seed: Optional[int] = None,
) -> AugmentationPipeline:
    """Backward-compatible alias used by older modules."""
    return AugmentationPipeline(
        augmentation_config=augmentation_config,
        input_shape=input_shape,
        seed=42 if seed is None else int(seed),
    )


def get_available_augmentations() -> list[str]:
    """Return supported augmentation labels."""
    return [
        "flip",
        "horizontal_flip",
        "vertical_flip",
        "rotate",
        "random_rotation",
        "brightness",
        "contrast",
    ]
