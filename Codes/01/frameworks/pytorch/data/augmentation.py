"""Deterministic augmentation for paired image-mask samples."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for random training augmentations."""

    horizontal_flip: bool = True
    vertical_flip: bool = True
    random_rotation: bool = True
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    seed: int = 42


class PairAugmenter:
    """Apply deterministic augmentations to image and mask pairs."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.horizontal_flip and self.rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if self.config.vertical_flip and self.rng.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if self.config.random_rotation:
            k = self.rng.randint(0, 3)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        if self.config.brightness_range > 0.0:
            delta = (self.rng.random() * 2.0 - 1.0) * self.config.brightness_range
            image = np.clip(image + delta, 0.0, 1.0)
        if self.config.contrast_range > 0.0:
            factor = 1.0 + (self.rng.random() * 2.0 - 1.0) * self.config.contrast_range
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * factor + mean, 0.0, 1.0)
        return image.astype(np.float32), mask.astype(np.float32)
