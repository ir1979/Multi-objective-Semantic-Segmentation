"""Preprocessing utilities for satellite building segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


def read_rgb_image(path: Path, image_size: int) -> np.ndarray:
    """Read and normalize an RGB image.

    Parameters
    ----------
    path:
        Path to RGB image.
    image_size:
        Target square size.

    Returns
    -------
    np.ndarray
        Float32 image in range [0, 1] with shape (image_size, image_size, 3).
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array


def read_mask_image(path: Path, image_size: int, threshold: float = 0.5) -> np.ndarray:
    """Read and binarize a building mask.

    Parameters
    ----------
    path:
        Path to mask image.
    image_size:
        Target square size.
    threshold:
        Threshold on normalized mask values.

    Returns
    -------
    np.ndarray
        Float32 binary mask with shape (image_size, image_size, 1).
    """
    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size), Image.NEAREST)
        values = np.asarray(gray, dtype=np.float32) / 255.0
    binary = (values >= threshold).astype(np.float32)
    return binary[..., np.newaxis]


def validate_binary_mask(mask: np.ndarray) -> bool:
    """Return True when mask values are binary."""
    unique_values = np.unique(mask.astype(np.float32))
    return set(unique_values.tolist()).issubset({0.0, 1.0})


def compute_building_density(mask: np.ndarray) -> float:
    """Compute ratio of building pixels in a binary mask."""
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0.5))


def safe_load_pair(
    rgb_path: Path,
    mask_path: Path,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load paired image and mask with unified error behavior."""
    try:
        image = read_rgb_image(rgb_path, image_size)
        mask = read_mask_image(mask_path, image_size)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not load sample ({rgb_path}, {mask_path}): {exc}") from exc
    return image, mask
