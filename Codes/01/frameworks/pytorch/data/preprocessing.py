"""Preprocessing utilities for PyTorch data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: Path, image_size: int) -> np.ndarray:
    """Load RGB image into float32 [0, 1] array."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def read_mask(path: Path, image_size: int) -> np.ndarray:
    """Load mask and binarize to {0,1} float32 with channel axis."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((image_size, image_size), Image.NEAREST)
        mask = np.asarray(gray, dtype=np.float32) / 255.0
    mask = (mask >= 0.5).astype(np.float32)
    return mask[..., np.newaxis]


def compute_density(mask: np.ndarray) -> float:
    """Compute foreground/building ratio."""
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0.5))
