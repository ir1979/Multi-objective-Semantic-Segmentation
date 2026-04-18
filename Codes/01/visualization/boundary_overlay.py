"""Boundary overlay visualization for model comparisons."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from visualization.style import save_figure


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask.squeeze() > 0.5).astype(np.float32)
    dil = np.pad(mask, 1, mode="edge")
    center = dil[1:-1, 1:-1]
    neighbors = (
        dil[:-2, 1:-1]
        + dil[2:, 1:-1]
        + dil[1:-1, :-2]
        + dil[1:-1, 2:]
    )
    return ((center > 0) & (neighbors < 4)).astype(np.float32)


def generate_boundary_overlay(
    models_dict: Dict[str, tf.keras.Model],
    dataset: tf.data.Dataset,
    num_samples: int = 6,
    save_path: str = "figures/boundary_overlay",
) -> None:
    """Generate boundary overlay comparison figure."""
    samples = []
    for image, mask in dataset.unbatch().take(num_samples):
        samples.append((image.numpy(), mask.numpy()))
    if not samples:
        return

    model_names = list(models_dict.keys())
    fig, axes = plt.subplots(len(samples), len(model_names), figsize=(3.5 * len(model_names), 3 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(model_names) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (image, gt_mask) in enumerate(samples):
        gt_boundary = _boundary(gt_mask)
        x = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
        for col_idx, model_name in enumerate(model_names):
            pred = models_dict[model_name](x, training=False)
            if isinstance(pred, list):
                pred = pred[-1]
            pred_boundary = _boundary(pred.numpy()[0])
            overlay = image.copy()
            overlay[gt_boundary > 0.5] = [0.0, 1.0, 0.0]
            overlay[pred_boundary > 0.5] = [1.0, 0.0, 0.0]
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_title(model_name)
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
