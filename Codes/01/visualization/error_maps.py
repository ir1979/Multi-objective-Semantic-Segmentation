"""Error map visualizations for segmentation results."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from visualization.style import save_figure


def _error_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt.squeeze() > 0.5
    pred = pred.squeeze() > 0.5
    overlay = np.ones((*gt.shape, 3), dtype=np.float32)
    overlay[gt & pred] = [0.0, 1.0, 0.0]  # TP
    overlay[(~gt) & pred] = [1.0, 0.0, 0.0]  # FP
    overlay[gt & (~pred)] = [0.0, 0.0, 1.0]  # FN
    blended = 0.5 * np.clip(rgb, 0.0, 1.0) + 0.5 * overlay
    return blended


def generate_error_maps(
    models_dict: Dict[str, tf.keras.Model],
    dataset: tf.data.Dataset,
    num_samples: int = 4,
    save_path: str = "figures/error_maps",
) -> None:
    """Generate per-sample and aggregate error maps."""
    samples = []
    for image, mask in dataset.unbatch().take(num_samples):
        samples.append((image.numpy(), mask.numpy()))
    if not samples:
        return

    model_names = list(models_dict.keys())
    fig, axes = plt.subplots(
        len(samples),
        2 + len(model_names),
        figsize=(3.2 * (2 + len(model_names)), 2.8 * len(samples)),
    )
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (image, gt_mask) in enumerate(samples):
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title("RGB")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(gt_mask.squeeze(), cmap="gray")
        axes[row_idx, 1].set_title("GT")
        axes[row_idx, 1].axis("off")
        x = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
        for col_idx, model_name in enumerate(model_names, start=2):
            pred = models_dict[model_name](x, training=False)
            if isinstance(pred, list):
                pred = pred[-1]
            error_map = _error_overlay(image, gt_mask, pred.numpy()[0])
            axes[row_idx, col_idx].imshow(error_map)
            axes[row_idx, col_idx].set_title(model_name)
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
