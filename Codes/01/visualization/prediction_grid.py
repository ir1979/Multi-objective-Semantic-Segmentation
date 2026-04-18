"""Prediction grid visualization."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from visualization.style import save_figure


def _overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).copy()
    gt = gt.squeeze() > 0.5
    pred = pred.squeeze() > 0.5
    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)
    rgb[tp] = np.array([0.0, 1.0, 0.0]) * 0.7 + rgb[tp] * 0.3
    rgb[fp] = np.array([1.0, 0.0, 0.0]) * 0.7 + rgb[fp] * 0.3
    rgb[fn] = np.array([0.0, 0.0, 1.0]) * 0.7 + rgb[fn] * 0.3
    return rgb


def generate_prediction_grid(
    models_dict: Dict[str, tf.keras.Model],
    dataset: tf.data.Dataset,
    num_samples: int = 8,
    save_path: str = "figures/sample_predictions",
) -> None:
    """Generate qualitative prediction grid figure."""
    samples = []
    for images, masks in dataset.unbatch().take(num_samples):
        samples.append((images.numpy(), masks.numpy()))
    if not samples:
        return

    model_names = list(models_dict.keys())
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(len(samples), n_cols, figsize=(14, 2.5 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (image, mask) in enumerate(samples):
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title("RGB")
        axes[row_idx, 1].imshow(mask.squeeze(), cmap="gray")
        axes[row_idx, 1].set_title("GT")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].axis("off")

        x = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
        for col_idx, model_name in enumerate(model_names, start=2):
            pred = models_dict[model_name](x, training=False)
            if isinstance(pred, list):
                pred = pred[-1]
            pred_np = (pred.numpy()[0] > 0.5).astype(np.float32)
            axes[row_idx, col_idx].imshow(_overlay(image, mask, pred_np))
            axes[row_idx, col_idx].set_title(model_name)
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
