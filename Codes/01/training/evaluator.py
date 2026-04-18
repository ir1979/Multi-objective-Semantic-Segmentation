"""Evaluation helpers for trained segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from training.metrics import (
    boundary_f1,
    boundary_iou,
    compactness_score,
    dice_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)


@dataclass
class Evaluator:
    """Evaluate segmentation models on tf.data datasets."""

    threshold: float = 0.5

    def _predict(self, model: tf.keras.Model, x_batch: tf.Tensor) -> tf.Tensor:
        y_pred = model(x_batch, training=False)
        if isinstance(y_pred, list):
            y_pred = y_pred[-1]
        return tf.cast(y_pred, tf.float32)

    def evaluate(self, model: tf.keras.Model, dataset: tf.data.Dataset, threshold: float = 0.5) -> Dict[str, float]:
        """Compute aggregate metrics over a dataset."""
        self.threshold = threshold
        metric_values: Dict[str, List[float]] = {
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
            "pixel_accuracy": [],
            "boundary_iou": [],
            "boundary_f1": [],
            "compactness": [],
        }

        for x_batch, y_batch in dataset:
            y_pred = self._predict(model, x_batch)
            y_pred = tf.cast(y_pred > threshold, tf.float32)
            metric_values["iou"].append(float(iou_score(y_batch, y_pred).numpy()))
            metric_values["dice"].append(float(dice_score(y_batch, y_pred).numpy()))
            metric_values["precision"].append(float(precision_score(y_batch, y_pred).numpy()))
            metric_values["recall"].append(float(recall_score(y_batch, y_pred).numpy()))
            metric_values["pixel_accuracy"].append(float(pixel_accuracy(y_batch, y_pred).numpy()))
            metric_values["boundary_iou"].append(float(boundary_iou(y_batch, y_pred).numpy()))
            metric_values["boundary_f1"].append(float(boundary_f1(y_batch, y_pred).numpy()))
            metric_values["compactness"].append(float(compactness_score(y_batch, y_pred).numpy()))

        return {name: float(np.mean(values)) if values else 0.0 for name, values in metric_values.items()}

    def evaluate_per_image(self, model: tf.keras.Model, dataset: tf.data.Dataset) -> pd.DataFrame:
        """Return per-image metric table."""
        rows: List[Dict[str, float]] = []
        image_index = 0
        for x_batch, y_batch in dataset:
            y_pred = self._predict(model, x_batch)
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
            batch_size = int(x_batch.shape[0])
            for idx in range(batch_size):
                gt = y_batch[idx : idx + 1]
                pred = y_pred[idx : idx + 1]
                rows.append(
                    {
                        "image_index": image_index,
                        "iou": float(iou_score(gt, pred).numpy()),
                        "dice": float(dice_score(gt, pred).numpy()),
                        "precision": float(precision_score(gt, pred).numpy()),
                        "recall": float(recall_score(gt, pred).numpy()),
                        "pixel_accuracy": float(pixel_accuracy(gt, pred).numpy()),
                        "boundary_iou": float(boundary_iou(gt, pred).numpy()),
                        "boundary_f1": float(boundary_f1(gt, pred).numpy()),
                    }
                )
                image_index += 1
        return pd.DataFrame(rows)

    def compute_confusion_matrix(self, model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
        """Compute pixel-level confusion matrix."""
        tp = tn = fp = fn = 0
        for x_batch, y_batch in dataset:
            y_pred = self._predict(model, x_batch)
            y_pred = tf.cast(y_pred > self.threshold, tf.int32).numpy()
            y_true = tf.cast(y_batch > 0.5, tf.int32).numpy()
            tp += int(np.sum((y_true == 1) & (y_pred == 1)))
            tn += int(np.sum((y_true == 0) & (y_pred == 0)))
            fp += int(np.sum((y_true == 0) & (y_pred == 1)))
            fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)
