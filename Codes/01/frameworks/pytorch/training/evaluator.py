"""Model evaluation helpers for PyTorch segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from frameworks.pytorch.training.metrics import compute_batch_metrics


@dataclass
class Evaluator:
    """Evaluate PyTorch models on DataLoader objects."""

    device: torch.device
    threshold: float = 0.5

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        """Compute aggregate metrics for a dataset."""
        model.eval()
        metrics: Dict[str, List[float]] = {
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "pixel_accuracy": [],
        }
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                preds = model(images)
                if isinstance(preds, list):
                    preds = preds[-1]
                batch = compute_batch_metrics(masks, preds, threshold=threshold)
                for key, value in batch.items():
                    metrics[key].append(value)
        return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}

    def evaluate_per_image(self, model: torch.nn.Module, dataloader: DataLoader) -> pd.DataFrame:
        """Return per-image metrics table."""
        rows: List[Dict[str, float]] = []
        image_idx = 0
        model.eval()
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                preds = model(images)
                if isinstance(preds, list):
                    preds = preds[-1]
                for idx in range(images.shape[0]):
                    metrics = compute_batch_metrics(
                        masks[idx : idx + 1],
                        preds[idx : idx + 1],
                        threshold=self.threshold,
                    )
                    rows.append({"image_index": image_idx, **metrics})
                    image_idx += 1
        return pd.DataFrame(rows)

    def compute_confusion_matrix(self, model: torch.nn.Module, dataloader: DataLoader) -> np.ndarray:
        """Compute pixel-level confusion matrix."""
        tp = tn = fp = fn = 0
        model.eval()
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                preds = model(images)
                if isinstance(preds, list):
                    preds = preds[-1]
                pred_bin = (preds > self.threshold).int()
                true_bin = (masks > 0.5).int()
                tp += int(((true_bin == 1) & (pred_bin == 1)).sum().item())
                tn += int(((true_bin == 0) & (pred_bin == 0)).sum().item())
                fp += int(((true_bin == 0) & (pred_bin == 1)).sum().item())
                fn += int(((true_bin == 1) & (pred_bin == 0)).sum().item())
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

