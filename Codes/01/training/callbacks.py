"""Custom callbacks used by the training workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf


@dataclass
class MGDAAlphaLogger:
    """Track MGDA alpha values per epoch."""

    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, epoch: int, alphas: Dict[str, float]) -> None:
        self.history.append({"epoch": float(epoch), **{k: float(v) for k, v in alphas.items()}})


@dataclass
class ValidationImageLogger:
    """Persist validation predictions every N epochs."""

    output_dir: Path
    interval: int = 5
    sample_batch: Optional[tuple[tf.Tensor, tf.Tensor]] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, epoch: int, model: tf.keras.Model) -> None:
        if self.sample_batch is None or epoch % self.interval != 0:
            return
        images, _ = self.sample_batch
        preds = model(images, training=False)
        if isinstance(preds, list):
            preds = preds[-1]
        preds = tf.cast(preds > 0.5, tf.float32).numpy()
        np.save(self.output_dir / f"val_preds_epoch_{epoch:03d}.npy", preds)


@dataclass
class GradientNormLogger:
    """Track gradient norms per epoch."""

    norms: List[Dict[str, float]] = field(default_factory=list)

    def log(self, epoch: int, gradients: List[tf.Tensor | None]) -> None:
        valid = [float(tf.norm(g).numpy()) for g in gradients if g is not None]
        self.norms.append(
            {
                "epoch": float(epoch),
                "mean_norm": float(np.mean(valid)) if valid else 0.0,
                "max_norm": float(np.max(valid)) if valid else 0.0,
            }
        )


@dataclass
class TrainingTimeLogger:
    """Record per-epoch and cumulative timing."""

    start_time: float = field(default_factory=time.time)
    epoch_times: List[float] = field(default_factory=list)

    def log_epoch(self, epoch_duration: float) -> Dict[str, float]:
        self.epoch_times.append(float(epoch_duration))
        return {
            "epoch_time_seconds": float(epoch_duration),
            "total_time_seconds": float(time.time() - self.start_time),
        }

