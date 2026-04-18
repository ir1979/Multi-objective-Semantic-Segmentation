"""TensorBoard logging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import tensorflow as tf


class TensorBoardLogger:
    """Log scalars, images, and histograms to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(str(self.log_dir))

    def log_scalars(self, epoch: int, scalars_dict: Mapping[str, float]) -> None:
        with self.writer.as_default():
            for name, value in scalars_dict.items():
                tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()

    def log_images(self, epoch: int, images_dict: Mapping[str, tf.Tensor]) -> None:
        with self.writer.as_default():
            for name, value in images_dict.items():
                tf.summary.image(name, value, step=epoch, max_outputs=min(8, int(value.shape[0])))
            self.writer.flush()

    def log_mgda_alphas(self, epoch: int, alphas_dict: Mapping[str, float]) -> None:
        self.log_scalars(epoch, {f"mgda/{name}": value for name, value in alphas_dict.items()})

    def log_gradient_norms(self, epoch: int, norms_dict: Mapping[str, float]) -> None:
        self.log_scalars(epoch, {f"gradients/{name}": value for name, value in norms_dict.items()})

    def close(self) -> None:
        self.writer.close()
