"""Training loop implementation for single, weighted, and MGDA strategies."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import tensorflow as tf

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from losses.loss_manager import LossManager, build_single_loss
from logging_utils.csv_logger import CSVLogger
from logging_utils.logger import DualLogger
from logging_utils.tensorboard_logger import TensorBoardLogger
from models.model_factory import build_model
from optimization.mgda import MGDASolver, MGDATrainStep
from optimization.schedulers import CosineAnnealingLR, PlateauScheduler
from training.callbacks import MGDAAlphaLogger, TrainingTimeLogger, ValidationImageLogger
from training.checkpoint_manager import CheckpointManager
from training.early_stopping import MultiMetricEarlyStopping
from training.metrics import (
    boundary_f1,
    boundary_iou,
    compactness_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
    topological_correctness,
)


def _metric_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - inter
    return tf.math.divide_no_nan(inter + 1e-7, union + 1e-7)


def _metric_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return tf.math.divide_no_nan(2.0 * inter + 1e-7, denom + 1e-7)


def _metric_precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return tf.math.divide_no_nan(tp + 1e-7, tp + fp + 1e-7)


def _metric_recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return tf.math.divide_no_nan(tp + 1e-7, tp + fn + 1e-7)


def _metric_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


METRIC_FNS = {
    "iou": _metric_iou,
    "dice": _metric_dice,
    "precision": _metric_precision,
    "recall": _metric_recall,
    "f1": _metric_dice,
    "pixel_accuracy": _metric_accuracy,
}


@dataclass
class TrainingResult:
    """Container for train/validation history and checkpoint metadata."""

    history: Dict[str, list] = field(default_factory=dict)
    best_epoch: int = 0
    best_metric: float = 0.0
    stopped_early: bool = False
    total_time_seconds: float = 0.0
    mgda_alpha_history: list = field(default_factory=list)
    resumed_from_checkpoint: bool = False
    start_epoch: int = 0


class Trainer:
    """Train a segmentation model using resolved configuration and tf.data datasets."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: Mapping[str, object],
        results_dir: Path,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.model = model
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = checkpoint_manager
        self.loss_manager = LossManager(config)

        train_cfg = dict(config.get("training", {}))
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            clipnorm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        )
        self.strategy = str(config.get("loss", {}).get("strategy", "single")).lower()
        self.mgda_solver = MGDASolver(
            max_iterations=int(config.get("mgda", {}).get("max_iterations", 50)),
            tolerance=float(config.get("mgda", {}).get("tolerance", 1e-6)),
            normalize_gradients=bool(config.get("mgda", {}).get("normalize_gradients", True)),
        )
        self.mgda_stepper = MGDATrainStep(
            model=self.model,
            optimizer=self.optimizer,
            solver=self.mgda_solver,
            loss_names=self.loss_manager.get_loss_names(),
        )

        epochs = int(train_cfg.get("epochs", 100))
        scheduler_cfg = dict(train_cfg.get("lr_scheduler", {}))
        self.lr_scheduler = CosineAnnealingLR(
            base_lr=float(train_cfg.get("learning_rate", 1e-4)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
            total_epochs=epochs,
            warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
        )
        self.plateau_scheduler = PlateauScheduler(
            initial_lr=float(train_cfg.get("learning_rate", 1e-4)),
            mode=str(train_cfg.get("early_stopping", {}).get("mode", "max")),
        )
        early_cfg = dict(train_cfg.get("early_stopping", {}))
        self.early_stopping = MultiMetricEarlyStopping(
            monitor=str(early_cfg.get("monitor", "val_iou")),
            patience=int(early_cfg.get("patience", 15)),
            mode=str(early_cfg.get("mode", "max")),
            min_delta=float(early_cfg.get("min_delta", 1e-4)),
        )

        log_cfg = dict(config.get("logging", {}))
        self.dual_logger = DualLogger(
            self.results_dir / "training.log",
            console_level=str(log_cfg.get("console_level", "INFO")),
            file_level=str(log_cfg.get("file_level", "DEBUG")),
        )
        self.tensorboard_logger = TensorBoardLogger(self.results_dir / "tensorboard")
        self.csv_logger = CSVLogger(self.results_dir / "logs.csv")
        self.alpha_logger = MGDAAlphaLogger()
        self.time_logger = TrainingTimeLogger()
        self.validation_image_logger = ValidationImageLogger(
            output_dir=self.results_dir / "figures",
            interval=int(log_cfg.get("validation_image_interval", 5)),
        )

    def _load_history(self) -> Dict[str, list]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "lr": [],
        }
        csv_history = self.csv_logger.load()
        if csv_history.empty:
            return history
        column_map = {
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "val_iou": "val_iou",
            "val_dice": "val_dice",
            "lr": "learning_rate",
        }
        for key, column in column_map.items():
            if column in csv_history.columns:
                history[key] = [float(v) for v in csv_history[column].tolist()]
        return history

    def _restore_training_state(self) -> tuple[int, Dict[str, list], bool]:
        checkpoint_cfg = dict(self.config.get("checkpointing", {}))
        auto_resume = bool(checkpoint_cfg.get("auto_resume", False))
        if not auto_resume:
            return 0, self._load_history(), False

        model_path, state = self.checkpoint_manager.load_latest()
        history = self._load_history()
        if model_path is None or state is None:
            return 0, history, False

        self._ensure_model_built()
        self.model.load_weights(model_path)
        restored_optimizer = self.checkpoint_manager.restore_optimizer_state(self.optimizer, state, self.model)
        extra_state = dict(state.get("extra_state", {}))
        if extra_state.get("plateau_scheduler"):
            self.plateau_scheduler.load_state_dict(dict(extra_state["plateau_scheduler"]))
        if extra_state.get("early_stopping"):
            self.early_stopping.load_state_dict(dict(extra_state["early_stopping"]))
        if self.strategy == "mgda" and state.get("mgda_alpha_history"):
            self.mgda_solver.alpha_history = list(state["mgda_alpha_history"])

        start_epoch = int(state.get("epoch", 0))
        self.dual_logger.info(
            f"Resuming training from epoch {start_epoch + 1} "
            f"(optimizer_restored={restored_optimizer}, checkpoint={model_path})"
        )
        return start_epoch, history, True

    def _ensure_model_built(self) -> None:
        """Build subclassed models before loading weights from checkpoint."""
        if self.model.built:
            return
        image_size = int(self.config.get("data", {}).get("image_size", 256))
        dummy = tf.zeros((1, image_size, image_size, 3), dtype=tf.float32)
        _ = self.model(dummy, training=False)

    @tf.function
    def _weighted_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            if isinstance(predictions, list):
                predictions = predictions[-1]
            loss_dict = self.loss_manager.compute_losses(y_batch, predictions)
            total_loss = self.loss_manager.compute_weighted_total(loss_dict)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": total_loss, **loss_dict}

    @tf.function
    def _eval_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        predictions = self.model(x_batch, training=False)
        if isinstance(predictions, list):
            predictions = predictions[-1]
        loss_dict = self.loss_manager.compute_losses(y_batch, predictions)
        total_loss = self.loss_manager.compute_weighted_total(loss_dict)
        metrics = {name: fn(y_batch, predictions) for name, fn in METRIC_FNS.items()}
        return {"loss": total_loss, **loss_dict, **metrics}

    @staticmethod
    def _aggregate(values: Dict[str, list]) -> Dict[str, float]:
        return {key: float(np.mean(series)) for key, series in values.items() if series}

    def _set_lr(self, epoch: int) -> float:
        scheduler_type = str(self.config.get("training", {}).get("lr_scheduler", {}).get("type", "cosine")).lower()
        if scheduler_type == "plateau":
            return float(self.optimizer.learning_rate.numpy())
        lr = self.lr_scheduler.get_lr(epoch)
        self.optimizer.learning_rate.assign(lr)
        return float(lr)

    def _update_plateau(self, metric_value: float) -> None:
        scheduler_type = str(self.config.get("training", {}).get("lr_scheduler", {}).get("type", "cosine")).lower()
        if scheduler_type == "plateau":
            self.optimizer.learning_rate.assign(self.plateau_scheduler.step(metric_value))

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        start_epoch: int = 0,
    ) -> TrainingResult:
        """Train the configured model and return history."""
        train_cfg = dict(self.config.get("training", {}))
        epochs = int(train_cfg.get("epochs", 100))
        resumed_from_checkpoint = False
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "lr": [],
        }
        start_time = time.time()
        best_metric = -float("inf")
        best_epoch = start_epoch
        mgda_history = []
        should_stop = False
        current_epoch = start_epoch

        try:
            if start_epoch == 0:
                start_epoch, history, resumed_from_checkpoint = self._restore_training_state()
            else:
                history = self._load_history()
            if not any(history.values()):
                history = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_iou": [],
                    "val_dice": [],
                    "lr": [],
                }
            best_metric = max(history["val_iou"]) if history["val_iou"] else -float("inf")
            best_epoch = int(np.argmax(history["val_iou"])) + 1 if history["val_iou"] else start_epoch
            mgda_history = list(self.mgda_solver.get_alpha_history()) if self.strategy == "mgda" else []

            self.validation_image_logger.sample_batch = next(iter(val_dataset.take(1)))
        except Exception:
            self.validation_image_logger.sample_batch = None

        try:
            for epoch in range(start_epoch, epochs):
                current_epoch = epoch
                epoch_start = time.time()
                current_lr = self._set_lr(epoch)
                train_values: Dict[str, list] = {"loss": [], "pixel": [], "boundary": [], "shape": []}

                for x_batch, y_batch in train_dataset:
                    if self.strategy == "mgda":
                        loss_functions = {"pixel": self.loss_manager.pixel_loss}
                        if self.loss_manager.boundary_enabled:
                            loss_functions["boundary"] = self.loss_manager.boundary_loss
                        if self.loss_manager.shape_enabled:
                            loss_functions["shape"] = self.loss_manager.shape_loss
                        mgda_metrics = self.mgda_stepper.step(x_batch, y_batch, loss_functions)
                        batch_loss = np.mean([v for k, v in mgda_metrics.items() if not k.startswith("alpha_")])
                        train_values["loss"].append(float(batch_loss))
                        train_values["pixel"].append(float(mgda_metrics.get("pixel", 0.0)))
                        train_values["boundary"].append(float(mgda_metrics.get("boundary", 0.0)))
                        train_values["shape"].append(float(mgda_metrics.get("shape", 0.0)))
                        alpha_payload = {k: v for k, v in mgda_metrics.items() if k.startswith("alpha_")}
                        if alpha_payload:
                            mgda_history.append(alpha_payload)
                    else:
                        batch_metrics = self._weighted_train_step(x_batch, y_batch)
                        train_values["loss"].append(float(batch_metrics["loss"]))
                        train_values["pixel"].append(float(batch_metrics.get("pixel", 0.0)))
                        train_values["boundary"].append(float(batch_metrics.get("boundary", 0.0)))
                        train_values["shape"].append(float(batch_metrics.get("shape", 0.0)))

                val_values: Dict[str, list] = {
                    "loss": [],
                    "iou": [],
                    "dice": [],
                    "precision": [],
                    "recall": [],
                    "pixel_accuracy": [],
                }
                for x_batch, y_batch in val_dataset:
                    metrics = self._eval_step(x_batch, y_batch)
                    for key in val_values:
                        val_values[key].append(float(metrics[key]))

                train_agg = self._aggregate(train_values)
                val_agg = self._aggregate(val_values)
                history["train_loss"].append(train_agg.get("loss", 0.0))
                history["val_loss"].append(val_agg.get("loss", 0.0))
                history["val_iou"].append(val_agg.get("iou", 0.0))
                history["val_dice"].append(val_agg.get("dice", 0.0))
                history["lr"].append(current_lr)

                epoch_duration = time.time() - epoch_start
                time_metrics = self.time_logger.log_epoch(epoch_duration)

                csv_metrics = {
                    "train_loss": train_agg.get("loss", 0.0),
                    "train_pixel_loss": train_agg.get("pixel", 0.0),
                    "train_boundary_loss": train_agg.get("boundary", 0.0),
                    "train_shape_loss": train_agg.get("shape", 0.0),
                    "val_loss": val_agg.get("loss", 0.0),
                    "val_pixel_loss": 0.0,
                    "val_boundary_loss": 0.0,
                    "val_shape_loss": 0.0,
                    "val_iou": val_agg.get("iou", 0.0),
                    "val_dice": val_agg.get("dice", 0.0),
                    "val_precision": val_agg.get("precision", 0.0),
                    "val_recall": val_agg.get("recall", 0.0),
                    "val_pixel_accuracy": val_agg.get("pixel_accuracy", 0.0),
                    "learning_rate": current_lr,
                    **time_metrics,
                }
                if mgda_history:
                    csv_metrics.update(mgda_history[-1])
                self.csv_logger.log_epoch(epoch=epoch + 1, metrics_dict=csv_metrics)

                scalar_payload = {
                    "train/loss": train_agg.get("loss", 0.0),
                    "val/loss": val_agg.get("loss", 0.0),
                    "val/iou": val_agg.get("iou", 0.0),
                    "val/dice": val_agg.get("dice", 0.0),
                    "lr": current_lr,
                }
                self.tensorboard_logger.log_scalars(epoch + 1, scalar_payload)
                self.dual_logger.log_epoch_summary(
                    epoch + 1,
                    train_metrics=train_agg,
                    val_metrics=val_agg,
                    lr=current_lr,
                    mgda_alphas=mgda_history[-1] if mgda_history else None,
                )
                if mgda_history:
                    self.alpha_logger.log(epoch + 1, mgda_history[-1])
                    self.tensorboard_logger.log_mgda_alphas(epoch + 1, mgda_history[-1])
                self.validation_image_logger.log(epoch + 1, self.model)

                self._update_plateau(val_agg.get("iou", 0.0))
                should_stop = self.early_stopping.step(
                    {
                        "val_iou": val_agg.get("iou", 0.0),
                        "val_loss": val_agg.get("loss", 0.0),
                    }
                )

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics={
                        "val_iou": val_agg.get("iou", 0.0),
                        "val_boundary": 1.0 - val_agg.get("dice", 0.0),
                    },
                    mgda_solver=self.mgda_solver if self.strategy == "mgda" else None,
                    extra_state={
                        "early_stopping": self.early_stopping.state_dict(),
                        "plateau_scheduler": self.plateau_scheduler.state_dict(),
                        "history_lengths": {key: len(value) for key, value in history.items()},
                    },
                )

                if val_agg.get("iou", 0.0) > best_metric:
                    best_metric = val_agg["iou"]
                    best_epoch = epoch + 1

                if should_stop:
                    break

            if best_epoch == start_epoch and history["val_iou"]:
                best_epoch = int(np.argmax(history["val_iou"])) + 1
                best_metric = float(max(history["val_iou"]))

            result = TrainingResult(
                history=history,
                best_epoch=best_epoch,
                best_metric=best_metric if np.isfinite(best_metric) else 0.0,
                stopped_early=bool(should_stop),
                total_time_seconds=time.time() - start_time,
                mgda_alpha_history=mgda_history,
                resumed_from_checkpoint=resumed_from_checkpoint,
                start_epoch=start_epoch,
            )
            self._save_history(result)
            return result
        except Exception as exc:
            failure_payload = {
                "epoch": current_epoch + 1,
                "strategy": self.strategy,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            with (self.results_dir / "training_error.json").open("w", encoding="utf-8") as handle:
                json.dump(failure_payload, handle, indent=2)
            self.dual_logger.exception(f"Training failed at epoch {current_epoch + 1}")
            raise
        finally:
            self.tensorboard_logger.close()
            self.dual_logger.close()

    def _save_history(self, result: TrainingResult) -> None:
        summary = {
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "stopped_early": result.stopped_early,
            "total_time_seconds": result.total_time_seconds,
            "resumed_from_checkpoint": result.resumed_from_checkpoint,
            "start_epoch": result.start_epoch,
        }
        with (self.results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if result.mgda_alpha_history:
            with (self.results_dir / "mgda_alphas.json").open("w", encoding="utf-8") as handle:
                json.dump(result.mgda_alpha_history, handle, indent=2)


def train_from_config(
    config: Mapping[str, object],
    output_dir: str = "./results",
    verbose: int = 1,
    resume: bool = False,
    auto_batch_size: bool = False,
) -> Dict[str, object]:
    """Legacy adapter preserving old call-sites with a minimal dry-run summary.

    The old optimization stack imports this function directly. The publication
    pipeline now uses ``ExperimentRunner`` and ``Trainer`` directly, but this
    shim avoids hard failures when legacy modules are imported.
    """
    del verbose, resume, auto_batch_size
    output_path = Path(output_dir)
    run_id = str(config.get("experiment_id", "legacy_run"))
    run_dir = output_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": run_id,
        "final_loss": 0.0,
        "final_val_loss": 0.0,
        "final_val_iou": 0.0,
        "final_val_dice": 0.0,
        "history": {"loss": [], "val_loss": []},
        "final_model_path": "",
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "train_time_seconds": 0.0,
        "memory_footprint_mb": 0.0,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
