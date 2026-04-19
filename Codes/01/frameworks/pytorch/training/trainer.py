"""PyTorch trainer implementing single/weighted/MGDA strategies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from frameworks.pytorch.losses.loss_manager import LossManager
from frameworks.pytorch.optimization.mgda import MGDASolver, MGDATrainStep
from frameworks.pytorch.optimization.schedulers import CosineAnnealingLR, PlateauScheduler
from frameworks.pytorch.training.callbacks import (
    CSVLogger,
    DualLogger,
    MGDAAlphaLogger,
    TensorBoardLogger,
    TrainingTimeLogger,
    ValidationImageLogger,
)
from frameworks.pytorch.training.early_stopping import MultiMetricEarlyStopping
from frameworks.pytorch.training.metrics import compute_batch_metrics


@dataclass
class TrainingResult:
    """Container for training history and best-epoch information."""

    history: Dict[str, list] = field(default_factory=dict)
    best_epoch: int = 0
    best_metric: float = 0.0
    stopped_early: bool = False
    total_time_seconds: float = 0.0
    mgda_alpha_history: list = field(default_factory=list)


class Trainer:
    """Train PyTorch segmentation model with configurable objective strategy."""

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, object],
        device: torch.device,
        checkpoint_manager,
        results_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = dict(config.get("training", {}))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
        )
        self.loss_manager = LossManager(config)
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
        self.cosine_scheduler = CosineAnnealingLR(
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
        self.csv_logger = CSVLogger(self.results_dir / "logs.csv")
        self.tensorboard_logger = TensorBoardLogger(self.results_dir / "tensorboard")
        self.alpha_logger = MGDAAlphaLogger()
        self.time_logger = TrainingTimeLogger()
        self.validation_image_logger = ValidationImageLogger(
            output_dir=self.results_dir / "figures",
            interval=int(log_cfg.get("validation_image_interval", 5)),
        )

    def _set_lr(self, epoch: int) -> float:
        scheduler_type = str(self.config.get("training", {}).get("lr_scheduler", {}).get("type", "cosine")).lower()
        if scheduler_type == "plateau":
            return float(self.optimizer.param_groups[0]["lr"])
        lr = self.cosine_scheduler.get_lr(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return float(lr)

    def _update_plateau(self, metric_value: float) -> None:
        scheduler_type = str(self.config.get("training", {}).get("lr_scheduler", {}).get("type", "cosine")).lower()
        if scheduler_type == "plateau":
            lr = self.plateau_scheduler.step(metric_value)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.model(x)
        if isinstance(prediction, list):
            prediction = prediction[-1]
        return prediction

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Train over epochs and return aggregated training result."""
        epochs = int(self.config.get("training", {}).get("epochs", 100))
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "lr": [],
        }
        start = time.time()
        best_metric = -float("inf")
        best_epoch = 0
        should_stop = False
        mgda_alphas = []

        for epoch in range(epochs):
            epoch_start = time.time()
            current_lr = self._set_lr(epoch)
            self.model.train()
            train_losses = []

            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.strategy == "mgda":
                    losses = {"pixel": self.loss_manager.pixel_loss}
                    if self.loss_manager.boundary_enabled:
                        losses["boundary"] = self.loss_manager.boundary_loss
                    if self.loss_manager.shape_enabled:
                        losses["shape"] = self.loss_manager.shape_loss
                    metrics = self.mgda_stepper.step(images, masks, losses)
                    batch_losses = [value for key, value in metrics.items() if not key.startswith("alpha_")]
                    train_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
                    alpha_info = {k: v for k, v in metrics.items() if k.startswith("alpha_")}
                    if alpha_info:
                        mgda_alphas.append(alpha_info)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    preds = self._forward(images)
                    loss_dict = self.loss_manager.compute_losses(masks, preds)
                    loss = self.loss_manager.compute_weighted_total(loss_dict)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    preds = self._forward(images)
                    loss_dict = self.loss_manager.compute_losses(masks, preds)
                    val_losses.append(float(self.loss_manager.compute_weighted_total(loss_dict).item()))
                    val_metrics.append(compute_batch_metrics(masks, preds))

            mean_train = float(np.mean(train_losses)) if train_losses else 0.0
            mean_val = float(np.mean(val_losses)) if val_losses else 0.0
            mean_iou = float(np.mean([m["iou"] for m in val_metrics])) if val_metrics else 0.0
            mean_dice = float(np.mean([m["dice"] for m in val_metrics])) if val_metrics else 0.0

            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)
            history["val_iou"].append(mean_iou)
            history["val_dice"].append(mean_dice)
            history["lr"].append(current_lr)

            epoch_duration = time.time() - epoch_start
            time_info = self.time_logger.log_epoch(epoch_duration)
            csv_payload = {
                "train_loss": mean_train,
                "train_pixel_loss": 0.0,
                "train_boundary_loss": 0.0,
                "train_shape_loss": 0.0,
                "val_loss": mean_val,
                "val_pixel_loss": 0.0,
                "val_boundary_loss": 0.0,
                "val_shape_loss": 0.0,
                "val_iou": mean_iou,
                "val_dice": mean_dice,
                "val_precision": float(np.mean([m["precision"] for m in val_metrics])) if val_metrics else 0.0,
                "val_recall": float(np.mean([m["recall"] for m in val_metrics])) if val_metrics else 0.0,
                "val_pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in val_metrics])) if val_metrics else 0.0,
                "learning_rate": current_lr,
                **time_info,
            }
            if mgda_alphas:
                csv_payload.update(mgda_alphas[-1])
            self.csv_logger.log_epoch(epoch + 1, csv_payload)

            self.tensorboard_logger.log_scalars(
                epoch + 1,
                {
                    "train/loss": mean_train,
                    "val/loss": mean_val,
                    "val/iou": mean_iou,
                    "val/dice": mean_dice,
                    "lr": current_lr,
                },
            )
            self.dual_logger.log_epoch_summary(
                epoch + 1,
                {"loss": mean_train},
                {"loss": mean_val, "iou": mean_iou, "dice": mean_dice},
                current_lr,
                mgda_alphas[-1] if mgda_alphas else None,
            )
            if mgda_alphas:
                self.alpha_logger.log(epoch + 1, mgda_alphas[-1])
                self.tensorboard_logger.log_mgda_alphas(epoch + 1, mgda_alphas[-1])
            self.validation_image_logger.log(epoch + 1)

            self._update_plateau(mean_iou)
            should_stop = self.early_stopping.step({"val_iou": mean_iou, "val_loss": mean_val})

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                metrics={"val_iou": mean_iou, "val_boundary": 1.0 - mean_dice},
                mgda_solver=self.mgda_solver if self.strategy == "mgda" else None,
            )

            if mean_iou > best_metric:
                best_metric = mean_iou
                best_epoch = epoch + 1
            if should_stop:
                break

        result = TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=best_metric if np.isfinite(best_metric) else 0.0,
            stopped_early=bool(should_stop),
            total_time_seconds=time.time() - start,
            mgda_alpha_history=mgda_alphas,
        )
        self._save_history(result)
        self.tensorboard_logger.close()
        return result

    def _save_history(self, result: TrainingResult) -> None:
        with (self.results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_epoch": result.best_epoch,
                    "best_metric": result.best_metric,
                    "stopped_early": result.stopped_early,
                    "total_time_seconds": result.total_time_seconds,
                },
                handle,
                indent=2,
            )
        if result.mgda_alpha_history:
            with (self.results_dir / "mgda_alphas.json").open("w", encoding="utf-8") as handle:
                json.dump(result.mgda_alpha_history, handle, indent=2)
