"""Checkpoint manager with optimizer and MGDA state persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from optimization.mgda import MGDASolver


@dataclass
class CheckpointManager:
    """Manage model/optimizer checkpoint lifecycle."""

    checkpoint_dir: Path
    best_metric_name: str = "val_iou"

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_iou = -float("inf")
        self.best_boundary = float("inf")

    def save(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        epoch: int,
        metrics: Mapping[str, float],
        mgda_solver: Optional[MGDASolver] = None,
        extra_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist model snapshots and training metadata."""
        last_weights_path = self.checkpoint_dir / "last_epoch.weights.h5"
        legacy_model_path_h5 = self.checkpoint_dir / "last_epoch.h5"
        legacy_model_save_dir = self.checkpoint_dir / "last_epoch"
        model.save_weights(last_weights_path)
        # Preserve a full-model artifact only when Keras can serialize the model graph.
        if self._supports_hdf5_export(model):
            try:
                model.save(legacy_model_path_h5, include_optimizer=False)
            except (NotImplementedError, ValueError, OSError):
                pass
        elif self._supports_saved_model_export(model):
            try:
                model.save(legacy_model_save_dir, include_optimizer=False)
            except (NotImplementedError, ValueError, OSError):
                pass

        val_iou = float(metrics.get("val_iou", -float("inf")))
        if val_iou >= self.best_iou:
            self.best_iou = val_iou
            model.save_weights(self.checkpoint_dir / "best_iou.weights.h5")

        val_boundary = float(metrics.get("val_boundary", float("inf")))
        if val_boundary <= self.best_boundary:
            self.best_boundary = val_boundary
            model.save_weights(self.checkpoint_dir / "best_boundary.weights.h5")

        training_state = {
            "epoch": epoch,
            "best_iou": self.best_iou,
            "best_boundary": self.best_boundary,
            "optimizer_weights": self._serialize_optimizer_state(optimizer),
            "mgda_alpha_history": mgda_solver.get_alpha_history() if mgda_solver else None,
            "metrics": dict(metrics),
        }
        if extra_state:
            training_state["extra_state"] = dict(extra_state)
        with (self.checkpoint_dir / "training_state.pkl").open("wb") as handle:
            pickle.dump(training_state, handle)

    @staticmethod
    def _serialize_optimizer_state(optimizer: tf.keras.optimizers.Optimizer) -> list:
        """Capture optimizer state across TF/Keras variants."""
        if hasattr(optimizer, "get_weights"):
            try:
                return list(optimizer.get_weights())
            except Exception:
                pass
        return [np.array(variable.numpy()) for variable in optimizer.variables]

    @staticmethod
    def _supports_hdf5_export(model: tf.keras.Model) -> bool:
        """Return True for graph-network models that can be written to HDF5."""
        return bool(getattr(model, "_is_graph_network", False))

    @staticmethod
    def _supports_saved_model_export(model: tf.keras.Model) -> bool:
        """Return True when a non-graph model is still worth exporting."""
        return isinstance(model, tf.keras.Sequential)

    def load_latest(self) -> Tuple[Optional[Path], Optional[dict]]:
        """Load latest checkpoint metadata."""
        model_path = self.checkpoint_dir / "last_epoch.weights.h5"
        state_path = self.checkpoint_dir / "training_state.pkl"
        if not model_path.exists() or not state_path.exists():
            return None, None

        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        self.best_iou = float(state.get("best_iou", self.best_iou))
        self.best_boundary = float(state.get("best_boundary", self.best_boundary))
        return model_path, state

    @staticmethod
    def restore_optimizer_state(
        optimizer: tf.keras.optimizers.Optimizer,
        state: Optional[dict],
        model: Optional[tf.keras.Model] = None,
    ) -> bool:
        """Best-effort optimizer state restore across TF/Keras variants."""
        if not state:
            return False
        weights = state.get("optimizer_weights")
        if not isinstance(weights, list) or not weights:
            return False
        try:
            if hasattr(optimizer, "build") and model is not None:
                optimizer.build(model.trainable_variables)
        except Exception:
            pass
        if model is not None:
            try:
                zero_grads = [(tf.zeros_like(variable), variable) for variable in model.trainable_variables]
                if zero_grads:
                    optimizer.apply_gradients(zero_grads)
            except Exception:
                pass
        try:
            if hasattr(optimizer, "set_weights"):
                optimizer.set_weights(weights)
                return True
        except Exception:
            pass
        try:
            variables = list(optimizer.variables)
            if len(variables) != len(weights):
                return False
            for variable, value in zip(variables, weights):
                variable.assign(value)
            return True
        except Exception:
            return False

    def has_checkpoint(self) -> bool:
        """Return True if resumable checkpoint artifacts exist."""
        return (self.checkpoint_dir / "last_epoch.weights.h5").exists()

    def get_best_metric(self) -> float:
        """Return best tracked metric value."""
        return self.best_iou
