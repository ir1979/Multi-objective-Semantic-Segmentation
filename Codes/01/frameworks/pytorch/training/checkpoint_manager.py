"""Checkpointing utilities for PyTorch training."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch

from frameworks.pytorch.optimization.mgda import MGDASolver


@dataclass
class CheckpointManager:
    """Manage PyTorch model checkpoints and optimizer/MGDA state."""

    checkpoint_dir: Path
    best_metric_name: str = "val_iou"

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_iou = -float("inf")
        self.best_boundary = float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Mapping[str, float],
        mgda_solver: Optional[MGDASolver] = None,
    ) -> None:
        """Save model checkpoints and training-state payload."""
        torch.save(model.state_dict(), self.checkpoint_dir / "last_epoch.pt")

        val_iou = float(metrics.get("val_iou", -float("inf")))
        if val_iou >= self.best_iou:
            self.best_iou = val_iou
            torch.save(model.state_dict(), self.checkpoint_dir / "best_iou.pt")

        val_boundary = float(metrics.get("val_boundary", float("inf")))
        if val_boundary <= self.best_boundary:
            self.best_boundary = val_boundary
            torch.save(model.state_dict(), self.checkpoint_dir / "best_boundary.pt")

        state = {
            "epoch": epoch,
            "best_iou": self.best_iou,
            "best_boundary": self.best_boundary,
            "optimizer_state": optimizer.state_dict(),
            "mgda_alpha_history": mgda_solver.get_alpha_history() if mgda_solver else None,
        }
        with (self.checkpoint_dir / "training_state.pkl").open("wb") as handle:
            pickle.dump(state, handle)

    def has_checkpoint(self) -> bool:
        """Return True when last checkpoint exists."""
        return (self.checkpoint_dir / "last_epoch.pt").exists()

    def load_latest(self) -> Tuple[Optional[Path], Optional[dict]]:
        """Load latest checkpoint metadata."""
        model_path = self.checkpoint_dir / "last_epoch.pt"
        state_path = self.checkpoint_dir / "training_state.pkl"
        if not model_path.exists() or not state_path.exists():
            return None, None
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        self.best_iou = float(state.get("best_iou", self.best_iou))
        self.best_boundary = float(state.get("best_boundary", self.best_boundary))
        return model_path, state

    def get_best_metric(self) -> float:
        """Return best tracked primary metric."""
        return self.best_iou
