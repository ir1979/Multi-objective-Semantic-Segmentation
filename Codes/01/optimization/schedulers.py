"""Learning-rate schedulers used by training loops."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CosineAnnealingLR:
    """Cosine annealing schedule with warmup support."""

    base_lr: float
    min_lr: float
    total_epochs: int
    warmup_epochs: int = 0

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        progress = float(epoch - self.warmup_epochs) / float(
            max(1, self.total_epochs - self.warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


@dataclass
class WarmupScheduler:
    """Linear warmup then delegates to a base scheduler."""

    warmup_epochs: int
    base_scheduler: CosineAnnealingLR

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_scheduler.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        return self.base_scheduler.get_lr(epoch)


@dataclass
class PlateauScheduler:
    """Reduce learning rate when monitored metric plateaus."""

    initial_lr: float
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7
    mode: str = "max"

    def __post_init__(self) -> None:
        self.best = -float("inf") if self.mode == "max" else float("inf")
        self.wait = 0
        self.current_lr = self.initial_lr

    def step(self, value: float) -> float:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.wait = 0
            return self.current_lr

        self.wait += 1
        if self.wait >= self.patience:
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.wait = 0
        return self.current_lr

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpoint resume."""
        return {
            "best": self.best,
            "wait": self.wait,
            "current_lr": self.current_lr,
            "initial_lr": self.initial_lr,
            "factor": self.factor,
            "patience": self.patience,
            "min_lr": self.min_lr,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state."""
        self.best = float(state.get("best", self.best))
        self.wait = int(state.get("wait", self.wait))
        self.current_lr = float(state.get("current_lr", self.current_lr))
