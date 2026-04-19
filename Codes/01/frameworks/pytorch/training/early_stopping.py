"""Early stopping utility for PyTorch training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MultiMetricEarlyStopping:
    """Monitor primary metric and stop when no improvement is observed."""

    monitor: str
    patience: int
    mode: str = "max"
    min_delta: float = 1e-4
    best_value: float = field(init=False)
    wait_count: int = field(default=0, init=False)
    epoch: int = field(default=0, init=False)
    best_epoch: int = field(default=0, init=False)
    best_metrics: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.best_value = -float("inf") if self.mode == "max" else float("inf")

    def step(self, metrics_dict: Dict[str, float]) -> bool:
        """Update stop state and return True when stopping criterion is met."""
        self.epoch += 1
        value = float(metrics_dict.get(self.monitor, float("nan")))
        improved = (
            value > (self.best_value + self.min_delta)
            if self.mode == "max"
            else value < (self.best_value - self.min_delta)
        )
        if improved:
            self.best_value = value
            self.wait_count = 0
            self.best_epoch = self.epoch
            self.best_metrics = dict(metrics_dict)
            return False
        self.wait_count += 1
        return self.wait_count >= self.patience
