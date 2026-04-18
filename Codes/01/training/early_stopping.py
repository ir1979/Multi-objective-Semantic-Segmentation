"""Multi-metric early stopping utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MultiMetricEarlyStopping:
    """Monitor one primary metric and track best epoch/values."""

    monitor: str
    patience: int
    mode: str = "max"
    min_delta: float = 1e-4
    best_value: float = field(init=False)
    best_epoch: int = field(default=0, init=False)
    wait_count: int = field(default=0, init=False)
    _epoch: int = field(default=0, init=False)
    _best_metrics: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.best_value = -float("inf") if self.mode == "max" else float("inf")

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        return value < (self.best_value - self.min_delta)

    def step(self, metrics_dict: Dict[str, float]) -> bool:
        """Update stopping state and return ``True`` when training should stop."""
        self._epoch += 1
        value = float(metrics_dict.get(self.monitor, float("nan")))
        if value != value:  # NaN guard
            self.wait_count += 1
            return self.wait_count >= self.patience

        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = self._epoch
            self.wait_count = 0
            self._best_metrics = dict(metrics_dict)
            return False

        self.wait_count += 1
        return self.wait_count >= self.patience

    def get_best_epoch(self) -> int:
        """Return the best epoch number (1-indexed)."""
        return self.best_epoch

    def get_best_metrics(self) -> Dict[str, float]:
        """Return metrics snapshot captured at best epoch."""
        return dict(self._best_metrics)
