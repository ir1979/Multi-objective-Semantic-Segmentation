"""Weighted sum combination strategy for PyTorch losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch


@dataclass
class WeightedSumStrategy:
    """Combine multiple loss components with static/dynamic weights."""

    base_weights: Mapping[str, float]
    schedule_type: str = "static"
    ramp_epochs: int = 20
    milestones: Optional[Mapping[int, Mapping[str, float]]] = None

    def get_weights(self, epoch: int) -> Dict[str, float]:
        weights = dict(self.base_weights)
        if self.schedule_type == "static":
            return weights
        if self.schedule_type == "linear_ramp":
            factor = min(1.0, max(0.0, float(epoch) / float(max(1, self.ramp_epochs))))
            return {name: float(weight) * factor for name, weight in weights.items()}
        if self.schedule_type == "step":
            if not self.milestones:
                return weights
            merged = dict(weights)
            for mark in sorted(self.milestones):
                if epoch >= mark:
                    merged.update({k: float(v) for k, v in self.milestones[mark].items()})
            return merged
        if self.schedule_type == "cosine":
            factor = 0.5 * (1.0 + np.cos(np.pi * min(1.0, float(epoch) / float(max(1, self.ramp_epochs)))))
            return {name: float(weight) * factor for name, weight in weights.items()}
        raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def combine(self, losses: Mapping[str, torch.Tensor], epoch: int) -> torch.Tensor:
        weights = self.get_weights(epoch)
        numer = []
        denom = 0.0
        for name, value in losses.items():
            w = float(weights.get(name, 0.0))
            if w <= 0.0:
                continue
            numer.append(value * w)
            denom += w
        if not numer:
            return torch.tensor(0.0)
        return torch.stack(numer).sum() / max(1e-8, denom)
