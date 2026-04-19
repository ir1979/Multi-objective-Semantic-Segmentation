"""Deep supervision loss for UNet++ auxiliary outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DeepSupervisionLoss:
    """Compute weighted average loss across deep supervision outputs."""

    base_loss: nn.Module
    weights: Sequence[float]

    def __call__(self, y_true: torch.Tensor, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(outputs) != len(self.weights):
            raise ValueError("Output count must match deep supervision weight count.")
        weighted = []
        for output, weight in zip(outputs, self.weights):
            resized = F.interpolate(output, size=y_true.shape[-2:], mode="bilinear", align_corners=False)
            weighted.append(weight * self.base_loss(y_true, resized))
        return sum(weighted) / max(1e-8, float(sum(self.weights)))
