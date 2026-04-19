"""Model complexity helpers for PyTorch models."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class ModelComplexityAnalyzer:
    """Compute parameter count and inference-time statistics."""

    input_shape: Tuple[int, int, int] = (3, 256, 256)
    num_runs: int = 100
    warmup_runs: int = 10

    def analyze(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        """Measure model complexity values."""
        model = model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        sample = torch.randn(1, *self.input_shape, device=device)
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            times = []
            for _ in range(self.num_runs):
                start = time.perf_counter()
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000.0)

        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

        return {
            "trainable_params": float(trainable_params),
            "total_params": float(total_params),
            # Lightweight approximate value from parameter count; replace with profiler if desired.
            "flops": float(total_params * 2.0),
            "inference_time_ms": float(statistics.mean(times)),
            "peak_memory_mb": peak_memory_mb,
        }
