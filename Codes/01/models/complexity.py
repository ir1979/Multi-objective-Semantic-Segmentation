"""Model complexity analysis utilities."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class ModelComplexityAnalyzer:
    """Compute complexity and runtime metrics for a Keras model."""

    input_shape: Tuple[int, int, int] = (256, 256, 3)
    num_runs: int = 100
    num_warmup: int = 10

    def _count_params(self, model: tf.keras.Model) -> Dict[str, int]:
        trainable_params = int(np.sum([np.prod(var.shape) for var in model.trainable_variables]))
        non_trainable_params = int(np.sum([np.prod(var.shape) for var in model.non_trainable_variables]))
        return {
            "trainable_params": trainable_params,
            "total_params": trainable_params + non_trainable_params,
        }

    def _estimate_flops(self, model: tf.keras.Model) -> int:
        # Lightweight fallback FLOPs estimate.
        dummy = tf.random.uniform((1,) + self.input_shape, dtype=tf.float32)
        _ = model(dummy, training=False)
        flops = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                output_shape = getattr(layer, "output_shape", None)
                if output_shape is None and hasattr(layer, "output"):
                    output_shape = tuple(layer.output.shape)
                if isinstance(output_shape, list):
                    continue
                _, h, w, c_out = output_shape
                input_shape = getattr(layer, "input_shape", None)
                if input_shape is None and hasattr(layer, "input"):
                    input_shape = tuple(layer.input.shape)
                c_in = int(input_shape[-1])
                kh, kw = layer.kernel_size
                flops += int(2 * h * w * c_in * c_out * kh * kw)
            elif isinstance(layer, tf.keras.layers.Dense):
                input_shape = getattr(layer, "input_shape", None)
                if input_shape is None and hasattr(layer, "input"):
                    input_shape = tuple(layer.input.shape)
                flops += int(2 * int(input_shape[-1]) * int(layer.units))
        return int(flops)

    def _measure_inference(self, model: tf.keras.Model) -> float:
        @tf.function
        def infer(x: tf.Tensor) -> tf.Tensor:
            return model(x, training=False)

        x = tf.random.uniform((1,) + self.input_shape, dtype=tf.float32)
        for _ in range(self.num_warmup):
            _ = infer(x)
        if hasattr(tf.experimental, "async_wait"):
            tf.experimental.async_wait()

        times = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            _ = infer(x)
            if hasattr(tf.experimental, "async_wait"):
                tf.experimental.async_wait()
            times.append((time.perf_counter() - start) * 1_000.0)
        return float(statistics.mean(times))

    def _peak_memory_mb(self) -> float:
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            return float(info.get("peak", 0) / (1024 * 1024))
        except Exception:
            return 0.0

    def analyze(self, model: tf.keras.Model) -> Dict[str, float]:
        """Analyze model complexity and return a unified metrics dictionary."""
        params = self._count_params(model)
        flops = self._estimate_flops(model)
        inference_ms = self._measure_inference(model)
        peak_memory_mb = self._peak_memory_mb()
        return {
            **params,
            "flops": int(flops),
            "inference_time_ms": float(inference_ms),
            "peak_memory_mb": float(peak_memory_mb),
        }
