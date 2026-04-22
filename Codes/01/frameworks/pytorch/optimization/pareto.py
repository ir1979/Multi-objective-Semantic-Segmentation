"""Pareto front utilities for PyTorch experiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ParetoFrontComputer:
    """Compute non-dominated points and simple quality indicators."""

    def compute(
        self,
        df: pd.DataFrame,
        objective_columns: List[str],
        minimize: Sequence[bool],
    ) -> pd.DataFrame:
        if len(objective_columns) != len(minimize):
            raise ValueError("objective_columns and minimize lengths must match.")
        if df.empty:
            return df.copy()
        values = df[objective_columns].to_numpy(dtype=np.float64)
        transformed = values.copy()
        for idx, to_minimize in enumerate(minimize):
            if not to_minimize:
                transformed[:, idx] = -transformed[:, idx]
        dominated = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i == j:
                    continue
                if np.all(transformed[j] <= transformed[i]) and np.any(transformed[j] < transformed[i]):
                    dominated[i] = True
                    break
        return df.loc[~dominated].copy()

    def compute_hypervolume(self, pareto_front: pd.DataFrame, reference_point: np.ndarray) -> float:
        """Compute basic dominated hypervolume estimate."""
        if pareto_front.empty:
            return 0.0
        points = pareto_front.to_numpy(dtype=np.float64)
        clipped = np.maximum(reference_point - points, 0.0)
        return float(np.sum(np.prod(clipped, axis=1)))

    def compute_spacing(self, pareto_front: pd.DataFrame) -> float:
        """Compute spacing metric over nearest-neighbor distances."""
        if len(pareto_front) <= 1:
            return 0.0
        points = pareto_front.to_numpy(dtype=np.float64)
        distances = []
        for idx, point in enumerate(points):
            others = np.delete(points, idx, axis=0)
            distances.append(float(np.min(np.linalg.norm(others - point, axis=1))))
        return float(np.std(np.asarray(distances)))
