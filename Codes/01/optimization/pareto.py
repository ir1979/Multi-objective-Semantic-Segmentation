"""Pareto front computation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ParetoFrontComputer:
    """Compute Pareto-optimal points and summary indicators."""

    def compute(
        self,
        df: pd.DataFrame,
        objective_columns: List[str],
        minimize: Sequence[bool],
    ) -> pd.DataFrame:
        """Return non-dominated rows from ``df``."""
        if len(objective_columns) != len(minimize):
            raise ValueError("objective_columns and minimize lengths must match.")
        if df.empty:
            return df.copy()

        values = df[objective_columns].to_numpy(dtype=np.float64)
        transformed = values.copy()
        for idx, to_minimize in enumerate(minimize):
            if not to_minimize:
                transformed[:, idx] = -transformed[:, idx]

        is_dominated = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if is_dominated[i]:
                continue
            for j in range(len(df)):
                if i == j:
                    continue
                if np.all(transformed[j] <= transformed[i]) and np.any(transformed[j] < transformed[i]):
                    is_dominated[i] = True
                    break
        return df.loc[~is_dominated].copy()

    def compute_hypervolume(self, pareto_front: pd.DataFrame, reference_point: np.ndarray) -> float:
        """Compute 2D/3D hypervolume using axis-aligned dominated cuboids."""
        if pareto_front.empty:
            return 0.0
        points = pareto_front.to_numpy(dtype=np.float64)
        if points.shape[1] not in (2, 3):
            # Lightweight approximation for higher dimensions.
            clipped = np.maximum(reference_point - points, 0.0)
            return float(np.mean(np.prod(clipped, axis=1)))
        clipped = np.maximum(reference_point - points, 0.0)
        return float(np.sum(np.prod(clipped, axis=1)))

    def compute_spacing(self, pareto_front: pd.DataFrame) -> float:
        """Compute spacing metric for Pareto-point uniformity."""
        if len(pareto_front) <= 1:
            return 0.0
        points = pareto_front.to_numpy(dtype=np.float64)
        distances = []
        for idx, point in enumerate(points):
            others = np.delete(points, idx, axis=0)
            nearest = np.min(np.linalg.norm(others - point, axis=1))
            distances.append(nearest)
        distances = np.asarray(distances)
        return float(np.std(distances))
