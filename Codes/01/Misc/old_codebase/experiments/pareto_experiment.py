"""Pareto sweep experiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from optimization.pareto import ParetoFrontComputer
from visualization.latex_tables import dataframe_to_latex
from visualization.pareto_plot import generate_pareto_2d, generate_pareto_3d


@dataclass
class ParetoExperiment:
    """Generate Pareto sweep combinations and front visualizations."""

    output_dir: Path
    random_seed: int = 42
    max_points: int = 30

    def generate_weight_grid(self) -> List[Dict[str, float]]:
        """Generate subsampled weight combinations."""
        pixel = [0.2, 0.4, 0.6, 0.8, 1.0]
        boundary = [0.0, 0.1, 0.2, 0.3, 0.5]
        shape = [0.0, 0.05, 0.1, 0.2]
        candidates = [
            {"pixel_weight": p, "boundary_weight": b, "shape_weight": s}
            for p, b, s in product(pixel, boundary, shape)
            if (p + b + s) > 0.0
        ]
        rng = np.random.default_rng(self.random_seed)
        if len(candidates) <= self.max_points:
            return candidates
        chosen = rng.choice(len(candidates), size=self.max_points, replace=False)
        return [candidates[int(idx)] for idx in sorted(chosen.tolist())]

    def compute_pareto_front(self, results: pd.DataFrame) -> pd.DataFrame:
        """Compute Pareto front from sweep results."""
        if results.empty:
            return results.copy()
        frame = self._add_objective_columns(results)
        computer = ParetoFrontComputer()
        front = computer.compute(
            frame,
            objective_columns=["obj_iou", "obj_boundary", "obj_convexity"],
            minimize=[True, True, True],
        )
        return front.sort_values("obj_iou")

    @staticmethod
    def _add_objective_columns(results: pd.DataFrame) -> pd.DataFrame:
        """Attach optimization-objective columns used by Pareto exports."""
        frame = results.copy()
        frame["obj_iou"] = 1.0 - frame["iou"]
        frame["obj_boundary"] = frame["hausdorff"]
        frame["obj_convexity"] = 1.0 - frame["convexity"]
        return frame

    @staticmethod
    def is_dominated(point: np.ndarray, other_points: np.ndarray) -> bool:
        """Return True if point is dominated by any row in other_points."""
        for other in other_points:
            if np.all(other <= point) and np.any(other < point):
                return True
        return False

    def save_outputs(self, results: pd.DataFrame, pareto_front: pd.DataFrame) -> None:
        """Export Pareto data tables and figures."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_with_objectives = self._add_objective_columns(results)
        front_with_objectives = self._add_objective_columns(pareto_front)
        results_with_objectives.to_csv(self.output_dir / "pareto_points.csv", index=False)
        front_with_objectives.to_csv(self.output_dir / "pareto_front.csv", index=False)
        dataframe_to_latex(
            front_with_objectives[
                [
                    "pixel_weight",
                    "boundary_weight",
                    "shape_weight",
                    "iou",
                    "hausdorff",
                    "convexity",
                ]
            ],
            caption="Pareto-optimal configurations from weight sweep.",
            label="tab:pareto_points",
            highlight_best=True,
            highlight_col_direction={"iou": "max", "hausdorff": "min", "convexity": "max"},
            save_path=str(self.output_dir / "pareto_points.tex"),
        )
        generate_pareto_2d(
            results_with_objectives,
            front_with_objectives,
            x_col="obj_iou",
            y_col="obj_boundary",
            save_path=str(self.output_dir / "pareto_2d"),
        )
        generate_pareto_3d(
            results_with_objectives,
            front_with_objectives,
            x_col="obj_iou",
            y_col="obj_boundary",
            z_col="obj_convexity",
            save_path=str(self.output_dir / "pareto_3d"),
        )
