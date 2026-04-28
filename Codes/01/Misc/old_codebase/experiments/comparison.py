"""Comparison experiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from visualization.latex_tables import dataframe_to_latex


@dataclass
class ModelComparisonExperiment:
    """Build comparison tables from completed experiment summaries."""

    output_dir: Path

    def run(self, experiment_results: Dict[str, dict]) -> pd.DataFrame:
        rows = []
        for name, payload in experiment_results.items():
            metrics = payload.get("test_metrics", {})
            complexity = payload.get("model_complexity", {})
            rows.append(
                {
                    "experiment": name,
                    "iou": metrics.get("iou", 0.0),
                    "dice": metrics.get("dice", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "hausdorff": metrics.get("boundary_iou", 0.0),
                    "convexity": metrics.get("compactness", 0.0),
                    "params": complexity.get("total_params", 0),
                    "flops": complexity.get("flops", 0),
                    "time_ms": complexity.get("inference_time_ms", 0.0),
                }
            )
        df = pd.DataFrame(rows).sort_values("iou", ascending=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "metrics.csv", index=False)
        dataframe_to_latex(
            df,
            caption="Model comparison across architectures and strategies.",
            label="tab:model_comparison",
            highlight_best=True,
            highlight_col_direction={"iou": "max", "dice": "max", "time_ms": "min"},
            save_path=str(self.output_dir / "metrics.tex"),
        )
        return df
