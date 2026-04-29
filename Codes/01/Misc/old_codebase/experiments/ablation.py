"""Ablation study utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from visualization.latex_tables import dataframe_to_latex


@dataclass
class AblationExperiment:
    """Generate ablation result table from experiment summaries."""

    output_dir: Path
    reference_experiment: str

    def run(self, experiment_results: Dict[str, dict]) -> pd.DataFrame:
        baseline = experiment_results.get(self.reference_experiment, {})
        baseline_iou = baseline.get("test_metrics", {}).get("iou", 0.0)
        baseline_boundary = baseline.get("test_metrics", {}).get("boundary_iou", 0.0)

        rows = []
        for name, payload in experiment_results.items():
            if "ablation" not in name and name != self.reference_experiment:
                continue
            metrics = payload.get("test_metrics", {})
            iou = metrics.get("iou", 0.0)
            boundary = metrics.get("boundary_iou", 0.0)
            rows.append(
                {
                    "configuration": name,
                    "iou": iou,
                    "dice": metrics.get("dice", 0.0),
                    "hausdorff": boundary,
                    "convexity": metrics.get("compactness", 0.0),
                    "delta_iou": iou - baseline_iou,
                    "delta_hausdorff": boundary - baseline_boundary,
                }
            )

        df = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "ablation.csv", index=False)
        dataframe_to_latex(
            df,
            caption="Ablation study results.",
            label="tab:ablation",
            highlight_best=True,
            highlight_col_direction={"iou": "max", "delta_iou": "max", "hausdorff": "min"},
            save_path=str(self.output_dir / "ablation.tex"),
        )
        return df
