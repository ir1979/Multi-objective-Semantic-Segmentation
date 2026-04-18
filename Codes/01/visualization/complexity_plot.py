"""Model complexity visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.style import COLORS, save_figure


def generate_complexity_plot(
    complexity_data: pd.DataFrame,
    save_path: str = "figures/flops_vs_accuracy",
) -> None:
    """Generate FLOPs-vs-IoU scatter with parameter-scaled markers."""
    if complexity_data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    gflops = complexity_data["flops"] / 1e9
    iou = complexity_data["iou"]
    params = complexity_data["total_params"]
    strategy = complexity_data.get("strategy", pd.Series(["unknown"] * len(complexity_data)))

    colors = [COLORS.get(str(s), "#4C72B0") for s in strategy]
    marker_sizes = 30 + 170 * (params / np.maximum(params.max(), 1.0))
    ax.scatter(gflops, iou, s=marker_sizes, c=colors, alpha=0.8)

    for _, row in complexity_data.iterrows():
        ax.annotate(
            str(row.get("experiment_name", "exp")),
            (row["flops"] / 1e9, row["iou"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # Pareto-like frontier in FLOPs-IoU plane (maximize IoU, minimize FLOPs).
    sorted_df = complexity_data.sort_values("flops")
    frontier_x = []
    frontier_y = []
    best_iou = -float("inf")
    for _, row in sorted_df.iterrows():
        if float(row["iou"]) > best_iou:
            best_iou = float(row["iou"])
            frontier_x.append(float(row["flops"] / 1e9))
            frontier_y.append(float(row["iou"]))
    if frontier_x:
        ax.plot(frontier_x, frontier_y, color="#E74C3C", linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("FLOPs (GFLOPs)")
    ax.set_ylabel("IoU")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, save_path)
