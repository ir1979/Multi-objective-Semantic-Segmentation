"""Pareto front plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from visualization.style import COLORS, save_figure


def generate_pareto_2d(
    results_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mgda_points: pd.DataFrame | None = None,
    save_path: str = "figures/pareto_2d",
) -> None:
    """Generate 2D Pareto scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        results_df[x_col],
        results_df[y_col],
        c=COLORS["pareto_dominated"],
        label=f"Dominated ({len(results_df)})",
        alpha=0.6,
    )
    if not pareto_df.empty:
        sorted_front = pareto_df.sort_values(by=x_col)
        ax.plot(sorted_front[x_col], sorted_front[y_col], color=COLORS["pareto_front"], linewidth=1.5)
        ax.scatter(
            sorted_front[x_col],
            sorted_front[y_col],
            c=COLORS["pareto_front"],
            marker="D",
            s=60,
            label=f"Pareto ({len(sorted_front)})",
        )
    if mgda_points is not None and not mgda_points.empty:
        ax.scatter(
            mgda_points[x_col],
            mgda_points[y_col],
            c=COLORS["mgda_point"],
            marker="*",
            s=120,
            label="MGDA",
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.annotate("better", xy=(0.95, 0.05), xycoords="axes fraction", ha="right")
    fig.tight_layout()
    save_figure(fig, save_path)


def generate_pareto_3d(
    results_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    save_path: str = "figures/pareto_3d",
) -> None:
    """Generate 3D Pareto scatter plot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[arg-type]
    ax.scatter(results_df[x_col], results_df[y_col], results_df[z_col], c=COLORS["pareto_dominated"], alpha=0.3)
    if not pareto_df.empty:
        ax.scatter(
            pareto_df[x_col],
            pareto_df[y_col],
            pareto_df[z_col],
            c=COLORS["pareto_front"],
            s=50,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    fig.tight_layout()
    save_figure(fig, save_path)
