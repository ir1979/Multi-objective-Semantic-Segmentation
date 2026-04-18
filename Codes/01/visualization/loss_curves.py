"""Loss curve plotting utilities."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import COLORS, save_figure


def _ema(series: pd.Series, alpha: float = 0.9) -> pd.Series:
    return series.ewm(alpha=1 - alpha, adjust=False).mean()


def generate_loss_curves(
    csv_paths: Dict[str, str],
    save_path: str = "figures/loss_curves",
) -> None:
    """Generate a 2x2 grid of train/val loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_specs = [
        ("train_loss", "val_loss", "Total Loss"),
        ("train_pixel_loss", "val_pixel_loss", "Pixel Loss"),
        ("train_boundary_loss", "val_boundary_loss", "Boundary Loss"),
        ("train_shape_loss", "val_shape_loss", "Shape Loss"),
    ]
    for experiment_name, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        color = COLORS.get(experiment_name, None)
        for axis, (train_col, val_col, title) in zip(axes.flatten(), plot_specs):
            if train_col in df:
                axis.plot(df["epoch"], df[train_col], color=color, alpha=0.2)
                axis.plot(df["epoch"], _ema(df[train_col]), color=color, label=f"{experiment_name} train")
            if val_col in df:
                axis.plot(df["epoch"], df[val_col], color=color, alpha=0.15, linestyle="--")
                axis.plot(
                    df["epoch"],
                    _ema(df[val_col]),
                    color=color,
                    linestyle="--",
                    label=f"{experiment_name} val",
                )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save_figure(fig, save_path)
