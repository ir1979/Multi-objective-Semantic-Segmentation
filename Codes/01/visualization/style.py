"""Journal-quality plotting style helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


JOURNAL_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
}

COLORS = {
    # Architecture colours
    "unet":              "#1f77b4",
    "unetpp":            "#d62728",
    "attunet":           "#8c564b",
    "r2attunet":         "#e377c2",
    "seunet":            "#7f7f7f",
    "scse_unet":         "#aec7e8",
    "resunet":           "#98df8a",
    "resunetpp":         "#c5b0d5",
    # Pixel-loss colours
    "bce":               "#2ca02c",
    "bce":           "#ff7f0e",
    "dice":              "#9467bd",
    # Encoder-depth colours
    "deep":              "#17becf",
    "shallow":           "#bcbd22",
    # Pareto plot colours
    "pareto_dominated":  "#cccccc",
    "pareto_front":      "#e74c3c",
}


def apply_journal_style() -> None:
    """Apply journal-style plotting defaults."""
    plt.rcParams.update(JOURNAL_STYLE)


def save_figure(fig: plt.Figure, path_stem: str, formats: Iterable[str] = ("pdf", "png")) -> None:
    """Save figure in all requested formats at 300 DPI."""
    apply_journal_style()
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
