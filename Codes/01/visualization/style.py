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
    "unet_single": "#1f77b4",
    "unet_weighted": "#ff7f0e",
    "unet_mgda": "#2ca02c",
    "unetpp_single": "#d62728",
    "unetpp_weighted": "#9467bd",
    "unetpp_mgda": "#8c564b",
    "unetpp_deepsup": "#e377c2",
    "pareto_dominated": "#cccccc",
    "pareto_front": "#e74c3c",
    "mgda_point": "#2ecc71",
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
