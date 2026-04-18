"""Visualization utilities for semantic segmentation experiments."""

from .boundary_overlay import generate_boundary_overlay
from .complexity_plot import generate_complexity_plot
from .error_maps import generate_error_maps
from .latex_tables import dataframe_to_latex, generate_all_paper_tables
from .loss_curves import generate_loss_curves
from .pareto_plot import generate_pareto_2d, generate_pareto_3d
from .prediction_grid import generate_prediction_grid
from .style import COLORS, JOURNAL_STYLE, apply_journal_style, save_figure
from .visualization import (
    plot_loss_curves,
    save_boundary_overlay,
    save_error_maps,
    save_pareto_plot,
    save_prediction_grid,
    save_sample_predictions,
)

__all__ = [
    "COLORS",
    "JOURNAL_STYLE",
    "apply_journal_style",
    "save_figure",
    "generate_prediction_grid",
    "generate_boundary_overlay",
    "generate_pareto_2d",
    "generate_pareto_3d",
    "generate_loss_curves",
    "generate_error_maps",
    "generate_complexity_plot",
    "dataframe_to_latex",
    "generate_all_paper_tables",
    "save_sample_predictions",
    "save_boundary_overlay",
    "plot_loss_curves",
    "save_pareto_plot",
    "save_error_maps",
    "save_prediction_grid",
]
