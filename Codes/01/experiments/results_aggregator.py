"""Paper-quality output generation from completed grid search results.

Produces the following outputs under ``paper_outputs/``:

tables/
    results_all.csv              – full results table (all points)
    results_main.tex             – booktabs LaTeX table, top-N configs, all metrics
    hyperparameter_ablation.tex  – mean±std per hyperparameter value
    pareto_front.csv             – 3-objective Pareto-optimal configurations
    pareto_front.tex             – LaTeX table of 3-objective Pareto front
    pareto_per_architecture.csv  – per-architecture Pareto membership summary
    pareto_per_architecture.tex  – LaTeX table of per-architecture Pareto summary

figures/
    pareto_iou_vs_boundary.png/pdf       – 2-D pairwise Pareto: IoU × Boundary-F1
    pareto_iou_vs_compactness.png/pdf    – 2-D pairwise Pareto: IoU × Compactness
    pareto_boundary_vs_compactness.png/pdf – 2-D pairwise Pareto: Boundary-F1 × Compactness
    pareto_combined.png/pdf              – single figure with all 3 pairwise panels
    pareto_3d.png/pdf                    – 3-objective 3-D scatter
    comparison_by_*.png/pdf              – box plots per hyperparameter (6 total)
    predictions_comparison_grid.png      – image grid of top-N prediction samples

diagnostics/
    summary_statistics.json          – mean/std/min/max per metric
    metric_distributions.png         – histogram grid for all 8 metrics
    correlation_matrix.png           – Pearson correlation of metrics
    grid_search.log                  – full training/eval log (already created during run)

Pareto objectives
-----------------
* test_iou          – maximise  (region overlap)
* test_boundary_f1  – maximise  (boundary sharpness)
* test_compactness  – minimise  (lower = more regular/compact building shape)
"""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from optimization.pareto import ParetoFrontComputer
from visualization.style import apply_journal_style, save_figure, COLORS
from visualization.latex_tables import dataframe_to_latex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HYPERPARAMS: List[str] = [
    "model_architecture",
    "encoder_filters",
    "pixel_loss_type",
    "boundary_loss_weight",
    "shape_loss_weight",
    "learning_rate",
]

HYPERPARAM_LABELS: Dict[str, str] = {
    "model_architecture":   "Architecture",
    "encoder_filters":      "Encoder Depth",
    "pixel_loss_type":      "Pixel Loss",
    "boundary_loss_weight": "Boundary Weight",
    "shape_loss_weight":    "Shape Weight",
    "learning_rate":        "Learning Rate",
}

ALL_TEST_METRICS: List[str] = [
    "test_iou",
    "test_dice",
    "test_precision",
    "test_recall",
    "test_pixel_acc",
    "test_boundary_iou",
    "test_boundary_f1",
    "test_compactness",
]

METRIC_LABELS: Dict[str, str] = {
    "test_iou":          "IoU",
    "test_dice":         "Dice",
    "test_precision":    "Precision",
    "test_recall":       "Recall",
    "test_pixel_acc":    "Pixel Acc.",
    "test_boundary_iou": "Boundary IoU",
    "test_boundary_f1":  "Boundary F1",
    "test_compactness":  "Compactness",
    "val_iou":           "Val IoU",
}

PARETO_METRICS: List[str] = ["test_iou", "test_boundary_f1", "test_compactness"]
# IoU and Boundary F1 are maximised; Compactness is minimised (lower = more regular shape)
PARETO_MINIMIZE: List[bool] = [False, False, True]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class GridSearchResultsAggregator:
    """Load completed grid search results and generate all paper outputs."""

    def __init__(self, state_file: Path, paper_output_dir: Path) -> None:
        self.state_file = Path(state_file)
        self.out_dir    = Path(paper_output_dir)
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

        self.df = self._load_results()
        self._results_root = self.state_file.parent   # grid_search_results/

    # ------------------------------------------------------------------ load

    def _load_results(self) -> pd.DataFrame:
        """Read completed grid-point records from the JSON state file."""
        if not self.state_file.exists():
            logger.warning(f"State file not found: {self.state_file}")
            return pd.DataFrame()

        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for pid, pdata in data.items():
            if pdata.get("status") != "completed":
                continue
            row: Dict[str, Any] = {"point_id": int(pid)}
            row.update(pdata.get("params", {}))
            row.update(pdata.get("metrics", {}))
            row["result_dir"] = pdata.get("result_dir", "")
            records.append(row)

        df = pd.DataFrame(records)
        # normalise encoder_filters to a readable depth label
        if "encoder_filters" in df.columns:
            def _depth_label(v: Any) -> str:
                if not isinstance(v, list):
                    return str(v)
                m = max(v)
                if m >= 1024:
                    return "deep"
                if m >= 512:
                    return "shallow"
                return "micro"
            df["encoder_filters"] = df["encoder_filters"].apply(_depth_label)
        logger.info(f"Loaded {len(df)} completed grid points")
        return df

    def _require_data(self) -> bool:
        if self.df.empty:
            logger.warning("No completed results available – skipping output generation")
            return False
        return True

    # ------------------------------------------------------------------ tables

    def save_all_results_csv(self) -> Path:
        """Dump the full results DataFrame."""
        out = self.out_dir / "tables" / "results_all.csv"
        self.df.to_csv(out, index=False)
        logger.info(f"Saved full results CSV → {out}")
        return out

    def save_latex_main_results_table(self, top_n: int = 15) -> Optional[Path]:
        """Booktabs LaTeX table of the top-N configurations ranked by test IoU."""
        if not self._require_data():
            return None
        avail_metrics = [m for m in ALL_TEST_METRICS if m in self.df.columns]
        rank_by = "test_iou" if "test_iou" in self.df.columns else avail_metrics[0]
        top = self.df.nlargest(top_n, rank_by).copy()

        show_cols = ["point_id"] + [h for h in HYPERPARAM_LABELS if h in top.columns] + avail_metrics
        show_cols = [c for c in show_cols if c in top.columns]
        display = top[show_cols].copy()

        # Pretty-print numeric metric columns
        for col in avail_metrics:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")

        # Rename columns for the paper
        rename = {**METRIC_LABELS, **HYPERPARAM_LABELS, "point_id": "ID"}
        display.rename(columns={k: v for k, v in rename.items() if k in display.columns}, inplace=True)

        out = self.out_dir / "tables" / "results_main.tex"
        latex_str = dataframe_to_latex(
            display,
            caption=f"Top-{top_n} configurations ranked by test IoU. "
                    "Best value in each column is shown in bold.",
            label="tab:results_main",
            highlight_best=True,
            highlight_col_direction={METRIC_LABELS.get(m, m): "max" for m in avail_metrics},
            save_path=str(out),
        )
        logger.info(f"Saved main results LaTeX table → {out}")
        return out

    def save_latex_hyperparameter_ablation(self) -> Optional[Path]:
        """One LaTeX sub-table per hyperparameter: mean±std of each metric."""
        if not self._require_data():
            return None
        avail_metrics = [m for m in ALL_TEST_METRICS if m in self.df.columns]
        lines: List[str] = [
            r"\section*{Hyperparameter Ablation}",
            "",
        ]
        for hp in HYPERPARAM_LABELS:
            if hp not in self.df.columns:
                continue
            grp = self.df.groupby(hp)[avail_metrics]
            mean_df = grp.mean().round(4)
            std_df  = grp.std().round(4)
            # combine mean±std
            combined = mean_df.copy()
            for col in avail_metrics:
                if col in mean_df.columns and col in std_df.columns:
                    combined[col] = mean_df[col].map(lambda x: f"{x:.4f}") + r" \pm " + std_df[col].map(lambda x: f"{x:.4f}")
            combined.reset_index(inplace=True)
            combined.rename(columns={**METRIC_LABELS, **HYPERPARAM_LABELS}, inplace=True)

            caption = (
                f"Effect of \\textbf{{{HYPERPARAM_LABELS[hp]}}} on evaluation metrics. "
                "Values are mean$\\pm$std across all configurations sharing that setting."
            )
            label = f"tab:ablation_{hp}"
            tex = dataframe_to_latex(combined, caption=caption, label=label,
                                     highlight_best=False, save_path=None)
            lines.append(tex)
            lines.append("")

        combined_tex = "\n".join(lines)
        out = self.out_dir / "tables" / "hyperparameter_ablation.tex"
        out.write_text(combined_tex, encoding="utf-8")
        logger.info(f"Saved hyperparameter ablation tables → {out}")
        return out

    # ------------------------------------------------------------------ Pareto

    def compute_and_save_pareto_front(self) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
        """Compute Pareto front across (test_iou, test_boundary_f1, test_compactness)."""
        if not self._require_data():
            return pd.DataFrame(), None, None

        available = [m for m in PARETO_METRICS if m in self.df.columns]
        if len(available) < 2:
            logger.warning("Not enough Pareto metrics available – need at least 2")
            return pd.DataFrame(), None, None

        minimize = [PARETO_MINIMIZE[PARETO_METRICS.index(m)] for m in available]
        computer = ParetoFrontComputer()
        pareto_df = computer.compute(self.df, available, minimize)

        csv_out = self.out_dir / "tables" / "pareto_front.csv"
        pareto_df.to_csv(csv_out, index=False)

        show_cols = ["point_id"] + [h for h in HYPERPARAM_LABELS if h in pareto_df.columns] + available
        show_cols = [c for c in show_cols if c in pareto_df.columns]
        display = pareto_df[show_cols].copy()
        for col in available:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:.4f}")
        display.rename(columns={**METRIC_LABELS, **HYPERPARAM_LABELS, "point_id": "ID"}, inplace=True)

        tex_out = self.out_dir / "tables" / "pareto_front.tex"
        dataframe_to_latex(
            display,
            caption=(
                "Pareto-optimal configurations under the objectives "
                r"\textit{Region IoU}, \textit{Boundary F1}, and \textit{Shape Compactness}. "
                "Configurations are non-dominated when all three objectives are considered jointly."
            ),
            label="tab:pareto_front",
            highlight_best=True,
            highlight_col_direction={METRIC_LABELS.get(m, m): "max" for m in available},
            save_path=str(tex_out),
        )
        logger.info(f"Pareto front: {len(pareto_df)} non-dominated configs → {csv_out}")
        return pareto_df, csv_out, tex_out

    # ------------------------------------------------------------------ figures

    def save_pareto_scatter_plots(self, pareto_df: Optional[pd.DataFrame] = None) -> List[Path]:
        """2-D scatter plots for each pair of Pareto objectives.

        Each plot uses the **pairwise** 2-objective Pareto front (not the full 3-obj
        front) so that the trade-off envelope is correct for the shown dimensions.
        Points are colour-coded by architecture to reveal which models reach the
        trade-off surface.  The compactness axis is inverted so that "better"
        always points up/right in every panel.
        """
        if not self._require_data():
            return []

        apply_journal_style()
        computer  = ParetoFrontComputer()
        arch_col  = "model_architecture" if "model_architecture" in self.df.columns else None
        out_paths: List[Path] = []

        # (x_col, y_col, file_stem, x_minimize, y_minimize)
        pairs = [
            ("test_iou",         "test_boundary_f1",  "pareto_iou_vs_boundary",        False, False),
            ("test_iou",         "test_compactness",  "pareto_iou_vs_compactness",      False, True),
            ("test_boundary_f1", "test_compactness",  "pareto_boundary_vs_compactness", False, True),
        ]

        for x_col, y_col, stem, x_min, y_min in pairs:
            if x_col not in self.df.columns or y_col not in self.df.columns:
                continue

            # Compute 2-objective Pareto front for this pair
            valid = self.df[[x_col, y_col, "point_id"]].dropna()
            subset = self.df.loc[valid.index].copy()
            pairwise_pareto = computer.compute(subset, [x_col, y_col], [x_min, y_min])
            front_ids = (
                set(pairwise_pareto["point_id"].tolist())
                if "point_id" in pairwise_pareto.columns
                else set()
            )

            fig, ax = plt.subplots(figsize=(6.5, 5.5))

            if arch_col:
                archs = sorted(self.df[arch_col].dropna().unique(), key=str)
                for arch in archs:
                    color    = COLORS.get(arch, "#888888")
                    arch_mask = self.df[arch_col] == arch
                    dom_mask  = arch_mask & ~self.df["point_id"].isin(front_ids)
                    frt_mask  = arch_mask & self.df["point_id"].isin(front_ids)
                    if dom_mask.any():
                        ax.scatter(
                            self.df.loc[dom_mask, x_col], self.df.loc[dom_mask, y_col],
                            c=color, s=35, alpha=0.40, zorder=2,
                        )
                    if frt_mask.any():
                        ax.scatter(
                            self.df.loc[frt_mask, x_col], self.df.loc[frt_mask, y_col],
                            c=color, s=110, alpha=0.95,
                            edgecolors="black", linewidths=0.7, zorder=4,
                            label=arch,
                        )
            else:
                # Fallback: single colour
                dom_mask = ~self.df["point_id"].isin(front_ids)
                ax.scatter(
                    self.df.loc[dom_mask, x_col], self.df.loc[dom_mask, y_col],
                    c=COLORS.get("pareto_dominated", "#cccccc"), s=35, alpha=0.55,
                    zorder=2,
                )
                if not pairwise_pareto.empty:
                    ax.scatter(
                        pairwise_pareto[x_col], pairwise_pareto[y_col],
                        c=COLORS.get("pareto_front", "#e74c3c"), s=100, alpha=0.95,
                        edgecolors="black", linewidths=0.7, zorder=4, label="Pareto front",
                    )

            # Pareto-front step line (sorted by x for the visual envelope)
            if not pairwise_pareto.empty:
                sorted_front = pairwise_pareto.sort_values(x_col)
                ax.step(
                    sorted_front[x_col], sorted_front[y_col],
                    color="#e74c3c", linewidth=1.6, alpha=0.75,
                    where="post", zorder=3, linestyle="--",
                )

            # Invert axis for objectives that are minimised
            if x_min:
                ax.invert_xaxis()
            if y_min:
                ax.invert_yaxis()

            # "Better" direction arrows in corners
            ax.annotate(
                ("← better" if x_min else "→ better"),
                xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="#444444",
            )
            ax.annotate(
                ("↓ better" if y_min else "↑ better"),
                xy=(0.03, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color="#444444",
            )

            ax.set_xlabel(METRIC_LABELS.get(x_col, x_col))
            ax.set_ylabel(METRIC_LABELS.get(y_col, y_col))
            ax.set_title(
                f"{METRIC_LABELS.get(x_col, x_col)} vs {METRIC_LABELS.get(y_col, y_col)}\n"
                f"({len(pairwise_pareto)} Pareto-optimal / {len(self.df)} total)",
                fontsize=10,
            )
            if arch_col:
                ax.legend(fontsize=7, ncol=2, loc="lower right", title="Architecture")
            else:
                ax.legend(fontsize=8, loc="lower right")
            plt.tight_layout()
            dest = self.out_dir / "figures" / stem
            save_figure(fig, str(dest), formats=("png", "pdf"))
            out_paths.append(dest.with_suffix(".png"))
            plt.close(fig)

        logger.info(f"Saved {len(out_paths)} Pareto scatter plots")
        return out_paths

    def save_pareto_combined_figure(self) -> Optional[Path]:
        """Publication-ready single figure: 3 pairwise Pareto panels side-by-side.

        This is the primary figure for the paper – one A4-wide image containing
        all three trade-off planes so readers can evaluate the architectures at a
        glance.
        """
        if not self._require_data():
            return None

        apply_journal_style()
        computer = ParetoFrontComputer()
        arch_col = "model_architecture" if "model_architecture" in self.df.columns else None

        # (x_col, y_col, panel_title, x_minimize, y_minimize)
        panels = [
            ("test_iou",         "test_boundary_f1",  "(a) IoU vs Boundary F1",        False, False),
            ("test_iou",         "test_compactness",  "(b) IoU vs Compactness",         False, True),
            ("test_boundary_f1", "test_compactness",  "(c) Boundary F1 vs Compactness", False, True),
        ]

        # Collect colours and architectures once
        archs = sorted(self.df[arch_col].dropna().unique(), key=str) if arch_col else []

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

        for ax, (x_col, y_col, title, x_min, y_min) in zip(axes, panels):
            if x_col not in self.df.columns or y_col not in self.df.columns:
                ax.axis("off")
                continue

            # 2-objective Pareto front for this pair
            valid = self.df[[x_col, y_col, "point_id"]].dropna()
            subset = self.df.loc[valid.index].copy()
            pairwise_pareto = computer.compute(subset, [x_col, y_col], [x_min, y_min])
            front_ids = (
                set(pairwise_pareto["point_id"].tolist())
                if "point_id" in pairwise_pareto.columns
                else set()
            )

            if archs:
                for arch in archs:
                    color    = COLORS.get(arch, "#888888")
                    a_mask   = self.df[arch_col] == arch
                    dom_mask = a_mask & ~self.df["point_id"].isin(front_ids)
                    frt_mask = a_mask & self.df["point_id"].isin(front_ids)
                    if dom_mask.any():
                        ax.scatter(
                            self.df.loc[dom_mask, x_col], self.df.loc[dom_mask, y_col],
                            c=color, s=28, alpha=0.35, zorder=2,
                        )
                    if frt_mask.any():
                        ax.scatter(
                            self.df.loc[frt_mask, x_col], self.df.loc[frt_mask, y_col],
                            c=color, s=90, alpha=0.95,
                            edgecolors="black", linewidths=0.6, zorder=4,
                            label=arch,
                        )
            else:
                dom_mask = ~self.df["point_id"].isin(front_ids)
                ax.scatter(
                    self.df.loc[dom_mask, x_col], self.df.loc[dom_mask, y_col],
                    c="#cccccc", s=28, alpha=0.50, zorder=2,
                )
                if not pairwise_pareto.empty:
                    ax.scatter(
                        pairwise_pareto[x_col], pairwise_pareto[y_col],
                        c="#e74c3c", s=90, alpha=0.95,
                        edgecolors="black", linewidths=0.6, zorder=4, label="Pareto front",
                    )

            # Step-line connecting the Pareto envelope
            if not pairwise_pareto.empty:
                sorted_front = pairwise_pareto.sort_values(x_col)
                ax.step(
                    sorted_front[x_col], sorted_front[y_col],
                    color="#e74c3c", linewidth=1.5, alpha=0.70,
                    where="post", zorder=3, linestyle="--",
                )

            if x_min:
                ax.invert_xaxis()
            if y_min:
                ax.invert_yaxis()

            ax.annotate(
                ("← better" if x_min else "→ better"),
                xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7.5, color="#444444",
            )
            ax.annotate(
                ("↓ better" if y_min else "↑ better"),
                xy=(0.03, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=7.5, color="#444444",
            )

            ax.set_xlabel(METRIC_LABELS.get(x_col, x_col), fontsize=10)
            ax.set_ylabel(METRIC_LABELS.get(y_col, y_col), fontsize=10)
            ax.set_title(title, fontsize=10, pad=4)

        # Shared legend (architectures) on the right of the last panel
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(
                handles, labels,
                fontsize=7, ncol=1,
                loc="lower right", title="Architecture", title_fontsize=7,
            )

        fig.suptitle(
            "Pairwise Pareto Fronts Across Three Competing Objectives\n"
            r"(IoU $\uparrow$, Boundary F1 $\uparrow$, Compactness $\downarrow$)",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        dest = self.out_dir / "figures" / "pareto_combined"
        save_figure(fig, str(dest), formats=("png", "pdf"))
        plt.close(fig)
        logger.info(f"Saved combined Pareto figure → {dest}.png")
        return dest.with_suffix(".png")

    def save_pareto_3d_figure(self, pareto_df: Optional[pd.DataFrame] = None) -> Optional[Path]:
        """3D scatter of all three Pareto objectives; Pareto-optimal points highlighted.

        Provides an intuitive overview of the full 3-objective trade-off space for
        inclusion in the paper as a supplementary figure.
        """
        if not self._require_data():
            return None
        available = [m for m in PARETO_METRICS if m in self.df.columns]
        if len(available) < 3:
            logger.warning("Need all 3 Pareto metrics for the 3-D figure – skipping")
            return None

        if pareto_df is None or pareto_df.empty:
            pareto_df, _, _ = self.compute_and_save_pareto_front()

        apply_journal_style()
        front_ids = (
            set(pareto_df["point_id"].tolist())
            if pareto_df is not None and "point_id" in pareto_df.columns
            else set()
        )

        arch_col = "model_architecture" if "model_architecture" in self.df.columns else None

        fig = plt.figure(figsize=(8, 6.5))
        ax  = fig.add_subplot(111, projection="3d")

        x_col, y_col, z_col = "test_iou", "test_boundary_f1", "test_compactness"
        archs = sorted(self.df[arch_col].dropna().unique(), key=str) if arch_col else []

        if archs:
            for arch in archs:
                color    = COLORS.get(arch, "#888888")
                a_mask   = self.df[arch_col] == arch
                dom_mask = a_mask & ~self.df["point_id"].isin(front_ids)
                frt_mask = a_mask & self.df["point_id"].isin(front_ids)
                if dom_mask.any():
                    ax.scatter(
                        self.df.loc[dom_mask, x_col],
                        self.df.loc[dom_mask, y_col],
                        self.df.loc[dom_mask, z_col],
                        c=color, s=22, alpha=0.30, zorder=2,
                    )
                if frt_mask.any():
                    ax.scatter(
                        self.df.loc[frt_mask, x_col],
                        self.df.loc[frt_mask, y_col],
                        self.df.loc[frt_mask, z_col],
                        c=color, s=100, alpha=0.95,
                        edgecolors="black", linewidths=0.6, zorder=4,
                        label=arch,
                    )
        else:
            dom_mask = ~self.df["point_id"].isin(front_ids)
            ax.scatter(
                self.df.loc[dom_mask, x_col],
                self.df.loc[dom_mask, y_col],
                self.df.loc[dom_mask, z_col],
                c="#cccccc", s=22, alpha=0.35,
            )
            if not pareto_df.empty:
                ax.scatter(
                    pareto_df[x_col], pareto_df[y_col], pareto_df[z_col],
                    c="#e74c3c", s=100, alpha=0.95,
                    edgecolors="black", linewidths=0.6, label="Pareto front",
                )

        ax.set_xlabel(METRIC_LABELS.get(x_col, x_col), fontsize=9, labelpad=6)
        ax.set_ylabel(METRIC_LABELS.get(y_col, y_col), fontsize=9, labelpad=6)
        ax.set_zlabel(METRIC_LABELS.get(z_col, z_col), fontsize=9, labelpad=6)
        ax.set_title(
            "3-Objective Trade-off Space\n"
            r"(IoU $\uparrow$, Boundary F1 $\uparrow$, Compactness $\downarrow$)",
            fontsize=10, pad=10,
        )
        if archs:
            ax.legend(fontsize=7, loc="upper left", title="Architecture", title_fontsize=7)
        else:
            ax.legend(fontsize=8)

        plt.tight_layout()
        dest = self.out_dir / "figures" / "pareto_3d"
        save_figure(fig, str(dest), formats=("png", "pdf"))
        plt.close(fig)
        logger.info(f"Saved 3D Pareto figure → {dest}.png")
        return dest.with_suffix(".png")

    def save_pareto_per_architecture_table(self, pareto_df: Optional[pd.DataFrame] = None) -> Optional[Path]:
        """LaTeX table showing how often each architecture reaches the Pareto front.

        Reports per-architecture: number of configs, best value on each objective,
        and number / fraction of 3-objective Pareto-optimal configs.  This is a
        key summary table for newcomers wanting to select an architecture.
        """
        if not self._require_data():
            return None
        if pareto_df is None or pareto_df.empty:
            pareto_df, _, _ = self.compute_and_save_pareto_front()

        arch_col = "model_architecture"
        if arch_col not in self.df.columns:
            return None

        avail_obj = [m for m in PARETO_METRICS if m in self.df.columns]

        rows = []
        for arch, group in self.df.groupby(arch_col):
            n_total = len(group)
            pareto_count = (
                group["point_id"].isin(pareto_df["point_id"]).sum()
                if pareto_df is not None and "point_id" in pareto_df.columns
                else 0
            )
            rec: Dict[str, Any] = {
                "Architecture": arch,
                "# Configs": n_total,
                "# Pareto": int(pareto_count),
                "Pareto %": f"{100*pareto_count/n_total:.0f}\\%",
            }
            for m in avail_obj:
                if m not in group.columns:
                    continue
                vals = group[m].dropna()
                if m == "test_compactness":
                    rec[f"Best {METRIC_LABELS[m]}"] = f"{vals.min():.4f}" if len(vals) else "--"
                else:
                    rec[f"Best {METRIC_LABELS[m]}"] = f"{vals.max():.4f}" if len(vals) else "--"
                rec[f"Mean {METRIC_LABELS[m]}"] = f"{vals.mean():.4f}" if len(vals) else "--"
            rows.append(rec)

        table_df = pd.DataFrame(rows).sort_values("# Pareto", ascending=False)
        out = self.out_dir / "tables" / "pareto_per_architecture.tex"
        dataframe_to_latex(
            table_df,
            caption=(
                "Per-architecture summary of Pareto optimality. "
                "\\# Pareto is the number of configurations that are non-dominated "
                "under the three objectives (IoU $\\uparrow$, Boundary F1 $\\uparrow$, "
                "Compactness $\\downarrow$) jointly."
            ),
            label="tab:pareto_per_arch",
            highlight_best=False,
            save_path=str(out),
        )
        # Also write a plain CSV for inspection
        csv_out = self.out_dir / "tables" / "pareto_per_architecture.csv"
        table_df.to_csv(csv_out, index=False)
        logger.info(f"Saved per-architecture Pareto table → {out}")
        return out

    # ------------------------------------------------------------------ figures (non-Pareto)

    def save_metric_comparison_plots(self) -> List[Path]:
        """Box plots of test IoU grouped by each of the 6 hyperparameters."""
        if not self._require_data():
            return []
        apply_journal_style()
        dpi = 300
        out_paths: List[Path] = []
        primary = "test_iou" if "test_iou" in self.df.columns else ALL_TEST_METRICS[0]

        for hp in HYPERPARAM_LABELS:
            if hp not in self.df.columns or primary not in self.df.columns:
                continue

            groups = sorted(self.df[hp].dropna().unique(), key=str)
            data_by_group = [self.df.loc[self.df[hp] == g, primary].dropna().values for g in groups]
            group_labels = [str(g) for g in groups]

            fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.2 + 1.5), 4.5))
            bp = ax.boxplot(data_by_group, labels=group_labels, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5))
            palette = list(COLORS.values()) if COLORS else None
            for i, patch in enumerate(bp["boxes"]):
                if palette:
                    patch.set_facecolor(palette[i % len(palette)])
                    patch.set_alpha(0.75)
            ax.set_xlabel(HYPERPARAM_LABELS[hp])
            ax.set_ylabel(METRIC_LABELS.get(primary, primary))
            ax.set_title(f"Effect of {HYPERPARAM_LABELS[hp]} on {METRIC_LABELS.get(primary, primary)}")
            if len(max(group_labels, key=len)) > 6:
                plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            stem = f"comparison_by_{hp}"
            dest = self.out_dir / "figures" / stem
            save_figure(fig, str(dest), formats=("png", "pdf"))
            out_paths.append(dest.with_suffix(".png"))
            plt.close(fig)

        logger.info(f"Saved {len(out_paths)} hyperparameter comparison plots")
        return out_paths

    def save_prediction_comparison_figure(
        self,
        top_n: int = 5,
        n_samples_per_config: int = 3,
    ) -> Optional[Path]:
        """Build a paper-ready prediction grid from saved per-point PNG files.

        Rows = top-N configurations (ranked by test IoU).
        Columns = up to n_samples_per_config prediction samples.
        Each cell = the 4-panel image saved by ``_save_prediction_samples``.
        """
        if not self._require_data():
            return None

        rank_by = "test_iou" if "test_iou" in self.df.columns else ALL_TEST_METRICS[0]
        top_rows = self.df.nlargest(top_n, rank_by)

        # Collect available prediction PNGs
        config_images: List[Tuple[str, List[Path]]] = []
        for _, row in top_rows.iterrows():
            rdir = Path(str(row.get("result_dir", "")))
            pred_dir = rdir / "predictions"
            if not pred_dir.exists():
                continue
            pngs = sorted(pred_dir.glob("sample_*.png"))[:n_samples_per_config]
            if not pngs:
                continue
            arch    = row.get("model_architecture", "?")
            depth   = row.get("encoder_filters", "?")
            loss    = row.get("pixel_loss_type", "?")
            b_w     = row.get("boundary_loss_weight", 0.0)
            s_w     = row.get("shape_loss_weight",    0.0)
            iou_val = row.get(rank_by, float("nan"))
            label   = (
                f"#{int(row['point_id'])}: {arch}/{depth}\n"
                f"loss={loss} b={b_w} s={s_w}\nIoU={iou_val:.4f}"
            )
            config_images.append((label, pngs))

        if not config_images:
            logger.warning("No prediction images found; skipping prediction grid figure")
            return None

        n_rows = len(config_images)
        n_cols = max(len(imgs) for _, imgs in config_images)
        apply_journal_style()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.8))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        for r, (label, pngs) in enumerate(config_images):
            for c in range(n_cols):
                ax = axes[r][c]
                if c < len(pngs):
                    img = plt.imread(str(pngs[c]))
                    ax.imshow(img)
                    ax.axis("off")
                    if c == 0:
                        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=80,
                                      va="center", ha="right")
                else:
                    ax.axis("off")

        plt.suptitle(
            f"Prediction examples for the top-{n_rows} configurations "
            "(RGB | Ground Truth | Prediction | TP/FP/FN)",
            fontsize=9, y=1.01,
        )
        plt.tight_layout()
        dest = self.out_dir / "figures" / "predictions_comparison_grid"
        save_figure(fig, str(dest), formats=("png",))
        plt.close(fig)
        logger.info(f"Saved prediction comparison grid → {dest}.png")
        return dest.with_suffix(".png")

    # ------------------------------------------------------------------ diagnostics

    def save_diagnostic_plots(self) -> List[Path]:
        """Metric distribution histograms and Pearson correlation matrix."""
        if not self._require_data():
            return []
        apply_journal_style()
        out_paths: List[Path] = []
        avail = [m for m in ALL_TEST_METRICS if m in self.df.columns]

        # ---- Histograms ----
        n = len(avail)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, metric in enumerate(avail):
            ax = axes_flat[i]
            vals = self.df[metric].dropna().values
            ax.hist(vals, bins=min(15, max(5, len(vals) // 3)), edgecolor="white", linewidth=0.5,
                    color=COLORS.get("unet", "#3498db") if COLORS else "#3498db", alpha=0.8)
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Count",  fontsize=8)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")
        plt.suptitle("Distribution of Evaluation Metrics Across All Configurations", fontsize=10)
        plt.tight_layout()
        hist_path = self.out_dir / "diagnostics" / "metric_distributions"
        save_figure(fig, str(hist_path), formats=("png",))
        out_paths.append(hist_path.with_suffix(".png"))
        plt.close(fig)

        # ---- Correlation matrix ----
        if len(avail) >= 3:
            corr = self.df[avail].corr()
            fig, ax = plt.subplots(figsize=(len(avail) * 0.9 + 1, len(avail) * 0.9))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            labels = [METRIC_LABELS.get(m, m) for m in avail]
            ax.set_xticks(range(len(avail))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(avail))); ax.set_yticklabels(labels, fontsize=8)
            for ii in range(len(avail)):
                for jj in range(len(avail)):
                    ax.text(jj, ii, f"{corr.iloc[ii, jj]:.2f}", ha="center", va="center", fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Pearson Correlation of Evaluation Metrics")
            plt.tight_layout()
            corr_path = self.out_dir / "diagnostics" / "correlation_matrix"
            save_figure(fig, str(corr_path), formats=("png",))
            out_paths.append(corr_path.with_suffix(".png"))
            plt.close(fig)

        logger.info(f"Saved {len(out_paths)} diagnostic plots")
        return out_paths

    def save_summary_statistics(self) -> Path:
        """Persist mean/std/min/max per metric as JSON for the paper's text."""
        avail = [m for m in ALL_TEST_METRICS if m in self.df.columns]
        stats: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(),
            "n_completed":  len(self.df),
            "metrics": {},
        }
        for m in avail:
            vals = self.df[m].dropna()
            stats["metrics"][m] = {
                "mean": round(float(vals.mean()), 6),
                "std":  round(float(vals.std()),  6),
                "min":  round(float(vals.min()),  6),
                "max":  round(float(vals.max()),  6),
                "median": round(float(vals.median()), 6),
            }
        # Per-hyperparameter group stats for best metric
        primary = "test_iou" if "test_iou" in self.df.columns else (avail[0] if avail else None)
        if primary:
            stats["by_hyperparameter"] = {}
            for hp in HYPERPARAM_LABELS:
                if hp not in self.df.columns:
                    continue
                grp_stats: Dict[str, Any] = {}
                for g, sub in self.df.groupby(hp):
                    grp_stats[str(g)] = {
                        "mean": round(float(sub[primary].mean()), 6),
                        "std":  round(float(sub[primary].std()),  6),
                        "n":    len(sub),
                    }
                stats["by_hyperparameter"][hp] = grp_stats

        out = self.out_dir / "diagnostics" / "summary_statistics.json"
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info(f"Saved summary statistics → {out}")
        return out

    # ------------------------------------------------------------------ orchestrator

    def generate_full_paper_report(self) -> Dict[str, Any]:
        """Run all generators and return a manifest of output paths."""
        if not self._require_data():
            return {}

        manifest: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(),
            "n_configurations": len(self.df),
        }

        # ---- Tables ----
        manifest["tables"] = {}
        manifest["tables"]["csv_all"]         = str(self.save_all_results_csv())
        manifest["tables"]["latex_main"]      = str(self.save_latex_main_results_table(top_n=15))
        manifest["tables"]["latex_ablation"]  = str(self.save_latex_hyperparameter_ablation())
        pareto_df, csv_p, tex_p = self.compute_and_save_pareto_front()
        manifest["tables"]["pareto_csv"]      = str(csv_p) if csv_p else None
        manifest["tables"]["pareto_tex"]      = str(tex_p) if tex_p else None
        manifest["n_pareto_points"]           = len(pareto_df)

        # ---- Figures ----
        manifest["figures"] = {}
        pareto_plots   = self.save_pareto_scatter_plots()
        combined_fig   = self.save_pareto_combined_figure()
        pareto_3d      = self.save_pareto_3d_figure(pareto_df)
        metric_plots   = self.save_metric_comparison_plots()
        pred_grid      = self.save_prediction_comparison_figure()
        manifest["figures"]["pareto_scatter"]   = [str(p) for p in pareto_plots]
        manifest["figures"]["pareto_combined"]  = str(combined_fig) if combined_fig else None
        manifest["figures"]["pareto_3d"]        = str(pareto_3d) if pareto_3d else None
        manifest["figures"]["metric_boxplots"]  = [str(p) for p in metric_plots]
        manifest["figures"]["prediction_grid"]  = str(pred_grid) if pred_grid else None

        # ---- Per-architecture Pareto table ----
        arch_pareto_tex = self.save_pareto_per_architecture_table(pareto_df)
        manifest["tables"]["pareto_per_arch_tex"] = str(arch_pareto_tex) if arch_pareto_tex else None

        # ---- Diagnostics ----
        manifest["diagnostics"] = {}
        diag_plots = self.save_diagnostic_plots()
        stat_file  = self.save_summary_statistics()
        manifest["diagnostics"]["plots"]              = [str(p) for p in diag_plots]
        manifest["diagnostics"]["summary_statistics"] = str(stat_file)

        # ---- Write manifest ----
        mf_path = self.out_dir / "report_manifest.json"
        mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Paper report complete. Manifest → {mf_path}")
        return manifest

