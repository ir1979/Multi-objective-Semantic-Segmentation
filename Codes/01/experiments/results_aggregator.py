"""Results aggregator for grid search analysis and paper generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GridSearchResultsAggregator:
    """Aggregate and analyze grid search results."""

    def __init__(self, state_file: Path, results_dir: Path) -> None:
        self.state_file = Path(state_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_df: Optional[pd.DataFrame] = None
        self._load_results()

    def _load_results(self) -> None:
        """Load results from state file."""
        if not self.state_file.exists():
            self.results_df = pd.DataFrame()
            return

        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for point_id, point_data in data.items():
            if point_data["status"] == "completed":
                row = {"point_id": int(point_id)}
                row.update(point_data.get("params", {}))
                row.update(point_data.get("metrics", {}))
                records.append(row)

        self.results_df = pd.DataFrame(records)

    def save_csv_report(self) -> Path:
        """Save results as CSV."""
        if self.results_df.empty:
            return None

        csv_file = self.results_dir / "grid_search_results.csv"
        self.results_df.to_csv(csv_file, index=False)
        return csv_file

    def save_latex_table(self, top_n: int = 10, metrics: Optional[List[str]] = None) -> Path:
        """Generate LaTeX table of top performing configurations."""
        if self.results_df.empty:
            return None

        if metrics is None:
            metrics = [col for col in self.results_df.columns if col.endswith("_iou") or col.endswith("_loss")]

        # Select top N by primary metric
        if metrics:
            primary_metric = metrics[0]
            top_results = self.results_df.nlargest(top_n, primary_metric)
        else:
            top_results = self.results_df.head(top_n)

        # Format for LaTeX
        latex_cols = ["point_id", "model_architecture", "loss_strategy"] + metrics
        latex_df = top_results[[col for col in latex_cols if col in top_results.columns]].copy()

        # Round numeric columns
        for col in latex_df.select_dtypes(include=[np.number]).columns:
            if col not in ["point_id"]:
                latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

        latex_str = latex_df.to_latex(index=False, caption="Top 10 Grid Search Results", label="tab:grid_results")

        latex_file = self.results_dir / "grid_search_table.tex"
        latex_file.write_text(latex_str, encoding="utf-8")
        return latex_file

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics per configuration category."""
        if self.results_df.empty:
            return {}

        summary = {}

        # By model architecture
        for col in ["model_architecture", "loss_strategy"]:
            if col in self.results_df.columns:
                grouped = self.results_df.groupby(col)
                metrics = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
                summary[col] = grouped[metrics].mean().to_dict()

        return summary

    def save_summary_statistics(self) -> Path:
        """Save summary statistics."""
        summary = self.get_summary_statistics()
        summary_file = self.results_dir / "summary_statistics.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary_file

    def generate_comparison_plots(self) -> List[Path]:
        """Generate comparison plots for paper."""
        if self.results_df.empty or not MATPLOTLIB_AVAILABLE:
            return []

        plots = []
        plt.style.use("seaborn-v0_8-darkgrid")

        # Plot 1: Performance by model architecture
        if "model_architecture" in self.results_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_cols = [col for col in self.results_df.columns if col.endswith("_iou")]
            if metric_cols:
                self.results_df.groupby("model_architecture")[metric_cols].mean().plot(kind="bar", ax=ax)
                ax.set_title("Performance by Model Architecture")
                ax.set_ylabel("Metric Value")
                ax.legend(title="Metrics")
                plt.tight_layout()
                plot_file = self.results_dir / "comparison_by_architecture.png"
                fig.savefig(plot_file, dpi=300)
                plots.append(plot_file)
                plt.close()

        # Plot 2: Performance by strategy
        if "loss_strategy" in self.results_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_cols = [col for col in self.results_df.columns if col.endswith("_iou")]
            if metric_cols:
                self.results_df.groupby("loss_strategy")[metric_cols].mean().plot(kind="bar", ax=ax)
                ax.set_title("Performance by Loss Strategy")
                ax.set_ylabel("Metric Value")
                ax.legend(title="Metrics")
                plt.tight_layout()
                plot_file = self.results_dir / "comparison_by_strategy.png"
                fig.savefig(plot_file, dpi=300)
                plots.append(plot_file)
                plt.close()

        # Plot 3: Distribution of results
        if "val_iou" in self.results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(self.results_df["val_iou"], bins=20, edgecolor="black", alpha=0.7)
            axes[0].set_title("Distribution of Validation IoU")
            axes[0].set_xlabel("IoU")
            axes[0].set_ylabel("Frequency")

            axes[1].scatter(range(len(self.results_df)), sorted(self.results_df["val_iou"]), alpha=0.6)
            axes[1].set_title("Sorted Validation IoU Scores")
            axes[1].set_xlabel("Configuration Index (sorted)")
            axes[1].set_ylabel("IoU")

            plt.tight_layout()
            plot_file = self.results_dir / "distribution_results.png"
            fig.savefig(plot_file, dpi=300)
            plots.append(plot_file)
            plt.close()

        return plots

    def generate_heatmap(self) -> Optional[Path]:
        """Generate heatmap of results if 2D parameter space."""
        if self.results_df.empty or not MATPLOTLIB_AVAILABLE:
            return None

        categorical_cols = self.results_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = [col for col in self.results_df.columns if col.endswith("_iou")]

        if len(categorical_cols) >= 2 and numeric_cols:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(6 * len(numeric_cols), 5))
            if len(numeric_cols) == 1:
                axes = [axes]

            for idx, metric in enumerate(numeric_cols):
                try:
                    pivot = self.results_df.pivot_table(
                        values=metric, index=categorical_cols[0], columns=categorical_cols[1], aggfunc="mean"
                    )
                    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axes[idx], cbar_kws={"label": metric})
                    axes[idx].set_title(f"{metric} by {categorical_cols[0]} and {categorical_cols[1]}")
                except Exception:
                    pass

            plt.tight_layout()
            plot_file = self.results_dir / "heatmap_results.png"
            fig.savefig(plot_file, dpi=300)
            plots.append(plot_file)
            plt.close()
            return plot_file

        return None

    def get_best_configurations(self, n: int = 5) -> pd.DataFrame:
        """Get top N configurations by validation metric."""
        if self.results_df.empty:
            return pd.DataFrame()

        if "val_iou" in self.results_df.columns:
            return self.results_df.nlargest(n, "val_iou")
        elif "test_iou" in self.results_df.columns:
            return self.results_df.nlargest(n, "test_iou")
        else:
            return self.results_df.head(n)

    def generate_full_report(self) -> Dict[str, Path]:
        """Generate complete analysis report."""
        report = {}

        # CSV report
        csv_file = self.save_csv_report()
        if csv_file:
            report["csv"] = csv_file

        # LaTeX table
        latex_file = self.save_latex_table()
        if latex_file:
            report["latex_table"] = latex_file

        # Summary statistics
        summary_file = self.save_summary_statistics()
        if summary_file:
            report["summary_statistics"] = summary_file

        # Plots
        plots = self.generate_comparison_plots()
        for i, plot in enumerate(plots):
            report[f"plot_{i}"] = plot

        # Heatmap
        heatmap = self.generate_heatmap()
        if heatmap:
            report["heatmap"] = heatmap

        # Best configurations
        best_configs = self.get_best_configurations()
        if not best_configs.empty:
            best_file = self.results_dir / "best_configurations.csv"
            best_configs.to_csv(best_file, index=False)
            report["best_configurations"] = best_file

        return report
