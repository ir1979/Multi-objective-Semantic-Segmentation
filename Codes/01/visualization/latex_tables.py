"""LaTeX table helpers for publication-ready exports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    highlight_best: bool = True,
    highlight_col_direction: Dict[str, str] | None = None,
    save_path: str | None = "tables/metrics.tex",
) -> str:
    """Convert a dataframe into a Booktabs-style LaTeX table."""
    working = df.copy()
    highlight_col_direction = highlight_col_direction or {}

    if highlight_best and not working.empty:
        working = working.astype(object)
        for column, direction in highlight_col_direction.items():
            if column not in working.columns:
                continue
            if direction == "max":
                best_index = working[column].astype(float).idxmax()
            else:
                best_index = working[column].astype(float).idxmin()
            working.loc[best_index, column] = f"\\textbf{{{working.loc[best_index, column]}}}"

    latex = working.to_latex(index=False, escape=False, caption=caption, label=label, bold_rows=False)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex, encoding="utf-8")
    return latex


def generate_all_paper_tables(results_registry: dict, save_dir: str) -> None:
    """Generate placeholder paper tables from experiment registry records."""
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, payload in results_registry.items():
        rows.append(
            {
                "experiment_name": name,
                "status": payload.get("status"),
                "test_iou": payload.get("test_iou", 0.0),
                "results_path": payload.get("results_path", ""),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(save_root / "strategy_comparison.csv", index=False)
    dataframe_to_latex(
        df,
        caption="Strategy Comparison",
        label="tab:strategy_comparison",
        highlight_best=True,
        highlight_col_direction={"test_iou": "max"},
        save_path=str(save_root / "strategy_comparison.tex"),
    )
