#!/usr/bin/env python3
"""Smoke test for the updated grid-search pipeline.

Runs a single grid point (UNet, 1 epoch, BCE loss, no boundary/shape terms)
and verifies that:
  - all imports succeed
  - a GridPoint trains end-to-end without error
  - all 8 evaluation metrics are recorded
  - prediction PNG panels are saved
  - GridSearchResultsAggregator can load results and run generate_full_paper_report()
"""

from __future__ import annotations

import copy
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    if not ok:
        raise SystemExit(f"\nSmoke test FAILED at: {label}")


def main() -> None:
    t0 = time.time()
    banner("SMOKE TEST — grid search pipeline")

    # ------------------------------------------------------------------
    # 1. Imports
    # ------------------------------------------------------------------
    print("\n--- Step 1: imports ---")
    try:
        from experiments.grid_search import (
            GridPoint,
            GridSearchConfig,
            GridSearchRunner,
        )
        from experiments.results_aggregator import GridSearchResultsAggregator
        from models.factory import get_model
        from visualization.style import COLORS, apply_journal_style
        check("experiments.grid_search import", True)
        check("experiments.results_aggregator import", True)
        check("models.factory import", True)
        check("visualization.style import", True)
        check("COLORS has pareto keys", "pareto_front" in COLORS and "pareto_dominated" in COLORS)
        check("COLORS has no MGDA keys", "unet_mgda" not in COLORS and "mgda_point" not in COLORS)
    except Exception as exc:
        traceback.print_exc()
        check("imports", False, str(exc))

    # ------------------------------------------------------------------
    # 2. Grid config
    # ------------------------------------------------------------------
    print("\n--- Step 2: grid config ---")
    cfg_path = ROOT / "configs" / "grid_search.yaml"
    check("grid_search.yaml exists", cfg_path.exists())

    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    params = cfg.get("grid_search", {}).get("parameters", {})
    expected_params = {
        "model_architecture", "encoder_filters",
        "pixel_loss_type", "boundary_loss_weight", "shape_loss_weight", "learning_rate",
    }
    check("exactly 6 grid parameters", set(params.keys()) == expected_params,
          f"found: {set(params.keys())}")
    check("loss.strategy is 'weighted'",
          cfg.get("loss", {}).get("strategy") == "weighted")
    check("mgda.enabled is False",
          cfg.get("mgda", {}).get("enabled") is False)
    check("save_predictions flag present",
          cfg.get("export", {}).get("save_predictions") is not None)

    # Generate grid
    gs_cfg_obj = GridSearchConfig(cfg)
    all_points = gs_cfg_obj.generate_points()
    check("grid generates > 0 points", len(all_points) > 0, f"{len(all_points)} points")

    # Constraint: no point should have both b=0 and s=0 simultaneously
    bad = [p for p in all_points
           if p.get("boundary_loss_weight", 1) == 0.0 and p.get("shape_loss_weight", 1) == 0.0]
    check("constraint: no all-zero loss point", len(bad) == 0,
          f"{len(bad)} violating points" if bad else "")

    # ------------------------------------------------------------------
    # 3. End-to-end training (1 epoch)
    # ------------------------------------------------------------------
    print("\n--- Step 3: end-to-end training (1 epoch) ---")
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["training"]["epochs"] = 1
    smoke_cfg["training"].setdefault("early_stopping", {})["enabled"] = False
    smoke_cfg["data"]["batch_size"] = 2
    smoke_cfg["export"]["save_predictions"] = True
    smoke_cfg["export"]["n_prediction_samples"] = 2

    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_grid_"))
    tmp_cfg = tmp_dir / "smoke_config.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as fh:
        yaml.dump(smoke_cfg, fh, default_flow_style=False)

    test_point = GridPoint(
        point_id=0,
        params={
            "model_architecture":   "unet",
            "encoder_filters":      [32, 64, 128, 256, 512],
            "pixel_loss_type":      "bce",
            "boundary_loss_weight": 0.1,
            "shape_loss_weight":    0.0,
            "learning_rate":        1e-3,
        },
    )

    runner = GridSearchRunner(
        config_path=str(tmp_cfg),
        results_dir=str(tmp_dir),
        resume=False,
        max_retries=1,
    )
    runner.initialize_grid(force=True)
    runner.state.points = {0: test_point}
    runner.state.save()

    ok = runner.run_point(test_point)
    check("run_point returns True", ok, test_point.error_message or "")
    check("point status == completed", test_point.status == "completed")

    expected_metrics = [
        "test_iou", "test_dice", "test_precision", "test_recall",
        "test_pixel_acc", "test_boundary_iou", "test_boundary_f1", "test_compactness",
    ]
    for m in expected_metrics:
        check(f"metric recorded: {m}", m in test_point.metrics,
              f"recorded: {list(test_point.metrics.keys())}")

    pred_dir = Path(test_point.result_dir) / "predictions"
    pngs = list(pred_dir.glob("sample_*.png")) if pred_dir.exists() else []
    check("prediction PNGs saved", len(pngs) > 0, f"found {len(pngs)} PNGs")

    # ------------------------------------------------------------------
    # 4. GridSearchResultsAggregator
    # ------------------------------------------------------------------
    print("\n--- Step 4: results aggregator ---")
    state_file = tmp_dir / "grid_search_state.json"
    paper_dir  = tmp_dir / "paper_outputs"

    agg = GridSearchResultsAggregator(state_file, paper_dir)
    check("aggregator loaded data", not agg.df.empty, f"rows={len(agg.df)}")
    check("df has test_iou column", "test_iou" in agg.df.columns)

    try:
        manifest = agg.generate_full_paper_report()
        check("generate_full_paper_report runs", True)
        check("manifest is non-empty dict", isinstance(manifest, dict) and len(manifest) > 0)
        manifest_file = paper_dir / "report_manifest.json"
        check("report_manifest.json written", manifest_file.exists())
    except Exception as exc:
        traceback.print_exc()
        check("generate_full_paper_report runs", False, str(exc))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    banner(f"ALL CHECKS PASSED  —  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
