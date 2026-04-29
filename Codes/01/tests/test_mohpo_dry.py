"""Dry tests for the upgraded MOHPO framework.

These tests avoid full training runs and instead verify the framework's control
plane: strategy selection, config validation, state persistence, Pareto/report
artifact generation, and resumable experiment bookkeeping.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.grid_search import GridPoint, GridSearchState
from experiments.results_aggregator import GridSearchResultsAggregator
from optimization.search_strategy import get_search_strategy
from utils.config_validator import validate_config_file


class TestMOHPODry(unittest.TestCase):
    """Fast dry tests for architecture-level MOHPO behaviour."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_grid_search_strategy_is_explicit_method(self) -> None:
        strategy = get_search_strategy("grid_search")
        points = [{"a": 1}, {"a": 2}]
        self.assertEqual(strategy.select(points, {}), points)

    def test_sample_config_with_grid_search_validates(self) -> None:
        cfg = self.root / "cfg.yaml"
        cfg.write_text(
            """
project:
  name: sample
model:
  architecture: unet
loss:
  strategy: weighted
  pixel:
    type: bce
training:
  epochs: 1
  lr_scheduler:
    type: cosine
evaluation: {}
export: {}
data:
  rgb_dir: RGB
  mask_dir: Mask
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
grid_search:
  enabled: true
  parameters:
    model_architecture: [unet, unetpp]
    pixel_loss_type: [bce]
    boundary_loss_weight: [0.3]
    shape_loss_weight: [0.1]
    learning_rate: [0.001]
    encoder_filters:
      - [16, 32, 64, 128, 256]
  selection:
    strategy: grid_search
  persistence:
    backend: json
    checkpoint_interval: 1
  objectives:
    - metric: test_iou
      direction: max
    - metric: test_boundary_f1
      direction: max
    - metric: test_compactness
      direction: min
            """.strip(),
            encoding="utf-8",
        )
        ok, errors, _ = validate_config_file(str(cfg))
        self.assertTrue(ok, msg="; ".join(errors))

    def test_json_state_backend_round_trip(self) -> None:
        state = GridSearchState(self.root / "state.json", backend="json", checkpoint_interval=1)
        p = GridPoint(point_id=1, params={"learning_rate": 1e-3}, status="completed", metrics={"test_iou": 0.5})
        state.add_point(p)
        state.flush()

        reloaded = GridSearchState(self.root / "state.json", backend="json", checkpoint_interval=1)
        self.assertEqual(len(reloaded.points), 1)
        self.assertEqual(reloaded.get_point(1).metrics["test_iou"], 0.5)

    def test_sqlite_state_backend_round_trip(self) -> None:
        state = GridSearchState(self.root / "state.json", backend="sqlite", checkpoint_interval=1)
        p = GridPoint(point_id=2, params={"learning_rate": 5e-4}, status="failed", error_message="dry failure")
        state.add_point(p)
        state.flush()

        reloaded = GridSearchState(self.root / "state.json", backend="sqlite", checkpoint_interval=1)
        self.assertEqual(len(reloaded.points), 1)
        self.assertEqual(reloaded.get_point(2).error_message, "dry failure")

    def test_report_generation_creates_academic_exports(self) -> None:
        state_payload = {
            "0": {
                "point_id": 0,
                "params": {
                    "model_architecture": "unet",
                    "encoder_filters": [16, 32, 64, 128, 256],
                    "pixel_loss_type": "bce",
                    "boundary_loss_weight": 0.3,
                    "shape_loss_weight": 0.1,
                    "learning_rate": 0.001,
                },
                "status": "completed",
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:01:00",
                "error_message": None,
                "result_dir": str(self.root / "point_000000"),
                "metrics": {
                    "train_loss": 0.2,
                    "val_iou": 0.4,
                    "test_iou": 0.45,
                    "test_dice": 0.62,
                    "test_precision": 0.74,
                    "test_recall": 0.53,
                    "test_pixel_acc": 0.88,
                    "test_boundary_iou": 0.14,
                    "test_boundary_f1": 0.25,
                    "test_compactness": 0.0003,
                    "best_epoch": 5,
                    "total_epochs": 5,
                    "stopped_early": False,
                },
            }
        }
        state_file = self.root / "grid_search_state.json"
        state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        out_dir = self.root / "paper_outputs"
        agg = GridSearchResultsAggregator(state_file, out_dir)
        manifest = agg.generate_full_paper_report()

        self.assertIn("reports", manifest)
        self.assertTrue((out_dir / "academic_report.md").exists())
        self.assertTrue((out_dir / "report_citation.bib").exists())
        self.assertTrue((out_dir / "report_citation_metadata.json").exists())
        self.assertTrue((out_dir / "report_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
