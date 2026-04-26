"""Visualization module tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from experiments.pareto_experiment import ParetoExperiment
from models.complexity import ModelComplexityAnalyzer
from models.unet import UNet
from models.unetpp import UNetPlusPlus
from visualization.boundary_overlay import generate_boundary_overlay
from visualization.complexity_plot import generate_complexity_plot
from visualization.error_maps import generate_error_maps
from visualization.latex_tables import dataframe_to_latex
from visualization.loss_curves import generate_loss_curves
from visualization.pareto_plot import generate_pareto_2d, generate_pareto_3d
from visualization.prediction_grid import generate_prediction_grid


def _write_rgb(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_mask(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask = (rng.random((256, 256)) > 0.85).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


class TestVisualization(unittest.TestCase):
    """Ensure visualization utilities generate expected artifacts."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rgb_dir = self.root / "RGB"
        mask_dir = self.root / "Mask"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(6):
            _write_rgb(rgb_dir / f"tile_{idx:03d}.png", idx)
            _write_mask(mask_dir / f"tile_{idx:03d}.tif", idx)

        loader = BuildingSegmentationDataset(
            DatasetConfig(
                rgb_dir=str(rgb_dir),
                mask_dir=str(mask_dir),
                image_size=256,
                batch_size=2,
                seed=42,
            ),
            skipped_log_path=str(self.root / "skipped.txt"),
        )
        loader.validate_pairs()
        split = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=42).split(loader.get_density_labels())
        self.dataset = loader.get_tf_dataset(split["test"] or split["val"] or split["train"], augment=False)
        self.models = {"unet": UNet()}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_prediction_grid_saves(self) -> None:
        generate_prediction_grid(self.models, self.dataset, num_samples=2, save_path=str(self.root / "pred_grid"))
        self.assertTrue((self.root / "pred_grid.png").exists())
        self.assertTrue((self.root / "pred_grid.pdf").exists())

    def test_boundary_overlay_saves(self) -> None:
        generate_boundary_overlay(self.models, self.dataset, num_samples=2, save_path=str(self.root / "boundary"))
        self.assertTrue((self.root / "boundary.png").exists())

    def test_error_maps_saves(self) -> None:
        generate_error_maps(self.models, self.dataset, num_samples=2, save_path=str(self.root / "errors"))
        self.assertTrue((self.root / "errors.png").exists())

    def test_pareto_plot_saves(self) -> None:
        results = pd.DataFrame(
            {
                "obj_iou": [0.2, 0.1, 0.3],
                "obj_boundary": [0.4, 0.5, 0.2],
                "obj_convexity": [0.3, 0.2, 0.4],
            }
        )
        generate_pareto_2d(results, results.iloc[[0, 1]], "obj_iou", "obj_boundary", save_path=str(self.root / "pareto2d"))
        generate_pareto_3d(
            results,
            results.iloc[[0, 1]],
            "obj_iou",
            "obj_boundary",
            "obj_convexity",
            save_path=str(self.root / "pareto3d"),
        )
        self.assertTrue((self.root / "pareto2d.png").exists())
        self.assertTrue((self.root / "pareto3d.png").exists())

    def test_loss_curves_saves(self) -> None:
        csv_path = self.root / "logs.csv"
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.6],
                "val_loss": [1.1, 0.9, 0.7],
            }
        ).to_csv(csv_path, index=False)
        generate_loss_curves({"exp": str(csv_path)}, save_path=str(self.root / "loss_curves"))
        self.assertTrue((self.root / "loss_curves.png").exists())

    def test_complexity_plot_saves(self) -> None:
        df = pd.DataFrame(
            {
                "experiment_name": ["a", "b"],
                "flops": [1e9, 2e9],
                "iou": [0.6, 0.7],
                "total_params": [1e6, 2e6],
                "strategy": ["unet_single", "unet_mgda"],
            }
        )
        generate_complexity_plot(df, save_path=str(self.root / "complexity"))
        self.assertTrue((self.root / "complexity.png").exists())

    def test_complexity_analyzer_handles_subclassed_models(self) -> None:
        analyzer = ModelComplexityAnalyzer(input_shape=(256, 256, 3), num_runs=1, num_warmup=1)
        metrics = analyzer.analyze(UNetPlusPlus(deep_supervision=True))
        self.assertIn("flops", metrics)
        self.assertGreaterEqual(metrics["total_params"], 0)

    def test_latex_table_valid(self) -> None:
        df = pd.DataFrame({"Model": ["A"], "IoU": [0.9]})
        latex = dataframe_to_latex(
            df,
            caption="Test",
            label="tab:test",
            save_path=str(self.root / "table.tex"),
        )
        self.assertIn("\\begin{tabular}", latex)
        self.assertTrue((self.root / "table.tex").exists())

    def test_pareto_experiment_save_outputs_uses_objective_columns(self) -> None:
        experiment = ParetoExperiment(self.root / "pareto_tables")
        results = pd.DataFrame(
            {
                "pixel_weight": [0.6, 0.8],
                "boundary_weight": [0.2, 0.1],
                "shape_weight": [0.1, 0.2],
                "iou": [0.75, 0.70],
                "hausdorff": [0.2, 0.25],
                "convexity": [0.9, 0.85],
            }
        )
        front = experiment.compute_pareto_front(results)
        experiment.save_outputs(results, front)
        saved = pd.read_csv(self.root / "pareto_tables" / "pareto_points.csv")
        self.assertIn("obj_iou", saved.columns)
        self.assertTrue((self.root / "pareto_tables" / "pareto_2d.png").exists())
