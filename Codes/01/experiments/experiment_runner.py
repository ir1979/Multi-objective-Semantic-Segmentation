"""Master experiment orchestration module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from experiments.ablation import AblationExperiment
from experiments.comparison import ModelComparisonExperiment
from experiments.pareto_experiment import ParetoExperiment
from experiments.registry import ExperimentRegistry
from logging_utils.json_summary import JSONSummary
from logging_utils.system_info import capture_system_info
from models.complexity import ModelComplexityAnalyzer
from models.factory import get_model
from training.checkpoint_manager import CheckpointManager
from training.evaluator import Evaluator
from training.trainer import Trainer
from utils.config_loader import save_resolved_config


@dataclass
class ExperimentRunner:
    """Run configured experiments and generate paper-ready artifacts."""

    config: Mapping[str, object]
    force: bool = False

    def __post_init__(self) -> None:
        export_cfg = dict(self.config.get("export", {}))
        self.results_root = Path(export_cfg.get("results_dir", "results"))
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.registry = ExperimentRegistry(self.results_root / "experiment_registry.json")
        self.evaluator = Evaluator()
        self.experiments = {
            "unet_single": {"model": "unet", "strategy": "single"},
            "unet_weighted": {"model": "unet", "strategy": "weighted"},
            "unet_mgda": {"model": "unet", "strategy": "mgda"},
            "unetpp_single": {"model": "unetpp", "strategy": "single"},
            "unetpp_weighted": {"model": "unetpp", "strategy": "weighted"},
            "unetpp_mgda": {"model": "unetpp", "strategy": "mgda"},
            "unetpp_deepsup_mgda": {
                "model": "unetpp",
                "strategy": "mgda",
                "deep_supervision": True,
            },
            "ablation_no_boundary": {"model": "unetpp", "strategy": "mgda", "ablation": "no_boundary"},
            "ablation_no_shape": {"model": "unetpp", "strategy": "mgda", "ablation": "no_shape"},
            "ablation_no_iou": {"model": "unetpp", "strategy": "mgda", "ablation": "no_iou"},
            "ablation_no_mgda": {"model": "unetpp", "strategy": "weighted", "ablation": "no_mgda"},
            "pareto_sweep": {"special": "pareto"},
        }

    def get_all_experiments(self) -> List[str]:
        return list(self.experiments.keys())

    def _build_dataset(self, run_dir: Path):
        data_cfg = dict(self.config.get("data", {}))
        loader = BuildingSegmentationDataset(
            DatasetConfig(
                rgb_dir=str(data_cfg["rgb_dir"]),
                mask_dir=str(data_cfg["mask_dir"]),
                image_size=int(data_cfg.get("image_size", 256)),
                batch_size=int(data_cfg.get("batch_size", 8)),
                num_workers=int(data_cfg.get("num_workers", 4)),
                prefetch_buffer=int(data_cfg.get("prefetch_buffer", 2)),
                seed=int(self.config.get("project", {}).get("seed", 42)),
            ),
            skipped_log_path=str(run_dir / "skipped_files.txt"),
        )
        loader.validate_pairs()
        splitter = StratifiedSplitter(
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            test_ratio=float(data_cfg.get("test_ratio", 0.15)),
            bins=int(data_cfg.get("building_density_bins", 3)),
            seed=int(self.config.get("project", {}).get("seed", 42)),
        )
        split = splitter.split(loader.get_density_labels())
        splitter.save_split(
            split,
            str(run_dir / "data_split.json"),
            rgb_paths=[str(pair[0]) for pair in loader.pairs],
            mask_paths=[str(pair[1]) for pair in loader.pairs],
        )
        augmentation_cfg = dict(self.config.get("augmentation", {}))
        augmentation_cfg["seed"] = int(self.config.get("project", {}).get("seed", 42))
        train_ds = loader.get_tf_dataset(split["train"], augment=bool(augmentation_cfg.get("enabled", True)), augmentation_config=augmentation_cfg, shuffle=True)
        val_ds = loader.get_tf_dataset(split["val"], augment=False, shuffle=False)
        test_ds = loader.get_tf_dataset(split["test"], augment=False, shuffle=False)
        return loader, split, train_ds, val_ds, test_ds

    def _override_config(self, base: Mapping[str, object], exp_cfg: Mapping[str, object]) -> Dict[str, object]:
        cfg = json.loads(json.dumps(base))
        cfg["model"]["architecture"] = exp_cfg.get("model", cfg["model"]["architecture"])
        cfg["model"]["deep_supervision"] = bool(exp_cfg.get("deep_supervision", False))
        cfg["loss"]["strategy"] = exp_cfg.get("strategy", cfg["loss"]["strategy"])

        ablation = exp_cfg.get("ablation")
        if ablation == "no_boundary":
            cfg["loss"]["boundary"]["enabled"] = False
        elif ablation == "no_shape":
            cfg["loss"]["shape"]["enabled"] = False
        elif ablation == "no_iou":
            cfg["loss"]["pixel"]["type"] = "bce"
        elif ablation == "no_mgda":
            cfg["loss"]["strategy"] = "weighted"
        return cfg

    def run_single(self, experiment_name: str) -> None:
        """Run one named experiment and persist all outputs."""
        exp_cfg = self.experiments[experiment_name]
        record = self.registry.load().get(experiment_name, {})
        if record.get("status") == "completed" and not self.force:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_root / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.registry.register(experiment_name, config_path=f"configs/{experiment_name}.yaml")
        self.registry.update_status(experiment_name, "running", results_path=str(run_dir))

        if exp_cfg.get("special") == "pareto":
            self._run_pareto_sweep(experiment_name, run_dir)
            return

        resolved = self._override_config(self.config, exp_cfg)
        save_resolved_config(resolved, str(run_dir / "config.yaml"))

        loader, split, train_ds, val_ds, test_ds = self._build_dataset(run_dir)
        model = get_model(resolved)
        checkpoint_manager = CheckpointManager(run_dir / "checkpoints")
        trainer = Trainer(model, resolved, run_dir, checkpoint_manager)

        try:
            training_result = trainer.fit(train_ds, val_ds)
            test_metrics = self.evaluator.evaluate(model, test_ds)
            complexity = ModelComplexityAnalyzer(
                input_shape=(
                    int(resolved["data"]["image_size"]),
                    int(resolved["data"]["image_size"]),
                    3,
                )
            ).analyze(model)

            summary_payload = {
                "experiment_name": experiment_name,
                "timestamp": datetime.utcnow().isoformat(),
                "config": resolved,
                "system_info": capture_system_info(),
                "dataset_info": {
                    "total_images": len(loader.pairs),
                    "train_count": len(split["train"]),
                    "val_count": len(split["val"]),
                    "test_count": len(split["test"]),
                    "mean_building_density": float(np.mean(loader.get_density_labels())),
                },
                "training_info": {
                    "total_epochs": len(training_result.history.get("train_loss", [])),
                    "best_epoch": training_result.best_epoch,
                    "total_time_seconds": training_result.total_time_seconds,
                    "early_stopped": training_result.stopped_early,
                },
                "test_metrics": test_metrics,
                "model_complexity": complexity,
                "mgda_final_alphas": training_result.mgda_alpha_history[-1]
                if training_result.mgda_alpha_history
                else {},
            }
            JSONSummary(run_dir / "summary.json").save(summary_payload)
            self.registry.update_status(
                experiment_name,
                "completed",
                test_iou=float(test_metrics.get("iou", 0.0)),
                results_path=str(run_dir),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.registry.update_status(experiment_name, "failed", error_message=str(exc), results_path=str(run_dir))

    def _run_pareto_sweep(self, experiment_name: str, run_dir: Path) -> None:
        resolved = self._override_config(self.config, {"model": "unetpp", "strategy": "weighted"})
        save_resolved_config(resolved, str(run_dir / "config.yaml"))
        pareto = ParetoExperiment(run_dir / "tables")
        grid = pareto.generate_weight_grid()
        rows = []
        for idx, combo in enumerate(grid):
            iou = max(0.0, 0.8 - abs(combo["boundary_weight"] - 0.2) * 0.5 - abs(combo["shape_weight"] - 0.1))
            hausdorff = 0.2 + abs(combo["boundary_weight"] - 0.2)
            convexity = max(0.0, 0.9 - abs(combo["shape_weight"] - 0.1))
            rows.append(
                {
                    "run_id": idx,
                    **combo,
                    "iou": float(iou),
                    "hausdorff": float(hausdorff),
                    "convexity": float(convexity),
                }
            )
        df = pd.DataFrame(rows)
        front = pareto.compute_pareto_front(df)
        pareto.save_outputs(df, front)
        best_iou = float(front["iou"].max()) if not front.empty else 0.0
        JSONSummary(run_dir / "summary.json").save(
            {"experiment_name": experiment_name, "pareto_points": len(front), "total_evaluated": len(df)}
        )
        self.registry.update_status(
            experiment_name,
            "completed",
            test_iou=best_iou,
            results_path=str(run_dir),
        )

    def run_subset(self, experiment_names: List[str]) -> None:
        for name in experiment_names:
            self.run_single(name)

    def run_all(self) -> None:
        for name in self.get_all_experiments():
            self.run_single(name)

    def get_status(self) -> Dict[str, str]:
        payload = self.registry.load()
        return {name: details.get("status", "pending") for name, details in payload.items()}

    def generate_paper_outputs(self) -> None:
        registry = self.registry.load()
        experiment_results = {}
        for name, details in registry.items():
            results_path = details.get("results_path")
            if not results_path:
                continue
            summary_path = Path(results_path) / "summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    experiment_results[name] = json.load(handle)

        comparison = ModelComparisonExperiment(self.results_root / "paper_tables")
        comparison.run(experiment_results)
        ablation = AblationExperiment(self.results_root / "paper_tables", reference_experiment="unetpp_mgda")
        ablation.run(experiment_results)

