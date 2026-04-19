"""PyTorch experiment runner for building segmentation framework."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from data.integrity import compute_dataset_hash
from frameworks.pytorch.data.loader import DataConfig, PairScanner, build_dataloaders
from frameworks.pytorch.data.splitter import StratifiedSplitter
from frameworks.pytorch.models.complexity import ModelComplexityAnalyzer
from frameworks.pytorch.models.factory import get_model
from frameworks.pytorch.training.checkpoint_manager import CheckpointManager
from frameworks.pytorch.training.evaluator import Evaluator
from frameworks.pytorch.training.trainer import Trainer
from frameworks.pytorch.utils.reproducibility import set_global_seed
from utils.config_loader import load_config, save_resolved_config


def _to_float_dict(mapping: Mapping[str, object]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in mapping.items()}


@dataclass
class PyTorchExperimentRunner:
    """Run a single PyTorch experiment from YAML config."""

    config: Mapping[str, object]
    force: bool = False

    def run(self, experiment_name: str = "pytorch_unet_mgda") -> Path:
        """Execute training/evaluation and write run artifacts."""
        del self.force  # Reserved for future resume/overwrite policy.
        seed = int(self.config.get("project", {}).get("seed", 42))
        set_global_seed(seed)

        export_cfg = dict(self.config.get("export", {}))
        results_root = Path(export_cfg.get("results_dir", "results"))
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = results_root / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)

        save_resolved_config(self.config, str(run_dir / "config.yaml"))

        data_cfg = dict(self.config.get("data", {}))
        scanner = PairScanner(rgb_dir=str(data_cfg["rgb_dir"]), mask_dir=str(data_cfg["mask_dir"]))
        pairs, warnings = scanner.scan()
        with (run_dir / "skipped_files.txt").open("w", encoding="utf-8") as handle:
            for warning in warnings:
                handle.write(warning + "\n")

        densities = scanner.densities(pairs, image_size=int(data_cfg.get("image_size", 256)))
        splitter = StratifiedSplitter(
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            test_ratio=float(data_cfg.get("test_ratio", 0.15)),
            bins=int(data_cfg.get("building_density_bins", 3)),
            seed=seed,
        )
        split_indices = splitter.split(densities)
        splitter.save_split(split_indices, str(run_dir / "data_split.json"))

        loaders = build_dataloaders(
            DataConfig(
                rgb_dir=str(data_cfg["rgb_dir"]),
                mask_dir=str(data_cfg["mask_dir"]),
                image_size=int(data_cfg.get("image_size", 256)),
                batch_size=int(data_cfg.get("batch_size", 4)),
                num_workers=int(data_cfg.get("num_workers", 0)),
                seed=seed,
            ),
            pairs=pairs,
            split_indices=split_indices,
            augmentation_cfg=dict(self.config.get("augmentation", {})),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(self.config)
        ckpt = CheckpointManager(run_dir / "checkpoints")
        trainer = Trainer(model, self.config, device=device, checkpoint_manager=ckpt, results_dir=run_dir)
        training_result = trainer.fit(loaders["train"], loaders["val"])

        evaluator = Evaluator(device=device)
        test_metrics = evaluator.evaluate(model.to(device), loaders["test"], threshold=0.5)
        complexity = ModelComplexityAnalyzer(
            input_shape=(3, int(data_cfg.get("image_size", 256)), int(data_cfg.get("image_size", 256)))
        ).analyze(model, device=device)

        summary = {
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "framework": "pytorch",
            "config": self.config,
            "dataset_info": {
                "hash": compute_dataset_hash(str(data_cfg["rgb_dir"]), str(data_cfg["mask_dir"])),
                "total_images": len(pairs),
                "train_count": len(split_indices["train"]),
                "val_count": len(split_indices["val"]),
                "test_count": len(split_indices["test"]),
                "mean_building_density": float(np.mean(densities)) if len(densities) else 0.0,
            },
            "training_info": {
                "total_epochs": len(training_result.history.get("train_loss", [])),
                "best_epoch": training_result.best_epoch,
                "total_time_seconds": training_result.total_time_seconds,
                "early_stopped": training_result.stopped_early,
            },
            "test_metrics": _to_float_dict(test_metrics),
            "model_complexity": _to_float_dict(complexity),
            "mgda_final_alphas": training_result.mgda_alpha_history[-1]
            if training_result.mgda_alpha_history
            else {},
        }

        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return run_dir


def run_from_config(config_path: str, experiment_name: str = "pytorch_unet_mgda") -> Path:
    """Load config and run PyTorch experiment."""
    config = load_config(config_path)
    return PyTorchExperimentRunner(config=config).run(experiment_name=experiment_name)
