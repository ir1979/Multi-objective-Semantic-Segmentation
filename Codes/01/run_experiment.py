#!/usr/bin/env python
"""Entry point for single training-run experiments from a flat YAML config.

Usage
-----
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --config my_saved_config.yaml --results-dir results/run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single segmentation training experiment from a YAML config"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--results-dir", default="results/single_run", help="Results directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Load raw YAML to detect grid_search mode before importing heavy deps
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    grid_enabled = (
        raw.get("grid_search_enabled")
        or (isinstance(raw.get("grid_search"), dict) and raw["grid_search"].get("enabled"))
    )
    if grid_enabled:
        print(
            "INFO: grid_search is enabled in this config — delegating to run_grid_search.py",
            file=sys.stderr,
        )
        import subprocess
        cmd = [sys.executable, "run_grid_search.py", "--config", str(config_path),
               "--results-dir", args.results_dir]
        return subprocess.call(cmd)

    # ── Heavy imports (TF / Keras) only after mode check ──────────────────────
    from utils.config_loader import load_config
    from data.loader import BuildingSegmentationDataset, DatasetConfig
    from data.splitter import StratifiedSplitter
    from models.model_factory import build_model
    from training.trainer import Trainer
    from training.checkpoint_manager import CheckpointManager
    from logging_utils.logger import DualLogger

    config = load_config(str(config_path))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = DualLogger(
        results_dir / "experiment.log",
        console_level=str(config.get("logging_console_level", "INFO")),
        file_level=str(config.get("logging_file_level", "DEBUG")),
    )

    logger.info("=" * 70)
    logger.info("EXPERIMENT STARTED")
    logger.info(f"  Config : {config_path}")
    logger.info(f"  Results: {results_dir}")
    logger.info("=" * 70)

    start = time.time()

    try:
        # ── Data ──────────────────────────────────────────────────────────────
        image_size = (
            int(config.get("data_image_height", 256)),
            int(config.get("data_image_width", 256)),
        )
        dataset_cfg = DatasetConfig(
            rgb_dir=str(config.get("data_rgb_dir", "datasets/RGB")),
            mask_dir=str(config.get("data_mask_dir", "datasets/Mask")),
            image_size=image_size,
            batch_size=int(config.get("training_batch_size", 8)),
            seed=int(config.get("data_seed", 42)),
        )
        loader = BuildingSegmentationDataset(
            dataset_cfg,
            skipped_log_path=str(results_dir / "skipped_files.txt"),
        )
        loader.validate_pairs()
        density_labels = loader.get_density_labels()

        train_ratio = float(config.get("data_train_ratio", 0.7))
        val_ratio   = float(config.get("data_val_ratio",   0.15))
        test_ratio  = float(config.get("data_test_ratio",  0.15))
        splitter = StratifiedSplitter(
            train_ratio, val_ratio, test_ratio,
            bins=int(config.get("data_split_bins", 10)),
            seed=int(config.get("data_seed", 42)),
        )
        split = splitter.split(density_labels)

        train_ds = loader.get_tf_dataset(split["train"], augment=True,  shuffle=True)
        val_ds   = loader.get_tf_dataset(split["val"],   augment=False, shuffle=False)

        logger.info(
            f"Dataset split — train: {len(split['train'])}, "
            f"val: {len(split['val'])}, test: {len(split['test'])}"
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = build_model(config)
        logger.info(f"Model: {config.get('model_architecture', 'unet')}  "
                    f"params={model.count_params():,}")

        # ── Trainer ───────────────────────────────────────────────────────────
        ckpt_manager = CheckpointManager(
            results_dir / "checkpoints",
            best_metric_name=str(config.get("training_early_stopping_monitor", "val_iou")),
        )
        trainer = Trainer(model, config, results_dir, ckpt_manager)
        result = trainer.fit(train_ds, val_ds)

        elapsed = time.time() - start
        logger.info("=" * 70)
        logger.info(f"EXPERIMENT FINISHED  ({elapsed:.1f}s)")
        logger.info(f"  Best epoch : {result.best_epoch}")
        logger.info(f"  Best metric: {result.best_metric:.4f}")
        logger.info(f"  Early stop : {result.stopped_early}")
        logger.info("=" * 70)
        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"Experiment failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
