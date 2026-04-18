#!/usr/bin/env python3
"""Small smoke test for the multi-objective segmentation pipeline."""

import argparse
import json
import os
import sys
from copy import deepcopy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.config_manager import ConfigManager
from plugins import load_plugins_from_config
from src.optimize import evaluate_experiment_config


def build_smoke_config(config_path: str, epochs: int, tile_size: int):
    manager = ConfigManager(config_path)
    root_config = deepcopy(manager.config)
    load_plugins_from_config(root_config)

    root_config.setdefault("models", {})
    root_config["models"]["architectures"] = ["Unet"]
    root_config["models"]["encoders"] = ["resnet34"]
    root_config["models"]["deep_supervision"] = False

    root_config.setdefault("training", {})
    root_config["training"]["loss_functions"] = ["dice_loss"]
    root_config["training"]["optimizers"] = ["Adam"]
    root_config["training"]["learning_rates"] = [1e-3]
    root_config["training"]["batch_sizes"] = [2]
    root_config["training"]["epochs"] = [epochs]

    root_config.setdefault("dataset", {})
    root_config["dataset"]["tile_size"] = [tile_size]
    root_config["dataset"]["augmentations"] = [["flip"]]

    experiment = manager.get_experiment_configs(mode="quick")[0]
    experiment.architecture = "Unet"
    experiment.encoder = "resnet34"
    experiment.loss_function = "dice_loss"
    experiment.optimizer = "Adam"
    experiment.learning_rate = 1e-3
    experiment.batch_size = 2
    experiment.epochs = epochs
    experiment.tile_size = tile_size
    experiment.augmentation = ["flip"]
    experiment.experiment_id = f"smoke_unet_{tile_size}_{epochs}ep"

    return experiment, root_config


def main():
    parser = argparse.ArgumentParser(description="Run a fast smoke test experiment.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    experiment, root_config = build_smoke_config(args.config, args.epochs, args.tile_size)
    result = evaluate_experiment_config(
        experiment,
        root_config,
        output_dir=args.output_dir,
        epoch_override=args.epochs,
    )

    payload = {
        "experiment_id": experiment.experiment_id,
        "iou": result["iou"],
        "f1_score": result["f1_score"],
        "summary_path": os.path.join(args.output_dir, experiment.experiment_id, "summary.json"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
