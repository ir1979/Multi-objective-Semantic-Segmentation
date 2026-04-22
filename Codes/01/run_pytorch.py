"""Run PyTorch-based segmentation experiments from YAML config."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import importlib
from utils.config_loader import ConfigValidationError, load_config


def validate_environment() -> Tuple[bool, List[str], List[str]]:
    """Validate Python and package availability for PyTorch pipeline."""
    warnings: List[str] = []
    errors: List[str] = []

    if sys.version_info < (3, 7):
        errors.append("Python 3.7+ is required.")
    elif sys.version_info >= (3, 8):
        warnings.append(
            "Target environment is Python 3.7.x (tested for 3.7.16)."
        )

    required_packages = ["torch", "numpy", "pandas", "yaml", "PIL"]
    optional_packages = ["cv2", "matplotlib", "scipy", "sklearn", "tensorboard"]

    for package in required_packages:
        try:
            importlib.import_module(package)
        except Exception:
            errors.append(f"Missing required package: {package}")

    for package in optional_packages:
        try:
            importlib.import_module(package)
        except Exception:
            warnings.append(f"Optional package not installed: {package}")

    free_disk_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    if free_disk_gb < 10:
        warnings.append(f"Free disk space is low ({free_disk_gb:.2f} GB).")

    try:
        import torch  # local import to avoid import-side effects when unavailable

        if not torch.cuda.is_available():
            warnings.append("CUDA GPU not detected. Training will run on CPU.")
    except Exception:
        pass

    return len(errors) == 0, warnings, errors


def validate_dataset(config: Dict[str, object]) -> Tuple[bool, Dict[str, object]]:
    """Validate dataset paths and pair counts."""
    data_cfg = dict(config.get("data", {}))
    rgb_dir = Path(str(data_cfg.get("rgb_dir", "")))
    mask_dir = Path(str(data_cfg.get("mask_dir", "")))
    if not rgb_dir.exists() or not mask_dir.exists():
        return False, {"error": "RGB or mask directory does not exist."}

    rgb_files = sorted(rgb_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.tif"))
    if not rgb_files or not mask_files:
        return False, {"error": "No RGB or mask files found."}

    matched = len({path.stem for path in rgb_files} & {path.stem for path in mask_files})
    return True, {
        "rgb_count": len(rgb_files),
        "mask_count": len(mask_files),
        "matched_pairs": matched,
    }


def main() -> None:
    """Entry point for running PyTorch experiments."""
    parser = argparse.ArgumentParser(description="PyTorch Building Segmentation Runner")
    parser.add_argument("--config", default="configs/pytorch_default.yaml", help="Path to YAML config")
    parser.add_argument("--name", default="pytorch_unet_mgda", help="Experiment run name prefix")
    args = parser.parse_args()

    env_valid, env_warnings, env_errors = validate_environment()
    if not env_valid:
        print("CRITICAL: Environment validation failed")
        for err in env_errors:
            print(f"  ERROR: {err}")
        sys.exit(2)
    for warn in env_warnings:
        print(f"  WARNING: {warn}")

    try:
        config = load_config(args.config)
    except ConfigValidationError as exc:
        print(f"CRITICAL: Config validation failed: {exc}")
        sys.exit(2)

    ds_valid, ds_info = validate_dataset(config)
    if not ds_valid:
        print("CRITICAL: Dataset validation failed")
        print(ds_info)
        sys.exit(2)
    print(f"Dataset info: {json.dumps(ds_info, indent=2)}")

    from frameworks.pytorch.experiments.runner import PyTorchExperimentRunner

    runner = PyTorchExperimentRunner(config=config, force=False)
    run_dir = runner.run(experiment_name=args.name)
    print("\n" + "=" * 60)
    print("PYTORCH PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
