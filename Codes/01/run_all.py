"""Master execution script for the publication-grade segmentation framework."""

from __future__ import annotations

import argparse
import json
import importlib
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - exercised in dependency-missing envs
    tf = None

import hashlib
from utils.config_loader import ConfigValidationError, load_config
from utils.reproducibility import set_global_seed


def compute_dataset_hash(rgb_dir: str, mask_dir: str) -> str:
    """Compute lightweight hash from filenames and sizes."""
    hasher = hashlib.sha256()
    for directory in sorted([Path(rgb_dir), Path(mask_dir)], key=lambda item: item.as_posix()):
        if not directory.exists():
            hasher.update(("missing:" + str(directory)).encode("utf-8"))
            continue
        for file_path in sorted(directory.glob("*")):
            if not file_path.is_file():
                continue
            hasher.update(file_path.name.encode("utf-8"))
            hasher.update(str(file_path.stat().st_size).encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class TestResult:
    """Simple test-suite result payload."""

    total: int
    failed: int
    critical_failures: int


def run_test_suite(quick: bool = False) -> TestResult:
    """Run test suite using unittest discovery."""
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2 if not quick else 1)
    result = runner.run(suite)

    report_path = Path("results/test_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                f"tests_run={result.testsRun}",
                f"failures={len(result.failures)}",
                f"errors={len(result.errors)}",
                f"was_successful={result.wasSuccessful()}",
            ]
        ),
        encoding="utf-8",
    )
    failed = len(result.failures) + len(result.errors)
    return TestResult(
        total=result.testsRun,
        failed=failed,
        critical_failures=failed,
    )


def validate_environment() -> Tuple[bool, List[str], List[str]]:
    """Check Python/TF/runtime requirements."""
    warnings: List[str] = []
    errors: List[str] = []

    if sys.version_info < (3, 7):
        errors.append("Python 3.7+ is required.")
    elif sys.version_info >= (3, 8):
        warnings.append("Target environment is Python 3.7.x (tested for 3.7.16).")

    if tf is None:
        errors.append("TensorFlow is not installed.")
        gpus = []
    else:
        version_parts = []
        for part in tf.__version__.split(".")[:2]:
            digits = "".join(ch for ch in part if ch.isdigit())
            version_parts.append(int(digits) if digits else 0)
        tf_version = tuple(version_parts)
        if tf_version < (2, 6):
            errors.append(f"TensorFlow>=2.6 required, found {tf.__version__}.")
        elif tf_version >= (2, 7):
            warnings.append(
                "Target environment is TensorFlow 2.6.x; newer versions may behave differently."
            )
        gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        warnings.append("No GPU detected; full pipeline will be significantly slower.")
    else:
        try:
            memory_info = tf.config.experimental.get_memory_info("GPU:0")
            gpu_memory_gb = memory_info.get("current", 0) / (1024**3)
            if gpu_memory_gb < 8:
                warnings.append("Detected GPU memory appears below 8 GB.")
        except Exception:
            warnings.append("Unable to read GPU memory information.")

    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "yaml",
        "tqdm",
        "PIL",
        "cv2",
        "sklearn",
    ]
    for package in required_packages:
        try:
            __import__(package)
        except Exception:
            errors.append(f"Missing required package: {package}")

    free_disk_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    if free_disk_gb < 10:
        warnings.append(f"Free disk space is low ({free_disk_gb:.2f} GB).")

    return len(errors) == 0, warnings, errors


def validate_dataset(config: Dict[str, object]) -> Tuple[bool, dict]:
    """Validate dataset paths and basic sample consistency."""
    data_cfg = dict(config.get("data", {}))
    rgb_dir = Path(str(data_cfg.get("rgb_dir", "")))
    mask_dir = Path(str(data_cfg.get("mask_dir", "")))
    if not rgb_dir.exists() or not mask_dir.exists():
        return False, {"error": "RGB or mask directory does not exist."}

    rgb_files = sorted(rgb_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.tif"))
    if not rgb_files or not mask_files:
        return False, {"error": "No RGB or mask files found."}

    dataset_hash = compute_dataset_hash(str(rgb_dir), str(mask_dir))
    sample_match = len({path.stem for path in rgb_files} & {path.stem for path in mask_files})
    return True, {
        "rgb_count": len(rgb_files),
        "mask_count": len(mask_files),
        "matched_pairs": sample_match,
        "dataset_hash": dataset_hash,
    }


def main() -> None:
    """Execute the full end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Building Segmentation MOO Pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config")
    parser.add_argument("--experiments-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--experiment", type=str, default=None, help="Run single experiment")
    parser.add_argument("--force", action="store_true", help="Re-run completed experiments")
    parser.add_argument("--no-pareto", action="store_true", help="Skip Pareto sweep")
    args = parser.parse_args()

    start_time = time.time()
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

    set_global_seed(int(config.get("project", {}).get("seed", 42)))

    ds_valid, ds_info = validate_dataset(config)
    if not ds_valid:
        print("CRITICAL: Dataset validation failed")
        print(ds_info)
        sys.exit(2)
    print(f"Dataset info: {json.dumps(ds_info, indent=2)}")

    if not args.experiments_only and not args.figures_only:
        test_result = run_test_suite(quick=args.experiment is not None)
        if test_result.critical_failures > 0:
            print(f"CRITICAL: {test_result.critical_failures} critical tests failed")
            sys.exit(3)

    if args.test_only:
        print("Tests completed. Exiting.")
        sys.exit(0)

    from experiments.experiment_runner import ExperimentRunner
    from experiments.registry import ExperimentRegistry

    runner = ExperimentRunner(config, force=args.force)
    if not args.figures_only:
        if args.experiment:
            runner.run_single(args.experiment)
        elif args.no_pareto:
            runner.run_subset([name for name in runner.get_all_experiments() if name != "pareto_sweep"])
        else:
            runner.run_all()

    runner.generate_paper_outputs()

    elapsed = time.time() - start_time
    registry = ExperimentRegistry(Path(config.get("export", {}).get("results_dir", "results")) / "experiment_registry.json")
    payload = registry.load()
    completed = [name for name, status in payload.items() if status.get("status") == "completed"]
    failed = [name for name, status in payload.items() if status.get("status") == "failed"]

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Experiments completed: {len(completed)}/{len(payload)}")
    if failed:
        print(f"Experiments failed: {', '.join(failed)}")
    print("Paper outputs: paper/figures and paper/tables")
    print("=" * 60)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
