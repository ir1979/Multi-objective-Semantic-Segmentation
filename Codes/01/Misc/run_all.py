"""Master execution script for the publication-grade segmentation framework."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - exercised in dependency-missing envs
    tf = None

from data.integrity import compute_dataset_hash
from logging_utils.logger import DualLogger
from utils.config_loader import ConfigValidationError, load_config
from utils.reproducibility import set_global_seed
from utils.test_reporting import run_suite_with_logging


@dataclass
class TestResult:
    """Simple test-suite result payload."""

    total: int
    failed: int
    critical_failures: int


def resolve_console_level(verbose: int, quiet: bool) -> str:
    """Map CLI verbosity flags to a console log level."""
    if quiet:
        return "ERROR"
    if verbose >= 2:
        return "DEBUG"
    if verbose == 1:
        return "INFO"
    return "WARNING"


def normalize_log_level(level: str) -> str:
    """Normalize a user-provided logging level string."""
    return str(level).upper()


def apply_logging_overrides(config: Dict[str, object], console_level: str, file_level: str) -> Dict[str, object]:
    """Apply CLI logging overrides while keeping file logs verbose."""
    resolved = dict(config)
    resolved["logging_console_level"] = normalize_log_level(console_level)
    resolved["logging_file_level"] = normalize_log_level(file_level)
    return resolved


def configure_runtime_warnings(
    results_dir: str | Path = "results",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> DualLogger:
    """Set predictable warning/logging behavior for production runs."""
    results_path = Path(results_dir)
    logger = DualLogger(
        results_path / "pipeline.log",
        console_level=normalize_log_level(console_level),
        file_level=normalize_log_level(file_level),
    )
    logging.captureWarnings(True)
    warning_logger = logging.getLogger("py.warnings")
    warning_logger.setLevel(logging.WARNING)
    for handler in logger.logger.handlers:
        warning_logger.addHandler(handler)
    warning_logger.propagate = False

    warnings.filterwarnings(
        "once",
        message=r".*Compiled the loaded model, but the compiled metrics have yet to be built.*",
    )
    warnings.filterwarnings(
        "once",
        message=r".*This file format is considered legacy.*",
    )
    warnings.filterwarnings(
        "once",
        category=FutureWarning,
        module=r"pandas.*",
    )

    if tf is not None:
        tf.get_logger().setLevel("WARNING" if console_level == "DEBUG" else "ERROR")
    logging.getLogger("tensorflow").setLevel(logging.WARNING if console_level == "DEBUG" else logging.ERROR)
    logging.getLogger("absl").setLevel(logging.WARNING if console_level == "DEBUG" else logging.ERROR)
    logger.info("Runtime warning filters configured.")
    return logger


def run_test_suite(quick: bool = False) -> TestResult:
    """Run test suite using unittest discovery."""
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    result, report_path, detail_path = run_suite_with_logging(
        suite,
        report_dir="results",
        verbosity=2 if not quick else 1,
    )
    failed = len(result.failures) + len(result.errors)
    print(f"Test summary written to {report_path}")
    print(f"Detailed test log written to {detail_path}")
    return TestResult(
        total=result.testsRun,
        failed=failed,
        critical_failures=failed,
    )


def validate_environment() -> Tuple[bool, List[str], List[str]]:
    """Check Python/TF/runtime requirements."""
    warnings: List[str] = []
    errors: List[str] = []

    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required.")

    if tf is None:
        errors.append("TensorFlow is not installed.")
        gpus = []
    else:
        tf_version = tuple(int(part) for part in tf.__version__.split(".")[:2])
        if tf_version < (2, 10):
            errors.append(f"TensorFlow>=2.10 required, found {tf.__version__}.")
        gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        warnings.append("No GPU detected; full pipeline will be significantly slower.")
    else:
        try:
            device_details = tf.config.experimental.get_device_details(gpus[0])
            device_name = device_details.get("device_name") or getattr(gpus[0], "name", "GPU:0")
            warnings.append(f"Detected GPU device: {device_name}.")
        except Exception:
            warnings.append("GPU detected, but detailed device information is unavailable.")

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
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase console verbosity; repeat for DEBUG")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show errors on the console")
    parser.add_argument(
        "--log-file-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the minimum level written to log files",
    )
    args = parser.parse_args()

    start_time = time.time()
    console_level = resolve_console_level(args.verbose, args.quiet)
    file_level = normalize_log_level(args.log_file_level)
    pipeline_logger = configure_runtime_warnings("results", console_level=console_level, file_level=file_level)
    pipeline_logger.info(f"Starting pipeline with args: {vars(args)}")
    env_valid, env_warnings, env_errors = validate_environment()
    if not env_valid:
        print("CRITICAL: Environment validation failed")
        for err in env_errors:
            print(f"  ERROR: {err}")
            pipeline_logger.error(f"Environment error: {err}")
        pipeline_logger.close()
        sys.exit(2)
    for warn in env_warnings:
        print(f"  WARNING: {warn}")
        pipeline_logger.warning(f"Environment warning: {warn}")

    try:
        config = load_config(args.config)
    except ConfigValidationError as exc:
        print(f"CRITICAL: Config validation failed: {exc}")
        pipeline_logger.exception("Configuration validation failed")
        pipeline_logger.close()
        sys.exit(2)

    config = apply_logging_overrides(config, console_level, file_level)
    set_global_seed(int(config.get("project_seed", 42)))
    pipeline_logger.log_config({"config_path": args.config, "seed": int(config.get("project_seed", 42))})

    ds_valid, ds_info = validate_dataset(config)
    if not ds_valid:
        print("CRITICAL: Dataset validation failed")
        print(ds_info)
        pipeline_logger.error(f"Dataset validation failed: {ds_info}")
        pipeline_logger.close()
        sys.exit(2)
    print(f"Dataset info: {json.dumps(ds_info, indent=2)}")
    pipeline_logger.info(f"Dataset info: {ds_info}")

    if not args.experiments_only and not args.figures_only:
        test_result = run_test_suite(quick=args.experiment is not None)
        pipeline_logger.info(
            f"Test suite finished | total={test_result.total} failed={test_result.failed} "
            f"critical_failures={test_result.critical_failures}"
        )
        if test_result.critical_failures > 0:
            print(f"CRITICAL: {test_result.critical_failures} critical tests failed")
            pipeline_logger.error(f"Aborting due to {test_result.critical_failures} critical test failures.")
            pipeline_logger.close()
            sys.exit(3)

    if args.test_only:
        print("Tests completed. Exiting.")
        pipeline_logger.info("Exiting after test-only run.")
        pipeline_logger.close()
        sys.exit(0)

    from experiments import ExperimentRegistry, ExperimentRunner

    runner = ExperimentRunner(config, force=args.force, console_level=console_level, file_level=file_level)
    try:
        if not args.figures_only:
            if args.experiment:
                runner.run_single(args.experiment)
            elif args.no_pareto:
                runner.run_subset([name for name in runner.get_all_experiments() if name != "pareto_sweep"])
            else:
                runner.run_all()

        runner.generate_paper_outputs()
    except Exception as exc:  # pragma: no cover - safety net for production runs
        error_log = Path(config.get("export_results_dir", "results")) / "pipeline_error.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"CRITICAL: Pipeline failed with {exc.__class__.__name__}: {exc}")
        print(f"Detailed traceback written to {error_log}")
        pipeline_logger.exception("Pipeline execution failed")
        pipeline_logger.close()
        sys.exit(4)

    elapsed = time.time() - start_time
    registry = ExperimentRegistry(Path(config.get("export_results_dir", "results")) / "experiment_registry.json")
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
    pipeline_logger.info(
        f"Pipeline finished | completed={len(completed)} failed={len(failed)} elapsed_seconds={elapsed:.2f}"
    )
    if failed:
        pipeline_logger.warning(f"Failed experiments: {failed}")
    pipeline_logger.close()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
