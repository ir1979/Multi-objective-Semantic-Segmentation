"""Run a focused dry-test suite and persist artifacts under results/dry_tests/."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

from utils.test_reporting import run_suite_with_logging


DRY_TEST_MODULES = [
    "tests.test_mohpo_dry",
    "tests.test_run_all_cli",
]


def build_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module_name in DRY_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path("results") / "dry_tests" / ts
    suite = build_suite()
    result, report_path, detail_path = run_suite_with_logging(suite, report_dir=root, verbosity=2)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite_modules": DRY_TEST_MODULES,
        "tests_run": result.testsRun,
        "was_successful": result.wasSuccessful(),
        "summary_path": str(report_path),
        "detail_path": str(detail_path),
        "timings_path": str(root / "test_timings.json"),
        "failures_path": str(root / "test_failures.log"),
    }
    manifest_path = root / "dry_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Dry test summary written to {report_path}")
    print(f"Dry test detail log written to {detail_path}")
    print(f"Dry test manifest written to {manifest_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
