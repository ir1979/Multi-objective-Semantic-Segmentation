"""Run the complete discovered test suite and persist timestamped artifacts."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

from utils.test_reporting import run_suite_with_logging


def build_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover("tests", pattern="test_*.py")


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path("results") / "complete_tests" / ts
    suite = build_suite()
    result, report_path, detail_path = run_suite_with_logging(suite, report_dir=root, verbosity=2)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": "tests/test_*.py",
        "tests_run": result.testsRun,
        "was_successful": result.wasSuccessful(),
        "summary_path": str(report_path),
        "detail_path": str(detail_path),
        "timings_path": str(root / "test_timings.json"),
        "failures_path": str(root / "test_failures.log"),
    }
    manifest_path = root / "complete_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Complete test summary written to {report_path}")
    print(f"Complete test detail log written to {detail_path}")
    print(f"Complete test manifest written to {manifest_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
