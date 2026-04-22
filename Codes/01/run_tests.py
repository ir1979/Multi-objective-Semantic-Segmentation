"""Master test runner for unit and integration suites."""

from __future__ import annotations

import argparse
import sys
import unittest

from utils.test_reporting import run_suite_with_logging


def _select_pattern(args: argparse.Namespace) -> str:
    if args.quick:
        return "test_*.py"
    if args.unit:
        return "test_*.py"
    if args.integration:
        return "test_*integration*.py"
    return "test_*.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run framework tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern=_select_pattern(args))
    if args.quick:
        suite = unittest.TestSuite(
            [test for index, test in enumerate(suite) if index < 5]
        )
    result, report_path, detail_path = run_suite_with_logging(suite, report_dir="results", verbosity=2)
    print(f"Test summary written to {report_path}")
    print(f"Detailed test log written to {detail_path}")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
