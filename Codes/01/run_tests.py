"""Master test runner for unit and integration suites."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


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
    result = unittest.TextTestRunner(verbosity=2).run(suite)

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
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
