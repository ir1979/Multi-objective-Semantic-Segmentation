"""Utilities for detailed unittest logging."""

from __future__ import annotations

import io
import json
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO


class TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class DetailedTextTestResult(unittest.TextTestResult):
    """Text test result that records per-test timing and errors."""

    def __init__(self, stream, descriptions, verbosity, detail_handle: TextIO) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.detail_handle = detail_handle
        self.test_timings: list[dict[str, object]] = []
        self._started_at = 0.0

    def startTest(self, test: unittest.case.TestCase) -> None:
        self._started_at = time.time()
        self.detail_handle.write(f"[START] {self.getDescription(test)}\n")
        self.detail_handle.flush()
        super().startTest(test)

    def stopTest(self, test: unittest.case.TestCase) -> None:
        elapsed = time.time() - self._started_at
        self.test_timings.append({"test": self.getDescription(test), "seconds": round(elapsed, 4)})
        self.detail_handle.write(f"[STOP] {self.getDescription(test)} ({elapsed:.2f}s)\n")
        self.detail_handle.flush()
        super().stopTest(test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        self.detail_handle.write(f"[PASS] {self.getDescription(test)}\n")
        self.detail_handle.flush()
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err) -> None:
        self.detail_handle.write(f"[FAIL] {self.getDescription(test)}\n")
        self.detail_handle.write(self._exc_info_to_string(err, test) + "\n")
        self.detail_handle.flush()
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err) -> None:
        self.detail_handle.write(f"[ERROR] {self.getDescription(test)}\n")
        self.detail_handle.write(self._exc_info_to_string(err, test) + "\n")
        self.detail_handle.flush()
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        self.detail_handle.write(f"[SKIP] {self.getDescription(test)} :: {reason}\n")
        self.detail_handle.flush()
        super().addSkip(test, reason)

    def addExpectedFailure(self, test: unittest.case.TestCase, err) -> None:
        self.detail_handle.write(f"[XFAIL] {self.getDescription(test)}\n")
        self.detail_handle.flush()
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        self.detail_handle.write(f"[XPASS] {self.getDescription(test)}\n")
        self.detail_handle.flush()
        super().addUnexpectedSuccess(test)


class DetailedTextTestRunner(unittest.TextTestRunner):
    """Text runner that writes full progress and traceback logs to disk."""

    resultclass = DetailedTextTestResult

    def __init__(self, *args, detail_handle: TextIO, **kwargs) -> None:
        self.detail_handle = detail_handle
        super().__init__(*args, **kwargs)

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity, self.detail_handle)


def run_suite_with_logging(
    suite: unittest.TestSuite,
    report_dir: str | Path = "results",
    verbosity: int = 2,
) -> tuple[unittest.TestResult, Path, Path]:
    """Run a unittest suite and persist detailed logs and summary artifacts."""
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "test_report.txt"
    detail_path = report_root / "test_details.log"
    timings_path = report_root / "test_timings.json"

    with detail_path.open("w", encoding="utf-8") as detail_handle:
        tee_stream = TeeStream(sys.stderr, detail_handle)
        detail_handle.write(f"Detailed test log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        detail_handle.flush()
        runner = DetailedTextTestRunner(
            stream=tee_stream,
            verbosity=verbosity,
            detail_handle=detail_handle,
        )
        with redirect_stdout(tee_stream), redirect_stderr(tee_stream):
            result = runner.run(suite)
        summary_path.write_text(
            "\n".join(
                [
                    f"tests_run={result.testsRun}",
                    f"failures={len(result.failures)}",
                    f"errors={len(result.errors)}",
                    f"skipped={len(getattr(result, 'skipped', []))}",
                    f"expected_failures={len(getattr(result, 'expectedFailures', []))}",
                    f"unexpected_successes={len(getattr(result, 'unexpectedSuccesses', []))}",
                    f"was_successful={result.wasSuccessful()}",
                    f"detail_log={detail_path}",
                    f"timings_log={timings_path}",
                ]
            ),
            encoding="utf-8",
        )
        timings_path.write_text(
            json.dumps(getattr(result, "test_timings", []), indent=2),
            encoding="utf-8",
        )
    return result, summary_path, detail_path
