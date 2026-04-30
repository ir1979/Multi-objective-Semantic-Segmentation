"""CLI logging configuration tests for the pipeline entrypoint."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MISC_DIR = PROJECT_ROOT / "Misc"
if str(MISC_DIR) not in sys.path:
    sys.path.insert(0, str(MISC_DIR))

from run_all import apply_logging_overrides, normalize_log_level, resolve_console_level


class TestRunAllCLI(unittest.TestCase):
    """Validate CLI verbosity mapping and logging overrides."""

    def test_resolve_console_level_defaults_to_info(self) -> None:
        self.assertEqual(resolve_console_level(verbose=1, quiet=False), "INFO")

    def test_resolve_console_level_quiet_wins(self) -> None:
        self.assertEqual(resolve_console_level(verbose=2, quiet=True), "ERROR")

    def test_resolve_console_level_debug(self) -> None:
        self.assertEqual(resolve_console_level(verbose=2, quiet=False), "DEBUG")

    def test_apply_logging_overrides_preserves_debug_file_logs(self) -> None:
        config = {"logging_file_level": "DEBUG", "logging_validation_image_interval": 5}
        updated = apply_logging_overrides(config, "WARNING", "INFO")
        self.assertEqual(updated["logging_console_level"], "WARNING")
        self.assertEqual(updated["logging_file_level"], "INFO")
        self.assertEqual(updated["logging_validation_image_interval"], 5)

    def test_normalize_log_level_uppercases(self) -> None:
        self.assertEqual(normalize_log_level("warning"), "WARNING")
