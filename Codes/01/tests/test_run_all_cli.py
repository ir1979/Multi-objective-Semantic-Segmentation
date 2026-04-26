"""CLI logging configuration tests for the pipeline entrypoint."""

from __future__ import annotations

import unittest

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
        config = {"logging": {"file_level": "DEBUG", "validation_image_interval": 5}}
        updated = apply_logging_overrides(config, "WARNING", "INFO")
        self.assertEqual(updated["logging"]["console_level"], "WARNING")
        self.assertEqual(updated["logging"]["file_level"], "INFO")
        self.assertEqual(updated["logging"]["validation_image_interval"], 5)

    def test_normalize_log_level_uppercases(self) -> None:
        self.assertEqual(normalize_log_level("warning"), "WARNING")
