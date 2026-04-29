"""Specialized logging for grid search experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from logging_utils.logger import DualLogger


class GridSearchLogger(DualLogger):
    """Enhanced logger for grid search with structured logging."""

    def __init__(self, log_file: Path, console_level: str = "INFO", file_level: str = "DEBUG") -> None:
        super().__init__(log_file, console_level=console_level, file_level=file_level)
        self.metrics_file = log_file.parent / "metrics.jsonl"
        self.events_file = log_file.parent / "events.jsonl"
        self.start_time = time.time()

    def log_point_started(self, point_id: int, params: Dict[str, Any]) -> None:
        """Log grid point start."""
        self.info(f"Started Grid Point {point_id}")
        self.debug(f"Parameters: {json.dumps(params, indent=2)}")
        self._append_event("point_started", {"point_id": point_id, "params": params})

    def log_point_completed(self, point_id: int, metrics: Dict[str, float], duration: float) -> None:
        """Log grid point completion."""
        self.info(f"Completed Grid Point {point_id} in {duration:.2f}s | Metrics: {metrics}")
        self._append_event(
            "point_completed",
            {
                "point_id": point_id,
                "metrics": metrics,
                "duration": duration,
            },
        )
        self._append_metrics(point_id, metrics)

    def log_point_failed(self, point_id: int, error: str, duration: float) -> None:
        """Log grid point failure."""
        self.error(f"FAILED Grid Point {point_id} after {duration:.2f}s: {error}")
        self._append_event(
            "point_failed",
            {
                "point_id": point_id,
                "error": error,
                "duration": duration,
            },
        )

    def log_batch_summary(
        self,
        batch_num: int,
        points_completed: int,
        points_failed: int,
        avg_duration: float,
        avg_metrics: Dict[str, float],
    ) -> None:
        """Log batch summary."""
        self.info(
            f"Batch {batch_num} Summary | Completed: {points_completed}, "
            f"Failed: {points_failed}, Avg Duration: {avg_duration:.2f}s"
        )
        self.debug(f"Average Metrics: {avg_metrics}")

    def log_grid_summary(
        self,
        total_points: int,
        completed: int,
        failed: int,
        best_point_id: int,
        best_metric: float,
        total_duration: float,
    ) -> None:
        """Log final grid search summary."""
        self.info("\n" + "=" * 80)
        self.info("GRID SEARCH SUMMARY")
        self.info("=" * 80)
        self.info(f"Total Points: {total_points}")
        self.info(f"Completed: {completed}")
        self.info(f"Failed: {failed}")
        self.info(f"Best Point ID: {best_point_id} (Metric: {best_metric:.4f})")
        self.info(f"Total Duration: {total_duration:.2f}s ({total_duration/3600:.2f}h)")
        self.info("=" * 80 + "\n")

    def _append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append structured event to events log."""
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": event_type,
                "elapsed": time.time() - self.start_time,
                **data,
            }
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            self.debug(f"Error writing event log: {exc}")

    def _append_metrics(self, point_id: int, metrics: Dict[str, float]) -> None:
        """Append metrics to metrics log."""
        try:
            record = {
                "point_id": point_id,
                "timestamp": datetime.utcnow().isoformat(),
                **metrics,
            }
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            self.debug(f"Error writing metrics log: {exc}")

    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger start."""
        return time.time() - self.start_time
