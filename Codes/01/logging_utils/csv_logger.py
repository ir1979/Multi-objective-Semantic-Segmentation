"""CSV logging helper for per-epoch metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pandas as pd


class CSVLogger:
    """Append per-epoch metrics to a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log_epoch(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics_dict}
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)
