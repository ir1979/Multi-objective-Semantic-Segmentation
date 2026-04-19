"""PyTorch callback-like logging helpers."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MGDAAlphaLogger:
    """Store MGDA alpha values per epoch."""

    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, epoch: int, alphas: Dict[str, float]) -> None:
        self.history.append({"epoch": float(epoch), **{k: float(v) for k, v in alphas.items()}})


@dataclass
class TrainingTimeLogger:
    """Track epoch and cumulative training times."""

    start_time: float = field(default_factory=time.time)
    epoch_times: List[float] = field(default_factory=list)

    def log_epoch(self, epoch_duration: float) -> Dict[str, float]:
        self.epoch_times.append(float(epoch_duration))
        return {
            "epoch_time_seconds": float(epoch_duration),
            "total_time_seconds": float(time.time() - self.start_time),
        }


@dataclass
class ValidationImageLogger:
    """Persist periodic validation metadata placeholders."""

    output_dir: Path
    interval: int = 5
    sample_batch: Optional[tuple] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, epoch: int) -> None:
        if epoch % self.interval != 0:
            return
        with (self.output_dir / f"val_epoch_{epoch:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump({"epoch": epoch, "logged": True}, handle, indent=2)


class DualLogger:
    """Simple dual console/file logger."""

    def __init__(self, log_file: Path, console_level: str = "INFO", file_level: str = "DEBUG") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"pytorch.{self.log_file.stem}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))

        fh = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(
            logging.Formatter(
                "[%(asctime)s.%(msecs)03d] %(levelname)s [%(module)s:%(lineno)d] - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        mgda_alphas: Optional[Dict[str, float]] = None,
    ) -> None:
        message = (
            f"Epoch {epoch:03d} | lr={lr:.3e} | train={train_metrics} | val={val_metrics}"
        )
        if mgda_alphas:
            message += f" | mgda_alphas={mgda_alphas}"
        self.logger.info(message)


class CSVLogger:
    """Append per-epoch metrics to CSV."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log_epoch(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics_dict}
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


class TensorBoardLogger:
    """JSON-backed placeholder scalar logger."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "scalars.jsonl"

    def log_scalars(self, epoch: int, scalars_dict: Dict[str, float]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, **{k: float(v) for k, v in scalars_dict.items()}}) + "\n")

    def log_mgda_alphas(self, epoch: int, alphas_dict: Dict[str, float]) -> None:
        self.log_scalars(epoch, {f"mgda/{k}": v for k, v in alphas_dict.items()})

    def close(self) -> None:
        return None
