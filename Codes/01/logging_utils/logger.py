"""Dual console/file logger utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional


class DualLogger:
    """Logger writing INFO to console and DEBUG to file."""

    def __init__(
        self,
        log_file: Path,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
    ) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"building_segmentation.{self.log_file.stem}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))

        file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s.%(msecs)03d] %(levelname)s [%(module)s:%(lineno)d] - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
        lr: float,
        mgda_alphas: Optional[Mapping[str, float]] = None,
    ) -> None:
        message = (
            f"Epoch {epoch:03d} | lr={lr:.3e} | "
            f"train={dict(train_metrics)} | val={dict(val_metrics)}"
        )
        if mgda_alphas:
            message += f" | mgda_alphas={dict(mgda_alphas)}"
        self.info(message)

    def log_system_info(self, system_info_dict: Dict[str, object]) -> None:
        self.info(f"System info: {system_info_dict}")

    def log_config(self, config_dict: Dict[str, object]) -> None:
        self.info(f"Resolved config: {config_dict}")

    def close(self) -> None:
        """Close all logging handlers to release file locks."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
