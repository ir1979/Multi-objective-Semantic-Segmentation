"""Compatibility wrappers for experiment folder creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def create_experiment_folder(base_path: str, experiment_name: str) -> str:
    """Create canonical experiment folder structure."""
    path = Path(base_path) / experiment_name
    for subdir in ("checkpoints", "figures", "tables", "results", "tensorboard"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return str(path)


def write_experiment_config(path: str, config: Mapping[str, Any]) -> str:
    """Write resolved experiment config to YAML file."""
    output_path = Path(path) / "config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    return str(output_path)
