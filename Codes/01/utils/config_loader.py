"""Configuration loading and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a configuration file fails validation."""


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass
class ConfigValidator:
    """Schema-aware configuration validator."""

    architecture_values: Iterable[str] = field(default_factory=lambda: ("unet", "unetpp"))
    strategy_values: Iterable[str] = field(default_factory=lambda: ("single", "weighted", "mgda"))
    pixel_loss_values: Iterable[str] = field(
        default_factory=lambda: ("bce", "iou", "dice", "bce_iou", "focal")
    )
    scheduler_values: Iterable[str] = field(
        default_factory=lambda: ("cosine", "step", "plateau")
    )

    def validate(self, config: Mapping[str, Any]) -> None:
        """Validate required sections and enum values.

        Parameters
        ----------
        config:
            Resolved configuration dictionary.
        """
        required_sections = (
            "project",
            "data",
            "model",
            "loss",
            "training",
            "evaluation",
            "export",
        )
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Missing required config section: '{section}'")

        architecture = str(config["model"].get("architecture", "")).lower()
        if architecture not in set(self.architecture_values):
            raise ConfigValidationError(
                f"Unknown model architecture '{architecture}'. "
                f"Expected one of {list(self.architecture_values)}."
            )

        strategy = str(config["loss"].get("strategy", "")).lower()
        if strategy not in set(self.strategy_values):
            raise ConfigValidationError(
                f"Unknown loss strategy '{strategy}'. Expected {list(self.strategy_values)}."
            )

        pixel_type = str(config["loss"].get("pixel", {}).get("type", "")).lower()
        if pixel_type not in set(self.pixel_loss_values):
            raise ConfigValidationError(
                f"Unknown pixel loss '{pixel_type}'. Expected {list(self.pixel_loss_values)}."
            )

        scheduler = str(config["training"].get("lr_scheduler", {}).get("type", "")).lower()
        if scheduler not in set(self.scheduler_values):
            raise ConfigValidationError(
                f"Unknown scheduler '{scheduler}'. Expected {list(self.scheduler_values)}."
            )

        self._validate_ratios(config)
        self._validate_paths(config)

    @staticmethod
    def _validate_ratios(config: Mapping[str, Any]) -> None:
        data = config["data"]
        train_ratio = float(data.get("train_ratio", 0.0))
        val_ratio = float(data.get("val_ratio", 0.0))
        test_ratio = float(data.get("test_ratio", 0.0))
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"Data split ratios must sum to 1.0, got {ratio_sum:.6f} "
                f"({train_ratio}, {val_ratio}, {test_ratio})."
            )

    @staticmethod
    def _validate_paths(config: Mapping[str, Any]) -> None:
        data = config["data"]
        for key in ("rgb_dir", "mask_dir"):
            value = data.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(f"Invalid path value for data.{key}: {value!r}")


def resolve_config(base_config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge overrides into a base configuration."""
    return _deep_merge(dict(base_config), dict(overrides))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(f"Config file must contain a mapping: {path}")
    return loaded


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance and validation.

    Inheritance is declared via an optional top-level key:
    ``inherits: configs/default.yaml``.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigValidationError(f"Configuration file not found: {config_path}")

    loaded = _load_yaml(config_path)
    parent_ref = loaded.pop("inherits", None)

    if parent_ref:
        parent_path = (config_path.parent / str(parent_ref)).resolve()
        if not parent_path.exists():
            raise ConfigValidationError(
                f"Parent config '{parent_ref}' declared in {config_path} was not found."
            )
        base_config = load_config(str(parent_path))
        resolved = resolve_config(base_config, loaded)
    else:
        resolved = loaded

    ConfigValidator().validate(resolved)
    return resolved


def config_to_flat_dict(config: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings for logger-friendly output."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(config_to_flat_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def save_resolved_config(config: Mapping[str, Any], output_path: str) -> None:
    """Persist resolved config as YAML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
