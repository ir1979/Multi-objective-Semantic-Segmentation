"""Configuration loading and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a configuration file fails validation."""


@dataclass
class ConfigValidator:
    """Schema-aware configuration validator."""

    architecture_values: Iterable[str] = field(
        default_factory=lambda: (
            "unet",
            "unetpp",
            "attunet",
            "r2attunet",
            "seunet",
            "scse_unet",
            "resunet",
            "resunetpp",
        )
    )
    strategy_values: Iterable[str] = field(default_factory=lambda: ("single", "weighted"))
    pixel_loss_values: Iterable[str] = field(
        default_factory=lambda: ("bce", "iou", "dice", "focal")
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
        required_keys = ("project_name", "data_rgb_dir", "model_architecture", "loss_strategy")
        for key in required_keys:
            if key not in config:
                raise ConfigValidationError(f"Missing required config key: '{key}'")

        architecture = str(config.get("model_architecture", "")).lower()
        if architecture not in set(self.architecture_values):
            raise ConfigValidationError(
                f"Unknown model architecture '{architecture}'. "
                f"Expected one of {list(self.architecture_values)}."
            )

        strategy = str(config.get("loss_strategy", "")).lower()
        if strategy not in set(self.strategy_values):
            raise ConfigValidationError(
                f"Unknown loss strategy '{strategy}'. Expected {list(self.strategy_values)}."
            )

        pixel_type = str(config.get("loss_pixel_type", "")).lower()
        if pixel_type not in set(self.pixel_loss_values):
            raise ConfigValidationError(
                f"Unknown pixel loss '{pixel_type}'. Expected {list(self.pixel_loss_values)}."
            )

        scheduler = str(config.get("training_lr_scheduler_type", "")).lower()
        if scheduler not in set(self.scheduler_values):
            raise ConfigValidationError(
                f"Unknown scheduler '{scheduler}'. Expected {list(self.scheduler_values)}."
            )

        self._validate_ratios(config)
        self._validate_paths(config)

    @staticmethod
    def _validate_ratios(config: Mapping[str, Any]) -> None:
        train_ratio = float(config.get("data_train_ratio", 0.0))
        val_ratio = float(config.get("data_val_ratio", 0.0))
        test_ratio = float(config.get("data_test_ratio", 0.0))
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"Data split ratios must sum to 1.0, got {ratio_sum:.6f} "
                f"({train_ratio}, {val_ratio}, {test_ratio})."
            )

    @staticmethod
    def _validate_paths(config: Mapping[str, Any]) -> None:
        for key in ("data_rgb_dir", "data_mask_dir"):
            value = config.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(f"Invalid path value for data.{key}: {value!r}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(f"Config file must contain a mapping: {path}")
    return loaded


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigValidationError(f"Configuration file not found: {config_path}")

    loaded = _load_yaml(config_path)
    # Preserve the nested grid_search block before flattening; GridSearchConfig
    # and related code require the full nested structure under this key.
    raw_grid_search = loaded.get("grid_search")
    loaded = config_to_flat_dict(loaded)
    if isinstance(raw_grid_search, dict):
        loaded["grid_search"] = raw_grid_search

    ConfigValidator().validate(loaded)
    return loaded


def config_to_flat_dict(config: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings for logger-friendly output."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}_{key}" if prefix else str(key)
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
