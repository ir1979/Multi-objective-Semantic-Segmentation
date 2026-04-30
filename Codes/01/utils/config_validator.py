"""Configuration validation for grid search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import yaml

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_with_inheritance(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    parent = cfg.pop("inherits", None)
    if not parent:
        return cfg
    import os
    parent_path = os.path.join(os.path.dirname(config_path), str(parent))
    parent_cfg = _load_with_inheritance(parent_path)
    return _deep_merge(parent_cfg, cfg)


class GridSearchConfigValidator:
    """Validate grid search configurations."""

    VALID_STRATEGIES = ["single", "weighted"]
    VALID_ARCHITECTURES = [
        "unet",
        "unetpp",
        "attunet",
        "r2attunet",
        "seunet",
        "scse_unet",
        "resunet",
        "resunetpp",
    ]
    VALID_PIXEL_LOSSES = ["bce", "bce", "dice", "focal"]
    VALID_SELECTION_STRATEGIES = ["full", "grid_search", "random", "latin_hypercube", "nsga2"]
    VALID_STATE_BACKENDS = ["json", "sqlite"]

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate grid search configuration.

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        grid_cfg = config.get("grid_search", {})
        if not grid_cfg.get("enabled", True):
            self.warnings.append("Grid search is disabled in configuration")
            return True, self.errors, self.warnings

        # Validate parameters space
        self._validate_parameters_space(grid_cfg)

        # Validate constraints
        self._validate_constraints(grid_cfg)

        # Validate selection strategy
        self._validate_selection(grid_cfg)

        # Validate training config
        self._validate_training_config(config)
        self._validate_persistence(config)
        self._validate_objectives(config)

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_parameters_space(self, grid_cfg: Dict[str, Any]) -> None:
        """Validate parameter space definition."""
        params = grid_cfg.get("parameters", {})

        if not params:
            self.errors.append("No parameters defined in grid_search.parameters")
            return

        # Validate parameter ranges
        for param_name, param_values in params.items():
            if not isinstance(param_values, list):
                self.errors.append(f"Parameter '{param_name}' must be a list, got {type(param_values)}")
                continue

            if len(param_values) == 0:
                self.errors.append(f"Parameter '{param_name}' has empty value list")
                continue

            # Validate known parameters
            if param_name == "model_architecture":
                for val in param_values:
                    if val not in self.VALID_ARCHITECTURES:
                        self.errors.append(f"Invalid architecture: {val}. Must be one of {self.VALID_ARCHITECTURES}")

            elif param_name == "loss_strategy":
                for val in param_values:
                    if val not in self.VALID_STRATEGIES:
                        self.errors.append(f"Invalid loss strategy: {val}. Must be one of {self.VALID_STRATEGIES}")

            elif param_name == "pixel_loss_type":
                for val in param_values:
                    if val not in self.VALID_PIXEL_LOSSES:
                        self.errors.append(f"Invalid pixel loss: {val}. Must be one of {self.VALID_PIXEL_LOSSES}")

            elif param_name == "learning_rate":
                for val in param_values:
                    if not isinstance(val, (int, float)) or val <= 0:
                        self.errors.append(f"Invalid learning_rate: {val}. Must be positive number")

            elif param_name == "batch_size":
                for val in param_values:
                    if not isinstance(val, int) or val <= 0:
                        self.errors.append(f"Invalid batch_size: {val}. Must be positive integer")

            elif param_name == "dropout_rate":
                for val in param_values:
                    if not isinstance(val, (int, float)) or not (0.0 <= val < 1.0):
                        self.errors.append(f"Invalid dropout_rate: {val}. Must be in [0.0, 1.0)")

            elif param_name == "boundary_loss_weight":
                for val in param_values:
                    if not isinstance(val, (int, float)) or val < 0:
                        self.errors.append(f"Invalid boundary_loss_weight: {val}. Must be non-negative")

            elif param_name == "shape_loss_weight":
                for val in param_values:
                    if not isinstance(val, (int, float)) or val < 0:
                        self.errors.append(f"Invalid shape_loss_weight: {val}. Must be non-negative")

            elif param_name == "deep_supervision":
                for val in param_values:
                    if not isinstance(val, bool):
                        self.errors.append(f"Invalid deep_supervision: {val}. Must be boolean")

            elif param_name == "encoder_filters":
                for val in param_values:
                    if not isinstance(val, list):
                        self.errors.append(f"Invalid encoder_filters: {val}. Must be list of integers")
                    elif not all(isinstance(v, int) and v > 0 for v in val):
                        self.errors.append(f"Invalid encoder_filters: {val}. Must be positive integers")

    def _validate_constraints(self, grid_cfg: Dict[str, Any]) -> None:
        """Validate constraint definitions."""
        constraints = grid_cfg.get("constraints", [])

        if not isinstance(constraints, list):
            self.errors.append("Constraints must be a list")
            return

        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                self.errors.append(f"Constraint {i} is not a dictionary")
                continue

            # Check that constraint has condition keys
            has_condition = any(k.startswith("if_") or k.startswith("and_") for k in constraint.keys())
            has_action = any(k.startswith("then_") for k in constraint.keys())

            if not has_condition or not has_action:
                self.warnings.append(f"Constraint {i} may be malformed (missing conditions or actions)")

    def _validate_selection(self, grid_cfg: Dict[str, Any]) -> None:
        """Validate selection strategy."""
        selection = grid_cfg.get("selection", {})
        strategy = selection.get("strategy", "full").lower()

        if strategy not in self.VALID_SELECTION_STRATEGIES:
            self.errors.append(f"Invalid selection strategy: {strategy}. Must be one of {self.VALID_SELECTION_STRATEGIES}")

        if strategy in ["random", "latin_hypercube", "nsga2"]:
            n_points = selection.get("n_points")
            if n_points is None:
                self.warnings.append(f"Selection strategy '{strategy}' requires 'n_points' parameter")
            elif not isinstance(n_points, int) or n_points <= 0:
                self.errors.append(f"Invalid n_points: {n_points}. Must be positive integer")

    def _validate_persistence(self, config: Dict[str, Any]) -> None:
        """Validate persistence backend and checkpoint settings."""
        persistence = config.get("grid_search_persistence", {})
        if not persistence:
            return
        backend = str(persistence.get("backend", "json")).lower()
        if backend not in self.VALID_STATE_BACKENDS:
            self.errors.append(
                f"Invalid persistence backend: {backend}. Must be one of {self.VALID_STATE_BACKENDS}"
            )
        interval = persistence.get("checkpoint_interval", 1)
        if not isinstance(interval, int) or interval <= 0:
            self.errors.append(
                f"Invalid checkpoint_interval: {interval}. Must be a positive integer"
            )

    def _validate_objectives(self, config: Dict[str, Any]) -> None:
        """Validate optional objective metadata used by Pareto/reporting modules."""
        objectives = config.get("grid_search_objectives", [])
        if not objectives:
            return
        if not isinstance(objectives, list):
            self.errors.append("grid_search.objectives must be a list")
            return
        for idx, obj in enumerate(objectives):
            if not isinstance(obj, dict):
                self.errors.append(f"Objective at index {idx} must be a dictionary")
                continue
            metric = obj.get("metric")
            direction = str(obj.get("direction", "")).lower()
            if not metric or not isinstance(metric, str):
                self.errors.append(f"Objective at index {idx} missing string field 'metric'")
            if direction not in ("max", "min"):
                self.errors.append(
                    f"Objective '{metric}' has invalid direction '{direction}'. Use 'max' or 'min'."
                )

    def _validate_training_config(self, config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        training = config.get("training", {})
        epochs = training.get("epochs")

        if epochs is None:
            self.errors.append("training.epochs must be defined")
        elif not isinstance(epochs, int) or epochs <= 0:
            self.errors.append(f"Invalid epochs: {epochs}. Must be positive integer")

        data = config.get("data", {})
        if not data.get("rgb_dir"):
            self.errors.append("data.rgb_dir must be defined")
        if not data.get("mask_dir"):
            self.errors.append("data.mask_dir must be defined")

    @staticmethod
    def estimate_grid_size(config: Dict[str, Any]) -> int:
        """Estimate total grid size before filtering."""
        grid_cfg = config.get("grid_search", {})
        params = grid_cfg.get("parameters", {})

        total = 1
        for param_values in params.values():
            total *= len(param_values)

        return total

    def print_summary(self) -> None:
        """Print validation summary."""
        if self.errors:
            print("\n❌ VALIDATION ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  VALIDATION WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid!")


def validate_config_file(config_path: str) -> Tuple[bool, List[str], List[str]]:
    """Load and validate a config file."""
    try:
        config = _load_with_inheritance(config_path)
    except Exception as exc:
        return False, [f"Failed to load config: {exc}"], []

    validator = GridSearchConfigValidator()
    is_valid, errors, warnings = validator.validate(config)

    if is_valid:
        grid_size = validator.estimate_grid_size(config)
        selection = config.get("grid_search_selection", {})
        strategy = selection.get("strategy", "full")
        n_points = selection.get("n_points", grid_size)

        print(f"\n📊 Grid Search Estimation:")
        print(f"  - Grid size (before filtering): {grid_size}")
        print(f"  - Selection strategy: {strategy}")
        if strategy != "full":
            print(f"  - Sampled points: {min(n_points, grid_size)}")

    validator.print_summary()
    return is_valid, errors, warnings
