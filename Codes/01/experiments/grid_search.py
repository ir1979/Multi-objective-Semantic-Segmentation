"""Grid search runner for systematic hyperparameter experimentation."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from logging_utils.grid_search_logger import GridSearchLogger
from logging_utils.logger import DualLogger
from utils.error_handling import ErrorHandler, RecoveryStrategy


@dataclass
class GridPoint:
    """Single configuration point in grid search space."""

    point_id: int
    params: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    result_dir: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "point_id": self.point_id,
            "params": self.params,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "result_dir": self.result_dir,
            "metrics": self.metrics,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> GridPoint:
        """Deserialize from dictionary."""
        return GridPoint(
            point_id=data["point_id"],
            params=data["params"],
            status=data.get("status", "pending"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            error_traceback=data.get("error_traceback"),
            result_dir=data.get("result_dir"),
            metrics=data.get("metrics", {}),
        )


class GridSearchState:
    """Persistent state management for grid search resumability."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.points: Dict[int, GridPoint] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        """Load existing state or initialize new one."""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.points = {int(k): GridPoint.from_dict(v) for k, v in data.items()}

    def save(self) -> None:
        """Persist state to disk."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v.to_dict() for k, v in self.points.items()},
                f,
                indent=2,
                default=str,
            )

    def add_point(self, point: GridPoint) -> None:
        """Add or update a grid point."""
        self.points[point.point_id] = point
        self.save()

    def get_point(self, point_id: int) -> Optional[GridPoint]:
        """Retrieve a grid point."""
        return self.points.get(point_id)

    def get_pending_points(self) -> List[GridPoint]:
        """Get all pending points."""
        return [p for p in self.points.values() if p.status == "pending"]

    def get_failed_points(self) -> List[GridPoint]:
        """Get all failed points."""
        return [p for p in self.points.values() if p.status == "failed"]

    def get_completed_points(self) -> List[GridPoint]:
        """Get all completed points."""
        return [p for p in self.points.values() if p.status == "completed"]

    def summary(self) -> Dict[str, int]:
        """Get status summary."""
        return {
            "total": len(self.points),
            "pending": len([p for p in self.points.values() if p.status == "pending"]),
            "running": len([p for p in self.points.values() if p.status == "running"]),
            "completed": len([p for p in self.points.values() if p.status == "completed"]),
            "failed": len([p for p in self.points.values() if p.status == "failed"]),
            "skipped": len([p for p in self.points.values() if p.status == "skipped"]),
        }


class GridSearchConfig:
    """Parser and validator for grid search configuration."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.config = config_dict
        self.grid_cfg = dict(config_dict.get("grid_search", {}))
        self.param_space = self.grid_cfg.get("parameters", {})
        self.constraints = self.grid_cfg.get("constraints", [])
        self.selection = self.grid_cfg.get("selection", {})

    def generate_points(self) -> List[Dict[str, Any]]:
        """Generate grid points from parameter space."""
        if not self.param_space:
            raise ValueError("No parameters defined in grid search configuration")

        # Generate all combinations (Cartesian product)
        param_names = sorted(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]
        all_combinations = list(product(*param_values))

        # Convert to dictionaries
        points = [dict(zip(param_names, combo)) for combo in all_combinations]

        # Apply constraints
        points = self._apply_constraints(points)

        # Apply selection strategy
        points = self._apply_selection(points)

        return points

    def _apply_constraints(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter points based on constraints."""
        filtered = []
        for point in points:
            skip = False
            for constraint in self.constraints:
                # Simple if-then constraint checking
                if "then_skip" in constraint:
                    conditions_met = all(
                        point.get(key) == value
                        for key, value in constraint.items()
                        if key.startswith("if_")
                    )
                    if conditions_met:
                        skip = True
                        break

                # If-then-not constraint
                if "then_not_" in str(constraint):
                    if_key = next((k for k in constraint.keys() if k.startswith("if_")), None)
                    if if_key:
                        if_param = if_key.replace("if_", "")
                        then_not_key = next((k for k in constraint.keys() if k.startswith("then_not_")), None)
                        if then_not_key:
                            then_param = then_not_key.replace("then_not_", "")
                            if point.get(if_param) == constraint[if_key]:
                                point[then_param] = not constraint[then_not_key]

            if not skip:
                filtered.append(point)

        return filtered

    def _apply_selection(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply point selection strategy."""
        strategy = self.selection.get("strategy", "full").lower()

        if strategy == "full":
            return points
        elif strategy == "random":
            n_points = self.selection.get("n_points", min(50, len(points)))
            seed = self.selection.get("random_seed", 42)
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(points), size=min(n_points, len(points)), replace=False)
            return [points[i] for i in sorted(indices)]
        elif strategy == "latin_hypercube":
            # Simple LH sampling approximation
            n_points = self.selection.get("n_points", min(50, len(points)))
            seed = self.selection.get("random_seed", 42)
            rng = np.random.RandomState(seed)
            step = max(1, len(points) // n_points)
            shuffled = rng.permutation(len(points))
            indices = shuffled[::step][:n_points]
            return [points[i] for i in sorted(indices)]
        else:
            return points


@dataclass
class GridSearchRunner:
    """Main grid search orchestrator with error handling and monitoring."""

    config_path: str
    results_dir: str = "grid_search_results"
    resume: bool = True
    max_retries: int = 3
    logger: Optional[GridSearchLogger] = None
    error_handler: Optional[ErrorHandler] = None
    recovery_strategy: Optional[RecoveryStrategy] = None

    def __post_init__(self) -> None:
        """Initialize runner."""
        self.results_root = Path(self.results_dir)
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.grid_config = GridSearchConfig(self.config)

        # Initialize logger
        log_file = self.results_root / "grid_search.log"
        if self.logger is None:
            self.logger = DualLogger(log_file, console_level="INFO", file_level="DEBUG")

        # Initialize state management
        self.state = GridSearchState(self.results_root / "grid_search_state.json")

        self.logger.info(f"Grid Search Runner initialized at {self.results_root}")

    def initialize_grid(self, force: bool = False) -> None:
        """Generate or resume grid points."""
        if force or not self.state.points:
            self.logger.info("Generating new grid points...")
            points_data = self.grid_config.generate_points()
            self.state.points = {
                i: GridPoint(point_id=i, params=params) for i, params in enumerate(points_data)
            }
            self.state.save()
            self.logger.info(f"Generated {len(self.state.points)} grid points")
        else:
            self.logger.info(f"Resuming with {len(self.state.points)} existing points")

        summary = self.state.summary()
        self.logger.info(
            f"Grid Summary: Total={summary['total']}, "
            f"Pending={summary['pending']}, Completed={summary['completed']}, "
            f"Failed={summary['failed']}, Skipped={summary['skipped']}"
        )

    def get_point_config(self, base_config: Dict[str, Any], point: GridPoint) -> Dict[str, Any]:
        """Merge grid point parameters into base configuration."""
        cfg = json.loads(json.dumps(base_config))

        # Map grid parameters to config paths
        param_mapping = {
            "model_architecture": ("model", "architecture"),
            "pixel_loss_type": ("loss", "pixel", "type"),
            "boundary_loss_weight": ("loss", "boundary", "weight"),
            "shape_loss_weight": ("loss", "shape", "weight"),
            "learning_rate": ("training", "learning_rate"),
            "encoder_filters": ("model", "encoder_filters"),
            "dropout_rate": ("model", "dropout_rate"),
            "deep_supervision": ("model", "deep_supervision"),
            "batch_size": ("data", "batch_size"),
            "loss_strategy": ("loss", "strategy"),
        }

        for param_name, param_value in point.params.items():
            if param_name in param_mapping:
                path = param_mapping[param_name]
                self._set_nested_config(cfg, path, param_value)

        return cfg

    @staticmethod
    def _set_nested_config(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
        """Set nested config value."""
        current = cfg
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def run_point(self, point: GridPoint) -> bool:
        """Execute a single grid point with error handling."""
        point.status = "running"
        point.started_at = datetime.utcnow().isoformat()
        self.state.add_point(point)

        try:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Running Grid Point {point.point_id}")
            self.logger.info(f"Parameters: {json.dumps(point.params, indent=2)}")
            self.logger.info(f"{'=' * 80}")

            # Create result directory
            result_dir = self.results_root / f"point_{point.point_id:06d}"
            result_dir.mkdir(parents=True, exist_ok=True)
            point.result_dir = str(result_dir)

            # Get merged config
            merged_config = self.get_point_config(self.config, point)

            # Save config for this point
            config_file = result_dir / "config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(merged_config, f, default_flow_style=False)

            # TODO: Run the actual training here
            # This would call the experiment runner or trainer
            # For now, we'll just simulate it
            point.metrics = {
                "train_loss": float(np.random.random()),
                "val_iou": float(np.random.random()),
                "test_iou": float(np.random.random()),
            }

            self.logger.info(f"Completed Point {point.point_id} | Metrics: {point.metrics}")
            point.status = "completed"
            point.completed_at = datetime.utcnow().isoformat()
            self.state.add_point(point)
            return True

        except Exception as exc:
            self.logger.error(f"FAILED Point {point.point_id}: {exc}")
            self.logger.exception(f"Traceback for Point {point.point_id}")
            point.status = "failed"
            point.completed_at = datetime.utcnow().isoformat()
            point.error_message = str(exc)
            point.error_traceback = traceback.format_exc()
            self.state.add_point(point)
            return False

    def run_search(self, start_from_pending: bool = True) -> None:
        """Execute the full grid search with resumability."""
        self.initialize_grid()

        start_time = time.time()
        points_to_run = self.state.get_pending_points() if start_from_pending else list(self.state.points.values())

        self.logger.info(f"Starting grid search with {len(points_to_run)} points to evaluate")

        for point in points_to_run:
            self.run_point(point)

            # Log intermediate summary
            summary = self.state.summary()
            self.logger.info(
                f"Progress: {summary['completed']}/{summary['total']} completed, "
                f"{summary['failed']} failed, {summary['pending']} pending"
            )

        elapsed = time.time() - start_time
        summary = self.state.summary()

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("GRID SEARCH SUMMARY")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total Time: {elapsed:.2f} seconds")
        self.logger.info(f"Total Points: {summary['total']}")
        self.logger.info(f"Completed: {summary['completed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Pending: {summary['pending']}")

    def generate_results_report(self) -> Path:
        """Generate aggregated results report."""
        report_file = self.results_root / "grid_search_results.json"

        completed = self.state.get_completed_points()
        results = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_points": len(self.state.points),
            "completed_points": len(completed),
            "results": [p.to_dict() for p in completed],
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results report saved to {report_file}")
        return report_file

    def get_best_point(self, metric: str = "val_iou") -> Optional[GridPoint]:
        """Get best performing point by metric."""
        completed = self.state.get_completed_points()
        if not completed:
            return None

        best = max(
            completed,
            key=lambda p: p.metrics.get(metric, float("-inf")),
            default=None,
        )
        return best
