"""Grid search runner for systematic hyperparameter experimentation."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from logging_utils.grid_search_logger import GridSearchLogger
from logging_utils.logger import DualLogger
from models.factory import get_model
from optimization.search_strategy import get_search_strategy
from training.checkpoint_manager import CheckpointManager
from training.evaluator import Evaluator
from training.trainer import Trainer
from utils.config_loader import load_config
from utils.error_handling import ErrorHandler, RecoveryStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_prediction_samples(
    model: Any,
    dataset: Any,
    n_samples: int,
    save_dir: Path,
    point_label: str = "",
) -> None:
    """Save RGB / GT-mask / prediction / TP-FP-FN overlay panels for each sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf  # imported here to avoid circular TF init at module level

    save_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for x_batch, y_batch in dataset:
        y_pred = model(x_batch, training=False)
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[-1]
        y_pred_np = (y_pred.numpy() > 0.5).astype(np.float32)

        for i in range(int(x_batch.shape[0])):
            if count >= n_samples:
                break
            img = np.clip(x_batch[i].numpy(), 0.0, 1.0)
            gt = y_batch[i].numpy().squeeze()
            pred = y_pred_np[i].squeeze()

            # TP/FP/FN colour overlay
            overlay = img.copy()
            tp = (gt > 0.5) & (pred > 0.5)
            fp = (gt <= 0.5) & (pred > 0.5)
            fn = (gt > 0.5) & (pred <= 0.5)
            overlay[tp] = overlay[tp] * 0.35 + np.array([0.0, 0.82, 0.0]) * 0.65
            overlay[fp] = overlay[fp] * 0.35 + np.array([0.9, 0.1, 0.1]) * 0.65
            overlay[fn] = overlay[fn] * 0.35 + np.array([0.1, 0.1, 0.9]) * 0.65

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(img);              axes[0].set_title("RGB Input",     fontsize=9); axes[0].axis("off")
            axes[1].imshow(gt,   cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground Truth",  fontsize=9); axes[1].axis("off")
            axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction",    fontsize=9); axes[2].axis("off")
            axes[3].imshow(overlay);          axes[3].set_title("TP/FP/FN\n(G/R/B)", fontsize=9); axes[3].axis("off")

            if point_label:
                fig.suptitle(point_label, fontsize=9, y=1.01)
            plt.tight_layout()
            fig.savefig(save_dir / f"sample_{count:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1
        if count >= n_samples:
            break


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridPoint:
    """Single configuration point in grid search space."""

    point_id: int
    params: Dict[str, Any]
    status: str = "pending"  # pending | running | completed | failed | skipped
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    result_dir: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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


class _StateBackend(ABC):
    """Backend interface for persistent grid-search state."""

    @abstractmethod
    def load(self) -> Dict[int, Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, payload: Dict[int, Dict[str, Any]]) -> None:
        pass


class _JsonStateBackend(_StateBackend):
    """JSON backend with atomic write semantics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[int, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid state payload in JSON backend: {self.path}")
        return {int(k): v for k, v in raw.items()}

    def save(self, payload: Dict[int, Dict[str, Any]]) -> None:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.path.stem}_",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
                json.dump({str(k): v for k, v in payload.items()}, tmp, indent=2, default=str)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, str(self.path))
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


class _SqliteStateBackend(_StateBackend):
    """SQLite backend for larger experiments and transactional durability."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS grid_points (
                    point_id INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def load(self) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        conn = self._connect()
        try:
            rows = conn.execute("SELECT point_id, payload FROM grid_points").fetchall()
            for pid, payload in rows:
                out[int(pid)] = json.loads(payload)
        finally:
            conn.close()
        return out

    def save(self, payload: Dict[int, Dict[str, Any]]) -> None:
        ts = datetime.utcnow().isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM grid_points")
                conn.executemany(
                    "INSERT INTO grid_points(point_id, payload, updated_at) VALUES (?, ?, ?)",
                    [(int(pid), json.dumps(data, default=str), ts) for pid, data in payload.items()],
                )
        finally:
            conn.close()


class GridSearchState:
    """Persistent state management with JSON/SQLite backends."""

    def __init__(
        self,
        state_file: Path,
        backend: str = "json",
        checkpoint_interval: int = 1,
    ) -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.points: Dict[int, GridPoint] = {}
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self._updates_since_save = 0
        self.backend_name = str(backend).lower()
        self._backend: _StateBackend = self._build_backend(self.backend_name)
        self._load_or_init()

    def _build_backend(self, backend: str) -> _StateBackend:
        if backend == "json":
            return _JsonStateBackend(self.state_file)
        if backend == "sqlite":
            db_path = self.state_file.with_suffix(".sqlite")
            return _SqliteStateBackend(db_path)
        raise ValueError(f"Unsupported state backend '{backend}'. Use 'json' or 'sqlite'.")

    def _validate_integrity(self) -> None:
        for pid, point in self.points.items():
            if point.point_id != int(pid):
                raise ValueError(f"State integrity error at key={pid}: point_id mismatch")

    def _load_or_init(self) -> None:
        data = self._backend.load()
        self.points = {int(k): GridPoint.from_dict(v) for k, v in data.items()}
        self._validate_integrity()

    def save(self, force: bool = False) -> None:
        self._updates_since_save += 1
        if not force and self._updates_since_save < self.checkpoint_interval:
            return
        payload = {int(k): v.to_dict() for k, v in self.points.items()}
        self._backend.save(payload)
        self._updates_since_save = 0

    def add_point(self, point: GridPoint) -> None:
        self.points[point.point_id] = point
        self.save()

    def flush(self) -> None:
        self.save(force=True)

    def get_point(self, point_id: int) -> Optional[GridPoint]:
        return self.points.get(point_id)

    def get_pending_points(self) -> List[GridPoint]:
        return [p for p in self.points.values() if p.status == "pending"]

    def get_failed_points(self) -> List[GridPoint]:
        return [p for p in self.points.values() if p.status == "failed"]

    def get_completed_points(self) -> List[GridPoint]:
        return [p for p in self.points.values() if p.status == "completed"]

    def summary(self) -> Dict[str, int]:
        return {
            "total": len(self.points),
            "pending": sum(1 for p in self.points.values() if p.status == "pending"),
            "running": sum(1 for p in self.points.values() if p.status == "running"),
            "completed": sum(1 for p in self.points.values() if p.status == "completed"),
            "failed": sum(1 for p in self.points.values() if p.status == "failed"),
            "skipped": sum(1 for p in self.points.values() if p.status == "skipped"),
        }


class GridSearchConfig:
    """Parse and generate the Cartesian-product grid with constraints and sampling."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.config = config_dict
        self.grid_cfg = dict(config_dict.get("grid_search", {}))
        self.param_space = self.grid_cfg.get("parameters", {})
        self.constraints = self.grid_cfg.get("constraints", [])
        self.selection = self.grid_cfg.get("selection", {})

    def generate_points(self) -> List[Dict[str, Any]]:
        if not self.param_space:
            raise ValueError("No parameters defined in grid search configuration")
        # Skip parameters whose value list is empty — they represent disabled dimensions.
        active_space = {k: v for k, v in self.param_space.items() if v}
        if not active_space:
            raise ValueError("No parameters defined in grid search configuration")
        param_names = sorted(active_space.keys())
        param_values = [active_space[name] for name in param_names]
        all_combinations = list(product(*param_values))
        points = [dict(zip(param_names, combo)) for combo in all_combinations]
        points = self._apply_constraints(points)
        points = self._apply_selection(points)
        return points

    def _apply_constraints(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for point in points:
            skip = False
            for constraint in self.constraints:
                if "then_skip" in constraint:
                    # Support both if_ and and_ prefixes as additional conditions
                    cond_keys = [k for k in constraint if k.startswith("if_") or k.startswith("and_")]
                    conditions_met = all(
                        point.get(k.replace("if_", "").replace("and_", "")) == v
                        for k, v in constraint.items()
                        if k in cond_keys
                    )
                    if conditions_met:
                        skip = True
                        break
            if not skip:
                filtered.append(point)
        return filtered

    def _apply_selection(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        strategy_name = str(self.selection.get("strategy", "full")).lower()
        strategy = get_search_strategy(strategy_name)
        return strategy.select(points, dict(self.selection))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class GridSearchRunner:
    """Main grid search orchestrator with error handling and resumability."""

    config_path: str
    results_dir: str = "grid_search_results"
    resume: bool = True
    max_retries: int = 3
    logger: Optional[GridSearchLogger] = None
    error_handler: Optional[ErrorHandler] = None
    recovery_strategy: Optional[RecoveryStrategy] = None

    def __post_init__(self) -> None:
        self.results_root = Path(self.results_dir)
        self.results_root.mkdir(parents=True, exist_ok=True)
        # Use shared config loader so `inherits` chains are fully resolved.
        self.config = load_config(self.config_path)
        self.grid_config = GridSearchConfig(self.config)
        _gs_block = dict(self.config.get("grid_search", {}))
        persistence_cfg = dict(_gs_block.get("persistence", {}))
        state_backend = str(persistence_cfg.get("backend") or self.config.get("grid_search_persistence_backend", "json")).lower()
        checkpoint_interval = int(persistence_cfg.get("checkpoint_interval") or self.config.get("grid_search_persistence_checkpoint_interval", 1))
        log_file = self.results_root / "grid_search.log"
        if self.logger is None:
            self.logger = DualLogger(log_file, console_level="INFO", file_level="DEBUG")
        self.state = GridSearchState(
            self.results_root / "grid_search_state.json",
            backend=state_backend,
            checkpoint_interval=checkpoint_interval,
        )
        self.logger.info(f"Grid Search Runner initialised at {self.results_root}")

    def initialize_grid(self, force: bool = False) -> None:
        if force or not self.state.points:
            self.logger.info("Generating new grid points …")
            points_data = self.grid_config.generate_points()
            self.state.points = {
                i: GridPoint(point_id=i, params=params)
                for i, params in enumerate(points_data)
            }
            self.state.flush()
            self.logger.info(f"Generated {len(self.state.points)} grid points")
        else:
            self.logger.info(f"Resuming with {len(self.state.points)} existing points")
        s = self.state.summary()
        self.logger.info(
            f"Grid summary – total={s['total']} pending={s['pending']} "
            f"completed={s['completed']} failed={s['failed']} skipped={s['skipped']}"
        )

    # ------------------------------------------------------------------
    # Config merging
    # ------------------------------------------------------------------

    def get_point_config(self, base_config: Dict[str, Any], point: GridPoint) -> Dict[str, Any]:
        """Merge grid point parameters into a full flat training config."""
        cfg = json.loads(json.dumps(base_config))

        # Direct parameter → flat config key mapping
        param_mapping: Dict[str, str] = {
            "model_architecture":   "model_architecture",
            "encoder_filters":      "model_encoder_filters",
            "pixel_loss_type":      "loss_pixel_type",
            "boundary_loss_weight": "loss_boundary_weight",
            "shape_loss_weight":    "loss_shape_weight",
            "learning_rate":        "training_learning_rate",
        }
        for param_name, param_value in point.params.items():
            if param_name in param_mapping:
                cfg[param_mapping[param_name]] = param_value

        # Derived: enable/disable boundary and shape terms based on weight
        b_weight = float(point.params.get("boundary_loss_weight",
                         cfg.get("loss_boundary_weight", 0.0)))
        s_weight = float(point.params.get("shape_loss_weight",
                         cfg.get("loss_shape_weight", 0.0)))
        cfg["loss_boundary_enabled"] = b_weight > 0.0
        cfg["loss_shape_enabled"]    = s_weight > 0.0
        cfg["loss_strategy"]         = "weighted"

        return cfg

    @staticmethod
    def _set_nested_config(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
        cur = cfg
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_point(self, point: GridPoint) -> bool:
        """Train and evaluate a single grid point; save metrics and predictions."""
        point.status = "running"
        point.started_at = datetime.utcnow().isoformat()
        self.state.add_point(point)

        try:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Grid Point {point.point_id:4d}  |  params: {json.dumps(point.params)}")
            self.logger.info(f"{'=' * 80}")

            result_dir = self.results_root / f"point_{point.point_id:06d}"
            result_dir.mkdir(parents=True, exist_ok=True)
            point.result_dir = str(result_dir)

            merged_config = self.get_point_config(self.config, point)

            # Persist resolved config for reproducibility
            with open(result_dir / "config.yaml", "w", encoding="utf-8") as fh:
                yaml.dump(merged_config, fh, default_flow_style=False)

            # ------ Data ------------------------------------------------
            _seed = int(merged_config.get("project_seed", 42))

            loader = BuildingSegmentationDataset(
                DatasetConfig(
                    rgb_dir=str(merged_config["data_rgb_dir"]),
                    mask_dir=str(merged_config["data_mask_dir"]),
                    image_size=int(merged_config.get("data_image_size", 256)),
                    batch_size=int(merged_config.get("data_batch_size", 4)),
                    num_workers=int(merged_config.get("data_num_workers", 4)),
                    prefetch_buffer=int(merged_config.get("data_prefetch_buffer", 2)),
                    seed=_seed,
                ),
                skipped_log_path=str(result_dir / "skipped_files.txt"),
            )
            loader.validate_pairs()

            splitter = StratifiedSplitter(
                train_ratio=float(merged_config.get("data_train_ratio", 0.7)),
                val_ratio=float(merged_config.get("data_val_ratio", 0.15)),
                test_ratio=float(merged_config.get("data_test_ratio", 0.15)),
                bins=int(merged_config.get("data_building_density_bins", 3)),
                seed=_seed,
            )
            split = splitter.split(loader.get_density_labels())
            train_ds = loader.get_tf_dataset(split["train"], augment=bool(merged_config.get("augmentation_enabled", True)), shuffle=True)
            val_ds   = loader.get_tf_dataset(split["val"],   augment=False, shuffle=False)
            test_ds  = loader.get_tf_dataset(split["test"],  augment=False, shuffle=False)

            self.logger.info(
                f"Point {point.point_id} | pairs={len(loader.pairs)} "
                f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
            )

            # ------ Train -----------------------------------------------
            model = get_model(merged_config)
            ckpt_mgr = CheckpointManager(result_dir / "checkpoints")
            trainer  = Trainer(model, merged_config, result_dir, ckpt_mgr)
            result   = trainer.fit(train_ds, val_ds)

            self.logger.info(
                f"Point {point.point_id} | best_epoch={result.best_epoch} "
                f"best_val_iou={float(result.best_metric):.4f} "
                f"early_stop={result.stopped_early}"
            )

            # ------ Evaluate --------------------------------------------
            evaluator    = Evaluator()
            test_metrics = evaluator.evaluate(model, test_ds)
            history      = result.history

            point.metrics = {
                "train_loss":       float(history.get("train_loss", [0.0])[-1]),
                "val_iou":          float(result.best_metric),
                "test_iou":         float(test_metrics.get("iou",          0.0)),
                "test_dice":        float(test_metrics.get("dice",         0.0)),
                "test_precision":   float(test_metrics.get("precision",    0.0)),
                "test_recall":      float(test_metrics.get("recall",       0.0)),
                "test_pixel_acc":   float(test_metrics.get("pixel_accuracy", 0.0)),
                "test_boundary_iou":float(test_metrics.get("boundary_iou", 0.0)),
                "test_boundary_f1": float(test_metrics.get("boundary_f1",  0.0)),
                "test_compactness": float(test_metrics.get("compactness",  0.0)),
                "best_epoch":       int(result.best_epoch),
                "total_epochs":     int(len(history.get("train_loss", []))),
                "stopped_early":    bool(result.stopped_early),
            }

            self.logger.info(
                f"Point {point.point_id} | test_iou={point.metrics['test_iou']:.4f} "
                f"boundary_f1={point.metrics['test_boundary_f1']:.4f} "
                f"compactness={point.metrics['test_compactness']:.4f}"
            )

            # ------ Save prediction images ------------------------------
            if merged_config.get("export_save_predictions", True):
                n_samples   = int(merged_config.get("export_n_prediction_samples", 8))
                arch        = point.params.get("model_architecture", "?")
                depth       = "deep" if len(point.params.get("encoder_filters", [])) > 4 and \
                               point.params.get("encoder_filters", [0])[-1] >= 1024 else "shallow"
                loss_lbl    = point.params.get("pixel_loss_type", "?")
                b_w         = point.params.get("boundary_loss_weight", 0.0)
                s_w         = point.params.get("shape_loss_weight",    0.0)
                lr_lbl      = point.params.get("learning_rate", 0.0)
                point_label = (
                    f"Point {point.point_id}: {arch} | {depth} | loss={loss_lbl} "
                    f"| b={b_w} s={s_w} | lr={lr_lbl}"
                )
                _save_prediction_samples(
                    model, test_ds, n_samples,
                    result_dir / "predictions",
                    point_label=point_label,
                )
                self.logger.info(f"Point {point.point_id} | Saved {n_samples} prediction images")

            point.status = "completed"
            point.completed_at = datetime.utcnow().isoformat()
            self.state.add_point(point)

            # Release GPU memory between points to reduce fragmentation
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
            return True

        except Exception as exc:
            exc_type = type(exc).__name__
            exc_str = str(exc)
            is_oom = (
                "ResourceExhausted" in exc_type
                or "OOM" in exc_str
                or "out of memory" in exc_str.lower()
                or "allocat" in exc_str.lower() and "memory" in exc_str.lower()
            )
            if is_oom:
                self.logger.warning(
                    f"Point {point.point_id}: GPU out of memory — "
                    f"skipping point and freeing GPU memory."
                )
                # Free GPU memory so subsequent points can run
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
                gc.collect()
                point.error_message = f"[OOM] {exc_str}"
            else:
                self.logger.error(f"FAILED Point {point.point_id}: {exc}")
                self.logger.exception(f"Traceback for Point {point.point_id}")
                point.error_message = exc_str
            point.status = "failed"
            point.completed_at = datetime.utcnow().isoformat()
            point.error_traceback = traceback.format_exc()
            self.state.add_point(point)
            return False

    def run_search(self, start_from_pending: bool = True) -> None:
        """Execute the full grid search with resumability."""
        self.initialize_grid()
        t0 = time.time()
        points_to_run = (
            self.state.get_pending_points() if start_from_pending
            else list(self.state.points.values())
        )
        self.logger.info(f"Starting grid search with {len(points_to_run)} points")
        for point in points_to_run:
            self.run_point(point)
            s = self.state.summary()
            self.logger.info(
                f"Progress: {s['completed']}/{s['total']} completed | "
                f"failed={s['failed']} pending={s['pending']}"
            )
        elapsed = time.time() - t0
        s = self.state.summary()
        self.logger.info(
            f"\nGrid search finished in {elapsed:.1f}s | "
            f"completed={s['completed']} failed={s['failed']} skipped={s['skipped']}"
        )
        self.state.flush()

    def generate_results_report(self) -> Path:
        """Persist aggregated JSON report."""
        report_file = self.results_root / "grid_search_results.json"
        completed = self.state.get_completed_points()
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at":    datetime.utcnow().isoformat(),
                    "total_points":    len(self.state.points),
                    "completed_points": len(completed),
                    "results":         [p.to_dict() for p in completed],
                },
                f, indent=2, default=str,
            )
        self.logger.info(f"Results report → {report_file}")
        return report_file

    def get_best_point(self, metric: str = "test_iou") -> Optional[GridPoint]:
        completed = self.state.get_completed_points()
        if not completed:
            return None
        return max(completed, key=lambda p: p.metrics.get(metric, float("-inf")))

    def get_top_points(self, n: int = 10, metric: str = "test_iou") -> List[GridPoint]:
        completed = self.state.get_completed_points()
        return sorted(completed, key=lambda p: p.metrics.get(metric, float("-inf")), reverse=True)[:n]

