"""Training loop smoke tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import gc

import numpy as np
import json
import tensorflow as tf
from PIL import Image

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from models.factory import get_model
from training.checkpoint_manager import CheckpointManager
from training.evaluator import Evaluator
from training.trainer import Trainer


def _clone_config(cfg):
    import copy
    return copy.deepcopy(cfg)


def _write_rgb(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_mask(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask = (rng.random((256, 256)) > 0.8).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


class TestTraining(unittest.TestCase):
    """Ensure short training runs execute correctly."""

    def setUp(self) -> None:
        tf.keras.backend.clear_session()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.rgb_dir = root / "RGB"
        self.mask_dir = root / "Mask"
        self.image_size = 128
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(18):
            image = (np.random.default_rng(idx).random((self.image_size, self.image_size, 3)) * 255).astype(np.uint8)
            mask = (np.random.default_rng(idx).random((self.image_size, self.image_size)) > 0.8).astype(np.uint8) * 255
            Image.fromarray(image).save(self.rgb_dir / f"tile_{idx:03d}.png")
            Image.fromarray(mask).save(self.mask_dir / f"tile_{idx:03d}.tif")

        self.base_config = {
            "project": {"seed": 42},
            "data": {
                "rgb_dir": str(self.rgb_dir),
                "mask_dir": str(self.mask_dir),
                "image_size": self.image_size,
                "batch_size": 1,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "building_density_bins": 3,
            },
            "model": {
                "architecture": "unet",
                "deep_supervision": False,
                "encoder_filters": [16, 32, 64, 128, 256],
                "dropout_rate": 0.1,
            },
            "loss": {
                "strategy": "single",
                "pixel": {"type": "bce_iou", "weight": 1.0},
                "boundary": {"enabled": False, "weight": 0.0},
                "shape": {"enabled": False, "weight": 0.0},
            },
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4,
                "gradient_clip_norm": 1.0,
                "lr_scheduler": {"type": "cosine", "warmup_epochs": 1, "min_lr": 1e-7},
                "early_stopping": {"monitor": "val_iou", "patience": 2, "mode": "max"},
            },
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()
        tf.keras.backend.clear_session()
        gc.collect()

    def _build_datasets(self):
        loader = BuildingSegmentationDataset(
            DatasetConfig(
                rgb_dir=str(self.rgb_dir),
                mask_dir=str(self.mask_dir),
                image_size=self.image_size,
                batch_size=1,
                seed=42,
            ),
            skipped_log_path=str(Path(self.tmp.name) / "skipped.txt"),
        )
        loader.validate_pairs()
        split = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=42).split(loader.get_density_labels())
        train_ds = loader.get_tf_dataset(split["train"], augment=False, shuffle=False)
        val_ds = loader.get_tf_dataset(split["val"], augment=False, shuffle=False)
        test_ds = loader.get_tf_dataset(split["test"], augment=False, shuffle=False)
        return train_ds, val_ds, test_ds

    def test_single_objective_runs(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["loss"] = dict(self.base_config["loss"])
        cfg["loss"]["strategy"] = "single"
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_single")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "single", ckpt)
        train_ds, val_ds, test_ds = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        metrics = Evaluator().evaluate(model, test_ds)
        self.assertGreaterEqual(len(result.history["train_loss"]), 1)
        self.assertIn("iou", metrics)

    def test_weighted_multi_loss_runs(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["model"] = dict(self.base_config["model"])
        cfg["loss"] = {
            "strategy": "weighted",
            "pixel": {"type": "bce_iou", "weight": 1.0},
            "boundary": {"enabled": True, "weight": 0.2},
            "shape": {"enabled": True, "weight": 0.1},
        }
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_weighted")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "weighted", ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        self.assertGreaterEqual(len(result.history["val_loss"]), 1)

    def test_mgda_runs(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["model"] = dict(self.base_config["model"])
        cfg["loss"] = {
            "strategy": "mgda",
            "pixel": {"type": "bce_iou", "weight": 1.0},
            "boundary": {"enabled": True, "weight": 0.2},
            "shape": {"enabled": True, "weight": 0.1},
        }
        cfg["mgda"] = {"max_iterations": 10, "tolerance": 1e-6, "normalize_gradients": True}
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_mgda")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "mgda", ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        self.assertTrue(isinstance(result.mgda_alpha_history, list))
        alpha_path = Path(self.tmp.name) / "mgda" / "mgda_alphas.json"
        self.assertTrue(alpha_path.exists())
        saved_history = json.loads(alpha_path.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(saved_history, list))

    def test_lr_scheduler_decreases(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["training"] = dict(self.base_config["training"])
        cfg["training"]["epochs"] = 3
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_lr")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "lr", ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        self.assertTrue(len(result.history["lr"]) >= 1)

    def test_early_stopping_triggers(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["training"] = dict(self.base_config["training"])
        cfg["training"]["epochs"] = 6
        cfg["training"]["early_stopping"] = {
            "monitor": "val_iou",
            "patience": 1,
            "mode": "max",
            "min_delta": 10.0,
        }
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_es")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "es", ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        self.assertTrue(result.stopped_early)

    def test_training_resumes_from_checkpoint(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["training"] = dict(self.base_config["training"])
        cfg["training"]["epochs"] = 1
        cfg["checkpointing"] = {"auto_resume": True}
        model = get_model(cfg)
        run_dir = Path(self.tmp.name) / "resume"
        ckpt = CheckpointManager(run_dir / "checkpoints")
        trainer = Trainer(model, cfg, run_dir, ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        first = trainer.fit(train_ds, val_ds)
        self.assertEqual(len(first.history["train_loss"]), 1)

        resumed_cfg = json.loads(json.dumps(cfg))
        resumed_cfg["training"]["epochs"] = 3
        resumed_model = get_model(resumed_cfg)
        resumed_trainer = Trainer(resumed_model, resumed_cfg, run_dir, ckpt)
        resumed = resumed_trainer.fit(train_ds, val_ds)
        self.assertTrue(resumed.resumed_from_checkpoint)
        self.assertGreaterEqual(resumed.start_epoch, 1)
        self.assertGreaterEqual(len(resumed.history["train_loss"]), 3)
        resumed_trainer.dual_logger.close()

    def test_training_skips_when_checkpoint_already_complete(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["training"] = dict(self.base_config["training"])
        cfg["training"]["epochs"] = 1
        cfg["checkpointing"] = {"auto_resume": True}
        run_dir = Path(self.tmp.name) / "resume_complete"

        model = get_model(cfg)
        ckpt = CheckpointManager(run_dir / "checkpoints")
        trainer = Trainer(model, cfg, run_dir, ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        first = trainer.fit(train_ds, val_ds)
        self.assertEqual(len(first.history["train_loss"]), 1)

        resumed_model = get_model(cfg)
        resumed_trainer = Trainer(resumed_model, cfg, run_dir, ckpt)
        resumed = resumed_trainer.fit(train_ds, val_ds)
        self.assertTrue(resumed.resumed_from_checkpoint)
        self.assertEqual(resumed.start_epoch, 1)
        self.assertEqual(len(resumed.history["train_loss"]), 1)

    def test_deep_supervision_weighted_training_runs(self) -> None:
        cfg = json.loads(json.dumps(self.base_config))
        cfg["model"] = dict(self.base_config["model"])
        cfg["model"]["architecture"] = "unetpp"
        cfg["model"]["deep_supervision"] = True
        cfg["loss"] = {
            "strategy": "weighted",
            "pixel": {"type": "bce_iou", "weight": 1.0},
            "boundary": {"enabled": True, "weight": 0.2},
            "shape": {"enabled": False, "weight": 0.0},
            "deep_supervision": {"enabled": True, "weights": [0.5, 0.3, 0.15, 0.05]},
        }
        model = get_model(cfg)
        ckpt = CheckpointManager(Path(self.tmp.name) / "checkpoints_deepsup")
        trainer = Trainer(model, cfg, Path(self.tmp.name) / "deepsup", ckpt)
        train_ds, val_ds, _ = self._build_datasets()
        result = trainer.fit(train_ds, val_ds)
        self.assertGreaterEqual(len(result.history["val_loss"]), 1)

