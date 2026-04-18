"""Reproducibility tests for seed-controlled behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter
from models.unet import UNet
from utils.reproducibility import compute_dataset_hash, set_global_seed


def _write_rgb(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_mask(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask = (rng.random((256, 256)) > 0.85).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


class TestReproducibility(unittest.TestCase):
    """Ensure deterministic behavior under fixed seeds."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rgb_dir = self.root / "RGB"
        self.mask_dir = self.root / "Mask"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(12):
            _write_rgb(self.rgb_dir / f"tile_{idx:03d}.png", idx)
            _write_mask(self.mask_dir / f"tile_{idx:03d}.tif", idx)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _load_densities(self):
        loader = BuildingSegmentationDataset(
            DatasetConfig(
                rgb_dir=str(self.rgb_dir),
                mask_dir=str(self.mask_dir),
                image_size=256,
                batch_size=2,
                seed=42,
            ),
            skipped_log_path=str(self.root / "skipped.txt"),
        )
        loader.validate_pairs()
        return loader.get_density_labels()

    def test_same_seed_same_split(self) -> None:
        densities = self._load_densities()
        split_a = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=42).split(densities)
        split_b = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=42).split(densities)
        self.assertEqual(split_a, split_b)

    def test_same_seed_same_init(self) -> None:
        set_global_seed(123)
        model_a = UNet()
        weights_a = model_a(tf.random.uniform((1, 256, 256, 3)), training=False).numpy()
        set_global_seed(123)
        model_b = UNet()
        weights_b = model_b(tf.random.uniform((1, 256, 256, 3)), training=False).numpy()
        self.assertTrue(np.allclose(weights_a, weights_b))

    def test_same_seed_same_augmentation(self) -> None:
        set_global_seed(99)
        image = tf.random.uniform((1, 256, 256, 3))
        set_global_seed(99)
        image_again = tf.random.uniform((1, 256, 256, 3))
        self.assertTrue(np.allclose(image.numpy(), image_again.numpy()))

    def test_dataset_hash_consistent(self) -> None:
        hash_a = compute_dataset_hash(str(self.root))
        hash_b = compute_dataset_hash(str(self.root))
        self.assertEqual(hash_a, hash_b)

    def test_different_seed_different_results(self) -> None:
        densities = self._load_densities()
        split_a = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=1).split(densities)
        split_b = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=2).split(densities)
        self.assertNotEqual(split_a, split_b)
