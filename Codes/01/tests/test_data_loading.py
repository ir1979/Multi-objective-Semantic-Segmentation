"""Tests for data loading, pairing, and splitting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.loader import BuildingSegmentationDataset, DatasetConfig
from data.splitter import StratifiedSplitter


def _write_rgb(path: Path, value: int = 128) -> None:
    array = np.full((256, 256, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_mask(path: Path, fill: int = 0) -> None:
    array = np.full((256, 256), fill, dtype=np.uint8)
    Image.fromarray(array).save(path)


class TestDataLoading(unittest.TestCase):
    """Dataset loader and split tests."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rgb_dir = self.root / "RGB"
        self.mask_dir = self.root / "Mask"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(12):
            stem = f"tile_{idx:03d}"
            _write_rgb(self.rgb_dir / f"{stem}.png", value=idx * 10)
            mask_fill = 255 if idx % 3 == 0 else 0
            _write_mask(self.mask_dir / f"{stem}.tif", fill=mask_fill)

        self.config = DatasetConfig(
            rgb_dir=str(self.rgb_dir),
            mask_dir=str(self.mask_dir),
            image_size=256,
            batch_size=4,
            seed=42,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_finds_all_files(self) -> None:
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        pairs = loader.scan_and_pair()
        self.assertEqual(len(pairs), 12)

    def test_pairing_matches_correctly(self) -> None:
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        pairs, orphaned = loader.validate_pairs()
        self.assertFalse(orphaned)
        for rgb_path, mask_path in pairs:
            self.assertEqual(rgb_path.stem, mask_path.stem)

    def test_orphan_detection(self) -> None:
        _write_rgb(self.rgb_dir / "orphan.png", value=10)
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        pairs = loader.scan_and_pair()
        self.assertEqual(len(pairs), 12)
        _, orphaned = loader.validate_pairs()
        self.assertFalse(any(path.stem == "orphan" for path in orphaned))

    def test_mask_binarization(self) -> None:
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        loader.validate_pairs()
        image, mask = loader.load_sample(loader.pairs[0][0], loader.pairs[0][1])
        self.assertEqual(image.shape, (256, 256, 3))
        self.assertEqual(mask.shape, (256, 256, 1))
        self.assertTrue(set(np.unique(mask).tolist()).issubset({0.0, 1.0}))

    def test_output_dtypes_and_ranges(self) -> None:
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        loader.validate_pairs()
        image, mask = loader.load_sample(loader.pairs[1][0], loader.pairs[1][1])
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(mask.dtype, np.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_stratified_split_ratios(self) -> None:
        loader = BuildingSegmentationDataset(self.config, skipped_log_path=str(self.root / "skipped.txt"))
        loader.validate_pairs()
        densities = loader.get_density_labels()
        splitter = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=42)
        split = splitter.split(densities)
        total = len(densities)
        self.assertEqual(len(split["train"]) + len(split["val"]) + len(split["test"]), total)

    def test_split_determinism(self) -> None:
        densities = np.linspace(0.0, 1.0, 50)
        splitter_a = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=123)
        splitter_b = StratifiedSplitter(0.7, 0.15, 0.15, bins=3, seed=123)
        self.assertEqual(splitter_a.split(densities), splitter_b.split(densities))
