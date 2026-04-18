"""Dataset loader for building segmentation with robust pairing and tf.data output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from data.augmentation import build_augmentation_pipeline
from data.preprocessing import (
    compute_building_density,
    read_mask_image,
    safe_load_pair,
    validate_binary_mask,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Minimal loader configuration."""

    rgb_dir: str
    mask_dir: str
    image_size: int
    batch_size: int
    num_workers: int = 4
    prefetch_buffer: int = 2
    seed: int = 42


class BuildingSegmentationDataset:
    """Load and pair RGB/mask tiles, and construct tf.data datasets."""

    def __init__(self, config: DatasetConfig | Dict[str, object], skipped_log_path: str) -> None:
        if isinstance(config, dict):
            self.config = DatasetConfig(
                rgb_dir=str(config["rgb_dir"]),
                mask_dir=str(config["mask_dir"]),
                image_size=int(config["image_size"]),
                batch_size=int(config["batch_size"]),
                num_workers=int(config.get("num_workers", 4)),
                prefetch_buffer=int(config.get("prefetch_buffer", 2)),
                seed=int(config.get("seed", 42)),
            )
        else:
            self.config = config

        self.rgb_dir = Path(self.config.rgb_dir)
        self.mask_dir = Path(self.config.mask_dir)
        self.skipped_log_path = Path(skipped_log_path)
        self.skipped_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.pairs: List[Tuple[Path, Path]] = []
        self.densities: List[float] = []
        self.background_flags: List[bool] = []

    def _warn_skip(self, message: str) -> None:
        LOGGER.warning(message)
        with self.skipped_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def scan_and_pair(self) -> List[Tuple[Path, Path]]:
        """Scan directories and pair RGB images to masks by basename."""
        rgb_files = sorted(self.rgb_dir.glob("*.png"))
        mask_files = sorted(self.mask_dir.glob("*.tif"))

        mask_lookup = {path.stem: path for path in mask_files}
        pairs: List[Tuple[Path, Path]] = []
        unmatched_rgb: List[Path] = []

        for rgb_path in rgb_files:
            mask_path = mask_lookup.get(rgb_path.stem)
            if mask_path is not None:
                pairs.append((rgb_path, mask_path))
            else:
                unmatched_rgb.append(rgb_path)

        matched_mask_stems = {mask.stem for _, mask in pairs}
        orphan_masks = [mask for mask in mask_files if mask.stem not in matched_mask_stems]

        if unmatched_rgb or orphan_masks:
            self._warn_skip(
                "Pairing mismatch detected: "
                f"{len(unmatched_rgb)} unmatched RGB, {len(orphan_masks)} orphan masks."
            )

        if not pairs:
            # Fallback: sorted-order pairing for partially corrupted naming.
            self._warn_skip("No basename matches; applying sorted-order fallback pairing.")
            for rgb_path, mask_path in zip(rgb_files, mask_files):
                pairs.append((rgb_path, mask_path))

        self.pairs = pairs
        return pairs

    def validate_pairs(self) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
        """Validate pair readability and report orphaned/corrupted files."""
        if not self.pairs:
            self.scan_and_pair()

        valid_pairs: List[Tuple[Path, Path]] = []
        orphaned: List[Path] = []
        densities: List[float] = []
        background_flags: List[bool] = []

        for rgb_path, mask_path in self.pairs:
            try:
                _, mask = safe_load_pair(rgb_path, mask_path, self.config.image_size)
            except ValueError as exc:
                orphaned.extend([rgb_path, mask_path])
                self._warn_skip(str(exc))
                continue

            density = compute_building_density(mask)
            if not validate_binary_mask(mask):
                self._warn_skip(f"Non-binary mask values detected after thresholding: {mask_path}")

            valid_pairs.append((rgb_path, mask_path))
            densities.append(density)
            background_flags.append(bool(np.isclose(density, 0.0)))

        self.pairs = valid_pairs
        self.densities = densities
        self.background_flags = background_flags
        return valid_pairs, orphaned

    def compute_density(self, mask_path: Path) -> float:
        """Compute building density for a single mask path."""
        mask = read_mask_image(mask_path, self.config.image_size)
        return compute_building_density(mask)

    def load_sample(self, rgb_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess an image-mask sample."""
        return safe_load_pair(rgb_path, mask_path, self.config.image_size)

    def get_tf_dataset(
        self,
        indices: Sequence[int],
        augment: bool = False,
        augmentation_config: Dict[str, object] | None = None,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """Build tf.data dataset yielding normalized float32 image/mask tensors."""
        if not self.pairs:
            self.validate_pairs()

        selected_pairs = [self.pairs[idx] for idx in indices]
        image_paths = [str(pair[0]) for pair in selected_pairs]
        mask_paths = [str(pair[1]) for pair in selected_pairs]

        def _read_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            def _loader(image_path_bytes: bytes, mask_path_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
                image, mask = self.load_sample(Path(image_path_bytes.decode()), Path(mask_path_bytes.decode()))
                return image.astype(np.float32), mask.astype(np.float32)

            image, mask = tf.numpy_function(
                _loader,
                [image_path, mask_path],
                [tf.float32, tf.float32],
            )
            image.set_shape((self.config.image_size, self.config.image_size, 3))
            mask.set_shape((self.config.image_size, self.config.image_size, 1))
            return image, mask

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=max(len(image_paths), 1),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(_read_pair, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            augmentation_config = augmentation_config or {}
            augmentation_config.setdefault("seed", self.config.seed)
            aug_fn = build_augmentation_pipeline(augmentation_config)
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.config.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        return dataset

    def get_density_labels(self) -> np.ndarray:
        """Return cached density values as a NumPy array."""
        if not self.densities:
            self.validate_pairs()
        return np.asarray(self.densities, dtype=np.float32)
