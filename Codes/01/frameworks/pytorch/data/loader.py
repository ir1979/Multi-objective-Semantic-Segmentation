"""PyTorch dataset and dataloader construction for building segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from frameworks.pytorch.data.augmentation import AugmentationConfig, PairAugmenter
from frameworks.pytorch.data.preprocessing import compute_density, read_mask, read_rgb


@dataclass
class DataConfig:
    """Data-related configuration."""

    rgb_dir: str
    mask_dir: str
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 42


class BuildingSegmentationTorchDataset(Dataset):
    """Torch dataset that loads paired RGB/mask tiles by index list."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        indices: Sequence[int],
        image_size: int,
        augmenter: Optional[PairAugmenter] = None,
    ) -> None:
        self.pairs = [pairs[idx] for idx in indices]
        self.image_size = image_size
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rgb_path, mask_path = self.pairs[idx]
        image = read_rgb(rgb_path, self.image_size)
        mask = read_mask(mask_path, self.image_size)
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_t = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
        return image_t, mask_t


class PairScanner:
    """Scan dataset directories and prepare paired file metadata."""

    def __init__(self, rgb_dir: str, mask_dir: str) -> None:
        self.rgb_dir = Path(rgb_dir)
        self.mask_dir = Path(mask_dir)

    def scan(self) -> Tuple[List[Tuple[Path, Path]], List[str]]:
        """Pair RGB and mask files by stem and report warnings."""
        warnings: List[str] = []
        rgb_files = sorted(self.rgb_dir.glob("*.png"))
        mask_files = sorted(self.mask_dir.glob("*.tif"))
        mask_map = {m.stem: m for m in mask_files}
        pairs: List[Tuple[Path, Path]] = []
        for rgb in rgb_files:
            mask = mask_map.get(rgb.stem)
            if mask is None:
                warnings.append(f"Orphan RGB: {rgb}")
                continue
            pairs.append((rgb, mask))
        orphan_masks = [m for m in mask_files if m.stem not in {r.stem for r, _ in pairs}]
        warnings.extend([f"Orphan mask: {m}" for m in orphan_masks])
        if not pairs:
            warnings.append("No strict basename pairs found, applying sorted fallback pairing.")
            for rgb, mask in zip(rgb_files, mask_files):
                pairs.append((rgb, mask))
        return pairs, warnings

    @staticmethod
    def densities(pairs: Sequence[Tuple[Path, Path]], image_size: int) -> np.ndarray:
        """Compute building densities for stratified splitting."""
        values = []
        for _, mask_path in pairs:
            mask = read_mask(mask_path, image_size=image_size)
            values.append(compute_density(mask))
        return np.asarray(values, dtype=np.float32)


def build_dataloaders(
    config: DataConfig,
    pairs: Sequence[Tuple[Path, Path]],
    split_indices: Dict[str, List[int]],
    augmentation_cfg: Optional[Dict[str, object]] = None,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders."""
    augmentation_cfg = augmentation_cfg or {}
    augmenter = PairAugmenter(
        AugmentationConfig(
            horizontal_flip=bool(augmentation_cfg.get("horizontal_flip", True)),
            vertical_flip=bool(augmentation_cfg.get("vertical_flip", True)),
            random_rotation=bool(augmentation_cfg.get("random_rotation", True)),
            brightness_range=float(augmentation_cfg.get("brightness_range", 0.1)),
            contrast_range=float(augmentation_cfg.get("contrast_range", 0.1)),
            seed=int(augmentation_cfg.get("seed", config.seed)),
        )
    )
    train_ds = BuildingSegmentationTorchDataset(
        pairs,
        split_indices["train"],
        image_size=config.image_size,
        augmenter=augmenter if bool(augmentation_cfg.get("enabled", True)) else None,
    )
    val_ds = BuildingSegmentationTorchDataset(
        pairs,
        split_indices["val"],
        image_size=config.image_size,
        augmenter=None,
    )
    test_ds = BuildingSegmentationTorchDataset(
        pairs,
        split_indices["test"],
        image_size=config.image_size,
        augmenter=None,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=generator,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }
