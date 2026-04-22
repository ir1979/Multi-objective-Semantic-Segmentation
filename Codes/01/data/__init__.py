"""Data package for building segmentation experiments."""

try:
    from .dataset import BuildingDataset
except Exception:  # pragma: no cover - allows environment checks without TF installed
    BuildingDataset = None

from .loader import BuildingSegmentationDataset, DatasetConfig
from .splitter import StratifiedSplitter
from .integrity import compute_dataset_hash

__all__ = [
    "BuildingDataset",
    "BuildingSegmentationDataset",
    "DatasetConfig",
    "StratifiedSplitter",
    "compute_dataset_hash",
]
