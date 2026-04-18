"""Data package for building segmentation experiments."""

from .dataset import BuildingDataset
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
