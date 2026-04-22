"""Data package for building segmentation experiments."""

try:
    from .dataset import BuildingDataset
except ModuleNotFoundError:  # pragma: no cover - allows env checks before TF install
    BuildingDataset = None
try:
    from .loader import BuildingSegmentationDataset, DatasetConfig
except ModuleNotFoundError:  # pragma: no cover - allows env checks before TF install
    BuildingSegmentationDataset = None
    DatasetConfig = None
from .splitter import StratifiedSplitter
from .integrity import compute_dataset_hash

__all__ = [
    "BuildingDataset",
    "BuildingSegmentationDataset",
    "DatasetConfig",
    "StratifiedSplitter",
    "compute_dataset_hash",
]
