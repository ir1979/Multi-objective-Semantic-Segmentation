"""Dataset utilities for building footprint segmentation.

This module provides dataset classes and utilities for loading and
processing aerial/satellite imagery with building masks.
"""

import os
import hashlib
import glob
import math
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from tensorflow.keras.utils import Sequence

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from .augmentation import AugmentationPipeline, build_augmentation_from_config
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False


def _import_cv2():
    """Import cv2 with error handling."""
    if not CV2_AVAILABLE:
        raise ImportError(
            "opencv-python is required for image loading. "
            "Install it with: pip install opencv-python"
        )
    import cv2
    return cv2


def load_rgb_image(path: str, normalize: bool = True) -> np.ndarray:
    """Load an RGB image from file.
    
    Parameters
    ----------
    path : str
        Path to the image file
    normalize : bool
        Whether to normalize to [0, 1] range
    
    Returns
    -------
    np.ndarray
        Loaded image as float32 array with shape (H, W, 3)
    
    Raises
    ------
    FileNotFoundError
        If image cannot be loaded
    """
    if CV2_AVAILABLE:
        cv2 = _import_cv2()
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"RGB image not found or could not be loaded: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif PIL_AVAILABLE:
        with Image.open(path) as image_file:
            image = np.array(image_file.convert("RGB"))
    else:
        raise ImportError(
            "Image loading requires either opencv-python or Pillow. "
            "Install one of them to load RGB images."
        )
    
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def load_mask_image(path: str, normalize: bool = True) -> np.ndarray:
    """Load a mask image from file.
    
    Parameters
    ----------
    path : str
        Path to the mask file
    normalize : bool
        Whether to normalize to [0, 1] range
    
    Returns
    -------
    np.ndarray
        Loaded mask as float32 array with shape (H, W, 1)
    
    Raises
    ------
    FileNotFoundError
        If mask cannot be loaded
    """
    if CV2_AVAILABLE:
        cv2 = _import_cv2()
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found or could not be loaded: {path}")
    elif PIL_AVAILABLE:
        with Image.open(path) as mask_file:
            mask = np.array(mask_file.convert("L"))
    else:
        raise ImportError(
            "Mask loading requires either opencv-python or Pillow. "
            "Install one of them to load masks."
        )
    
    if normalize:
        mask = mask.astype(np.float32) / 255.0
    
    mask = np.clip(mask, 0.0, 1.0)
    mask = np.expand_dims(mask, axis=-1)
    
    return mask


def pair_files_by_filename(
    rgb_files: List[str], 
    mask_files: List[str],
    rgb_pattern: str = "*",
    mask_pattern: str = "*"
) -> List[Tuple[str, str]]:
    """Pair RGB and mask files by matching filenames.
    
    This function matches files based on their base filenames (without extension),
    which is the expected format for the dataset where files are paired by filename.
    
    Parameters
    ----------
    rgb_files : List[str]
        List of RGB image file paths
    mask_files : List[str]
        List of mask file paths
    rgb_pattern : str
        Pattern to filter RGB files
    mask_pattern : str
        Pattern to filter mask files
    
    Returns
    -------
    List[Tuple[str, str]]
        List of paired (rgb_file, mask_file) tuples
    
    Raises
    ------
    ValueError
        If no valid pairs are found
    """
    # Create lookup by basename without extension
    mask_lookup = {}
    for mask_path in mask_files:
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        mask_lookup[basename] = mask_path
    
    pairs = []
    unmatched_rgb = []
    
    for rgb_path in rgb_files:
        basename = os.path.splitext(os.path.basename(rgb_path))[0]
        
        if basename in mask_lookup:
            pairs.append((rgb_path, mask_lookup[basename]))
        else:
            unmatched_rgb.append(rgb_path)
    
    if not pairs:
        raise ValueError(
            f"No valid RGB-mask pairs found. "
            f"RGB files: {len(rgb_files)}, Mask files: {len(mask_files)}. "
            f"Unmatched RGB: {unmatched_rgb[:5]}"
        )
    
    return pairs


def load_dataset_files(
    rgb_glob: str,
    mask_glob: str,
    rgb_pattern: str = "*",
    mask_pattern: str = "*"
) -> Tuple[List[str], List[str]]:
    """Load and pair dataset files using glob patterns.
    
    Parameters
    ----------
    rgb_glob : str
        Glob pattern for RGB files (e.g., "./Datasets/RGB/*.png")
    mask_glob : str
        Glob pattern for mask files (e.g., "./Datasets/Mask/*.tif")
    rgb_pattern : str
        Pattern to filter RGB files
    mask_pattern : str
        Pattern to filter mask files
    
    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (rgb_files, mask_files) lists
    
    Raises
    ------
    ValueError
        If no files found or no valid pairs
    """
    rgb_files = sorted(glob.glob(rgb_glob))
    mask_files = sorted(glob.glob(mask_glob))
    
    if not rgb_files:
        raise ValueError(f"No RGB files found matching pattern: {rgb_glob}")
    if not mask_files:
        raise ValueError(f"No mask files found matching pattern: {mask_glob}")
    
    print(f"Found {len(rgb_files)} RGB files and {len(mask_files)} mask files")
    
    # Pair files by filename
    pairs = pair_files_by_filename(rgb_files, mask_files, rgb_pattern, mask_pattern)
    
    # Unzip pairs
    rgb_paired = [p[0] for p in pairs]
    mask_paired = [p[1] for p in pairs]
    
    print(f"Successfully paired {len(pairs)} RGB-mask pairs")
    
    return rgb_paired, mask_paired


def resolve_dataset_path(path: str) -> str:
    """Resolve dataset paths while tolerating case-only directory mismatches."""
    if os.path.exists(path):
        return path

    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    if not parts:
        return path

    candidates = []
    if parts[0] == "Datasets":
        candidates.append(os.path.join("datasets", *parts[1:]))
    elif parts[0] == "datasets":
        candidates.append(os.path.join("Datasets", *parts[1:]))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return path


class BuildingDataset(Sequence):
    """Keras Sequence for satellite image building segmentation.
    
    This dataset class supports:
    - Paired RGB and mask loading
    - Data augmentation via albumentations
    - Configurable batching and shuffling
    - Reproducible random seeding
    
    Parameters
    ----------
    rgb_paths : List[str]
        List of RGB image file paths
    mask_paths : List[str]
        List of mask file paths
    batch_size : int
        Batch size for training
    shuffle : bool
        Whether to shuffle data each epoch
    augment_fn : Callable, optional
        Augmentation function to apply to images and masks
    deterministic : bool
        Whether to use deterministic shuffling
    seed : int, optional
        Random seed for reproducibility
    tile_size : int, optional
        Expected tile size (for validation)
    """
    
    def __init__(
        self,
        rgb_paths: List[str],
        mask_paths: List[str],
        batch_size: int = 4,
        shuffle: bool = True,
        augment_fn: Optional[Callable] = None,
        deterministic: bool = False,
        seed: Optional[int] = None,
        tile_size: Optional[int] = None,
    ):
        if len(rgb_paths) != len(mask_paths):
            raise ValueError(
                f"RGB and mask lists must have equal length. "
                f"Got {len(rgb_paths)} RGB and {len(mask_paths)} mask files."
            )
        
        self.rgb_paths = list(rgb_paths)
        self.mask_paths = list(mask_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_fn = augment_fn
        self.deterministic = deterministic
        self.seed = seed
        self.tile_size = tile_size
        
        self.indices = np.arange(len(self.rgb_paths), dtype=np.int32)
        self.on_epoch_end()
        
        # Validate first few images
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that dataset files can be loaded and have correct shapes."""
        if len(self.rgb_paths) == 0:
            return
        
        try:
            sample_img = load_rgb_image(self.rgb_paths[0], normalize=False)
            sample_mask = load_mask_image(self.mask_paths[0], normalize=False)
            
            if self.tile_size:
                if sample_img.shape[0] != self.tile_size or sample_img.shape[1] != self.tile_size:
                    print(f"Warning: Image shape {sample_img.shape[:2]} doesn't match expected tile_size {self.tile_size}")
            
            if sample_img.shape[:2] != sample_mask.shape[:2]:
                print(f"Warning: RGB shape {sample_img.shape[:2]} doesn't match mask shape {sample_mask.shape[:2]}")
            
            print(f"Dataset validation passed. Sample shapes: RGB={sample_img.shape}, Mask={sample_mask.shape}")
            
        except Exception as e:
            print(f"Warning: Dataset validation failed: {e}")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        if not self.rgb_paths:
            return 0
        return math.ceil(len(self.rgb_paths) / self.batch_size)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data.
        
        Parameters
        ----------
        idx : int
            Batch index
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (images_batch, masks_batch)
        """
        # Get indices for this batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Handle case where batch would exceed dataset
        if len(batch_indices) < self.batch_size:
            # Pad with random samples from the dataset
            extra_needed = self.batch_size - len(batch_indices)
            if self.deterministic and self.seed is not None:
                rng = np.random.RandomState(self.seed + idx)
                extra_indices = rng.choice(self.indices, extra_needed, replace=len(self.indices) < extra_needed)
            else:
                extra_indices = np.random.choice(self.indices, extra_needed, replace=len(self.indices) < extra_needed)
            batch_indices = np.concatenate([batch_indices, extra_indices])
        
        # Load batch
        x_batch = []
        y_batch = []
        
        for i in batch_indices:
            rgb_path = self.rgb_paths[i]
            mask_path = self.mask_paths[i]
            
            try:
                x = load_rgb_image(rgb_path)
                y = load_mask_image(mask_path)
                
                # Apply augmentation if provided
                if self.augment_fn is not None:
                    x, y = self.augment_fn(x, y)
                
                x_batch.append(x)
                y_batch.append(y)
                
            except Exception as e:
                print(f"Error loading {rgb_path} or {mask_path}: {e}")
                # Use a previously loaded sample as fallback
                if len(x_batch) > 0:
                    x_batch.append(x_batch[-1])
                    y_batch.append(y_batch[-1])
        if not x_batch:
            raise RuntimeError(f"Failed to load any samples for batch index {idx}")
        
        return np.stack(x_batch, axis=0), np.stack(y_batch, axis=0)
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        if self.shuffle:
            if self.deterministic and self.seed is not None:
                rng = np.random.RandomState(self.seed)
                rng.shuffle(self.indices)
            else:
                np.random.shuffle(self.indices)
    
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with dataset statistics
        """
        return {
            "samples": len(self.rgb_paths),
            "batch_size": self.batch_size,
            "batches_per_epoch": len(self),
            "steps": len(self),
            "shuffle": self.shuffle,
            "deterministic": self.deterministic,
            "augmentation": self.augment_fn is not None,
            "seed": self.seed,
            "tile_size": self.tile_size,
        }
    
    def get_sample(self, idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single sample for visualization or debugging.
        
        Parameters
        ----------
        idx : int
            Sample index
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Single image and mask
        """
        x = load_rgb_image(self.rgb_paths[idx])
        y = load_mask_image(self.mask_paths[idx])
        
        if self.augment_fn is not None:
            x, y = self.augment_fn(x, y)
        
        return x, y


def create_datasets_from_config(
    config: Dict[str, Any],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Dict[str, BuildingDataset]:
    """Create train/val/test datasets from configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - rgb_path: path to RGB images directory
        - mask_path: path to mask images directory
        - batch_size: batch size
        - tile_size: input tile size
        - augmentations: list of augmentation names
        - seed: random seed
        - train_val_test_split: list of [train, val, test] splits
    train_split : float
        Fraction for training (default 0.7)
    val_split : float
        Fraction for validation (default 0.15)
    test_split : float
        Fraction for testing (default 0.15)
    
    Returns
    -------
    Dict[str, BuildingDataset]
        Dictionary with 'train', 'val', 'test' datasets
    
    Raises
    ------
    ValueError
        If splits don't sum to 1.0 or no files found
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0):
        raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")
    
    # Get paths from config
    rgb_path = resolve_dataset_path(config.get('rgb_path', 'datasets/RGB'))
    mask_path = resolve_dataset_path(config.get('mask_path', 'datasets/Mask'))
    
    # Build glob patterns
    rgb_glob = os.path.join(rgb_path, "*")
    mask_glob = os.path.join(mask_path, "*")
    
    # Load and pair files
    rgb_files, mask_files = load_dataset_files(rgb_glob, mask_glob)
    
    split_config = config.get('train_val_test_split')
    if split_config is not None:
        if len(split_config) != 3:
            raise ValueError("train_val_test_split must contain exactly three values")
        train_split, val_split, test_split = [float(v) for v in split_config]
        total_split = train_split + val_split + test_split
        if not np.isclose(total_split, 1.0):
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")

    # Shuffle pairs before splitting to avoid lexical/spatial bias.
    paired_files = list(zip(rgb_files, mask_files))
    seed = config.get('seed', 42)
    rng = np.random.RandomState(seed)
    rng.shuffle(paired_files)
    rgb_files = [rgb for rgb, _ in paired_files]
    mask_files = [mask for _, mask in paired_files]

    # Calculate split indices
    n_total = len(rgb_files)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    # Create splits
    train_rgb = rgb_files[:n_train]
    train_mask = mask_files[:n_train]
    
    val_rgb = rgb_files[n_train:n_train + n_val]
    val_mask = mask_files[n_train:n_train + n_val]
    
    test_rgb = rgb_files[n_train + n_val:]
    test_mask = mask_files[n_train + n_val:]
    
    # Build augmentation pipeline for training
    augment_fn = None
    if AUGMENTATION_AVAILABLE:
        augment_fn = build_augmentation_from_config(config)
    
    # Get parameters
    batch_size = config.get('batch_size', 4)
    tile_size = config.get('tile_size', 256)
    seed = config.get('seed', 42)
    
    # Create datasets
    train_ds = BuildingDataset(
        train_rgb, train_mask,
        batch_size=batch_size,
        shuffle=True,
        augment_fn=augment_fn,
        deterministic=False,
        seed=seed,
        tile_size=tile_size
    )
    
    val_ds = BuildingDataset(
        val_rgb, val_mask,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,  # No augmentation for validation
        deterministic=True,
        seed=seed,
        tile_size=tile_size
    )
    
    test_ds = BuildingDataset(
        test_rgb, test_mask,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,  # No augmentation for test
        deterministic=True,
        seed=seed,
        tile_size=tile_size
    )
    
    print(f"Dataset splits: Train={len(train_rgb)}, Val={len(val_rgb)}, Test={len(test_rgb)}")
    
    return {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }


def split_dataset(
    rgb_files: List[str],
    mask_files: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Split dataset files into train/val/test sets.
    
    Parameters
    ----------
    rgb_files : List[str]
        List of RGB file paths
    mask_files : List[str]
        List of mask file paths
    train_split : float
        Fraction for training
    val_split : float
        Fraction for validation
    test_split : float
        Fraction for testing
    seed : int, optional
        Random seed for shuffling
    
    Returns
    -------
    Dict[str, Tuple[List[str], List[str]]]
        Dictionary with 'train', 'val', 'test' keys containing (rgb, mask) tuples
    """
    if len(rgb_files) != len(mask_files):
        raise ValueError("RGB and mask file lists must have same length")
    
    # Validate splits
    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    # Shuffle if seed provided
    if seed is not None:
        indices = np.arange(len(rgb_files))
        np.random.seed(seed)
        np.random.shuffle(indices)
        rgb_files = [rgb_files[i] for i in indices]
        mask_files = [mask_files[i] for i in indices]
    
    # Calculate split indices
    n_total = len(rgb_files)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    # Create splits
    train_rgb = rgb_files[:n_train]
    train_mask = mask_files[:n_train]
    
    val_rgb = rgb_files[n_train:n_train + n_val]
    val_mask = mask_files[n_train:n_train + n_val]
    
    test_rgb = rgb_files[n_train + n_val:]
    test_mask = mask_files[n_train + n_val:]
    
    return {
        'train': (train_rgb, train_mask),
        'val': (val_rgb, val_mask),
        'test': (test_rgb, test_mask)
    }


def get_dataset_info(dataset: BuildingDataset) -> Dict[str, Any]:
    """Get detailed information about a dataset.
    
    Parameters
    ----------
    dataset : BuildingDataset
        Dataset to analyze
    
    Returns
    -------
    Dict[str, Any]
        Dataset information
    """
    info = dataset.summary()
    
    # Load a sample to get shape information
    if len(dataset) > 0:
        try:
            x_sample, y_sample = dataset[0]
            info['image_shape'] = x_sample.shape[1:]
            info['mask_shape'] = y_sample.shape[1:]
            info['image_dtype'] = str(x_sample.dtype)
            info['mask_dtype'] = str(y_sample.dtype)
            info['pixel_range'] = (float(x_sample.min()), float(x_sample.max()))
        except Exception as e:
            info['sample_error'] = str(e)
    
    return info
