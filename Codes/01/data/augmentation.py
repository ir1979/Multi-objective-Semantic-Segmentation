"""Data augmentation pipeline for building footprint segmentation.

This module provides image augmentation strategies using albumentations
for training semantic segmentation models on aerial/satellite imagery.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class AugmentationPipeline:
    """Configurable augmentation pipeline for building segmentation.
    
    This class creates augmentation pipelines based on configuration
    and applies them consistently to both images and masks.
    
    Parameters
    ----------
    augmentation_config : List[str] or str
        List of augmentation names or single augmentation strategy name
    input_shape : Tuple[int, int]
        Input image shape (height, width)
    
    Examples
    --------
    >>> pipeline = AugmentationPipeline(['flip', 'rotate', 'brightness'])
    >>> augmented_image, augmented_mask = pipeline(image, mask)
    """
    
    # Available augmentation strategies
    AUGMENTATION_STRATEGIES = {
        'flip': A.HorizontalFlip(p=0.5),
        'vertical_flip': A.VerticalFlip(p=0.5),
        'rotate': A.RandomRotate90(p=0.5),
        'rotation': A.Rotate(limit=30, p=0.5),
        'brightness': A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        'contrast': A.RandomContrast(limit=0.2, p=0.5),
        'elastic_transform': A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50, p=0.3
        ),
        'grid_distortion': A.GridDistortion(p=0.3),
        'optical_distortion': A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3),
        'blur': A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        'noise': A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        'hue_sat': A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
        ),
        'rgb_shift': A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        'gamma': A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        'scale': A.RandomScale(scale_limit=0.2, p=0.3),
        'shift': A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
        ),
        'cutout': A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, 
            min_holes=1, min_height=8, min_width=8, p=0.3
        ),
    }
    
    def __init__(
        self, 
        augmentation_config: List[str] or str = None,
        input_shape: Tuple[int, int] = (256, 256),
        seed: Optional[int] = None
    ):
        self.augmentation_config = augmentation_config or ['flip', 'rotate']
        self.input_shape = input_shape
        self.seed = seed
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> A.Compose:
        """Build the augmentation pipeline from configuration.
        
        Returns
        -------
        A.Compose
            Composed augmentation pipeline
        """
        transforms = []
        
        # Parse augmentation config
        aug_list = self._parse_config(self.augmentation_config)
        
        for aug_name in aug_list:
            aug_name = aug_name.lower().strip()
            
            if aug_name in self.AUGMENTATION_STRATEGIES:
                transforms.append(self.AUGMENTATION_STRATEGIES[aug_name])
            else:
                # Try to match partial names
                matched = self._match_augmentation(aug_name)
                if matched:
                    transforms.append(matched)
        
        # Always add normalization at the end
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]))
        
        # Build pipeline with mask support
        pipeline = A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            p=1.0
        )
        
        return pipeline
    
    def _parse_config(self, config) -> List[str]:
        """Parse augmentation configuration.
        
        Parameters
        ----------
        config : List[str] or str
            Augmentation configuration
        
        Returns
        -------
        List[str]
            List of augmentation names
        """
        if isinstance(config, str):
            # Try to parse comma-separated or space-separated
            return [x.strip() for x in config.replace(',', ' ').split()]
        elif isinstance(config, (list, tuple)):
            # Flatten nested lists
            result = []
            for item in config:
                if isinstance(item, (list, tuple)):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        else:
            return ['flip', 'rotate']
    
    def _match_augmentation(self, name: str) -> Optional[A.BasicTransform]:
        """Try to match an augmentation name.
        
        Parameters
        ----------
        name : str
            Augmentation name to match
        
        Returns
        -------
        Optional[A.BasicTransform]
            Matched augmentation or None
        """
        # Try exact match first
        if name in self.AUGMENTATION_STRATEGIES:
            return self.AUGMENTATION_STRATEGIES[name]
        
        # Try partial match
        for key, transform in self.AUGMENTATION_STRATEGIES.items():
            if name in key or key in name:
                return transform
        
        return None
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to image and mask.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, C) in range [0, 1]
        mask : np.ndarray, optional
            Input mask (H, W, 1) or (H, W)
        **kwargs
            Additional arguments for augmentation
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Augmented image and mask
        """
        # Set seed for reproducibility if specified
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Ensure image is in correct format
        if image.max() <= 1.0:
            # Convert from [0, 1] to [0, 255] for albumentations
            image = (image * 255).astype(np.uint8)
        
        # Prepare mask
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
            mask = (mask * 255).astype(np.uint8)
        
        # Apply augmentation
        if mask is not None:
            augmented = self.pipeline(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Convert back to [0, 1] range
            aug_image = aug_image.astype(np.float32)
            aug_mask = aug_mask.astype(np.float32) / 255.0
            
            # Ensure mask has channel dimension
            if aug_mask.ndim == 2:
                aug_mask = aug_mask[..., np.newaxis]
            
            return aug_image, aug_mask
        else:
            augmented = self.pipeline(image=image)
            aug_image = augmented['image']
            aug_image = aug_image.astype(np.float32)
            return aug_image, None
    
    def get_transform(self) -> Callable:
        """Get the transformation function for dataset integration.
        
        Returns
        -------
        Callable
            Function that takes (image, mask) and returns (augmented_image, augmented_mask)
        """
        def transform_fn(image, mask):
            return self(image, mask)
        
        return transform_fn


def create_augmentation_pipeline(
    augmentation_config: List[str] = None,
    input_shape: Tuple[int, int] = (256, 256),
    seed: Optional[int] = None
) -> AugmentationPipeline:
    """Factory function to create augmentation pipeline.
    
    Parameters
    ----------
    augmentation_config : List[str]
        List of augmentation names
    input_shape : Tuple[int, int]
        Input image shape
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    AugmentationPipeline
        Configured augmentation pipeline
    """
    return AugmentationPipeline(augmentation_config, input_shape, seed)


def get_available_augmentations() -> List[str]:
    """Get list of available augmentation strategies.
    
    Returns
    -------
    List[str]
        List of available augmentation names
    """
    return list(AugmentationPipeline.AUGMENTATION_STRATEGIES.keys())


def validate_augmentation_config(config: List[str]) -> Tuple[bool, List[str]]:
    """Validate augmentation configuration.
    
    Parameters
    ----------
    config : List[str]
        Augmentation configuration to validate
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, invalid_items)
    """
    available = get_available_augmentations()
    invalid = []
    
    for item in config:
        item_lower = item.lower().strip()
        if item_lower not in available:
            # Try partial match
            matched = False
            for aug in available:
                if item_lower in aug or aug in item_lower:
                    matched = True
                    break
            if not matched:
                invalid.append(item)
    
    return len(invalid) == 0, invalid


class RandomAugmentation(AugmentationPipeline):
    """Random augmentation pipeline with fixed seed per call.
    
    This is useful for creating different augmentations for each
    sample in a batch while maintaining reproducibility.
    """
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs):
        """Apply augmentation with random seed."""
        # Set random seed based on current time or provided seed
        if self.seed is not None:
            current_seed = self.seed + int(np.random.randint(0, 10000))
            np.random.seed(current_seed)
        
        return super().__call__(image, mask, **kwargs)


# Predefined augmentation strategies
PREDEFINED_PIPELINES = {
    'basic': ['flip', 'rotate'],
    'medium': ['flip', 'rotate', 'brightness', 'contrast'],
    'heavy': ['flip', 'rotate', 'brightness', 'contrast', 'elastic_transform', 
              'grid_distortion', 'blur', 'noise'],
    'light': ['flip'],
    'geometric': ['flip', 'rotate', 'elastic_transform', 'grid_distortion'],
    'photometric': ['brightness', 'contrast', 'hue_sat', 'rgb_shift', 'gamma'],
}


def get_predefined_pipeline(name: str, input_shape: Tuple[int, int] = (256, 256)) -> AugmentationPipeline:
    """Get a predefined augmentation pipeline.
    
    Parameters
    ----------
    name : str
        Name of predefined pipeline ('basic', 'medium', 'heavy', 'light', etc.)
    input_shape : Tuple[int, int]
        Input image shape
    
    Returns
    -------
    AugmentationPipeline
        Configured pipeline
    
    Raises
    ------
    ValueError
        If predefined pipeline not found
    """
    if name not in PREDEFINED_PIPELINES:
        raise ValueError(f"Predefined pipeline '{name}' not found. "
                        f"Available: {list(PREDEFINED_PIPELINES.keys())}")
    
    return AugmentationPipeline(PREDEFINED_PIPELINES[name], input_shape)


def build_augmentation_from_config(config: Dict[str, Any]) -> Optional[AugmentationPipeline]:
    """Build augmentation pipeline from configuration dictionary.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration containing augmentation settings
    
    Returns
    -------
    Optional[AugmentationPipeline]
        Augmentation pipeline or None if disabled
    
    Examples
    --------
    >>> config = {
    ...     'augmentations': ['flip', 'rotate', 'brightness'],
    ...     'tile_size': 256,
    ...     'seed': 42
    ... }
    >>> pipeline = build_augmentation_from_config(config)
    """
    augmentations = config.get('augmentations')
    
    if augmentations is None or augmentations == [] or augmentations == 'none':
        return None
    
    # Handle predefined pipeline
    if isinstance(augmentations, str) and augmentations in PREDEFINED_PIPELINES:
        return get_predefined_pipeline(
            augmentations,
            (config.get('tile_size', 256), config.get('tile_size', 256))
        )
    
    # Handle list of augmentations
    tile_size = config.get('tile_size', 256)
    seed = config.get('seed', None)
    
    return AugmentationPipeline(augmentations, (tile_size, tile_size), seed)
