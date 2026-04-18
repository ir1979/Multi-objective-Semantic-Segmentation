"""Loss function manager for building footprint segmentation.

This module provides a unified interface for configuring and building
loss functions from configuration dictionaries.
"""

from typing import List, Tuple, Callable, Dict, Any
import tensorflow as tf
from tensorflow import keras

from .pixel_losses import (
    bce, iou_loss, pixel_bce_iou, dice_loss, focal_loss, 
    combo_loss, tversky_loss, pixel_bce_dice,
    FocalLoss, DiceLoss, ComboLoss
)
from .boundary_losses import boundary_hausdorff
from .shape_losses import shape_convexity


# Registry of available loss functions
LOSS_REGISTRY = {
    # Pixel-level losses
    'binary_crossentropy': bce,
    'bce': bce,
    'dice_loss': dice_loss,
    'dice': dice_loss,
    'iou_loss': iou_loss,
    'iou': iou_loss,
    'focal_loss': focal_loss,
    'focal': focal_loss,
    'tversky_loss': tversky_loss,
    'tversky': tversky_loss,
    'combo_loss': combo_loss,
    'combo': combo_loss,
    'pixel_bce_iou': pixel_bce_iou,
    'bce_iou': pixel_bce_iou,
    'bce+iou': pixel_bce_iou,
    'pixel_bce_dice': pixel_bce_dice,
    'bce_dice': pixel_bce_dice,
    'bce+dice': pixel_bce_dice,
    
    # Boundary losses
    'hausdorff': boundary_hausdorff,
    'boundary_hausdorff': boundary_hausdorff,
    
    # Shape losses
    'convexity': shape_convexity,
    'shape_convexity': shape_convexity,
}


# Keras Loss classes for model.compile()
KERAS_LOSS_CLASSES = {
    'focal_loss': FocalLoss,
    'focal': FocalLoss,
    'dice_loss': DiceLoss,
    'dice': DiceLoss,
    'combo_loss': ComboLoss,
    'combo': ComboLoss,
}


def get_loss_function(loss_name: str) -> Callable:
    """Get a loss function by name.
    
    Parameters
    ----------
    loss_name : str
        Name of the loss function
    
    Returns
    -------
    Callable
        Loss function
    
    Raises
    ------
    ValueError
        If loss function not found
    """
    loss_key = loss_name.lower().replace('_', '').replace('-', '')
    
    # Try direct lookup
    if loss_name in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_name]
    
    # Try normalized lookup
    normalized_registry = {k.lower().replace('_', ''): v for k, v in LOSS_REGISTRY.items()}
    if loss_key in normalized_registry:
        return normalized_registry[loss_key]
    
    raise ValueError(f"Loss function '{loss_name}' not found. "
                    f"Available: {list(LOSS_REGISTRY.keys())}")


def get_keras_loss(loss_name: str, **kwargs) -> keras.losses.Loss:
    """Get a Keras Loss class instance.
    
    Parameters
    ----------
    loss_name : str
        Name of the loss
    **kwargs
        Additional arguments for loss initialization
    
    Returns
    -------
    keras.losses.Loss
        Keras loss instance
    """
    loss_key = loss_name.lower().replace('_', '').replace('-', '')
    
    normalized_registry = {k.lower().replace('_', ''): v for k, v in KERAS_LOSS_CLASSES.items()}
    
    if loss_key in normalized_registry:
        return normalized_registry[loss_key](**kwargs)
    
    # Fall back to function-based losses
    loss_fn = get_loss_function(loss_name)
    return loss_fn


def build_losses(config: Dict[str, Any]) -> Tuple[List[Callable], List[float], List[str]]:
    """Build a list of loss functions and their weights from configuration.
    
    This function supports multiple configuration formats:
    - Simple format: pixel_loss, boundary_loss, shape_loss with weights
    - List format: loss_functions and loss_weights
    - Single format: just specifying loss
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing loss specifications
    
    Returns
    -------
    Tuple[List[Callable], List[float], List[str]]
        Tuple of (loss functions, weights, loss names)
    
    Examples
    --------
    >>> config = {'pixel_loss': 'bce', 'pixel_weight': 1.0}
    >>> losses, weights, names = build_losses(config)
    
    >>> config = {'loss_functions': ['bce', 'dice'], 'loss_weights': [1.0, 1.0]}
    >>> losses, weights, names = build_losses(config)
    """
    losses = []
    weights = []
    names = []
    
    # Handle list format
    if 'loss_functions' in config:
        loss_fns = config['loss_functions']
        loss_weights = config.get('loss_weights', [1.0] * len(loss_fns))
        
        if len(loss_weights) != len(loss_fns):
            loss_weights = [1.0] * len(loss_fns)
        
        for loss_name, weight in zip(loss_fns, loss_weights):
            losses.append(get_loss_function(loss_name))
            weights.append(float(weight))
            names.append(loss_name)
        
        return losses, weights, names
    
    # Handle simple format with individual components
    # Pixel loss
    pixel_loss = config.get("pixel_loss")
    if pixel_loss:
        losses.append(get_loss_function(pixel_loss))
        weights.append(float(config.get("pixel_weight", 1.0)))
        names.append(pixel_loss)
    
    # Boundary loss
    boundary_loss = config.get("boundary_loss")
    if boundary_loss and boundary_loss != "none":
        losses.append(get_loss_function(boundary_loss))
        weights.append(float(config.get("boundary_weight", 1.0)))
        names.append(boundary_loss)
    
    # Shape loss
    shape_loss = config.get("shape_loss")
    if shape_loss and shape_loss != "none":
        losses.append(get_loss_function(shape_loss))
        weights.append(float(config.get("shape_weight", 1.0)))
        names.append(shape_loss)
    
    # Validation
    if not losses:
        # Default to BCE if nothing specified
        losses.append(bce)
        weights.append(1.0)
        names.append("bce")
    
    return losses, weights, names


def build_single_loss(config: Dict[str, Any]) -> Callable:
    """Build a single combined loss function from configuration.
    
    This is useful when you want a single loss function to pass to
    model.compile(), especially for multi-component losses.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    
    Returns
    -------
    Callable
        Combined loss function
    
    Examples
    --------
    >>> config = {'pixel_loss': 'bce', 'boundary_loss': 'hausdorff', 
    ...           'pixel_weight': 1.0, 'boundary_weight': 0.5}
    >>> loss_fn = build_single_loss(config)
    >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    losses, weights, names = build_losses(config)
    
    if len(losses) == 1:
        return losses[0]
    
    def combined_loss(y_true, y_pred):
        total_loss = 0.0
        total_weight = sum(weights)
        
        for loss_fn, weight in zip(losses, weights):
            loss_val = loss_fn(y_true, y_pred)
            # Handle scalar vs tensor
            if hasattr(loss_val, 'numpy'):
                total_loss += weight * loss_val
            else:
                total_loss += weight * tf.reduce_mean(loss_val)
        
        return total_loss / total_weight
    
    return combined_loss


def get_loss_info(loss_name: str) -> Dict[str, Any]:
    """Get information about a loss function.
    
    Parameters
    ----------
    loss_name : str
        Name of the loss function
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with loss information
    """
    info = {
        'name': loss_name,
        'available': loss_name in LOSS_REGISTRY or loss_name in KERAS_LOSS_CLASSES,
    }
    
    if info['available']:
        loss_fn = get_loss_function(loss_name)
        info['function'] = loss_fn
        info['docstring'] = loss_fn.__doc__
    
    return info


def list_available_losses() -> List[str]:
    """List all available loss functions.
    
    Returns
    -------
    List[str]
        List of available loss function names
    """
    return sorted(list(LOSS_REGISTRY.keys()))


def validate_loss_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a loss configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration to validate
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        losses, weights, names = build_losses(config)
        
        if not losses:
            return False, "No loss functions specified"
        
        if len(losses) != len(weights):
            return False, "Mismatch between losses and weights"
        
        if any(w < 0 for w in weights):
            return False, "Loss weights must be non-negative"
        
        return True, "Configuration valid"
    
    except Exception as e:
        return False, str(e)
