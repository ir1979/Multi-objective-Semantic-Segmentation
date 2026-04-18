"""Model factory with segmentation_models library support.

This module provides a unified interface for building various semantic
segmentation architectures using the segmentation_models library with
support for multiple encoder backbones.
"""

import os
import logging
from typing import Callable, Dict, Any, Optional, Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.registry import Registry
from .factory import get_model as get_research_model

# Try to import segmentation_models
try:
    import segmentation_models as sm
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False
    logging.warning("segmentation_models not installed. Using custom implementations.")

logger = logging.getLogger(__name__)
ModelBuilder = Callable[..., keras.Model]
MODEL_REGISTRY: Registry[ModelBuilder] = Registry("model")

# Encoder mapping for segmentation_models
ENCODER_MAP = {
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    'efficientnetb0': 'efficientnetb0',
    'efficientnetb1': 'efficientnetb1',
    'efficientnetb2': 'efficientnetb2',
    'efficientnetb3': 'efficientnetb3',
    'efficientnetb4': 'efficientnetb4',
    'efficientnetb5': 'efficientnetb5',
    'efficientnetb6': 'efficientnetb6',
    'efficientnetb7': 'efficientnetb7',
    'vgg16': 'vgg16',
    'vgg19': 'vgg19',
    'densenet121': 'densenet121',
    'densenet169': 'densenet169',
    'densenet201': 'densenet201',
    'inceptionv3': 'inceptionv3',
    'inceptionresnetv2': 'inceptionresnetv2',
    'mobilenet': 'mobilenet',
    'mobilenetv2': 'mobilenetv2',
    'seresnet18': 'seresnet18',
    'seresnet34': 'seresnet34',
    'seresnet50': 'seresnet50',
    'seresnet101': 'seresnet101',
    'seresnet152': 'seresnet152',
    'seresnext50': 'seresnext50',
    'seresnext101': 'seresnext101',
    'senet154': 'senet154',
}


def get_available_architectures() -> List[str]:
    """Get list of available architectures.
    
    Returns
    -------
    List[str]
        List of available architecture names
    """
    return MODEL_REGISTRY.list_names()


def get_available_encoders() -> List[str]:
    """Get list of available encoders.
    
    Returns
    -------
    List[str]
        List of available encoder names
    """
    return list(ENCODER_MAP.keys())


def build_model(
    architecture: str,
    encoder: str = 'resnet34',
    encoder_weights: Optional[str] = 'imagenet',
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    activation: str = 'sigmoid',
    deep_supervision: bool = False,
    **kwargs
) -> keras.Model:
    """Build a segmentation model with specified architecture and encoder.
    
    This factory function creates models using segmentation_models library
    when available, falling back to custom implementations for specific
    architectures.
    
    Parameters
    ----------
    architecture : str
        Model architecture name (e.g., 'Unet', 'UnetPlusPlus', 'FPN')
    encoder : str, optional
        Encoder backbone name (e.g., 'resnet34', 'efficientnetb0')
    encoder_weights : str or None, optional
        Pretrained weights for encoder ('imagenet' or None)
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
    activation : str
        Activation function for output layer
    deep_supervision : bool
        Whether to use deep supervision (for supported architectures)
    **kwargs
        Additional arguments passed to model builder
    
    Returns
    -------
    keras.Model
        Compiled segmentation model
    
    Raises
    ------
    ValueError
        If architecture or encoder is not supported
    
    Examples
    --------
    >>> model = build_model('Unet', 'resnet34', input_shape=(256, 256, 3))
    >>> model = build_model('FPN', 'efficientnetb0', num_classes=1)
    """
    # Map encoder name
    if encoder.lower() in ENCODER_MAP:
        enc_name = ENCODER_MAP[encoder.lower()]
    else:
        enc_name = encoder

    # Route canonical research architectures to the publication-grade implementations.
    architecture_key = architecture.lower().replace("_", "").replace("-", "")
    if architecture_key in {"unet", "unetplusplus", "unet++", "unetpp"}:
        resolved_arch = "unetpp" if architecture_key != "unet" else "unet"
        model = get_research_model(
            {
                "model": {
                    "architecture": resolved_arch,
                    "encoder_filters": kwargs.get("encoder_filters", [64, 128, 256, 512, 1024]),
                    "dropout_rate": kwargs.get("dropout_rate", 0.3),
                    "batch_norm": kwargs.get("batch_norm", True),
                    "activation": kwargs.get("activation", "relu"),
                    "output_activation": activation,
                    "deep_supervision": deep_supervision,
                }
            }
        )
        model.build((None,) + tuple(input_shape))
        return model
    
    entry = MODEL_REGISTRY.get(architecture)
    logger.info(f"Building model: {entry.name} with encoder {enc_name}")
    return entry.value(
        architecture=entry.name,
        encoder=enc_name,
        encoder_weights=encoder_weights,
        input_shape=input_shape,
        num_classes=num_classes,
        activation=activation,
        deep_supervision=deep_supervision,
        **kwargs,
    )


def register_model_builder(
    name: str,
    builder: ModelBuilder,
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
):
    """Register a new model builder for experiment configs."""
    return MODEL_REGISTRY.register(
        name=name,
        value=builder,
        aliases=aliases,
        metadata=metadata,
        overwrite=overwrite,
    )


def get_model_metadata(name: str) -> Dict[str, Any]:
    """Get model metadata for a registered architecture."""
    return dict(MODEL_REGISTRY.get(name).metadata)


def is_model_configuration_supported(
    architecture: str,
    input_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check whether a registered architecture is supported in this environment."""
    try:
        entry = MODEL_REGISTRY.get(architecture)
    except KeyError as exc:
        return False, str(exc)

    metadata = entry.metadata
    if metadata.get("requires_segmentation_models") and not SM_AVAILABLE:
        return False, f"{entry.name} requires the segmentation_models package"

    fixed_tile_size = metadata.get("fixed_tile_size")
    if fixed_tile_size is not None and input_shape is not None:
        if input_shape[0] != fixed_tile_size or input_shape[1] != fixed_tile_size:
            return False, f"{entry.name} currently supports only {fixed_tile_size}x{fixed_tile_size} tiles"

    return True, None


def _build_sm_model(
    architecture: str,
    encoder: str,
    encoder_weights: Optional[str],
    input_shape: Tuple[int, int, int],
    num_classes: int,
    activation: str
) -> keras.Model:
    """Build model using segmentation_models library.
    
    Parameters
    ----------
    architecture : str
        Architecture name for segmentation_models
    encoder : str
        Encoder name
    encoder_weights : str or None
        Pretrained weights
    input_shape : tuple
        Input shape
    num_classes : int
        Number of classes
    activation : str
        Output activation
    
    Returns
    -------
    keras.Model
        Built model
    """
    # Map architecture names to segmentation_models functions
    arch_map = {
        'Unet': sm.Unet,
        'UnetPlusPlus': sm.UnetPlusPlus,
        'FPN': sm.FPN,
        'Linknet': sm.Linknet,
        'PSPNet': sm.PSPNet,
        'DeepLabV3': sm.DeepLabV3,
        'DeepLabV3Plus': sm.DeepLabV3Plus,
        'PAN': sm.PAN,
    }
    
    if architecture not in arch_map:
        raise ValueError(f"Architecture {architecture} not available in segmentation_models")
    
    model_fn = arch_map[architecture]
    
    model = model_fn(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        input_shape=input_shape,
        classes=num_classes,
        activation=activation,
    )
    
    return model


def _register_builtin_models():
    register_model_builder(
        "Unet",
        lambda **kwargs: __import__("models.Unet", fromlist=["build_unet"]).build_unet(
            kwargs["input_shape"], kwargs["num_classes"], kwargs.get("deep_supervision", False)
        ),
        aliases=["unet_basic"],
        metadata={"fixed_tile_size": 256},
        overwrite=True,
    )
    register_model_builder(
        "UnetPlusPlus",
        lambda **kwargs: __import__("models.unet_pp", fromlist=["build_unet_pp"]).build_unet_pp(
            kwargs["input_shape"], kwargs["num_classes"], deep_supervision=kwargs.get("deep_supervision", False)
        ),
        aliases=["unet++", "unetplusplus", "unet_pp"],
        overwrite=True,
    )
    register_model_builder(
        "NestedUnet",
        lambda **kwargs: __import__("models.NestedUnet", fromlist=["NestedUNet"]).NestedUNet(),
        aliases=["nestedunet"],
        overwrite=True,
    )
    register_model_builder(
        "AttUNet",
        lambda **kwargs: __import__("models.AttUNet", fromlist=["build_att_unet"]).build_att_unet(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["attunet"],
        overwrite=True,
    )
    register_model_builder(
        "R2AttUNet",
        lambda **kwargs: __import__("models.R2AttUNet", fromlist=["build_r2att_unet"]).build_r2att_unet(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["r2attunet"],
        overwrite=True,
    )
    register_model_builder(
        "ResUnet",
        lambda **kwargs: __import__("models.ResUnet", fromlist=["build_res_unet"]).build_res_unet(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["resunet"],
        overwrite=True,
    )
    register_model_builder(
        "ResUnetPlusPlus",
        lambda **kwargs: __import__("models.ResUnetPlusPlus", fromlist=["build_res_unet_pp"]).build_res_unet_pp(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["resunet++", "resunetplusplus"],
        overwrite=True,
    )
    register_model_builder(
        "SEUnet",
        lambda **kwargs: __import__("models.SEUnet", fromlist=["build_se_unet"]).build_se_unet(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["seunet"],
        overwrite=True,
    )
    register_model_builder(
        "scSEUnet",
        lambda **kwargs: __import__("models.scSEUnet", fromlist=["build_scse_unet"]).build_scse_unet(
            kwargs["input_shape"], kwargs["num_classes"]
        ),
        aliases=["scseunet"],
        overwrite=True,
    )
    register_model_builder(
        "Unet3p",
        lambda **kwargs: __import__("models.Unet3p", fromlist=["build_unet_3p"]).build_unet_3p(
            kwargs["input_shape"], kwargs["num_classes"], deep_supervision=kwargs.get("deep_supervision", False)
        ),
        aliases=["unet3p", "unet3+"],
        overwrite=True,
    )

    sm_architectures = {
        "FPN": ["fpn"],
        "Linknet": ["linknet"],
        "PSPNet": ["pspnet"],
        "DeepLabV3": ["deeplabv3"],
        "DeepLabV3Plus": ["deeplabv3plus"],
        "PAN": ["pan"],
    }
    for architecture_name, aliases in sm_architectures.items():
        register_model_builder(
            architecture_name,
            lambda architecture=architecture_name, **kwargs: _build_sm_model(
                architecture=architecture,
                encoder=kwargs["encoder"],
                encoder_weights=kwargs["encoder_weights"],
                input_shape=kwargs["input_shape"],
                num_classes=kwargs["num_classes"],
                activation=kwargs["activation"],
            ),
            aliases=aliases,
            metadata={"requires_segmentation_models": True},
            overwrite=True,
        )


_register_builtin_models()


def get_model_info(model: keras.Model) -> Dict[str, Any]:
    """Get information about a model.
    
    Parameters
    ----------
    model : keras.Model
        Keras model to analyze
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing model information
    """
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
    }
    return info


def estimate_flops(model: keras.Model, input_shape: Optional[Tuple[int, ...]] = None) -> int:
    """Estimate FLOPs for a model.
    
    Parameters
    ----------
    model : keras.Model
        Model to analyze
    input_shape : tuple or None
        Input shape (if None, uses model.input_shape)
    
    Returns
    -------
    int
        Estimated FLOPs
    """
    if input_shape is None:
        input_shape = model.input_shape
    
    # Handle None in batch dimension
    if input_shape[0] is None:
        input_shape = (1,) + input_shape[1:]
    
    total_flops = 0
    
    def _shape_tuple(shape_obj):
        if shape_obj is None:
            return None
        if hasattr(shape_obj, "as_list"):
            return tuple(shape_obj.as_list())
        try:
            return tuple(shape_obj)
        except TypeError:
            return None

    for layer in model.layers:
        layer_flops = 0
        layer_input_shape = getattr(layer, "input_shape", None)
        if layer_input_shape is None and hasattr(layer, "input"):
            layer_input_shape = _shape_tuple(getattr(layer.input, "shape", None))
        layer_output_shape = getattr(layer, "output_shape", None)
        if layer_output_shape is None and hasattr(layer, "output"):
            layer_output_shape = _shape_tuple(getattr(layer.output, "shape", None))
        
        # Convolutional layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            if hasattr(layer, 'kernel_size') and hasattr(layer, 'filters'):
                # Conv2D: 2 * kernel_h * kernel_w * input_channels * output_h * output_w * output_channels
                kernel_h, kernel_w = layer.kernel_size
                if layer_input_shape is None or layer_output_shape is None:
                    continue
                input_channels = layer_input_shape[-1]
                output_channels = layer.filters
                
                if layer_output_shape is not None:
                    output_h = layer_output_shape[1]
                    output_w = layer_output_shape[2]
                    layer_flops = 2 * kernel_h * kernel_w * input_channels * output_h * output_w * output_channels
        
        # Dense layers
        elif isinstance(layer, keras.layers.Dense):
            if hasattr(layer, 'units') and layer_input_shape is not None:
                input_units = layer_input_shape[-1]
                output_units = layer.units
                layer_flops = 2 * input_units * output_units
        
        # Batch normalization
        elif isinstance(layer, keras.layers.BatchNormalization):
            if layer_output_shape is not None:
                # Approximately 5 operations per element
                layer_flops = 5 * np.prod(layer_output_shape[1:])
        
        total_flops += layer_flops
    
    return int(total_flops)


def measure_inference_time(
    model: keras.Model,
    input_shape: Tuple[int, ...] = (1, 256, 256, 3),
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """Measure inference time for a model.
    
    Parameters
    ----------
    model : keras.Model
        Model to benchmark
    input_shape : tuple
        Shape of input tensor
    num_runs : int
        Number of inference runs for timing
    warmup_runs : int
        Number of warmup runs before timing
    
    Returns
    -------
    Dict[str, float]
        Dictionary with timing statistics
    """
    import time
    
    # Create dummy input
    dummy_input = tf.random.normal(input_shape)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)
    
    # Timing
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(dummy_input, training=False)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
    }
