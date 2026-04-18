"""Models package for the segmentation framework."""

from .model_factory import (
    build_model,
    get_available_architectures,
    get_model_metadata,
    is_model_configuration_supported,
    register_model_builder,
)

__all__ = [
    "build_model",
    "get_available_architectures",
    "get_model_metadata",
    "is_model_configuration_supported",
    "register_model_builder",
]
