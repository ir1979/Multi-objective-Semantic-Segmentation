"""Factory module for creating segmentation models."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf

from models.unet import UNet
from models.unetpp import UNetPlusPlus


def get_model(config: Mapping[str, Any]) -> tf.keras.Model:
    """Instantiate a model from resolved configuration.

    Parameters
    ----------
    config:
        Resolved configuration dictionary.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model instance.
    """
    model_cfg = dict(config.get("model", {}))
    architecture = str(model_cfg.get("architecture", "unet")).lower()
    filters = list(model_cfg.get("encoder_filters", [64, 128, 256, 512, 1024]))
    kwargs = {
        "encoder_filters": filters,
        "dropout_rate": float(model_cfg.get("dropout_rate", 0.3)),
        "batch_norm": bool(model_cfg.get("batch_norm", True)),
        "activation": str(model_cfg.get("activation", "relu")),
        "output_activation": str(model_cfg.get("output_activation", "sigmoid")),
    }

    if architecture == "unet":
        return UNet(**kwargs)
    if architecture == "unetpp":
        kwargs["deep_supervision"] = bool(model_cfg.get("deep_supervision", False))
        return UNetPlusPlus(**kwargs)
    raise ValueError(f"Unknown architecture '{architecture}'. Expected one of ['unet', 'unetpp'].")
