"""Factory helpers for PyTorch segmentation models."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from frameworks.pytorch.models.unet import UNet
from frameworks.pytorch.models.unetpp import UNetPlusPlus


def get_model(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate a PyTorch model from configuration mapping."""
    model_cfg = dict(config.get("model", {}))
    architecture = str(model_cfg.get("architecture", "unet")).lower()
    kwargs = {
        "encoder_filters": list(model_cfg.get("encoder_filters", [64, 128, 256, 512, 1024])),
        "dropout_rate": float(model_cfg.get("dropout_rate", 0.3)),
        "batch_norm": bool(model_cfg.get("batch_norm", True)),
    }
    if architecture == "unet":
        return UNet(**kwargs)
    if architecture == "unetpp":
        kwargs["deep_supervision"] = bool(model_cfg.get("deep_supervision", False))
        return UNetPlusPlus(**kwargs)
    raise ValueError(f"Unknown PyTorch architecture: {architecture}")
