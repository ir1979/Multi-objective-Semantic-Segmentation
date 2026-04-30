"""Loss functions package for segmentation training."""

from .boundary_losses import ApproxHausdorffLoss
from .deep_supervision_loss import DeepSupervisionLoss
from .loss_manager import LossManager, build_losses, build_single_loss
from .pixel_losses import BCELoss, DiceLoss, FocalLoss, IoULoss
from .shape_losses import ConvexityLoss, RegularityLoss

__all__ = [
    "ApproxHausdorffLoss",
    "DeepSupervisionLoss",
    "LossManager",
    "build_losses",
    "build_single_loss",
    "BCELoss",
    "BCEIoULoss",
    "DiceLoss",
    "FocalLoss",
    "IoULoss",
    "ConvexityLoss",
    "RegularityLoss",
]
