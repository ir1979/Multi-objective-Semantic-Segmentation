"""Pixel-level loss functions for semantic segmentation.

This module provides various pixel-level loss functions including
binary cross-entropy, Dice loss, IoU loss, Focal loss, and combined losses
for building footprint segmentation tasks.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def bce(y_true, y_pred, eps=1e-7):
    """Binary cross-entropy loss.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks (batch, height, width, channels)
    y_pred : tf.Tensor
        Predicted masks (batch, height, width, channels)
    eps : float
        Small value to avoid numerical instability
    
    Returns
    -------
    tf.Tensor
        Binary cross-entropy loss
    """
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    loss = -tf.reduce_mean(
        y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    )
    return loss


def dice_loss(y_true, y_pred, eps=1e-7, smooth=1.0):
    """Dice loss for segmentation.
    
    The Dice coefficient measures the overlap between two samples.
    This loss is particularly useful for imbalanced datasets where
    the target class (buildings) may occupy a small portion of the image.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    eps : float
        Small value to avoid division by zero
    smooth : float
        Smoothing factor for stability
    
    Returns
    -------
    tf.Tensor
        Dice loss (1 - Dice coefficient)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2.0 * intersection + smooth) / (union + smooth + eps)
    return 1.0 - dice


def iou_loss(y_true, y_pred, eps=1e-7, smooth=1.0):
    """IoU (Intersection over Union) loss.
    
    Also known as Jaccard loss. Maximizes the overlap between
    predicted and ground truth regions.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    eps : float
        Small value to avoid division by zero
    smooth : float
        Smoothing factor for stability
    
    Returns
    -------
    tf.Tensor
        IoU loss (1 - IoU coefficient)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou = (intersection + smooth) / (union + smooth + eps)
    return 1.0 - iou


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, eps=1e-7):
    """Focal loss for addressing class imbalance.
    
    Down-weights easy examples and focuses on hard examples.
    Particularly useful when buildings occupy a small fraction of the image.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks (after sigmoid)
    alpha : float
        Weighting factor for the rare class (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    eps : float
        Small value to avoid log(0)
    
    Returns
    -------
    tf.Tensor
        Focal loss value
    """
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    # Cross-entropy
    ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    # Focal weight
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    loss = tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * ce)
    return loss


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, eps=1e-7, smooth=1.0):
    """Tversky loss for handling class imbalance.
    
    Generalization of Dice loss with alpha and beta parameters controlling
    the penalty for false positives and false negatives.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    alpha : float
        Weight for false negatives (higher = more penalty on missing buildings)
    beta : float
        Weight for false positives (higher = more penalty on extra predictions)
    eps : float
        Small value to avoid division by zero
    smooth : float
        Smoothing factor for stability
    
    Returns
    -------
    tf.Tensor
        Tversky loss (1 - Tversky coefficient)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth + eps)
    return 1.0 - tversky


def pixel_bce_iou(y_true, y_pred, bce_weight=1.0, iou_weight=1.0, eps=1e-7):
    """Combined BCE and IoU loss.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    bce_weight : float
        Weight for BCE component
    iou_weight : float
        Weight for IoU component
    eps : float
        Small value for numerical stability
    
    Returns
    -------
    tf.Tensor
        Combined loss
    """
    bce_val = bce(y_true, y_pred, eps)
    iou_val = iou_loss(y_true, y_pred, eps)
    
    return (bce_weight * bce_val + iou_weight * iou_val) / (bce_weight + iou_weight)


def pixel_bce_dice(y_true, y_pred, bce_weight=1.0, dice_weight=1.0, eps=1e-7):
    """Combined BCE and Dice loss.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    bce_weight : float
        Weight for BCE component
    dice_weight : float
        Weight for Dice component
    eps : float
        Small value for numerical stability
    
    Returns
    -------
    tf.Tensor
        Combined loss
    """
    bce_val = bce(y_true, y_pred, eps)
    dice_val = dice_loss(y_true, y_pred, eps)
    
    return (bce_weight * bce_val + dice_weight * dice_val) / (bce_weight + dice_weight)


def combo_loss(y_true, y_pred, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, 
               alpha=0.25, gamma=2.0, eps=1e-7):
    """Combined BCE, Dice, and Focal loss.
    
    This is a comprehensive loss that combines pixel-wise classification (BCE),
    region overlap (Dice), and class imbalance handling (Focal).
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    bce_weight : float
        Weight for BCE component
    dice_weight : float
        Weight for Dice component
    focal_weight : float
        Weight for Focal component
    alpha : float
        Focal loss alpha parameter
    gamma : float
        Focal loss gamma parameter
    eps : float
        Small value for numerical stability
    
    Returns
    -------
    tf.Tensor
        Combined loss
    """
    bce_val = bce(y_true, y_pred, eps)
    dice_val = dice_loss(y_true, y_pred, eps)
    focal_val = focal_loss(y_true, y_pred, alpha, gamma, eps)
    
    total_weight = bce_weight + dice_weight + focal_weight
    return (bce_weight * bce_val + dice_weight * dice_val + focal_weight * focal_val) / total_weight


def lovasz_hinge(y_true, y_pred):
    """Lovász hinge loss for direct optimization of IoU.
    
    This loss is specifically designed to optimize the IoU metric directly.
    Best used when the primary evaluation metric is IoU.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks (flattened or per-image)
    y_pred : tf.Tensor
        Predicted logits (before sigmoid)
    
    Returns
    -------
    tf.Tensor
        Lovász hinge loss
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # Sort errors
    signs = 2.0 * y_true - 1.0
    errors = 1.0 - y_pred * signs
    
    errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
    gt_sorted = tf.gather(y_true, perm)
    
    # Compute grad
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / union
    jaccard = tf.concat([jaccard[0:1], jaccard[1:] - jaccard[:-1]], axis=0)
    
    loss = tf.tensordot(tf.nn.relu(errors_sorted), jaccard, 1)
    return loss


class FocalLoss(keras.losses.Loss):
    """Focal loss as a Keras Loss class for use in model.compile().
    
    Parameters
    ----------
    alpha : float
        Weighting factor for the rare class
    gamma : float
        Focusing parameter
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, self.alpha, self.gamma)
    
    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class DiceLoss(keras.losses.Loss):
    """Dice loss as a Keras Loss class.
    
    Parameters
    ----------
    smooth : float
        Smoothing factor for stability
    """
    
    def __init__(self, smooth=1.0, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        return dice_loss(y_true, y_pred, smooth=self.smooth)
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config


class ComboLoss(keras.losses.Loss):
    """Combo loss as a Keras Loss class.
    
    Parameters
    ----------
    bce_weight : float
        Weight for BCE component
    dice_weight : float
        Weight for Dice component
    focal_weight : float
        Weight for Focal component
    """
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, 
                 alpha=0.25, gamma=2.0, name="combo_loss"):
        super().__init__(name=name)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        return combo_loss(y_true, y_pred, self.bce_weight, self.dice_weight,
                         self.focal_weight, self.alpha, self.gamma)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "alpha": self.alpha,
            "gamma": self.gamma
        })
        return config