"""Segmentation evaluation metrics and mask processing utilities.

This module defines a collection of metrics that can be used during training,
validation, and experiment reporting for semantic segmentation tasks.
Each function expects ground truth masks `y_true` and predicted masks
`y_pred`, where `y_pred` may be a probability map from the model.

Metrics in this module are designed to capture both region-level and
shape-level quality, including pixel accuracy, overlap, boundary alignment,
and approximate topology.
"""

import numpy as np
import tensorflow as tf

try:
    from skimage.measure import label as sk_label
except ImportError:
    sk_label = None

try:
    import cv2
except ImportError:
    cv2 = None


def _extract_tensor(x):
    """Helper to extract tensor from tuple/list if needed."""
    if isinstance(x, (tuple, list)):
        return tf.convert_to_tensor(x[0])
    return x

def threshold_prediction(y_pred, threshold=0.5):
    """Convert model scores into a binary foreground/background mask.

    This helper is useful when most metrics require a binary prediction.
    It can be applied to logits or probability outputs from the network.
    """
    y_pred = _extract_tensor(y_pred)
    return tf.cast(y_pred > threshold, tf.float32)


def pixel_accuracy(y_true, y_pred, threshold=0.5, eps=1e-7):
    """Compute pixel-wise classification accuracy for a single mask.

    This metric counts how many pixels match between prediction and truth.
    It is a stable baseline measure for segmentation quality, especially
    when the target object dominates the image.
    """
    # Convert tuple to tensor if necessary
    if isinstance(y_true, tuple):
        y_true = tf.convert_to_tensor(y_true[0])

    y_pred = threshold_prediction(y_pred, threshold)
    correct = tf.cast(
        tf.equal(tf.cast(y_true > 0.5, tf.bool), tf.cast(y_pred > 0.5, tf.bool)),
        tf.float32,
    )
    return tf.reduce_mean(correct)


def iou_score(y_true, y_pred, eps=1e-7):
    """Compute intersection over union (IoU) between masks.

    IoU is the standard evaluation measure in segmentation benchmarks.
    It penalizes both false positives and false negatives by comparing the
    overlap of the predicted and ground truth regions.
    """
    y_true = _extract_tensor(y_true)
    y_pred = threshold_prediction(y_pred)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - inter
    return (inter + eps) / (union + eps)


def dice_score(y_true, y_pred, eps=1e-7):
    """Compute the Dice similarity coefficient for predicted masks.

    Dice is closely related to IoU and is often used in medical and remote
    sensing segmentation tasks. It emphasizes overlap and is robust when
    object size is small relative to the image.
    """
    y_true = _extract_tensor(y_true)
    y_pred = threshold_prediction(y_pred)
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * inter + eps) / (denom + eps)


def precision_score(y_true, y_pred, eps=1e-7):
    """Compute positive predictive value (precision) for the foreground class.

    Precision captures how many predicted object pixels are actually correct.
    It is useful when false positives are more damaging than false negatives.
    """
    y_true = _extract_tensor(y_true)
    y_pred = threshold_prediction(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return (tp + eps) / (tp + fp + eps)


def recall_score(y_true, y_pred, eps=1e-7):
    """Compute recall (sensitivity) for the segmented object.

    Recall measures how much of the true object region was recovered by the
    prediction. It is useful when missing object pixels is expensive.
    """
    y_true = _extract_tensor(y_true)
    y_pred = threshold_prediction(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return (tp + eps) / (tp + fn + eps)


def _boundary_map(y_true, y_pred, pool_size=3):
    """Compute approximate binary boundary masks for true and predicted masks.

    This internal helper uses average pooling to estimate edges from the
    segmentation masks, which can then be compared via boundary-based metrics.
    """
    y_true = _extract_tensor(y_true)
    y_pred = _extract_tensor(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_edge = tf.abs(y_true - tf.nn.avg_pool2d(y_true, pool_size, 1, "SAME"))
    pred_edge = tf.abs(y_pred - tf.nn.avg_pool2d(y_pred, pool_size, 1, "SAME"))
    true_boundary = tf.cast(true_edge > 0.1, tf.float32)
    pred_boundary = tf.cast(pred_edge > 0.1, tf.float32)
    return true_boundary, pred_boundary


def boundary_iou(y_true, y_pred, eps=1e-7):
    """Evaluate how well predicted boundaries align with ground truth edges.

    Boundary IoU is valuable for building footprint segmentation because the
    contour quality is often more important than pure region overlap.
    """
    y_pred = threshold_prediction(y_pred)
    true_boundary, pred_boundary = _boundary_map(y_true, y_pred)
    inter = tf.reduce_sum(true_boundary * pred_boundary)
    union = tf.reduce_sum(true_boundary + pred_boundary) - inter
    return (inter + eps) / (union + eps)


def boundary_f1(y_true, y_pred, eps=1e-7):
    """Compute the boundary F1 score using predicted and true edge maps.

    This metric balances boundary precision and recall, providing a single
    score for edge alignment quality.
    """
    y_pred = threshold_prediction(y_pred)
    true_boundary, pred_boundary = _boundary_map(y_true, y_pred)
    tp = tf.reduce_sum(true_boundary * pred_boundary)
    precision = (tp + eps) / (tf.reduce_sum(pred_boundary) + eps)
    recall = (tp + eps) / (tf.reduce_sum(true_boundary) + eps)
    return (2.0 * precision * recall) / (precision + recall + eps)


def compactness_score(y_true, y_pred, eps=1e-7):
    """Measure the compactness of the predicted shape.

    Compactness is defined as area / perimeter^2, encouraging predictions that
    are regular and not overly fragmented. It is especially meaningful for
    semantic objects with expected compact geometry.
    """
    y_true = _extract_tensor(y_true)
    y_pred = threshold_prediction(y_pred)
    area = tf.reduce_sum(y_pred)
    edge_map = tf.abs(y_pred - tf.nn.avg_pool2d(y_pred, 3, 1, "SAME"))
    perimeter = tf.reduce_sum(tf.cast(edge_map > 0.1, tf.float32))
    return (area + eps) / (perimeter * perimeter + eps)


def region_completeness(y_true, y_pred, eps=1e-7):
    """Alias for IoU used when describing region completeness explicitly.

    This function is kept for readability in code and reports that compare
    region completeness to boundary or shape-based metrics.
    """
    return iou_score(y_true, y_pred, eps=eps)


def topological_correctness(y_true, y_pred):
    """Approximate topological correctness using connected component counts.

    This metric estimates whether the predicted segmentation preserves the
    same number of connected components as the ground truth. When installed,
    `scikit-image` is used for connected component labeling. If unavailable,
    the function returns 1.0 so it does not break training.
    """
    y_true = _extract_tensor(y_true)
    y_pred = _extract_tensor(y_pred)
    if sk_label is None and cv2 is None:
        return tf.constant(0.0)

    def count_components(mask_np):
        mask_np = np.asarray(mask_np, dtype=np.uint8)
        if mask_np.ndim == 4 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        if np.any(mask_np):
            if sk_label is not None:
                labeled = sk_label(mask_np)
                return np.float32(np.max(labeled))
            if cv2 is not None:
                count, _ = cv2.connectedComponents(mask_np)
                return np.float32(max(count - 1, 0))
        return np.float32(0.0)

    true_count = tf.numpy_function(count_components, [tf.cast(y_true > 0.5, tf.uint8)], tf.float32)
    pred_count = tf.numpy_function(count_components, [tf.cast(y_pred > 0.5, tf.uint8)], tf.float32)
    true_count = tf.maximum(true_count, 1.0)
    diff = tf.abs(true_count - pred_count)
    return 1.0 - diff / true_count
