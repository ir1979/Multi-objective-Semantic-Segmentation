import tensorflow as tf


def boundary_hausdorff(y_true, y_pred):
    """Approximate a boundary-aware loss using local edge differences."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_edge = tf.abs(y_true - tf.nn.avg_pool2d(y_true, 3, 1, "SAME"))
    pred_edge = tf.abs(y_pred - tf.nn.avg_pool2d(y_pred, 3, 1, "SAME"))
    return tf.reduce_mean(tf.abs(true_edge - pred_edge))
