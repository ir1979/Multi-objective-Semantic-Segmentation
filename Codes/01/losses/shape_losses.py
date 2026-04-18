import tensorflow as tf


def shape_convexity(y_true, y_pred):
    """Regularize shape predictions with a smoothness penalty."""
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.total_variation(y_pred))
