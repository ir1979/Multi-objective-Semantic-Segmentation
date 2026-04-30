"""Model architecture tests."""

from __future__ import annotations

import unittest

import tensorflow as tf

from models.factory import get_model
from models.unet import UNet
from models.unetpp import UNetPlusPlus


class TestModels(unittest.TestCase):
    """Validate model factory and output behavior."""

    def test_unet_output_shape(self) -> None:
        model = UNet()
        x = tf.random.uniform((1, 256, 256, 3))
        y = model(x, training=False)
        self.assertEqual(tuple(y.shape), (1, 256, 256, 1))

    def test_unetpp_output_shape(self) -> None:
        model = UNetPlusPlus(deep_supervision=False)
        x = tf.random.uniform((1, 256, 256, 3))
        y = model(x, training=False)
        self.assertEqual(tuple(y.shape), (1, 256, 256, 1))

    def test_unetpp_deep_supervision_shapes(self) -> None:
        model = UNetPlusPlus(deep_supervision=True)
        x = tf.random.uniform((1, 256, 256, 3))
        outputs = model(x, training=False)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(tuple(out.shape), (1, 256, 256, 1))

    def test_model_factory_unet(self) -> None:
        model = get_model({"model": {"architecture": "unet"}})
        self.assertIsInstance(model, UNet)

    def test_model_factory_unetpp(self) -> None:
        model = get_model({"model": {"architecture": "unetpp"}})
        self.assertIsInstance(model, UNetPlusPlus)

    def test_model_factory_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _ = get_model({"model": {"architecture": "invalid"}})

    def test_output_range(self) -> None:
        model = UNet()
        y = model(tf.random.uniform((1, 256, 256, 3)), training=False)
        self.assertGreaterEqual(float(tf.reduce_min(y).numpy()), 0.0)
        self.assertLessEqual(float(tf.reduce_max(y).numpy()), 1.0)

    def test_gradient_flow(self) -> None:
        model = UNet()
        x = tf.random.uniform((1, 256, 256, 3))
        y_true = tf.random.uniform((1, 256, 256, 1))
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_pred - y_true))
        grads = tape.gradient(loss, model.trainable_variables)
        non_zero = [g for g in grads if g is not None and float(tf.reduce_sum(tf.abs(g)).numpy()) > 0.0]
        self.assertTrue(len(non_zero) > 0)
