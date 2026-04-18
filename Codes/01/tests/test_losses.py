"""Loss function tests."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from losses.boundary_losses import ApproxHausdorffLoss
from losses.loss_manager import LossManager
from losses.pixel_losses import BCELoss, BCEIoULoss, DiceLoss, FocalLoss, IoULoss
from losses.shape_losses import ConvexityLoss, RegularityLoss


class TestLosses(unittest.TestCase):
    """Validate loss values and differentiability."""

    def setUp(self) -> None:
        self.y_true = tf.constant(np.ones((1, 64, 64, 1), dtype=np.float32))
        self.y_pred_good = tf.constant(np.ones((1, 64, 64, 1), dtype=np.float32))
        self.y_pred_bad = tf.constant(np.zeros((1, 64, 64, 1), dtype=np.float32))

    def test_bce_perfect_prediction(self) -> None:
        loss = BCELoss()(self.y_true, self.y_pred_good)
        self.assertLess(float(loss.numpy()), 1e-3)

    def test_iou_loss_perfect(self) -> None:
        loss = IoULoss()(self.y_true, self.y_pred_good)
        self.assertLess(float(loss.numpy()), 1e-3)

    def test_iou_loss_no_overlap(self) -> None:
        loss = IoULoss()(self.y_true, self.y_pred_bad)
        self.assertGreater(float(loss.numpy()), 0.9)

    def test_dice_loss_symmetry(self) -> None:
        dice = DiceLoss()
        a = tf.cast(tf.random.uniform((1, 64, 64, 1)) > 0.5, tf.float32)
        b = tf.cast(tf.random.uniform((1, 64, 64, 1)) > 0.5, tf.float32)
        self.assertAlmostEqual(float(dice(a, b).numpy()), float(dice(b, a).numpy()), places=5)

    def test_boundary_loss_perfect(self) -> None:
        loss = ApproxHausdorffLoss()(self.y_true, self.y_true)
        self.assertGreaterEqual(float(loss.numpy()), 0.0)

    def test_shape_losses(self) -> None:
        convex = ConvexityLoss()(self.y_true, self.y_true)
        reg = RegularityLoss()(self.y_true, self.y_true)
        self.assertTrue(np.isfinite(float(convex.numpy())))
        self.assertTrue(np.isfinite(float(reg.numpy())))

    def test_focal_loss_reduces_easy_examples(self) -> None:
        focal = FocalLoss()
        easy = focal(self.y_true, self.y_pred_good)
        hard = focal(self.y_true, self.y_pred_bad + 1e-3)
        self.assertLess(float(easy.numpy()), float(hard.numpy()))

    def test_all_losses_differentiable(self) -> None:
        losses = [BCELoss(), IoULoss(), DiceLoss(), BCEIoULoss(), FocalLoss(), ApproxHausdorffLoss(), ConvexityLoss(), RegularityLoss()]
        pred = tf.Variable(tf.random.uniform((1, 64, 64, 1), minval=0.1, maxval=0.9))
        for loss_fn in losses:
            with tf.GradientTape() as tape:
                value = loss_fn(self.y_true, pred)
            grad = tape.gradient(value, pred)
            self.assertIsNotNone(grad)

    def test_loss_manager_weighted(self) -> None:
        manager = LossManager(
            {
                "loss": {
                    "strategy": "weighted",
                    "pixel": {"type": "bce_iou", "weight": 1.0},
                    "boundary": {"enabled": True, "weight": 0.5},
                    "shape": {"enabled": True, "weight": 0.25},
                }
            }
        )
        losses_dict = manager.compute_losses(self.y_true, self.y_pred_good)
        total = manager.compute_weighted_total(losses_dict)
        self.assertTrue(np.isfinite(float(total.numpy())))

    def test_loss_manager_returns_all(self) -> None:
        manager = LossManager(
            {
                "loss": {
                    "strategy": "mgda",
                    "pixel": {"type": "bce", "weight": 1.0},
                    "boundary": {"enabled": True, "weight": 0.2},
                    "shape": {"enabled": True, "weight": 0.1},
                }
            }
        )
        losses_dict = manager.compute_losses(self.y_true, self.y_pred_good)
        self.assertEqual(set(losses_dict.keys()), {"pixel", "boundary", "shape"})

