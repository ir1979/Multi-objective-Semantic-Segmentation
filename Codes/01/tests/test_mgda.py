"""MGDA solver tests."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from optimization.mgda import MGDASolver


class TestMGDA(unittest.TestCase):
    """Validate MGDA simplex and convergence properties."""

    def setUp(self) -> None:
        self.solver = MGDASolver(max_iterations=100, tolerance=1e-8)

    def test_alphas_sum_to_one(self) -> None:
        gradients = [[tf.constant([1.0, 0.0])], [tf.constant([0.0, 1.0])]]
        alphas, _ = self.solver.solve(gradients)
        self.assertAlmostEqual(float(np.sum(alphas)), 1.0, places=6)

    def test_alphas_non_negative(self) -> None:
        gradients = [[tf.constant([1.0, 0.0])], [tf.constant([0.0, 1.0])]]
        alphas, _ = self.solver.solve(gradients)
        self.assertTrue(np.all(alphas >= -1e-8))

    def test_identical_gradients_uniform(self) -> None:
        gradients = [[tf.constant([1.0, 2.0])], [tf.constant([1.0, 2.0])]]
        alphas, _ = self.solver.solve(gradients)
        self.assertAlmostEqual(float(alphas[0]), float(alphas[1]), places=2)

    def test_single_objective(self) -> None:
        gradients = [[tf.constant([1.0, 2.0])]]
        alphas, combined = self.solver.solve(gradients)
        self.assertEqual(alphas.tolist(), [1.0])
        self.assertEqual(len(combined), 1)

    def test_zero_gradient_handling(self) -> None:
        gradients = [[tf.constant([0.0, 0.0])], [tf.constant([1.0, 0.0])]]
        alphas, _ = self.solver.solve(gradients)
        self.assertAlmostEqual(float(np.sum(alphas)), 1.0, places=6)

    def test_gradient_combination_shape(self) -> None:
        gradients = [
            [tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)],
            [tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)],
            [tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)],
        ]
        _, combined = self.solver.solve(gradients)
        self.assertEqual(tuple(combined[0].shape), (2, 2))

