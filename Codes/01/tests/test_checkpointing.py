"""Checkpoint manager tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tensorflow as tf

from optimization.mgda import MGDASolver
from training.checkpoint_manager import CheckpointManager


class TestCheckpointing(unittest.TestCase):
    """Validate checkpoint save/load behavior."""

    def test_save_and_load_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(Path(tmpdir))
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(8,)),
                    tf.keras.layers.Dense(4, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            optimizer = tf.keras.optimizers.Adam(1e-3)
            dummy_x = tf.random.uniform((2, 8))
            dummy_y = tf.random.uniform((2, 1))
            with tf.GradientTape() as tape:
                pred = model(dummy_x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(dummy_y, pred))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            solver = MGDASolver()
            solver.alpha_history.append([0.5, 0.5])

            ckpt.save(
                model=model,
                optimizer=optimizer,
                epoch=3,
                metrics={"val_iou": 0.7, "val_boundary": 0.2},
                mgda_solver=solver,
            )
            self.assertTrue((Path(tmpdir) / "last_epoch.h5").exists())
            self.assertTrue((Path(tmpdir) / "training_state.pkl").exists())

            model_path, state = ckpt.load_latest()
            self.assertIsNotNone(model_path)
            self.assertIsNotNone(state)
            self.assertEqual(int(state["epoch"]), 3)

            restored = tf.keras.models.load_model(model_path, compile=False)
            self.assertEqual(len(restored.weights), len(model.weights))
            self.assertTrue(isinstance(state.get("optimizer_weights"), list))
            self.assertTrue(isinstance(state.get("mgda_alpha_history"), list))
            self.assertAlmostEqual(ckpt.get_best_metric(), 0.7, places=5)

            self.assertTrue(ckpt.has_checkpoint())

