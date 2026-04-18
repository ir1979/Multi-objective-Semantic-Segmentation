"""MGDA solver and train-step helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


def _flatten_gradients(gradients: Sequence[tf.Tensor | None]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for grad in gradients:
        if grad is None:
            continue
        vectors.append(tf.reshape(grad, [-1]).numpy())
    if not vectors:
        return np.zeros((1,), dtype=np.float64)
    return np.concatenate(vectors, axis=0).astype(np.float64)


def _gradients_non_zero(gradient_list: Sequence[tf.Tensor | None], epsilon: float) -> bool:
    """Return True when any gradient tensor has non-trivial norm."""
    for grad in gradient_list:
        if grad is None:
            continue
        if float(tf.norm(grad).numpy()) > epsilon:
            return True
    return False


@dataclass
class MGDASolver:
    """Frank-Wolfe solver for MGDA simplex optimization."""

    max_iterations: int = 50
    tolerance: float = 1e-6
    normalize_gradients: bool = True
    epsilon: float = 1e-12
    alpha_history: List[np.ndarray] = field(default_factory=list)

    def _normalize(self, grads: np.ndarray) -> np.ndarray:
        if not self.normalize_gradients:
            return grads
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        norms = np.maximum(norms, self.epsilon)
        return grads / norms

    def solve(self, gradients: List[List[tf.Tensor | None]]) -> Tuple[np.ndarray, List[tf.Tensor]]:
        """Solve for minimum-norm convex combination of gradients."""
        if not gradients:
            raise ValueError("Expected at least one objective gradient list.")

        active_indices = [
            idx for idx, gradient_list in enumerate(gradients) if _gradients_non_zero(gradient_list, self.epsilon)
        ]
        if not active_indices:
            active_indices = [0]
        active_gradients = [gradients[idx] for idx in active_indices]

        vectors = np.stack([_flatten_gradients(grad_list) for grad_list in active_gradients], axis=0)
        vectors = self._normalize(vectors)
        num_objectives = vectors.shape[0]
        if num_objectives == 1:
            alpha = np.array([1.0], dtype=np.float64)
            combined = [grad if grad is not None else tf.constant(0.0) for grad in active_gradients[0]]
            full_alpha = np.zeros(len(gradients), dtype=np.float64)
            full_alpha[active_indices[0]] = 1.0
            self.alpha_history.append(full_alpha.copy())
            return full_alpha, combined

        alpha = np.ones(num_objectives, dtype=np.float64) / float(num_objectives)
        prev_direction = vectors.T @ alpha
        self.alpha_history.append(alpha.copy())

        for _ in range(self.max_iterations):
            current = vectors.T @ alpha
            inner = vectors @ current
            target = int(np.argmin(inner))
            direction = vectors[target] - current
            denom = float(np.dot(direction, direction))
            if denom <= self.epsilon:
                break
            gamma = float(np.clip(-np.dot(current, direction) / denom, 0.0, 1.0))
            vertex = np.zeros_like(alpha)
            vertex[target] = 1.0
            new_alpha = (1.0 - gamma) * alpha + gamma * vertex
            new_direction = vectors.T @ new_alpha
            if np.linalg.norm(new_direction - prev_direction) < self.tolerance:
                alpha = new_alpha
                self.alpha_history.append(alpha.copy())
                break
            alpha = new_alpha
            prev_direction = new_direction
            self.alpha_history.append(alpha.copy())

        combined_gradients: List[tf.Tensor] = []
        param_count = len(active_gradients[0])
        for param_index in range(param_count):
            reference = next(
                (grad_list[param_index] for grad_list in active_gradients if grad_list[param_index] is not None),
                None,
            )
            if reference is None:
                combined_gradients.append(None)
                continue
            stacked = []
            for objective_index, grad_list in enumerate(active_gradients):
                grad = grad_list[param_index]
                if grad is None:
                    grad = tf.zeros_like(reference)
                stacked.append(tf.cast(alpha[objective_index], tf.float32) * grad)
            combined_gradients.append(tf.add_n(stacked))

        full_alpha = np.zeros(len(gradients), dtype=np.float64)
        for local_index, global_index in enumerate(active_indices):
            full_alpha[global_index] = alpha[local_index]
        self.alpha_history.append(full_alpha.copy())
        return full_alpha, combined_gradients

    def get_alpha_history(self) -> List[np.ndarray]:
        """Return saved alpha values."""
        return list(self.alpha_history)


@dataclass
class MGDATrainStep:
    """One MGDA optimization step for a batch."""

    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    solver: MGDASolver
    loss_names: Sequence[str]

    def step(
        self,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        losses: Dict[str, tf.keras.losses.Loss],
    ) -> Dict[str, float]:
        """Execute an MGDA step and apply combined gradient."""
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.model(x_batch, training=True)
            if isinstance(predictions, list):
                predictions_main = predictions[-1]
            else:
                predictions_main = predictions
            loss_values = {
                name: tf.cast(loss_fn(y_batch, predictions_main), tf.float32)
                for name, loss_fn in losses.items()
            }

        gradients_per_loss: List[List[tf.Tensor | None]] = []
        ordered_names = [name for name in self.loss_names if name in loss_values]
        for name in ordered_names:
            gradients_per_loss.append(tape.gradient(loss_values[name], self.model.trainable_variables))
        del tape

        alphas, combined_gradients = self.solver.solve(gradients_per_loss)
        self.optimizer.apply_gradients(zip(combined_gradients, self.model.trainable_variables))

        metrics = {name: float(loss_values[name].numpy()) for name in ordered_names}
        for idx, alpha in enumerate(alphas.tolist()):
            metrics[f"alpha_{ordered_names[idx]}"] = float(alpha)
        return metrics
