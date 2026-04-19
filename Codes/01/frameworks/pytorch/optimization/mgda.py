"""MGDA implementation for PyTorch training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for grad in grads:
        if grad is None:
            continue
        chunks.append(grad.detach().reshape(-1).cpu().numpy().astype(np.float64))
    if not chunks:
        return np.zeros((1,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _non_zero(grads: Sequence[Optional[torch.Tensor]], epsilon: float) -> bool:
    for grad in grads:
        if grad is None:
            continue
        if float(torch.norm(grad).item()) > epsilon:
            return True
    return False


@dataclass
class MGDASolver:
    """Frank-Wolfe solver for MGDA simplex weights."""

    max_iterations: int = 50
    tolerance: float = 1e-6
    normalize_gradients: bool = True
    epsilon: float = 1e-12
    alpha_history: List[np.ndarray] = field(default_factory=list)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize_gradients:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, self.epsilon)
        return vectors / norms

    def solve(
        self, gradients: List[List[Optional[torch.Tensor]]]
    ) -> Tuple[np.ndarray, List[Optional[torch.Tensor]]]:
        """Solve for minimum-norm convex combination of gradients."""
        if not gradients:
            raise ValueError("At least one objective gradient list is required.")

        active = [idx for idx, grad in enumerate(gradients) if _non_zero(grad, self.epsilon)]
        if not active:
            active = [0]
        active_grads = [gradients[idx] for idx in active]
        vectors = np.stack([_flatten_grads(g) for g in active_grads], axis=0)
        vectors = self._normalize(vectors)
        n_obj = vectors.shape[0]

        if n_obj == 1:
            alpha_local = np.array([1.0], dtype=np.float64)
        else:
            alpha_local = np.ones(n_obj, dtype=np.float64) / float(n_obj)
            prev = vectors.T @ alpha_local
            for _ in range(self.max_iterations):
                current = vectors.T @ alpha_local
                inner = vectors @ current
                target = int(np.argmin(inner))
                direction = vectors[target] - current
                denom = float(np.dot(direction, direction))
                if denom <= self.epsilon:
                    break
                gamma = float(np.clip(-np.dot(current, direction) / denom, 0.0, 1.0))
                vertex = np.zeros_like(alpha_local)
                vertex[target] = 1.0
                next_alpha = (1.0 - gamma) * alpha_local + gamma * vertex
                if np.linalg.norm((vectors.T @ next_alpha) - prev) < self.tolerance:
                    alpha_local = next_alpha
                    break
                alpha_local = next_alpha
                prev = vectors.T @ alpha_local

        full_alpha = np.zeros(len(gradients), dtype=np.float64)
        for idx, source in enumerate(active):
            full_alpha[source] = alpha_local[idx]
        if full_alpha.sum() <= self.epsilon:
            full_alpha[:] = 1.0 / float(len(full_alpha))
        else:
            full_alpha /= full_alpha.sum()
        self.alpha_history.append(full_alpha.copy())

        param_count = len(gradients[0])
        combined: List[Optional[torch.Tensor]] = []
        for param_idx in range(param_count):
            reference = next(
                (grad_list[param_idx] for grad_list in gradients if grad_list[param_idx] is not None),
                None,
            )
            if reference is None:
                combined.append(None)
                continue
            mixed = torch.zeros_like(reference)
            for obj_idx, grad_list in enumerate(gradients):
                grad = grad_list[param_idx]
                if grad is None:
                    grad = torch.zeros_like(reference)
                mixed = mixed + grad * float(full_alpha[obj_idx])
            combined.append(mixed)
        return full_alpha, combined

    def get_alpha_history(self) -> List[np.ndarray]:
        """Return tracked alpha values."""
        return list(self.alpha_history)


@dataclass
class MGDATrainStep:
    """Single MGDA training step wrapper."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    solver: MGDASolver
    loss_names: Sequence[str]

    def step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        losses: Dict[str, nn.Module],
    ) -> Dict[str, float]:
        self.model.train()
        prediction = self.model(x_batch)
        if isinstance(prediction, list):
            prediction = prediction[-1]

        loss_values: Dict[str, torch.Tensor] = {}
        gradients_per_loss: List[List[Optional[torch.Tensor]]] = []
        ordered = [name for name in self.loss_names if name in losses]

        for name in ordered:
            self.optimizer.zero_grad(set_to_none=True)
            value = losses[name](y_batch, prediction)
            value.backward(retain_graph=True)
            grads = [
                None if param.grad is None else param.grad.detach().clone()
                for param in self.model.parameters()
            ]
            gradients_per_loss.append(grads)
            loss_values[name] = value.detach()

        alphas, combined = self.solver.solve(gradients_per_loss)
        self.optimizer.zero_grad(set_to_none=True)
        for param, grad in zip(self.model.parameters(), combined):
            if grad is not None:
                param.grad = grad
        self.optimizer.step()

        metrics = {name: float(loss_values[name].item()) for name in ordered}
        for idx, name in enumerate(ordered):
            if idx < len(alphas):
                metrics[f"alpha_{name}"] = float(alphas[idx])
        return metrics
