"""Pluggable search-strategy interfaces and implementations.

This module provides a lightweight strategy layer so the grid search runner can
swap exploration policies without changing orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from utils.registry import Registry


class BaseSearchStrategy(ABC):
    """Abstract search strategy interface."""

    name: str = "base"

    @abstractmethod
    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select a subset/order of candidate points for evaluation."""


@dataclass
class FullSearchStrategy(BaseSearchStrategy):
    """Evaluate all generated points."""

    name: str = "full"

    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(points)


@dataclass
class GridSearchStrategy(BaseSearchStrategy):
    """Explicit Cartesian grid-search strategy.

    Behaviour is identical to full enumeration, but it is exposed separately so
    configuration, logging, and future reporting can refer to grid search as a
    formal optimization method.
    """

    name: str = "grid_search"

    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(points)


@dataclass
class RandomSearchStrategy(BaseSearchStrategy):
    """Uniform random subset without replacement."""

    name: str = "random"

    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        n_points = int(selection_cfg.get("n_points", min(50, len(points))))
        seed = int(selection_cfg.get("random_seed", 42))
        if not points:
            return []
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(points), size=min(n_points, len(points)), replace=False)
        return [points[i] for i in sorted(idx)]


@dataclass
class LatinHypercubeSearchStrategy(BaseSearchStrategy):
    """Simple quasi-uniform subsampling over an existing candidate list."""

    name: str = "latin_hypercube"

    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        n_points = int(selection_cfg.get("n_points", min(50, len(points))))
        seed = int(selection_cfg.get("random_seed", 42))
        if not points:
            return []
        n_points = max(1, min(n_points, len(points)))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(points))
        step = max(1, len(points) // n_points)
        idx = sorted(perm[::step][:n_points].tolist())
        return [points[i] for i in idx]


# Placeholder for future evolutionary strategy integration.
@dataclass
class NSGA2SearchStrategy(BaseSearchStrategy):
    """Registration-only placeholder strategy for future NSGA-II extension."""

    name: str = "nsga2"

    def select(self, points: List[Dict[str, Any]], selection_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        # NSGA-II needs online objective feedback; here we keep deterministic
        # behaviour by returning points unchanged until a full MOEA loop is added.
        return list(points)


SEARCH_STRATEGY_REGISTRY: Registry[BaseSearchStrategy] = Registry("search strategy")
SEARCH_STRATEGY_REGISTRY.register("full", FullSearchStrategy(), aliases=("grid",))
SEARCH_STRATEGY_REGISTRY.register("grid_search", GridSearchStrategy(), aliases=("gridsearch",))
SEARCH_STRATEGY_REGISTRY.register("random", RandomSearchStrategy())
SEARCH_STRATEGY_REGISTRY.register("latin_hypercube", LatinHypercubeSearchStrategy(), aliases=("lhs",))
SEARCH_STRATEGY_REGISTRY.register("nsga2", NSGA2SearchStrategy(), aliases=("nsga-ii",))


def get_search_strategy(name: str) -> BaseSearchStrategy:
    """Resolve a search strategy from the registry."""
    return SEARCH_STRATEGY_REGISTRY.get(name).value
