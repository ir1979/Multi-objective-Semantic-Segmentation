"""Stratified split utilities for density-aware partitioning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class StratifiedSplitter:
    """Split sample indices into train/val/test sets with density stratification."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    bins: int
    seed: int

    def split(self, densities: Sequence[float]) -> Dict[str, List[int]]:
        """Compute deterministic stratified split.

        Parameters
        ----------
        densities:
            Building density per sample in [0, 1].
        """
        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0.")

        densities_array = np.asarray(densities, dtype=np.float32)
        if densities_array.size == 0:
            return {"train": [], "val": [], "test": []}

        bin_edges = np.linspace(0.0, 1.0, self.bins + 1, dtype=np.float32)
        labels = np.digitize(densities_array, bin_edges[1:-1], right=False)

        rng = np.random.default_rng(self.seed)
        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for label in range(self.bins):
            bin_idx = np.where(labels == label)[0]
            if bin_idx.size == 0:
                continue
            permuted = rng.permutation(bin_idx)
            n_total = permuted.size
            n_train = int(round(self.train_ratio * n_total))
            n_val = int(round(self.val_ratio * n_total))
            n_train = min(n_train, n_total)
            n_val = min(n_val, max(0, n_total - n_train))
            n_test_start = n_train + n_val

            train_indices.extend(permuted[:n_train].tolist())
            val_indices.extend(permuted[n_train:n_test_start].tolist())
            test_indices.extend(permuted[n_test_start:].tolist())

        # Deterministic ordering keeps downstream processing reproducible.
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        return {"train": train_indices, "val": val_indices, "test": test_indices}

    @staticmethod
    def save_split(
        split: Dict[str, List[int]],
        output_path: str,
        rgb_paths: Sequence[str] | None = None,
        mask_paths: Sequence[str] | None = None,
    ) -> None:
        """Save split indices (and optional file paths) to JSON."""
        serializable: Dict[str, object] = {
            "indices": split,
        }
        if rgb_paths is not None:
            serializable["rgb_paths"] = list(rgb_paths)
        if mask_paths is not None:
            serializable["mask_paths"] = list(mask_paths)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
