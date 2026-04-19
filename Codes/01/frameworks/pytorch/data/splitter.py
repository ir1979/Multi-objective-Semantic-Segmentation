"""Density-stratified splitting utilities for PyTorch pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class StratifiedSplitter:
    """Split indices into train/val/test partitions by density bins."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    bins: int
    seed: int

    def split(self, densities: Sequence[float]) -> Dict[str, List[int]]:
        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0.")
        densities = np.asarray(densities, dtype=np.float32)
        labels = np.digitize(densities, np.linspace(0.0, 1.0, self.bins + 1)[1:-1])
        rng = np.random.default_rng(self.seed)

        train, val, test = [], [], []
        for label in range(self.bins):
            members = np.where(labels == label)[0]
            if members.size == 0:
                continue
            order = rng.permutation(members)
            n = len(order)
            n_train = int(round(self.train_ratio * n))
            n_val = int(round(self.val_ratio * n))
            n_train = min(n_train, n)
            n_val = min(n_val, max(0, n - n_train))
            train.extend(order[:n_train].tolist())
            val.extend(order[n_train : n_train + n_val].tolist())
            test.extend(order[n_train + n_val :].tolist())

        train.sort()
        val.sort()
        test.sort()
        used = set(train) | set(val) | set(test)
        missing = sorted(set(range(len(densities))) - used)
        if missing:
            train.extend(missing)
            train.sort()
        return {"train": train, "val": val, "test": test}

    @staticmethod
    def save_split(split: Dict[str, List[int]], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"indices": split}, handle, indent=2)
