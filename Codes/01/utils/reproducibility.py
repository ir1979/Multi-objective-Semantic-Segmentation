"""Reproducibility helpers for deterministic experiments."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - allows environment checks without TF
    tf = None


def set_global_seed(seed: int = 42) -> None:
    """Set all known random seeds.

    Parameters
    ----------
    seed:
        Random seed used for Python, NumPy, and TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Some TF builds do not expose deterministic kernels for every platform.
            pass


def compute_dataset_hash(data_dir: str) -> str:
    """Compute lightweight dataset hash from file names and sizes.

    Parameters
    ----------
    data_dir:
        Root directory containing dataset files.

    Returns
    -------
    str
        SHA-256 hex digest prefix (16 chars).
    """
    hasher = hashlib.sha256()
    root = Path(data_dir)
    if not root.exists():
        return "missing-dataset"

    for path in sorted(root.rglob("*")):
        if path.is_file():
            hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
            hasher.update(str(path.stat().st_size).encode("utf-8"))
    return hasher.hexdigest()[:16]
