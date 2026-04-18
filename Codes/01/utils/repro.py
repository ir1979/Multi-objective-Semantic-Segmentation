import hashlib
import json
import os
import random
import subprocess

import numpy as np
import tensorflow as tf


def set_global_seed(seed):
    """Set the random seed for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_file_hash(path, chunk_size=65536):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def dataset_hash(file_paths):
    """Compute a reproducible hash for a list of dataset file paths."""
    hasher = hashlib.sha256()
    for path in sorted(file_paths):
        hasher.update(path.encode("utf-8"))
        if os.path.exists(path):
            hasher.update(compute_file_hash(path).encode("utf-8"))
    return hasher.hexdigest()


def get_git_commit():
    """Return the current git commit hash or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_json(path, data):
    """Save a Python dictionary to JSON with indentation."""
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
