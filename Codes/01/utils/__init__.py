"""Utility helpers for reproducibility and experiment export."""

try:
    from .reproducibility import compute_dataset_hash
except Exception:  # pragma: no cover - allows torch-only environments
    compute_dataset_hash = None

try:
    from .repro import dataset_hash, get_git_commit, save_json, set_global_seed
except Exception:  # pragma: no cover - allows torch-only environments
    dataset_hash = None
    get_git_commit = None
    save_json = None
    set_global_seed = None

__all__ = [
    "dataset_hash",
    "get_git_commit",
    "save_json",
    "set_global_seed",
    "compute_dataset_hash",
]
