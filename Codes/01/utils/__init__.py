"""Utility helpers for reproducibility, configuration, and experiment export."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "dataset_hash",
    "get_git_commit",
    "save_json",
    "set_global_seed",
    "compute_dataset_hash",
]


def __getattr__(name: str) -> Any:
    if name in {"dataset_hash", "get_git_commit", "save_json", "set_global_seed"}:
        module = import_module(".repro", __name__)
        return getattr(module, name)
    if name == "compute_dataset_hash":
        module = import_module(".reproducibility", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
