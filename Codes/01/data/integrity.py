"""Dataset integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_dataset_hash(rgb_dir: str, mask_dir: str) -> str:
    """Compute SHA-256 hash from filenames and file sizes.

    This hash excludes file contents by design for performance.
    """
    hasher = hashlib.sha256()
    for directory in sorted([Path(rgb_dir), Path(mask_dir)], key=lambda item: item.as_posix()):
        if not directory.exists():
            hasher.update(f"missing:{directory}".encode("utf-8"))
            continue
        for file_path in sorted(directory.glob("*")):
            if not file_path.is_file():
                continue
            hasher.update(file_path.name.encode("utf-8"))
            hasher.update(str(file_path.stat().st_size).encode("utf-8"))
    return hasher.hexdigest()
