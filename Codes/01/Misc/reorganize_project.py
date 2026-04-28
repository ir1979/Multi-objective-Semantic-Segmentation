"""Reorganize project structure."""

import shutil
from pathlib import Path

base_path = Path(".")

# Create new directories
new_dirs = [
    "analysis",
    "scripts/grid_search",
    "docs/notebooks",
]

for dir_path in new_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {dir_path}")

# Move files to Misc if they exist
files_to_move = [
    ("scripts/smoke_test.py", "Misc/smoke_test.py"),
    ("tmp", "Misc/tmp"),
    ("01.rar", "Misc/01.rar"),
]

for src, dst in files_to_move:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            shutil.rmtree(src_path)
        else:
            shutil.copy2(src_path, dst_path)
            src_path.unlink()
        print(f"✓ Moved {src} → {dst}")
    else:
        print(f"⊘ Skipped {src} (not found)")

print("\nProject structure reorganized successfully!")
