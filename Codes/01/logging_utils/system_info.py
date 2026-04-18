"""System information capture utilities."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

import tensorflow as tf


def _run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "N/A"


def capture_system_info() -> dict:
    """Collect runtime system information."""
    disk = shutil.disk_usage(Path.cwd())
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for gpu in gpus:
        try:
            gpu_names.append(gpu.name)
        except Exception:
            gpu_names.append("Unknown GPU")

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_version": _run_command(
            "python3 - <<'PY'\n"
            "import re, subprocess\n"
            "try:\n"
            "    out = subprocess.check_output(['nvcc', '--version'], text=True)\n"
            "    m = re.search(r'release\\s+([0-9.]+)', out)\n"
            "    print(m.group(1) if m else 'N/A')\n"
            "except Exception:\n"
            "    print('N/A')\n"
            "PY"
        ),
        "cudnn_version": _run_command("python3 - <<'PY'\nimport tensorflow as tf\nprint(tf.sysconfig.get_build_info().get('cudnn_version', 'N/A'))\nPY"),
        "gpu_names": gpu_names or ["CPU-only"],
        "cpu": platform.processor() or "N/A",
        "cpu_cores": int(_run_command("python3 - <<'PY'\nimport os\nprint(os.cpu_count() or 0)\nPY") or 0),
        "total_ram": _run_command("python3 - <<'PY'\nimport os\nimport psutil\nprint(psutil.virtual_memory().total)\nPY"),
        "disk_free_bytes": disk.free,
        "git_commit": _run_command("git rev-parse HEAD"),
    }
