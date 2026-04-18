import os

try:
    import psutil
except ImportError:
    psutil = None


def get_process_memory_mb():
    """Return the current process memory usage in megabytes, or None if unavailable."""
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0 / 1024.0
