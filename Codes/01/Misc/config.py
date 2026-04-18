import os

try:
    import yaml
except ImportError:
    yaml = None

# Default hyperparameter configuration for all experiments.
# These values can be overridden via YAML config files or CLI arguments.
DEFAULT_CONFIG = {
    "mode": "mgda",
    "model": "unet++",
    "deep_supervision": False,
    "pixel_loss": "bce+iou",
    "boundary_loss": "hausdorff",
    "shape_loss": "convexity",
    "pixel_weight": 1.0,
    "boundary_weight": 1.0,
    "shape_weight": 1.0,
    "batch_size": 4,
    "epochs": 50,
    "lr": 5e-4,
    "seed": 42,
    "verbose": 2,
    "output_dir": "results",
    "experiment_name": None,
    "rgb_glob": "Datasets/RGB/*.png",
    "mask_glob": "Datasets/Mask/*.tif",
    "train_split": 0.8,
    "input_height": 256,
    "input_width": 256,
    "input_channels": 3,
    "log_every": 50,
    "resume": False,
}


def load_config(path):
    """Load a YAML configuration file and return a config dictionary."""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML config files. Install it with pip install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} must define a top-level mapping.")
    return config


def save_config(path, config):
    """Save a configuration dictionary back to a YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required to save YAML config files. Install it with pip install pyyaml.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def merge_config(base, override):
    """Merge override values into the base config, ignoring None values."""
    merged = base.copy()
    for key, value in override.items():
        if value is None:
            continue
        merged[key] = value
    return merged
