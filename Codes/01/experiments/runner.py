import os

try:
    import yaml
except ImportError:
    yaml = None


def create_experiment_folder(base_path, experiment_name):
    """Create a structured experiment folder for outputs."""
    path = os.path.join(base_path, experiment_name)
    subdirs = ["checkpoints", "figures", "tables", "results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)
    return path


def write_experiment_config(path, config):
    """Write experiment configuration to YAML if PyYAML is installed."""
    if yaml is None:
        raise ImportError("PyYAML is required to write config.yaml. Install it with pip install pyyaml.")
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path
