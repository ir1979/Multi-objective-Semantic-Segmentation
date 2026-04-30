"""Minimal quick-start demonstration script."""

from __future__ import annotations

from utils.config_loader import load_config


def main() -> None:
    config = load_config("configs/default.yaml")
    print("Loaded project:", config["project_name"])
    print("Model architecture:", config["model_architecture"])
    print("Loss strategy:", config["loss_strategy"])
    print("Run `python run_all.py --experiment unet_single` to start training.")


if __name__ == "__main__":
    main()
