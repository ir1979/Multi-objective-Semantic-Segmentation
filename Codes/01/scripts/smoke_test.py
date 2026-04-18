#!/usr/bin/env python3
"""Run a minimal smoke test against the publication pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.experiment_runner import ExperimentRunner
from utils.config_loader import load_config
from utils.reproducibility import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one short experiment run.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    parser.add_argument("--experiment", default="unet_single", help="Experiment name")
    args = parser.parse_args()

    config = load_config(args.config)
    config = dict(config)
    config["training"] = dict(config.get("training", {}))
    config["training"]["epochs"] = 1
    set_global_seed(int(config.get("project", {}).get("seed", 42)))

    runner = ExperimentRunner(config, force=True)
    runner.run_single(args.experiment)
    status = runner.get_status()
    print(json.dumps({"experiment": args.experiment, "status": status.get(args.experiment, "unknown")}, indent=2))


if __name__ == "__main__":
    main()
