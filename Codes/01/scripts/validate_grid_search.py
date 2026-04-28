#!/usr/bin/env python
"""Utility script to validate and prepare grid search setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from experiments.grid_search import GridSearchConfig
from utils.config_validator import GridSearchConfigValidator, validate_config_file


def main() -> int:
    """Validate grid search setup."""
    parser = argparse.ArgumentParser(description="Validate Grid Search Configuration")
    parser.add_argument("--config", default="configs/grid_search.yaml", help="Path to grid search config")
    parser.add_argument("--dry-run", action="store_true", help="Generate points without running")
    parser.add_argument("--show-points", action="store_true", help="Display generated grid points")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    print(f"📋 Validating: {config_path}\n")

    # Validate configuration
    is_valid, errors, warnings = validate_config_file(str(config_path))

    if not is_valid:
        print("\n❌ Configuration validation failed!")
        return 1

    if args.dry_run or args.show_points:
        print("\n" + "=" * 80)
        print("GRID POINTS GENERATION (DRY RUN)")
        print("=" * 80)

        # Load and generate points
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        grid_config = GridSearchConfig(config)
        points = grid_config.generate_points()

        print(f"\n✅ Generated {len(points)} grid points\n")

        if args.show_points:
            print("First 10 points:")
            for i, point in enumerate(points[:10]):
                print(f"\n  Point {i}:")
                for key, val in point.items():
                    print(f"    {key}: {val}")

            if len(points) > 10:
                print(f"\n  ... and {len(points) - 10} more points")

        print("\n✅ Grid generation successful! Ready to run grid search.")
        print(f"\nRun with: python run_grid_search.py --config {config_path}")

        return 0

    print("\n✅ Configuration is valid and ready!")
    print(f"Run with: python run_grid_search.py --config {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
