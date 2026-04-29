#!/usr/bin/env python
"""Quick utility commands for grid search management."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import yaml


def list_results(results_dir: str = "grid_search_results", top_n: int = 10) -> None:
    """List top results from completed grid search."""
    state_file = Path(results_dir) / "grid_search_state.json"

    if not state_file.exists():
        print(f"No grid search results found at {results_dir}")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract completed points
    completed = []
    for point_id, point_data in data.items():
        if point_data["status"] == "completed":
            row = {"point_id": int(point_id)}
            row.update(point_data.get("params", {}))
            row.update(point_data.get("metrics", {}))
            completed.append(row)

    if not completed:
        print("No completed points found")
        return

    df = pd.DataFrame(completed)

    # Sort by val_iou if available
    if "val_iou" in df.columns:
        df = df.sort_values("val_iou", ascending=False)

    print(f"\nTop {min(top_n, len(df))} Results:")
    print("-" * 100)
    print(df.head(top_n).to_string(index=False))
    print("-" * 100)


def show_status(results_dir: str = "grid_search_results") -> None:
    """Show current grid search status."""
    state_file = Path(results_dir) / "grid_search_state.json"

    if not state_file.exists():
        print(f"No grid search state found at {results_dir}")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count statuses
    statuses = {}
    for point_id, point_data in data.items():
        status = point_data["status"]
        statuses[status] = statuses.get(status, 0) + 1

    print(f"\nGrid Search Status:")
    print("-" * 40)
    total = len(data)
    for status, count in sorted(statuses.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {status.upper():12} : {count:4} ({pct:5.1f}%)")
    print("-" * 40)
    print(f"  {'TOTAL':12} : {total:4}")
    print("-" * 40)


def show_config(results_dir: str = "grid_search_results", point_id: int = 0) -> None:
    """Show configuration for a specific grid point."""
    config_file = Path(results_dir) / f"point_{point_id:06d}" / "config.yaml"

    if not config_file.exists():
        print(f"Config not found for point {point_id} at {config_file}")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration for Point {point_id}:")
    print("-" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 60)


def show_error_summary(results_dir: str = "grid_search_results") -> None:
    """Show error summary if any."""
    error_file = Path(results_dir) / "error_log.json"

    if not error_file.exists():
        print("No errors recorded")
        return

    with open(error_file, "r", encoding="utf-8") as f:
        errors = json.load(f)

    print(f"\nError Summary ({len(errors)} errors):")
    print("-" * 60)

    # Group by type
    by_type = {}
    for error in errors:
        error_type = error.get("type", "Unknown")
        by_type[error_type] = by_type.get(error_type, 0) + 1

    for error_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")

    print("\nFirst 5 errors:")
    for error in errors[:5]:
        print(f"  - {error['type']}: {error['message']}")
    print("-" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Grid Search Management Utility")
    parser.add_argument("--results-dir", default="grid_search_results", help="Results directory")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List top results")
    list_parser.add_argument("--top-n", type=int, default=10, help="Number of top results to show")

    # Status command
    subparsers.add_parser("status", help="Show grid search status")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration for a point")
    config_parser.add_argument("--point-id", type=int, default=0, help="Point ID")

    # Errors command
    subparsers.add_parser("errors", help="Show error summary")

    args = parser.parse_args()

    if args.command == "list":
        list_results(args.results_dir, args.top_n)
    elif args.command == "status":
        show_status(args.results_dir)
    elif args.command == "config":
        show_config(args.results_dir, args.point_id)
    elif args.command == "errors":
        show_error_summary(args.results_dir)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
