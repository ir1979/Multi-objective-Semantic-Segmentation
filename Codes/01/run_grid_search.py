#!/usr/bin/env python
"""Main entry point for grid search experimentation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from experiments.grid_search import GridSearchRunner
from experiments.results_aggregator import GridSearchResultsAggregator
from logging_utils.logger import DualLogger
from utils.error_handling import ErrorHandler


def main() -> int:
    """Execute grid search pipeline."""
    parser = argparse.ArgumentParser(description="Grid Search for Multi-Objective Semantic Segmentation")
    parser.add_argument("--config", default="configs/grid_search.yaml", help="Path to grid search config")
    parser.add_argument("--results-dir", default="grid_search_results", help="Results directory")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--force-restart", action="store_true", help="Force restart (ignore checkpoints)")
    parser.add_argument("--generate-reports-only", action="store_true", help="Only generate reports from existing results")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per point")
    args = parser.parse_args()

    # Setup logging
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = DualLogger(results_dir / "grid_search_main.log", console_level="INFO", file_level="DEBUG")
    error_handler = ErrorHandler(logger)

    start_time = time.time()

    try:
        logger.info("=" * 80)
        logger.info("GRID SEARCH PIPELINE STARTED")
        logger.info("=" * 80)

        # Validate config file
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        logger.info(f"Using config: {config_path}")
        logger.info(f"Results directory: {results_dir}")

        if args.generate_reports_only:
            logger.info("Generating reports from existing results only")
            state_file = results_dir / "grid_search_state.json"
            if not state_file.exists():
                logger.error(f"State file not found: {state_file}")
                return 1

            aggregator = GridSearchResultsAggregator(state_file, results_dir / "paper_outputs")
            manifest = aggregator.generate_full_paper_report()
            logger.info(f"Paper report complete. {len(manifest)} top-level sections")
            return 0

        # Initialize grid search runner
        runner = GridSearchRunner(
            config_path=str(config_path),
            results_dir=str(results_dir),
            resume=not args.force_restart,
            max_retries=args.max_retries,
            logger=logger,
        )

        # Initialize or resume grid
        runner.initialize_grid(force=args.force_restart)

        # Run search
        runner.run_search(start_from_pending=not args.force_restart)

        # Generate results report
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING RESULTS REPORT")
        logger.info("=" * 80)

        state_file = results_dir / "grid_search_state.json"
        paper_dir = results_dir / "paper_outputs"
        paper_dir.mkdir(parents=True, exist_ok=True)

        aggregator = GridSearchResultsAggregator(state_file, paper_dir)
        manifest = aggregator.generate_full_paper_report()

        logger.info(f"\nPaper report complete. Manifest written to {paper_dir / 'report_manifest.json'}")

        # Print best configurations
        if not aggregator.df.empty:
            top5 = aggregator.df.nlargest(5, "test_iou")[["model_architecture", "encoder_filters", "pixel_loss_type", "test_iou", "test_boundary_f1"]]
            logger.info("\nTop 5 Configurations by IoU:")
            logger.info(top5.to_string())

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH PIPELINE COMPLETED")
        logger.info(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/3600:.2f} hours)")
        logger.info("=" * 80)

        return 0

    except Exception as exc:
        error_handler.handle_exception(exc, context="grid_search_main", allow_continue=False)
        logger.exception("Fatal error in grid search pipeline")

        elapsed = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed:.2f} seconds")

        # Save error log
        error_log_file = results_dir / "error_log.json"
        error_handler.save_error_log(str(error_log_file))
        logger.info(f"Error log saved to {error_log_file}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
