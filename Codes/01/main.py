#!/usr/bin/env python3
"""
Multi-Objective Optimization Framework for Building Footprint Recognition

This module serves as the main orchestrator for running multi-objective 
optimization experiments for building footprint extraction from aerial/satellite imagery.
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf

# Configure logging
os.makedirs('outputs/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'outputs/logs/main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from config.config_manager import ConfigManager, create_default_config
from plugins import load_plugins_from_config
from src.pareto import extract_pareto_front, plot_pareto_front, generate_pareto_report
from src.evaluate import generate_comparison_table, generate_latex_table
from utils.repro import set_global_seed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Multi-Objective Optimization for Building Footprint Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config configs/experiment_config.yaml --mode full
    python main.py --config configs/experiment_config.yaml --mode quick
    python main.py --config configs/experiment_config.yaml --mode evaluate
        """
    )
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Path to the experiment configuration YAML file')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'quick', 'evaluate', 'create-config'],
                       help='Experiment mode')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--eval-dir', type=str, default=None, help='Evaluation directory')
    parser.add_argument('--create-config', type=str, default=None, help='Create default config')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2])
    return parser.parse_args()


def setup_environment(config: Dict[str, Any], gpu_id: str = '0'):
    """Set environment variables for reproducibility and GPU configuration."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    seeds = config.get('experiment', {}).get('seed', [42])
    seed = seeds[0] if isinstance(seeds, list) else seeds
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_global_seed(seed)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    
    if gpus:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.warning(f"Mixed precision not available: {e}")
    else:
        logger.info("Mixed precision disabled on CPU-only runtime")


def create_output_structure(experiment_name: str) -> Dict[str, str]:
    """Create output directory structure for experiment."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{experiment_name}_{timestamp}"
    
    output_dirs = {
        'base': f'outputs/{run_id}',
        'figures': f'outputs/{run_id}/figures',
        'tables': f'outputs/{run_id}/tables',
        'masks': f'outputs/{run_id}/masks',
        'logs': f'outputs/{run_id}/logs',
        'checkpoints': f'outputs/{run_id}/checkpoints',
        'optimization': f'outputs/{run_id}/optimization',
        'predictions': f'outputs/{run_id}/predictions',
    }
    
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return output_dirs


def save_run_metadata(output_dirs: Dict[str, str], config: Dict[str, Any], args: argparse.Namespace):
    """Save metadata about the experiment run."""
    metadata = {
        'run_id': os.path.basename(output_dirs['base']),
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'arguments': vars(args),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
    }
    
    metadata_path = os.path.join(output_dirs['base'], 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Run metadata saved to {metadata_path}")


def main():
    """Main entry point for the multi-objective optimization framework."""
    args = parse_arguments()
    
    # Handle create-config mode
    if args.mode == 'create-config' or args.create_config:
        config_path = args.create_config or 'configs/default_config.yaml'
        create_default_config(config_path)
        logger.info(f"Default configuration created at {config_path}")
        return
    
    # Validate config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Use --create-config to generate a default configuration")
        sys.exit(1)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        loaded_plugins = load_plugins_from_config(config)
        if loaded_plugins:
            logger.info("Loaded %d plugin module(s)", len(loaded_plugins))
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup environment
    setup_environment(config, args.gpu)
    
    # Create output structure
    experiment_name = config.get('experiment', {}).get('name', 'building_footprint_moo')
    output_dirs = create_output_structure(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"Starting Multi-Objective Optimization Experiment")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Run ID: {os.path.basename(output_dirs['base'])}")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 80)
    
    # Save metadata
    save_run_metadata(output_dirs, config, args)
    
    # Create experiment configuration
    experiment_config = {
        **config,
        'output_dirs': output_dirs,
        'config_path': args.config,
    }
    
    try:
        if args.mode == 'evaluate':
            # Evaluation mode
            logger.info("Running evaluation mode")
            eval_dir = args.eval_dir or config.get('eval_run_dir')
            if not eval_dir:
                logger.error("Evaluation directory not specified")
                sys.exit(1)
            logger.info(f"Evaluating models from {eval_dir}")
            
        else:
            # Run multi-objective optimization
            logger.info("Running multi-objective optimization")
            
            # Get configurations
            config_manager = ConfigManager(args.config)
            configs = config_manager.get_experiment_configs(mode=args.mode)
            
            logger.info(f"Total configurations to evaluate: {len(configs)}")
            
            # Import and run optimization
            from src.optimize import run_moo_experiment
            results = run_moo_experiment(experiment_config, args.mode)
            
            # Save raw results
            results_path = os.path.join(output_dirs['tables'], 'raw_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Raw results saved to {results_path}")
            
            # Generate comparison tables
            logger.info("Generating comparison tables")
            csv_path = os.path.join(output_dirs['tables'], 'results.csv')
            results_df = generate_comparison_table(results, csv_path)
            
            # Generate LaTeX table
            latex_path = os.path.join(output_dirs['tables'], 'results.tex')
            generate_latex_table(results, latex_path, top_k=10, sort_by='iou')
            logger.info(f"LaTeX table saved to {latex_path}")
            
            objective_names = config.get('multi_objective', {}).get('objectives')

            # Extract Pareto front
            logger.info("Extracting Pareto-optimal solutions")
            pareto_results, pareto_values = extract_pareto_front(results, objectives=objective_names)
            
            # Generate Pareto visualizations
            logger.info("Generating Pareto front visualizations")
            pareto_plots = plot_pareto_front(pareto_results, output_dirs['figures'], objectives=objective_names)
            logger.info(f"Generated {len(pareto_plots)} Pareto front plots")
            
            # Generate Pareto report
            pareto_report_path = os.path.join(output_dirs['tables'], 'pareto_report.txt')
            generate_pareto_report(pareto_results, pareto_report_path)
            
            # Summary
            logger.info("=" * 80)
            logger.info("Experiment Summary")
            logger.info("=" * 80)
            logger.info(f"Total configurations evaluated: {len(results)}")
            logger.info(f"Pareto-optimal solutions: {len(pareto_results)}")
            logger.info(f"Results saved to: {output_dirs['base']}")
            logger.info("=" * 80)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
