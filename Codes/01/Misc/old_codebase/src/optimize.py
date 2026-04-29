"""Multi-objective optimization using NSGA-II.

This module implements multi-objective optimization for building footprint
segmentation using the NSGA-II algorithm from pymoo. It searches the
architecture-hyperparameter space to find Pareto-optimal trade-offs between
accuracy, model complexity, and inference time.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json

# Import pymoo for multi-objective optimization
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    from pymoo.indicators.hv import Hypervolume
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logging.warning("pymoo not installed. NSGA-II optimization will not be available.")
    Problem = object
    NSGA2 = None
    minimize = None
    Scatter = None
    Hypervolume = None

import tensorflow as tf
from tensorflow import keras

from config.config_manager import ExperimentConfig
from models.model_factory import build_model, get_model_info, estimate_flops, is_model_configuration_supported, measure_inference_time
from plugins import load_plugins_from_config
from src.objectives import ObjectiveSpec, resolve_objective_specs
from training.trainer import train_from_config

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveVector:
    """Container for dynamic objective values."""
    specs: List[ObjectiveSpec]
    values: Dict[str, float]

    def to_array(self) -> np.ndarray:
        return np.array([spec.to_minimization_value(self.values) for spec in self.specs], dtype=float)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)


class SegmentationProblem(Problem):
    """Multi-objective optimization problem for segmentation.
    
    This class defines the optimization problem for finding Pareto-optimal
    segmentation models. Each solution represents a model configuration,
    and objectives include accuracy, complexity, and speed.
    
    Parameters
    ----------
    config_space : List[ExperimentConfig]
        List of possible experiment configurations
    dataset : Any
        Dataset for evaluation
    n_objectives : int
        Number of objectives (default: 5)
    """
    
    def __init__(
        self,
        config_space: List[ExperimentConfig],
        dataset: Any,
        objective_specs: List[ObjectiveSpec],
        inference_runs: int = 50,
        warmup_runs: int = 10,
        **kwargs
    ):
        self.config_space = config_space
        self.dataset = dataset
        self.objective_specs = objective_specs
        self.inference_runs = inference_runs
        self.warmup_runs = warmup_runs
        self.n_configs = len(config_space)
        self.evaluation_cache: Dict[str, ObjectiveVector] = {}
        
        super().__init__(
            n_var=1,  # Single variable: index into config space
            n_obj=len(objective_specs),
            n_constr=0,
            xl=0,  # Lower bound: index 0
            xu=self.n_configs - 1,  # Upper bound: last index
            vtype=int,  # Integer variables
            **kwargs
        )
    
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """Evaluate a set of solutions.
        
        Parameters
        ----------
        x : np.ndarray
            Array of solution indices
        out : Dict
            Output dictionary for objective values
        """
        objectives = []
        
        for idx in x:
            config = self.config_space[int(idx)]
            obj_vector = self._evaluate_config(config)
            objectives.append(obj_vector.to_array())
        
        out["F"] = np.array(objectives)
    
    def _evaluate_config(self, config: ExperimentConfig) -> ObjectiveVector:
        """Evaluate a single configuration.
        
        Parameters
        ----------
        config : ExperimentConfig
            Configuration to evaluate
        
        Returns
        -------
        ObjectiveVector
            Objective values for this configuration
        """
        # Check cache
        if config.experiment_id in self.evaluation_cache:
            logger.info(f"Using cached evaluation for {config.experiment_id}")
            return self.evaluation_cache[config.experiment_id]
        
        logger.info(f"Evaluating configuration: {config.experiment_id}")
        
        try:
            # Build model
            model = build_model(
                architecture=config.architecture,
                encoder=config.encoder,
                encoder_weights=config.encoder_weights,
                input_shape=(config.tile_size, config.tile_size, 3),
                num_classes=1,
            )
            
            # Get model complexity metrics
            model_info = get_model_info(model)
            param_count = model_info['total_params']
            flops = estimate_flops(model)
            
            # Measure inference time
            timing = measure_inference_time(
                model,
                input_shape=(1, config.tile_size, config.tile_size, 3),
                num_runs=self.inference_runs,
                warmup_runs=self.warmup_runs,
            )
            inference_time = timing['mean_ms']
            
            # Evaluate using real training/validation instead of random placeholders.
            eval_output_dir = None
            if isinstance(self.dataset, dict):
                eval_output_dir = self.dataset.get('output_dirs', {}).get('optimization')
            result = evaluate_experiment_config(config, self.dataset, output_dir=eval_output_dir)
            result['param_count'] = param_count
            result['flops'] = flops
            result['inference_time'] = inference_time

            obj_vector = ObjectiveVector(
                specs=self.objective_specs,
                values={spec.name: spec.extract(result) for spec in self.objective_specs},
            )
            
            # Cache result
            self.evaluation_cache[config.experiment_id] = obj_vector
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return obj_vector
            
        except Exception as e:
            logger.error(f"Error evaluating {config.experiment_id}: {e}")
            # Return worst possible values
            return ObjectiveVector(
                specs=self.objective_specs,
                values={
                    spec.name: (1e12 if spec.minimize else 0.0)
                    for spec in self.objective_specs
                },
            )
    
def build_training_config(
    experiment: ExperimentConfig,
    root_config: Dict[str, Any],
    epoch_override: Optional[int] = None
) -> Dict[str, Any]:
    """Flatten experiment and root config into the trainer input format."""
    dataset_cfg = root_config.get('dataset', {})
    models_cfg = root_config.get('models', {})
    training_cfg = root_config.get('training', {})

    train_config = experiment.to_dict()
    train_config.update({
        'rgb_path': dataset_cfg.get('rgb_path', 'datasets/RGB'),
        'mask_path': dataset_cfg.get('mask_path', 'datasets/Mask'),
        'train_val_test_split': dataset_cfg.get('train_val_test_split', [0.7, 0.15, 0.15]),
        'deep_supervision': models_cfg.get('deep_supervision', False),
        'num_classes': models_cfg.get('num_classes', 1),
        'early_stopping': training_cfg.get('early_stopping', {'enabled': True, 'patience': 10}),
        'scheduler_factor': training_cfg.get('scheduler_factor', 0.5),
        'scheduler_patience': training_cfg.get('scheduler_patience', 5),
    })
    if epoch_override is not None:
        train_config['epochs'] = epoch_override
    return train_config


def evaluate_experiment_config(
    experiment: ExperimentConfig,
    root_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    epoch_override: Optional[int] = None
) -> Dict[str, Any]:
    """Train/evaluate a configuration and return real validation metrics."""
    train_config = build_training_config(
        experiment,
        root_config,
        epoch_override=epoch_override,
    )
    summary = train_from_config(
        train_config,
        output_dir=output_dir or "outputs/optimization",
        verbose=0,
        resume=False,
        auto_batch_size=False,
    )
    result = {
        'experiment_id': experiment.experiment_id,
        'architecture': experiment.architecture,
        'encoder': experiment.encoder,
        'loss_function': experiment.loss_function,
        'learning_rate': experiment.learning_rate,
        'batch_size': experiment.batch_size,
        'epochs': experiment.epochs,
        'seed': experiment.seed,
        'tile_size': experiment.tile_size,
        'summary': summary,
        'summary_path': os.path.join(output_dir or "outputs/optimization", experiment.experiment_id, "summary.json"),
        'iou': summary.get('final_val_iou', 0.0) or 0.0,
        'f1_score': summary.get('final_val_dice', 0.0) or 0.0,
        'precision': summary.get('final_val_precision', summary.get('final_val_precision_score', 0.0)) or 0.0,
        'recall': summary.get('final_val_recall', summary.get('final_val_recall_score', 0.0)) or 0.0,
        'accuracy': summary.get('final_val_accuracy', summary.get('final_val_pixel_accuracy', 0.0)) or 0.0,
        'boundary_iou': summary.get('final_val_boundary_iou', 0.0) or 0.0,
        'boundary_f1': summary.get('final_val_boundary_f1', 0.0) or 0.0,
        'train_time_seconds': summary.get('train_time_seconds', 0.0) or 0.0,
        'memory_footprint_mb': summary.get('memory_footprint_mb', 0.0) or 0.0,
    }
    return result


def filter_supported_configs(configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
    """Drop configurations that are known to be unsupported in the current env."""
    filtered = []
    for config in configs:
        supported, reason = is_model_configuration_supported(
            config.architecture,
            input_shape=(config.tile_size, config.tile_size, 3),
        )
        if not supported:
            logger.warning("Skipping %s because %s", config.experiment_id, reason)
            continue
        filtered.append(config)
    return filtered


def run_nsga2_optimization(
    config_space: List[ExperimentConfig],
    dataset: Any,
    objective_specs: List[ObjectiveSpec],
    population_size: int = 50,
    n_generations: int = 100,
    seed: int = 42,
    output_dir: str = "outputs/optimization",
    inference_runs: int = 50,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """Run NSGA-II optimization on the configuration space.
    
    Parameters
    ----------
    config_space : List[ExperimentConfig]
        List of possible configurations
    dataset : Any
        Dataset for evaluation
    population_size : int
        Population size for NSGA-II
    n_generations : int
        Number of generations
    seed : int
        Random seed
    output_dir : str
        Directory to save results
    
    Returns
    -------
    Dict[str, Any]
        Optimization results including Pareto front
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo is required for NSGA-II optimization. "
                         "Install with: pip install pymoo")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting NSGA-II optimization")
    logger.info(f"Configuration space size: {len(config_space)}")
    logger.info(f"Population size: {population_size}")
    logger.info(f"Generations: {n_generations}")
    
    # Define problem
    problem = SegmentationProblem(
        config_space=config_space,
        dataset=dataset,
        objective_specs=objective_specs,
        inference_runs=inference_runs,
        warmup_runs=warmup_runs,
    )
    
    # Configure algorithm
    algorithm = NSGA2(
        pop_size=population_size,
        seed=seed,
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=seed,
        verbose=True,
        save_history=True
    )
    
    # Extract Pareto front
    pareto_front = res.F
    pareto_configs = [config_space[int(res.X[i])] for i in range(len(res.X))]
    
    # Calculate hypervolume
    ref_point = []
    for spec in objective_specs:
        observed = pareto_front[:, len(ref_point)]
        ref_point.append(float(np.max(observed) * 1.05 + 1e-9))
    ref_point = np.array(ref_point, dtype=float)
    hv = Hypervolume(ref_point=ref_point)
    hypervolume = hv.do(pareto_front)
    
    # Save results
    results = {
        'pareto_front': pareto_front.tolist(),
        'pareto_config_ids': [c.experiment_id for c in pareto_configs],
        'objective_names': [spec.name for spec in objective_specs],
        'hypervolume': float(hypervolume),
        'n_evaluations': len(problem.evaluation_cache),
        'config_space_size': len(config_space),
    }
    
    results_path = os.path.join(output_dir, 'nsga2_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization complete. Hypervolume: {hypervolume:.4f}")
    logger.info(f"Results saved to {results_path}")
    
    return results


def extract_pareto_front(
    results: List[Dict[str, Any]],
    objectives: List[str] = ['iou', 'f1_score', 'param_count', 'flops', 'inference_time']
) -> List[Dict[str, Any]]:
    """Extract Pareto front from a list of results.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of experiment results
    objectives : List[str]
        Names of objectives to consider
    
    Returns
    -------
    List[Dict[str, Any]]
        Pareto-optimal solutions
    """
    if not results:
        return []
    
    # Convert to objective vectors
    # For minimization: negate accuracy metrics (iou, f1)
    obj_matrix = []
    objective_specs = resolve_objective_specs(objectives)
    for r in results:
        vec = [spec.to_minimization_value(r) for spec in objective_specs]
        obj_matrix.append(vec)
    
    obj_matrix = np.array(obj_matrix)
    
    # Find non-dominated solutions
    pareto_mask = np.ones(len(obj_matrix), dtype=bool)
    for i, vec in enumerate(obj_matrix):
        if pareto_mask[i]:
            # Check if any other solution dominates this one
            for j, other in enumerate(obj_matrix):
                if i != j and pareto_mask[j]:
                    # Check if other dominates vec
                    if np.all(other <= vec) and np.any(other < vec):
                        pareto_mask[i] = False
                        break
    
    pareto_results = [results[i] for i in range(len(results)) if pareto_mask[i]]
    
    logger.info(f"Extracted {len(pareto_results)} Pareto-optimal solutions from {len(results)} total")
    
    return pareto_results


def grid_search_optimization(
    config_space: List[ExperimentConfig],
    dataset: Any,
    output_dir: str = "outputs/optimization",
    max_configs: Optional[int] = None,
    epoch_override: Optional[int] = None,
    inference_runs: int = 50,
    warmup_runs: int = 10,
    objective_specs: Optional[List[ObjectiveSpec]] = None,
) -> List[Dict[str, Any]]:
    """Run exhaustive grid search over configuration space.
    
    Parameters
    ----------
    config_space : List[ExperimentConfig]
        List of configurations to evaluate
    dataset : Any
        Dataset for evaluation
    output_dir : str
        Directory to save results
    max_configs : int or None
        Maximum number of configurations to evaluate (None for all)
    
    Returns
    -------
    List[Dict[str, Any]]
        List of evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs_to_eval = config_space[:max_configs] if max_configs else config_space
    
    logger.info(f"Starting grid search over {len(configs_to_eval)} configurations")
    
    results = []
    objective_specs = objective_specs or resolve_objective_specs()
    
    for i, config in enumerate(configs_to_eval):
        logger.info(f"Evaluating configuration {i+1}/{len(configs_to_eval)}: {config.experiment_id}")
        
        try:
            # Build and evaluate model
            model = build_model(
                architecture=config.architecture,
                encoder=config.encoder,
                encoder_weights=config.encoder_weights,
                input_shape=(config.tile_size, config.tile_size, 3),
                num_classes=1,
            )
            
            # Get complexity metrics
            model_info = get_model_info(model)
            flops = estimate_flops(model)
            timing = measure_inference_time(
                model,
                input_shape=(1, config.tile_size, config.tile_size, 3),
                num_runs=inference_runs,
                warmup_runs=warmup_runs,
            )
            
            eval_result = evaluate_experiment_config(
                config,
                dataset,
                output_dir=output_dir,
                epoch_override=epoch_override,
            )
            result = dict(eval_result)
            result.update({
                'param_count': model_info['total_params'],
                'trainable_params': model_info['trainable_params'],
                'non_trainable_params': model_info['non_trainable_params'],
                'flops': flops,
                'inference_time': timing['mean_ms'],
                'inference_time_std': timing['std_ms'],
                'model_info': model_info,
                'timing': timing,
                'train_summary_path': os.path.join(output_dir, config.experiment_id, 'summary.json'),
            })
            result['objectives'] = {
                spec.name: spec.extract(result)
                for spec in objective_specs
            }
            
            results.append(result)
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            logger.error(f"Error evaluating {config.experiment_id}: {e}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'grid_search_results.csv')
    df.to_csv(results_path, index=False)
    
    logger.info(f"Grid search complete. Results saved to {results_path}")
    
    return results


def run_moo_experiment(
    experiment_config: Dict[str, Any],
    mode: str = 'full'
) -> Dict[str, Any]:
    """Run multi-objective optimization experiment.
    
    Parameters
    ----------
    experiment_config : Dict[str, Any]
        Experiment configuration
    mode : str
        'full' for complete grid, 'quick' for subset
    
    Returns
    -------
    Dict[str, Any]
        Experiment results
    """
    from config.config_manager import ConfigManager
    
    # Create config manager
    config_manager = ConfigManager(experiment_config.get('config_path', 'config.yaml'))
    load_plugins_from_config(config_manager.config)
    
    # Get configurations
    configs = filter_supported_configs(config_manager.get_experiment_configs(mode=mode))
    
    # Get MOO configuration
    moo_config = experiment_config.get('multi_objective', {})
    objective_specs = resolve_objective_specs(moo_config.get('objectives'))
    optimization_method = moo_config.get('optimization_method', 'grid_search')
    
    output_dirs = experiment_config.get('output_dirs', {})
    opt_output_dir = output_dirs.get('optimization', 'outputs/optimization')
    quick_mode = mode == 'quick'
    inference_runs = moo_config.get('quick_inference_runs', 5) if quick_mode else moo_config.get('inference_runs', 50)
    warmup_runs = moo_config.get('quick_warmup_runs', 2) if quick_mode else moo_config.get('warmup_runs', 10)
    
    if optimization_method == 'NSGA-II' and PYMOO_AVAILABLE:
        results = run_nsga2_optimization(
            config_space=configs,
            dataset=experiment_config,
            objective_specs=objective_specs,
            population_size=moo_config.get('population_size', 50),
            n_generations=moo_config.get('generations', 100),
            output_dir=opt_output_dir,
            inference_runs=inference_runs,
            warmup_runs=warmup_runs,
        )
    else:
        # Default to grid search
        max_configs = moo_config.get('quick_max_configs', 4) if quick_mode else None
        epoch_override = moo_config.get('evaluation_epochs')
        if epoch_override is None and quick_mode:
            epoch_override = 3
        results = grid_search_optimization(
            config_space=configs,
            dataset=experiment_config,
            output_dir=opt_output_dir,
            max_configs=max_configs,
            epoch_override=epoch_override,
            inference_runs=inference_runs,
            warmup_runs=warmup_runs,
            objective_specs=objective_specs,
        )
    
    return results
