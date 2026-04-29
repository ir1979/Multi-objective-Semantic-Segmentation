"""Configuration management for multi-objective optimization experiments.

This module provides comprehensive configuration management with support for
YAML-based experiment definitions, hyperparameter grids, and experiment tracking.
"""

import os
import yaml
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import itertools


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.
    
    Attributes
    ----------
    experiment_id : str
        Unique identifier for this experiment
    architecture : str
        Model architecture name (e.g., 'Unet', 'UnetPlusPlus')
    encoder : str
        Encoder backbone name (e.g., 'resnet34', 'efficientnetb0')
    encoder_weights : str
        Pretrained weights for encoder ('imagenet' or None)
    loss_function : str
        Loss function name
    optimizer : str
        Optimizer name ('Adam', 'SGD')
    learning_rate : float
        Learning rate for training
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    seed : int
        Random seed for reproducibility
    tile_size : int
        Input image tile size
    augmentation : List[str]
        List of augmentation techniques
    """
    experiment_id: str = ""
    architecture: str = "Unet"
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    loss_function: str = "binary_crossentropy"
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 50
    seed: int = 42
    tile_size: int = 256
    augmentation: List[str] = field(default_factory=lambda: ["flip", "rotate"])
    scheduler: str = "ReduceLROnPlateau"
    
    def __post_init__(self):
        """Generate experiment ID if not provided."""
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.seed = int(self.seed)
        self.tile_size = int(self.tile_size)
        if not self.experiment_id:
            self.experiment_id = self.generate_id()
    
    def generate_id(self) -> str:
        """Generate a unique experiment ID based on configuration."""
        config_str = f"{self.architecture}_{self.encoder}_{self.loss_function}_{self.optimizer}_" \
                    f"lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}_seed{self.seed}"
        hash_obj = hashlib.md5(config_str.encode())
        short_hash = hash_obj.hexdigest()[:8]
        return f"{config_str}_{short_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConfigManager:
    """Manager for experiment configurations and hyperparameter grids.
    
    This class handles loading, validation, and expansion of configuration
    files for multi-objective optimization experiments.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    
    Attributes
    ----------
    config : Dict[str, Any]
        Loaded configuration dictionary
    experiment_configs : List[ExperimentConfig]
        List of all experiment configurations from grid expansion
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.experiment_configs: List[ExperimentConfig] = []
        self._expand_grid()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        
        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
               ValueError
            If config file is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a dictionary")
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        required_sections = ['experiment', 'dataset', 'models', 'training', 'multi_objective']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: '{section}'")
        
        # Validate models section
        models = config.get('models', {})
        if 'architectures' not in models or not models['architectures']:
            raise ValueError("Configuration must specify at least one model architecture")
        if 'encoders' not in models or not models['encoders']:
            raise ValueError("Configuration must specify at least one encoder")
        
        # Validate training section
        training = config.get('training', {})
        required_training = ['loss_functions', 'optimizers', 'learning_rates', 'batch_sizes', 'epochs']
        for key in required_training:
            if key not in training or not training[key]:
                raise ValueError(f"Training configuration must specify '{key}'")
    
    def _expand_grid(self) -> None:
        """Expand hyperparameter grid into individual experiment configurations."""
        models = self.config.get('models', {})
        training = self.config.get('training', {})
        dataset = self.config.get('dataset', {})
        experiment = self.config.get('experiment', {})
        
        # Get all hyperparameter lists
        architectures = models.get('architectures', ['Unet'])
        encoders = models.get('encoders', ['resnet34'])
        encoder_weights_list = models.get('encoder_weights', ['imagenet'])
        loss_functions = training.get('loss_functions', ['binary_crossentropy'])
        optimizers = training.get('optimizers', ['Adam'])
        learning_rates = training.get('learning_rates', [1e-3])
        batch_sizes = training.get('batch_sizes', [8])
        epochs_list = training.get('epochs', [50])
        schedulers = training.get('schedulers', ['ReduceLROnPlateau'])
        seeds = experiment.get('seed', [42])
        tile_sizes = dataset.get('tile_size', [256])
        augmentations = dataset.get('augmentations', [['flip', 'rotate']])
        
        # Generate all combinations
        grid = itertools.product(
            architectures, encoders, encoder_weights_list,
            loss_functions, optimizers, learning_rates,
            batch_sizes, epochs_list, schedulers, seeds, tile_sizes, augmentations
        )
        
        for combo in grid:
            (arch, enc, enc_w, loss, opt, lr, bs, ep, sched, seed, tile, aug) = combo
            
            exp_config = ExperimentConfig(
                architecture=arch,
                encoder=enc,
                encoder_weights=enc_w,
                loss_function=loss,
                optimizer=opt,
                learning_rate=lr,
                batch_size=bs,
                epochs=ep,
                scheduler=sched,
                seed=seed,
                tile_size=tile,
                augmentation=list(aug) if isinstance(aug, (list, tuple)) else [aug]
            )
            self.experiment_configs.append(exp_config)
    
    def get_experiment_configs(self, mode: str = 'full') -> List[ExperimentConfig]:
        """Get experiment configurations based on mode.
        
        Parameters
        ----------
        mode : str
            'full' for all configurations, 'quick' for subset
        
        Returns
        -------
        List[ExperimentConfig]
            List of experiment configurations
        """
        if mode == 'quick':
            # Return a representative subset
            subset_size = min(10, len(self.experiment_configs))
            indices = np.linspace(0, len(self.experiment_configs) - 1, subset_size, dtype=int)
            return [self.experiment_configs[i] for i in indices]
        return self.experiment_configs
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration section."""
        return self.config.get('dataset', {})
    
    def get_moo_config(self) -> Dict[str, Any]:
        """Get multi-objective optimization configuration section."""
        return self.config.get('multi_objective', {})
    
    def get_experiment_name(self) -> str:
        """Get experiment name from configuration."""
        return self.config.get('experiment', {}).get('name', 'building_footprint_moo')
    
    def save_results_table(self, results: List[Dict[str, Any]], output_dir: str) -> 'pd.DataFrame':
        """Save results to CSV and return DataFrame.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            List of result dictionaries
        output_dir : str
            Directory to save the CSV file
        
        Returns
        -------
        pd.DataFrame
            Results DataFrame
        """
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(results)
        
        csv_path = os.path.join(output_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        
        return df
    
    def save_latex_table(self, df: 'pd.DataFrame', output_path: str) -> None:
        """Save results as LaTeX table.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        output_path : str
            Path to save LaTeX file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Select and format columns for publication
        pub_columns = ['experiment_id', 'architecture', 'encoder', 'loss_function',
                       'learning_rate', 'batch_size', 'iou', 'f1_score', 
                       'param_count', 'flops', 'inference_time']
        
        pub_df = df[[col for col in pub_columns if col in df.columns]].copy()
        
        # Format numeric columns
        numeric_cols = ['learning_rate', 'iou', 'f1_score']
        for col in numeric_cols:
            if col in pub_df.columns:
                pub_df[col] = pub_df[col].apply(lambda x: f"{x:.4f}")
        
        # Format large numbers
        if 'param_count' in pub_df.columns:
            pub_df['param_count'] = pub_df['param_count'].apply(
                lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K"
            )
        if 'flops' in pub_df.columns:
            pub_df['flops'] = pub_df['flops'].apply(
                lambda x: f"{x/1e9:.2f}G" if x >= 1e9 else f"{x/1e6:.2f}M"
            )
        
        # Generate LaTeX
        latex_str = pub_df.to_latex(index=False, escape=False, 
                                     caption='Multi-Objective Optimization Results',
                                     label='tab:moo_results')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file.
        
        Parameters
        ----------
        output_path : str
            Path to save configuration
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """
experiment:
  name: "building_footprint_moo"
  seed: [42, 123, 456]
  description: "Multi-objective optimization for building footprint extraction"

dataset:
  name: "Inria Aerial"
  tile_size: [256, 512]
  augmentations:
    - [flip, rotate]
    - [flip, rotate, brightness]
    - [flip, rotate, brightness, elastic_transform]
  train_val_test_split: [0.7, 0.15, 0.15]
  rgb_path: "Datasets/RGB"
  mask_path: "Datasets/Mask"

models:
  architectures: [Unet, UnetPlusPlus, FPN, Linknet, DeepLabV3Plus]
  encoders: [resnet34, resnet50, efficientnetb0, efficientnetb3, vgg16]
  encoder_weights: [imagenet]

training:
  loss_functions: [binary_crossentropy, dice_loss, focal_loss, combo_loss]
  optimizers: [Adam, SGD]
  learning_rates: [1e-3, 1e-4, 5e-4]
  batch_sizes: [8, 16]
  epochs: [50, 100]
  schedulers: [ReduceLROnPlateau, CosineAnnealing]

multi_objective:
  objectives: [iou, f1_score, param_count, flops, inference_time]
  optimization_method: "NSGA-II"
  population_size: 50
  generations: 100
  pareto_front: true
  reference_point: [0.0, 0.0, 1e9, 1e12, 1000]  # For hypervolume calculation
"""


def create_default_config(output_path: str) -> None:
    """Create a default configuration file.
    
    Parameters
    ----------
    output_path : str
        Path to save the default configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)
