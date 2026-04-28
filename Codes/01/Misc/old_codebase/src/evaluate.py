"""Comprehensive evaluation module for segmentation models.

This module provides evaluation functions for computing all metrics,
generating visualizations, and saving results in publication-ready formats.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from training.metrics import (
    iou_score, dice_score, precision_score, recall_score,
    pixel_accuracy, boundary_iou, boundary_f1
)
from models.model_factory import get_model_info, estimate_flops, measure_inference_time
from src.objectives import list_objectives

logger = logging.getLogger(__name__)


def evaluate_model(
    model: keras.Model,
    dataset: Any,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Comprehensive model evaluation.
    
    Parameters
    ----------
    model : keras.Model
        Model to evaluate
    dataset : Any
        Evaluation dataset
    config : Dict[str, Any] or None
        Configuration dictionary
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    results = {}
    
    # 1. Segmentation metrics
    seg_metrics = evaluate_segmentation_metrics(model, dataset)
    results.update(seg_metrics)
    
    # 2. Model complexity metrics
    complexity_metrics = evaluate_model_complexity(model)
    results.update(complexity_metrics)
    
    # 3. Inference time metrics
    inference_metrics = evaluate_inference_time(model, config)
    results.update(inference_metrics)
    
    logger.info("Evaluation complete.")
    logger.info(f"IoU: {results.get('iou', 0):.4f}, F1: {results.get('f1_score', 0):.4f}")
    logger.info(f"Parameters: {results.get('param_count', 0):,}, FLOPs: {results.get('flops', 0):,}")
    
    return results


def evaluate_segmentation_metrics(
    model: keras.Model,
    dataset: Any
) -> Dict[str, float]:
    """Evaluate segmentation quality metrics.
    
    Parameters
    ----------
    model : keras.Model
        Model to evaluate
    dataset : Any
        Dataset with images and masks
    
    Returns
    -------
    Dict[str, float]
        Dictionary of segmentation metrics
    """
    metrics = {
        'iou': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'boundary_iou': [],
        'boundary_f1': [],
    }
    
    for batch_idx in range(len(dataset)):
        x_batch, y_batch = dataset[batch_idx]
        y_pred = model.predict(x_batch, verbose=0)
        
        for i in range(len(x_batch)):
            y_true = y_batch[i:i+1]
            y_pred_single = y_pred[i:i+1]
            
            # Compute metrics
            metrics['iou'].append(float(iou_score(y_true, y_pred_single)))
            metrics['f1_score'].append(float(dice_score(y_true, y_pred_single)))
            metrics['precision'].append(float(precision_score(y_true, y_pred_single)))
            metrics['recall'].append(float(recall_score(y_true, y_pred_single)))
            metrics['accuracy'].append(float(pixel_accuracy(y_true, y_pred_single)))
            metrics['boundary_iou'].append(float(boundary_iou(y_true, y_pred_single)))
            metrics['boundary_f1'].append(float(boundary_f1(y_true, y_pred_single)))
    
    # Aggregate metrics
    results = {}
    for key, values in metrics.items():
        results[key] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))
    
    return results


def evaluate_model_complexity(model: keras.Model) -> Dict[str, Any]:
    """Evaluate model complexity metrics.
    
    Parameters
    ----------
    model : keras.Model
        Model to analyze
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with complexity metrics
    """
    info = get_model_info(model)
    flops = estimate_flops(model)
    
    return {
        'param_count': info['total_params'],
        'trainable_params': info['trainable_params'],
        'non_trainable_params': info['non_trainable_params'],
        'flops': flops,
        'model_layers': info['layers'],
    }


def evaluate_inference_time(
    model: keras.Model,
    config: Dict[str, Any] = None
) -> Dict[str, float]:
    """Evaluate inference time metrics.
    
    Parameters
    ----------
    model : keras.Model
        Model to benchmark
    config : Dict[str, Any] or None
        Configuration with input shape
    
    Returns
    -------
    Dict[str, float]
        Dictionary with timing metrics
    """
    # Determine input shape
    if config and 'tile_size' in config:
        tile_size = config['tile_size']
        input_shape = (1, tile_size, tile_size, 3)
    else:
        input_shape = (1, 256, 256, 3)
    
    timing = measure_inference_time(model, input_shape=input_shape, num_runs=100)
    
    return {
        'inference_time': timing['mean_ms'],
        'inference_time_std': timing['std_ms'],
        'inference_time_min': timing['min_ms'],
        'inference_time_max': timing['max_ms'],
    }


def generate_comparison_table(
    results: List[Dict[str, Any]],
    output_path: str,
    metrics: List[str] = None
) -> pd.DataFrame:
    """Generate comparison table of multiple experiments.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of experiment results
    output_path : str
        Path to save table
    metrics : List[str] or None
        Metrics to include
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame
    """
    if metrics is None:
        metrics = ['experiment_id', 'architecture', 'encoder'] + [
            metric for metric in ['iou', 'f1_score', 'precision', 'recall', 'boundary_iou', 'boundary_f1', 'param_count', 'flops', 'inference_time']
            if metric in list_objectives() or metric in {'experiment_id', 'architecture', 'encoder'}
        ]
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select relevant columns
    available_cols = [col for col in metrics if col in df.columns]
    df = df[available_cols]
    
    # Format for display
    if 'param_count' in df.columns:
        df['param_count_formatted'] = df['param_count'].apply(
            lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K"
        )
    
    if 'flops' in df.columns:
        df['flops_formatted'] = df['flops'].apply(
            lambda x: f"{x/1e9:.2f}G" if x >= 1e9 else f"{x/1e6:.2f}M"
        )
    
    if 'inference_time' in df.columns:
        df['inference_time_formatted'] = df['inference_time'].apply(
            lambda x: f"{x:.2f} ms"
        )
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved comparison table to {output_path}")
    
    return df


def generate_latex_table(
    results: List[Dict[str, Any]],
    output_path: str,
    top_k: int = 10,
    sort_by: str = 'iou'
) -> str:
    """Generate LaTeX table of top results.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    output_path : str
        Path to save LaTeX file
    top_k : int
        Number of top results to include
    sort_by : str
        Metric to sort by
    
    Returns
    -------
    str
        LaTeX table string
    """
    # Sort results
    sorted_results = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=True)[:top_k]
    
    # Create DataFrame
    df = pd.DataFrame(sorted_results)
    
    # Select and rename columns
    col_map = {
        'architecture': 'Architecture',
        'encoder': 'Encoder',
        'iou': 'IoU',
        'f1_score': 'F1',
        'precision': 'Precision',
        'recall': 'Recall',
        'param_count': 'Params',
        'flops': 'FLOPs',
        'inference_time': 'Time (ms)',
    }
    
    available_cols = [col for col in col_map.keys() if col in df.columns]
    df = df[available_cols].rename(columns=col_map)
    
    # Format values
    if 'Params' in df.columns:
        df['Params'] = df['Params'].apply(lambda x: f"{x/1e6:.2f}M")
    if 'FLOPs' in df.columns:
        df['FLOPs'] = df['FLOPs'].apply(lambda x: f"{x/1e9:.2f}G")
    if 'Time (ms)' in df.columns:
        df['Time (ms)'] = df['Time (ms)'].apply(lambda x: f"{x:.1f}")
    
    for col in ['IoU', 'F1', 'Precision', 'Recall']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False)
    
    # Add table environment
    latex_str = f"""\\\\begin{{table}}[htbp]
\\\\centering
\\\\caption{{Top {top_k} Model Configurations by {sort_by.replace('_', ' ').title()}}}
\\\\label{{tab:top_results}}
{latex}
\\\\end{{table}}
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    logger.info(f"Saved LaTeX table to {output_path}")
    
    return latex_str


def plot_metric_distributions(
    results: List[Dict[str, Any]],
    output_dir: str,
    metrics: List[str] = None,
    group_by: str = 'architecture'
) -> Dict[str, str]:
    """Plot distributions of metrics across experiments.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    output_dir : str
        Directory to save plots
    metrics : List[str] or None
        Metrics to plot
    group_by : str
        Attribute to group by
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping metric names to plot paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if metrics is None:
        metrics = ['iou', 'f1_score', 'param_count', 'inference_time']
    
    df = pd.DataFrame(results)
    
    plot_paths = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if group_by in df.columns:
            # Box plot grouped by attribute
            df.boxplot(column=metric, by=group_by, ax=ax)
            plt.suptitle('')  # Remove default title
        else:
            # Simple histogram
            ax.hist(df[metric], bins=20, edgecolor='black', alpha=0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'{metric}_distribution.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        plot_paths[metric] = path
    
    logger.info(f"Generated {len(plot_paths)} distribution plots")
    
    return plot_paths


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    class_names: List[str] = None
) -> plt.Figure:
    """Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    output_path : str
        Path to save figure
    class_names : List[str] or None
        Class names for labels
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Background', 'Building']
    
    # Compute confusion matrix
    y_true_flat = (y_true > 0.5).astype(int).flatten()
    y_pred_flat = (y_pred > 0.5).astype(int).flatten()
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_qualitative_predictions(
    model: keras.Model,
    dataset: Any,
    output_dir: str,
    num_samples: int = 10
) -> List[str]:
    """Save qualitative prediction visualizations.
    
    Parameters
    ----------
    model : keras.Model
        Model to use for predictions
    dataset : Any
        Dataset with images and masks
    output_dir : str
        Directory to save visualizations
    num_samples : int
        Number of samples to visualize
    
    Returns
    -------
    List[str]
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for batch_idx in range(min(num_samples, len(dataset))):
        x_batch, y_batch = dataset[batch_idx]
        y_pred = model.predict(x_batch, verbose=0)
        
        for i in range(min(4, len(x_batch))):  # Max 4 per batch
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input image
            axes[0].imshow(np.clip(x_batch[i], 0, 1))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(y_batch[i].squeeze(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(y_pred[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            path = os.path.join(output_dir, f'sample_{batch_idx}_{i}.png')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            saved_paths.append(path)
    
    logger.info(f"Saved {len(saved_paths)} qualitative prediction images")
    
    return saved_paths
