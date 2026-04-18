"""Visualization utilities for training and evaluation.

This module provides comprehensive visualization functions for:
- Training curves (loss, metrics over epochs)
- Prediction comparisons
- Confusion matrices
- Metric distributions
- Qualitative results
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> Optional[str]:
    """Plot training history curves.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history dictionary from Keras
    output_path : str
        Path to save the plot
    metrics : List[str], optional
        Specific metrics to plot. If None, plots all
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        DPI for saving
    
    Returns
    -------
    Optional[str]
        Path to saved plot
    """
    # Filter metrics
    if metrics is None:
        metrics = [k for k in history.keys() if not k.startswith('val_')]
    
    # Group metrics for subplots
    n_metrics = len(metrics)
    if n_metrics == 0:
        logger.warning("No metrics to plot")
        return None
    
    # Create subplots
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Training metric
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
        
        # Validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Training history plot saved to {output_path}")
    return output_path


def plot_loss_curves(
    history: Dict[str, List[float]],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> str:
    """Plot loss curves (training and validation).
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history
    output_path : str
        Path to save plot
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        DPI for saving
    
    Returns
    -------
    str
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    # Training loss
    if 'loss' in history:
        ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    
    # Validation loss
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_prediction_grid(
    images: List[np.ndarray],
    masks_true: List[np.ndarray],
    masks_pred: List[np.ndarray],
    output_path: str,
    titles: List[str] = None,
    n_cols: int = 4,
    figsize: Tuple[int, int] = None,
    dpi: int = 300
) -> str:
    """Plot grid of input images, ground truth, and predictions.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    masks_true : List[np.ndarray]
        List of ground truth masks
    masks_pred : List[np.ndarray]
        List of predicted masks
    output_path : str
        Path to save plot
    titles : List[str], optional
        Titles for each sample
    n_cols : int
        Number of columns in grid
    figsize : Tuple[int, int], optional
        Figure size (auto-calculated if None)
    dpi : int
        DPI for saving
    
    Returns
    -------
    str
        Path to saved plot
    """
    n_samples = len(images)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Each sample has 3 columns (input, ground truth, prediction)
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        
        # Input image
        ax_input = axes[row * 3, col]
        ax_input.imshow(np.clip(images[i], 0, 1))
        ax_input.set_title('Input' if row == 0 else '', fontsize=10)
        ax_input.axis('off')
        
        # Ground truth
        ax_true = axes[row * 3 + 1, col]
        ax_true.imshow(masks_true[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_true.set_title('Ground Truth' if row == 0 else '', fontsize=10)
        ax_true.axis('off')
        
        # Prediction
        ax_pred = axes[row * 3 + 2, col]
        ax_pred.imshow(masks_pred[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_pred.set_title('Prediction' if row == 0 else '', fontsize=10)
        ax_pred.axis('off')
        
        # Sample title
        if titles and i < len(titles):
            fig.text(0.5, 0.98 - row * (0.95 / n_rows), titles[i], 
                    ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Prediction grid saved to {output_path}")
    return output_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    class_names: List[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300
) -> str:
    """Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    output_path : str
        Path to save plot
    class_names : List[str], optional
        Class names for labels
    normalize : bool
        Whether to normalize the confusion matrix
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        DPI for saving
    
    Returns
    -------
    str
        Path to saved plot
    """
    if class_names is None:
        class_names = ['Background', 'Building']
    
    # Binarize predictions
    y_true_flat = (y_true > 0.5).astype(int).flatten()
    y_pred_flat = (y_pred > 0.5).astype(int).flatten()
    
    # Compute confusion matrix
    cm = sklearn_confusion_matrix(y_true_flat, y_pred_flat)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix{title_suffix}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_training_report(
    summary: Dict[str, Any],
    output_dir: str,
    include_plots: bool = True
) -> Dict[str, str]:
    """Create comprehensive training report with plots.
    
    Parameters
    ----------
    summary : Dict[str, Any]
        Training summary dictionary
    output_dir : str
        Directory to save report
    include_plots : bool
        Whether to generate plots
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping report component names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_files = {}
    
    # Training history plots
    if 'history' in summary and include_plots:
        history = summary['history']
        
        # Loss curves
        loss_plot_path = os.path.join(output_dir, 'loss_curves.png')
        plot_loss_curves(history, loss_plot_path)
        report_files['loss_curves'] = loss_plot_path
        
        # All metrics
        metrics_plot_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(history, metrics_plot_path)
        report_files['training_history'] = metrics_plot_path
    
    # Save summary as formatted text
    report_text_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_text_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('TRAINING REPORT\n')
        f.write('=' * 80 + '\n\n')
        
        f.write(f"Experiment ID: {summary.get('experiment_id', 'N/A')}\n")
        f.write(f"Architecture: {summary.get('architecture', 'N/A')}\n")
        f.write(f"Encoder: {summary.get('encoder', 'N/A')}\n")
        f.write(f"Epochs Trained: {summary.get('epochs_trained', 'N/A')}\n")
        f.write(f"Training Time: {summary.get('train_time_minutes', 0):.2f} minutes\n\n")
        
        f.write('-' * 80 + '\n')
        f.write('FINAL METRICS\n')
        f.write('-' * 80 + '\n')
        f.write(f"Final Train Loss: {summary.get('final_loss', 'N/A')}\n")
        f.write(f"Final Val Loss: {summary.get('final_val_loss', 'N/A')}\n")
        f.write(f"Final Val IoU: {summary.get('final_val_iou', 0):.4f}\n\n")
        
        if 'model_info' in summary:
            f.write('-' * 80 + '\n')
            f.write('MODEL INFO\n')
            f.write('-' * 80 + '\n')
            info = summary['model_info']
            f.write(f"Total Parameters: {info.get('total_params', 'N/A'):,}\n")
            f.write(f"Trainable Parameters: {info.get('trainable_params', 'N/A'):,}\n")
            f.write(f"Layers: {info.get('layers', 'N/A')}\n\n")
    
    report_files['report_text'] = report_text_path
    
    return report_files