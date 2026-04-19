"""Pareto front extraction and visualization.

This module provides tools for extracting Pareto-optimal solutions from
multi-objective optimization results and generating publication-quality
visualizations.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.patches import Patch
import seaborn as sns
from .objectives import resolve_objective_specs

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def is_pareto_dominated(
    point: np.ndarray,
    others: np.ndarray,
    minimize: List[bool] = None
) -> bool:
    """Check if a point is Pareto dominated by any other point.
    
    Parameters
    ----------
    point : np.ndarray
        The point to check (shape: [n_objectives])
    others : np.ndarray
        Other points to compare against (shape: [n_points, n_objectives])
    minimize : List[bool] or None
        For each objective, True if minimizing, False if maximizing.
        If None, assumes all objectives are minimized.
    
    Returns
    -------
    bool
        True if point is dominated by at least one other point
    """
    if minimize is None:
        minimize = [True] * len(point)
    
    # Adjust for maximization objectives
    point_adj = point.copy()
    others_adj = others.copy()
    for i, min_obj in enumerate(minimize):
        if not min_obj:
            point_adj[i] = -point[i]
            others_adj[:, i] = -others[:, i]
    
    # Check if any point dominates
    for other in others_adj:
        if np.all(other <= point_adj) and np.any(other < point_adj):
            return True
    
    return False


def extract_pareto_front(
    results: List[Dict[str, Any]],
    objectives: List[str] = None,
    minimize: List[bool] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Extract Pareto-optimal solutions from results.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of experiment results
    objectives : List[str] or None
        Names of objectives to consider. If None, uses all available.
    minimize : List[bool] or None
        For each objective, True if minimizing, False if maximizing.
        If None, assumes accuracy metrics should be maximized.
    
    Returns
    -------
    Tuple[List[Dict[str, Any]], np.ndarray]
        Pareto-optimal results and their objective values
    """
    if not results:
        logger.warning("No results provided for Pareto extraction")
        return [], np.array([])
    
    # Determine objectives
    if objectives is None:
        objectives = ['iou', 'f1_score', 'param_count', 'flops', 'inference_time']
        objectives = [obj for obj in objectives if obj in results[0] or any(obj == key for key in results[0].get('objectives', {}))]
    
    if minimize is None:
        minimize = [spec.minimize for spec in resolve_objective_specs(objectives)]
    
    logger.info(f"Extracting Pareto front for objectives: {objectives}")
    
    # Extract objective values
    obj_matrix = []
    valid_results = []
    
    for result in results:
        try:
            vec = []
            for obj in objectives:
                val = result.get(obj, result.get('objectives', {}).get(obj, 0))
                # For maximization objectives, negate
                if not minimize[objectives.index(obj)]:
                    val = -val
                vec.append(val)
            obj_matrix.append(vec)
            valid_results.append(result)
        except Exception as e:
            logger.warning(f"Skipping result due to error: {e}")
    
    if not obj_matrix:
        logger.warning("No valid results for Pareto extraction")
        return [], np.array([])
    
    obj_matrix = np.array(obj_matrix)
    
    # Find non-dominated solutions
    pareto_indices = []
    for i in range(len(obj_matrix)):
        others = np.delete(obj_matrix, i, axis=0)
        if not is_pareto_dominated(obj_matrix[i], others, minimize=[True]*len(objectives)):
            pareto_indices.append(i)
    
    pareto_results = [valid_results[i] for i in pareto_indices]
    pareto_values = obj_matrix[pareto_indices]
    
    # Restore original values (undo negation)
    for i, obj in enumerate(objectives):
        if not minimize[i]:
            pareto_values[:, i] = -pareto_values[:, i]
    
    logger.info(f"Extracted {len(pareto_results)} Pareto-optimal solutions from {len(results)} total")
    
    return pareto_results, pareto_values


def plot_pareto_front_2d(
    results: List[Dict[str, Any]],
    obj1: str,
    obj2: str,
    minimize1: bool = True,
    minimize2: bool = True,
    color_by: str = None,
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = None
) -> plt.Figure:
    """Create 2D Pareto front plot.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    obj1 : str
        First objective (x-axis)
    obj2 : str
        Second objective (y-axis)
    minimize1 : bool
        Whether first objective should be minimized
    minimize2 : bool
        Whether second objective should be minimized
    color_by : str or None
        Attribute to color points by
    output_path : str or None
        Path to save figure
    figsize : tuple
        Figure size
    title : str or None
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract values
    x_vals = [r.get(obj1, 0) for r in results]
    y_vals = [r.get(obj2, 0) for r in results]
    
    # Extract Pareto front
    pareto_results, pareto_vals = extract_pareto_front(
        results, [obj1, obj2], [minimize1, minimize2]
    )
    
    # Plot all points
    if color_by and color_by in results[0]:
        colors = [r.get(color_by, 0) for r in results]
        if all(isinstance(value, (int, float, np.number)) for value in colors):
            scatter = ax.scatter(x_vals, y_vals, c=colors, alpha=0.5, s=50, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            categories = {value: idx for idx, value in enumerate(sorted({str(value) for value in colors}))}
            encoded = [categories[str(value)] for value in colors]
            scatter = ax.scatter(x_vals, y_vals, c=encoded, alpha=0.5, s=50, cmap='tab10')
            handles = [
                Patch(color=scatter.cmap(scatter.norm(idx)), label=category)
                for category, idx in categories.items()
            ]
            ax.legend(handles=handles, title=color_by, loc='best', fontsize=10)
    else:
        ax.scatter(x_vals, y_vals, alpha=0.5, s=50, label='All solutions', color='lightgray')
    
    # Plot Pareto front
    if len(pareto_results) > 0:
        pareto_x = pareto_vals[:, 0]
        pareto_y = pareto_vals[:, 1]
        
        # Sort for line plot
        sort_idx = np.argsort(pareto_x)
        if not minimize1:
            sort_idx = sort_idx[::-1]
        
        ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r-', linewidth=2, 
                label='Pareto front', marker='o', markersize=6)
        ax.scatter(pareto_x, pareto_y, s=100, c='red', marker='*', 
                  edgecolors='black', linewidths=1, label='Pareto-optimal', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel(obj1.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(obj2.replace('_', ' ').title(), fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    if not (color_by and color_by in results[0] and not all(isinstance(value, (int, float, np.number)) for value in [r.get(color_by, 0) for r in results])):
        ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Pareto front plot to {output_path}")
    
    return fig


def plot_pareto_front_3d(
    results: List[Dict[str, Any]],
    obj1: str,
    obj2: str,
    obj3: str,
    minimize: List[bool] = None,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 10),
    title: str = None
) -> plt.Figure:
    """Create 3D Pareto front plot.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    obj1 : str
        First objective (x-axis)
    obj2 : str
        Second objective (y-axis)
    obj3 : str
        Third objective (z-axis)
    minimize : List[bool] or None
        Whether each objective should be minimized
    output_path : str or None
        Path to save figure
    figsize : tuple
        Figure size
    title : str or None
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract Pareto front
    objectives = [obj1, obj2, obj3]
    if minimize is None:
        minimize = [obj not in ['iou', 'f1_score', 'precision', 'recall', 'accuracy'] 
                   for obj in objectives]
    
    pareto_results, pareto_vals = extract_pareto_front(results, objectives, minimize)
    
    # Plot all points
    x_vals = [r.get(obj1, 0) for r in results]
    y_vals = [r.get(obj2, 0) for r in results]
    z_vals = [r.get(obj3, 0) for r in results]
    
    ax.scatter(x_vals, y_vals, z_vals, alpha=0.3, s=30, label='All solutions', color='lightgray')
    
    # Plot Pareto front
    if len(pareto_results) > 0:
        ax.scatter(pareto_vals[:, 0], pareto_vals[:, 1], pareto_vals[:, 2],
                  s=100, c='red', marker='*', edgecolors='black', linewidths=1,
                  label='Pareto-optimal', zorder=5)
    
    # Labels
    ax.set_xlabel(obj1.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(obj2.replace('_', ' ').title(), fontsize=11)
    ax.set_zlabel(obj3.replace('_', ' ').title(), fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 3D Pareto front plot to {output_path}")
    
    return fig


def plot_pareto_front(
    results: List[Dict[str, Any]],
    output_dir: str,
    objectives: List[str] = None
) -> Dict[str, str]:
    """Generate all Pareto front visualizations.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    output_dir : str
        Directory to save plots
    objectives : List[str] or None
        Objectives to plot. If None, uses defaults.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if objectives is None:
        if results and results[0].get('objectives'):
            objectives = list(results[0]['objectives'].keys())
        else:
            objectives = ['iou', 'f1_score', 'param_count', 'flops', 'inference_time']
            objectives = [obj for obj in objectives if any(obj in r for r in results)]
    
    plot_paths = {}
    
    # Generate 2D plots for key pairs
    key_pairs = [
        ('iou', 'param_count', False, True),
        ('iou', 'inference_time', False, True),
        ('f1_score', 'param_count', False, True),
        ('f1_score', 'inference_time', False, True),
        ('param_count', 'inference_time', True, True),
    ]
    
    for obj1, obj2, min1, min2 in key_pairs:
        if obj1 in objectives and obj2 in objectives:
            path = os.path.join(output_dir, f'pareto_{obj1}_vs_{obj2}.png')
            plot_pareto_front_2d(
                results, obj1, obj2, min1, min2,
                color_by='architecture',
                output_path=path,
                title=f'Pareto Front: {obj1.replace("_", " ").title()} vs {obj2.replace("_", " ").title()}'
            )
            plot_paths[f'{obj1}_vs_{obj2}'] = path
    
    # Generate 3D plot if we have enough objectives
    if len(objectives) >= 3:
        path = os.path.join(output_dir, 'pareto_3d.png')
        plot_pareto_front_3d(
            results, objectives[0], objectives[1], objectives[2],
            output_path=path,
            title='3D Pareto Front'
        )
        plot_paths['3d'] = path
    
    logger.info(f"Generated {len(plot_paths)} Pareto front plots")
    
    return plot_paths


def plot_objective_space(
    results: List[Dict[str, Any]],
    output_path: str,
    objectives: List[str] = None
) -> plt.Figure:
    """Create parallel coordinates plot of objective space.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Experiment results
    output_path : str
        Path to save figure
    objectives : List[str] or None
        Objectives to include
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if objectives is None:
        objectives = ['iou', 'f1_score', 'param_count', 'flops', 'inference_time']
        objectives = [obj for obj in objectives if any(obj in r for r in results)]
    
    # Normalize values for visualization
    data = []
    for r in results:
        row = []
        for obj in objectives:
            val = r.get(obj, 0)
            # Normalize to [0, 1]
            vals = [res.get(obj, 0) for res in results]
            min_val, max_val = min(vals), max(vals)
            if max_val > min_val:
                norm_val = (val - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            row.append(norm_val)
        data.append(row)
    
    data = np.array(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each solution
    for i, row in enumerate(data):
        color = 'blue' if i < len(data) * 0.1 else 'gray'  # Highlight top 10%
        alpha = 0.8 if i < len(data) * 0.1 else 0.2
        ax.plot(range(len(objectives)), row, color=color, alpha=alpha, linewidth=1)
    
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels([obj.replace('_', ' ').title() for obj in objectives], rotation=45, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Objective Space Visualization (Parallel Coordinates)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved objective space plot to {output_path}")
    
    return fig


def plot_hypervolume_history(
    history: List[float],
    output_path: str
) -> plt.Figure:
    """Plot hypervolume convergence over generations.
    
    Parameters
    ----------
    history : List[float]
        Hypervolume values over generations
    output_path : str
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(1, len(history) + 1)
    ax.plot(generations, history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(generations, history, alpha=0.3)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.set_title('Hypervolume Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved hypervolume history plot to {output_path}")
    
    return fig


def generate_pareto_report(
    pareto_results: List[Dict[str, Any]],
    output_path: str
) -> None:
    """Generate a text report of Pareto-optimal solutions.
    
    Parameters
    ----------
    pareto_results : List[Dict[str, Any]]
        Pareto-optimal results
    output_path : str
        Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARETO-OPTIMAL SOLUTIONS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Number of Pareto-optimal solutions: {len(pareto_results)}\n\n")
        
        for i, result in enumerate(pareto_results, 1):
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Solution {i}: {result.get('experiment_id', 'N/A')}\n")
            f.write(f"{'-' * 80}\n")
            
            # Configuration
            f.write("\nConfiguration:\n")
            config_keys = ['architecture', 'encoder', 'loss_function', 'optimizer',
                          'learning_rate', 'batch_size', 'epochs']
            for key in config_keys:
                if key in result:
                    f.write(f"  {key}: {result[key]}\n")
            
            # Objectives
            f.write("\nObjectives:\n")
            obj_keys = ['iou', 'f1_score', 'precision', 'recall', 'accuracy',
                       'param_count', 'flops', 'inference_time']
            for key in obj_keys:
                if key in result:
                    val = result[key]
                    if key in ['param_count']:
                        f.write(f"  {key}: {val/1e6:.2f}M\n")
                    elif key in ['flops']:
                        f.write(f"  {key}: {val/1e9:.2f}G\n")
                    elif key in ['inference_time']:
                        f.write(f"  {key}: {val:.2f} ms\n")
                    else:
                        f.write(f"  {key}: {val:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Saved Pareto report to {output_path}")
