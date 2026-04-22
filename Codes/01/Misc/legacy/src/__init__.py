"""Core framework modules for multi-objective optimization.

This package contains the core modules for multi-objective optimization,
evaluation, and Pareto front analysis.
"""

from .optimize import (
    run_moo_experiment,
    run_nsga2_optimization,
    grid_search_optimization,
    ObjectiveVector,
    SegmentationProblem,
)

from .pareto import (
    extract_pareto_front,
    plot_pareto_front,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    generate_pareto_report,
)

from .evaluate import (
    evaluate_model,
    evaluate_segmentation_metrics,
    evaluate_model_complexity,
    evaluate_inference_time,
    generate_comparison_table,
    generate_latex_table,
    plot_metric_distributions,
)
from .objectives import (
    ObjectiveSpec,
    get_objective_spec,
    list_objectives,
    register_objective,
    resolve_objective_specs,
)

__all__ = [
    # Optimization
    'run_moo_experiment',
    'run_nsga2_optimization',
    'grid_search_optimization',
    'ObjectiveVector',
    'SegmentationProblem',
    # Pareto
    'extract_pareto_front',
    'plot_pareto_front',
    'plot_pareto_front_2d',
    'plot_pareto_front_3d',
    'generate_pareto_report',
    # Evaluation
    'evaluate_model',
    'evaluate_segmentation_metrics',
    'evaluate_model_complexity',
    'evaluate_inference_time',
    'generate_comparison_table',
    'generate_latex_table',
    'plot_metric_distributions',
    # Objective registry
    'ObjectiveSpec',
    'register_objective',
    'get_objective_spec',
    'resolve_objective_specs',
    'list_objectives',
]
