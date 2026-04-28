# Grid Search Framework for Multi-Objective Semantic Segmentation

## Overview

This grid search framework enables systematic hyperparameter exploration for the multi-objective building segmentation pipeline. It provides:

- **Efficient Grid Exploration**: Cartesian product, random sampling, or Latin hypercube sampling
- **Checkpointing & Resumability**: Save state after each point for fault tolerance
- **Comprehensive Logging**: Detailed logs at multiple levels with structured event tracking
- **Error Resilience**: Automatic retry with exponential backoff
- **Results Aggregation**: Automatic generation of tables, figures, and statistics
- **Paper-Ready Outputs**: CSV, LaTeX, and visualizations for publication

## Quick Start

### 1. Configure Grid Space

Create or edit a grid search config file (e.g., `configs/grid_search.yaml`):

```yaml
grid_search:
  enabled: true
  auto_checkpoint: true
  parameters:
    model_architecture: ["unet", "unetpp"]
    loss_strategy: ["single", "weighted", "mgda"]
    learning_rate: [1.0e-4, 5.0e-4, 1.0e-3]
    batch_size: [4, 8]
  
  selection:
    strategy: "random"  # or "full", "latin_hypercube"
    n_points: 50
    random_seed: 42
```

### 2. Run Grid Search

```bash
# Start fresh grid search
python run_grid_search.py --config configs/grid_search.yaml --force-restart

# Resume existing grid search
python run_grid_search.py --config configs/grid_search.yaml

# Only generate reports from existing results
python run_grid_search.py --generate-reports-only
```

### 3. Monitor Progress

Check the log files in `grid_search_results/`:

```bash
# Real-time log viewing
tail -f grid_search_results/grid_search.log

# View event log (JSON lines format)
cat grid_search_results/events.jsonl | jq '.'

# View metrics (JSON lines format)
cat grid_search_results/metrics.jsonl | jq '.'
```

## Project Structure

```
experiments/
├── grid_search.py              # Main grid search runner
├── results_aggregator.py       # Results analysis and visualization
├── registry.py                 # Experiment state tracking
└── ...

logging_utils/
├── grid_search_logger.py       # Specialized logging for grid search
├── logger.py                   # Base dual logger
└── ...

utils/
├── error_handling.py           # Error recovery and retry logic
├── config_loader.py            # Configuration management
└── ...

configs/
├── grid_search.yaml            # Grid search configuration template
├── default.yaml                # Default training config
└── ...

grid_search_results/            # Generated during execution
├── grid_search_state.json      # Current grid state (resumable)
├── grid_search.log             # Main log file
├── events.jsonl                # Event stream
├── metrics.jsonl               # Metrics stream
└── point_XXXXXX/               # Results per grid point
    ├── config.yaml             # Point-specific config
    ├── model.h5                # Trained model
    └── logs/
        ├── run.log
        └── metrics.csv
```

## Features

### Error Handling & Resilience

The framework includes comprehensive error handling:

- **Automatic Retries**: Exponential backoff retry for transient failures
- **Graceful Degradation**: Continue with next point on recoverable errors
- **Error Logging**: Detailed error logs with context and traceback
- **Recovery State**: Resume from last successful point

```python
from utils.error_handling import ErrorHandler, RecoveryStrategy

error_handler = ErrorHandler(logger)
recovery = RecoveryStrategy(logger, max_retries=3, backoff_factor=2.0)

# Use decorator for automatic retry
@recovery.retry_decorator(error_context="data_loading")
def load_data():
    ...
```

### Configuration Constraints

Apply constraints to reduce search space:

```yaml
grid_search:
  constraints:
    # Only use deep supervision with UNetPP
    - if_model_architecture: "unet"
      then_not_deep_supervision: true
    
    # Skip configurations with both losses disabled
    - if_boundary_loss_weight: 0.0
      and_shape_loss_weight: 0.0
      then_skip: true
```

### Selection Strategies

Choose how to sample from the grid:

- **`full`**: Generate all combinations (Cartesian product)
- **`random`**: Randomly sample N points uniformly
- **`latin_hypercube`**: Latin hypercube sampling for better coverage

```yaml
grid_search:
  selection:
    strategy: "random"
    n_points: 50
    random_seed: 42
```

### Checkpointing & Resumability

Each grid point is tracked in `grid_search_state.json`:

```json
{
  "0": {
    "point_id": 0,
    "params": {"model_architecture": "unet", "learning_rate": 0.0001},
    "status": "completed",
    "result_dir": "grid_search_results/point_000000",
    "metrics": {"val_iou": 0.845}
  },
  "1": {
    "point_id": 1,
    "status": "failed",
    "error_message": "Out of memory"
  }
}
```

Resume after failure:
```bash
# Continues from where it left off
python run_grid_search.py --config configs/grid_search.yaml
```

### Results Analysis

Automatic generation of analysis outputs:

```
grid_search_results/reports/
├── grid_search_results.csv           # Full results table
├── best_configurations.csv           # Top N configurations
├── grid_search_table.tex             # LaTeX table for paper
├── summary_statistics.json           # Per-category statistics
├── comparison_by_architecture.png    # Bar plots
├── comparison_by_strategy.png
├── distribution_results.png          # Distribution analysis
└── heatmap_results.png              # 2D parameter space heatmap
```

## Configuration Reference

### Grid Search Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | bool | Enable grid search mode |
| `auto_checkpoint` | bool | Save state after each point |
| `replicate_points` | int | Number of runs per point (for robustness) |
| `parameters` | dict | Hyperparameter space |
| `constraints` | list | Filtering constraints |
| `selection.strategy` | str | "full", "random", or "latin_hypercube" |
| `selection.n_points` | int | Number of points to sample (for random/LH) |

### Hyperparameter Space

All hyperparameters that can be grid searched:

- `model_architecture`: "unet" or "unetpp"
- `loss_strategy`: "single", "weighted", or "mgda"
- `pixel_loss_type`: "bce", "bce_iou", "dice", "focal"
- `boundary_loss_weight`: float (0.0 to 1.0)
- `shape_loss_weight`: float (0.0 to 1.0)
- `learning_rate`: float (typically 1e-5 to 1e-2)
- `encoder_filters`: list of ints
- `dropout_rate`: float (0.0 to 0.8)
- `deep_supervision`: bool (only valid for UNetPP)
- `batch_size`: int (4, 8, 16, 32)

## Advanced Usage

### Custom Metric Collection

Extend `GridSearchLogger` to collect custom metrics:

```python
from logging_utils.grid_search_logger import GridSearchLogger

class CustomLogger(GridSearchLogger):
    def log_custom_metric(self, name: str, value: float):
        self._append_metrics({"custom_" + name: value})
```

### Integration with Experiment Runner

The grid search framework integrates with the existing experiment runner:

```python
from experiments.experiment_runner import ExperimentRunner
from experiments.grid_search import GridPoint

# Create experiment runner
runner = ExperimentRunner(config)

# Run each grid point
for point in grid_points:
    config = runner.get_point_config(base_config, point)
    runner.run_experiment(config, point.result_dir)
```

## Performance Tips

1. **Reduce Logging Overhead**:
   ```yaml
   logging:
     tensorboard: false        # Disable TensorBoard
     validation_image_interval: 0
     log_gradient_norms: false
   ```

2. **Parallelize Points** (future feature):
   - Run multiple grid points in parallel
   - Use separate GPUs per point

3. **Smart Sampling**:
   - Use `random` sampling instead of `full` grid
   - Focus on promising regions iteratively

4. **Early Stopping**:
   ```yaml
   training:
     early_stopping:
       enabled: true
       patience: 3  # Stop after 3 epochs without improvement
   ```

## Troubleshooting

### Grid Search Hangs

Check the log file for stuck processes:
```bash
tail -100 grid_search_results/grid_search.log
```

### Memory Issues

Reduce batch size in grid search config:
```yaml
parameters:
  batch_size: [4]  # Use only 4 instead of [4, 8, 16]
```

### Resume Fails

Force restart the grid search:
```bash
python run_grid_search.py --force-restart
```

### Missing Results Files

Regenerate reports only:
```bash
python run_grid_search.py --generate-reports-only
```

## Examples

### Example 1: Quick Grid Search

Explore 50 random configurations in small parameter space:

```yaml
grid_search:
  parameters:
    model_architecture: ["unet", "unetpp"]
    learning_rate: [1.0e-4, 5.0e-4, 1.0e-3]
    batch_size: [4, 8]
  selection:
    strategy: "random"
    n_points: 50
```

### Example 2: Loss Strategy Comparison

Compare all combinations of loss strategies:

```yaml
grid_search:
  parameters:
    loss_strategy: ["single", "weighted", "mgda"]
    pixel_loss_type: ["bce", "bce_iou", "dice"]
    boundary_loss_weight: [0.0, 0.3, 0.5]
    shape_loss_weight: [0.0, 0.1, 0.2]
  selection:
    strategy: "full"  # All 108 combinations
```

### Example 3: Architecture Ablation

Deep study of UNet++ configurations:

```yaml
grid_search:
  parameters:
    model_architecture: ["unetpp"]
    encoder_filters:
      - [32, 64, 128, 256, 512]
      - [64, 128, 256, 512, 1024]
    dropout_rate: [0.2, 0.3, 0.5]
    deep_supervision: [false, true]
  selection:
    strategy: "random"
    n_points: 40
```

## Citation

If you use this grid search framework in your research, please cite:

```bibtex
@inproceedings{moo_segmentation,
  title={Multi-Objective Semantic Segmentation for Building Footprints},
  author={...},
  year={2026}
}
```

## Support

For issues, feature requests, or contributions, please refer to the main project README.
