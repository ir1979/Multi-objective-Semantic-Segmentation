# Grid Search Quick Reference

## Quick Start (30 seconds)

```bash
# 1. Validate configuration
python scripts/validate_grid_search.py --show-points

# 2. Run grid search
python run_grid_search.py

# 3. Check results
python scripts/grid_search_utils.py list --top-n 20
```

## Common Commands

### Running Grid Search
```bash
# Start fresh
python run_grid_search.py --force-restart

# Resume from checkpoint
python run_grid_search.py

# Run specific config
python run_grid_search.py --config configs/my_grid.yaml

# Only generate reports
python run_grid_search.py --generate-reports-only
```

### Monitoring Progress
```bash
# Main log
tail -f grid_search_results/grid_search.log

# Event stream
tail -f grid_search_results/events.jsonl | jq '.type' | sort | uniq -c

# Metrics stream
tail -f grid_search_results/metrics.jsonl | jq '.val_iou' | sort -n
```

### Inspecting Results
```bash
# List top results
python scripts/grid_search_utils.py --results-dir grid_search_results list --top-n 20

# Show status
python scripts/grid_search_utils.py status

# Show config for a point
python scripts/grid_search_utils.py config --point-id 0

# Show errors
python scripts/grid_search_utils.py errors
```

## Configuration Quick Guide

### Minimal Grid Search
```yaml
grid_search:
  parameters:
    model_architecture: ["unet", "unetpp"]
    learning_rate: [1.0e-4, 1.0e-3]
```

### Random Sampling (50 points)
```yaml
grid_search:
  selection:
    strategy: "random"
    n_points: 50
    random_seed: 42
```

### With Constraints
```yaml
grid_search:
  constraints:
    - if_model_architecture: "unet"
      then_not_deep_supervision: true
```

### All Parameters
```yaml
grid_search:
  parameters:
    model_architecture: ["unet", "unetpp"]
    loss_strategy: ["single", "weighted", "mgda"]
    pixel_loss_type: ["bce", "bce_iou", "dice"]
    learning_rate: [1.0e-4, 5.0e-4, 1.0e-3]
    batch_size: [4, 8]
    dropout_rate: [0.2, 0.3, 0.5]
    deep_supervision: [false, true]
    boundary_loss_weight: [0.0, 0.3, 0.5]
    shape_loss_weight: [0.0, 0.1, 0.2]
```

## Folder Structure

```
grid_search_results/
├── grid_search_state.json        # Checkpoint (resumable)
├── grid_search.log               # Main log
├── events.jsonl                  # Event stream
├── metrics.jsonl                 # Metrics stream
├── error_log.json                # Errors (if any)
├── point_000000/
│   ├── config.yaml              # Point-specific config
│   ├── model.h5                 # Trained model
│   ├── run.log                  # Point log
│   └── metrics.csv              # Point metrics
├── point_000001/
└── reports/
    ├── grid_search_results.csv   # All results
    ├── best_configurations.csv   # Top 5 configs
    ├── grid_search_table.tex     # LaTeX table
    ├── summary_statistics.json   # Aggregated stats
    ├── comparison_by_architecture.png
    ├── comparison_by_strategy.png
    ├── distribution_results.png
    └── heatmap_results.png
```

## Troubleshooting

### Restart Stuck Search
```bash
# Force restart
python run_grid_search.py --force-restart
```

### Low Memory
```yaml
# Reduce batch size
parameters:
  batch_size: [4]

# Disable TensorBoard logging
logging:
  tensorboard: false
```

### Check Point Failure
```bash
# View point-specific config
python scripts/grid_search_utils.py config --point-id 42

# Check point logs
cat grid_search_results/point_000042/run.log
```

### Extract Results
```bash
# Read raw results
cat grid_search_results/reports/grid_search_results.csv

# Parse JSON
cat grid_search_results/grid_search_state.json | jq '.[].metrics | sort_by(.val_iou)'
```

## Performance Tips

1. **Use random sampling** instead of full grid
2. **Enable early stopping** to reduce training time
3. **Disable logging overhead**:
   ```yaml
   logging:
     tensorboard: false
     validation_image_interval: 0
   ```
4. **Use constraints** to filter invalid combinations
5. **Start small**, expand after validation

## One-Liners

```bash
# Best result
cat grid_search_results/reports/best_configurations.csv | head -2

# Count by architecture
cat grid_search_results/grid_search_state.json | jq -r '.[].params.model_architecture' | sort | uniq -c

# Average IoU
cat grid_search_results/metrics.jsonl | jq '.val_iou' | awk '{s+=$1; n++} END {print s/n}'

# Failed points
cat grid_search_results/grid_search_state.json | jq 'map(select(.status=="failed"))'

# Export results to Excel
python -c "import pandas as pd; df = pd.read_csv('grid_search_results/reports/grid_search_results.csv'); df.to_excel('results.xlsx')"
```

## Files to Review

1. **Implementation Details**: `GRID_SEARCH_IMPLEMENTATION.md`
2. **Full Documentation**: `docs/GRID_SEARCH.md`
3. **Configuration Template**: `configs/grid_search.yaml`
4. **Validation Script**: `scripts/validate_grid_search.py`
5. **Management Utility**: `scripts/grid_search_utils.py`
6. **Main Entry Point**: `run_grid_search.py`

## Key Features

- ✅ Automatic checkpointing for resumability
- ✅ Error handling with retry logic
- ✅ Comprehensive logging and monitoring
- ✅ Results aggregation and visualization
- ✅ Configuration validation
- ✅ Paper-ready output generation
- ✅ Multiple selection strategies
- ✅ Constraint filtering
- ✅ Error categorization and recovery
- ✅ Progress tracking

## Integration Checklist

- [ ] Validate grid search config: `python scripts/validate_grid_search.py --show-points`
- [ ] Test on small grid: `python run_grid_search.py --config configs/test_grid.yaml`
- [ ] Check output directory structure
- [ ] Verify report generation: `python run_grid_search.py --generate-reports-only`
- [ ] Test resumption (Ctrl+C and restart)
- [ ] Review logs and error handling
- [ ] Inspect generated tables/figures
- [ ] Document results

---

For full documentation, see `docs/GRID_SEARCH.md`
