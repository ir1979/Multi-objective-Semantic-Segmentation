# Grid Search Workflow & Architecture

## Execution Flow Diagram

```
START: run_grid_search.py
    │
    ├─→ Load Configuration (grid_search.yaml)
    │   └─→ Validate Config (config_validator.py)
    │       ├─ Parameter validation
    │       ├─ Constraint validation
    │       └─ Grid size estimation
    │
    ├─→ Initialize Grid Search Runner
    │   ├─ Setup Logging (GridSearchLogger)
    │   ├─ Setup Error Handling (ErrorHandler)
    │   ├─ Setup Recovery Strategy
    │   └─ Load/Create State (grid_search_state.json)
    │
    ├─→ Generate Grid Points
    │   ├─ Create Cartesian product (or sample)
    │   ├─ Apply constraints
    │   └─ Save state
    │
    ├─→ FOR EACH Grid Point:
    │   │
    │   ├─→ Create Point Directory
    │   │   └─ point_XXXXXX/
    │   │
    │   ├─→ Generate Point Config
    │   │   └─ Save config.yaml
    │   │
    │   ├─→ Execute Training
    │   │   ├─ Load data
    │   │   ├─ Train model
    │   │   ├─ Evaluate
    │   │   └─ Save metrics
    │   │
    │   ├─→ Error Handling
    │   │   ├─ Success? → Save metrics
    │   │   ├─ Recoverable? → Retry with backoff
    │   │   └─ Critical? → Mark failed, continue
    │   │
    │   └─→ Update State
    │       └─ Save checkpoint
    │
    ├─→ Generate Results Report
    │   ├─ Aggregate results
    │   ├─ Generate CSV
    │   ├─ Generate LaTeX table
    │   ├─ Generate plots
    │   ├─ Generate statistics
    │   └─ Identify best configs
    │
    └─→ END with Summary
        ├─ Total execution time
        ├─ Success/failure count
        ├─ Best configuration
        └─ Report location
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_grid_search.py                           │
│                     (Entry Point)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Validation  │ │ Logging      │ │ Error        │
    │ System      │ │ System       │ │ Handling     │
    │             │ │              │ │              │
    │ • Config    │ │ • Events     │ │ • Recovery   │
    │   Validator │ │   (JSONL)    │ │   Strategy   │
    │ • Grid      │ │ • Metrics    │ │ • Error      │
    │   Estimator │ │ • Logs       │ │   Handler    │
    └─────────────┘ └──────────────┘ └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   GridSearchRunner                 │
        │   (Main Orchestrator)              │
        │                                    │
        │  • Initialize grid                 │
        │  • Run each point                  │
        │  • Track state                     │
        │  • Aggregate results               │
        └────────────────┬───────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
    │ GridSearch  │ │ Results      │ │ State        │
    │ Config      │ │ Aggregator   │ │ Management   │
    │             │ │              │ │              │
    │ • Parameter │ │ • CSV/LaTeX  │ │ • Save       │
    │   space     │ │ • Plots      │ │   checkpoint │
    │ • Sampling  │ │ • Statistics │ │ • Resume     │
    │ • Filtering │ │ • Best-config│ │ • Resume     │
    └─────────────┘ └──────────────┘ └──────────────┘
```

## State Management Flow

```
┌──────────────────────────────────────────────┐
│      grid_search_state.json (Checkpoint)     │
│                                              │
│  {                                           │
│    "0": {                                    │
│      "point_id": 0,                          │
│      "params": {...},                        │
│      "status": "completed",                  │
│      "metrics": {...},                       │
│      "result_dir": "point_000000"            │
│    },                                        │
│    "1": {                                    │
│      "status": "pending",                    │
│      ...                                     │
│    }                                         │
│  }                                           │
└────────────────┬─────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
LOAD STATE                  SAVE STATE
(Resume from)            (After each point)
    │                         │
    │   ┌─────────────────────┘
    │   │
    └───┴─→ Continue execution
            from last point
```

## Error Handling Flow

```
Error Occurs in Point Execution
    │
    ├─→ ErrorHandler categorizes error
    │   │
    │   ├─→ Is Critical? (KeyboardInterrupt, SystemExit)
    │   │   └─→ Abort, Save State, Exit
    │   │
    │   ├─→ Is Recoverable? (OSError, TimeoutError, RuntimeError)
    │   │   └─→ Pass to RecoveryStrategy
    │   │
    │   └─→ Other Error
    │       └─→ Log and Mark Failed
    │
    └─→ RecoveryStrategy executes retry
        │
        ├─→ Attempt 1: Wait 1.0s, retry
        │   ├─ Success? → Continue
        │   └─ Fail? → Try 2
        │
        ├─→ Attempt 2: Wait 2.0s, retry
        │   ├─ Success? → Continue
        │   └─ Fail? → Try 3
        │
        ├─→ Attempt 3: Wait 4.0s, retry
        │   ├─ Success? → Continue
        │   └─ Fail? → Try 4
        │
        └─→ Attempt 4: Wait 8.0s, retry
            ├─ Success? → Continue
            └─ Fail? → Mark Failed, Continue with next point
```

## Logging Architecture

```
┌────────────────────────────────────────────────┐
│          GridSearchLogger                      │
│  (Specialized for grid search)                 │
└────────┬───────────────────────────────────────┘
         │
    ┌────┴─────────┬───────────────┬──────────────┐
    │              │               │              │
    ▼              ▼               ▼              ▼
CONSOLE       MAIN LOG       EVENTS JSONL    METRICS JSONL
(INFO)        (DEBUG)        (Structured)    (Metrics)

Console:      grid_search.  events.jsonl    metrics.jsonl
INFO          log            
              (All levels)   {              {
[HH:MM:SS]                     "timestamp":  "point_id": 0,
level -                        "type":       "timestamp":
message       Detailed         "point_id":   "val_iou":
              debug info       0,            0.847,
              and traces       "status":     "train_loss":
              (full context)   "running"     0.234,
                             }              }
```

## Configuration Space Visualization

```
Parameter Space: {A, B, C, D, E}

Full Cartesian Product (strategy: "full")
├─→ 5 × 4 × 3 × 2 × 3 = 360 combinations
└─→ All evaluated

Random Sampling (strategy: "random")
├─→ Generate 360 combinations
├─→ Sample 50 random points uniformly
└─→ Evaluate 50 points

Latin Hypercube (strategy: "latin_hypercube")
├─→ Generate 360 combinations
├─→ Stratify into 50 bins
├─→ Sample 1 from each bin
└─→ Evaluate 50 strategically distributed points

Constraints Applied (if defined)
├─→ Filter invalid combinations
└─→ Reduce search space before sampling
```

## Output Generation Pipeline

```
Completed Points
    │
    ├─→ Extract Metrics
    │   └─ Collect all results
    │
    ├─→ Generate CSV
    │   ├─ Point ID
    │   ├─ All parameters
    │   └─ All metrics
    │
    ├─→ Generate LaTeX
    │   ├─ Select top N
    │   ├─ Format columns
    │   └─ Create table
    │
    ├─→ Generate Plots
    │   ├─ Comparison by architecture
    │   ├─ Comparison by strategy
    │   ├─ Distribution
    │   └─ Heatmap
    │
    ├─→ Calculate Statistics
    │   ├─ Mean, std, min, max
    │   ├─ Per-architecture
    │   ├─ Per-strategy
    │   └─ Overall
    │
    └─→ Identify Best
        ├─ Sort by metric
        ├─ Select top N
        └─ Save configurations
```

## Directory Structure After Execution

```
project_root/
│
├── grid_search_results/
│   ├── grid_search_state.json           [Checkpoint - use for resume]
│   ├── grid_search.log                  [Main log file]
│   ├── grid_search_main.log             [Runner log]
│   ├── events.jsonl                     [Event stream]
│   ├── metrics.jsonl                    [Metrics stream]
│   ├── error_log.json                   [Error summary]
│   │
│   ├── point_000000/                    [Per-point results]
│   │   ├── config.yaml                  [Point config]
│   │   ├── model.h5                     [Trained model]
│   │   ├── run.log                      [Execution log]
│   │   └── metrics.csv                  [Metrics]
│   │
│   ├── point_000001/
│   ├── point_000002/
│   ├── ... (more points)
│   │
│   └── reports/                         [Generated reports]
│       ├── grid_search_results.csv      [All results table]
│       ├── best_configurations.csv      [Top configs]
│       ├── grid_search_table.tex        [LaTeX table]
│       ├── summary_statistics.json      [Stats]
│       ├── comparison_by_architecture.png
│       ├── comparison_by_strategy.png
│       ├── distribution_results.png
│       └── heatmap_results.png
│
└── [Original project structure preserved]
```

## Integration Points

```
Existing Project Components
│
├─→ GridSearchRunner ◀───────────┐
│   (New)                        │
│   │                            │
│   ├─→ Experiment Runner       │
│   │   (Existing)              │
│   │   ├─ Models               │
│   │   ├─ Loss Manager          │
│   │   ├─ Trainer               │
│   │   └─ Evaluator             │
│   │                            │
│   └─→ Data Loaders             │
│       (Existing)              │
│       ├─ Augmentation          │
│       ├─ Preprocessing         │
│       └─ Splitting             │
│
├─→ Configuration System         │
│   (New + Existing)            │
│   ├─ Grid Search Config        │
│   ├─ Validator                 │
│   └─ Loader                    │
│
├─→ Logging System               │
│   (New + Existing)            │
│   ├─ Grid Search Logger        │
│   ├─ Dual Logger               │
│   └─ TensorBoard Logger        │
│
└─→ Results Analysis             │
    (New)
    ├─ Results Aggregator
    ├─ Plotting
    └─ Statistics
```

## Resume/Recovery Mechanism

```
START: python run_grid_search.py
    │
    ├─→ Check: state file exists?
    │   │
    │   ├─ NO: Create new state
    │   │       │
    │   │       └─→ Generate grid points
    │   │           │
    │   │           └─→ All points status = "pending"
    │   │
    │   └─ YES: Load existing state
    │           │
    │           ├─ Found completed points? → Skip
    │           ├─ Found running points?   → Retry
    │           ├─ Found failed points?    → Retry
    │           └─ Found pending points?   → Process
    │
    └─→ FOR EACH pending/running/failed point:
        │
        ├─→ Check checkpoint exists?
        │   │
        │   ├─ YES: Resume from checkpoint
        │   │       └─ Load model state
        │   │
        │   └─ NO: Start fresh
        │           └─ Train from scratch
        │
        └─→ Update state and continue
```

---

This comprehensive architecture enables:
- ✅ Systematic hyperparameter exploration
- ✅ Automatic recovery from failures
- ✅ Efficient monitoring and logging
- ✅ Reproducible results
- ✅ Paper-ready output generation
