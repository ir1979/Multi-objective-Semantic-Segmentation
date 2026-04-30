# Building Footprint Segmentation — Grid Search Framework

A research framework for **binary semantic segmentation of building footprints** from RGB satellite imagery. It runs a systematic grid search over six hyperparameters, measures eight evaluation metrics, identifies the Pareto-optimal configurations, and produces publication-ready figures and LaTeX tables automatically.

---

## What the Project Does

Given a set of 256×256 RGB satellite tiles and matching binary building masks, the pipeline:

1. **Trains** multiple segmentation models (UNet / UNet++) using different combinations of hyperparameters.
2. **Evaluates** every trained model on a held-out test split using eight metrics.
3. **Identifies the Pareto front** — configurations that are not dominated across the three most important objectives (IoU, Boundary F1, Shape Compactness).
4. **Exports paper-ready outputs**: prediction images, metric tables in LaTeX, box plots, Pareto scatter plots, and a diagnostic report.

The entire pipeline is driven by a single command and resumes automatically if interrupted.

---

## Dataset Layout

Place your data in the following directories before running anything:

```
datasets/
    RGB/      ← RGB satellite tiles    (256×256×3, PNG)
    Mask/     ← Binary building masks  (256×256×1, TIF or PNG)
```

Files are matched by **basename** — `tile_001.png` is paired with `tile_001.tif`.  
Pixel values in the mask should be 0 (background) or 255 / 1 (building).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+ and TensorFlow 2.10+. A GPU is recommended but not required.

### 2. Run the full grid search

```bash
python run_grid_search.py
```

This trains 36 randomly-sampled configurations (out of 324 possible combinations), evaluates each one, and writes all results to `grid_search_results/`.

### 3. Resume an interrupted run

The pipeline checkpoints after every configuration. Simply re-run the same command:

```bash
python run_grid_search.py --resume
```

### 4. Re-generate paper outputs only (no training)

```bash
python run_grid_search.py --generate-reports-only
```

### 5. Run the smoke test (verify the setup works)

```bash
python scripts/smoke_test_grid.py
```

This runs a single 1-epoch configuration and checks all steps end-to-end in a few minutes.

### 6. Launch the config wizard

```bash
python run_config_wizard.py
```

The wizard lets you build a single-run or grid-search YAML config, load an existing config, choose a parent config via `inherits`, preview the YAML that will be written, and save either an inherited override file or a fully resolved copy.

---

## The 6 Hyperparameters Searched

| # | Parameter | Values searched |
|---|-----------|----------------|
| 1 | **Architecture** | UNet, UNet++ |
| 2 | **Encoder depth** | Deep `[64,128,256,512,1024]`, Shallow `[32,64,128,256,512]` |
| 3 | **Pixel loss** | BCE, IoU, Dice |
| 4 | **Boundary loss weight** | 0.0, 0.3, 0.5 |
| 5 | **Shape loss weight** | 0.0, 0.1, 0.2 |
| 6 | **Learning rate** | 1e-4, 5e-4, 1e-3 |

> Configurations where both boundary weight and shape weight are 0 are automatically skipped (degenerate pixel-only variant). The default run randomly samples **36 configurations** from the constrained space.

---

## The 8 Evaluation Metrics

Each configuration is evaluated on the held-out test split:

| Metric | What it measures |
|--------|-----------------|
| **IoU** | Intersection over Union (region overlap) |
| **Dice** | F1 score over pixels |
| **Precision** | Fraction of predicted building pixels that are correct |
| **Recall** | Fraction of true building pixels that are detected |
| **Pixel Accuracy** | Overall per-pixel classification accuracy |
| **Boundary IoU** | IoU computed only near building edges |
| **Boundary F1** | F1 score near building edges |
| **Compactness** | Regularity/roundness of predicted building shapes |

---

## What Gets Produced

After the run completes, all outputs are written to `grid_search_results/`:

```
grid_search_results/
│
├── grid_search_state.json          ← checkpoint of every configuration's status
├── grid_search_main.log            ← full diagnostic log
│
├── point_000000/                   ← one folder per trained configuration
│   ├── config.yaml                 ← exact settings used
│   ├── checkpoints/                ← best model weights
│   ├── predictions/
│   │   ├── sample_000.png          ← 4-panel: RGB | Ground Truth | Prediction | TP/FP/FN
│   │   └── sample_001.png
│   └── logs.csv                    ← per-epoch training metrics
│
└── paper_outputs/
    ├── tables/
    │   ├── results_all.csv                   ← all 36 results
    │   ├── results_main.tex                  ← booktabs LaTeX table (top configurations)
    │   ├── hyperparameter_ablation.tex       ← mean ± std per hyperparameter value
    │   ├── pareto_front.csv
    │   └── pareto_front.tex
    ├── figures/
    │   ├── pareto_iou_vs_boundary.png/pdf    ← Pareto scatter: IoU vs Boundary F1
    │   ├── pareto_iou_vs_compactness.png/pdf ← Pareto scatter: IoU vs Compactness
    │   ├── comparison_by_model_architecture.png/pdf
    │   ├── comparison_by_pixel_loss_type.png/pdf
    │   ├── comparison_by_*.png/pdf           ← box plots for all 6 hyperparameters
    │   └── predictions_comparison_grid.png   ← top-5 configs side by side
    ├── diagnostics/
    │   ├── summary_statistics.json           ← mean/std/min/max per metric
    │   ├── metric_distributions.png          ← histogram grid for all 8 metrics
    │   └── correlation_matrix.png            ← Pearson correlation between metrics
    └── report_manifest.json                  ← paths to all generated files
```

---

## Understanding the Prediction Images

Each `sample_*.png` shows four panels side by side:

| Panel | Content |
|-------|---------|
| **RGB Input** | The satellite tile fed to the model |
| **Ground Truth** | The reference building mask |
| **Prediction** | The model's binary output |
| **TP / FP / FN** | Green = correct building, Red = false alarm, Blue = missed building |

---

## Understanding the Pareto Front

No single configuration is best on all metrics simultaneously. For example, a model that maximises Boundary F1 may sacrifice overall IoU. The **Pareto front** is the set of configurations where improving any one objective requires accepting a trade-off on another.

Three objectives are used: **IoU** (region accuracy), **Boundary F1** (edge accuracy), **Compactness** (shape regularity). The Pareto front tables and scatter plots let you choose the best configuration for your specific priority.

---

## Project Structure

```
run_grid_search.py      ← main entry point
configs/
    grid_search.yaml    ← all settings (grid space, training, evaluation, export)
    default.yaml        ← base defaults referenced by grid_search.yaml
data/                   ← data loading, augmentation, splitting, preprocessing
models/                 ← UNet, UNet++ (and other architectures in models/)
losses/                 ← pixel losses (BCE, Dice, BCE+IoU), boundary loss, shape loss
training/               ← trainer, evaluator, checkpointing, early stopping, callbacks
experiments/
    grid_search.py      ← grid point generation, training loop, prediction saving
    results_aggregator.py ← paper output generation (tables, figures, Pareto)
optimization/
    pareto.py           ← post-hoc Pareto front computation
visualization/          ← plot style, LaTeX table helpers, Pareto plots
logging_utils/          ← console/file/CSV logging
scripts/
    smoke_test_grid.py  ← quick end-to-end verification
tests/                  ← unit tests (run with: python run_tests.py)
Misc/                   ← archived / superseded code (not used by the pipeline)
```

---

## Configuration

All settings are in `configs/grid_search.yaml`. Key sections:

- **`grid_search.parameters`** — the six hyperparameter value lists
- **`grid_search.selection`** — set `n_points` to control how many configurations are trained
- **`training.epochs`** — default 50, with early stopping (patience 10) on validation IoU
- **`data`** — paths to `RGB/` and `Mask/` directories, batch size, train/val/test split
- **`export.save_predictions`** — set to `true` to save prediction images (recommended)

---

## Reproducibility

- Global random seed is set at startup (`project.seed: 1000`).
- Every trained configuration saves its exact resolved `config.yaml` inside its result folder.
- The `grid_search_state.json` file records all results and can be used to regenerate reports at any time.

---

## Running the Tests

```bash
python run_tests.py
```

Tests run on synthetic (randomly generated) data and do not require the real dataset.
