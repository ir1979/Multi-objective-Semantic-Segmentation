# Multi-Objective Optimization Framework for Building Footprint Recognition

A comprehensive, publication-ready Python framework for multi-objective optimization of deep learning models for building footprint extraction from aerial/satellite imagery using semantic segmentation.

## Overview

This framework explores Pareto-optimal trade-offs among three competing objectives:

1. **Maximize Segmentation Accuracy**: IoU, F1-score, Precision, Recall
2. **Minimize Model Complexity**: Parameter count, FLOPs
3. **Minimize Inference Time**: Latency per image (ms)

## Features

### Model Architectures
- **U-Net** - Classic encoder-decoder architecture
- **U-Net++** - Nested skip connections with deep supervision
- **FPN** - Feature Pyramid Network
- **LinkNet** - Efficient encoder-decoder with residual blocks
- **DeepLabV3+** - Atrous convolution with ASPP
- **PSPNet** - Pyramid Scene Parsing Network
- **PAN** - Pyramid Attention Network
- **Custom Architectures**: NestedUNet, AttUNet, R2AttUNet, ResUNet, ResUNet++, SEUNet, scSEUNet, UNet3+

### Encoder Backbones
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0, B1, B2, B3, B4, B5, B6, B7)
- VGG (16, 19)
- DenseNet (121, 169, 201)
- MobileNet, MobileNetV2
- SE-ResNet, SE-ResNeXt, SENet

### Multi-Objective Optimization
- **NSGA-II** algorithm via `pymoo`
- **Grid Search** for exhaustive evaluation
- Pareto front extraction and visualization
- Hypervolume calculation for convergence analysis
- Registry-based extension points for new models and new objectives

### Evaluation Metrics
- **Segmentation**: IoU (Jaccard), Dice/F1, Precision, Recall, Pixel Accuracy
- **Boundary**: Boundary IoU, Boundary F1
- **Complexity**: Parameter count, FLOPs
- **Speed**: Inference time (mean, std, min, max)

### Publication-Ready Outputs
- LaTeX-formatted tables
- High-resolution Pareto front plots (2D and 3D)
- Metric distribution visualizations
- Qualitative prediction comparisons
- Comprehensive experiment reports

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd building_footprint_moo
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install segmentation-models**:
```bash
pip install segmentation-models
```

## Quick Start

### 1. Create Configuration File

Generate a default configuration:
```bash
python main.py --create-config configs/my_experiment.yaml
```

Or use the provided example:
```bash
cp configs/example_config.yaml configs/my_experiment.yaml
```

### 2. Edit Configuration

Update `configs/my_experiment.yaml` with your settings:

```yaml
experiment:
  name: "my_building_experiment"
  seed: [42, 123, 456]

dataset:
  name: "Inria Aerial"
  rgb_path: "Datasets/RGB"
  mask_path: "Datasets/Mask"
  tile_size: [256, 512]

models:
  architectures: [Unet, UnetPlusPlus, FPN]
  encoders: [resnet34, efficientnetb0]

training:
  loss_functions: [binary_crossentropy, dice_loss]
  learning_rates: [1e-3, 1e-4]
  batch_sizes: [8, 16]
  epochs: [50]
```

### 3. Run Experiment

**Full grid search**:
```bash
python main.py --config configs/my_experiment.yaml --mode full --gpu 0
```

**Quick test** (subset of configurations):
```bash
python main.py --config configs/my_experiment.yaml --mode quick --gpu 0
```

**Evaluate existing models**:
```bash
python main.py --config configs/my_experiment.yaml --mode evaluate --eval-dir outputs/run_001
```

## New Starting Points

- Use `configs/paper_ready_config.yaml` as the main template for paper experiments.
- Use `plugins/README.md` to add custom models and objectives without editing core files.
- Use `docs/tutorial.md` for a full concept review and step-by-step worked example.

## Project Structure

```
building_footprint_moo/
├── configs/                    # Configuration files
│   ├── experiment_config.yaml
│   └── dataset_config.yaml
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   └── dataset.py
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── model_factory.py        # Unified model builder
│   ├── Unet.py
│   ├── unet_pp.py
│   ├── NestedUnet.py
│   ├── AttUNet.py
│   ├── R2AttUNet.py
│   ├── ResUnet.py
│   ├── ResUnetPlusPlus.py
│   ├── SEUnet.py
│   ├── scSEUnet.py
│   └── Unet3p.py
├── losses/                     # Loss functions
│   ├── __init__.py
│   ├── loss_manager.py
│   ├── pixel_losses.py
│   ├── boundary_losses.py
│   └── shape_losses.py
├── training/                   # Training and metrics
│   ├── __init__.py
│   ├── trainer.py
│   ├── trainer_mgda.py
│   └── metrics.py
├── src/                        # Core framework modules
│   ├── optimize.py            # MOO algorithms (NSGA-II, grid search)
│   ├── objectives.py          # Objective registry
│   ├── pareto.py              # Pareto front extraction and plotting
│   └── evaluate.py            # Comprehensive evaluation
├── config/                     # Configuration management
│   └── config_manager.py
├── experiments/                # Experiment utilities
│   └── runner.py
├── visualization/              # Visualization utilities
│   └── visualization.py
├── utils/                      # Utility functions
│   ├── repro.py               # Reproducibility utilities
│   └── system.py              # System utilities
├── outputs/                    # Experiment outputs (auto-created)
│   ├── figures/               # Plots and visualizations
│   ├── tables/                # CSV and LaTeX tables
│   ├── masks/                 # Output masks
│   ├── logs/                  # Training logs
│   ├── checkpoints/           # Model checkpoints
│   └── optimization/          # Optimization results
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb
├── main.py                     # Main orchestrator
├── requirements.txt
└── README.md
```

## Configuration System

The framework uses YAML-based configuration files that define the complete experimental grid.

### Example Configuration

```yaml
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
  train_val_test_split: [0.7, 0.15, 0.15]
  rgb_path: "Datasets/RGB"
  mask_path: "Datasets/Mask"

models:
  architectures: [Unet, UnetPlusPlus, FPN, Linknet]
  encoders: [resnet34, resnet50, efficientnetb0]
  encoder_weights: [imagenet]

training:
  loss_functions: [binary_crossentropy, dice_loss, focal_loss]
  optimizers: [Adam, SGD]
  learning_rates: [1e-3, 1e-4, 5e-4]
  batch_sizes: [8, 16]
  epochs: [50, 100]
  schedulers: [ReduceLROnPlateau, CosineAnnealing]

multi_objective:
  objectives: [iou, f1_score, param_count, flops, inference_time]
  optimization_method: "NSGA-II"  # or "grid_search"
  population_size: 50
  generations: 100
  pareto_front: true
```

### Configuration Expansion

The framework automatically expands the hyperparameter grid into individual experiment configurations. Each configuration receives a unique experiment ID based on its parameters.

## Extending The Framework

### Register a New Model

```python
from tensorflow import keras
from models import register_model_builder

def build_tiny_unet(**kwargs):
    inputs = keras.Input(shape=kwargs["input_shape"])
    outputs = keras.layers.Conv2D(kwargs["num_classes"], 1, activation="sigmoid")(inputs)
    return keras.Model(inputs, outputs, name="TinyUNet")

register_model_builder(
    name="TinyUNet",
    builder=build_tiny_unet,
    aliases=["tiny_unet"],
)
```

After registration, `TinyUNet` can be used in `models.architectures` in your YAML config.

### Register a New Objective

```python
from src import register_objective

register_objective(
    name="memory_footprint_mb",
    direction="min",
    getter=lambda result: result["summary"]["memory_footprint_mb"],
)
```

Then add it to `multi_objective.objectives` in the YAML config:

```yaml
multi_objective:
  objectives: [iou, f1_score, memory_footprint_mb, inference_time]
```

Objective getters receive the full evaluated result dictionary, including config metadata, timing, model info, and the nested training `summary`.

### Auto-Load Plugin Modules

If you prefer not to edit core files, place your custom registrations in `plugins/` and list them in the YAML config:

```yaml
extensions:
  plugin_modules:
    - plugins.objectives
    - plugins.models
```

Each listed module is imported at runtime, so any `register_model_builder(...)` and `register_objective(...)` calls inside those files become available automatically. Example plugin files are provided in `plugins/models.py` and `plugins/objectives.py`.

## Multi-Objective Optimization

### NSGA-II Algorithm

The framework implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization:

```python
from src.optimize import run_nsga2_optimization

results = run_nsga2_optimization(
    config_space=configs,
    dataset=dataset,
    population_size=50,
    n_generations=100,
    seed=42
)
```

### Grid Search

For exhaustive evaluation of all hyperparameter combinations:

```python
from src.optimize import grid_search_optimization

results = grid_search_optimization(
    config_space=configs,
    dataset=dataset,
    max_configs=100  # Optional: limit number of configs
)
```

## Evaluation and Metrics

### Segmentation Metrics

- **IoU (Intersection over Union)**: Standard segmentation metric
- **Dice / F1**: Harmonic mean of precision and recall
- **Precision**: Pixel-level positive predictive value
- **Recall**: Pixel-level sensitivity
- **Accuracy**: Overall pixel classification rate
- **Boundary IoU/F1**: Edge alignment quality

### Model Complexity Metrics

- **Parameter Count**: Total trainable parameters
- **FLOPs**: Floating point operations
- **Inference Time**: Milliseconds per image

### Usage

```python
from src.evaluate import evaluate_model

results = evaluate_model(model, dataset, config)
print(f"IoU: {results['iou']:.4f}")
print(f"Parameters: {results['param_count']:,}")
print(f"Inference time: {results['inference_time']:.2f} ms")
```

## Visualization

### Pareto Front Plots

```python
from src.pareto import plot_pareto_front

plot_paths = plot_pareto_front(
    results,
    output_dir='outputs/figures',
    objectives=['iou', 'f1_score', 'param_count', 'flops', 'inference_time']
)
```

Generated plots include:
- IoU vs. Parameter Count
- IoU vs. Inference Time
- F1 vs. Parameter Count
- F1 vs. Inference Time
- 3D Pareto front (if applicable)

### Metric Distributions

```python
from src.evaluate import plot_metric_distributions

plot_paths = plot_metric_distributions(
    results,
    output_dir='outputs/figures',
    group_by='architecture'
)
```

## Publication-Ready Outputs

### LaTeX Tables

The framework automatically generates LaTeX-formatted tables:

```latex
\begin{table}[htbp]
\centering
\caption{Top 10 Model Configurations by IoU}
\label{tab:top_results}
\begin{tabular}{llcccccc}
\toprule
Architecture & Encoder & IoU & F1 & Params & FLOPs & Time (ms) \\
\midrule
UnetPlusPlus & efficientnetb0 & 0.8234 & 0.9012 & 7.2M & 15.3G & 45.2 \\
FPN & resnet50 & 0.8156 & 0.8987 & 25.6M & 42.1G & 78.5 \\
...
\bottomrule
\end{tabular}
\end{table}
```

### Figures

All figures are saved as high-resolution PNG (300 DPI) suitable for publication:
- Pareto front plots
- Metric distributions
- Training curves
- Qualitative predictions

## Reproducibility

### Seed Control

All random seeds are controllable via configuration:

```yaml
experiment:
  seed: [42, 123, 456]  # Multiple seeds for statistical significance
```

### Deterministic Training

The framework sets seeds for:
- Python `random`
- NumPy
- TensorFlow
- Dataset shuffling

### Experiment Tracking

Each run creates:
- Unique run ID with timestamp
- Complete configuration snapshot
- Git commit hash (if available)
- Dataset hash for verification

## Advanced Usage

### Custom Architectures

Add new architectures to `models/model_factory.py`:

```python
def build_custom_model(input_shape, num_classes):
    # Your implementation
    return model

# Register in ARCHITECTURE_MAP
ARCHITECTURE_MAP['custom'] = 'CustomModel'
```

### Custom Loss Functions

Add to `losses/loss_manager.py`:

```python
def my_custom_loss(y_true, y_pred):
    # Your implementation
    return loss
```

### Custom Objectives

Extend the multi-objective optimization:

```python
from src.optimize import ObjectiveVector

class CustomObjectiveVector(ObjectiveVector):
    def __init__(self, iou, f1, params, flops, time, custom_metric):
        super().__init__(iou, f1, params, flops, time)
        self.custom_metric = custom_metric
```

## Troubleshooting

### GPU Memory Issues

Reduce batch size or use mixed precision:
```yaml
training:
  batch_sizes: [4, 8]  # Smaller batches
```

### Out of Disk Space

Limit output generation:
```yaml
experiment:
  save_predictions: false
  save_checkpoints: false
```

### Slow Optimization

Use NSGA-II instead of grid search:
```yaml
multi_objective:
  optimization_method: "NSGA-II"
  population_size: 30
  generations: 50
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{building_footprint_moo,
  title = {Multi-Objective Optimization Framework for Building Footprint Recognition},
  author = {Research Team},
  year = {2026},
  url = {https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [segmentation_models](https://github.com/qubvel/segmentation_models) for the model implementations
- [pymoo](https://pymoo.org/) for multi-objective optimization algorithms
- [TensorFlow](https://tensorflow.org/) and [Keras](https://keras.io/) for deep learning framework

## Contact

For questions or issues, please open an issue on GitHub or contact the research team.

---

**Last Updated**: April 2026

### Dataset configuration

- `rgb_glob` - glob pattern for RGB tiles, e.g. `./Datasets/RGB/*.png`
- `mask_glob` - glob pattern for binary masks, e.g. `./Datasets/Mask/*.tif`
- `train_split` - fraction of samples used for training (e.g. `0.8` for 80% train, 20% validation)

### Loss configuration

- `pixel_loss` - supported values: `bce`, `iou`, `bce+iou`
- `boundary_loss` - supported values: `hausdorff` or `none`
- `shape_loss` - supported values: `convexity` or `none`
- `pixel_weight`, `boundary_weight`, `shape_weight` - relative weights for each loss term

### Training configuration

- `mode` - `standard` or `mgda`
- `batch_size` - training batch size
- `epochs` - number of epochs to train
- `lr` - learning rate
- `seed` - random seed for reproducibility
- `resume` - whether to continue from the latest checkpoint
- `log_every` - MGDA logging frequency (prints per batch)

### Output configuration

- `output_dir` - base directory for results
- `experiment_name` - optional folder name for a stable experiment path

## How parameter changes affect final objectives

- `mode`
  - `standard`: uses weighted loss sum and is simpler to tune.
  - `mgda`: uses multi-objective gradient balancing for more stable trade-offs between objectives.

- `pixel_loss`
  - `bce` focuses on per-pixel classification.
  - `iou` focuses on overlap / segmentation quality.
  - `bce+iou` combines both and is often a good research default.

- `boundary_loss`
  - Adds shape boundary sensitivity, useful when building outlines are important.
  - Use `hausdorff` to push predictions closer to mask boundaries.

- `shape_loss`
  - Adds a shape regularization component.
  - `convexity` encourages compact, building-like shapes.

- `*_weight`
  - Increase a weight to place more importance on that objective.
  - Decrease it to reduce its influence relative to the other losses.

- `batch_size`, `lr`, `epochs`
  - Larger `batch_size` can improve training stability but may require more memory.
  - Lower `lr` often improves convergence; higher values may make training faster but less stable.
  - More `epochs` give the model more time to learn; monitor validation metrics.

- `train_split`
  - A lower split (e.g. `0.7`) gives more validation data and less training data.
  - A higher split (e.g. `0.9`) gives more training data, but less validation support.

## Objectives and assessment measures

The framework now supports tracking and evaluating the following objectives:

- `pixel_accuracy` — per-pixel correctness across the predicted mask
- `boundary_fidelity` — boundary IoU and boundary F1 on the predicted edges
- `region_completeness` — IoU / Dice measuring how completely buildings are recovered
- `shape_regularization` — compactness-based shape quality for building masks
- `precision` / `recall` — false positive vs false negative trade-offs
- `data_fit` — training metrics measure how well the model fits the training set
- `generalization` — validation metrics measure how well the model generalizes
- `boundary_F1` / `boundary_IoU` — evaluate how well boundaries align with ground truth
- `compactness` / `shape_priors` — encourage regular building geometries
- `topological_correctness` — approximate object count consistency in the prediction
- `train_runtime` / `validation_runtime` — timing information for performance comparison
- `memory_footprint` — process memory usage during training

These objectives are tracked in the training logs and saved experiment summaries.

## Running training

### Run from defaults in `config.py`

```bash
python main.py
```

### Override with CLI arguments

```bash
python main.py --mode mgda --model unet++ --deep_supervision --batch_size 8 --epochs 100 --lr 0.0003
```

### Use a YAML config file

```bash
python main.py --config config_example.yaml
```

### Override a config file value

```bash
python main.py --config config_example.yaml --epochs 120 --lr 0.0002
```

## Example YAML configs

The repository includes three example configs:

- `config_example.yaml` — generic template for a single configurable experiment
- `config_standard.yaml` — explicit standard training example
- `config_mgda.yaml` — explicit MGDA training example

Use one of these directly, or copy them and modify parameters for your own experiment.

### Run standard training example

```bash
python main.py --config config_standard.yaml
```

### Run MGDA training example

```bash
python main.py --config config_mgda.yaml
```

### Example YAML template

Use `config_example.yaml` as a starting point. It includes the most important experiment settings:

```yaml
mode: mgda
model: unet++
deep_supervision: true
pixel_loss: bce+iou
boundary_loss: hausdorff
shape_loss: convexity
pixel_weight: 1.0
boundary_weight: 1.0
shape_weight: 1.0
batch_size: 4
epochs: 100
lr: 0.0005
seed: 42
output_dir: ./results
experiment_name: moss_experiment
rgb_glob: ./Datasets/RGB/*.png
mask_glob: ./Datasets/Mask/*.tif
train_split: 0.8
input_height: 256
input_width: 256
input_channels: 3
log_every: 50
resume: false
```

## Output

Each experiment run creates a structured folder under `output_dir` containing:

- `checkpoints/`
- `tensorboard/`
- `logs.csv`
- `summary.json`
- `config.yaml` (when YAML writing is available)

## Reproducibility

The framework records:

- `dataset_hash` for the complete dataset used in the run
- `git_commit` when the repository is available
- `loss_config` and training settings

## Tests

Run the core tests with:

```bash
python -m pytest tests/test_core.py
```

## Notes

- `config.py` is the canonical place for the default hyperparameters.
- `config_example.yaml` is provided as a template for experiments.
- Use CLI overrides to change a small number of values without editing files.
- `psutil` is used for memory footprint reporting; install from `requirements.txt`.
- `scikit-image` is used for approximate topological correctness measurement if available.
- For multi-objective tuning, try varying `pixel_weight`, `boundary_weight`, and `shape_weight` while comparing `mode: standard` and `mode: mgda`.
