# Building Segmentation with Multi-Objective Optimization

Publication-oriented framework for building footprint segmentation from RGB
satellite tiles using TensorFlow, UNet/UNet++, weighted losses, and MGDA.

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py --test-only
python run_all.py --no-pareto
```

## Dataset Layout

Expected input directories:

- `datasets/RGB/*.png` (RGB images, `256x256x3`)
- `datasets/Mask/*.tif` (binary masks, `256x256x1`)

Matching is done by basename (`tile_001.png` ↔ `tile_001.tif`).

## Main Commands

```bash
python run_all.py                          # full pipeline
python run_all.py --experiment unet_mgda  # single experiment
python run_all.py --figures-only          # regenerate paper outputs
python run_tests.py                        # run all tests
```

## Project Structure

```
configs/            YAML configs (base + experiment variants)
data/               data loader, splitter, preprocessing, augmentation
models/             UNet, UNet++, blocks, complexity
losses/             pixel/boundary/shape losses + manager
optimization/       MGDA, Pareto, weighted sum, schedulers
training/           trainer, evaluator, checkpointing, callbacks
experiments/        experiment runner, registry, comparison/ablation
visualization/      figures and LaTeX table helpers
logging_utils/      console/file/TensorBoard/CSV/JSON logging
tests/              unit/integration-style tests on synthetic data
```

## Reproducibility

`utils/reproducibility.py` pins random seeds, enables deterministic TF ops,
and computes dataset hashes for run tracking.

## Notes

- Figures are saved as PNG + PDF at 300 DPI.
- Tables are exported as CSV + LaTeX.
- `main.py` proxies to `run_all.py` for backward compatibility.
