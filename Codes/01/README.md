# Building Segmentation with Multi-Objective Optimization

Publication-oriented framework for building footprint segmentation from RGB
satellite tiles with two parallel backends:

- TensorFlow (`run_all.py`)
- PyTorch (`run_pytorch.py`)

## Quick Start

```bash
pip install -r requirements.txt
python run_tests.py
python run_all.py --no-pareto
python run_pytorch.py --config configs/pytorch_default.yaml --name pt_run
```

## Windows/Conda (your environment-style setup)

If you are on Windows + Conda similar to your provided `seg4` environment:

```bash
conda activate seg4
pip install -r requirements.txt
```

For CUDA-specific PyTorch builds, use the official wheel selector if needed:
https://pytorch.org/get-started/locally/

## Dataset Layout

Expected input directories:

- `datasets/RGB/*.png` (RGB images, `256x256x3`)
- `datasets/Mask/*.tif` (binary masks, `256x256x1`)

Matching is done by basename (`tile_001.png` <-> `tile_001.tif`).

## Main Commands

```bash
python run_all.py                                  # TensorFlow full pipeline
python run_all.py --experiment unet_mgda           # TensorFlow single experiment
python run_pytorch.py --config configs/pytorch_default.yaml --name pt_mgda
python run_tests.py                                # test suite
```

## Project Structure

```
configs/              YAML configs (TF + PyTorch variants)
frameworks/pytorch/   PyTorch pipeline (data/models/losses/training/optimization)
data/                 TensorFlow data pipeline
models/               TensorFlow models
losses/               TensorFlow losses
optimization/         TensorFlow optimization + MGDA
training/             TensorFlow trainer/evaluator/checkpointing
experiments/          TensorFlow experiment orchestration
visualization/        figures and LaTeX table helpers
logging_utils/        console/file/TensorBoard/CSV/JSON logging
tests/                unit/integration-style tests
Misc/legacy/          relocated legacy code not used by active pipelines
```

## PyTorch Configs

- `configs/pytorch_default.yaml`
- `configs/pytorch_unet_single.yaml`
- `configs/pytorch_unet_mgda.yaml`
- `configs/pytorch_unetpp_mgda.yaml`

## Reproducibility

- TensorFlow: `utils/reproducibility.py`
- PyTorch: `frameworks/pytorch/utils/reproducibility.py`

Both pin seeds and deterministic options where supported.

## Notes

- Figures are saved as PNG + PDF at 300 DPI.
- Tables are exported as CSV + LaTeX.
- `main.py` proxies to `run_all.py` for backward compatibility.
