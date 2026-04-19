# Building Segmentation with Multi-Objective Optimization

This branch targets your environment:

- Python **3.7.16**
- TensorFlow **2.6.0** (`tensorflow-gpu` / `keras-gpu` stack)
- Windows + Conda (`seg4`-style setup)

The TensorFlow runner is the primary entrypoint:

- `run_all.py` (full TensorFlow pipeline)

PyTorch runner remains available, but the branch is tuned first for TF2.6 compatibility.

## Quick Start (TF2.6 / Py3.7)

```bash
conda activate seg4
pip install -r requirements.txt
python run_tests.py
python run_all.py --no-pareto
```

## Dataset Layout

Expected input directories:

- `datasets/RGB/*.png` (RGB images, `256x256x3`)
- `datasets/Mask/*.tif` (binary masks, `256x256x1`)

Matching is done by basename (`tile_001.png` <-> `tile_001.tif`).

## Main Commands

```bash
python run_all.py                                  # TensorFlow full pipeline
python run_all.py --experiment unet_mgda           # TensorFlow single experiment
python run_tests.py                                # test suite
python run_pytorch.py --config configs/pytorch_default.yaml --name pt_mgda
```

## Project Structure

```
configs/              YAML configs (TF + PyTorch variants)
data/                 TensorFlow data pipeline
models/               TensorFlow models
losses/               TensorFlow losses
optimization/         TensorFlow optimization + MGDA
training/             TensorFlow trainer/evaluator/checkpointing
experiments/          TensorFlow experiment orchestration
frameworks/pytorch/   Parallel PyTorch pipeline
visualization/        figures and LaTeX table helpers
logging_utils/        console/file/TensorBoard/CSV/JSON logging
tests/                unit/integration-style tests
Misc/legacy/          relocated legacy code not used by active pipelines
```

## Compatibility Notes

- `requirements.txt` is pinned to versions compatible with Python 3.7.16 + TF 2.6.0.
- `setup.py` requires `>=3.7,<3.8`.
- Runtime guards in `run_all.py` target TensorFlow 2.6.x and Python 3.7.x.

## Notes

- Figures are saved as PNG + PDF at 300 DPI.
- Tables are exported as CSV + LaTeX.
- `main.py` proxies to `run_all.py` for backward compatibility.
