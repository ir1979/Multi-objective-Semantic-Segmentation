# Plugin Guide

This folder is the recommended place for project-specific extensions that should remain separate from the core framework.

## Why Use Plugins

Use plugins when you want to:

- register a new semantic segmentation architecture,
- register a new optimization objective,
- define paper-specific experimental components,
- keep custom research code isolated from framework internals.

The runtime imports any modules listed in the YAML configuration under `extensions.plugin_modules`, and any registration side effects inside those modules become available automatically.

## Folder Pattern

You can keep the simple two-file layout already included in this repository:

- `plugins/models.py` for `register_model_builder(...)`
- `plugins/objectives.py` for `register_objective(...)`

For larger studies, split them further:

- `plugins/models_baselines.py`
- `plugins/models_ablation.py`
- `plugins/objectives_paper.py`
- `plugins/objectives_deployment.py`

## Register a Custom Model

```python
from tensorflow import keras

from models import register_model_builder


def build_my_model(**kwargs):
    inputs = keras.Input(shape=kwargs["input_shape"])
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    outputs = keras.layers.Conv2D(kwargs["num_classes"], 1, activation=kwargs.get("activation", "sigmoid"))(x)
    return keras.Model(inputs, outputs, name="MyModel")


register_model_builder(
    name="MyModel",
    builder=build_my_model,
    aliases=["my_model"],
    metadata={"family": "paper_baseline"},
)
```

## Register a Custom Objective

```python
from src import register_objective


register_objective(
    name="paper_score",
    direction="max",
    getter=lambda result: (
        0.6 * float(result["summary"].get("final_val_iou", 0.0))
        + 0.2 * float(result["summary"].get("final_val_boundary_iou", 0.0))
        + 0.2 * float(result["summary"].get("final_val_f1_score", result["summary"].get("final_val_dice", 0.0)))
    ),
    description="Weighted score for ranking models in the paper.",
)
```

## Activate Plugins From YAML

```yaml
extensions:
  plugin_modules:
    - plugins.objectives
    - plugins.models
```

## Good Research Practice

- Keep the objective definition close to the paper wording.
- Prefer interpretable objective names like `boundary_quality` or `compactness_penalty`.
- Store composite scores in plugins, not buried in notebooks.
- If an objective is only for ranking, document the formula and rationale in the paper and in the plugin file.

## Current Examples

- `plugins/models.py` registers `TinyPaperUNet`.
- `plugins/objectives.py` registers `paper_quality` and `efficiency_score`.
