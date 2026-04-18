import os
import numpy as np
import tensorflow as tf

from data.dataset import BuildingDataset, resolve_dataset_path, create_datasets_from_config
from models import build_model, register_model_builder
from losses.loss_manager import build_losses
from src.objectives import register_objective, resolve_objective_specs
from training.metrics import (
    boundary_f1,
    boundary_iou,
    compactness_score,
    pixel_accuracy,
    region_completeness,
    topological_correctness,
)
from utils.repro import set_global_seed


def test_build_model_shapes():
    model = build_model("unet", input_shape=(128, 128, 3), num_classes=1)
    assert model.output_shape == (None, 128, 128, 1)
    model2 = build_model("unet++", input_shape=(128, 128, 3), num_classes=1)
    assert model2.output_shape[1:] == (128, 128, 1)


def test_loss_manager_returns_functions():
    config = {"pixel_loss": "bce+iou", "boundary_loss": "hausdorff", "shape_loss": "convexity"}
    losses, weights, names = build_losses(config)
    assert len(losses) >= 1
    assert len(weights) == len(losses)
    assert len(names) == len(losses)
    x = np.zeros((1, 128, 128, 1), dtype=np.float32)
    y = np.zeros((1, 128, 128, 1), dtype=np.float32)
    for loss_fn in losses:
        value = loss_fn(tf.constant(y), tf.constant(x))
        assert value.shape == ()


def test_dataset_sequence_length():
    rgb_paths = [f"rgb_{i}.png" for i in range(8)]
    mask_paths = [f"mask_{i}.png" for i in range(8)]
    ds = BuildingDataset(rgb_paths, mask_paths, batch_size=4, shuffle=False)
    assert len(ds) == 2


def test_dataset_sequence_keeps_partial_batch():
    rgb_paths = [f"rgb_{i}.png" for i in range(10)]
    mask_paths = [f"mask_{i}.png" for i in range(10)]
    ds = BuildingDataset(rgb_paths, mask_paths, batch_size=4, shuffle=False)
    assert len(ds) == 3


def test_resolve_dataset_path_handles_case_mismatch():
    resolved = resolve_dataset_path("Datasets/RGB")
    assert resolved in {"Datasets/RGB", "datasets/RGB"}


def test_register_custom_model_builder():
    def build_tiny_model(**kwargs):
        inputs = tf.keras.Input(shape=kwargs["input_shape"])
        outputs = tf.keras.layers.Conv2D(kwargs["num_classes"], 1, activation="sigmoid")(inputs)
        return tf.keras.Model(inputs, outputs)

    register_model_builder("TinyRegistryModel", build_tiny_model, overwrite=True)
    model = build_model("TinyRegistryModel", input_shape=(32, 32, 3), num_classes=1)
    assert model.output_shape == (None, 32, 32, 1)


def test_register_custom_objective():
    register_objective(
        "dummy_objective",
        getter=lambda result: result["summary"]["custom_value"],
        direction="max",
        overwrite=True,
    )
    spec = resolve_objective_specs(["dummy_objective"])[0]
    assert spec.extract({"summary": {"custom_value": 0.75}}) == 0.75


def test_additional_metrics_return_scalars():
    y_true = np.zeros((1, 128, 128, 1), dtype=np.float32)
    y_pred = np.zeros((1, 128, 128, 1), dtype=np.float32)
    y_pred[0, 32:96, 32:96, 0] = 1.0
    y_true[0, 32:96, 32:96, 0] = 1.0

    funcs = [
        pixel_accuracy,
        region_completeness,
        boundary_iou,
        boundary_f1,
        compactness_score,
        topological_correctness,
    ]

    for func in funcs:
        value = func(tf.constant(y_true), tf.constant(y_pred))
        assert value.shape == ()
        assert np.isfinite(value.numpy())


def test_set_global_seed_deterministic():
    set_global_seed(1234)
    assert tf.random.uniform([1]).numpy().shape == (1,)
