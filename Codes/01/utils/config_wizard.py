"""Helpers for building, loading, and saving experiment configs from a wizard UI."""

from __future__ import annotations

from copy import deepcopy
from math import prod
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import yaml

from utils.config_loader import ConfigValidator, config_to_flat_dict, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
TEMPLATE_PATHS = {
    "single": CONFIG_DIR / "default.yaml",
    "grid_search": CONFIG_DIR / "grid_search.yaml",
}

MODEL_ARCHITECTURES = [
    "unet",
    "unetpp",
    "attunet",
    "r2attunet",
    "seunet",
    "scse_unet",
    "resunet",
    "resunetpp",
]
PIXEL_LOSSES = ["bce", "iou", "dice", "focal"]
LOSS_STRATEGIES = ["single", "weighted"]
SEARCH_STRATEGIES = ["grid_search", "random", "latin_hypercube", "nsga2"]
SCHEDULERS = ["cosine", "step", "plateau"]
ENCODER_FILTER_PRESETS = {
    "deep": [64, 128, 256, 512, 1024],
    "shallow": [32, 64, 128, 256, 512],
    "micro": [16, 32, 64, 128, 256],
}
SUPPORTED_GRID_PARAMETERS = {
    "model_architecture": ("model", "architecture"),
    "encoder_filters": ("model", "encoder_filters"),
    "pixel_loss_type": ("loss", "pixel", "type"),
    "boundary_loss_weight": ("loss", "boundary", "weight"),
    "shape_loss_weight": ("loss", "shape", "weight"),
    "learning_rate": ("training", "learning_rate"),
}
GRID_PARAMETER_STATE_KEYS = {
    "model_architecture": "grid_include_model_architecture",
    "encoder_filters": "grid_include_encoder_filters",
    "pixel_loss_type": "grid_include_pixel_loss_type",
    "boundary_loss_weight": "grid_include_boundary_loss_weight",
    "shape_loss_weight": "grid_include_shape_loss_weight",
    "learning_rate": "grid_include_learning_rate",
}
ZERO_LOSS_CONSTRAINT = [
    {
        "if_boundary_loss_weight": 0.0,
        "and_shape_loss_weight": 0.0,
        "then_skip": True,
    }
]

# Keys that must always be written to the saved YAML even when they equal defaults.
REQUIRED_KEYS: frozenset[str] = frozenset({
    "project_name",
    "data_rgb_dir",
    "data_mask_dir",
    "data_train_ratio",
    "data_val_ratio",
    "data_test_ratio",
    "model_architecture",
    "loss_strategy",
    "loss_pixel_type",
    "training_epochs",
    "training_learning_rate",
    "training_lr_scheduler_type",
})


def _read_yaml_mapping(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def parse_scalar_list(text: str, cast: type = str) -> List[Any]:
    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    if cast is str:
        return [part.strip("\"'") for part in parts]
    return [cast(part) for part in parts]


def parse_constraints_text(text: str) -> List[str]:
    """Parse a newline- or comma-separated list of constraint expressions into a list of strings."""
    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return parts


def parse_nested_list_text(text: str) -> List[List[Any]]:
    raw_text = text.strip()
    if not raw_text:
        return []
    # Try wrapping all content in an outer list so YAML can parse multiple flow-sequences
    wrapped = f"[{raw_text}]" if not raw_text.startswith("[[") else raw_text
    try:
        loaded = yaml.safe_load(wrapped)
        if isinstance(loaded, list):
            if loaded and isinstance(loaded[0], list):
                return loaded
            return [loaded]
    except yaml.YAMLError:
        pass
    # Fallback: parse line-by-line
    result = []
    for ln in [ln.strip() for ln in raw_text.splitlines() if ln.strip()]:
        try:
            parsed = yaml.safe_load(ln)
            if isinstance(parsed, list):
                result.append(parsed)
        except yaml.YAMLError:
            pass
    return result if result else []



def get_template_path(mode: str) -> str:
    key = "grid_search" if "grid" in mode else "single"
    path = TEMPLATE_PATHS.get(key)
    if path is None:
        raise FileNotFoundError(f"No template path registered for mode '{mode}'")
    return str(path)


def config_to_wizard_state(resolved: dict, raw: dict, path: str) -> dict:
    state: Dict[str, Any] = {"mode": "single", "grid_enabled": False}
    state.update(resolved)

    # Reconstruct text-widget fields from list values so the GUI is populated.
    if "model_encoder_filters" in resolved:
        val = resolved["model_encoder_filters"]
        state["model_encoder_filters_text"] = str(val) if isinstance(val, list) else str(val)
    if "evaluation_metrics" in resolved:
        val = resolved["evaluation_metrics"]
        state["evaluation_metrics_text"] = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
    if "evaluation_complexity_metrics" in resolved:
        val = resolved["evaluation_complexity_metrics"]
        state["evaluation_complexity_metrics_text"] = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
    if "visualization_formats" in resolved:
        val = resolved["visualization_formats"]
        state["visualization_formats_text"] = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)

    # Map flattened scheduler keys back to wizard state keys.
    for flat_key, state_key in [
        ("training_lr_scheduler_type", "training_scheduler_type"),
        ("training_lr_scheduler_warmup_epochs", "training_scheduler_warmup_epochs"),
        ("training_lr_scheduler_min_lr", "training_scheduler_min_lr"),
    ]:
        if flat_key in resolved:
            state[state_key] = resolved[flat_key]

    # Reconstruct grid search dimension text fields.
    grid_params = {k: v for k, v in resolved.items() if k.startswith("grid_search_parameters_")}
    if grid_params:
        state["grid_enabled"] = True
        state["mode"] = "grid_search"
        for k, v in grid_params.items():
            param = k.replace("grid_search_parameters_", "")
            state[f"grid_{param}_text"] = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            state[f"grid_include_{param}"] = True
    return state
def default_wizard_state(mode: str = "single") -> Dict[str, Any]:
    # Always build from the single template first so every wizard field has a value.
    single_path = get_template_path("single")
    single_raw = _read_yaml_mapping(single_path)
    single_resolved = load_config(str(single_path))
    state = config_to_wizard_state(single_resolved, single_raw, single_path)
    if mode == "grid_search":
        gs_path = get_template_path("grid_search")
        gs_raw = _read_yaml_mapping(gs_path)
        gs_resolved = load_config(str(gs_path))
        gs_state = config_to_wizard_state(gs_resolved, gs_raw, gs_path)
        state.update(gs_state)   # grid values win where present
    return state


def load_wizard_state(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config into wizard state without failing on validation errors."""
    raw = _read_yaml_mapping(config_path)
    # Flatten without validation so broken/partial configs can still populate the GUI.
    resolved = config_to_flat_dict(raw)
    return config_to_wizard_state(resolved, raw, str(config_path))


def _assemble_config(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the config dict from wizard state without running validation."""
    mode = "grid_search" if str(state.get("mode", "single")).lower() == "grid_search" else "single"
    try:
        template = load_config(str(get_template_path(mode)))
    except Exception:
        template = {}
    config = deepcopy(template)

    config['project_name'] = str(state.get("project_name", "experiment"))
    config['project_seed'] = int(state.get("project_seed", 42))
    config['project_deterministic'] = bool(state.get("project_deterministic", True))

    config['data_rgb_dir'] = str(state.get("data_rgb_dir", "datasets/RGB"))
    config['data_mask_dir'] = str(state.get("data_mask_dir", "datasets/Mask"))
    config['data_image_size'] = int(state.get("data_image_size", 256))
    config['data_batch_size'] = int(state.get("data_batch_size", 4))
    config['data_train_ratio'] = float(state.get("data_train_ratio", 0.7))
    config['data_val_ratio'] = float(state.get("data_val_ratio", 0.15))
    config['data_test_ratio'] = float(state.get("data_test_ratio", 0.15))

    model_filters = parse_nested_list_text(str(state.get("model_encoder_filters_text", "")))
    selected_filters = model_filters[0] if model_filters else list(ENCODER_FILTER_PRESETS["deep"])
    config['model_architecture'] = str(state.get("model_architecture", "unet"))
    config['model_encoder_filters'] = selected_filters
    config['model_dropout_rate'] = float(state.get("model_dropout_rate", 0.3))
    config['model_batch_norm'] = bool(state.get("model_batch_norm", True))
    config['model_deep_supervision'] = bool(state.get("model_deep_supervision", False))

    config['loss_strategy'] = str(state.get("loss_strategy", "single"))
    config['loss_pixel_type'] = str(state.get("loss_pixel_type", "bce"))
    config['loss_pixel_weight'] = float(state.get("loss_pixel_weight", 1.0))
    config['loss_boundary_enabled'] = bool(state.get("loss_boundary_enabled", False))
    config['loss_boundary_weight'] = float(state.get("loss_boundary_weight", 0.0))
    config['loss_boundary_distance_threshold'] = int(state.get("loss_boundary_distance_threshold", 3),
    )
    config['loss_shape_enabled'] = bool(state.get("loss_shape_enabled", False))
    config['loss_shape_weight'] = float(state.get("loss_shape_weight", 0.0))


    config['training_epochs'] = int(state.get("training_epochs", 10))
    config['training_learning_rate'] = float(state.get("training_learning_rate", 1.0e-4))
    config['training_lr_scheduler_type'] = str(state.get("training_scheduler_type", "cosine"))
    config['training_lr_scheduler_warmup_epochs'] = int(state.get("training_scheduler_warmup_epochs", 2),
    )
    config['training_lr_scheduler_min_lr'] = float(state.get("training_scheduler_min_lr", 1.0e-7),
    )
    config['training_early_stopping_enabled'] = bool(state.get("training_early_stopping_enabled", True),
    )
    config['training_early_stopping_patience'] = int(state.get("training_early_stopping_patience", 5),
    )

    config['evaluation_metrics'] = parse_scalar_list(str(state.get("evaluation_metrics_text", "")),
    )
    config['evaluation_complexity_metrics'] = parse_scalar_list(str(state.get("evaluation_complexity_metrics_text", "")),
    )
    config['visualization_formats'] = parse_scalar_list(str(state.get("visualization_formats_text", "")),
    )
    config['export_results_dir'] = str(state.get("export_results_dir", "results"))

    if mode == "grid_search" or bool(state.get("grid_enabled", False)):
        grid_cfg = config.setdefault("grid_search", {})
        grid_cfg["enabled"] = bool(state.get("grid_enabled", True))
        grid_cfg["auto_checkpoint"] = bool(state.get("grid_auto_checkpoint", True))
        grid_cfg["replicate_points"] = int(state.get("grid_replicate_points", 1))
        grid_cfg["intermediate_reporting"] = bool(state.get("grid_intermediate_reporting", True))
        grid_cfg["selection"] = {
            "strategy": str(state.get("grid_selection_strategy", "grid_search")),
            "n_points": int(state.get("grid_selection_n_points", 36)),
            "random_seed": int(state.get("grid_selection_random_seed", 42)),
        }
        grid_cfg["persistence"] = {
            "backend": str(state.get("grid_persistence_backend", "json")),
            "checkpoint_interval": int(state.get("grid_checkpoint_interval", 1)),
        }
        parameters: Dict[str, Any] = {}
        if bool(state.get("grid_include_model_architecture", True)):
            parameters["model_architecture"] = parse_scalar_list(str(state.get("grid_model_architecture_text", "")))
        if bool(state.get("grid_include_encoder_filters", True)):
            parameters["encoder_filters"] = parse_nested_list_text(str(state.get("grid_encoder_filters_text", "")))
        if bool(state.get("grid_include_pixel_loss_type", True)):
            parameters["pixel_loss_type"] = parse_scalar_list(str(state.get("grid_pixel_loss_type_text", "")))
        if bool(state.get("grid_include_boundary_loss_weight", True)):
            parameters["boundary_loss_weight"] = parse_scalar_list(
                str(state.get("grid_boundary_loss_weight_text", "")), float
            )
        if bool(state.get("grid_include_shape_loss_weight", True)):
            parameters["shape_loss_weight"] = parse_scalar_list(str(state.get("grid_shape_loss_weight_text", "")), float)
        if bool(state.get("grid_include_learning_rate", True)):
            parameters["learning_rate"] = parse_scalar_list(str(state.get("grid_learning_rate_text", "")), float)
        if not parameters:
            raise ValueError("Grid search is enabled, but no search dimensions are selected.")
        grid_cfg["parameters"] = parameters
        grid_cfg["constraints"] = parse_constraints_text(str(state.get("grid_constraints_text", "")))
    else:
        config.pop("grid_search", None)

    return config


def build_full_config(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble and validate the config. Raises on validation errors."""
    config = _assemble_config(state)
    ConfigValidator().validate(config)
    return config


def build_best_effort_config(state: Mapping[str, Any]) -> tuple[Dict[str, Any], str]:
    """Assemble config from state, returning (config, error_comment).
    Never raises — if validation fails the error is returned as a YAML comment string."""
    try:
        config = _assemble_config(state)
    except Exception as exc:
        return {k: v for k, v in state.items() if isinstance(k, str) and not k.startswith("_")}, \
            f"# BUILD ERROR: {exc}\n"
    error_comment = ""
    try:
        ConfigValidator().validate(config)
    except Exception as exc:
        error_comment = f"# VALIDATION ERROR: {exc}\n# Fix the issues above before running training.\n\n"
    return config, error_comment


def build_save_payload(state: Mapping[str, Any], output_path: str | Path | None = None) -> Dict[str, Any]:
    return build_full_config(state)


def save_payload_to_file(payload: Mapping[str, Any], path: str | Path, error_comment: str = "") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(dict(payload), sort_keys=False)
    with output_path.open("w", encoding="utf-8") as handle:
        if error_comment:
            handle.write(error_comment)
        handle.write(yaml_text)


def estimate_grid_point_count(state: Mapping[str, Any]) -> int:
    if not bool(state.get("grid_enabled", False)):
        return 0
    counts = []
    if bool(state.get("grid_include_model_architecture", True)):
        counts.append(len(parse_scalar_list(str(state.get("grid_model_architecture_text", "")))))
    if bool(state.get("grid_include_encoder_filters", True)):
        counts.append(len(parse_nested_list_text(str(state.get("grid_encoder_filters_text", "")))))
    if bool(state.get("grid_include_pixel_loss_type", True)):
        counts.append(len(parse_scalar_list(str(state.get("grid_pixel_loss_type_text", "")))))
    if bool(state.get("grid_include_boundary_loss_weight", True)):
        counts.append(len(parse_scalar_list(str(state.get("grid_boundary_loss_weight_text", "")), float)))
    if bool(state.get("grid_include_shape_loss_weight", True)):
        counts.append(len(parse_scalar_list(str(state.get("grid_shape_loss_weight_text", "")), float)))
    if bool(state.get("grid_include_learning_rate", True)):
        counts.append(len(parse_scalar_list(str(state.get("grid_learning_rate_text", "")), float)))
    if not counts:
        return 0
    return prod(max(1, count) for count in counts)


def build_hint_summary(state: Mapping[str, Any]) -> str:
    hints: List[str] = []
    hints.append(
        "Available model architectures: " + ", ".join(MODEL_ARCHITECTURES)
    )
    hints.append(
        "Available loss strategies: " + ", ".join(LOSS_STRATEGIES)
    )
    hints.append(
        "Available pixel losses: " + ", ".join(PIXEL_LOSSES)
    )
    hints.append(
        "Available schedulers: " + ", ".join(SCHEDULERS)
    )
    hints.append(
        "Search-space dimensions supported by the current grid runner: model_architecture, encoder_filters, pixel_loss_type, boundary_loss_weight, shape_loss_weight, learning_rate."
    )
    total_ratio = (
        float(state.get("data_train_ratio", 0.0))
        + float(state.get("data_val_ratio", 0.0))
        + float(state.get("data_test_ratio", 0.0))
    )
    if abs(total_ratio - 1.0) > 1e-6:
        hints.append(f"Data split ratios currently sum to {total_ratio:.3f}; they must sum to 1.0.")
    if not bool(state.get("loss_boundary_enabled", False)) and not bool(state.get("loss_shape_enabled", False)):
        hints.append("Boundary and shape losses are both disabled, so the experiment reduces to pixel loss only.")
    if bool(state.get("model_deep_supervision", False)) and str(state.get("model_architecture", "")).lower() != "unetpp":
        hints.append("Deep supervision is only meaningful for UNet++; other architectures will ignore it.")
    hints.append(
        "The model section defines the base training configuration. In single mode it is the exact model used. In grid-search mode it becomes the default/fallback, and any selected search dimension overrides it per sampled point."
    )
    if bool(state.get("grid_enabled", False)):
        grid_points = estimate_grid_point_count(state)
        active_dimensions = [
            name for name, include_key in GRID_PARAMETER_STATE_KEYS.items() if bool(state.get(include_key, True))
        ]
        hints.append(
            "Active search dimensions: " + (", ".join(active_dimensions) if active_dimensions else "none")
        )
        hints.append(f"The current search space expands to {grid_points} grid points before constraints or sampling.")
        if not active_dimensions:
            hints.append("Select at least one search dimension before saving a grid-search config.")
        if grid_points > 200:
            hints.append("This is a large sweep. Consider random search or fewer architecture/filter combinations.")
        strategy = str(state.get("grid_selection_strategy", "grid_search")).lower()
        if strategy in {"random", "latin_hypercube"}:
            hints.append(
                f"Selection strategy '{strategy}' will subsample the generated candidates using n_points={int(state.get('grid_selection_n_points', 36))}."
            )
    architecture = str(state.get("model_architecture", "unet")).lower()
    selected_filters = parse_nested_list_text(str(state.get("model_encoder_filters_text", "")))
    if selected_filters and architecture in {"r2attunet", "attunet", "scse_unet", "resunetpp"}:
        if selected_filters[0] == ENCODER_FILTER_PRESETS["deep"]:
            hints.append("This architecture/filter combination is memory-heavy and may exceed a 12 GB GPU.")
    return "\n".join(hints) if hints else "Configuration looks internally consistent."


def minimize_config(config: Dict, mode: str = "single") -> Dict:
    """Return *config* with keys that equal the template default removed.

    ``REQUIRED_KEYS`` and keys absent from the template are always kept.
    The nested ``grid_search`` key is kept intact — its sub-keys are not diffed.
    """
    template_key = "grid_search" if "grid" in mode else "single"
    template_path = TEMPLATE_PATHS.get(template_key)
    defaults: Dict[str, Any] = {}
    if template_path is not None and template_path.exists():
        try:
            defaults = load_config(str(template_path))
        except Exception:
            pass
    minimized: Dict[str, Any] = {}
    for k, v in config.items():
        if k == "grid_search":
            minimized[k] = v           # keep nested section as-is
        elif k in REQUIRED_KEYS:
            minimized[k] = v           # always required
        elif k not in defaults or defaults[k] != v:
            minimized[k] = v           # differs from default — keep
        # else: equals default — omit
    return minimized


def build_file_header(config: Dict) -> str:
    """Return a multi-line YAML comment block summarising the experiment."""
    from datetime import datetime
    from math import prod

    W = 58
    sep = "# " + "\u2500" * W
    NA = "\u2014"  # em-dash used as "not set" placeholder

    def _line(label: str, value: object) -> str:
        return f"#  {label:<22}{value}"

    lines = [sep, "# Experiment Configuration"]
    lines.append(_line("Generated  :", datetime.now().strftime("%Y-%m-%d %H:%M")))
    lines.append(_line("Project    :", config.get("project_name", NA)))
    lines.append(_line("Seed       :", config.get("project_seed", NA)))
    lines.append("#")

    lines.append("#  Data")
    img_px = config.get("data_image_size", NA)
    lines.append(_line("  Image size  :", f"{img_px} px"))
    lines.append(_line("  Batch size  :", config.get("data_batch_size", NA)))
    train = config.get("data_train_ratio", "?")
    val   = config.get("data_val_ratio",   "?")
    test  = config.get("data_test_ratio",  "?")
    lines.append(_line("  Split       :", f"train {train} / val {val} / test {test}"))
    lines.append("#")

    lines.append("#  Model")
    lines.append(_line("  Architecture:", config.get("model_architecture", NA)))
    lines.append(_line("  Filters     :", config.get("model_encoder_filters", NA)))
    lines.append("#")

    lines.append("#  Loss")
    lines.append(_line("  Strategy    :", config.get("loss_strategy", NA)))
    lines.append(_line("  Pixel loss  :", config.get("loss_pixel_type", NA)))
    boundary_s = "on" if config.get("loss_boundary_enabled") else "off"
    shape_s    = "on" if config.get("loss_shape_enabled")    else "off"
    lines.append(_line("  Boundary    :", f"{boundary_s}  Shape: {shape_s}"))
    lines.append("#")

    lines.append("#  Training")
    lines.append(_line("  Epochs      :", config.get("training_epochs", NA)))
    lines.append(_line("  Learning rate:", config.get("training_learning_rate", NA)))
    lines.append(_line("  Scheduler   :", config.get("training_lr_scheduler_type", NA)))
    lines.append("#")

    # Grid search — config from _assemble_config keeps this as a nested dict.
    gs = config.get("grid_search")
    if isinstance(gs, dict):
        params = gs.get("parameters", {})
        if params:
            lines.append("#  Grid Search")
            dim_names = sorted(params.keys())
            lines.append(_line("  Dimensions  :", ", ".join(dim_names)))
            counts = [len(v) for v in params.values() if isinstance(v, list)]
            total = prod(counts) if counts else 0
            lines.append(_line("  Grid points :", f"{total}  (before constraints)"))
            sel      = gs.get("selection", {})
            strategy = sel.get("strategy", "\u2014")
            lines.append(_line("  Strategy    :", strategy))
            if strategy in {"random", "latin_hypercube"}:
                lines.append(_line("  Sampled pts :", sel.get("n_points", "\u2014")))
            lines.append("#")

    lines.append(sep)
    lines.append("")
    return "\n".join(lines) + "\n"


def preview_yaml(state: Mapping[str, Any]) -> str:
    """Always returns a YAML string with a file header and optional error comment."""
    payload, error_comment = build_best_effort_config(state)
    mode = "grid_search" if bool(state.get("grid_enabled")) else "single"
    minimized = minimize_config(payload, mode)
    header = build_file_header(payload)
    return header + (error_comment or "") + yaml.safe_dump(dict(minimized), sort_keys=False)