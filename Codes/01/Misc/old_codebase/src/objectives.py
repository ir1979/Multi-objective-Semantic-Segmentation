"""Objective registry for multi-objective optimization experiments."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


ObjectiveGetter = Callable[[Dict[str, Any]], float]


def _normalize_objective_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").strip()


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of an optimization objective."""

    name: str
    direction: str
    getter: ObjectiveGetter
    aliases: tuple[str, ...] = ()
    description: str = ""

    def extract(self, result: Dict[str, Any]) -> float:
        return float(self.getter(result))

    @property
    def minimize(self) -> bool:
        return self.direction.lower() == "min"

    def to_minimization_value(self, result: Dict[str, Any]) -> float:
        value = self.extract(result)
        return value if self.minimize else -value


_OBJECTIVE_REGISTRY: Dict[str, ObjectiveSpec] = {}


def register_objective(
    name: str,
    getter: ObjectiveGetter,
    direction: str,
    aliases: Optional[Iterable[str]] = None,
    description: str = "",
    overwrite: bool = False,
) -> ObjectiveSpec:
    """Register a new optimization objective."""
    normalized_names = [_normalize_objective_name(name)]
    if aliases:
        normalized_names.extend(_normalize_objective_name(alias) for alias in aliases)

    if not overwrite:
        for registry_name in normalized_names:
            if registry_name in _OBJECTIVE_REGISTRY:
                raise ValueError(f"Objective '{name}' is already registered")

    spec = ObjectiveSpec(
        name=name,
        direction=direction,
        getter=getter,
        aliases=tuple(aliases or ()),
        description=description,
    )
    for registry_name in normalized_names:
        _OBJECTIVE_REGISTRY[registry_name] = spec
    return spec


def get_objective_spec(name: str) -> ObjectiveSpec:
    """Return a registered objective definition."""
    normalized_name = _normalize_objective_name(name)
    if normalized_name not in _OBJECTIVE_REGISTRY:
        available = ", ".join(sorted(list_objectives()))
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return _OBJECTIVE_REGISTRY[normalized_name]


def resolve_objective_specs(names: Optional[Iterable[str]] = None) -> List[ObjectiveSpec]:
    """Resolve objective names into ordered objective specs."""
    if names is None:
        names = ["iou", "f1_score", "param_count", "flops", "inference_time"]

    resolved: List[ObjectiveSpec] = []
    seen = set()
    for name in names:
        spec = get_objective_spec(name)
        if spec.name not in seen:
            resolved.append(spec)
            seen.add(spec.name)
    return resolved


def list_objectives() -> List[str]:
    """List canonical objective names."""
    canonical_names = {spec.name for spec in _OBJECTIVE_REGISTRY.values()}
    return sorted(canonical_names)


def objective_directions(names: Iterable[str]) -> List[bool]:
    """Return whether each objective should be minimized."""
    return [get_objective_spec(name).minimize for name in names]


def _get_nested(result: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in result and result[key] is not None:
            return float(result[key])
    summary = result.get("summary", {})
    for key in keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    history = summary.get("history", {})
    for key in keys:
        if key in history and history[key]:
            return float(history[key][-1])
    return float(default)


def _register_builtin_objectives():
    register_objective(
        "iou",
        lambda result: _get_nested(result, "iou", "final_val_iou", "val_iou_score"),
        direction="max",
        aliases=["iou_score"],
        description="Validation intersection-over-union.",
    )
    register_objective(
        "f1_score",
        lambda result: _get_nested(result, "f1_score", "final_val_dice", "val_dice_score"),
        direction="max",
        aliases=["dice", "dice_score"],
        description="Validation Dice/F1 score.",
    )
    register_objective(
        "precision",
        lambda result: _get_nested(result, "precision", "final_val_precision", "val_precision_score"),
        direction="max",
        description="Validation precision.",
    )
    register_objective(
        "recall",
        lambda result: _get_nested(result, "recall", "final_val_recall", "val_recall_score"),
        direction="max",
        description="Validation recall.",
    )
    register_objective(
        "accuracy",
        lambda result: _get_nested(result, "accuracy", "final_val_accuracy", "val_pixel_accuracy", "val_pixel_accuracy"),
        direction="max",
        aliases=["pixel_accuracy"],
        description="Validation accuracy / pixel accuracy.",
    )
    register_objective(
        "boundary_iou",
        lambda result: _get_nested(result, "boundary_iou", "final_val_boundary_iou", "val_boundary_iou"),
        direction="max",
        description="Validation boundary IoU.",
    )
    register_objective(
        "boundary_f1",
        lambda result: _get_nested(result, "boundary_f1", "final_val_boundary_f1", "val_boundary_f1"),
        direction="max",
        description="Validation boundary F1.",
    )
    register_objective(
        "param_count",
        lambda result: _get_nested(result, "param_count", default=result.get("model_info", {}).get("total_params", 0.0)),
        direction="min",
        aliases=["params"],
        description="Number of model parameters.",
    )
    register_objective(
        "flops",
        lambda result: _get_nested(result, "flops"),
        direction="min",
        description="Estimated floating-point operations.",
    )
    register_objective(
        "inference_time",
        lambda result: _get_nested(result, "inference_time"),
        direction="min",
        aliases=["latency"],
        description="Mean inference latency in milliseconds.",
    )
    register_objective(
        "train_time_seconds",
        lambda result: _get_nested(result, "train_time_seconds"),
        direction="min",
        description="Training runtime in seconds.",
    )


_register_builtin_objectives()

