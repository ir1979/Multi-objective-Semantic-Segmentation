"""Example custom objectives for paper-ready experiments."""

from src import register_objective


register_objective(
    name="paper_quality",
    direction="max",
    getter=lambda result: (
        0.7 * float(result["summary"].get("final_val_iou", 0.0))
        + 0.3 * float(result["summary"].get("final_val_boundary_iou", 0.0))
    ),
    description="Weighted validation score combining region IoU and boundary IoU.",
    overwrite=True,
)

register_objective(
    name="efficiency_score",
    direction="max",
    getter=lambda result: 1.0 / (1.0 + float(result.get("inference_time", 0.0))),
    description="Inverse-latency score for paper tables and Pareto ranking.",
    overwrite=True,
)
