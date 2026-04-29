"""Experiment registry persisted as JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentRegistry:
    """Track experiment statuses and metadata."""

    registry_path: Path = Path("results/experiment_registry.json")
    _data: Dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.registry_path = Path(self.registry_path)
        if self.registry_path.exists():
            self._data = self.load()

    def register(self, name: str, config_path: str) -> None:
        self._data.setdefault(
            name,
            {
                "status": "pending",
                "config_path": config_path,
                "results_path": "",
                "started_at": "",
                "completed_at": "",
                "test_iou": 0.0,
                "error_message": None,
                "failure_log": "",
                "resume_count": 0,
            },
        )
        self.save()

    def update_status(self, name: str, status: str, **kwargs: object) -> None:
        payload = self._data.setdefault(name, {})
        payload["status"] = status
        if status == "running":
            payload["started_at"] = datetime.utcnow().isoformat()
        if status in {"completed", "failed"}:
            payload["completed_at"] = datetime.utcnow().isoformat()
        payload.update(kwargs)
        self.save()

    def get_completed(self) -> List[str]:
        return [name for name, payload in self._data.items() if payload.get("status") == "completed"]

    def get_failed(self) -> List[str]:
        return [name for name, payload in self._data.items() if payload.get("status") == "failed"]

    def load(self) -> Dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)

