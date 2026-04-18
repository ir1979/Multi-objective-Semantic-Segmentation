"""Final experiment JSON summary writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping


@dataclass
class JSONSummary:
    """Persist complete experiment metadata to JSON."""

    output_path: Path

    def save(self, payload: Mapping[str, object]) -> None:
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, default=str)
