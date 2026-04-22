"""Small reusable registry helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, List, Optional, TypeVar


T = TypeVar("T")


def normalize_registry_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(" ", "").strip()


@dataclass
class RegistryEntry(Generic[T]):
    name: str
    value: T
    aliases: tuple[str, ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)


class Registry(Generic[T]):
    """Simple string-keyed registry with aliases and metadata."""

    def __init__(self, label: str):
        self.label = label
        self._entries: Dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        value: T,
        aliases: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        overwrite: bool = False,
    ) -> RegistryEntry[T]:
        names = [normalize_registry_name(name)]
        if aliases:
            names.extend(normalize_registry_name(alias) for alias in aliases)

        if not overwrite:
            for candidate in names:
                if candidate in self._entries:
                    raise ValueError(f"{self.label} '{name}' is already registered")

        entry = RegistryEntry(
            name=name,
            value=value,
            aliases=tuple(aliases or ()),
            metadata=dict(metadata or {}),
        )
        for candidate in names:
            self._entries[candidate] = entry
        return entry

    def get(self, name: str) -> RegistryEntry[T]:
        normalized_name = normalize_registry_name(name)
        if normalized_name not in self._entries:
            available = ", ".join(self.list_names())
            raise KeyError(f"Unknown {self.label} '{name}'. Available: {available}")
        return self._entries[normalized_name]

    def list_names(self) -> List[str]:
        return sorted({entry.name for entry in self._entries.values()})

