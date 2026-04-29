"""Runtime plugin loading for custom experiment components."""

import importlib
import logging
from typing import Any, Dict, Iterable, List


logger = logging.getLogger(__name__)


def _normalized_module_list(module_names: Iterable[str]) -> List[str]:
    ordered = []
    seen = set()
    for module_name in module_names:
        if not module_name:
            continue
        module_name = str(module_name).strip()
        if not module_name or module_name in seen:
            continue
        seen.add(module_name)
        ordered.append(module_name)
    return ordered


def load_plugin_modules(module_names: Iterable[str]) -> List[str]:
    """Import plugin modules once so their registration side effects run."""
    loaded = []
    for module_name in _normalized_module_list(module_names):
        importlib.import_module(module_name)
        loaded.append(module_name)
        logger.info("Loaded plugin module: %s", module_name)
    return loaded


def load_plugins_from_config(config: Dict[str, Any]) -> List[str]:
    """Load plugin modules declared in the experiment config."""
    extensions = config.get("extensions", {})
    plugin_modules = extensions.get("plugin_modules", [])
    return load_plugin_modules(plugin_modules)
