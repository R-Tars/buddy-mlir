from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def load_generated_model(path: Path, ttnn: Any) -> ModuleType:
    module_name = "generated_buddy_ttnn_runtime_model"
    old_ttnn = sys.modules.get("ttnn")
    sys.modules["ttnn"] = ttnn
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load generated model: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        return module
    finally:
        if old_ttnn is None:
            sys.modules.pop("ttnn", None)
        else:
            sys.modules["ttnn"] = old_ttnn


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{name: to_namespace(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value
