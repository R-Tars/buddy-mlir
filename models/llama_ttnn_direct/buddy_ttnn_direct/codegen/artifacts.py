from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GENERATED_ARTIFACTS = ("model.py", "config.json", "plan.json", "README.md")


def planned_artifact_paths(out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    return {name: root / name for name in GENERATED_ARTIFACTS}


def ensure_output_dir(out_dir: str | Path) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text)
