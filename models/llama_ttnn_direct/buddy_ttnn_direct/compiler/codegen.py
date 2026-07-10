from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..codegen.artifacts import (
    ensure_output_dir,
    planned_artifact_paths,
    write_json,
    write_text,
)
from .config import build_codegen_config, validate_execution_plan_for_codegen
from .source_templates import render_codegen_readme, render_python_ttnn_model


def write_python_ttnn_skeleton(
    plan: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    validate_execution_plan_for_codegen(plan)
    root = ensure_output_dir(out_dir)
    paths = planned_artifact_paths(root)
    write_text(paths["model.py"], render_python_ttnn_model(plan))
    write_json(paths["config.json"], build_codegen_config(plan))
    write_json(paths["plan.json"], copy.deepcopy(plan))
    write_text(paths["README.md"], render_codegen_readme(plan))
    return paths


def dry_run_report(plan: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    config = build_codegen_config(plan)
    return {
        "dry_run": True,
        "model_name": config["model_name"],
        "num_layers": config["num_layers"],
        "out_dir": str(out_dir),
        "artifacts": sorted(planned_artifact_paths(out_dir)),
    }
