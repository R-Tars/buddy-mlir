from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..codegen.python_ttnn import write_python_ttnn_skeleton
from ..semantic.graph import LlamaModelGraph
from ..templates.registry import build_execution_plan, dump_execution_plan
from .space import enumerate_candidate_configs


SUPPORTED_METRICS = {"latency_ms"}


def run_lm_head_search(
    *,
    graph: LlamaModelGraph,
    base_config: dict[str, Any],
    space: dict[str, list[Any]],
    metric: str,
    out: str | Path,
    candidates_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"unsupported search metric {metric!r}; "
            f"expected one of {sorted(SUPPORTED_METRICS)}"
        )

    out_path = Path(out)
    candidate_root = (
        Path(candidates_dir)
        if candidates_dir is not None
        else out_path.parent / f"{out_path.stem}_candidates"
    )
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    for candidate in enumerate_candidate_configs(base_config, space):
        candidate_dir = candidate_root / candidate["id"]
        codegen_dir = candidate_dir / "codegen"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        config_path = candidate_dir / "config.json"
        plan_path = candidate_dir / "execution_plan.json"
        config = candidate["config"]
        plan = build_execution_plan(graph, config)

        _write_json(config_path, config)
        dump_execution_plan(plan, plan_path)
        codegen_paths = write_python_ttnn_skeleton(plan, codegen_dir)

        candidates.append(
            {
                "id": candidate["id"],
                "lm_head_split_count": candidate["lm_head_split_count"],
                "generation_template": candidate["generation_template"],
                "mlp_intermediate_dtype": candidate.get(
                    "mlp_intermediate_dtype"
                ),
                "attention_sdpa_output_memory_config": candidate.get(
                    "attention_sdpa_output_memory_config"
                ),
                "attention_concat_heads_output_memory_config": candidate.get(
                    "attention_concat_heads_output_memory_config"
                ),
                "config": _relative_or_absolute(config_path, out_path.parent),
                "execution_plan": _relative_or_absolute(
                    plan_path, out_path.parent
                ),
                "codegen_dir": _relative_or_absolute(
                    codegen_dir, out_path.parent
                ),
                "codegen_artifacts": {
                    name: _relative_or_absolute(path, out_path.parent)
                    for name, path in sorted(codegen_paths.items())
                },
                "status": (
                    "dry_run_generated"
                    if dry_run
                    else "generated_unmeasured"
                ),
                "metric": None,
            }
        )

    return {
        "schema_version": 1,
        "search": "lm_head_minimal",
        "metric": metric,
        "dry_run": bool(dry_run),
        "candidate_count": len(candidates),
        "candidates_dir": _relative_or_absolute(
            candidate_root, out_path.parent
        ),
        "candidates": candidates,
        "best": None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
