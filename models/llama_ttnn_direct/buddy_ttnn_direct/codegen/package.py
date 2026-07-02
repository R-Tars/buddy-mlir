from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import ensure_output_dir, write_json, write_text
from .program import PROGRAM_ARTIFACTS


PACKAGE_MANIFEST = "manifest.json"
PACKAGE_ARTIFACTS = (PACKAGE_MANIFEST,) + PROGRAM_ARTIFACTS
PACKAGE_BACKEND = "tenstorrent-ttnn-direct"
PACKAGE_PROGRAM_TYPE = "python-ttnn"


def build_package_manifest(program_dir: str | Path) -> dict[str, Any]:
    root = Path(program_dir)
    _validate_program_dir(root)
    config = json.loads((root / "config.json").read_text())
    semantic_graph = json.loads((root / "semantic_graph.json").read_text())
    execution_plan = json.loads((root / "execution_plan.json").read_text())
    weights_manifest = json.loads((root / "weights_manifest.json").read_text())
    return {
        "schema_version": 1,
        "backend": PACKAGE_BACKEND,
        "program_type": PACKAGE_PROGRAM_TYPE,
        "entrypoint": "model.py",
        "config": "config.json",
        "semantic_graph": "semantic_graph.json",
        "execution_plan": "execution_plan.json",
        "weights_manifest": "weights_manifest.json",
        "run_decode": "run_decode.py",
        "readme": "README.md",
        "model_name": semantic_graph.get("model_name"),
        "mode": execution_plan.get("mode"),
        "num_layers": execution_plan.get("num_layers")
        or len(execution_plan.get("layers", [])),
        "batch_size": execution_plan.get("batch_size"),
        "seq_len": execution_plan.get("seq_len"),
        "max_cache_len": execution_plan.get("max_cache_len"),
        "generation_template": _generation_template(execution_plan),
        "metadata_policy": copy.deepcopy(
            weights_manifest.get("metadata_policy", {})
        ),
        "runtime": {
            "buddy_cli_supported": False,
            "python_runner": "run_decode.py",
            "python_runner_supported": True,
            "runner_modes": [
                "inspect",
                "smoke",
                "profile",
                "validate-real",
            ],
            "dry_run_supported": True,
            "real_weight_validation_supported": True,
            "notes": (
                "buddy-cli runtime dispatch is intentionally not wired; use "
                "run_decode.py for inspect, smoke, profile, and real-weight "
                "validation flows."
            ),
        },
        "artifacts": {
            artifact: artifact
            for artifact in PROGRAM_ARTIFACTS
        },
        "config_summary": {
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "vocab_size": config.get("vocab_size"),
        },
    }


def package_dry_run_report(
    program_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    manifest = build_package_manifest(program_dir)
    return {
        "dry_run": True,
        "backend": manifest["backend"],
        "program_type": manifest["program_type"],
        "program_dir": str(program_dir),
        "out_dir": str(out_dir),
        "artifacts": list(PACKAGE_ARTIFACTS),
        "manifest": manifest,
    }


def package_ttnn_direct_program(
    program_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    source = Path(program_dir)
    target = ensure_output_dir(out_dir)
    manifest = build_package_manifest(source)
    paths = {name: target / name for name in PACKAGE_ARTIFACTS}
    for artifact in PROGRAM_ARTIFACTS:
        shutil.copy2(source / artifact, paths[artifact])
    write_json(paths[PACKAGE_MANIFEST], manifest)
    write_text(
        target / "PACKAGE_README.md",
        render_package_readme(manifest),
    )
    paths["PACKAGE_README.md"] = target / "PACKAGE_README.md"
    return paths


def render_package_readme(manifest: dict[str, Any]) -> str:
    return f"""# Buddy-TTNN Direct Program Package

Backend: `{manifest["backend"]}`
Program type: `{manifest["program_type"]}`
Entrypoint: `{manifest["entrypoint"]}`
Model: `{manifest.get("model_name")}`

This package directory contains a generated Python TTNN Direct decode program
and JSON metadata manifests. The package is not wired into `buddy-cli`, but the
Python runner can inspect and exercise the generated decode path:

```bash
python run_decode.py
python run_decode.py --mode smoke --dry-run --out /tmp/decode_step_smoke.json
python run_decode.py --mode profile --dry-run --out /tmp/decode_step_profile.json
python run_decode.py --mode validate-real --dry-run --require-trace \\
  --min-tokens-per-second-per-user 1.0 --out-dir /tmp/validate_real
```
"""


def _validate_program_dir(root: Path) -> None:
    missing = [
        artifact
        for artifact in PROGRAM_ARTIFACTS
        if not (root / artifact).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "TTNN Direct program directory is missing required artifacts: "
            + ", ".join(missing)
        )


def _generation_template(plan: dict[str, Any]) -> str | None:
    final = plan.get("final", [])
    if "device_argmax_greedy" in final:
        return "device_argmax_greedy"
    if "full_logits" in final:
        return "full_logits"
    return None
