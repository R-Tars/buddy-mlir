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
            "buddy_cli_supported": True,
            "python_runner": "run_decode.py",
            "python_runner_supported": True,
            "runner_modes": [
                "build",
                "generate",
                "profile",
                "validate",
                "inspect",
                "diagnose",
            ],
            "legacy_mode_mappings": {
                "smoke": "diagnose --stage decode-step",
                "prefill-smoke": "diagnose --stage prefill",
                "profile": "diagnose --stage decode-step",
                "decode-loop": "diagnose --stage decode-loop-legacy",
                "generate": "generate",
                "profile-generate": "profile --mode generate",
                "validate-real": "validate --suite device",
            },
            "dry_run_supported": True,
            "real_weight_validation_supported": True,
            "notes": (
                "run_decode.py forwards to the six-command product CLI and "
                "automatically supplies its package directory as --program-dir."
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
and JSON metadata manifests. The Python runner is a thin product CLI facade:

```bash
export TTNN_DIRECT_PACKAGE_DIR="$PWD"
export TTNN_DIRECT_BUILD="$(cd .. && pwd)"
export TTNN_DIRECT_REPORTS="$TTNN_DIRECT_BUILD/reports/package"
export TTNN_DIRECT_RUNTIME_ARTIFACTS="$TTNN_DIRECT_BUILD/runtime_artifacts"
export TT_METAL_LOGS_PATH="$TTNN_DIRECT_RUNTIME_ARTIFACTS"
mkdir -p "$TTNN_DIRECT_REPORTS" "$TTNN_DIRECT_RUNTIME_ARTIFACTS"
cd "$TTNN_DIRECT_RUNTIME_ARTIFACTS"

python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" inspect
python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" diagnose \\
  --stage decode-step --dry-run \\
  --out "$TTNN_DIRECT_REPORTS/decode_step_smoke.json"
python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" diagnose \\
  --stage prefill --dry-run \\
  --prefill-len 128 --out "$TTNN_DIRECT_REPORTS/prefill_smoke.json"
python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" generate --dry-run \\
  --prefill-len 128 --max-new-tokens 8 \\
  --out "$TTNN_DIRECT_REPORTS/generate.json"
python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" profile --mode generate --dry-run \\
  --prefill-len 128 --max-new-tokens 8 \\
  --out "$TTNN_DIRECT_REPORTS/generate_profile.json"
python "$TTNN_DIRECT_PACKAGE_DIR/run_decode.py" validate --suite dryrun \\
  --out-dir "$TTNN_DIRECT_REPORTS/validation"
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
