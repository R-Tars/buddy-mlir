from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from ..smoke_single_layer_decode import profile_decode_step
from ..templates.lm_head import build_lm_head_split_ranges
from .space import enumerate_candidate_configs


SUPPORTED_DECODE_STEP_METRICS = {"latency_ms"}


def run_decode_step_autotune(
    *,
    program_dir: str | Path,
    space: dict[str, list[Any]],
    out: str | Path,
    layers: int,
    batch_size: int | None = None,
    cache_len: int | None = None,
    metric: str = "latency_ms",
    candidates_dir: str | Path | None = None,
    dry_run: bool = False,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if metric not in SUPPORTED_DECODE_STEP_METRICS:
        raise ValueError(
            f"unsupported decode-step autotune metric {metric!r}; "
            f"expected one of {sorted(SUPPORTED_DECODE_STEP_METRICS)}"
        )
    if layers <= 0:
        raise ValueError("layers must be positive")

    program_root = Path(program_dir)
    out_path = Path(out)
    base_config = json.loads((program_root / "config.json").read_text())
    candidate_root = (
        Path(candidates_dir)
        if candidates_dir is not None
        else out_path.parent / f"{out_path.stem}_candidates"
    )
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    base_template_config = dict(base_config.get("template_config", {}))
    if not base_template_config:
        base_template_config = {
            "lm_head_split_count": base_config["lm_head"]["split_count"],
            "generation_template": base_config["generation"]["template"],
        }

    candidates = []
    best: dict[str, Any] | None = None
    for candidate in enumerate_candidate_configs(base_template_config, space):
        candidate_dir = candidate_root / candidate["id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_config = apply_decode_step_candidate_config(
            base_config,
            candidate,
        )
        config_path = candidate_dir / "config.json"
        model_path = candidate_dir / "model.py"
        report_path = candidate_dir / "profile_report.json"
        _write_json(config_path, candidate_config)
        shutil.copy2(program_root / "model.py", model_path)

        record = {
            "id": candidate["id"],
            "config": _relative_or_absolute(config_path, out_path.parent),
            "model": _relative_or_absolute(model_path, out_path.parent),
            "knobs": _candidate_knobs(candidate),
            "status": "dry_run_planned" if dry_run else "pending",
            "metric": None,
            "profile_report": None,
            "bottleneck_summary": None,
        }
        if not dry_run:
            profile = profile_decode_step(
                out=report_path,
                program_dir=candidate_dir,
                layers=layers,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
            record["status"] = str(profile["status"])
            record["profile_report"] = _relative_or_absolute(
                report_path,
                out_path.parent,
            )
            if profile.get("passed"):
                record["metric"] = float(profile[metric])
                record["bottleneck_summary"] = profile.get(
                    "bottleneck_summary"
                )
                if best is None or float(record["metric"]) < float(best["metric"]):
                    best = dict(record)
        candidates.append(record)

    return {
        "schema_version": 1,
        "search": "decode_step_minimal",
        "metric": metric,
        "dry_run": bool(dry_run),
        "program_dir": str(program_root),
        "layers": layers,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations if trace else 0,
        "candidate_count": len(candidates),
        "candidates_dir": _relative_or_absolute(candidate_root, out_path.parent),
        "candidates": candidates,
        "best": best,
    }


def apply_decode_step_candidate_config(
    base_config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    split_count = int(candidate["lm_head_split_count"])
    generation_template = str(candidate["generation_template"])
    retain_logits = generation_template != "device_argmax_greedy"

    template_config = config.setdefault("template_config", {})
    template_config["lm_head_split_count"] = split_count
    template_config["generation_template"] = generation_template
    template_config["mlp_intermediate_dtype"] = candidate.get(
        "mlp_intermediate_dtype"
    )
    template_config["attention_sdpa_output_memory_config"] = candidate.get(
        "attention_sdpa_output_memory_config"
    )
    template_config["attention_concat_heads_output_memory_config"] = (
        candidate.get("attention_concat_heads_output_memory_config")
    )

    lm_head = config.setdefault("lm_head", {})
    lm_head["split_count"] = split_count
    lm_head["retain_logits"] = retain_logits
    lm_head["program_configs"] = [None] * split_count
    lm_head["splits"] = build_lm_head_split_ranges(
        int(config["vocab_size"]),
        split_count,
    )

    generation = config.setdefault("generation", {})
    generation["template"] = generation_template
    generation["mode"] = "greedy" if not retain_logits else "full_logits"
    generation["retain_logits"] = retain_logits

    if "final" in config and config["final"]:
        config["final"][-1] = generation_template

    mlp = config.setdefault("mlp", {})
    mlp["intermediate_dtype"] = candidate.get("mlp_intermediate_dtype")

    attention = config.setdefault("attention", {})
    attention["sdpa_output_memory_config"] = _memory_config_value(
        candidate.get("attention_sdpa_output_memory_config")
    )
    attention["concat_heads_output_memory_config"] = _memory_config_value(
        candidate.get("attention_concat_heads_output_memory_config")
    )
    return config


def _candidate_knobs(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "lm_head_split_count": candidate["lm_head_split_count"],
        "generation_template": candidate["generation_template"],
        "mlp_intermediate_dtype": candidate.get("mlp_intermediate_dtype"),
        "attention_sdpa_output_memory_config": candidate.get(
            "attention_sdpa_output_memory_config"
        ),
        "attention_concat_heads_output_memory_config": candidate.get(
            "attention_concat_heads_output_memory_config"
        ),
    }


def _memory_config_value(value: Any) -> Any:
    if value in (None, "default"):
        return None
    if value == "l1":
        return "L1_MEMORY_CONFIG"
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
