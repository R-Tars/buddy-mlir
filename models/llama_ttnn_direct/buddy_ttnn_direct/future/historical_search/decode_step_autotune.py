from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from ...smoke_single_layer_decode import profile_decode_step
from ...templates.lm_head import build_lm_head_split_ranges
from ...reports.autotune import DECODE_STEP_AUTOTUNE_KNOBS
from .space import enumerate_candidate_configs


DECODE_STEP_METRIC_DIRECTIONS = {
    "latency_ms": "minimize",
    "tokens_per_second_per_user": "maximize",
    "aggregate_tokens_per_second": "maximize",
}
SUPPORTED_DECODE_STEP_METRICS = set(DECODE_STEP_METRIC_DIRECTIONS)
def run_decode_step_autotune(
    *,
    program_dir: str | Path,
    space: dict[str, list[Any]],
    out: str | Path,
    layers: int,
    batch_size: int | None = None,
    cache_len: int | None = None,
    model_path: str | Path | None = None,
    metric: str = "latency_ms",
    candidates_dir: str | Path | None = None,
    dry_run: bool = False,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
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
    model_path_for_profile = Path(model_path) if model_path is not None else None
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
        candidate_model_path = candidate_dir / "model.py"
        report_path = candidate_dir / "profile_report.json"
        _write_json(config_path, candidate_config)
        shutil.copy2(program_root / "model.py", candidate_model_path)
        metadata_paths = _copy_profile_metadata(program_root, candidate_dir)

        record = {
            "id": candidate["id"],
            "config": _relative_or_absolute(config_path, out_path.parent),
            "model": _relative_or_absolute(candidate_model_path, out_path.parent),
            "profile_metadata": [
                _relative_or_absolute(path, out_path.parent)
                for path in metadata_paths
            ],
            "knobs": _candidate_knobs(candidate),
            "status": "dry_run_planned" if dry_run else "pending",
            "passed": None,
            "metric": None,
            "output_kind": _candidate_output_kind(candidate),
            "output_shapes": None,
            "lm_head_profile": None,
            "throughput_summary": None,
            "profile_report": None,
            "bottleneck_summary": None,
            "parameter_source": None,
            "parameter_setup": None,
            "trace_status": None,
            "reference_status": None,
            "reference_kind": None,
            "reference_failed_checks": [],
            "error": None,
        }
        if not dry_run:
            profile = profile_decode_step(
                out=report_path,
                program_dir=candidate_dir,
                layers=layers,
                model_path=model_path_for_profile,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
            record["status"] = str(profile["status"])
            record["passed"] = bool(profile.get("passed"))
            record["parameter_source"] = profile.get("parameter_source")
            record["parameter_setup"] = profile.get("parameter_setup")
            record["input_source"] = profile.get("input_source")
            record["prompt_tokenization"] = profile.get("prompt_tokenization")
            record["trace_status"] = (profile.get("trace") or {}).get("status")
            record.update(_reference_summary(profile))
            record["error"] = profile.get("error")
            record["output_kind"] = str(
                profile.get("output_kind")
                or (profile.get("output") or {}).get("kind")
                or record["output_kind"]
            )
            record["output_shapes"] = profile.get("output_shapes")
            record["lm_head_profile"] = profile.get("lm_head_profile")
            record["throughput_summary"] = profile.get("throughput_summary")
            record["profile_report"] = _relative_or_absolute(
                report_path,
                out_path.parent,
            )
            if profile.get("passed"):
                record["metric"] = _profile_metric_value(profile, metric)
                record["bottleneck_summary"] = profile.get(
                    "bottleneck_summary"
                )
                if best is None or _is_better_metric(
                    float(record["metric"]),
                    float(best["metric"]),
                    metric,
                ):
                    best = dict(record)
        candidates.append(record)

    status_counts = _candidate_field_counts(candidates, "status")
    reference_status_counts = _candidate_field_counts(
        candidates,
        "reference_status",
    )
    trace_status_counts = _candidate_field_counts(candidates, "trace_status")
    output_kind_counts = _candidate_field_counts(candidates, "output_kind")
    knob_coverage = _knob_coverage(candidates)
    leaderboard = _candidate_leaderboard(candidates, metric)
    best_candidate_summary = _best_candidate_summary(best, metric)
    return {
        "schema_version": 1,
        "search": "decode_step_minimal",
        "metric": metric,
        "metric_direction": DECODE_STEP_METRIC_DIRECTIONS[metric],
        "dry_run": bool(dry_run),
        "program_dir": str(program_root),
        "model_path": str(model_path_for_profile) if model_path_for_profile else None,
        "layers": layers,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations if trace else 0,
        "candidate_count": len(candidates),
        "status_counts": status_counts,
        "passed_candidate_count": sum(
            1 for candidate in candidates if candidate.get("passed") is True
        ),
        "failed_candidate_count": sum(
            1 for candidate in candidates if candidate.get("passed") is False
        ),
        "knob_coverage": knob_coverage,
        "search_space": knob_coverage["values"],
        "reference_status_counts": reference_status_counts,
        "trace_status_counts": trace_status_counts,
        "output_kind_counts": output_kind_counts,
        "candidates_dir": _relative_or_absolute(candidate_root, out_path.parent),
        "candidates": candidates,
        "leaderboard": leaderboard,
        "best": best,
        "best_candidate_summary": best_candidate_summary,
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


def _reference_summary(report: dict[str, Any]) -> dict[str, Any]:
    reference = report.get("reference") or {}
    checks = reference.get("checks") or []
    return {
        "reference_status": reference.get("status"),
        "reference_kind": reference.get("kind"),
        "reference_failed_checks": [
            check.get("name")
            for check in checks
            if isinstance(check, dict) and not check.get("passed")
        ],
    }


def _profile_metric_value(profile: dict[str, Any], metric: str) -> float:
    if metric == "latency_ms":
        return float(profile["latency_ms"])
    throughput = profile.get("throughput_summary") or {}
    value = throughput.get(metric)
    if value is None:
        raise ValueError(f"profile report did not produce metric {metric!r}")
    return float(value)


def _is_better_metric(candidate: float, current_best: float, metric: str) -> bool:
    direction = DECODE_STEP_METRIC_DIRECTIONS[metric]
    if direction == "maximize":
        return candidate > current_best
    return candidate < current_best


def _candidate_leaderboard(
    candidates: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: _leaderboard_sort_key(candidate, metric),
    )
    return [
        _candidate_leaderboard_entry(
            candidate,
            metric=metric,
            rank=rank,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]


def _leaderboard_sort_key(
    candidate: dict[str, Any],
    metric: str,
) -> tuple[int, float, str]:
    value = candidate.get("metric")
    if isinstance(value, (int, float)):
        metric_value = float(value)
        if DECODE_STEP_METRIC_DIRECTIONS[metric] == "maximize":
            metric_value = -metric_value
        return (0, metric_value, str(candidate.get("id", "")))
    return (1, 0.0, str(candidate.get("id", "")))


def _candidate_leaderboard_entry(
    candidate: dict[str, Any],
    *,
    metric: str,
    rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": candidate.get("id"),
        "status": candidate.get("status"),
        "passed": candidate.get("passed"),
        "metric": metric,
        "metric_direction": DECODE_STEP_METRIC_DIRECTIONS[metric],
        "metric_value": candidate.get("metric"),
        "knobs": candidate.get("knobs"),
        "output_kind": candidate.get("output_kind"),
        "output_shapes": candidate.get("output_shapes"),
        "profile_report": candidate.get("profile_report"),
        "parameter_source": candidate.get("parameter_source"),
        "reference_status": candidate.get("reference_status"),
        "trace_status": candidate.get("trace_status"),
        "throughput_summary": _compact_throughput_summary(
            candidate.get("throughput_summary")
        ),
        "bottleneck_summary": _compact_bottleneck_summary(
            candidate.get("bottleneck_summary")
        ),
        "lm_head_profile": _compact_lm_head_profile(
            candidate.get("lm_head_profile")
        ),
        "error": candidate.get("error"),
    }


def _best_candidate_summary(
    best: dict[str, Any] | None,
    metric: str,
) -> dict[str, Any] | None:
    if best is None:
        return None
    summary = _candidate_leaderboard_entry(best, metric=metric, rank=1)
    summary["config"] = best.get("config")
    summary["model"] = best.get("model")
    summary["profile_metadata"] = best.get("profile_metadata", [])
    return summary


def _compact_throughput_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    return {
        "status": summary.get("status"),
        "latency_ms": summary.get("latency_ms"),
        "tokens_per_second_per_user": summary.get(
            "tokens_per_second_per_user"
        ),
        "aggregate_tokens_per_second": summary.get(
            "aggregate_tokens_per_second"
        ),
        "trace_execute_mean_ms": summary.get("trace_execute_mean_ms"),
        "trace_execute_tokens_per_second_per_user": summary.get(
            "trace_execute_tokens_per_second_per_user"
        ),
        "trace_execute_aggregate_tokens_per_second": summary.get(
            "trace_execute_aggregate_tokens_per_second"
        ),
        "trace_iterations": summary.get("trace_iterations"),
    }


def _compact_bottleneck_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    return {
        "max_section": summary.get("max_section"),
        "max_section_ms": summary.get("max_section_ms"),
        "sections_ms": summary.get("sections_ms"),
    }


def _compact_lm_head_profile(profile: Any) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    return {
        "split_count": profile.get("split_count"),
        "lm_head_ms": profile.get("lm_head_ms"),
        "argmax_ms": profile.get("argmax_ms"),
        "argmax_status": profile.get("argmax_status"),
    }


def _candidate_field_counts(
    candidates: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        value = candidate.get(field)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _knob_coverage(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        knob: _unique_knob_values(candidates, knob)
        for knob in DECODE_STEP_AUTOTUNE_KNOBS
    }
    value_counts = {
        knob: _knob_value_counts(candidates, knob)
        for knob in DECODE_STEP_AUTOTUNE_KNOBS
    }
    varied_knobs = [
        knob
        for knob, knob_values in values.items()
        if len(knob_values) > 1
    ]
    missing_varied_knobs = [
        knob for knob in DECODE_STEP_AUTOTUNE_KNOBS if knob not in varied_knobs
    ]
    return {
        "knobs": list(DECODE_STEP_AUTOTUNE_KNOBS),
        "candidate_count": len(candidates),
        "values": values,
        "value_counts": value_counts,
        "varied_knobs": varied_knobs,
        "required_varied_knobs": list(DECODE_STEP_AUTOTUNE_KNOBS),
        "missing_varied_knobs": missing_varied_knobs,
        "all_knobs_varied": not missing_varied_knobs,
    }


def _unique_knob_values(
    candidates: list[dict[str, Any]],
    knob: str,
) -> list[Any]:
    values = []
    for candidate in candidates:
        value = _record_knob_value(candidate, knob)
        if value not in values:
            values.append(value)
    return sorted(values, key=_knob_sort_key)


def _knob_value_counts(
    candidates: list[dict[str, Any]],
    knob: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        label = _knob_value_label(_record_knob_value(candidate, knob))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _record_knob_value(candidate: dict[str, Any], knob: str) -> Any:
    knobs = candidate.get("knobs")
    if isinstance(knobs, dict) and knob in knobs:
        return knobs.get(knob)
    return candidate.get(knob)


def _knob_sort_key(value: Any) -> tuple[int, float, str]:
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), str(value))
    return (2, 0.0, str(value))


def _knob_value_label(value: Any) -> str:
    return "null" if value is None else str(value)


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


def _candidate_output_kind(candidate: dict[str, Any]) -> str:
    return (
        "token"
        if candidate["generation_template"] == "device_argmax_greedy"
        else "logits"
    )


def _copy_profile_metadata(program_root: Path, candidate_dir: Path) -> list[Path]:
    copied = []
    for filename in (
        "semantic_graph.json",
        "weights_manifest.json",
        "execution_plan.json",
    ):
        source = program_root / filename
        if not source.is_file():
            continue
        destination = candidate_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


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
