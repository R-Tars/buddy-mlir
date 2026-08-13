from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..autotune.measurement import (
    MeasurementContract,
    file_sha256,
    sha256_json,
    validate_decode_profile_contract,
)
from ..autotune.templates import (
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    KV_UPDATE_AXIS,
    ROPE_AXIS,
    SEPARATE_PAGED_UPDATE,
    SEPARATE_QK_ROPE,
    template_choice_from_runtime_config,
)
from ..runtime.reports import write_report

SCHEMA_VERSION = 1
_PROFILER_MEASUREMENT = MeasurementContract(
    warmup=0,
    iterations=1,
    repetitions=1,
    kind="profiler_audit_capture",
)
REGION_ORDER = (
    "embedding",
    "attention_rmsnorm",
    "qkv_linear",
    "create_heads",
    "rope",
    "kv_update",
    "sdpa",
    "concat_heads",
    "o_projection",
    "residual",
    "mlp_rmsnorm",
    "gate_linear",
    "up_linear",
    "silu_mul",
    "down_linear",
    "final_norm",
    "lm_head_shards",
    "lm_head_concat",
    "untilize",
    "argmax",
)
EXTRA_REGION_ORDER = ("runtime_inputs",)
BOTTLENECK_CLASSES = (
    "compute_bound",
    "dram_bound",
    "noc_bound",
    "cb_bound",
    "packing_bound",
    "unknown",
)

_OP = {
    "argmax": "ArgMaxDeviceOperation",
    "binary": "BinaryNgDeviceOperation",
    "concat": "ConcatDeviceOperation",
    "concat_heads": "NLPConcatHeadsDecodeDeviceOperation",
    "create_heads": "NLPCreateQKVHeadsDecodeDeviceOperation",
    "embedding": "EmbeddingsDeviceOperation",
    "interleaved_to_sharded": "InterleavedToShardedDeviceOperation",
    "kv_update": "PagedUpdateCacheDeviceOperation",
    "kv_update_fused": "PagedFusedUpdateCacheDeviceOperation",
    "layer_norm": "LayerNormDeviceOperation",
    "matmul": "MatmulDeviceOperation",
    "reshard": "ReshardDeviceOperation",
    "rope": "RotaryEmbeddingLlamaDeviceOperation",
    "rope_fused_qk": "RotaryEmbeddingLlamaFusedQKDeviceOperation",
    "sdpa": "SdpaDecodeDeviceOperation",
    "sharded_to_interleaved": "ShardedToInterleavedDeviceOperation",
    "tilize": "TilizeDeviceOperation",
    "untilize": "UntilizeDeviceOperation",
}
_LAYOUT_OPS = {
    _OP["interleaved_to_sharded"],
    _OP["reshard"],
    _OP["sharded_to_interleaved"],
    _OP["tilize"],
}
_TIME_COLUMNS = (
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE KERNEL DURATION DM START [ns]",
    "DEVICE KERNEL DURATION PER CORE MIN [ns]",
    "DEVICE KERNEL DURATION PER CORE MAX [ns]",
    "DEVICE KERNEL DURATION PER CORE AVG [ns]",
    "DEVICE KERNEL FIRST TO LAST START [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
)
_COMPONENT_COLUMNS = {
    "brisc_ms": "DEVICE BRISC KERNEL DURATION [ns]",
    "ncrisc_ms": "DEVICE NCRISC KERNEL DURATION [ns]",
    "trisc0_ms": "DEVICE TRISC0 KERNEL DURATION [ns]",
    "trisc1_ms": "DEVICE TRISC1 KERNEL DURATION [ns]",
    "trisc2_ms": "DEVICE TRISC2 KERNEL DURATION [ns]",
    "erisc_ms": "DEVICE ERISC KERNEL DURATION [ns]",
    "cb_wait_front_ms": "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "cb_reserve_back_ms": "DEVICE COMPUTE CB RESERVE BACK [ns]",
}
_METRIC_COLUMNS = {
    "fpu_util_percent": "PM FPU UTIL (%)",
    "noc_util_percent": "NOC UTIL (%)",
    "multicast_noc_util_percent": "MULTICAST NOC UTIL (%)",
    "dram_bw_util_percent": "DRAM BW UTIL (%)",
    "noc_congestion_impact_percent": "NPE CONG IMPACT (%)",
    "pm_ideal_ns": "PM IDEAL [ns]",
    "pm_compute_ns": "PM COMPUTE [ns]",
    "pm_bandwidth_ns": "PM BANDWIDTH [ns]",
}
_OPERATOR_CONFIG_KEYS = {
    "qkv_linear": "attention.qkv",
    "sdpa": "attention.sdpa",
    "o_projection": "attention.o_proj",
    "gate_linear": "mlp.gate",
    "up_linear": "mlp.up",
    "down_linear": "mlp.down",
    "lm_head_shards": "lm_head.shards",
}


class ProfilerAuditError(RuntimeError):
    pass


def run_autotune_profiler_audit(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path,
    input_prompts: str | Path,
    tokenizer_path: str | Path | None = None,
    instruct: bool = False,
    device: str = "p150a",
    device_id: int = 0,
    timeout_seconds: float = 3600.0,
    profiler_csv: str | Path | None = None,
    profile_report: str | Path | None = None,
) -> dict[str, Any]:
    """Capture or reuse a full decode trace and write the Phase 1 audit."""

    program = Path(program_dir).resolve()
    output_json, _ = _output_paths(Path(out))
    work_dir = output_json.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    resolved_csv = Path(profiler_csv).resolve() if profiler_csv else None
    resolved_profile = Path(profile_report).resolve() if profile_report else None
    if (resolved_csv is None) != (resolved_profile is None):
        return _write_failure(
            out,
            "--profiler-csv and --profile-report must be supplied together",
        )

    if resolved_csv is None:
        resolved_profile = work_dir / "profile.json"
        profiler_dir = work_dir / "profiler"
        profiler_dir.mkdir(parents=True, exist_ok=True)
        command = _profiler_command(
            program_dir=program,
            model_path=Path(model_path).resolve(),
            input_prompts=Path(input_prompts).resolve(),
            tokenizer_path=(Path(tokenizer_path).resolve() if tokenizer_path else None),
            instruct=instruct,
            device=device,
            device_id=device_id,
            profile_report=resolved_profile,
            profiler_dir=profiler_dir,
        )
        log_path = work_dir / "profiler_capture.log"
        environment = _profiler_environment()
        try:
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=_repository_root(),
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return _write_failure(
                out,
                f"profiler capture timed out after {timeout_seconds:.0f}s",
                command=command,
                profiler_log=log_path,
            )
        if completed.returncode != 0:
            return _write_failure(
                out,
                f"profiler capture failed with exit code {completed.returncode}",
                command=command,
                profiler_log=log_path,
            )
        candidates = sorted(
            profiler_dir.glob("reports/**/ops_perf_results*.csv"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            return _write_failure(
                out,
                "TT-Metal profiler did not produce an ops_perf_results CSV",
                command=command,
                profiler_log=log_path,
            )
        resolved_csv = candidates[-1]

    return build_autotune_profiler_audit(
        profiler_csv=resolved_csv,
        profile_report=resolved_profile,
        program_dir=program,
        out=out,
    )


def build_autotune_profiler_audit(
    *,
    profiler_csv: str | Path,
    profile_report: str | Path,
    program_dir: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Build a deterministic audit from persisted TT-Metal profiler data."""

    csv_path = Path(profiler_csv).resolve()
    profile_path = Path(profile_report).resolve()
    program = Path(program_dir).resolve()
    try:
        profile = _read_json(profile_path)
        config_path = program / "config.json"
        config = _read_json(config_path)
        measurement = validate_decode_profile_contract(
            profile,
            program_config=config,
            expected_layers=int(config["num_layers"]),
            measurement_contract=_PROFILER_MEASUREMENT,
            expected_trace_capture_count=1,
            expected_trace_execute_count=1,
            expected_workload={
                "mode": "decode-steady",
                "batch_size": 32,
                "prefill_len": 256,
                "cache_len": 1024,
                "prefill_execution_mode": "eager",
            },
            require_runtime_input_stability=True,
            handoff_evidence="runtime_inputs",
        )
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        trace_rows, selection = select_trace_replay(rows)
        e2e_latency_ms = _required_number(profile, "decode_step_ms_p50")
        normalized_rows, normalization = normalize_trace_timestamps(
            trace_rows,
            e2e_latency_ms=e2e_latency_ms,
        )
        assignments = assign_decode_regions(
            normalized_rows,
            num_layers=int(config["num_layers"]),
            lm_head_shards=_lm_head_shard_count(config),
            program_config=config,
        )
        regions = _aggregate_regions(normalized_rows, assignments, config)
        total_ns = sum(float(row["_duration_ns"]) for row in normalized_rows)
        device_total_ms = total_ns / 1_000_000.0
        ratio = device_total_ms / e2e_latency_ms
        if not 0.75 <= ratio <= 1.25:
            raise ProfilerAuditError(
                "normalized device total does not match decode p50: "
                f"device={device_total_ms:.6f}ms, "
                f"e2e={e2e_latency_ms:.6f}ms, ratio={ratio:.4f}"
            )
        budget_answers = _budget_answers(assignments, normalized_rows, total_ns)
        trace_identity = profile.get("trace_key") or {}
        precision_contract = _precision_contract(config)
        report = {
            "schema_version": SCHEMA_VERSION,
            "command": "diagnose",
            "stage": "autotune-profiler-audit",
            "status": "pass",
            "passed": True,
            "source": {
                "profiler_csv": str(csv_path),
                "profiler_csv_sha256": file_sha256(csv_path),
                "profile_report": str(profile_path),
                "profile_report_sha256": file_sha256(profile_path),
                "program_dir": str(program),
                "program_config": str(config_path),
                "program_config_sha256": file_sha256(config_path),
            },
            "measurement_contract": {
                **measurement,
                "precision": precision_contract,
                "precision_contract_sha256": sha256_json(precision_contract),
                "trace_identity": trace_identity,
                "trace_identity_sha256": sha256_json(trace_identity),
            },
            "trace_selection": selection,
            "clock_normalization": normalization,
            "latency": {
                "device_kernel_total_ms": device_total_ms,
                "decode_step_p50_ms": e2e_latency_ms,
                "device_to_e2e_ratio": ratio,
                "device_unattributed_ms": e2e_latency_ms - device_total_ms,
                "trace_op_count": len(normalized_rows),
            },
            "regions": regions,
            "budget_answers": budget_answers,
            "bottleneck_ranking": [
                {
                    "region": region["region"],
                    "device_kernel_latency_ms": region["device_kernel_latency_ms"],
                    "percent_of_decode": region["percent_of_decode"],
                    "classification": region["classification"],
                }
                for region in sorted(
                    regions,
                    key=lambda item: item["device_kernel_latency_ms"],
                    reverse=True,
                )
            ],
            "metric_availability": _metric_availability(normalized_rows),
        }
        output_json, output_csv = _output_paths(Path(out))
        write_report(output_json, report)
        _write_regions_csv(output_csv, regions)
        return report
    except (OSError, KeyError, TypeError, ValueError, ProfilerAuditError) as exc:
        return _write_failure(
            out,
            str(exc),
            profiler_csv=csv_path,
            profile_report=profile_path,
            program_dir=program,
        )


def select_trace_replay(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ProfilerAuditError("profiler CSV is empty")
    required = {
        "OP CODE",
        "DEVICE KERNEL DURATION [ns]",
        "METAL TRACE ID",
        "METAL TRACE REPLAY SESSION ID",
    }
    missing = required - set(rows[0])
    if missing:
        raise ProfilerAuditError(
            "profiler CSV is missing required columns: " + ", ".join(sorted(missing))
        )
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        trace_id = str(source.get("METAL TRACE ID", "")).strip()
        replay_id = str(source.get("METAL TRACE REPLAY SESSION ID", "")).strip()
        if not trace_id or trace_id == "-1" or not replay_id:
            continue
        sessions[(trace_id, replay_id)].append(dict(source))
    if not sessions:
        raise ProfilerAuditError(
            "profiler CSV has no complete Metal trace replay session"
        )

    def session_key(
        item: tuple[tuple[str, str], list[dict[str, Any]]],
    ) -> tuple[int, int]:
        (_, replay_id), session_rows = item
        try:
            ordinal = int(replay_id)
        except ValueError:
            ordinal = -1
        return ordinal, len(session_rows)

    selected_key, selected = max(sessions.items(), key=session_key)
    return selected, {
        "metal_trace_id": selected_key[0],
        "replay_session_id": selected_key[1],
        "selected_op_count": len(selected),
        "available_sessions": [
            {
                "metal_trace_id": trace_id,
                "replay_session_id": replay_id,
                "op_count": len(session_rows),
            }
            for (trace_id, replay_id), session_rows in sorted(sessions.items())
        ],
    }


def normalize_trace_timestamps(
    rows: Sequence[Mapping[str, Any]],
    *,
    e2e_latency_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    durations = [_required_float(row, "DEVICE KERNEL DURATION [ns]") for row in rows]
    huge_count = sum(value > 1_000_000_000.0 for value in durations)
    offset_ns = 0.0
    method = "none"
    candidates: list[float] = []
    median_absolute_deviation_ns = 0.0
    inlier_count = 0
    if huge_count:
        for row in rows:
            value = _optional_float(row.get("OP TO OP LATENCY [ns]"))
            if value is not None and value < -1_000_000_000.0:
                candidates.append(-value)
        if len(candidates) < max(10, int(huge_count * 0.75)):
            raise ProfilerAuditError(
                "trace durations contain a clock offset but there are not enough "
                "negative op-to-op samples to normalize it"
            )
        offset_ns = statistics.median(candidates)
        median_absolute_deviation_ns = statistics.median(
            abs(value - offset_ns) for value in candidates
        )
        tolerance = max(1_000_000.0, 10.0 * median_absolute_deviation_ns)
        inlier_count = sum(abs(value - offset_ns) <= tolerance for value in candidates)
        if inlier_count / len(candidates) < 0.95:
            raise ProfilerAuditError(
                "trace clock-offset evidence is not coherent enough to normalize"
            )
        method = "median_negative_op_to_op_offset"

    normalized: list[dict[str, Any]] = []
    corrected_count = 0
    for source, raw_duration in zip(rows, durations):
        row = dict(source)
        normalized_timing: dict[str, float | None] = {}
        for column in _TIME_COLUMNS:
            value = _optional_float(source.get(column))
            if value is not None and offset_ns and value > 1_000_000_000.0:
                corrected = value - offset_ns
                if corrected < 0.0 or corrected > 100_000_000.0:
                    raise ProfilerAuditError(
                        f"clock normalization produced an invalid {column}: "
                        f"{corrected:.3f}ns"
                    )
                value = corrected
            normalized_timing[column] = value
        duration = normalized_timing["DEVICE KERNEL DURATION [ns]"]
        if duration is None or duration <= 0.0:
            raise ProfilerAuditError("trace contains a non-positive kernel duration")
        if raw_duration > 1_000_000_000.0:
            corrected_count += 1
        row["_duration_ns"] = duration
        row["_normalized_timing_ns"] = normalized_timing
        normalized.append(row)

    total_ms = sum(float(row["_duration_ns"]) for row in normalized) / 1_000_000.0
    ratio = total_ms / e2e_latency_ms
    if not 0.75 <= ratio <= 1.25:
        raise ProfilerAuditError(
            "trace clock normalization failed the end-to-end latency check: "
            f"device={total_ms:.6f}ms, e2e={e2e_latency_ms:.6f}ms"
        )
    return normalized, {
        "required": bool(huge_count),
        "method": method,
        "offset_ns": offset_ns if huge_count else None,
        "offset_candidate_count": len(candidates),
        "offset_inlier_count": inlier_count,
        "offset_inlier_ratio": (inlier_count / len(candidates) if candidates else None),
        "median_absolute_deviation_ns": (
            median_absolute_deviation_ns if candidates else None
        ),
        "corrected_kernel_duration_count": corrected_count,
        "normalized_device_total_ms": total_ms,
        "decode_step_p50_ms": e2e_latency_ms,
        "device_to_e2e_ratio": ratio,
        "validation": "pass",
    }


def assign_decode_regions(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    lm_head_shards: int,
    program_config: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    if num_layers <= 0 or lm_head_shards <= 0:
        raise ProfilerAuditError("invalid model depth or LM-head shard count")
    codes = [str(row.get("OP CODE", "")) for row in rows]
    config = program_config or {}
    rope_template = template_choice_from_runtime_config(config, ROPE_AXIS)
    kv_update_template = template_choice_from_runtime_config(config, KV_UPDATE_AXIS)
    norm_indices = [i for i, code in enumerate(codes) if code == _OP["layer_norm"]]
    embedding_indices = [i for i, code in enumerate(codes) if code == _OP["embedding"]]
    if len(norm_indices) != num_layers * 2 + 1:
        raise ProfilerAuditError(
            f"expected {num_layers * 2 + 1} layer norms, found {len(norm_indices)}"
        )
    model_embeddings = [i for i in embedding_indices if i < norm_indices[0]]
    if not model_embeddings:
        raise ProfilerAuditError("could not locate token embedding before layer 0")
    model_embedding = model_embeddings[-1]
    assignments: list[dict[str, str] | None] = [None] * len(rows)

    def mark(index: int, region: str, semantic_stage: str) -> None:
        if assignments[index] is not None:
            raise ProfilerAuditError(f"trace op {index} was assigned twice")
        assignments[index] = {
            "region": region,
            "semantic_stage": semantic_stage,
        }

    def expect(index: int, code: str, region: str, semantic_stage: str) -> int:
        observed = codes[index] if index < len(codes) else "<end>"
        if observed != code:
            raise ProfilerAuditError(
                f"trace op {index}: expected {code}, found {observed}"
            )
        mark(index, region, semantic_stage)
        return index + 1

    for index in range(model_embedding):
        mark(index, "runtime_inputs", "runtime_inputs")
    mark(model_embedding, "embedding", "embedding")
    cursor = model_embedding + 1
    while cursor < norm_indices[0]:
        if codes[cursor] not in _LAYOUT_OPS:
            raise ProfilerAuditError(
                f"unexpected pre-layer op {cursor}: {codes[cursor]}"
            )
        mark(cursor, "attention_rmsnorm", "attention_rmsnorm")
        cursor += 1

    for layer in range(num_layers):
        cursor = expect(
            cursor,
            _OP["layer_norm"],
            "attention_rmsnorm",
            "attention_rmsnorm",
        )
        cursor = expect(cursor, _OP["matmul"], "qkv_linear", "qkv_linear")
        cursor = expect(
            cursor,
            _OP["create_heads"],
            "create_heads",
            "create_heads",
        )
        if rope_template == SEPARATE_QK_ROPE:
            for _ in range(2):
                cursor = expect(cursor, _OP["rope"], "rope", "rope")
        elif rope_template == FUSED_QK_ROPE:
            while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
                mark(cursor, "rope", "rope")
                cursor += 1
            cursor = expect(cursor, _OP["rope_fused_qk"], "rope", "rope")
        else:
            raise ProfilerAuditError(f"unsupported RoPE template: {rope_template}")

        if kv_update_template == SEPARATE_PAGED_UPDATE:
            for _ in range(2):
                cursor = expect(
                    cursor, _OP["kv_update"], "kv_update", "kv_update"
                )
        elif kv_update_template == FUSED_PAGED_UPDATE:
            while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
                mark(cursor, "kv_update", "kv_update")
                cursor += 1
            cursor = expect(
                cursor, _OP["kv_update_fused"], "kv_update", "kv_update"
            )
        else:
            raise ProfilerAuditError(
                f"unsupported KV-update template: {kv_update_template}"
            )
        cursor = expect(cursor, _OP["sdpa"], "sdpa", "sdpa")
        while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
            mark(cursor, "concat_heads", "concat_heads")
            cursor += 1
        cursor = expect(
            cursor,
            _OP["concat_heads"],
            "concat_heads",
            "concat_heads",
        )
        cursor = expect(
            cursor,
            _OP["matmul"],
            "o_projection",
            "o_projection",
        )
        while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
            mark(cursor, "residual", "attention_residual")
            cursor += 1
        cursor = expect(
            cursor,
            _OP["binary"],
            "residual",
            "attention_residual",
        )
        while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
            mark(cursor, "mlp_rmsnorm", "mlp_rmsnorm")
            cursor += 1
        cursor = expect(
            cursor,
            _OP["layer_norm"],
            "mlp_rmsnorm",
            "mlp_rmsnorm",
        )
        cursor = expect(cursor, _OP["matmul"], "gate_linear", "gate_linear")
        cursor = expect(cursor, _OP["matmul"], "up_linear", "up_linear")
        cursor = expect(cursor, _OP["binary"], "silu_mul", "silu_mul")
        cursor = expect(cursor, _OP["matmul"], "down_linear", "down_linear")
        cursor = expect(
            cursor,
            _OP["binary"],
            "residual",
            "mlp_residual",
        )
        next_region = "attention_rmsnorm" if layer + 1 < num_layers else "final_norm"
        while cursor < len(codes) and codes[cursor] in _LAYOUT_OPS:
            mark(cursor, next_region, next_region)
            cursor += 1

    cursor = expect(cursor, _OP["layer_norm"], "final_norm", "final_norm")
    for _ in range(lm_head_shards):
        cursor = expect(
            cursor,
            _OP["matmul"],
            "lm_head_shards",
            "lm_head_shards",
        )
        while cursor < len(codes) and codes[cursor] == _OP["sharded_to_interleaved"]:
            mark(cursor, "lm_head_concat", "lm_head_concat")
            cursor += 1
    cursor = expect(
        cursor,
        _OP["concat"],
        "lm_head_concat",
        "lm_head_concat",
    )
    cursor = expect(cursor, _OP["untilize"], "untilize", "untilize")
    cursor = expect(cursor, _OP["argmax"], "argmax", "argmax")
    while cursor < len(codes):
        mark(cursor, "runtime_inputs", "runtime_inputs")
        cursor += 1

    result = [item for item in assignments if item is not None]
    if len(result) != len(rows):
        raise ProfilerAuditError("not every trace operation received a region")
    observed = Counter(item["region"] for item in result)
    missing = [region for region in REGION_ORDER if observed[region] == 0]
    if missing:
        raise ProfilerAuditError(
            "required profiler regions are empty: " + ", ".join(missing)
        )
    return result


def _aggregate_regions(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    total_ns = sum(float(row["_duration_ns"]) for row in rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row, assignment in zip(rows, assignments):
        grouped[assignment["region"]].append(row)
    operators = (config.get("autotune") or {}).get("operators") or {}
    records: list[dict[str, Any]] = []
    for region in (*REGION_ORDER, *EXTRA_REGION_ORDER):
        region_rows = grouped.get(region, [])
        if not region_rows:
            continue
        duration_ns = sum(float(row["_duration_ns"]) for row in region_rows)
        metrics = _aggregate_hardware_metrics(region_rows)
        config_key = _OPERATOR_CONFIG_KEYS.get(region)
        operator_config = operators.get(config_key) if config_key else None
        classification, evidence = _classify_bottleneck(
            region_rows,
            metrics,
        )
        worker_cores = [
            value
            for value in (_optional_float(row.get("CORE COUNT")) for row in region_rows)
            if value is not None
        ]
        input_memory = sorted(
            {
                str(value)
                for row in region_rows
                for key, value in row.items()
                if key.startswith("INPUT_")
                and key.endswith("_MEMORY")
                and str(value).strip()
            }
        )
        output_memory = sorted(
            {
                str(value)
                for row in region_rows
                for key, value in row.items()
                if key.startswith("OUTPUT_")
                and key.endswith("_MEMORY")
                and str(value).strip()
            }
        )
        records.append(
            {
                "region": region,
                "op_count": len(region_rows),
                "op_codes": sorted({str(row["OP CODE"]) for row in region_rows}),
                "device_kernel_latency_ms": duration_ns / 1_000_000.0,
                "percent_of_decode": 100.0 * duration_ns / total_ns,
                "worker_core_count": {
                    "min": min(worker_cores) if worker_cores else None,
                    "max": max(worker_cores) if worker_cores else None,
                    "mean": statistics.fmean(worker_cores) if worker_cores else None,
                },
                "available_worker_core_count": _mean_column(
                    region_rows,
                    "AVAILABLE WORKER CORE COUNT",
                ),
                "program_family": _program_families(operator_config),
                "program_config_key": config_key,
                "program_config": operator_config,
                "input_memory_configs": input_memory,
                "output_memory_configs": output_memory,
                "hardware_metrics": metrics,
                "classification": classification,
                "classification_evidence": evidence,
            }
        )
    return records


def _aggregate_hardware_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for output_name, column in _METRIC_COLUMNS.items():
        metrics[output_name] = _mean_column(rows, column)
    for output_name, column in _COMPONENT_COLUMNS.items():
        values = [(row.get("_normalized_timing_ns") or {}).get(column) for row in rows]
        present = [float(value) for value in values if value is not None]
        metrics[output_name] = sum(present) / 1_000_000.0 if present else None
    metrics["packer_stall_ms"] = _mean_matching_columns(
        rows,
        include=("PACK", "STALL"),
        scale=1_000_000.0,
    )
    metrics["unpacker_stall_ms"] = _mean_matching_columns(
        rows,
        include=("UNPACK", "STALL"),
        scale=1_000_000.0,
    )
    metrics["l1_stall_ms"] = _mean_matching_columns(
        rows,
        include=("L1", "STALL"),
        scale=1_000_000.0,
    )
    metrics["achieved_weight_bandwidth_gbps"] = _mean_matching_columns(
        rows,
        include=("ACHIEVED", "WEIGHT", "BW"),
    )
    metrics["required_input_bandwidth"] = _mean_list_column(rows, "PM REQ I BW")
    metrics["required_output_bandwidth"] = _mean_list_column(rows, "PM REQ O BW")
    return metrics


def _classify_bottleneck(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    duration_ms = sum(float(row["_duration_ns"]) for row in rows) / 1_000_000.0
    cb_ms = sum(
        float(metrics.get(name) or 0.0)
        for name in ("cb_wait_front_ms", "cb_reserve_back_ms")
    )
    noc_util = max(
        float(metrics.get("noc_util_percent") or 0.0),
        float(metrics.get("multicast_noc_util_percent") or 0.0),
    )
    congestion = float(metrics.get("noc_congestion_impact_percent") or 0.0)
    pack_ms = sum(
        float(metrics.get(name) or 0.0)
        for name in ("packer_stall_ms", "unpacker_stall_ms")
    )
    compute_ns = metrics.get("pm_compute_ns")
    bandwidth_ns = metrics.get("pm_bandwidth_ns")
    evidence = {
        "device_kernel_latency_ms": duration_ms,
        "cb_stall_fraction": cb_ms / duration_ms if duration_ms else None,
        "noc_util_percent": noc_util if noc_util else None,
        "noc_congestion_impact_percent": congestion if congestion else None,
        "packing_stall_fraction": pack_ms / duration_ms if duration_ms else None,
        "pm_compute_ns": compute_ns,
        "pm_bandwidth_ns": bandwidth_ns,
    }
    if duration_ms and cb_ms / duration_ms >= 0.10:
        return "cb_bound", evidence
    if noc_util >= 70.0 or congestion >= 10.0:
        return "noc_bound", evidence
    if duration_ms and pack_ms / duration_ms >= 0.10:
        return "packing_bound", evidence
    if compute_ns is not None and bandwidth_ns is not None:
        if float(compute_ns) >= 1.15 * max(float(bandwidth_ns), 1.0):
            return "compute_bound", evidence
        if float(bandwidth_ns) >= 1.15 * max(float(compute_ns), 1.0):
            return "dram_bound", evidence
    return "unknown", evidence


def _budget_answers(
    assignments: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, Any]],
    total_ns: float,
) -> dict[str, dict[str, Any]]:
    definitions = {
        "mlp": {
            "semantic_stages": {
                "mlp_rmsnorm",
                "gate_linear",
                "up_linear",
                "silu_mul",
                "down_linear",
                "mlp_residual",
            },
            "definition": (
                "MLP RMSNorm, gate/up/down linears, SILU-mul, and MLP residual"
            ),
        },
        "attention_matmul": {
            "semantic_stages": {"qkv_linear", "o_projection"},
            "definition": "QKV and O-projection MatMul regions",
        },
        "sdpa": {
            "semantic_stages": {"sdpa"},
            "definition": "SDPA decode region",
        },
        "lm_head": {
            "semantic_stages": {
                "lm_head_shards",
                "lm_head_concat",
                "untilize",
                "argmax",
            },
            "definition": "LM-head shards, concat, untilize, and argmax",
        },
        "kv_update_plus_rope": {
            "semantic_stages": {"kv_update", "rope"},
            "definition": "K/V paged updates and Q/K RoPE",
        },
    }
    answers: dict[str, dict[str, Any]] = {}
    for name, spec in definitions.items():
        stages = spec["semantic_stages"]
        latency_ns = sum(
            float(row["_duration_ns"])
            for row, assignment in zip(rows, assignments)
            if assignment["semantic_stage"] in stages
        )
        answers[name] = {
            "definition": spec["definition"],
            "device_kernel_latency_ms": latency_ns / 1_000_000.0,
            "percent_of_decode": 100.0 * latency_ns / total_ns,
        }
    return answers


def _metric_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = {
        **_METRIC_COLUMNS,
        **_COMPONENT_COLUMNS,
    }
    availability = {}
    for metric, column in requested.items():
        count = sum(_optional_float(row.get(column)) is not None for row in rows)
        availability[metric] = {
            "column": column,
            "sample_count": count,
            "available": count > 0,
        }
    for metric, needles in {
        "packer_stall": ("PACK", "STALL"),
        "unpacker_stall": ("UNPACK", "STALL"),
        "l1_stall": ("L1", "STALL"),
        "achieved_weight_bandwidth": ("ACHIEVED", "WEIGHT", "BW"),
    }.items():
        columns = [
            key for key in rows[0] if all(needle in key.upper() for needle in needles)
        ]
        availability[metric] = {
            "columns": columns,
            "sample_count": sum(
                _optional_float(row.get(column)) is not None
                for row in rows
                for column in columns
            ),
            "available": bool(columns),
        }
    return availability


def _profiler_command(
    *,
    program_dir: Path,
    model_path: Path,
    input_prompts: Path,
    tokenizer_path: Path | None,
    instruct: bool,
    device: str,
    device_id: int,
    profile_report: Path,
    profiler_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tracy",
        "-p",
        "-r",
        "--check-exit-code",
        "--dump-device-data-mid-run",
        "--no-op-info-cache",
        "-o",
        str(profiler_dir),
        "-m",
        "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "profile",
        "--mode",
        "decode-steady",
        "--program-dir",
        str(program_dir),
        "--model-path",
        str(model_path),
        "--input-prompts",
        str(input_prompts),
        "--layers",
        "32",
        "--batch-size",
        "32",
        "--prefill-len",
        "256",
        "--cache-len",
        "1024",
        "--warmup",
        str(_PROFILER_MEASUREMENT.warmup),
        "--iterations",
        str(_PROFILER_MEASUREMENT.iterations),
        "--after-prefill",
        "--runtime-input-mode",
        "persistent",
        "--execution-mode",
        "trace",
        "--prefill-execution-mode",
        "eager",
        "--device",
        device,
        "--device-id",
        str(device_id),
        "--out",
        str(profile_report),
    ]
    if tokenizer_path is not None:
        command.extend(("--tokenizer-path", str(tokenizer_path)))
    if instruct:
        command.append("--instruct")
    return command


def _profiler_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["BUDDY_TTNN_PROFILER_AUDIT"] = "1"
    metal_home = Path(
        environment.get(
            "TT_METAL_HOME",
            _repository_root() / "thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal",
        )
    ).resolve()
    tools_path = str(metal_home / "tools")
    python_path = environment.get("PYTHONPATH", "")
    entries = [str(_repository_root()), tools_path]
    if python_path:
        entries.append(python_path)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["TT_METAL_HOME"] = str(metal_home)
    return environment


def _output_paths(out: Path) -> tuple[Path, Path]:
    destination = out.resolve()
    if destination.suffix.lower() == ".json":
        return destination, destination.with_suffix(".csv")
    return (
        destination / "autotune_profiler_audit.json",
        destination / "autotune_profiler_audit.csv",
    )


def _write_failure(
    out: str | Path,
    error: str,
    **context: Any,
) -> dict[str, Any]:
    output_json, output_csv = _output_paths(Path(out))
    report = {
        "schema_version": SCHEMA_VERSION,
        "command": "diagnose",
        "stage": "autotune-profiler-audit",
        "status": "fail",
        "passed": False,
        "error": error,
        "context": {
            key: (
                [str(item) for item in value] if isinstance(value, list) else str(value)
            )
            for key, value in context.items()
            if value is not None
        },
    }
    write_report(output_json, report)
    _write_regions_csv(output_csv, [])
    return report


def _write_regions_csv(path: Path, regions: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "region",
        "op_count",
        "device_kernel_latency_ms",
        "percent_of_decode",
        "classification",
        "worker_core_min",
        "worker_core_max",
        "worker_core_mean",
        "program_family",
        "input_memory_configs",
        "output_memory_configs",
        "fpu_util_percent",
        "brisc_ms",
        "ncrisc_ms",
        "trisc0_ms",
        "trisc1_ms",
        "trisc2_ms",
        "cb_wait_front_ms",
        "cb_reserve_back_ms",
        "noc_util_percent",
        "dram_bw_util_percent",
        "packer_stall_ms",
        "unpacker_stall_ms",
        "l1_stall_ms",
        "achieved_weight_bandwidth_gbps",
    )
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for region in regions:
        cores = region.get("worker_core_count") or {}
        metrics = region.get("hardware_metrics") or {}
        writer.writerow(
            {
                "region": region.get("region"),
                "op_count": region.get("op_count"),
                "device_kernel_latency_ms": region.get("device_kernel_latency_ms"),
                "percent_of_decode": region.get("percent_of_decode"),
                "classification": region.get("classification"),
                "worker_core_min": cores.get("min"),
                "worker_core_max": cores.get("max"),
                "worker_core_mean": cores.get("mean"),
                "program_family": ";".join(region.get("program_family") or []),
                "input_memory_configs": ";".join(
                    region.get("input_memory_configs") or []
                ),
                "output_memory_configs": ";".join(
                    region.get("output_memory_configs") or []
                ),
                **{name: metrics.get(name) for name in fields if name in metrics},
            }
        )
    _atomic_write_text(path, stream.getvalue())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfilerAuditError(f"expected a JSON object: {path}")
    return value


def _required_number(mapping: Mapping[str, Any], key: str) -> float:
    value = _optional_float(mapping.get(key))
    if value is None or value <= 0.0:
        raise ProfilerAuditError(f"profile report has no positive {key}")
    return value


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _optional_float(mapping.get(key))
    if value is None:
        raise ProfilerAuditError(f"profiler row has no numeric {key}")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _mean_column(rows: Sequence[Mapping[str, Any]], column: str) -> float | None:
    values = [
        value
        for value in (_optional_float(row.get(column)) for row in rows)
        if value is not None
    ]
    return statistics.fmean(values) if values else None


def _mean_list_column(rows: Sequence[Mapping[str, Any]], column: str) -> float | None:
    values: list[float] = []
    for row in rows:
        raw = str(row.get(column, "")).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            values.extend(
                float(value)
                for value in parsed
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            )
    return statistics.fmean(values) if values else None


def _mean_matching_columns(
    rows: Sequence[Mapping[str, Any]],
    *,
    include: Sequence[str],
    scale: float = 1.0,
) -> float | None:
    columns = [
        key for key in rows[0] if all(needle in key.upper() for needle in include)
    ]
    values = [
        value
        for row in rows
        for column in columns
        if (value := _optional_float(row.get(column))) is not None
    ]
    return statistics.fmean(values) / scale if values else None


def _program_families(operator_config: Any) -> list[str]:
    if not isinstance(operator_config, Mapping):
        return []
    families = []
    runtime_kind = operator_config.get("runtime_kind")
    if runtime_kind:
        families.append(str(runtime_kind))
    programs = operator_config.get("programs") or []
    if isinstance(programs, list):
        for program in programs:
            if not isinstance(program, Mapping):
                continue
            family = program.get("program_family") or program.get("runtime_kind")
            if family:
                families.append(str(family))
    return sorted(set(families))


def _lm_head_shard_count(config: Mapping[str, Any]) -> int:
    operators = (config.get("autotune") or {}).get("operators") or {}
    programs = (operators.get("lm_head.shards") or {}).get("programs") or []
    if programs:
        return len(programs)
    return int((config.get("lm_head") or {}).get("num_splits", 8))


def _precision_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    operators = (config.get("autotune") or {}).get("operators") or {}
    return {
        "official_config_profile": config.get("official_config_profile"),
        "template_config": config.get("template_config"),
        "operator_math_fidelity": {
            name: value.get("math_fidelity")
            for name, value in sorted(operators.items())
            if isinstance(value, Mapping) and value.get("math_fidelity") is not None
        },
        "dtype_seed": "bf16",
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


__all__ = [
    "BOTTLENECK_CLASSES",
    "ProfilerAuditError",
    "REGION_ORDER",
    "assign_decode_regions",
    "build_autotune_profiler_audit",
    "normalize_trace_timestamps",
    "run_autotune_profiler_audit",
    "select_trace_replay",
]
