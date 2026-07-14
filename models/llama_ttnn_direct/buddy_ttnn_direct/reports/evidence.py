from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from ..codegen.artifacts import write_json
from .performance import (
    performance_baseline_entry_summary as _performance_baseline_entry_summary,
    performance_gap_summary as _performance_gap_summary,
    throughput_baseline_summary as _throughput_baseline_summary,
    validate_real_generate_milestones as _validate_real_generate_milestones,
)
from .tensorization import (
    step_tensorization_summary,
    tensorized_physical_shape_mismatches,
)
from .validation import (
    final_acceptance_gate_matrix,
    model_end_to_end_readiness,
    real_decode_acceptance_scope,
)
from .runtime_diagnostics import real_decode_runtime_diagnostics


def artifact_index(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def artifact_evidence(name: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "directory"
    else:
        kind = "missing"
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
    }


def shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _append_option(args: list[str], option: str, value: Any) -> None:
    if value is not None:
        args.extend([option, str(value)])


def real_decode_cli_args(
    *,
    program_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    official_config_path: str | Path,
    decode_step_search_space_path: str | Path,
    performance_baselines_path: str | Path,
    layers: int,
    batch_size: int | None,
    cache_len: int | None,
    max_new_tokens: int,
    prefill_len: int | None,
    device: str,
    device_id: int,
    trace: bool,
    trace_iterations: int,
    skip_autotune: bool,
    skip_profile_decode_step: bool,
    require_full_decode_step: bool,
    require_model_end_to_end: bool,
    require_official_performance_parity: bool,
    require_trace: bool,
    require_official_config_match: bool,
    require_full_depth: bool,
    require_program_runtime_shape: bool,
    require_batch32_decode_step: bool,
    min_tokens_per_second_per_user: float | None,
    baseline_tokens_per_second_per_user: float | None,
    baseline_reference: str | None,
    min_baseline_ratio: float | None,
    decode_shell_pcc_threshold: float,
    require_decode_shell_numeric_reference: bool,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    dtype_seed: str | None = None,
    metric: str | None = None,
    dry_run: bool = False,
    preflight_only: bool = False,
    guard_device_busy: bool = False,
    guard_device_health: bool = False,
) -> list[str]:
    args = [
        "python",
        "-m",
        "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "validate-real-decode",
        "--program-dir",
        str(program_dir),
        "--model-path",
        str(model_path),
        "--out-dir",
        str(out_dir),
        "--official-config",
        str(official_config_path),
        "--decode-step-search-space",
        str(decode_step_search_space_path),
        "--performance-baselines",
        str(performance_baselines_path),
        "--layers",
        str(layers),
        "--device",
        str(device),
        "--device-id",
        str(device_id),
        "--trace-iterations",
        str(trace_iterations),
        "--decode-shell-pcc-threshold",
        str(decode_shell_pcc_threshold),
    ]
    _append_option(args, "--batch-size", batch_size)
    _append_option(args, "--cache-len", cache_len)
    _append_option(args, "--max-new-tokens", max_new_tokens)
    _append_option(args, "--prefill-len", prefill_len)
    _append_option(args, "--prompt", prompt)
    _append_option(args, "--tokenizer-path", tokenizer_path)
    _append_option(args, "--dtype-seed", dtype_seed)
    _append_option(args, "--metric", metric)
    _append_option(
        args,
        "--min-tokens-per-second-per-user",
        min_tokens_per_second_per_user,
    )
    _append_option(
        args,
        "--baseline-tokens-per-second-per-user",
        baseline_tokens_per_second_per_user,
    )
    _append_option(args, "--baseline-reference", baseline_reference)
    _append_option(args, "--min-baseline-ratio", min_baseline_ratio)
    if trace:
        args.append("--trace")
    if dry_run:
        args.append("--dry-run")
    if skip_autotune:
        args.append("--skip-autotune")
    if skip_profile_decode_step:
        args.append("--skip-profile-decode-step")
    if require_full_decode_step:
        args.append("--require-full-decode-step")
    if require_model_end_to_end:
        args.append("--require-model-end-to-end")
    if require_official_performance_parity:
        args.append("--require-official-performance-parity")
    if require_trace:
        args.append("--require-trace")
    if require_official_config_match:
        args.append("--require-official-config-match")
    if require_full_depth:
        args.append("--require-full-depth")
    if require_program_runtime_shape:
        args.append("--require-program-runtime-shape")
    if require_batch32_decode_step:
        args.append("--require-batch32-decode-step")
    if require_decode_shell_numeric_reference:
        args.append("--require-decode-shell-numeric-reference")
    if guard_device_busy:
        args.append("--guard-device-busy")
    if guard_device_health:
        args.append("--guard-device-health")
    if preflight_only:
        args.append("--preflight-only")
    return args


def real_decode_reproducibility(
    *,
    validation_args: list[str],
    preflight_args: list[str],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "validation_cli_args": validation_args,
        "validation_cli_command": shell_command(validation_args),
        "preflight_cli_args": preflight_args,
        "preflight_cli_command": shell_command(preflight_args),
        "artifact_index": artifact_index(artifact_paths),
    }


def step_names_with_status(
    results: Any,
    *,
    status: str | None = None,
    failing: bool = False,
) -> list[str]:
    if not isinstance(results, dict):
        return []
    names = []
    passing_statuses = {"pass", "dry_run", "skipped", "pending"}
    for name, value in results.items():
        value = str(value)
        if status is not None and value == status:
            names.append(str(name))
        elif failing and value not in passing_statuses:
            names.append(str(name))
    return names


def tensorization_evidence(step: dict[str, Any]) -> dict[str, Any]:
    tensorization = step_tensorization_summary(step)
    return {
        "status": tensorization.get("status"),
        "roles": tensorization.get("roles"),
        "tensor_count": tensorization.get("tensor_count"),
        "target_dtype_counts": tensorization.get("target_dtype_counts", {}),
        "layout_counts": tensorization.get("layout_counts", {}),
        "memory_config_counts": tensorization.get("memory_config_counts", {}),
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
        "ttnn_dtype_counts": tensorization.get("ttnn_dtype_counts", {}),
        "ttnn_layout_counts": tensorization.get("ttnn_layout_counts", {}),
        "ttnn_memory_config_counts": tensorization.get(
            "ttnn_memory_config_counts",
            {},
        ),
        "tensor_paths": tensorization.get("tensor_paths", []),
        "required_tensorized_tensor_paths": step.get(
            "required_tensorized_tensor_paths",
            [],
        ),
        "missing_required_tensorized_tensor_paths": step.get(
            "missing_required_tensorized_tensor_paths",
            [],
        ),
        "physical_shape_mismatches": (
            tensorized_physical_shape_mismatches(tensorization)
        ),
        "key_paths": tensorization.get("key_paths", []),
        "key_tensors": tensorization.get("key_tensors", {}),
    }


def reference_summary(runtime_report: dict[str, Any]) -> dict[str, Any]:
    reference = runtime_report.get("reference") or {}
    checks = reference.get("checks") or []
    return {
        "reference_status": reference.get("status"),
        "reference_kind": reference.get("kind"),
        "reference_planned_ops": reference.get("planned_ops"),
        "reference_planned_observed_ops": reference.get(
            "planned_observed_ops"
        ),
        "reference_observed_ops": reference.get("observed_ops"),
        "reference_failed_checks": [
            check.get("name")
            for check in checks
            if isinstance(check, dict) and not check.get("passed")
        ],
    }


def candidate_reference_status_counts(
    report: dict[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in report.get("candidates", []):
        status = candidate.get("reference_status")
        if status is None:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _prefill_cache_population_diagnostics(
    cache_population: Any,
) -> list[dict[str, Any]]:
    if not isinstance(cache_population, list):
        return []
    diagnostics = []
    for entry in cache_population:
        if not isinstance(entry, dict):
            continue
        diagnostics.append(
            {
                "layer_id": entry.get("layer_id"),
                "status": entry.get("status"),
                "write_policy": entry.get("write_policy"),
                "update_shape_layout": entry.get("update_shape_layout"),
                "key_update_shape": entry.get("key_update_shape"),
                "value_update_shape": entry.get("value_update_shape"),
                "key_cache_shape": entry.get("key_cache_shape"),
                "value_cache_shape": entry.get("value_cache_shape"),
                "page_table_shape": entry.get("page_table_shape"),
                "planned_user_count": entry.get("planned_user_count"),
                "filled_user_count": entry.get("filled_user_count"),
            }
        )
    return diagnostics


def dump_validation_report(report: dict[str, Any], out: str | Path) -> None:
    write_json_report(Path(out), report)


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)

def real_decode_evidence_manifest(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    steps = report.get("steps", {})
    official_config_diff = steps.get("official_config_diff", {})
    materialize = steps.get("materialize_parameters", {})
    decode_shell = steps.get("decode_shell", {})
    attention_primitives = steps.get("attention_primitives", {})
    attention_layer = steps.get("attention_layer", {})
    single_layer = steps.get("single_layer_decode", {})
    smoke = steps.get("smoke_decode_step", {})
    profile = steps.get("profile_decode_step", {})
    prompt_loop = steps.get("prompt_decode_loop", {})
    generate_step = steps.get("generate_prefill_decode", {})
    profile_generate = steps.get("profile_generate", {})
    generate_depth_sweep = steps.get("generate_depth_sweep", {})
    depth_sweep = steps.get("decode_depth_sweep", {})
    autotune = steps.get("decode_step_autotune", {})
    acceptance = report.get("acceptance", {})
    decode_step_contract = report.get("decode_step_contract") or {}
    failed_checks = [
        check
        for check in acceptance.get("checks", [])
        if isinstance(check, dict) and not check.get("passed")
    ]
    if report.get("dry_run"):
        status = "dry_run"
    elif report.get("status") == "pass" and acceptance.get("passed"):
        status = "accepted"
    else:
        status = "incomplete"
    results = report.get("results") or {}
    acceptance_scope = real_decode_acceptance_scope(report, acceptance)
    model_readiness = model_end_to_end_readiness(
        report,
        acceptance_scope,
    )

    return {
        "schema_version": 1,
        "status": status,
        "acceptance_scope": acceptance_scope,
        "model_end_to_end_readiness": model_readiness,
        "reproducibility": report.get("reproducibility"),
        "final_acceptance_plan": report.get("final_acceptance_plan"),
        "runtime_diagnostics": report.get("runtime_diagnostics")
        or real_decode_runtime_diagnostics(report),
        "acceptance_gate_matrix": final_acceptance_gate_matrix(
            report,
            acceptance,
        ),
        "validation": {
            "command": report.get("command"),
            "status": report.get("status"),
            "dry_run": report.get("dry_run"),
            "program_dir": report.get("program_dir"),
            "model_path": report.get("model_path"),
            "official_config": report.get("official_config"),
            "performance_baselines": report.get("performance_baselines"),
            "baseline_reference": report.get("baseline_reference"),
            "baseline_reference_entry": _performance_baseline_entry_summary(
                report.get("baseline_reference_entry")
            ),
            "decode_step_search_space": report.get(
                "decode_step_search_space"
            ),
            "decode_step_search_space_is_default": report.get(
                "decode_step_search_space_is_default"
            ),
            "program_num_layers": report.get("program_num_layers"),
            "program_batch_size": report.get("program_batch_size"),
            "program_cache_len": report.get("program_cache_len"),
            "program_seq_len": report.get("program_seq_len"),
            "program_hidden_size": report.get("program_hidden_size"),
            "program_intermediate_size": report.get(
                "program_intermediate_size"
            ),
            "program_vocab_size": report.get("program_vocab_size"),
            "program_num_attention_heads": report.get(
                "program_num_attention_heads"
            ),
            "program_num_key_value_heads": report.get(
                "program_num_key_value_heads"
            ),
            "program_head_dim": report.get("program_head_dim"),
            "program_generation": report.get("program_generation"),
            "program_kv_cache": report.get("program_kv_cache"),
            "layers": report.get("layers"),
            "requested_batch_size": report.get("requested_batch_size"),
            "requested_cache_len": report.get("requested_cache_len"),
            "batch_size": report.get("batch_size"),
            "cache_len": report.get("cache_len"),
            "max_new_tokens": report.get("max_new_tokens"),
            "prefill_len": report.get("prefill_len"),
            "device": report.get("device"),
            "device_id": report.get("device_id"),
            "dtype_seed": report.get("dtype_seed"),
            "trace_enabled": report.get("trace_enabled"),
            "trace_iterations": report.get("trace_iterations"),
            "metric": report.get("metric"),
            "skip_autotune": report.get("skip_autotune"),
            "baseline_tokens_per_second_per_user": report.get(
                "baseline_tokens_per_second_per_user"
            ),
            "min_baseline_ratio": report.get("min_baseline_ratio"),
            "require_official_config_match": report.get(
                "require_official_config_match"
            ),
            "require_official_performance_parity": report.get(
                "require_official_performance_parity"
            ),
            "require_full_depth": report.get("require_full_depth"),
            "require_program_runtime_shape": report.get(
                "require_program_runtime_shape"
            ),
            "require_batch32_decode_step": report.get(
                "require_batch32_decode_step"
            ),
            "require_full_decode_step": report.get(
                "require_full_decode_step"
            ),
            "require_model_end_to_end": report.get(
                "require_model_end_to_end"
            ),
            "results": dict(results),
            "failed_steps": step_names_with_status(
                results,
                failing=True,
            ),
            "skipped_steps": step_names_with_status(
                results,
                status="skipped",
            ),
        },
        "requirements": {
            "require_official_config_match": report.get(
                "require_official_config_match"
            ),
            "require_official_performance_parity": report.get(
                "require_official_performance_parity"
            ),
            "require_full_depth": report.get("require_full_depth"),
            "require_program_runtime_shape": report.get(
                "require_program_runtime_shape"
            ),
            "require_batch32_decode_step": report.get(
                "require_batch32_decode_step"
            ),
            "require_full_decode_step": report.get(
                "require_full_decode_step"
            ),
            "require_model_end_to_end": report.get(
                "require_model_end_to_end"
            ),
            "require_trace": report.get("require_trace"),
            "min_tokens_per_second_per_user": report.get(
                "min_tokens_per_second_per_user"
            ),
            "baseline_tokens_per_second_per_user": report.get(
                "baseline_tokens_per_second_per_user"
            ),
            "baseline_reference": report.get("baseline_reference"),
            "baseline_reference_entry": _performance_baseline_entry_summary(
                report.get("baseline_reference_entry")
            ),
            "performance_baselines": report.get("performance_baselines"),
            "min_baseline_ratio": report.get("min_baseline_ratio"),
            "decode_shell_pcc_threshold": report.get(
                "decode_shell_pcc_threshold"
            ),
            "require_decode_shell_numeric_reference": report.get(
                "require_decode_shell_numeric_reference"
            ),
        },
        "artifacts": [
            artifact_evidence(name, path)
            for name, path in paths.items()
            if name != "evidence_manifest"
        ],
        "device_evidence": {
            "device_preflight_diagnostics": report.get(
                "device_preflight_diagnostics"
            ),
            "tenstorrent_setup_environment": report.get(
                "tenstorrent_setup_environment"
            ),
            "tenstorrent_device_environment": report.get(
                "tenstorrent_device_environment"
            ),
            "tenstorrent_process_environment": report.get(
                "tenstorrent_process_environment"
            ),
            "tenstorrent_runtime_health": report.get(
                "tenstorrent_runtime_health"
            ),
            "attention_primitives_ttnn_environment": (
                attention_primitives.get("ttnn_environment")
            ),
            "attention_layer_ttnn_environment": attention_layer.get(
                "ttnn_environment"
            ),
            "single_layer_ttnn_environment": single_layer.get(
                "ttnn_environment"
            ),
            "smoke_ttnn_environment": smoke.get("ttnn_environment"),
            "profile_ttnn_environment": profile.get("ttnn_environment"),
            "prompt_decode_loop_ttnn_environment": prompt_loop.get(
                "ttnn_environment"
            ),
            "generate_prefill_decode_ttnn_environment": generate_step.get(
                "ttnn_environment"
            ),
            "profile_generate_ttnn_environment": profile_generate.get(
                "ttnn_environment"
            ),
            "generate_depth_sweep_ttnn_environment": (
                generate_depth_sweep.get("ttnn_environment")
            ),
        },
        "performance_evidence": {
            "throughput_baseline": _throughput_baseline_summary(
                report,
                profile,
            ),
            "performance_gap_summary": _performance_gap_summary(
                report,
                profile,
            ),
            "generate_milestones": _validate_real_generate_milestones(
                profile_generate,
                generate_depth_sweep,
            ),
            "profile_generate": {
                "status": profile_generate.get("status"),
                "runtime_status": profile_generate.get("runtime_status"),
                "profile_generate_report": profile_generate.get(
                    "profile_generate_report"
                ),
                "generate_report": profile_generate.get("generate_report"),
                "prefill_ms": profile_generate.get("prefill_ms"),
                "decode_step_ms_mean": profile_generate.get(
                    "decode_step_ms_mean"
                ),
                "tokens_per_second_per_user": profile_generate.get(
                    "tokens_per_second_per_user"
                ),
                "aggregate_tokens_per_second": profile_generate.get(
                    "aggregate_tokens_per_second"
                ),
                "throughput_summary": profile_generate.get(
                    "throughput_summary"
                ),
                "performance_milestones": profile_generate.get(
                    "performance_milestones"
                ),
                "sections": profile_generate.get("sections"),
                "per_layer": profile_generate.get("per_layer"),
                "host_copy_profile": profile_generate.get(
                    "host_copy_profile"
                ),
                "decode_token_runtime_handoff": profile_generate.get(
                    "decode_token_runtime_handoff"
                ),
                "decode_token_host_roundtrip_per_step": profile_generate.get(
                    "decode_token_host_roundtrip_per_step"
                ),
                "host_token_materialization_for_reporting_only": (
                    profile_generate.get(
                        "host_token_materialization_for_reporting_only"
                    )
                ),
                "section_profile": profile_generate.get("section_profile"),
                "acceptance": profile_generate.get("acceptance"),
                "official_performance_parity_claimed": profile_generate.get(
                    "official_performance_parity_claimed"
                ),
            },
        },
        "decode_step_contract": decode_step_contract,
        "config_evidence": {
            "official_config_diff": {
                "status": official_config_diff.get("status"),
                "report": official_config_diff.get("official_config_diff"),
                "official_config": official_config_diff.get(
                    "official_config"
                ),
                "diff_status": official_config_diff.get("diff_status"),
                "issue_count": official_config_diff.get("issue_count"),
                "missing_count": official_config_diff.get("missing_count"),
                "mismatch_count": official_config_diff.get("mismatch_count"),
                "extra_count": official_config_diff.get("extra_count"),
                "matching_count": official_config_diff.get("matching_count"),
                "ours_source_format": official_config_diff.get(
                    "ours_source_format"
                ),
                "official_source_format": official_config_diff.get(
                    "official_source_format"
                ),
                "official_source": official_config_diff.get(
                    "official_source"
                ),
                "gap_summary": official_config_diff.get("gap_summary"),
                "official_required_field_coverage": (
                    official_config_diff.get(
                        "official_required_field_coverage"
                    )
                ),
                "required_parity_fields": official_config_diff.get(
                    "required_parity_fields",
                    [],
                ),
                "sections": official_config_diff.get("sections", []),
                "section_statuses": official_config_diff.get(
                    "section_statuses",
                    {},
                ),
            },
        },
        "weight_evidence": {
            "materialization": {
                "status": materialize.get("status"),
                "materialized_layer_ids": materialize.get(
                    "materialized_layer_ids"
                ),
                "tensor_count": materialize.get("tensor_count"),
                "lm_head_split_count": materialize.get("lm_head_split_count"),
                "required_tensor_paths": materialize.get(
                    "required_tensor_paths",
                    [],
                ),
                "missing_required_tensor_paths": materialize.get(
                    "missing_required_tensor_paths",
                    [],
                ),
                "materialized_tensor_shape_mismatches": materialize.get(
                    "materialized_tensor_shape_mismatches",
                    [],
                ),
                "key_tensors": materialize.get("key_tensors", {}),
            },
            "decode_shell_tensorization": tensorization_evidence(
                decode_shell
            ),
            "single_layer_tensorization": tensorization_evidence(
                single_layer
            ),
            "smoke_tensorization": tensorization_evidence(smoke),
            "profile_tensorization": tensorization_evidence(profile),
            "prompt_decode_loop_tensorization": tensorization_evidence(
                prompt_loop
            ),
            "generate_prefill_decode_tensorization": tensorization_evidence(
                generate_step
            ),
            "profile_generate_tensorization": tensorization_evidence(
                profile_generate
            ),
        },
        "runtime_evidence": {
            "decode_shell": {
                "status": decode_shell.get("status"),
                "runtime_status": decode_shell.get("runtime_status"),
                "error": decode_shell.get("error"),
                "message": decode_shell.get("message"),
                "detail": decode_shell.get("detail"),
                "layers": decode_shell.get("layers"),
                "parameter_source": decode_shell.get("parameter_source"),
                "input_source": decode_shell.get("input_source"),
                "runtime_input_tensor_count": decode_shell.get(
                    "runtime_input_tensor_count"
                ),
                "reference_status": decode_shell.get("reference_status"),
                "numeric_reference_status": decode_shell.get(
                    "numeric_reference_status"
                ),
                "numeric_reference_kind": decode_shell.get(
                    "numeric_reference_kind"
                ),
                "numeric_reference_passed": decode_shell.get(
                    "numeric_reference_passed"
                ),
                "pcc": decode_shell.get("pcc"),
                "pcc_threshold": decode_shell.get("pcc_threshold"),
                "numeric_reference_failed_checks": decode_shell.get(
                    "numeric_reference_failed_checks",
                    [],
                ),
                "reference_planned_ops": decode_shell.get(
                    "reference_planned_ops"
                ),
                "reference_observed_ops": decode_shell.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": decode_shell.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "attention_primitives": {
                "status": attention_primitives.get("status"),
                "runtime_status_counts": attention_primitives.get(
                    "runtime_status_counts",
                    {},
                ),
                "primitive_count": attention_primitives.get(
                    "primitive_count"
                ),
                "primitive_sequence": attention_primitives.get(
                    "primitive_sequence"
                ),
                "primitive_reports": attention_primitives.get(
                    "primitive_reports",
                    [],
                ),
            },
            "attention_layer": {
                "status": attention_layer.get("status"),
                "runtime_status": attention_layer.get("runtime_status"),
                "layer": attention_layer.get("layer"),
                "batch_size": attention_layer.get("batch_size"),
                "cache_len": attention_layer.get("cache_len"),
                "hidden_size": attention_layer.get("hidden_size"),
                "num_heads": attention_layer.get("num_heads"),
                "num_kv_heads": attention_layer.get("num_kv_heads"),
                "head_dim": attention_layer.get("head_dim"),
                "latency_ms": attention_layer.get("latency_ms"),
                "primitive_count": attention_layer.get("primitive_count"),
                "primitive_sequence": attention_layer.get(
                    "primitive_sequence"
                ),
                "primitive_reports": attention_layer.get(
                    "primitive_reports",
                    [],
                ),
                "output_shapes": attention_layer.get("output_shapes"),
                "tensor_conversion_count": attention_layer.get(
                    "tensor_conversion_count"
                ),
                "memory_config_conversion_count": attention_layer.get(
                    "memory_config_conversion_count"
                ),
                "reference_status": attention_layer.get("reference_status"),
                "reference_kind": attention_layer.get("reference_kind"),
                "reference_planned_ops": attention_layer.get(
                    "reference_planned_ops"
                ),
                "reference_planned_observed_ops": attention_layer.get(
                    "reference_planned_observed_ops"
                ),
                "reference_observed_ops": attention_layer.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": attention_layer.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "single_layer_decode": {
                "status": single_layer.get("status"),
                "runtime_status": single_layer.get("runtime_status"),
                "layers": single_layer.get("layers"),
                "batch_size": single_layer.get("batch_size"),
                "cache_len": single_layer.get("cache_len"),
                "parameter_source": single_layer.get("parameter_source"),
                "input_source": single_layer.get("input_source"),
                "synthetic_runtime_input_tensor_count": single_layer.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": single_layer.get(
                    "synthetic_rotary_tensor_count"
                ),
                "tensor_conversion_count": single_layer.get(
                    "tensor_conversion_count"
                ),
                "input_shapes": single_layer.get("input_shapes"),
                "kv_cache": single_layer.get("kv_cache"),
                "output_shapes": single_layer.get("output_shapes"),
                "trace_status": single_layer.get("trace_status"),
                "trace": single_layer.get("trace"),
                "reference_status": single_layer.get("reference_status"),
                "reference_kind": single_layer.get("reference_kind"),
                "reference_planned_ops": single_layer.get(
                    "reference_planned_ops"
                ),
                "reference_observed_ops": single_layer.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": single_layer.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "smoke_decode_step": {
                "status": smoke.get("status"),
                "runtime_status": smoke.get("runtime_status"),
                "layers": smoke.get("layers"),
                "batch_size": smoke.get("batch_size"),
                "cache_len": smoke.get("cache_len"),
                "parameter_source": smoke.get("parameter_source"),
                "input_source": smoke.get("input_source"),
                "synthetic_runtime_input_tensor_count": smoke.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": smoke.get(
                    "synthetic_rotary_tensor_count"
                ),
                "tensor_conversion_count": smoke.get(
                    "tensor_conversion_count"
                ),
                "input_shapes": smoke.get("input_shapes"),
                "kv_cache": smoke.get("kv_cache"),
                "output_shapes": smoke.get("output_shapes"),
                "trace_status": smoke.get("trace_status"),
                "trace": smoke.get("trace"),
                "reference_status": smoke.get("reference_status"),
                "reference_kind": smoke.get("reference_kind"),
                "reference_planned_ops": smoke.get("reference_planned_ops"),
                "reference_observed_ops": smoke.get("reference_observed_ops"),
                "reference_failed_checks": smoke.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "profile_decode_step": {
                "status": profile.get("status"),
                "runtime_status": profile.get("runtime_status"),
                "layers": profile.get("layers"),
                "batch_size": profile.get("batch_size"),
                "cache_len": profile.get("cache_len"),
                "parameter_source": profile.get("parameter_source"),
                "input_source": profile.get("input_source"),
                "synthetic_runtime_input_tensor_count": profile.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": profile.get(
                    "synthetic_rotary_tensor_count"
                ),
                "tensor_conversion_count": profile.get(
                    "tensor_conversion_count"
                ),
                "tensor_conversion_ms": profile.get("tensor_conversion_ms"),
                "latency_ms": profile.get("latency_ms"),
                "section_latency_ms": profile.get("section_latency_ms"),
                "layer_profiles": profile.get("layer_profiles", []),
                "lm_head_profile": profile.get("lm_head_profile"),
                "input_shapes": profile.get("input_shapes"),
                "kv_cache": profile.get("kv_cache"),
                "output_shapes": profile.get("output_shapes"),
                "trace_status": profile.get("trace_status"),
                "trace": profile.get("trace"),
                "reference_status": profile.get("reference_status"),
                "reference_kind": profile.get("reference_kind"),
                "reference_planned_ops": profile.get("reference_planned_ops"),
                "reference_observed_ops": profile.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": profile.get(
                    "reference_failed_checks",
                    [],
                ),
                "throughput_summary": profile.get("throughput_summary"),
                "bottleneck_summary": profile.get("bottleneck_summary"),
                "max_section": profile.get("max_section"),
            },
            "prompt_decode_loop": {
                "status": prompt_loop.get("status"),
                "runtime_status": prompt_loop.get("runtime_status"),
                "decode_loop_runtime_owned": prompt_loop.get(
                    "decode_loop_runtime_owned"
                ),
                "runtime_owner": prompt_loop.get("runtime_owner"),
                "decode_steps": prompt_loop.get("decode_steps"),
                "layers": prompt_loop.get("layers"),
                "batch_size": prompt_loop.get("batch_size"),
                "cache_len": prompt_loop.get("cache_len"),
                "parameter_source": prompt_loop.get("parameter_source"),
                "input_source": prompt_loop.get("input_source"),
                "synthetic_runtime_input_tensor_count": prompt_loop.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": prompt_loop.get(
                    "synthetic_rotary_tensor_count"
                ),
                "prompt_runtime_input_tensor_count": prompt_loop.get(
                    "prompt_runtime_input_tensor_count"
                ),
                "decode_runtime_state_input_tensor_count": prompt_loop.get(
                    "decode_runtime_state_input_tensor_count"
                ),
                "rotary_runtime_input_tensor_count": prompt_loop.get(
                    "rotary_runtime_input_tensor_count"
                ),
                "kv_cache_runtime_input_tensor_count": prompt_loop.get(
                    "kv_cache_runtime_input_tensor_count"
                ),
                "tensor_conversion_count": prompt_loop.get(
                    "tensor_conversion_count"
                ),
                "latency_ms": prompt_loop.get("latency_ms"),
                "throughput_summary": prompt_loop.get("throughput_summary"),
                "output_shapes": prompt_loop.get("output_shapes"),
                "trace_status": prompt_loop.get("trace_status"),
                "trace": prompt_loop.get("trace"),
                "reference_status": prompt_loop.get("reference_status"),
                "reference_kind": prompt_loop.get("reference_kind"),
                "reference_failed_checks": prompt_loop.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "generate_prefill_decode": {
                "status": generate_step.get("status"),
                "generate_report": generate_step.get("generate_report"),
                "runtime_status": generate_step.get("runtime_status"),
                "prefill_status": generate_step.get("prefill_status"),
                "generate_runtime_owned": generate_step.get(
                    "generate_runtime_owned"
                ),
                "decode_loop_runtime_owned": generate_step.get(
                    "decode_loop_runtime_owned"
                ),
                "kv_cache_source": generate_step.get("kv_cache_source"),
                "model_semantics": generate_step.get("model_semantics"),
                "runtime_owner": generate_step.get("runtime_owner"),
                "layers": generate_step.get("layers"),
                "batch_size": generate_step.get("batch_size"),
                "cache_len": generate_step.get("cache_len"),
                "prefill_len": generate_step.get("prefill_len"),
                "max_new_tokens": generate_step.get("max_new_tokens"),
                "decode_steps": generate_step.get("decode_steps"),
                "generated_token_budget": generate_step.get(
                    "generated_token_budget"
                ),
                "parameter_source": generate_step.get("parameter_source"),
                "input_source": generate_step.get("input_source"),
                "synthetic_runtime_input_tensor_count": generate_step.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": generate_step.get(
                    "synthetic_rotary_tensor_count"
                ),
                "synthetic_kv_cache_tensor_count": generate_step.get(
                    "synthetic_kv_cache_tensor_count"
                ),
                "prompt_runtime_input_tensor_count": generate_step.get(
                    "prompt_runtime_input_tensor_count"
                ),
                "decode_runtime_state_input_tensor_count": generate_step.get(
                    "decode_runtime_state_input_tensor_count"
                ),
                "rotary_runtime_input_tensor_count": generate_step.get(
                    "rotary_runtime_input_tensor_count"
                ),
                "kv_cache_runtime_input_tensor_count": generate_step.get(
                    "kv_cache_runtime_input_tensor_count"
                ),
                "runtime_context": generate_step.get("runtime_context"),
                "decode_token_runtime_handoff": generate_step.get(
                    "decode_token_runtime_handoff"
                ),
                "decode_token_host_roundtrip_per_step": generate_step.get(
                    "decode_token_host_roundtrip_per_step"
                ),
                "host_token_materialization_for_reporting_only": (
                    generate_step.get(
                        "host_token_materialization_for_reporting_only"
                    )
                ),
                "prefill_cache_population": (
                    (generate_step.get("prefill") or {}).get(
                        "cache_population"
                    )
                    if isinstance(generate_step.get("prefill"), dict)
                    else None
                ),
                "prefill_cache_population_diagnostics": (
                    _prefill_cache_population_diagnostics(
                        (generate_step.get("prefill") or {}).get(
                            "cache_population"
                        )
                        if isinstance(generate_step.get("prefill"), dict)
                        else []
                    )
                ),
                "end_to_end_contract": generate_step.get(
                    "end_to_end_contract"
                ),
                "end_to_end_failed_checks": (
                    (generate_step.get("end_to_end_contract") or {}).get(
                        "failed_checks",
                        [],
                    )
                    if isinstance(
                        generate_step.get("end_to_end_contract"),
                        dict,
                    )
                    else []
                ),
                "generated_token_ids": generate_step.get(
                    "generated_token_ids"
                ),
                "generated_token_id_source": generate_step.get(
                    "generated_token_id_source"
                ),
                "token_materialization_status": generate_step.get(
                    "token_materialization_status"
                ),
                "generated_text": generate_step.get("generated_text"),
                "generated_text_status": generate_step.get(
                    "generated_text_status"
                ),
                "generated_text_source": generate_step.get(
                    "generated_text_source"
                ),
                "output_shapes": generate_step.get("output_shapes"),
                "latency_ms": generate_step.get("latency_ms"),
                "throughput_summary": generate_step.get(
                    "throughput_summary"
                ),
                "host_copy_profile": generate_step.get("host_copy_profile"),
                "section_profile": generate_step.get("section_profile"),
                "trace_status": generate_step.get("trace_status"),
                "reference_status": generate_step.get("reference_status"),
                "reference_kind": generate_step.get("reference_kind"),
                "reference_failed_checks": generate_step.get(
                    "reference_failed_checks",
                    [],
                ),
                "error": generate_step.get("error"),
                "detail": generate_step.get("detail"),
                "failure_diagnostics": generate_step.get(
                    "failure_diagnostics"
                ),
            },
            "profile_generate": {
                "status": profile_generate.get("status"),
                "profile_generate_report": profile_generate.get(
                    "profile_generate_report"
                ),
                "generate_report": profile_generate.get("generate_report"),
                "runtime_status": profile_generate.get("runtime_status"),
                "generate_status": profile_generate.get("generate_status"),
                "generate_passed": profile_generate.get("generate_passed"),
                "program_num_layers": profile_generate.get(
                    "program_num_layers"
                ),
                "prefill_status": profile_generate.get("prefill_status"),
                "kv_cache_source": profile_generate.get("kv_cache_source"),
                "model_semantics": profile_generate.get("model_semantics"),
                "runtime_owner": profile_generate.get("runtime_owner"),
                "generate_runtime_owned": profile_generate.get(
                    "generate_runtime_owned"
                ),
                "decode_loop_runtime_owned": profile_generate.get(
                    "decode_loop_runtime_owned"
                ),
                "runtime_context": profile_generate.get("runtime_context"),
                "decode_token_runtime_handoff": profile_generate.get(
                    "decode_token_runtime_handoff"
                ),
                "decode_token_host_roundtrip_per_step": profile_generate.get(
                    "decode_token_host_roundtrip_per_step"
                ),
                "host_token_materialization_for_reporting_only": (
                    profile_generate.get(
                        "host_token_materialization_for_reporting_only"
                    )
                ),
                "end_to_end_contract": profile_generate.get(
                    "end_to_end_contract"
                ),
                "synthetic_runtime_input_tensor_count": profile_generate.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "synthetic_rotary_tensor_count": profile_generate.get(
                    "synthetic_rotary_tensor_count"
                ),
                "synthetic_kv_cache_tensor_count": profile_generate.get(
                    "synthetic_kv_cache_tensor_count"
                ),
                "prompt_runtime_input_tensor_count": profile_generate.get(
                    "prompt_runtime_input_tensor_count"
                ),
                "decode_runtime_state_input_tensor_count": (
                    profile_generate.get(
                        "decode_runtime_state_input_tensor_count"
                    )
                ),
                "rotary_runtime_input_tensor_count": profile_generate.get(
                    "rotary_runtime_input_tensor_count"
                ),
                "kv_cache_runtime_input_tensor_count": profile_generate.get(
                    "kv_cache_runtime_input_tensor_count"
                ),
                "generated_text_status": profile_generate.get(
                    "generated_text_status"
                ),
                "generated_token_count_by_user": profile_generate.get(
                    "generated_token_count_by_user"
                ),
                "latency_ms": profile_generate.get("latency_ms"),
                "prefill_ms": profile_generate.get("prefill_ms"),
                "decode_step_ms_mean": profile_generate.get(
                    "decode_step_ms_mean"
                ),
                "decode_step_ms_samples": profile_generate.get(
                    "decode_step_ms_samples",
                    [],
                ),
                "tokens_per_second_per_user": profile_generate.get(
                    "tokens_per_second_per_user"
                ),
                "aggregate_tokens_per_second": profile_generate.get(
                    "aggregate_tokens_per_second"
                ),
                "throughput_summary": profile_generate.get(
                    "throughput_summary"
                ),
                "performance_milestones": profile_generate.get(
                    "performance_milestones"
                ),
                "sections": profile_generate.get("sections"),
                "per_layer": profile_generate.get("per_layer"),
                "host_copy_ms": profile_generate.get("host_copy_ms"),
                "host_copy_profile": profile_generate.get(
                    "host_copy_profile"
                ),
                "section_profile": profile_generate.get("section_profile"),
                "acceptance": profile_generate.get("acceptance"),
                "official_performance_parity_claimed": profile_generate.get(
                    "official_performance_parity_claimed"
                ),
                "reason": profile_generate.get("reason"),
                "error": profile_generate.get("error"),
            },
            "generate_depth_sweep": {
                "status": generate_depth_sweep.get("status"),
                "generate_depth_sweep_report": generate_depth_sweep.get(
                    "generate_depth_sweep_report"
                ),
                "reports_dir": generate_depth_sweep.get("reports_dir"),
                "depths": generate_depth_sweep.get("depths"),
                "depth_count": generate_depth_sweep.get("depth_count"),
                "max_depth": generate_depth_sweep.get("max_depth"),
                "covered_full_depth": generate_depth_sweep.get(
                    "covered_full_depth"
                ),
                "require_full_depth": generate_depth_sweep.get(
                    "require_full_depth"
                ),
                "status_counts": generate_depth_sweep.get(
                    "status_counts",
                    {},
                ),
                "prefill_status_counts": generate_depth_sweep.get(
                    "prefill_status_counts",
                    {},
                ),
                "generated_text_status_counts": generate_depth_sweep.get(
                    "generated_text_status_counts",
                    {},
                ),
                "model_semantics_counts": generate_depth_sweep.get(
                    "model_semantics_counts",
                    {},
                ),
                "passed_depth_count": generate_depth_sweep.get(
                    "passed_depth_count"
                ),
                "failed_depths": generate_depth_sweep.get(
                    "failed_depths",
                    [],
                ),
                "failed_depth_diagnostics": generate_depth_sweep.get(
                    "failed_depth_diagnostics",
                    [],
                ),
                "acceptance": generate_depth_sweep.get("acceptance"),
                "records": generate_depth_sweep.get("records", []),
            },
            "decode_depth_sweep": {
                "status": depth_sweep.get("status"),
                "depths": depth_sweep.get("depths"),
                "depth_count": depth_sweep.get("depth_count"),
                "max_depth": depth_sweep.get("max_depth"),
                "covered_full_depth": depth_sweep.get("covered_full_depth"),
                "require_full_depth": depth_sweep.get(
                    "require_full_depth"
                ),
                "status_counts": depth_sweep.get("status_counts", {}),
                "reference_status_counts": depth_sweep.get(
                    "reference_status_counts",
                    {},
                ),
                "trace_status_counts": depth_sweep.get(
                    "trace_status_counts",
                    {},
                ),
                "passed_depth_count": depth_sweep.get("passed_depth_count"),
                "failed_depths": depth_sweep.get("failed_depths", []),
                "acceptance": depth_sweep.get("acceptance"),
                "records": depth_sweep.get("records", []),
            },
            "decode_step_autotune": {
                "status": autotune.get("status"),
                "candidate_count": autotune.get("candidate_count"),
                "passed_candidate_count": autotune.get(
                    "passed_candidate_count"
                ),
                "failed_candidate_count": autotune.get(
                    "failed_candidate_count"
                ),
                "best": autotune.get("best"),
                "metric": autotune.get("metric"),
                "best_reference_status": autotune.get(
                    "best_reference_status"
                ),
                "best_trace_status": autotune.get("best_trace_status"),
                "best_parameter_source": autotune.get(
                    "best_parameter_source"
                ),
                "best_metric": autotune.get("best_metric"),
                "best_output_kind": autotune.get("best_output_kind"),
                "status_counts": autotune.get("status_counts", {}),
                "reference_status_counts": autotune.get(
                    "reference_status_counts",
                    {},
                ),
                "trace_status_counts": autotune.get(
                    "trace_status_counts",
                    {},
                ),
                "output_kind_counts": autotune.get(
                    "output_kind_counts",
                    {},
                ),
                "candidate_summaries": autotune.get(
                    "candidate_summaries",
                    [],
                ),
                "leaderboard": autotune.get("leaderboard", []),
                "best_candidate_summary": autotune.get(
                    "best_candidate_summary"
                ),
                "knob_coverage": autotune.get("knob_coverage"),
                "default_search_space": autotune.get(
                    "default_search_space"
                ),
                "all_knobs_varied": autotune.get("all_knobs_varied"),
                "missing_varied_knobs": autotune.get(
                    "missing_varied_knobs",
                    [],
                ),
                "search_space": autotune.get("search_space"),
            },
        },
        "acceptance": {
            "status": acceptance.get("status"),
            "passed": acceptance.get("passed"),
            "check_count": len(acceptance.get("checks", [])),
            "failed_checks": [check.get("name") for check in failed_checks],
        },
    }
