from __future__ import annotations

import gc
import json
import importlib
import py_compile
import shlex
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .codegen.artifacts import (
    prepare_offline_artifacts,
)
from .codegen.config_diff import (
    PARITY_SECTIONS,
    default_official_config_path as default_official_parity_config_path,
    diff_official_config,
    dump_config_diff,
)
from .codegen.config_emit import (
    dump_parameter_config,
    emit_parameter_config,
)
from .codegen.package import package_ttnn_direct_program
from .codegen.parameters import materialize_parameters_from_program
from .codegen.program import write_decode_program_bundle
from .codegen.ttnn_tensorizer import (
    LINEAR_WEIGHT_TRANSFORM,
    tensorize_parameters_from_program_dry_run,
)
from .runtime_environment import (
    collect_tenstorrent_device_environment,
    collect_tenstorrent_process_environment,
    collect_tenstorrent_setup_environment,
    collect_ttnn_runtime_health,
    collect_ttnn_environment,
)
from .decode_loop import run_prompt_decode_loop
from .diagnostics.legacy_validation import (
    diagnostic_excerpt as _diagnostic_excerpt,
    real_decode_runtime_diagnostics as _real_decode_runtime_diagnostics,
    recover_real_decode_process_failure,
    recover_real_decode_process_timeout,
    runtime_error_findings as _runtime_error_findings,
    tenstorrent_device_preflight_diagnostics as _tenstorrent_device_preflight_diagnostics,
    tenstorrent_device_preflight_diagnostics_from_report as _tenstorrent_device_preflight_diagnostics_from_report,
    tenstorrent_preflight_recommended_action as _tenstorrent_preflight_recommended_action,
    walk_strings as _walk_strings,
)
from .generate import run_generate, run_profile_generate
from .reports.evidence import (
    artifact_evidence as _artifact_evidence,
    artifact_index as _artifact_index,
    candidate_reference_status_counts as _candidate_reference_status_counts,
    dump_validation_report,
    reference_summary as _reference_summary,
    step_names_with_status as _step_names_with_status,
    tensorization_evidence as _tensorization_evidence,
    write_json_report as _write_json,
)
from .reports.performance import (
    OFFICIAL_PERFORMANCE_PARITY_METRIC,
    PROFILE_GENERATE_MILESTONE_IDS,
    default_performance_baselines_path,
    load_performance_baselines,
    official_performance_baseline_entry_complete as _official_performance_baseline_entry_complete,
    performance_gap_summary as _performance_gap_summary,
    performance_baseline_entry_complete as _performance_baseline_entry_complete,
    performance_baseline_entry_summary as _performance_baseline_entry_summary,
    profile_generate_milestones_complete as _profile_generate_milestones_complete,
    resolve_performance_baseline,
    throughput_baseline_summary as _throughput_baseline_summary,
    validate_real_generate_milestones as _validate_real_generate_milestones,
)
from .reports.schema import (
    acceptance_check as _acceptance_check,
    contains_all as _contains_all,
    field_keys as _field_keys,
    has_nonnegative_fields as _has_nonnegative_fields,
    int_equal as _int_equal,
    int_list as _int_list,
    int_list_contains as _int_list_contains,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    number_at_least as _number_at_least,
    numbers_equal as _numbers_equal,
    path_exists as _path_exists,
    path_exists_relative_to as _path_exists_relative_to,
    paths_exist as _paths_exist,
    paths_exist_relative_to as _paths_exist_relative_to,
    positive_count as _positive_count,
    positive_number as _positive_number,
    safe_int as _safe_int,
    status_count_matches_total as _status_count_matches_total,
)
from .reports.validation import (
    acceptance_check_passed as _acceptance_check_passed,
    final_acceptance_gate_matrix as _final_acceptance_gate_matrix,
)
from .search.decode_step_autotune import (
    DECODE_STEP_AUTOTUNE_KNOBS,
    run_decode_step_autotune,
)
from .search.decode_depth_sweep import run_decode_depth_sweep
from .search.generate_depth_sweep import run_generate_depth_sweep
from .search.report import dump_search_report
from .search.runner import run_lm_head_search
from .search.space import load_search_space
from .semantic.dump import dump_graph_json
from .semantic.graph import LlamaModelGraph
from .semantic.importer_hf_llama import import_hf_llama
from .smoke_attention_layer import ATTENTION_LAYER_OPS, run_smoke_attention_layer
from .smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
    run_smoke_attention_primitive,
)
from .smoke_decode_shell import run_smoke_decode_shell
from .smoke_single_layer_decode import (
    DECODE_PARAMETER_ROLES,
    profile_decode_step,
    run_smoke_decode_step,
    run_smoke_single_layer_decode,
)
from .templates.diff import (
    diff_plan_against_official,
    dump_plan_diff,
    load_official_template,
)
from .templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_template_config,
)


VALIDATION_STEPS = (
    "import_llama",
    "plan",
    "plan_diff",
    "emit_config",
    "prepare_artifacts",
    "build_program",
    "py_compile",
    "official_config_diff",
    "tensorize_parameters_dry_run",
    "decode_shell_dry_run",
    "attention_primitives_dry_run",
    "attention_layer_dry_run",
    "single_layer_decode_dry_run",
    "decode_step_smoke_dry_run",
    "decode_step_profile_dry_run",
    "search_dry_run",
    "decode_step_autotune_dry_run",
    "package_program",
)

REAL_DECODE_VALIDATION_STEPS = (
    "official_config_diff",
    "materialize_parameters",
    "decode_shell",
    "attention_primitives",
    "attention_layer",
    "single_layer_decode",
    "smoke_decode_step",
    "profile_decode_step",
    "prompt_decode_loop",
    "generate_prefill_decode",
    "profile_generate",
    "generate_depth_sweep",
    "decode_depth_sweep",
    "decode_step_autotune",
)

PROFILE_SECTION_LATENCY_KEYS = (
    "embedding_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
)

PROFILE_LAYER_LATENCY_KEYS = (
    "rms_norm_attn_ms",
    "attention_ms",
    "residual_add_attn_ms",
    "rms_norm_mlp_ms",
    "mlp_ms",
    "residual_add_mlp_ms",
    "total_ms",
)

PROFILE_BOTTLENECK_SECTION_KEYS = (
    "tensor_conversion_ms",
    "embedding_ms",
    "per_layer_attention_ms",
    "per_layer_mlp_ms",
    "layer_stack_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
    "trace_execute_ms",
)

PROFILE_GENERATE_SECTION_KEYS = (
    "prefill_ms",
    "decode_total_ms",
    "decode_step_ms_mean",
    "embedding_ms",
    "prefill_attention_ms",
    "decode_attention_ms",
    "mlp_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
)

def default_official_template_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "reference"
        / "official_llama31_decode_template.json"
    )


def default_search_space_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "search"
        / "spaces"
        / "lm_head_minimal.json"
    )


def default_decode_step_search_space_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "search"
        / "spaces"
        / "decode_step_minimal.json"
    )


def _same_path(lhs: Path, rhs: Path) -> bool:
    return lhs.resolve() == rhs.resolve()


def _shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _append_option(args: list[str], option: str, value: Any) -> None:
    if value is not None:
        args.extend([option, str(value)])


def _real_decode_cli_args(
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


def _real_decode_reproducibility(
    *,
    validation_args: list[str],
    preflight_args: list[str],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "validation_cli_args": validation_args,
        "validation_cli_command": _shell_command(validation_args),
        "preflight_cli_args": preflight_args,
        "preflight_cli_command": _shell_command(preflight_args),
        "artifact_index": _artifact_index(artifact_paths),
    }


def _real_decode_final_acceptance_plan(
    *,
    metric: str,
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
) -> dict[str, Any]:
    if require_official_performance_parity:
        target_scope = "official_performance_parity"
    elif require_model_end_to_end:
        target_scope = "model_end_to_end"
    elif require_full_decode_step:
        target_scope = "full_decode_step"
    else:
        target_scope = "bringup"

    runtime_steps = [
        step
        for step in REAL_DECODE_VALIDATION_STEPS
        if not (
            (skip_autotune and step == "decode_step_autotune")
            or (
                skip_profile_decode_step
                and step
                in {
                    "profile_decode_step",
                    "profile_generate",
                    "decode_depth_sweep",
                    "decode_step_autotune",
                }
            )
        )
    ]
    effective_requirements = {
        "require_full_decode_step": bool(require_full_decode_step),
        "require_model_end_to_end": bool(require_model_end_to_end),
        "require_official_performance_parity": bool(
            require_official_performance_parity
        ),
        "require_trace": bool(require_trace),
        "require_official_config_match": bool(require_official_config_match),
        "require_full_depth": bool(require_full_depth),
        "require_program_runtime_shape": bool(
            require_program_runtime_shape
        ),
        "require_batch32_decode_step": bool(require_batch32_decode_step),
        "require_decode_shell_numeric_reference": bool(
            require_decode_shell_numeric_reference
        ),
        "skip_profile_decode_step": bool(skip_profile_decode_step),
    }
    requested_flags = []
    for flag, enabled in (
        ("--require-full-decode-step", require_full_decode_step),
        ("--require-model-end-to-end", require_model_end_to_end),
        (
            "--require-official-performance-parity",
            require_official_performance_parity,
        ),
        ("--require-trace", require_trace),
        ("--require-official-config-match", require_official_config_match),
        ("--require-full-depth", require_full_depth),
        ("--require-program-runtime-shape", require_program_runtime_shape),
        ("--require-batch32-decode-step", require_batch32_decode_step),
        (
            "--require-decode-shell-numeric-reference",
            require_decode_shell_numeric_reference,
        ),
    ):
        if enabled:
            requested_flags.append(flag)

    full_decode_gate_names = []
    if require_full_decode_step:
        full_decode_gate_names = [
            "validation.full_depth_layers",
            "decode_depth_sweep.full_depth",
            "validation.program_batch_size",
            "validation.program_cache_len",
            "decode_step_contract.batch32",
            "single_layer_decode.trace_status",
            "smoke_decode_step.trace_status",
            "profile_decode_step.trace_status",
            "profile_decode_step.trace_profile",
            "decode_shell.numeric_reference",
        ]

    official_parity_gate_names = []
    if require_official_performance_parity:
        official_parity_gate_names = [
            "model_end_to_end_readiness.ready",
            "official_config_diff.official_reference_format",
            "official_config_diff.match",
            "profile_decode_step.baseline_reference",
            "profile_decode_step.official_baseline_reference",
            "profile_decode_step.official_min_baseline_ratio_positive",
            "profile_decode_step.min_baseline_ratio",
            "decode_step_autotune.metric",
            "decode_step_autotune.status",
            "decode_step_autotune.best_candidate_summary",
        ]

    model_end_to_end_gate_names = []
    if require_model_end_to_end or require_official_performance_parity:
        model_end_to_end_gate_names = [
            "model_end_to_end_readiness.ready",
        ]

    optional_gate_names = []
    if min_tokens_per_second_per_user is not None:
        optional_gate_names.append(
            "profile_decode_step.min_tokens_per_second_per_user"
        )
    if baseline_tokens_per_second_per_user is not None:
        optional_gate_names.append(
            "profile_decode_step.baseline_tokens_per_second_per_user"
        )
    if baseline_reference is not None and not require_official_performance_parity:
        optional_gate_names.append("profile_decode_step.baseline_reference")
    if min_baseline_ratio is not None and not require_official_performance_parity:
        optional_gate_names.append("profile_decode_step.min_baseline_ratio")

    return {
        "target_scope": target_scope,
        "required_runtime_steps": runtime_steps,
        "effective_requirements": effective_requirements,
        "requested_acceptance_flags": requested_flags,
        "full_decode_step_gate_names": full_decode_gate_names,
        "model_end_to_end_gate_names": model_end_to_end_gate_names,
        "official_performance_parity_gate_names": official_parity_gate_names,
        "optional_gate_names": optional_gate_names,
        "metric": metric,
        "thresholds": {
            "min_tokens_per_second_per_user": (
                min_tokens_per_second_per_user
            ),
            "min_baseline_ratio": min_baseline_ratio,
            "decode_shell_pcc_threshold": decode_shell_pcc_threshold,
        },
        "baseline": {
            "baseline_reference": baseline_reference,
            "baseline_tokens_per_second_per_user": (
                baseline_tokens_per_second_per_user
            ),
            "uses_reference_baseline": baseline_reference is not None,
        },
    }


def validate_direct(
    *,
    model_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    official_template_path: str | Path | None = None,
    official_config_path: str | Path | None = None,
    search_space_path: str | Path | None = None,
    decode_step_search_space_path: str | Path | None = None,
    metric: str = "latency_ms",
) -> dict[str, Any]:
    """Run all device-free TTNN Direct scaffold checks and write a report."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "validation_report.json"
    model_path = Path(model_path)
    config_path = Path(config_path)
    official_template_path = (
        Path(official_template_path)
        if official_template_path is not None
        else default_official_template_path()
    )
    official_config_path = (
        Path(official_config_path)
        if official_config_path is not None
        else default_official_parity_config_path()
    )
    search_space_path = (
        Path(search_space_path)
        if search_space_path is not None
        else default_search_space_path()
    )
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )
    decode_step_search_space_is_default = _same_path(
        decode_step_search_space_path,
        default_decode_step_search_space_path(),
    )

    paths = {
        "semantic_json": root / "semantic_graph.json",
        "execution_plan": root / "execution_plan.json",
        "plan_diff": root / "plan_diff.json",
        "official_config_diff": root / "official_config_diff.json",
        "parameter_config": root / "parameter_config.json",
        "artifacts_dir": root / "offline_artifacts",
        "program_dir": root / "program",
        "tensorize_report": root / "tensorize_report.json",
        "decode_shell_report": root / "decode_shell_report.json",
        "attention_primitives_dir": root / "attention_primitives",
        "attention_layer_report": root / "attention_layer_report.json",
        "single_layer_decode_report": root / "single_layer_decode_report.json",
        "decode_step_smoke_report": root / "decode_step_smoke_report.json",
        "decode_step_profile_report": root / "decode_step_profile_report.json",
        "search_report": root / "search_report.json",
        "search_candidates_dir": root / "search_candidates",
        "decode_step_autotune_report": root / "decode_step_autotune_report.json",
        "decode_step_autotune_candidates_dir": (
            root / "decode_step_autotune_candidates"
        ),
        "package_dir": root / "package",
        "report": report_path,
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "validate-direct",
        "status": "running",
        "model_path": str(model_path),
        "config": str(config_path),
        "out_dir": str(root),
        "official_template": str(official_template_path),
        "official_config": str(official_config_path),
        "search_space": str(search_space_path),
        "decode_step_search_space": str(decode_step_search_space_path),
        "decode_step_search_space_is_default": (
            decode_step_search_space_is_default
        ),
        "metric": metric,
        "results": {step: "pending" for step in VALIDATION_STEPS},
        "steps": {},
        "artifacts": {name: str(path) for name, path in paths.items()},
    }

    template_config: dict[str, Any] = {}
    graph: LlamaModelGraph | None = None
    plan: dict[str, Any] | None = None
    parameter_config: dict[str, Any] | None = None

    def persist() -> None:
        _write_json(report_path, report)

    def run_step(
        name: str,
        action: Callable[[], dict[str, Any] | None],
    ) -> bool:
        try:
            detail = action() or {}
        except Exception as exc:  # pragma: no cover - exercised by CLI users.
            report["results"][name] = "fail"
            report["steps"][name] = {
                "status": "fail",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _mark_remaining_skipped(report, name)
            report["status"] = "fail"
            persist()
            return False

        report["results"][name] = "pass"
        report["steps"][name] = {"status": "pass", **detail}
        persist()
        return True

    def _validation_device() -> str:
        return str(template_config.get("device", "p150a"))

    def _validation_decode_layers(max_layers: int = 2) -> int:
        _require(graph, "import_llama")
        assert graph is not None
        return max(1, min(max_layers, int(graph.num_layers)))

    def import_step() -> dict[str, Any]:
        nonlocal template_config, graph
        template_config = load_template_config(config_path)
        graph = import_hf_llama(
            model_path,
            mode="decode",
            batch_size=int(template_config["batch_size"]),
            seq_len=int(template_config["decode_seq_len"]),
            max_cache_len=int(template_config["max_cache_len"]),
            generation_mode=(
                "greedy"
                if template_config["generation_template"]
                == "device_argmax_greedy"
                else "sampling"
            ),
        )
        dump_graph_json(graph, paths["semantic_json"])
        return {
            "semantic_json": str(paths["semantic_json"]),
            "model_name": graph.model_name,
            "num_layers": graph.num_layers,
            "batch_size": graph.batch_size,
            "seq_len": graph.seq_len,
            "max_cache_len": graph.max_cache_len,
        }

    def plan_step() -> dict[str, Any]:
        nonlocal plan
        _require(graph, "import_llama")
        plan = build_execution_plan(graph, template_config)
        dump_execution_plan(plan, paths["execution_plan"])
        return {
            "execution_plan": str(paths["execution_plan"]),
            "num_layers": len(plan["layers"]),
            "final": list(plan["final"]),
        }

    def diff_step() -> dict[str, Any]:
        _require(plan, "plan")
        official_template = load_official_template(official_template_path)
        diff = diff_plan_against_official(plan, official_template)
        dump_plan_diff(diff, paths["plan_diff"])
        return {
            "plan_diff": str(paths["plan_diff"]),
            "missing_ops": list(diff["missing_ops"]),
            "extra_ops": list(diff["extra_ops"]),
            "order_mismatch": list(diff["order_mismatch"]),
        }

    def emit_config_step() -> dict[str, Any]:
        nonlocal parameter_config
        _require(graph, "import_llama")
        parameter_config = emit_parameter_config(
            graph,
            recipe=template_config["dtype_recipe"],
            lm_head_split_count=int(template_config["lm_head_split_count"]),
        )
        dump_parameter_config(parameter_config, paths["parameter_config"])
        return {
            "parameter_config": str(paths["parameter_config"]),
            "weight_count": len(parameter_config["weights"]),
            "lm_head_split_count": (
                parameter_config["lm_head"]["split_count"]
            ),
        }

    def prepare_artifacts_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        _require(parameter_config, "emit_config")
        artifact_paths = prepare_offline_artifacts(
            model_path,
            graph,
            parameter_config,
            paths["artifacts_dir"],
        )
        return {
            "artifacts_dir": str(paths["artifacts_dir"]),
            "manifests": {
                name: str(path) for name, path in sorted(artifact_paths.items())
            },
        }

    def build_program_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        _require(plan, "plan")
        program_paths = write_decode_program_bundle(
            graph=graph,
            plan=plan,
            template_config=template_config,
            model_path=model_path,
            out_dir=paths["program_dir"],
        )
        return {
            "program_dir": str(paths["program_dir"]),
            "artifacts": {
                name: str(path) for name, path in sorted(program_paths.items())
            },
        }

    def py_compile_step() -> dict[str, Any]:
        compiled = [
            paths["program_dir"] / "model.py",
            paths["program_dir"] / "run_decode.py",
        ]
        for source in compiled:
            py_compile.compile(str(source), doraise=True)
        return {"compiled": [str(source) for source in compiled]}

    def official_config_diff_step() -> dict[str, Any]:
        diff = diff_official_config(
            paths["program_dir"] / "config.json",
            official_config_path,
        )
        dump_config_diff(diff, paths["official_config_diff"])
        coverage = diff.get("required_field_coverage") or {}
        return {
            "official_config_diff": str(paths["official_config_diff"]),
            "diff_status": diff["status"],
            "issue_count": diff["summary"]["issue_count"],
            "missing_count": diff["summary"]["missing_count"],
            "mismatch_count": diff["summary"]["mismatch_count"],
            "extra_count": diff["summary"]["extra_count"],
            "official_required_field_coverage": coverage.get("official"),
            "required_parity_fields": coverage.get("required_fields", []),
            "ours_source_format": diff["ours"].get("source_format"),
            "official_source_format": diff["official"].get(
                "source_format"
            ),
            "official_source": diff["official"].get("source"),
        }

    def tensorize_parameters_dry_run_step() -> dict[str, Any]:
        tensor_report = tensorize_parameters_from_program_dry_run(
            program_dir=paths["program_dir"],
            roles=["embedding", "norm", "attention", "mlp", "lm_head"],
            layers=[0],
            device=_validation_device(),
            out=paths["tensorize_report"],
        )
        return {
            "tensorize_report": str(paths["tensorize_report"]),
            "roles": list(tensor_report["roles"]),
            "tensor_count": tensor_report["tensor_count"],
            "dry_run": tensor_report["dry_run"],
        }

    def decode_shell_dry_run_step() -> dict[str, Any]:
        shell_report = run_smoke_decode_shell(
            out=paths["decode_shell_report"],
            program_dir=paths["program_dir"],
            layers=1,
            disable_attention=True,
            device=_validation_device(),
            dry_run=True,
        )
        return {
            "decode_shell_report": str(paths["decode_shell_report"]),
            "layers": shell_report["layers_requested"],
            "dry_run": shell_report["dry_run"],
            "smoke_status": shell_report["status"],
        }

    def attention_primitives_dry_run_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        assert graph is not None
        reports = {}
        for primitive in ATTENTION_PRIMITIVES:
            report_path = paths["attention_primitives_dir"] / f"{primitive}.json"
            primitive_report = run_smoke_attention_primitive(
                out=report_path,
                primitive=primitive,
                device=_validation_device(),
                batch_size=int(template_config["batch_size"]),
                hidden_size=int(graph.hidden_size),
                num_heads=int(graph.num_attention_heads),
                num_kv_heads=int(graph.num_key_value_heads),
                head_dim=int(graph.head_dim),
                max_cache_len=int(template_config["max_cache_len"]),
                dry_run=True,
            )
            reports[primitive] = {
                "report": str(report_path),
                "status": primitive_report["status"],
                "dry_run": primitive_report["dry_run"],
            }
        return {
            "attention_primitives_dir": str(paths["attention_primitives_dir"]),
            "primitive_count": len(reports),
            "reports": reports,
        }

    def attention_layer_dry_run_step() -> dict[str, Any]:
        layer_report = run_smoke_attention_layer(
            out=paths["attention_layer_report"],
            program_dir=paths["program_dir"],
            layer=0,
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            dry_run=True,
        )
        return {
            "attention_layer_report": str(paths["attention_layer_report"]),
            "layer": layer_report["layer"],
            "primitive_count": len(layer_report["primitive_reports"]),
            "dry_run": layer_report["dry_run"],
            "smoke_status": layer_report["status"],
        }

    def single_layer_decode_dry_run_step() -> dict[str, Any]:
        smoke_report = run_smoke_single_layer_decode(
            out=paths["single_layer_decode_report"],
            program_dir=paths["program_dir"],
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            dry_run=True,
        )
        return {
            "single_layer_decode_report": str(
                paths["single_layer_decode_report"]
            ),
            "layers": smoke_report["layers"],
            "dry_run": smoke_report["dry_run"],
            "smoke_status": smoke_report["status"],
            "op_count": len(smoke_report["op_sequence"]),
        }

    def decode_step_smoke_dry_run_step() -> dict[str, Any]:
        smoke_report = run_smoke_decode_step(
            out=paths["decode_step_smoke_report"],
            program_dir=paths["program_dir"],
            layers=_validation_decode_layers(),
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            trace=True,
            trace_iterations=1,
            dry_run=True,
        )
        return {
            "decode_step_smoke_report": str(paths["decode_step_smoke_report"]),
            "layers": smoke_report["layers"],
            "dry_run": smoke_report["dry_run"],
            "trace_status": smoke_report["trace"]["status"],
            "ttnn_environment": smoke_report.get("ttnn_environment"),
            "smoke_status": smoke_report["status"],
            "op_count": len(smoke_report["op_sequence"]),
            **_reference_summary(smoke_report),
        }

    def decode_step_profile_dry_run_step() -> dict[str, Any]:
        profile_report = profile_decode_step(
            out=paths["decode_step_profile_report"],
            program_dir=paths["program_dir"],
            layers=_validation_decode_layers(),
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            trace=True,
            trace_iterations=1,
            dry_run=True,
        )
        return {
            "decode_step_profile_report": str(
                paths["decode_step_profile_report"]
            ),
            "layers": profile_report["layers"],
            "dry_run": profile_report["dry_run"],
            "trace_status": profile_report["trace"]["status"],
            "ttnn_environment": profile_report.get("ttnn_environment"),
            "profile_status": profile_report["status"],
            "bottleneck": profile_report["bottleneck_summary"]["max_section"],
            **_reference_summary(profile_report),
        }

    def search_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        search_report = run_lm_head_search(
            graph=graph,
            base_config=template_config,
            space=load_search_space(search_space_path),
            metric=metric,
            out=paths["search_report"],
            candidates_dir=paths["search_candidates_dir"],
            dry_run=True,
        )
        dump_search_report(search_report, paths["search_report"])
        return {
            "search_report": str(paths["search_report"]),
            "candidates_dir": str(paths["search_candidates_dir"]),
            "candidate_count": search_report["candidate_count"],
            "dry_run": search_report["dry_run"],
        }

    def decode_step_autotune_dry_run_step() -> dict[str, Any]:
        autotune_report = run_decode_step_autotune(
            program_dir=paths["program_dir"],
            space=load_search_space(decode_step_search_space_path),
            out=paths["decode_step_autotune_report"],
            layers=_validation_decode_layers(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            metric=metric,
            candidates_dir=paths["decode_step_autotune_candidates_dir"],
            dry_run=True,
            device=_validation_device(),
            trace=True,
            trace_iterations=1,
        )
        dump_search_report(autotune_report, paths["decode_step_autotune_report"])
        return {
            "decode_step_autotune_report": str(
                paths["decode_step_autotune_report"]
            ),
            "candidates_dir": str(
                paths["decode_step_autotune_candidates_dir"]
            ),
            "candidate_count": autotune_report["candidate_count"],
            "metric": autotune_report.get("metric"),
            "metric_direction": autotune_report.get("metric_direction"),
            "status_counts": autotune_report.get("status_counts", {}),
            "reference_status_counts": autotune_report.get(
                "reference_status_counts",
                {},
            ),
            "trace_status_counts": autotune_report.get(
                "trace_status_counts",
                {},
            ),
            "knob_coverage": autotune_report.get("knob_coverage"),
            "default_search_space": decode_step_search_space_is_default,
            "all_knobs_varied": (
                autotune_report.get("knob_coverage") or {}
            ).get("all_knobs_varied"),
            "missing_varied_knobs": (
                autotune_report.get("knob_coverage") or {}
            ).get("missing_varied_knobs", []),
            "search_space": autotune_report.get("search_space"),
            "dry_run": autotune_report["dry_run"],
            "trace_enabled": autotune_report["trace_enabled"],
        }

    def package_step() -> dict[str, Any]:
        package_paths = package_ttnn_direct_program(
            paths["program_dir"],
            paths["package_dir"],
        )
        return {
            "package_dir": str(paths["package_dir"]),
            "artifacts": {
                name: str(path) for name, path in sorted(package_paths.items())
            },
        }

    step_actions = {
        "import_llama": import_step,
        "plan": plan_step,
        "plan_diff": diff_step,
        "emit_config": emit_config_step,
        "prepare_artifacts": prepare_artifacts_step,
        "build_program": build_program_step,
        "py_compile": py_compile_step,
        "official_config_diff": official_config_diff_step,
        "tensorize_parameters_dry_run": tensorize_parameters_dry_run_step,
        "decode_shell_dry_run": decode_shell_dry_run_step,
        "attention_primitives_dry_run": attention_primitives_dry_run_step,
        "attention_layer_dry_run": attention_layer_dry_run_step,
        "single_layer_decode_dry_run": single_layer_decode_dry_run_step,
        "decode_step_smoke_dry_run": decode_step_smoke_dry_run_step,
        "decode_step_profile_dry_run": decode_step_profile_dry_run_step,
        "search_dry_run": search_step,
        "decode_step_autotune_dry_run": (
            decode_step_autotune_dry_run_step
        ),
        "package_program": package_step,
    }

    for step in VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            return report

    acceptance = _validate_direct_acceptance(report)
    report["acceptance"] = acceptance
    report["status"] = "pass" if acceptance["passed"] else "acceptance_failed"
    persist()
    return report


def preflight_real_decode(
    *,
    program_dir: str | Path,
    model_path: str | Path,
    out: str | Path,
    official_config_path: str | Path | None = None,
    decode_step_search_space_path: str | Path | None = None,
    performance_baselines_path: str | Path | None = None,
    layers: int = 1,
    batch_size: int | None = None,
    cache_len: int | None = None,
    max_new_tokens: int = 2,
    prefill_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    trace: bool = False,
    trace_iterations: int = 1,
    metric: str = "latency_ms",
    skip_autotune: bool = False,
    skip_profile_decode_step: bool = False,
    require_full_decode_step: bool = False,
    require_model_end_to_end: bool = False,
    require_official_performance_parity: bool = False,
    require_trace: bool = False,
    require_official_config_match: bool = False,
    require_full_depth: bool = False,
    require_program_runtime_shape: bool = False,
    require_batch32_decode_step: bool = False,
    min_tokens_per_second_per_user: float | None = None,
    baseline_tokens_per_second_per_user: float | None = None,
    baseline_reference: str | None = None,
    min_baseline_ratio: float | None = None,
    decode_shell_pcc_threshold: float = 0.99,
    require_decode_shell_numeric_reference: bool = False,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    ttnn_module: Any | None = None,
    device_environment: dict[str, Any] | None = None,
    guard_device_busy: bool = False,
    device_process_environment: dict[str, Any] | None = None,
    guard_device_health: bool = False,
) -> dict[str, Any]:
    """Check real-decode prerequisites without loading weights or opening a device."""
    program_dir = Path(program_dir)
    model_path = Path(model_path)
    out = Path(out)
    tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else None
    effective_tokenizer_path = tokenizer_path or model_path
    official_config_path = (
        Path(official_config_path)
        if official_config_path is not None
        else default_official_parity_config_path()
    )
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )
    performance_baselines_path = (
        Path(performance_baselines_path)
        if performance_baselines_path is not None
        else default_performance_baselines_path()
    )
    max_new_token_count = int(max_new_tokens)
    prefill_token_count = int(prefill_len) if prefill_len is not None else None
    if max_new_token_count <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prefill_token_count is not None and prefill_token_count <= 0:
        raise ValueError("prefill_len must be positive")

    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        *,
        observed: Any = None,
        expected: Any = None,
        required: bool = True,
        message: str | None = None,
    ) -> None:
        check = {
            "name": name,
            "passed": bool(passed),
            "required": required,
        }
        if observed is not None:
            check["observed"] = observed
        if expected is not None:
            check["expected"] = expected
        if message:
            check["message"] = message
        checks.append(check)

    normalized = {
        "trace": bool(trace),
        "require_full_decode_step": bool(require_full_decode_step),
        "require_model_end_to_end": bool(require_model_end_to_end),
        "require_official_performance_parity": bool(
            require_official_performance_parity
        ),
        "require_trace": bool(require_trace),
        "require_official_config_match": bool(require_official_config_match),
        "require_full_depth": bool(require_full_depth),
        "require_program_runtime_shape": bool(require_program_runtime_shape),
        "require_batch32_decode_step": bool(require_batch32_decode_step),
        "require_decode_shell_numeric_reference": bool(
            require_decode_shell_numeric_reference
        ),
    }
    if normalized["require_official_performance_parity"]:
        normalized["require_full_decode_step"] = True
        normalized["require_model_end_to_end"] = True
        normalized["require_official_config_match"] = True
        add(
            "requirements.baseline_reference",
            bool(baseline_reference),
            observed=baseline_reference,
            expected="non-empty baseline reference",
        )
        add(
            "requirements.min_baseline_ratio",
            min_baseline_ratio is not None,
            observed=min_baseline_ratio,
            expected="positive ratio",
        )
        add(
            "requirements.official_min_baseline_ratio_positive",
            (
                min_baseline_ratio is not None
                and min_baseline_ratio > 0.0
            ),
            observed=min_baseline_ratio,
            expected="> 0.0",
        )
        add(
            "requirements.official_performance_autotune_enabled",
            not skip_autotune,
            observed={"skip_autotune": skip_autotune},
            expected="skip_autotune=false",
            message=(
                "official performance parity requires decode-step autotune "
                "evidence; use --skip-autotune only for bring-up"
            ),
        )
        add(
            "requirements.official_performance_profile_enabled",
            not skip_profile_decode_step,
            observed={
                "skip_profile_decode_step": skip_profile_decode_step,
            },
            expected="skip_profile_decode_step=false",
            message=(
                "official performance parity requires profile-decode-step "
                "evidence; use --skip-profile-decode-step only for bring-up"
            ),
        )
        add(
            "requirements.official_performance_metric",
            metric == OFFICIAL_PERFORMANCE_PARITY_METRIC,
            observed=metric,
            expected=OFFICIAL_PERFORMANCE_PARITY_METRIC,
            message=(
                "official performance parity requires the throughput "
                "autotune metric"
            ),
        )
    if normalized["require_model_end_to_end"]:
        normalized["require_full_decode_step"] = True
    if normalized["require_full_decode_step"]:
        normalized["trace"] = True
        normalized["require_trace"] = True
        normalized["require_full_depth"] = True
        normalized["require_program_runtime_shape"] = True
        normalized["require_batch32_decode_step"] = True
        normalized["require_decode_shell_numeric_reference"] = True

    layer_count = _safe_positive_int("layers", layers, checks)
    trace_iteration_count = _safe_positive_int(
        "trace_iterations",
        trace_iterations,
        checks,
    )
    if min_tokens_per_second_per_user is not None:
        add(
            "requirements.min_tokens_per_second_per_user",
            min_tokens_per_second_per_user >= 0.0,
            observed=min_tokens_per_second_per_user,
            expected=">= 0.0",
        )
    if baseline_tokens_per_second_per_user is not None:
        add(
            "requirements.baseline_tokens_per_second_per_user",
            baseline_tokens_per_second_per_user > 0.0,
            observed=baseline_tokens_per_second_per_user,
            expected="positive number",
        )
    if min_baseline_ratio is not None:
        add(
            "requirements.min_baseline_ratio_nonnegative",
            min_baseline_ratio >= 0.0,
            observed=min_baseline_ratio,
            expected=">= 0.0",
        )
        add(
            "requirements.baseline_source_for_ratio",
            (
                baseline_tokens_per_second_per_user is not None
                or baseline_reference is not None
            ),
            observed={
                "baseline_tokens_per_second_per_user": (
                    baseline_tokens_per_second_per_user
                ),
                "baseline_reference": baseline_reference,
            },
            expected="baseline tokens/sec/user or baseline reference",
        )
    add(
        "requirements.decode_shell_pcc_threshold",
        0.0 <= decode_shell_pcc_threshold <= 1.0,
        observed=decode_shell_pcc_threshold,
        expected="0.0 <= threshold <= 1.0",
    )
    if prompt is not None or normalized["require_model_end_to_end"]:
        add(
            "prompt_runtime.prompt",
            isinstance(prompt, str) and prompt != "",
            observed={
                "provided": prompt is not None,
                "char_count": len(prompt) if isinstance(prompt, str) else None,
            },
            expected="non-empty prompt",
            required=normalized["require_model_end_to_end"],
        )
        add(
            "prompt_runtime.tokenizer_path",
            effective_tokenizer_path.exists(),
            observed=str(effective_tokenizer_path),
            expected="tokenizer directory/model path exists",
            required=normalized["require_model_end_to_end"],
        )

    required_program_files = [
        "config.json",
        "model.py",
        "execution_plan.json",
        "weights_manifest.json",
        "run_decode.py",
    ]
    add("program_dir.exists", program_dir.is_dir(), observed=str(program_dir))
    for filename in required_program_files:
        path = program_dir / filename
        add(
            f"program_file.{filename}",
            path.is_file(),
            observed=str(path),
            expected="file exists",
        )

    program_config: dict[str, Any] | None = None
    program_config_error = None
    try:
        program_config = _load_program_config(program_dir)
    except Exception as exc:
        program_config_error = f"{type(exc).__name__}: {exc}"
    add(
        "program_config.load",
        program_config is not None,
        observed=program_config_error,
        expected="valid generated config.json",
    )

    resolved_batch_size = None
    resolved_cache_len = None
    decode_step_contract = None
    if program_config is not None:
        program_num_layers = int(program_config["num_layers"])
        program_batch_size = int(program_config["batch_size"])
        program_cache_len = int(program_config["max_cache_len"])
        program_seq_len = int(program_config.get("seq_len", 1))
        program_num_kv_heads = int(program_config["num_key_value_heads"])
        program_head_dim = int(program_config["head_dim"])
        program_template_config = (
            program_config.get("template_config")
            if isinstance(program_config.get("template_config"), dict)
            else {}
        )
        program_generation = (
            program_config.get("generation")
            if isinstance(program_config.get("generation"), dict)
            else {}
        )
        program_kv_cache = (
            program_config.get("kv_cache")
            if isinstance(program_config.get("kv_cache"), dict)
            else _kv_cache_contract_from_template_config(
                program_template_config,
                cache_len=program_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
            )
        )
        add(
            "program_config.decode_seq_len",
            program_seq_len == 1,
            observed=program_seq_len,
            expected=1,
        )
        add(
            "program_config.paged_kv_cache",
            (program_kv_cache or {}).get("policy") == "paged",
            observed=(program_kv_cache or {}).get("policy"),
            expected="paged",
        )
        if layer_count is not None:
            add(
                "runtime.layers_within_program",
                layer_count <= program_num_layers,
                observed=layer_count,
                expected=f"<= {program_num_layers}",
            )
            if normalized["require_full_depth"]:
                add(
                    "runtime.full_depth_required",
                    layer_count == program_num_layers,
                    observed=layer_count,
                    expected=program_num_layers,
                )
        resolved_batch_size = _safe_runtime_dimension(
            "batch_size",
            requested=batch_size,
            fallback=program_batch_size,
            checks=checks,
        )
        resolved_cache_len = _safe_runtime_dimension(
            "cache_len",
            requested=cache_len,
            fallback=program_cache_len,
            checks=checks,
        )
        if (
            normalized["require_program_runtime_shape"]
            and resolved_batch_size is not None
            and resolved_cache_len is not None
        ):
            add(
                "runtime.program_batch_size_required",
                resolved_batch_size == program_batch_size,
                observed=resolved_batch_size,
                expected=program_batch_size,
            )
            add(
                "runtime.program_cache_len_required",
                resolved_cache_len == program_cache_len,
                observed=resolved_cache_len,
                expected=program_cache_len,
            )
        if normalized["require_batch32_decode_step"]:
            add(
                "runtime.batch32_required",
                resolved_batch_size == 32,
                observed=resolved_batch_size,
                expected=32,
            )
        if (
            layer_count is not None
            and resolved_batch_size is not None
            and resolved_cache_len is not None
        ):
            decode_step_contract = _decode_step_contract(
                layer_count=layer_count,
                batch_size=resolved_batch_size,
                seq_len=program_seq_len,
                cache_len=resolved_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                kv_cache=program_kv_cache,
                generation=program_generation,
            )

    add("model_path.exists", model_path.is_dir(), observed=str(model_path))
    add(
        "model_config.exists",
        (model_path / "config.json").is_file(),
        observed=str(model_path / "config.json"),
        expected="file exists",
    )
    safetensor_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    add(
        "model_weights.safetensors_present",
        bool(safetensor_files),
        observed=safetensor_files,
        expected="at least one .safetensors shard",
    )

    add(
        "official_config.exists",
        official_config_path.is_file(),
        observed=str(official_config_path),
        expected="file exists",
    )
    official_config_diff_summary = None
    if program_config is not None and official_config_path.is_file():
        try:
            official_diff = diff_official_config(
                program_dir / "config.json",
                official_config_path,
            )
            official_config_diff_summary = {
                "status": official_diff["status"],
                "issue_count": official_diff["summary"]["issue_count"],
                "missing_count": official_diff["summary"]["missing_count"],
                "mismatch_count": official_diff["summary"]["mismatch_count"],
                "extra_count": official_diff["summary"]["extra_count"],
                "ours_source_format": official_diff["ours"].get(
                    "source_format"
                ),
                "official_source_format": official_diff["official"].get(
                    "source_format"
                ),
                "official_source": official_diff["official"].get("source"),
                "sections_with_issues": official_diff["summary"][
                    "sections_with_issues"
                ],
            }
            if normalized["require_official_config_match"]:
                add(
                    "official_config.reference_format",
                    official_diff["official"].get("source_format")
                    == "normalized_parity_config",
                    observed={
                        "source_format": official_diff["official"].get(
                            "source_format"
                        ),
                        "source": official_diff["official"].get("source"),
                    },
                    expected=(
                        "normalized_parity_config official/reference JSON"
                    ),
                    message=(
                        "strong official config match must compare against "
                        "an external normalized parity reference, not a "
                        "generated TTNN Direct config"
                    ),
                )
                add(
                    "official_config.match",
                    official_diff["status"] == "match",
                    observed=official_diff["status"],
                    expected="match",
                    message=(
                        f"{official_diff['summary']['issue_count']} parity "
                        "issue(s)"
                    ),
                )
            else:
                add(
                    "official_config.diff_available",
                    True,
                    observed=official_diff["status"],
                    required=False,
                )
        except Exception as exc:
            add(
                "official_config.diff",
                False,
                observed=f"{type(exc).__name__}: {exc}",
                expected="diff can be computed",
                required=normalized["require_official_config_match"],
            )
    if not skip_autotune:
        add(
            "decode_step_search_space.exists",
            decode_step_search_space_path.is_file(),
            observed=str(decode_step_search_space_path),
            expected="file exists",
        )
    add(
        "performance_baselines.exists",
        performance_baselines_path.is_file(),
        observed=str(performance_baselines_path),
        expected="file exists",
    )

    baseline_reference_entry = None
    baseline_error = None
    if baseline_reference:
        try:
            baseline_reference_entry = resolve_performance_baseline(
                baseline_reference,
                path=performance_baselines_path,
            )
        except Exception as exc:
            baseline_error = f"{type(exc).__name__}: {exc}"
        add(
            "baseline_reference.resolve",
            baseline_reference_entry is not None,
            observed=baseline_error or baseline_reference,
            expected="baseline entry resolves",
        )
        if baseline_reference_entry is not None:
            reference_tps = baseline_reference_entry[
                "decode_tokens_per_second_per_user"
            ]
            add(
                "baseline_reference.tokens_match",
                (
                    baseline_tokens_per_second_per_user is None
                    or _numbers_equal(
                        baseline_tokens_per_second_per_user,
                        reference_tps,
                    )
                ),
                observed={
                    "provided": baseline_tokens_per_second_per_user,
                    "reference": reference_tps,
                },
                expected="provided baseline matches reference",
            )
        if normalized["require_official_performance_parity"]:
            add(
                "baseline_reference.official_performance_target",
                _official_performance_baseline_entry_complete(
                    baseline_reference_entry
                ),
                observed=_performance_baseline_entry_summary(
                    baseline_reference_entry
                ),
                expected={
                    "role": "official_8b_target",
                    "model": "Llama 3.1 8B",
                    "batch_size": 32,
                },
            )

    ttnn_import_error = None
    if ttnn_module is None:
        try:
            ttnn_module = importlib.import_module("ttnn")
        except ImportError as exc:
            ttnn_import_error = str(exc)
            ttnn_module = None
    ttnn_environment = collect_ttnn_environment(ttnn_module)
    add(
        "ttnn.module_available",
        ttnn_environment.get("module_available") is True,
        observed=ttnn_import_error or ttnn_environment.get("module_file"),
        expected=True,
    )
    add(
        "ttnn.version",
        _non_empty_string(ttnn_environment.get("version"))
        or _non_empty_string(ttnn_environment.get("module_file")),
        observed={
            "version": ttnn_environment.get("version"),
            "module_file": ttnn_environment.get("module_file"),
        },
        expected="non-empty version or importable source module path",
    )
    add(
        "ttnn.tt_metal_git_commit",
        _non_empty_string(ttnn_environment.get("tt_metal_git_commit")),
        observed=ttnn_environment.get("tt_metal_git_commit"),
        expected="non-empty tt-metal commit",
    )
    tenstorrent_setup_environment = collect_tenstorrent_setup_environment()
    add(
        "tenstorrent.environment_doc_available",
        tenstorrent_setup_environment.get("reference_doc_available") is True,
        observed=tenstorrent_setup_environment.get("reference_doc"),
        expected="docs/TenstorrentEnvironment.md exists",
        required=False,
    )
    add(
        "tenstorrent.ttrt_module_available",
        tenstorrent_setup_environment.get("ttrt_module_available") is True,
        observed={
            "python_executable": tenstorrent_setup_environment.get(
                "python_executable"
            ),
            "recommended_probe_commands": (
                tenstorrent_setup_environment.get(
                    "recommended_probe_commands",
                    [],
                )
            ),
        },
        expected="importable ttrt module in the active runtime env",
        required=False,
        message=(
            "TenstorrentEnvironment.md uses python -m ttrt query as the "
            "runtime smoke probe; this is advisory for preflight diagnosis"
        ),
    )
    tenstorrent_device_environment = (
        device_environment
        if device_environment is not None
        else collect_tenstorrent_device_environment()
    )
    tenstorrent_process_environment = (
        device_process_environment
        if device_process_environment is not None
        else collect_tenstorrent_process_environment()
    )
    add(
        "tenstorrent.device_available",
        tenstorrent_device_environment.get("device_available") is True,
        observed={
            "device_nodes": tenstorrent_device_environment.get(
                "device_nodes",
                [],
            ),
            "filesystem_entries": tenstorrent_device_environment.get(
                "filesystem_entries",
                [],
            ),
            "driver_loaded": tenstorrent_device_environment.get(
                "driver_loaded"
            ),
            "tt_smi_path": tenstorrent_device_environment.get("tt_smi_path"),
            "tt_smi": tenstorrent_device_environment.get("tt_smi"),
        },
        expected="at least one Tenstorrent character device node",
        message=(
            "real decode cannot open a TTNN device unless the current process "
            "can see /dev/tenstorrent* device nodes"
        ),
    )
    if guard_device_busy:
        add(
            "tenstorrent.device_exclusive",
            tenstorrent_process_environment.get("status") != "busy",
            observed={
                "status": tenstorrent_process_environment.get("status"),
                "conflict_count": tenstorrent_process_environment.get(
                    "conflict_count"
                ),
                "reset_in_progress": tenstorrent_process_environment.get(
                    "reset_in_progress"
                ),
                "conflicts": tenstorrent_process_environment.get(
                    "conflicts",
                    [],
                ),
            },
            expected="no external Tenstorrent reset/example process",
            message=(
                "real validation requires an exclusive board window when "
                "--guard-device-busy is set"
            ),
        )

    device_preflight_diagnostics = _tenstorrent_device_preflight_diagnostics(
        ttnn_environment=ttnn_environment,
        setup_environment=tenstorrent_setup_environment,
        device_environment=tenstorrent_device_environment,
        process_environment=tenstorrent_process_environment,
        runtime_health=None,
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )

    failed_checks = [
        check
        for check in checks
        if not check["passed"] and check.get("required", True)
    ]
    rerun_layer_count = layer_count if layer_count is not None else int(layers)
    rerun_trace_iterations = (
        trace_iteration_count
        if trace_iteration_count is not None
        else int(trace_iterations)
    )
    final_cli_args = _real_decode_cli_args(
        program_dir=program_dir,
        model_path=model_path,
        out_dir=out.parent,
        official_config_path=official_config_path,
        decode_step_search_space_path=decode_step_search_space_path,
        performance_baselines_path=performance_baselines_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        layers=rerun_layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        max_new_tokens=max_new_token_count,
        prefill_len=prefill_token_count,
        device=device,
        device_id=device_id,
        trace=normalized["trace"],
        trace_iterations=rerun_trace_iterations,
        metric=metric,
        skip_autotune=skip_autotune,
        skip_profile_decode_step=skip_profile_decode_step,
        require_full_decode_step=normalized["require_full_decode_step"],
        require_model_end_to_end=normalized["require_model_end_to_end"],
        require_official_performance_parity=normalized[
            "require_official_performance_parity"
        ],
        require_trace=normalized["require_trace"],
        require_official_config_match=normalized[
            "require_official_config_match"
        ],
        require_full_depth=normalized["require_full_depth"],
        require_program_runtime_shape=normalized[
            "require_program_runtime_shape"
        ],
        require_batch32_decode_step=normalized[
            "require_batch32_decode_step"
        ],
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        baseline_reference=baseline_reference,
        min_baseline_ratio=min_baseline_ratio,
        decode_shell_pcc_threshold=decode_shell_pcc_threshold,
        require_decode_shell_numeric_reference=normalized[
            "require_decode_shell_numeric_reference"
        ],
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )
    preflight_cli_args = _real_decode_cli_args(
        program_dir=program_dir,
        model_path=model_path,
        out_dir=out.parent,
        official_config_path=official_config_path,
        decode_step_search_space_path=decode_step_search_space_path,
        performance_baselines_path=performance_baselines_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        layers=rerun_layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        max_new_tokens=max_new_token_count,
        prefill_len=prefill_token_count,
        device=device,
        device_id=device_id,
        trace=normalized["trace"],
        trace_iterations=rerun_trace_iterations,
        metric=metric,
        skip_autotune=skip_autotune,
        skip_profile_decode_step=skip_profile_decode_step,
        require_full_decode_step=normalized["require_full_decode_step"],
        require_model_end_to_end=normalized["require_model_end_to_end"],
        require_official_performance_parity=normalized[
            "require_official_performance_parity"
        ],
        require_trace=normalized["require_trace"],
        require_official_config_match=normalized[
            "require_official_config_match"
        ],
        require_full_depth=normalized["require_full_depth"],
        require_program_runtime_shape=normalized[
            "require_program_runtime_shape"
        ],
        require_batch32_decode_step=normalized[
            "require_batch32_decode_step"
        ],
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        baseline_reference=baseline_reference,
        min_baseline_ratio=min_baseline_ratio,
        decode_shell_pcc_threshold=decode_shell_pcc_threshold,
        require_decode_shell_numeric_reference=normalized[
            "require_decode_shell_numeric_reference"
        ],
        preflight_only=True,
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )
    report = {
        "schema_version": 1,
        "command": "preflight-real-decode",
        "status": "pass" if not failed_checks else "fail",
        "ready_to_run": not failed_checks,
        "program_dir": str(program_dir),
        "model_path": str(model_path),
        "official_config": str(official_config_path),
        "decode_step_search_space": str(decode_step_search_space_path),
        "performance_baselines": str(performance_baselines_path),
        "device": device,
        "device_id": device_id,
        "layers": layer_count,
        "requested_batch_size": batch_size,
        "requested_cache_len": cache_len,
        "max_new_tokens": max_new_token_count,
        "prefill_len": prefill_token_count,
        "batch_size": resolved_batch_size,
        "cache_len": resolved_cache_len,
        "trace_iterations": trace_iteration_count,
        "metric": metric,
        "skip_profile_decode_step": skip_profile_decode_step,
        "requirements": normalized,
        "baseline_reference": baseline_reference,
        "baseline_reference_entry": _performance_baseline_entry_summary(
            baseline_reference_entry
        ),
        "prompt_runtime_requested": prompt is not None,
        "prompt_char_count": len(prompt) if isinstance(prompt, str) else None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "effective_tokenizer_path": str(effective_tokenizer_path),
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "baseline_tokens_per_second_per_user": (
            baseline_tokens_per_second_per_user
        ),
        "min_baseline_ratio": min_baseline_ratio,
        "decode_shell_pcc_threshold": decode_shell_pcc_threshold,
        "official_config_diff": official_config_diff_summary,
        "decode_step_contract": decode_step_contract,
        "ttnn_environment": ttnn_environment,
        "tenstorrent_setup_environment": tenstorrent_setup_environment,
        "tenstorrent_device_environment": tenstorrent_device_environment,
        "tenstorrent_process_environment": (
            tenstorrent_process_environment
        ),
        "device_preflight_diagnostics": device_preflight_diagnostics,
        "guard_device_busy": guard_device_busy,
        "guard_device_health": guard_device_health,
        "final_acceptance_plan": _real_decode_final_acceptance_plan(
            metric=metric,
            skip_autotune=skip_autotune,
            skip_profile_decode_step=skip_profile_decode_step,
            require_full_decode_step=normalized[
                "require_full_decode_step"
            ],
            require_model_end_to_end=normalized[
                "require_model_end_to_end"
            ],
            require_official_performance_parity=normalized[
                "require_official_performance_parity"
            ],
            require_trace=normalized["require_trace"],
            require_official_config_match=normalized[
                "require_official_config_match"
            ],
            require_full_depth=normalized["require_full_depth"],
            require_program_runtime_shape=normalized[
                "require_program_runtime_shape"
            ],
            require_batch32_decode_step=normalized[
                "require_batch32_decode_step"
            ],
            min_tokens_per_second_per_user=(
                min_tokens_per_second_per_user
            ),
            baseline_tokens_per_second_per_user=(
                baseline_tokens_per_second_per_user
            ),
            baseline_reference=baseline_reference,
            min_baseline_ratio=min_baseline_ratio,
            decode_shell_pcc_threshold=decode_shell_pcc_threshold,
            require_decode_shell_numeric_reference=normalized[
                "require_decode_shell_numeric_reference"
            ],
        ),
        "reproducibility": {
            "preflight_cli_args": preflight_cli_args,
            "preflight_cli_command": _shell_command(preflight_cli_args),
            "final_validation_cli_args": final_cli_args,
            "final_validation_cli_command": _shell_command(final_cli_args),
            "preflight_report": str(out),
            "out_dir": str(out.parent),
        },
        "check_count": len(checks),
        "failed_checks": [check["name"] for check in failed_checks],
        "checks": checks,
    }
    _write_json(out, report)
    return report


def validate_real_decode(
    *,
    program_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    official_config_path: str | Path | None = None,
    decode_step_search_space_path: str | Path | None = None,
    layers: int = 1,
    batch_size: int | None = None,
    cache_len: int | None = None,
    max_new_tokens: int = 2,
    prefill_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    metric: str = "latency_ms",
    dry_run: bool = False,
    skip_autotune: bool = False,
    skip_profile_decode_step: bool = False,
    require_full_decode_step: bool = False,
    require_model_end_to_end: bool = False,
    require_official_performance_parity: bool = False,
    require_trace: bool = False,
    require_official_config_match: bool = False,
    require_full_depth: bool = False,
    require_program_runtime_shape: bool = False,
    require_batch32_decode_step: bool = False,
    min_tokens_per_second_per_user: float | None = None,
    baseline_tokens_per_second_per_user: float | None = None,
    performance_baselines_path: str | Path | None = None,
    baseline_reference: str | None = None,
    min_baseline_ratio: float | None = None,
    decode_shell_pcc_threshold: float = 0.99,
    require_decode_shell_numeric_reference: bool = False,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    guard_device_busy: bool = False,
    device_process_environment: dict[str, Any] | None = None,
    guard_device_health: bool = False,
    device_health_environment: dict[str, Any] | None = None,
    isolate_runtime_steps: bool = False,
) -> dict[str, Any]:
    """Run the real-weight generated decode validation gates.

    This is intentionally separate from ``validate_direct``: that command stays
    device-free, while this one proves the materialize/tensorize/runtime path.
    """
    layer_count = int(layers)
    max_new_token_count = int(max_new_tokens)
    prefill_token_count = int(prefill_len) if prefill_len is not None else None
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if max_new_token_count <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prefill_token_count is not None and prefill_token_count <= 0:
        raise ValueError("prefill_len must be positive")
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")
    if (
        baseline_tokens_per_second_per_user is not None
        and baseline_tokens_per_second_per_user <= 0.0
    ):
        raise ValueError("baseline_tokens_per_second_per_user must be positive")
    if (
        min_tokens_per_second_per_user is not None
        and min_tokens_per_second_per_user < 0.0
    ):
        raise ValueError("min_tokens_per_second_per_user must be nonnegative")
    if not 0.0 <= decode_shell_pcc_threshold <= 1.0:
        raise ValueError("decode_shell_pcc_threshold must be between 0 and 1")
    if require_official_performance_parity:
        require_full_decode_step = True
        require_model_end_to_end = True
        require_official_config_match = True
        if not baseline_reference:
            raise ValueError(
                "require_official_performance_parity requires "
                "baseline_reference"
            )
        if min_baseline_ratio is None:
            raise ValueError(
                "require_official_performance_parity requires "
                "min_baseline_ratio"
            )
        if min_baseline_ratio <= 0.0:
            raise ValueError(
                "require_official_performance_parity requires positive "
                "min_baseline_ratio"
            )
        if skip_autotune:
            raise ValueError(
                "require_official_performance_parity cannot be used with "
                "skip_autotune"
            )
        if skip_profile_decode_step:
            raise ValueError(
                "require_official_performance_parity cannot be used with "
                "skip_profile_decode_step"
            )
        if metric != OFFICIAL_PERFORMANCE_PARITY_METRIC:
            raise ValueError(
                "require_official_performance_parity requires "
                f"{OFFICIAL_PERFORMANCE_PARITY_METRIC} autotune metric"
            )
    if require_model_end_to_end:
        require_full_decode_step = True
    if min_baseline_ratio is not None:
        if min_baseline_ratio < 0.0:
            raise ValueError("min_baseline_ratio must be nonnegative")
        if (
            baseline_tokens_per_second_per_user is None
            and baseline_reference is None
        ):
            raise ValueError(
                "baseline_tokens_per_second_per_user or baseline_reference "
                "is required when min_baseline_ratio is set"
            )
    if require_full_decode_step:
        trace = True
        require_trace = True
        require_full_depth = True
        require_program_runtime_shape = True
        require_batch32_decode_step = True
        require_decode_shell_numeric_reference = True

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "real_decode_validation_report.json"
    program_dir = Path(program_dir)
    model_path = Path(model_path)
    program_config = _load_program_config(program_dir)
    program_num_layers = int(program_config["num_layers"])
    program_batch_size = int(program_config["batch_size"])
    program_cache_len = int(program_config["max_cache_len"])
    program_seq_len = int(program_config.get("seq_len", 1))
    program_hidden_size = int(program_config["hidden_size"])
    program_intermediate_size = int(program_config["intermediate_size"])
    program_vocab_size = int(program_config["vocab_size"])
    program_num_attention_heads = int(program_config["num_attention_heads"])
    program_num_kv_heads = int(program_config["num_key_value_heads"])
    program_head_dim = int(program_config["head_dim"])
    program_template_config = (
        program_config.get("template_config")
        if isinstance(program_config.get("template_config"), dict)
        else {}
    )
    program_generation = (
        program_config.get("generation")
        if isinstance(program_config.get("generation"), dict)
        else {}
    )
    program_kv_cache = (
        program_config.get("kv_cache")
        if isinstance(program_config.get("kv_cache"), dict)
        else _kv_cache_contract_from_template_config(
            program_template_config,
            cache_len=program_cache_len,
            num_kv_heads=program_num_kv_heads,
            head_dim=program_head_dim,
        )
    )
    if layer_count > program_num_layers:
        raise ValueError(
            "layers must be <= generated config num_layers "
            f"({program_num_layers})"
        )
    resolved_batch_size = _resolve_runtime_dimension(
        "batch_size",
        requested=batch_size,
        fallback=program_config.get("batch_size"),
    )
    resolved_cache_len = _resolve_runtime_dimension(
        "cache_len",
        requested=cache_len,
        fallback=program_config.get("max_cache_len"),
    )
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )
    decode_step_search_space_is_default = _same_path(
        decode_step_search_space_path,
        default_decode_step_search_space_path(),
    )
    official_config_path = (
        Path(official_config_path)
        if official_config_path is not None
        else default_official_parity_config_path()
    )
    performance_baselines_path = (
        Path(performance_baselines_path)
        if performance_baselines_path is not None
        else default_performance_baselines_path()
    )
    baseline_reference_entry = None
    if baseline_reference is not None:
        baseline_reference_entry = resolve_performance_baseline(
            baseline_reference,
            path=performance_baselines_path,
        )
        baseline_from_reference = baseline_reference_entry[
            "decode_tokens_per_second_per_user"
        ]
        if baseline_tokens_per_second_per_user is not None and not (
            _numbers_equal(
                baseline_tokens_per_second_per_user,
                baseline_from_reference,
            )
        ):
            raise ValueError(
                "baseline_tokens_per_second_per_user must match the selected "
                f"baseline_reference {baseline_reference!r}"
            )
        baseline_tokens_per_second_per_user = float(baseline_from_reference)
    layers_to_materialize = list(range(layer_count))
    decode_step_contract = _decode_step_contract(
        layer_count=layer_count,
        batch_size=resolved_batch_size,
        seq_len=program_seq_len,
        cache_len=resolved_cache_len,
        num_kv_heads=program_num_kv_heads,
        head_dim=program_head_dim,
        kv_cache=program_kv_cache,
        generation=program_generation,
    )

    paths = {
        "official_config_diff": root / "official_config_diff.json",
        "materialize_report": root / "parameter_materialization_report.json",
        "decode_shell_report": root / "decode_shell_report.json",
        "attention_primitives_dir": root / "attention_primitives",
        "attention_layer_report": root / "attention_layer_report.json",
        "single_layer_decode_report": root / "single_layer_decode_report.json",
        "smoke_report": root / "decode_step_smoke_report.json",
        "profile_report": root / "decode_step_profile_report.json",
        "prompt_decode_loop_report": root / "prompt_decode_loop_report.json",
        "generate_report": root / "generate_prefill_decode_report.json",
        "profile_generate_report": root / "generate_profile_report.json",
        "profile_generate_underlying_report": (
            root / "profile_generate_underlying_generate_report.json"
        ),
        "generate_depth_sweep_report": root / "generate_depth_sweep_report.json",
        "generate_depth_reports_dir": root / "generate_depth_reports",
        "decode_depth_sweep_report": root / "decode_depth_sweep_report.json",
        "decode_depth_profiles_dir": root / "decode_depth_profiles",
        "autotune_report": root / "decode_step_autotune_report.json",
        "autotune_candidates_dir": root / "decode_step_autotune_candidates",
        "evidence_manifest": root / "real_decode_evidence_manifest.json",
        "report": report_path,
    }
    validation_cli_args = _real_decode_cli_args(
        program_dir=program_dir,
        model_path=model_path,
        out_dir=root,
        official_config_path=official_config_path,
        decode_step_search_space_path=decode_step_search_space_path,
        performance_baselines_path=performance_baselines_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        max_new_tokens=max_new_token_count,
        prefill_len=prefill_token_count,
        device=device,
        device_id=device_id,
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        metric=metric,
        dry_run=dry_run,
        skip_autotune=skip_autotune,
        skip_profile_decode_step=skip_profile_decode_step,
        require_full_decode_step=require_full_decode_step,
        require_model_end_to_end=require_model_end_to_end,
        require_official_performance_parity=require_official_performance_parity,
        require_trace=require_trace,
        require_official_config_match=require_official_config_match,
        require_full_depth=require_full_depth,
        require_program_runtime_shape=require_program_runtime_shape,
        require_batch32_decode_step=require_batch32_decode_step,
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        baseline_reference=baseline_reference,
        min_baseline_ratio=min_baseline_ratio,
        decode_shell_pcc_threshold=decode_shell_pcc_threshold,
        require_decode_shell_numeric_reference=(
            require_decode_shell_numeric_reference
        ),
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )
    preflight_cli_args = _real_decode_cli_args(
        program_dir=program_dir,
        model_path=model_path,
        out_dir=root,
        official_config_path=official_config_path,
        decode_step_search_space_path=decode_step_search_space_path,
        performance_baselines_path=performance_baselines_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        max_new_tokens=max_new_token_count,
        prefill_len=prefill_token_count,
        device=device,
        device_id=device_id,
        trace=trace,
        trace_iterations=trace_iterations,
        metric=metric,
        skip_autotune=skip_autotune,
        skip_profile_decode_step=skip_profile_decode_step,
        require_full_decode_step=require_full_decode_step,
        require_model_end_to_end=require_model_end_to_end,
        require_official_performance_parity=require_official_performance_parity,
        require_trace=require_trace,
        require_official_config_match=require_official_config_match,
        require_full_depth=require_full_depth,
        require_program_runtime_shape=require_program_runtime_shape,
        require_batch32_decode_step=require_batch32_decode_step,
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        baseline_reference=baseline_reference,
        min_baseline_ratio=min_baseline_ratio,
        decode_shell_pcc_threshold=decode_shell_pcc_threshold,
        require_decode_shell_numeric_reference=(
            require_decode_shell_numeric_reference
        ),
        preflight_only=True,
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )
    tenstorrent_process_environment = (
        device_process_environment
        if device_process_environment is not None
        else collect_tenstorrent_process_environment()
    )
    tenstorrent_process_environment_is_injected = (
        device_process_environment is not None
    )
    tenstorrent_runtime_health = (
        device_health_environment
        if device_health_environment is not None
        else {"status": "not_checked", "device_id": device_id}
    )
    tenstorrent_setup_environment = collect_tenstorrent_setup_environment()
    tenstorrent_device_environment = collect_tenstorrent_device_environment()
    device_preflight_diagnostics = _tenstorrent_device_preflight_diagnostics(
        ttnn_environment={"module_available": ttnn_module is not None},
        setup_environment=tenstorrent_setup_environment,
        device_environment=tenstorrent_device_environment,
        process_environment=tenstorrent_process_environment,
        runtime_health=tenstorrent_runtime_health,
        guard_device_busy=guard_device_busy,
        guard_device_health=guard_device_health,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "validate-real-decode",
        "status": "running",
        "program_dir": str(program_dir),
        "model_path": str(model_path),
        "out_dir": str(root),
        "official_config": str(official_config_path),
        "performance_baselines": str(performance_baselines_path),
        "baseline_reference": baseline_reference,
        "baseline_reference_entry": baseline_reference_entry,
        "decode_step_search_space": str(decode_step_search_space_path),
        "decode_step_search_space_is_default": (
            decode_step_search_space_is_default
        ),
        "program_num_layers": program_num_layers,
        "program_batch_size": program_batch_size,
        "program_cache_len": program_cache_len,
        "program_seq_len": program_seq_len,
        "program_hidden_size": program_hidden_size,
        "program_intermediate_size": program_intermediate_size,
        "program_vocab_size": program_vocab_size,
        "program_num_attention_heads": program_num_attention_heads,
        "program_num_key_value_heads": program_num_kv_heads,
        "program_head_dim": program_head_dim,
        "program_generation": program_generation,
        "program_kv_cache": program_kv_cache,
        "layers": layer_count,
        "requested_batch_size": batch_size,
        "requested_cache_len": cache_len,
        "batch_size": resolved_batch_size,
        "cache_len": resolved_cache_len,
        "max_new_tokens": max_new_token_count,
        "prefill_len": prefill_token_count,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations,
        "metric": metric,
        "dry_run": dry_run,
        "skip_autotune": skip_autotune,
        "skip_profile_decode_step": skip_profile_decode_step,
        "require_full_decode_step": require_full_decode_step,
        "require_model_end_to_end": require_model_end_to_end,
        "require_official_performance_parity": (
            require_official_performance_parity
        ),
        "require_trace": require_trace,
        "require_official_config_match": require_official_config_match,
        "require_full_depth": require_full_depth,
        "require_program_runtime_shape": require_program_runtime_shape,
        "require_batch32_decode_step": require_batch32_decode_step,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "baseline_tokens_per_second_per_user": (
            baseline_tokens_per_second_per_user
        ),
        "min_baseline_ratio": min_baseline_ratio,
        "decode_shell_pcc_threshold": decode_shell_pcc_threshold,
        "require_decode_shell_numeric_reference": (
            require_decode_shell_numeric_reference
        ),
        "guard_device_busy": guard_device_busy,
        "guard_device_health": guard_device_health,
        "tenstorrent_setup_environment": tenstorrent_setup_environment,
        "tenstorrent_device_environment": tenstorrent_device_environment,
        "tenstorrent_process_environment": tenstorrent_process_environment,
        "tenstorrent_runtime_health": tenstorrent_runtime_health,
        "device_preflight_diagnostics": device_preflight_diagnostics,
        "prompt_runtime_requested": prompt is not None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "results": {
            step: "pending" for step in REAL_DECODE_VALIDATION_STEPS
        },
        "decode_step_contract": decode_step_contract,
        "steps": {},
        "artifacts": {name: str(path) for name, path in paths.items()},
        "reproducibility": _real_decode_reproducibility(
            validation_args=validation_cli_args,
            preflight_args=preflight_cli_args,
            artifact_paths=paths,
        ),
        "final_acceptance_plan": _real_decode_final_acceptance_plan(
            metric=metric,
            skip_autotune=skip_autotune,
            skip_profile_decode_step=skip_profile_decode_step,
            require_full_decode_step=require_full_decode_step,
            require_model_end_to_end=require_model_end_to_end,
            require_official_performance_parity=(
                require_official_performance_parity
            ),
            require_trace=require_trace,
            require_official_config_match=require_official_config_match,
            require_full_depth=require_full_depth,
            require_program_runtime_shape=require_program_runtime_shape,
            require_batch32_decode_step=require_batch32_decode_step,
            min_tokens_per_second_per_user=min_tokens_per_second_per_user,
            baseline_tokens_per_second_per_user=(
                baseline_tokens_per_second_per_user
            ),
            baseline_reference=baseline_reference,
            min_baseline_ratio=min_baseline_ratio,
            decode_shell_pcc_threshold=decode_shell_pcc_threshold,
            require_decode_shell_numeric_reference=(
                require_decode_shell_numeric_reference
            ),
        ),
    }

    def persist() -> None:
        _write_json(report_path, report)

    def write_evidence_summary() -> dict[str, Any]:
        report["device_preflight_diagnostics"] = (
            _tenstorrent_device_preflight_diagnostics_from_report(report)
        )
        report["runtime_diagnostics"] = _real_decode_runtime_diagnostics(
            report
        )
        evidence = _real_decode_evidence_manifest(report, paths)
        _write_json(paths["evidence_manifest"], evidence)
        report["evidence"] = {
            "status": evidence["status"],
            "manifest": str(paths["evidence_manifest"]),
            "artifact_count": len(evidence["artifacts"]),
            "acceptance_scope": evidence.get("acceptance_scope", {}),
            "model_end_to_end_readiness": evidence.get(
                "model_end_to_end_readiness",
                {},
            ),
            "failed_acceptance_checks": evidence["acceptance"][
                "failed_checks"
            ],
        }
        persist()
        return evidence

    def mark_device_busy(blocked_step: str) -> bool:
        report["status"] = "device_busy"
        report["message"] = (
            "Tenstorrent device appears to be in use by another process; "
            "rerun validate-real-decode during an exclusive board window."
        )
        report["steps"]["device_exclusive_check"] = {
            "status": "device_busy",
            "blocked_step": blocked_step,
            "process_environment": tenstorrent_process_environment,
            "message": report["message"],
        }
        for step in REAL_DECODE_VALIDATION_STEPS:
            if report["results"].get(step) == "pending":
                report["results"][step] = "skipped"
                report["steps"][step] = {
                    "status": "skipped",
                    "reason": (
                        "blocked by failed step: device_exclusive_check"
                    ),
                }
        persist()
        return False

    def mark_device_unhealthy(blocked_step: str) -> bool:
        report["status"] = "device_unhealthy"
        report["message"] = (
            "Tenstorrent device is visible but failed an isolated TTNN "
            "runtime health probe; reset the board or rerun after the shared "
            "device recovers."
        )
        report["steps"]["device_health_check"] = {
            "status": "device_unhealthy",
            "blocked_step": blocked_step,
            "runtime_health": tenstorrent_runtime_health,
            "message": report["message"],
        }
        for step in REAL_DECODE_VALIDATION_STEPS:
            if report["results"].get(step) == "pending":
                report["results"][step] = "skipped"
                report["steps"][step] = {
                    "status": "skipped",
                    "reason": "blocked by failed step: device_health_check",
                }
        persist()
        return False

    def device_busy_guard(blocked_step: str) -> bool:
        nonlocal tenstorrent_process_environment
        if not guard_device_busy or dry_run:
            return False
        if not tenstorrent_process_environment_is_injected:
            tenstorrent_process_environment = (
                collect_tenstorrent_process_environment()
            )
        report["tenstorrent_process_environment"] = (
            tenstorrent_process_environment
        )
        if tenstorrent_process_environment.get("status") != "busy":
            return False
        mark_device_busy(blocked_step)
        return True

    device_health_checked = False

    def device_health_guard(blocked_step: str) -> bool:
        nonlocal device_health_checked, tenstorrent_runtime_health
        if not guard_device_health or dry_run or device_health_checked:
            return False
        if tenstorrent_runtime_health.get("status") == "not_checked":
            tenstorrent_runtime_health = collect_ttnn_runtime_health(
                device_id=device_id
            )
        device_health_checked = True
        report["tenstorrent_runtime_health"] = tenstorrent_runtime_health
        if tenstorrent_runtime_health.get("status") == "pass":
            report["steps"]["device_health_check"] = {
                "status": "pass",
                "blocked_step": blocked_step,
                "runtime_health": tenstorrent_runtime_health,
            }
            persist()
            return False
        mark_device_unhealthy(blocked_step)
        return True

    def cleanup_step_memory() -> None:
        if not dry_run:
            gc.collect()

    def use_runtime_step_isolation() -> bool:
        return (
            not dry_run
            and isolate_runtime_steps
            and ttnn_module is None
            and torch_module is None
            and tokenizer_module is None
        )

    def add_prompt_runtime_cli_args(command: list[str]) -> None:
        if prompt is not None:
            command.extend(["--prompt", prompt])
        if tokenizer_path is not None:
            command.extend(["--tokenizer-path", str(tokenizer_path)])

    def isolated_cli_report(
        *,
        command: list[str],
        report_path: Path,
        fallback_report: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        isolated = {
            "enabled": True,
            "returncode": result.returncode,
            "command": command,
            "stdout": _diagnostic_excerpt(
                _coerce_process_text(result.stdout),
                limit=2000,
            ),
            "stderr": _diagnostic_excerpt(
                _coerce_process_text(result.stderr),
                limit=2000,
            ),
        }
        if report_path.is_file():
            return json.loads(report_path.read_text()), isolated
        return fallback_report, isolated

    def run_step(
        name: str,
        action: Callable[[], dict[str, Any]],
    ) -> bool:
        if device_busy_guard(name):
            return False
        if device_health_guard(name):
            return False
        try:
            detail = action()
        except Exception as exc:  # pragma: no cover - exercised by CLI users.
            report["results"][name] = "fail"
            report["steps"][name] = {
                "status": "fail",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _mark_remaining_skipped(
                report,
                name,
                REAL_DECODE_VALIDATION_STEPS,
            )
            report["status"] = "fail"
            persist()
            cleanup_step_memory()
            return False

        status = str(detail.get("status", "pass"))
        report["results"][name] = status
        report["steps"][name] = detail
        persist()
        cleanup_step_memory()
        if status in {"pass", "dry_run", "skipped"}:
            return True

        _mark_remaining_skipped(
            report,
            name,
            REAL_DECODE_VALIDATION_STEPS,
        )
        report["status"] = status
        persist()
        return False

    def official_config_diff_step() -> dict[str, Any]:
        diff = diff_official_config(
            program_dir / "config.json",
            official_config_path,
        )
        dump_config_diff(diff, paths["official_config_diff"])
        coverage = diff.get("required_field_coverage") or {}
        section_statuses = {
            section: summary.get("status")
            for section, summary in sorted(diff["sections"].items())
            if isinstance(summary, dict)
        }
        return {
            "status": "pass",
            "official_config_diff": str(paths["official_config_diff"]),
            "official_config": str(official_config_path),
            "diff_status": diff["status"],
            "issue_count": diff["summary"]["issue_count"],
            "missing_count": diff["summary"]["missing_count"],
            "mismatch_count": diff["summary"]["mismatch_count"],
            "extra_count": diff["summary"]["extra_count"],
            "matching_count": diff["summary"]["matching_count"],
            "gap_summary": diff.get("gap_summary"),
            "official_required_field_coverage": coverage.get("official"),
            "required_parity_fields": coverage.get("required_fields", []),
            "ours_source_format": diff["ours"].get("source_format"),
            "official_source_format": diff["official"].get(
                "source_format"
            ),
            "official_source": diff["official"].get("source"),
            "sections": sorted(diff["sections"]),
            "section_statuses": section_statuses,
        }

    def materialize_step() -> dict[str, Any]:
        if dry_run:
            required_tensor_paths = _required_materialized_tensor_paths(
                layer_count=layer_count,
                lm_head_split_count=None,
            )
            materialize_report = {
                "schema_version": 1,
                "status": "dry_run",
                "backend": "torch",
                "model_path": str(model_path),
                "program_dir": str(program_dir),
                "materialized_layer_ids": layers_to_materialize,
                "required_tensor_paths": required_tensor_paths,
                "message": "Dry run only; safetensors payloads were not loaded.",
            }
            _write_json(paths["materialize_report"], materialize_report)
            return {
                "status": "dry_run",
                "materialize_report": str(paths["materialize_report"]),
                "materialized_layer_ids": layers_to_materialize,
                "required_tensor_paths": required_tensor_paths,
            }

        materialize_report = materialize_parameters_from_program(
            model_path=model_path,
            program_dir=program_dir,
            backend="torch",
            layers=layers_to_materialize,
            out=paths["materialize_report"],
        )
        materialized_tensor_paths = sorted(
            (materialize_report.get("tensors") or {}).keys()
        )
        lm_head_split_count = materialize_report["lm_head"]["split_count"]
        required_tensor_paths = _required_materialized_tensor_paths(
            layer_count=layer_count,
            lm_head_split_count=lm_head_split_count,
        )
        expected_tensor_shapes = _expected_materialized_tensor_shapes(
            layer_count=layer_count,
            hidden_size=program_hidden_size,
            intermediate_size=program_intermediate_size,
            vocab_size=program_vocab_size,
            num_attention_heads=program_num_attention_heads,
            num_kv_heads=program_num_kv_heads,
            head_dim=program_head_dim,
            lm_head_splits=(
                (materialize_report.get("lm_head") or {}).get("splits")
            ),
        )
        materialized_tensor_shape_mismatches = (
            _materialized_tensor_shape_mismatches(
                materialize_report.get("tensors") or {},
                expected_tensor_shapes,
            )
        )
        return {
            "status": "pass",
            "materialize_report": str(paths["materialize_report"]),
            "materialized_layer_ids": list(
                materialize_report["materialized_layer_ids"]
            ),
            "tensor_count": materialize_report["tensor_count"],
            "lm_head_split_count": lm_head_split_count,
            "materialized_tensor_paths": materialized_tensor_paths,
            "required_tensor_paths": required_tensor_paths,
            "missing_required_tensor_paths": sorted(
                set(required_tensor_paths) - set(materialized_tensor_paths)
            ),
            "materialized_tensor_shape_mismatches": (
                materialized_tensor_shape_mismatches
            ),
            "key_tensors": _materialization_key_tensors(
                materialize_report.get("tensors") or {},
                required_tensor_paths,
            ),
        }

    def materialized_lm_head_split_count() -> int | None:
        materialize = report.get("steps", {}).get("materialize_parameters", {})
        value = materialize.get("lm_head_split_count")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def tensorization_path_detail(
        runtime_report: dict[str, Any],
        *,
        required_layer_count: int = layer_count,
    ) -> dict[str, Any]:
        setup = runtime_report.get("parameter_setup") or {}
        tensorization = setup.get("tensorization") or {}
        if not isinstance(tensorization, dict):
            tensorization = {}
        required_tensor_paths = _required_tensorized_tensor_paths(
            layer_count=required_layer_count,
            lm_head_split_count=materialized_lm_head_split_count(),
        )
        tensorized_tensor_paths = _tensorized_tensor_paths(tensorization)
        return {
            "tensorized_tensor_paths": tensorized_tensor_paths,
            "required_tensorized_tensor_paths": required_tensor_paths,
            "missing_required_tensorized_tensor_paths": sorted(
                set(required_tensor_paths) - set(tensorized_tensor_paths)
            ),
        }

    def decode_shell_tensorization_path_detail(
        runtime_report: dict[str, Any],
    ) -> dict[str, Any]:
        setup = runtime_report.get("parameter_setup") or {}
        tensorization = setup.get("tensorization") or {}
        if not isinstance(tensorization, dict):
            tensorization = {}
        required_tensor_paths = _required_decode_shell_tensorized_tensor_paths(
            layer_count=layer_count,
            lm_head_split_count=materialized_lm_head_split_count(),
        )
        tensorized_tensor_paths = _tensorized_tensor_paths(tensorization)
        return {
            "tensorized_tensor_paths": tensorized_tensor_paths,
            "required_tensorized_tensor_paths": required_tensor_paths,
            "missing_required_tensorized_tensor_paths": sorted(
                set(required_tensor_paths) - set(tensorized_tensor_paths)
            ),
        }

    def decode_shell_step() -> dict[str, Any]:
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "smoke-decode-shell",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--layers",
                str(layer_count),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--disable-attention",
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--pcc-threshold",
                str(decode_shell_pcc_threshold),
                "--out",
                str(paths["decode_shell_report"]),
            ]
            add_prompt_runtime_cli_args(command)
            shell_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["decode_shell_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated smoke-decode-shell did not write a report"
                    ),
                },
            )
        else:
            shell_report = run_smoke_decode_shell(
                out=paths["decode_shell_report"],
                program_dir=program_dir,
                layers=layer_count,
                disable_attention=True,
                model_path=None if dry_run else model_path,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dry_run=dry_run,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                tokenizer_module=tokenizer_module,
                pcc_threshold=decode_shell_pcc_threshold,
            )
        numeric_reference = (
            shell_report.get("reference", {}).get("numeric_reference", {})
        )
        detail = {
            "status": _runtime_step_status(shell_report, dry_run=dry_run),
            "decode_shell_report": str(paths["decode_shell_report"]),
            "runtime_status": shell_report["status"],
            "error": shell_report.get("error"),
            "message": shell_report.get("message"),
            "detail": shell_report.get("detail"),
            "layers": shell_report.get("layers_requested"),
            "parameter_source": shell_report.get("parameter_source"),
            "input_source": shell_report.get("input_source"),
            "runtime_input_tensor_count": shell_report.get(
                "runtime_input_tensor_count"
            ),
            "synthetic_runtime_input_tensor_count": shell_report.get(
                "synthetic_runtime_input_tensor_count"
            ),
            "prompt_runtime_input_tensor_count": shell_report.get(
                "prompt_runtime_input_tensor_count"
            ),
            "prompt_tokenization": shell_report.get("prompt_tokenization"),
            "numeric_reference_status": numeric_reference.get("status"),
            "numeric_reference_kind": numeric_reference.get("kind"),
            "numeric_reference_passed": numeric_reference.get("passed"),
            "pcc": numeric_reference.get("pcc"),
            "pcc_threshold": numeric_reference.get(
                "pcc_threshold",
                decode_shell_pcc_threshold,
            ),
            "numeric_reference_failed_checks": [
                check.get("name")
                for check in numeric_reference.get("checks", [])
                if isinstance(check, dict) and not check.get("passed")
            ],
            "parameter_setup": shell_report.get("parameter_setup"),
            **decode_shell_tensorization_path_detail(shell_report),
            **_reference_summary(shell_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def attention_primitives_step() -> dict[str, Any]:
        primitive_reports = []
        for primitive in ATTENTION_PRIMITIVES:
            report_path = paths["attention_primitives_dir"] / f"{primitive}.json"
            isolated = None
            if use_runtime_step_isolation():
                command = [
                    sys.executable,
                    "-m",
                    "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                    "smoke-attention-primitive",
                    "--primitive",
                    primitive,
                    "--device",
                    device,
                    "--device-id",
                    str(device_id),
                    "--batch-size",
                    str(resolved_batch_size),
                    "--hidden-size",
                    str(program_hidden_size),
                    "--num-heads",
                    str(program_num_attention_heads),
                    "--num-kv-heads",
                    str(program_num_kv_heads),
                    "--head-dim",
                    str(program_head_dim),
                    "--max-cache-len",
                    str(resolved_cache_len),
                    "--dtype-seed",
                    dtype_seed,
                    "--out",
                    str(report_path),
                ]
                primitive_report, isolated = isolated_cli_report(
                    command=command,
                    report_path=report_path,
                    fallback_report={
                        "schema_version": 1,
                        "primitive": primitive,
                        "status": "fail",
                        "passed": False,
                        "error": (
                            "isolated smoke-attention-primitive did not "
                            "write a report"
                        ),
                    },
                )
            else:
                primitive_report = run_smoke_attention_primitive(
                    out=report_path,
                    primitive=primitive,
                    device=device,
                    device_id=device_id,
                    batch_size=resolved_batch_size,
                    hidden_size=program_hidden_size,
                    num_heads=program_num_attention_heads,
                    num_kv_heads=program_num_kv_heads,
                    head_dim=program_head_dim,
                    max_cache_len=resolved_cache_len,
                    dtype_seed=dtype_seed,
                    dry_run=dry_run,
                    ttnn_module=ttnn_module,
                    torch_module=torch_module,
                )
            primitive_detail = {
                "report": str(report_path),
                **primitive_report,
            }
            if isolated is not None:
                primitive_detail["isolated_subprocess"] = isolated
            primitive_reports.append(primitive_detail)
        return {
            "status": _attention_primitives_step_status(
                primitive_reports,
                dry_run=dry_run,
            ),
            "attention_primitives_dir": str(paths["attention_primitives_dir"]),
            "runtime_status_counts": _primitive_runtime_status_counts(
                primitive_reports
            ),
            "primitive_count": len(primitive_reports),
            "primitive_sequence": [
                report.get("primitive")
                for report in primitive_reports
                if isinstance(report, dict)
            ],
            "primitive_reports": primitive_reports,
            "ttnn_environment": _first_ttnn_environment(primitive_reports),
        }

    def attention_layer_step() -> dict[str, Any]:
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "smoke-attention-layer",
                "--program-dir",
                str(program_dir),
                "--layer",
                "0",
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--out",
                str(paths["attention_layer_report"]),
            ]
            layer_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["attention_layer_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated smoke-attention-layer did not write a report"
                    ),
                },
            )
        else:
            layer_report = run_smoke_attention_layer(
                out=paths["attention_layer_report"],
                program_dir=program_dir,
                layer=0,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
        primitive_reports = layer_report.get("primitive_reports") or []
        detail = {
            "status": _runtime_step_status(layer_report, dry_run=dry_run),
            "attention_layer_report": str(paths["attention_layer_report"]),
            "runtime_status": layer_report["status"],
            "layer": layer_report.get("layer"),
            "batch_size": layer_report.get("batch_size"),
            "cache_len": layer_report.get("cache_len"),
            "hidden_size": layer_report.get("hidden_size"),
            "num_heads": layer_report.get("num_heads"),
            "num_kv_heads": layer_report.get("num_kv_heads"),
            "head_dim": layer_report.get("head_dim"),
            "latency_ms": layer_report.get("latency_ms"),
            "primitive_count": len(primitive_reports),
            "primitive_sequence": [
                primitive.get("primitive")
                for primitive in primitive_reports
                if isinstance(primitive, dict)
            ],
            "primitive_reports": primitive_reports,
            "output_shapes": layer_report.get("output_shapes"),
            "tensor_conversion_count": layer_report.get(
                "tensor_conversion_count"
            ),
            "memory_config_conversion_count": layer_report.get(
                "memory_config_conversion_count"
            ),
            "ttnn_environment": layer_report.get("ttnn_environment"),
            **_reference_summary(layer_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def single_layer_decode_step() -> dict[str, Any]:
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "smoke-single-layer-decode",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--out",
                str(paths["single_layer_decode_report"]),
            ]
            if trace:
                command.append("--trace")
                command.extend(["--trace-iterations", str(trace_iterations)])
            add_prompt_runtime_cli_args(command)
            single_layer_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["single_layer_decode_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated smoke-single-layer-decode did not write "
                        "a report"
                    ),
                },
            )
        else:
            single_layer_report = run_smoke_single_layer_decode(
                out=paths["single_layer_decode_report"],
                program_dir=program_dir,
                model_path=None if dry_run else model_path,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                dry_run=dry_run,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
        detail = {
            "status": _runtime_step_status(
                single_layer_report,
                dry_run=dry_run,
            ),
            "single_layer_decode_report": str(
                paths["single_layer_decode_report"]
            ),
            "runtime_status": single_layer_report["status"],
            "layers": single_layer_report.get("layers"),
            "batch_size": single_layer_report.get("batch_size"),
            "cache_len": single_layer_report.get("cache_len"),
            "parameter_source": single_layer_report.get("parameter_source"),
            "input_source": single_layer_report.get("input_source"),
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(single_layer_report)
            ),
            "synthetic_rotary_tensor_count": (
                _step_synthetic_rotary_tensor_count(single_layer_report)
            ),
            "rotary_runtime_input_tensor_count": (
                (single_layer_report.get("parameter_setup") or {}).get(
                    "rotary_runtime_input_tensor_count"
                )
            ),
            "rotary_runtime_state": single_layer_report.get(
                "rotary_runtime_state"
            ),
            "kv_cache_runtime_input_tensor_count": (
                (single_layer_report.get("parameter_setup") or {}).get(
                    "kv_cache_runtime_input_tensor_count"
                )
            ),
            "kv_cache_runtime_state": single_layer_report.get(
                "kv_cache_runtime_state"
            ),
            "tensor_conversion_count": single_layer_report.get(
                "tensor_conversion_count"
            ),
            "input_shapes": single_layer_report.get("input_shapes"),
            "kv_cache": single_layer_report.get("kv_cache"),
            "output_shapes": single_layer_report.get("output_shapes"),
            "trace_status": single_layer_report.get("trace", {}).get("status"),
            "trace": _trace_summary(single_layer_report.get("trace")),
            "ttnn_environment": single_layer_report.get("ttnn_environment"),
            "parameter_setup": single_layer_report.get("parameter_setup"),
            "prompt_runtime_input_tensor_count": (
                (single_layer_report.get("parameter_setup") or {}).get(
                    "prompt_runtime_input_tensor_count"
                )
            ),
            "decode_runtime_state_input_tensor_count": (
                (single_layer_report.get("parameter_setup") or {}).get(
                    "decode_runtime_state_input_tensor_count"
                )
            ),
            "prompt_tokenization": single_layer_report.get(
                "prompt_tokenization"
            ),
            "decode_runtime_state": single_layer_report.get(
                "decode_runtime_state"
            ),
            **tensorization_path_detail(
                single_layer_report,
                required_layer_count=1,
            ),
            **_reference_summary(single_layer_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def smoke_step_detail(
        smoke_report: dict[str, Any],
        *,
        isolated_subprocess: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detail = {
            "status": _runtime_step_status(smoke_report, dry_run=dry_run),
            "smoke_report": str(paths["smoke_report"]),
            "runtime_status": smoke_report["status"],
            "layers": smoke_report.get("layers"),
            "batch_size": smoke_report.get("batch_size"),
            "cache_len": smoke_report.get("cache_len"),
            "parameter_source": smoke_report.get("parameter_source"),
            "input_source": smoke_report.get("input_source"),
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(smoke_report)
            ),
            "synthetic_rotary_tensor_count": (
                _step_synthetic_rotary_tensor_count(smoke_report)
            ),
            "rotary_runtime_input_tensor_count": (
                (smoke_report.get("parameter_setup") or {}).get(
                    "rotary_runtime_input_tensor_count"
                )
            ),
            "rotary_runtime_state": smoke_report.get("rotary_runtime_state"),
            "kv_cache_runtime_input_tensor_count": (
                (smoke_report.get("parameter_setup") or {}).get(
                    "kv_cache_runtime_input_tensor_count"
                )
            ),
            "kv_cache_runtime_state": smoke_report.get(
                "kv_cache_runtime_state"
            ),
            "tensor_conversion_count": smoke_report.get(
                "tensor_conversion_count"
            ),
            "input_shapes": smoke_report.get("input_shapes"),
            "kv_cache": smoke_report.get("kv_cache"),
            "output_shapes": smoke_report.get("output_shapes"),
            "trace_status": smoke_report.get("trace", {}).get("status"),
            "trace": _trace_summary(smoke_report.get("trace")),
            "ttnn_environment": smoke_report.get("ttnn_environment"),
            "parameter_setup": smoke_report.get("parameter_setup"),
            "prompt_runtime_input_tensor_count": (
                (smoke_report.get("parameter_setup") or {}).get(
                    "prompt_runtime_input_tensor_count"
                )
            ),
            "decode_runtime_state_input_tensor_count": (
                (smoke_report.get("parameter_setup") or {}).get(
                    "decode_runtime_state_input_tensor_count"
                )
            ),
            "prompt_tokenization": smoke_report.get("prompt_tokenization"),
            "decode_runtime_state": smoke_report.get("decode_runtime_state"),
            **tensorization_path_detail(smoke_report),
            **_reference_summary(smoke_report),
        }
        if isolated_subprocess is not None:
            detail["isolated_subprocess"] = isolated_subprocess
        return detail

    def smoke_step_subprocess_report() -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        command = [
            sys.executable,
            "-m",
            "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
            "smoke-decode-step",
            "--program-dir",
            str(program_dir),
            "--layers",
            str(layer_count),
            "--device",
            device,
            "--device-id",
            str(device_id),
            "--batch-size",
            str(resolved_batch_size),
            "--cache-len",
            str(resolved_cache_len),
            "--dtype-seed",
            dtype_seed,
            "--trace-iterations",
            str(trace_iterations),
            "--out",
            str(paths["smoke_report"]),
        ]
        if model_path is not None:
            command.extend(["--model-path", str(model_path)])
        if trace:
            command.append("--trace")
        if prompt is not None:
            command.extend(["--prompt", prompt])
        if tokenizer_path is not None:
            command.extend(["--tokenizer-path", str(tokenizer_path)])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        isolated = {
            "enabled": True,
            "returncode": result.returncode,
            "command": command,
            "stdout": _diagnostic_excerpt(
                _coerce_process_text(result.stdout),
                limit=2000,
            ),
            "stderr": _diagnostic_excerpt(
                _coerce_process_text(result.stderr),
                limit=2000,
            ),
        }
        if paths["smoke_report"].is_file():
            return json.loads(paths["smoke_report"].read_text()), isolated

        return (
            {
                "schema_version": 1,
                "status": "fail",
                "passed": False,
                "error": "isolated smoke-decode-step did not write a report",
                "detail": isolated,
                "layers": layer_count,
                "batch_size": resolved_batch_size,
                "cache_len": resolved_cache_len,
                "trace": _trace_report(
                    requested=trace,
                    status="unavailable" if trace else "disabled",
                    iterations=trace_iterations if trace else 0,
                ),
            },
            isolated,
        )

    def smoke_step() -> dict[str, Any]:
        if (
            not dry_run
            and model_path is not None
            and ttnn_module is None
            and torch_module is None
            and tokenizer_module is None
        ):
            smoke_report, isolated = smoke_step_subprocess_report()
            return smoke_step_detail(
                smoke_report,
                isolated_subprocess=isolated,
            )

        smoke_report = run_smoke_decode_step(
            out=paths["smoke_report"],
            program_dir=program_dir,
            layers=layer_count,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            dry_run=dry_run,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            tokenizer_module=tokenizer_module,
        )
        return smoke_step_detail(smoke_report)

    def profile_step() -> dict[str, Any]:
        if skip_profile_decode_step:
            profile_report = {
                "schema_version": 1,
                "command": "profile-decode-step",
                "status": "skipped",
                "reason": "skip_profile_decode_step requested",
                "dry_run": dry_run,
                "program_dir": str(program_dir),
                "model_path": str(model_path),
                "layers": layer_count,
                "batch_size": resolved_batch_size,
                "cache_len": resolved_cache_len,
                "device": device,
                "device_id": device_id,
                "trace_requested": trace,
                "trace_iterations": trace_iterations,
            }
            _write_json(paths["profile_report"], profile_report)
            return {
                "status": "skipped",
                "profile_report": str(paths["profile_report"]),
                "runtime_status": "skipped",
                "reason": profile_report["reason"],
                "layers": layer_count,
                "batch_size": resolved_batch_size,
                "cache_len": resolved_cache_len,
                "trace_status": "skipped",
                "trace": {"status": "skipped"},
            }

        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "profile-decode-step",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--layers",
                str(layer_count),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--trace-iterations",
                str(trace_iterations),
                "--out",
                str(paths["profile_report"]),
            ]
            if trace:
                command.append("--trace")
            add_prompt_runtime_cli_args(command)
            profile_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["profile_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated profile-decode-step did not write a report"
                    ),
                },
            )
        else:
            profile_report = profile_decode_step(
                out=paths["profile_report"],
                program_dir=program_dir,
                layers=layer_count,
                model_path=None if dry_run else model_path,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                dry_run=dry_run,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                tokenizer_module=tokenizer_module,
            )
        bottleneck = profile_report.get("bottleneck_summary", {})
        detail = {
            "status": _runtime_step_status(profile_report, dry_run=dry_run),
            "profile_report": str(paths["profile_report"]),
            "runtime_status": profile_report["status"],
            "layers": profile_report.get("layers"),
            "batch_size": profile_report.get("batch_size"),
            "cache_len": profile_report.get("cache_len"),
            "parameter_source": profile_report.get("parameter_source"),
            "input_source": profile_report.get("input_source"),
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(profile_report)
            ),
            "synthetic_rotary_tensor_count": (
                _step_synthetic_rotary_tensor_count(profile_report)
            ),
            "rotary_runtime_input_tensor_count": (
                (profile_report.get("parameter_setup") or {}).get(
                    "rotary_runtime_input_tensor_count"
                )
            ),
            "rotary_runtime_state": profile_report.get(
                "rotary_runtime_state"
            ),
            "kv_cache_runtime_input_tensor_count": (
                (profile_report.get("parameter_setup") or {}).get(
                    "kv_cache_runtime_input_tensor_count"
                )
            ),
            "kv_cache_runtime_state": profile_report.get(
                "kv_cache_runtime_state"
            ),
            "tensor_conversion_count": profile_report.get(
                "tensor_conversion_count"
            ),
            "tensor_conversion_ms": profile_report.get("tensor_conversion_ms"),
            "latency_ms": profile_report.get("latency_ms"),
            "section_latency_ms": profile_report.get("section_latency_ms"),
            "layer_profiles": profile_report.get("layer_profiles", []),
            "lm_head_profile": profile_report.get("lm_head_profile"),
            "input_shapes": profile_report.get("input_shapes"),
            "kv_cache": profile_report.get("kv_cache"),
            "output_shapes": profile_report.get("output_shapes"),
            "throughput_summary": profile_report.get("throughput_summary"),
            "bottleneck_summary": bottleneck,
            "max_section": bottleneck.get("max_section"),
            "trace_status": profile_report.get("trace", {}).get("status"),
            "trace": _trace_summary(profile_report.get("trace")),
            "ttnn_environment": profile_report.get("ttnn_environment"),
            "parameter_setup": profile_report.get("parameter_setup"),
            "prompt_runtime_input_tensor_count": (
                (profile_report.get("parameter_setup") or {}).get(
                    "prompt_runtime_input_tensor_count"
                )
            ),
            "decode_runtime_state_input_tensor_count": (
                (profile_report.get("parameter_setup") or {}).get(
                    "decode_runtime_state_input_tensor_count"
                )
            ),
            "prompt_tokenization": profile_report.get("prompt_tokenization"),
            "decode_runtime_state": profile_report.get(
                "decode_runtime_state"
            ),
            **tensorization_path_detail(profile_report),
            **_reference_summary(profile_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def prompt_decode_loop_step() -> dict[str, Any]:
        if not dry_run and prompt is None:
            report["decode_loop_runtime_owned"] = False
            return {
                "status": "skipped",
                "reason": (
                    "prompt is required for decode loop ownership evidence"
                ),
                "decode_loop_runtime_owned": False,
                "input_source": None,
            }
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "prompt-decode-loop",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--decode-steps",
                "2",
                "--layers",
                str(layer_count),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--out",
                str(paths["prompt_decode_loop_report"]),
            ]
            add_prompt_runtime_cli_args(command)
            loop_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["prompt_decode_loop_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated prompt-decode-loop did not write a report"
                    ),
                },
            )
        else:
            loop_report = run_prompt_decode_loop(
                out=paths["prompt_decode_loop_report"],
                program_dir=program_dir,
                model_path=None if dry_run else model_path,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                decode_steps=2,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
        report["decode_loop_runtime_owned"] = bool(
            loop_report.get("decode_loop_runtime_owned")
        )
        detail = {
            "status": _runtime_step_status(loop_report, dry_run=dry_run),
            "prompt_decode_loop_report": str(
                paths["prompt_decode_loop_report"]
            ),
            "runtime_status": loop_report.get("status"),
            "decode_loop_runtime_owned": loop_report.get(
                "decode_loop_runtime_owned"
            ),
            "decode_steps": loop_report.get("decode_steps"),
            "layers": loop_report.get("layers"),
            "batch_size": loop_report.get("batch_size"),
            "cache_len": loop_report.get("cache_len"),
            "parameter_source": loop_report.get("parameter_source"),
            "input_source": loop_report.get("input_source"),
            "runtime_owner": loop_report.get("runtime_owner"),
            "synthetic_runtime_input_tensor_count": (
                loop_report.get("synthetic_runtime_input_tensor_count")
            ),
            "synthetic_rotary_tensor_count": (
                loop_report.get("synthetic_rotary_tensor_count")
            ),
            "prompt_runtime_input_tensor_count": (
                loop_report.get("prompt_runtime_input_tensor_count")
            ),
            "decode_runtime_state_input_tensor_count": (
                loop_report.get("decode_runtime_state_input_tensor_count")
            ),
            "rotary_runtime_input_tensor_count": (
                loop_report.get("rotary_runtime_input_tensor_count")
            ),
            "rotary_runtime_state": loop_report.get("rotary_runtime_state"),
            "kv_cache_runtime_input_tensor_count": (
                loop_report.get("kv_cache_runtime_input_tensor_count")
            ),
            "kv_cache_runtime_state": loop_report.get(
                "kv_cache_runtime_state"
            ),
            "tensor_conversion_count": loop_report.get(
                "tensor_conversion_count"
            ),
            "latency_ms": loop_report.get("latency_ms"),
            "throughput_summary": loop_report.get("throughput_summary"),
            "step_reports": loop_report.get("step_reports", []),
            "input_shapes": loop_report.get("input_shapes"),
            "kv_cache": loop_report.get("kv_cache"),
            "output_shapes": loop_report.get("output_shapes"),
            "trace_status": loop_report.get("trace", {}).get("status"),
            "trace": _trace_summary(loop_report.get("trace")),
            "ttnn_environment": loop_report.get("ttnn_environment"),
            "parameter_setup": loop_report.get("parameter_setup"),
            "prompt_tokenization": loop_report.get("prompt_tokenization"),
            "decode_runtime_state": loop_report.get("decode_runtime_state"),
            **tensorization_path_detail(loop_report),
            **_reference_summary(loop_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def generate_prefill_decode_step() -> dict[str, Any]:
        if not dry_run and prompt is None:
            report["generate_runtime_owned"] = False
            return {
                "status": "skipped",
                "reason": (
                    "prompt is required for prefill+decode generate evidence"
                ),
                "generate_runtime_owned": False,
                "prefill_status": None,
                "kv_cache_source": None,
                "input_source": None,
            }
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "generate",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--max-new-tokens",
                str(max_new_token_count),
                "--layers",
                str(layer_count),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--out",
                str(paths["generate_report"]),
            ]
            if prefill_token_count is not None:
                command.extend(["--prefill-len", str(prefill_token_count)])
            add_prompt_runtime_cli_args(command)
            generate_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["generate_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": "isolated generate did not write a report",
                },
            )
        else:
            generate_report = run_generate(
                out=paths["generate_report"],
                program_dir=program_dir,
                model_path=None if dry_run else model_path,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                max_new_tokens=max_new_token_count,
                layers=layer_count,
                prefill_len=prefill_token_count,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
                tokenizer_module=tokenizer_module,
            )
        report["generate_runtime_owned"] = bool(
            generate_report.get("generate_runtime_owned")
        )
        setup = generate_report.get("parameter_setup")
        if not isinstance(setup, dict):
            setup = {}
        prefill_rotary_count = setup.get(
            "prefill_rotary_runtime_input_tensor_count"
        )
        decode_rotary_count = setup.get(
            "decode_rotary_runtime_input_tensor_count"
        )
        rotary_runtime_count = _sum_present_counts(
            prefill_rotary_count,
            decode_rotary_count,
        )
        kv_cache_state = generate_report.get("kv_cache_runtime_state")
        kv_cache_runtime_count = None
        if isinstance(kv_cache_state, dict):
            kv_cache_runtime_count = kv_cache_state.get("tensor_count")
        detail = {
            "status": _runtime_step_status(generate_report, dry_run=dry_run),
            "generate_report": str(paths["generate_report"]),
            "runtime_status": generate_report.get("runtime_status"),
            "prefill_status": generate_report.get("prefill_status"),
            "generate_runtime_owned": generate_report.get(
                "generate_runtime_owned"
            ),
            "decode_loop_runtime_owned": generate_report.get(
                "decode_loop_runtime_owned"
            ),
            "kv_cache_source": generate_report.get("kv_cache_source"),
            "model_semantics": generate_report.get("model_semantics"),
            "semantic_disclaimer": generate_report.get(
                "semantic_disclaimer"
            ),
            "layers": generate_report.get("layers"),
            "batch_size": generate_report.get("batch_size"),
            "cache_len": generate_report.get("cache_len"),
            "prefill_len": generate_report.get("prefill_len"),
            "max_new_tokens": generate_report.get("max_new_tokens"),
            "decode_steps": generate_report.get("decode_steps"),
            "generated_token_budget": generate_report.get(
                "generated_token_budget"
            ),
            "parameter_source": generate_report.get("parameter_source"),
            "input_source": generate_report.get("input_source"),
            "runtime_owner": generate_report.get("runtime_owner"),
            "generated_token_ids": generate_report.get(
                "generated_token_ids"
            ),
            "generated_token_id_source": generate_report.get(
                "generated_token_id_source"
            ),
            "token_materialization_status": generate_report.get(
                "token_materialization_status"
            ),
            "generated_text": generate_report.get("generated_text"),
            "generated_text_by_user": generate_report.get(
                "generated_text_by_user"
            ),
            "generated_text_status": generate_report.get(
                "generated_text_status"
            ),
            "generated_text_source": generate_report.get(
                "generated_text_source"
            ),
            "synthetic_runtime_input_tensor_count": generate_report.get(
                "synthetic_runtime_input_tensor_count"
            ),
            "synthetic_rotary_tensor_count": generate_report.get(
                "synthetic_rotary_tensor_count"
            ),
            "synthetic_kv_cache_tensor_count": generate_report.get(
                "synthetic_kv_cache_tensor_count"
            ),
            "prompt_runtime_input_tensor_count": setup.get(
                "prefill_prompt_runtime_input_tensor_count"
            ),
            "decode_runtime_state_input_tensor_count": setup.get(
                "decode_runtime_state_input_tensor_count"
            ),
            "rotary_runtime_input_tensor_count": rotary_runtime_count,
            "kv_cache_runtime_input_tensor_count": kv_cache_runtime_count,
            "prefill": generate_report.get("prefill"),
            "prompt_tokenization": generate_report.get("prompt_tokenization"),
            "prefill_tokenization": generate_report.get(
                "prefill_tokenization"
            ),
            "decode_runtime_state": generate_report.get(
                "decode_runtime_state"
            ),
            "rotary_runtime_state": generate_report.get(
                "rotary_runtime_state"
            ),
            "kv_cache_runtime_state": kv_cache_state,
            "runtime_context": generate_report.get("runtime_context"),
            "parameter_setup": generate_report.get("parameter_setup"),
            "decode_token_runtime_handoff": generate_report.get(
                "decode_token_runtime_handoff"
            ),
            "decode_token_host_roundtrip_per_step": generate_report.get(
                "decode_token_host_roundtrip_per_step"
            ),
            "host_token_materialization_for_reporting_only": (
                generate_report.get(
                    "host_token_materialization_for_reporting_only"
                )
            ),
            "end_to_end_contract": generate_report.get(
                "end_to_end_contract"
            ),
            "host_copy_profile": generate_report.get("host_copy_profile"),
            "section_profile": generate_report.get("section_profile"),
            "per_step_token_metadata": generate_report.get(
                "per_step_token_metadata",
                [],
            ),
            "step_reports": generate_report.get("step_reports", []),
            "output_shapes": generate_report.get("output_shapes"),
            "latency_ms": generate_report.get("latency_ms"),
            "throughput_summary": generate_report.get("throughput_summary"),
            "ttnn_environment": generate_report.get("ttnn_environment"),
            "trace_status": generate_report.get("trace", {}).get("status"),
            "trace": _trace_summary(generate_report.get("trace")),
            "error": generate_report.get("error"),
            "detail": generate_report.get("detail"),
            "failure_diagnostics": (
                _generate_prefill_decode_failure_diagnostics(
                    generate_report
                )
            ),
            **tensorization_path_detail(generate_report),
            **_reference_summary(generate_report),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def profile_generate_step() -> dict[str, Any]:
        if skip_profile_decode_step:
            profile_report = {
                "schema_version": 1,
                "command": "profile-generate",
                "mode": "profile-generate",
                "status": "skipped",
                "passed": False,
                "dry_run": dry_run,
                "reason": "skip_profile_decode_step requested",
                "generate_report": str(
                    paths["profile_generate_underlying_report"]
                ),
                "acceptance": {
                    "passed": False,
                    "failed_checks": [
                        "profile_generate.skipped_by_skip_profile_decode_step"
                    ],
                },
                "official_performance_parity_claimed": False,
            }
            _write_json(paths["profile_generate_report"], profile_report)
            return {
                "status": "skipped",
                "profile_generate_report": str(
                    paths["profile_generate_report"]
                ),
                "generate_report": profile_report["generate_report"],
                "runtime_status": "skipped",
                "reason": profile_report["reason"],
                "acceptance": profile_report["acceptance"],
                "official_performance_parity_claimed": False,
            }
        if not dry_run and prompt is None:
            profile_report = {
                "schema_version": 1,
                "command": "profile-generate",
                "mode": "profile-generate",
                "status": "skipped",
                "passed": False,
                "dry_run": dry_run,
                "reason": (
                    "prompt is required for prefill+decode generate profile"
                ),
                "generate_report": str(
                    paths["profile_generate_underlying_report"]
                ),
                "acceptance": {
                    "passed": False,
                    "failed_checks": ["profile_generate.prompt_required"],
                },
                "official_performance_parity_claimed": False,
            }
            _write_json(paths["profile_generate_report"], profile_report)
            return {
                "status": "skipped",
                "profile_generate_report": str(
                    paths["profile_generate_report"]
                ),
                "generate_report": profile_report["generate_report"],
                "runtime_status": "skipped",
                "reason": profile_report["reason"],
                "acceptance": profile_report["acceptance"],
                "official_performance_parity_claimed": False,
            }

        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "profile-generate",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--max-new-tokens",
                str(max_new_token_count),
                "--layers",
                str(layer_count),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--generate-report",
                str(paths["profile_generate_underlying_report"]),
                "--out",
                str(paths["profile_generate_report"]),
            ]
            if prefill_token_count is not None:
                command.extend(["--prefill-len", str(prefill_token_count)])
            add_prompt_runtime_cli_args(command)
            profile_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["profile_generate_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated profile-generate did not write a report"
                    ),
                },
            )
        else:
            profile_report = run_profile_generate(
                out=paths["profile_generate_report"],
                program_dir=program_dir,
                model_path=None if dry_run else model_path,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                max_new_tokens=max_new_token_count,
                layers=layer_count,
                prefill_len=prefill_token_count,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                generate_report=paths["profile_generate_underlying_report"],
                ttnn_module=ttnn_module,
                torch_module=torch_module,
                tokenizer_module=tokenizer_module,
            )
        profile_generate_rotary_count = _sum_present_counts(
            profile_report.get("prefill_rotary_runtime_input_tensor_count"),
            profile_report.get("decode_rotary_runtime_input_tensor_count"),
        )
        detail = {
            "status": _runtime_step_status(
                profile_report,
                dry_run=dry_run,
            ),
            "profile_generate_report": str(paths["profile_generate_report"]),
            "generate_report": profile_report.get("generate_report"),
            "runtime_status": profile_report.get("status"),
            "passed": profile_report.get("passed"),
            "generate_status": profile_report.get("generate_status"),
            "generate_passed": profile_report.get("generate_passed"),
            "program_num_layers": profile_report.get("program_num_layers"),
            "prefill_status": profile_report.get("prefill_status"),
            "kv_cache_source": profile_report.get("kv_cache_source"),
            "model_semantics": profile_report.get("model_semantics"),
            "layers": profile_report.get("layers"),
            "batch_size": profile_report.get("batch_size"),
            "cache_len": profile_report.get("cache_len"),
            "prefill_len": profile_report.get("prefill_len"),
            "max_new_tokens": profile_report.get("max_new_tokens"),
            "decode_steps": profile_report.get("decode_steps"),
            "parameter_source": profile_report.get("parameter_source"),
            "input_source": profile_report.get("input_source"),
            "runtime_owner": profile_report.get("runtime_owner"),
            "generate_runtime_owned": profile_report.get(
                "generate_runtime_owned"
            ),
            "decode_loop_runtime_owned": profile_report.get(
                "decode_loop_runtime_owned"
            ),
            "runtime_context": profile_report.get("runtime_context"),
            "parameter_setup": profile_report.get("parameter_setup"),
            "decode_token_runtime_handoff": profile_report.get(
                "decode_token_runtime_handoff"
            ),
            "decode_token_host_roundtrip_per_step": profile_report.get(
                "decode_token_host_roundtrip_per_step"
            ),
            "host_token_materialization_for_reporting_only": (
                profile_report.get(
                    "host_token_materialization_for_reporting_only"
                )
            ),
            "end_to_end_contract": profile_report.get(
                "end_to_end_contract"
            ),
            "synthetic_runtime_input_tensor_count": profile_report.get(
                "synthetic_runtime_input_tensor_count"
            ),
            "synthetic_rotary_tensor_count": profile_report.get(
                "synthetic_rotary_tensor_count"
            ),
            "synthetic_kv_cache_tensor_count": profile_report.get(
                "synthetic_kv_cache_tensor_count"
            ),
            "prompt_runtime_input_tensor_count": profile_report.get(
                "prefill_prompt_runtime_input_tensor_count"
            ),
            "prefill_rotary_runtime_input_tensor_count": profile_report.get(
                "prefill_rotary_runtime_input_tensor_count"
            ),
            "decode_runtime_state_input_tensor_count": profile_report.get(
                "decode_runtime_state_input_tensor_count"
            ),
            "decode_rotary_runtime_input_tensor_count": profile_report.get(
                "decode_rotary_runtime_input_tensor_count"
            ),
            "rotary_runtime_input_tensor_count": (
                profile_generate_rotary_count
            ),
            "kv_cache_runtime_input_tensor_count": profile_report.get(
                "kv_cache_runtime_input_tensor_count"
            ),
            "generated_text_status": profile_report.get(
                "generated_text_status"
            ),
            "generated_token_count_by_user": profile_report.get(
                "generated_token_count_by_user"
            ),
            "latency_ms": profile_report.get("latency_ms"),
            "prefill_ms": profile_report.get("prefill_ms"),
            "decode_step_ms_mean": profile_report.get(
                "decode_step_ms_mean"
            ),
            "decode_step_ms_min": profile_report.get("decode_step_ms_min"),
            "decode_step_ms_max": profile_report.get("decode_step_ms_max"),
            "decode_step_ms_samples": profile_report.get(
                "decode_step_ms_samples",
                [],
            ),
            "host_copy_ms": profile_report.get("host_copy_ms"),
            "host_copy_profile": profile_report.get("host_copy_profile"),
            "section_profile": profile_report.get("section_profile"),
            "sections": profile_report.get("sections"),
            "per_layer": profile_report.get("per_layer"),
            "tokens_per_second_per_user": profile_report.get(
                "tokens_per_second_per_user"
            ),
            "aggregate_tokens_per_second": profile_report.get(
                "aggregate_tokens_per_second"
            ),
            "throughput_summary": profile_report.get("throughput_summary"),
            "performance_milestones": profile_report.get(
                "performance_milestones"
            ),
            "acceptance": profile_report.get("acceptance"),
            "official_performance_parity_claimed": profile_report.get(
                "official_performance_parity_claimed"
            ),
            "message": profile_report.get("message"),
            "error": profile_report.get("error"),
            "ttnn_environment": profile_report.get("ttnn_environment"),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def generate_depth_sweep_step() -> dict[str, Any]:
        if not dry_run and prompt is None:
            sweep_report = {
                "schema_version": 1,
                "command": "generate-depth-sweep",
                "status": "skipped",
                "reason": (
                    "prompt is required for prefill+decode generate depth "
                    "sweep evidence"
                ),
                "dry_run": dry_run,
                "depths": [],
                "depth_count": 0,
                "records": [],
            }
            _write_json(paths["generate_depth_sweep_report"], sweep_report)
            return {
                "status": "skipped",
                "generate_depth_sweep_report": str(
                    paths["generate_depth_sweep_report"]
                ),
                "reports_dir": str(paths["generate_depth_reports_dir"]),
                "reason": sweep_report["reason"],
                "depths": [],
                "depth_count": 0,
                "status_counts": {"skipped": 1},
                "records": [],
                "acceptance": {
                    "status": "skipped",
                    "passed": False,
                    "failed_checks": ["generate_depth_sweep.prompt_required"],
                },
            }

        sweep_depths = _validation_depth_sweep_targets(
            layer_count=layer_count,
            program_num_layers=program_num_layers,
            require_full_depth=require_full_depth,
        )
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "generate-depth-sweep",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--depths",
                ",".join(str(depth) for depth in sweep_depths),
                "--reports-dir",
                str(paths["generate_depth_reports_dir"]),
                "--max-new-tokens",
                str(max_new_token_count),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--dtype-seed",
                dtype_seed,
                "--out",
                str(paths["generate_depth_sweep_report"]),
            ]
            if prefill_token_count is not None:
                command.extend(["--prefill-len", str(prefill_token_count)])
            if require_full_depth:
                command.append("--require-full-depth")
            add_prompt_runtime_cli_args(command)
            sweep_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["generate_depth_sweep_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated generate-depth-sweep did not write a report"
                    ),
                    "depths": sweep_depths,
                    "depth_count": len(sweep_depths),
                    "records": [],
                },
            )
        else:
            sweep_report = run_generate_depth_sweep(
                program_dir=program_dir,
                out=paths["generate_depth_sweep_report"],
                depths=sweep_depths,
                model_path=None if dry_run else model_path,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                reports_dir=paths["generate_depth_reports_dir"],
                max_new_tokens=max_new_token_count,
                prefill_len=prefill_token_count,
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                device=device,
                device_id=device_id,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                require_full_depth=require_full_depth,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
        detail = {
            "status": _generate_depth_sweep_step_status(
                sweep_report,
                dry_run=dry_run,
            ),
            "generate_depth_sweep_report": str(
                paths["generate_depth_sweep_report"]
            ),
            "reports_dir": str(paths["generate_depth_reports_dir"]),
            "depths": sweep_report.get("depths"),
            "depth_count": sweep_report.get("depth_count"),
            "max_depth": sweep_report.get("max_depth"),
            "covered_full_depth": sweep_report.get("covered_full_depth"),
            "require_full_depth": sweep_report.get("require_full_depth"),
            "status_counts": sweep_report.get("status_counts", {}),
            "prefill_status_counts": sweep_report.get(
                "prefill_status_counts",
                {},
            ),
            "generated_text_status_counts": sweep_report.get(
                "generated_text_status_counts",
                {},
            ),
            "model_semantics_counts": sweep_report.get(
                "model_semantics_counts",
                {},
            ),
            "passed_depth_count": sweep_report.get("passed_depth_count"),
            "failed_depths": sweep_report.get("failed_depths", []),
            "failed_depth_diagnostics": sweep_report.get(
                "failed_depth_diagnostics",
                [],
            ),
            "records": sweep_report.get("records", []),
            "acceptance": sweep_report.get("acceptance"),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def decode_depth_sweep_step() -> dict[str, Any]:
        if skip_profile_decode_step:
            sweep_report = {
                "schema_version": 1,
                "command": "decode-depth-sweep",
                "status": "skipped",
                "reason": (
                    "skip_profile_decode_step requested; "
                    "decode-depth-sweep requires profile-decode-step"
                ),
                "dry_run": dry_run,
                "depths": [],
                "depth_count": 0,
                "records": [],
            }
            _write_json(paths["decode_depth_sweep_report"], sweep_report)
            return {
                "status": "skipped",
                "decode_depth_sweep_report": str(
                    paths["decode_depth_sweep_report"]
                ),
                "profiles_dir": str(paths["decode_depth_profiles_dir"]),
                "reason": sweep_report["reason"],
                "depths": [],
                "depth_count": 0,
                "status_counts": {"skipped": 1},
                "records": [],
                "acceptance": {
                    "status": "skipped",
                    "passed": False,
                    "failed_checks": ["decode_depth_sweep.profile_required"],
                },
            }

        sweep_depths = _validation_depth_sweep_targets(
            layer_count=layer_count,
            program_num_layers=program_num_layers,
            require_full_depth=require_full_depth,
        )
        isolated = None
        if use_runtime_step_isolation():
            command = [
                sys.executable,
                "-m",
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
                "decode-depth-sweep",
                "--program-dir",
                str(program_dir),
                "--model-path",
                str(model_path),
                "--depths",
                ",".join(str(depth) for depth in sweep_depths),
                "--profiles-dir",
                str(paths["decode_depth_profiles_dir"]),
                "--device",
                device,
                "--device-id",
                str(device_id),
                "--batch-size",
                str(resolved_batch_size),
                "--cache-len",
                str(resolved_cache_len),
                "--dtype-seed",
                dtype_seed,
                "--trace-iterations",
                str(trace_iterations),
                "--out",
                str(paths["decode_depth_sweep_report"]),
            ]
            if trace:
                command.append("--trace")
            add_prompt_runtime_cli_args(command)
            sweep_report, isolated = isolated_cli_report(
                command=command,
                report_path=paths["decode_depth_sweep_report"],
                fallback_report={
                    "schema_version": 1,
                    "status": "fail",
                    "passed": False,
                    "error": (
                        "isolated decode-depth-sweep did not write a report"
                    ),
                    "depths": sweep_depths,
                    "depth_count": len(sweep_depths),
                    "records": [],
                },
            )
        else:
            sweep_report = run_decode_depth_sweep(
                program_dir=program_dir,
                out=paths["decode_depth_sweep_report"],
                depths=sweep_depths,
                model_path=None if dry_run else model_path,
                profiles_dir=paths["decode_depth_profiles_dir"],
                batch_size=resolved_batch_size,
                cache_len=resolved_cache_len,
                device=device,
                device_id=device_id,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                dry_run=dry_run,
                require_full_depth=require_full_depth,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
        detail = {
            "status": _decode_depth_sweep_step_status(
                sweep_report,
                dry_run=dry_run,
            ),
            "decode_depth_sweep_report": str(
                paths["decode_depth_sweep_report"]
            ),
            "profiles_dir": str(paths["decode_depth_profiles_dir"]),
            "depths": sweep_report.get("depths"),
            "depth_count": sweep_report.get("depth_count"),
            "max_depth": sweep_report.get("max_depth"),
            "covered_full_depth": sweep_report.get("covered_full_depth"),
            "require_full_depth": sweep_report.get("require_full_depth"),
            "status_counts": sweep_report.get("status_counts", {}),
            "reference_status_counts": sweep_report.get(
                "reference_status_counts",
                {},
            ),
            "trace_status_counts": sweep_report.get("trace_status_counts", {}),
            "passed_depth_count": sweep_report.get("passed_depth_count"),
            "failed_depths": sweep_report.get("failed_depths", []),
            "records": sweep_report.get("records", []),
            "acceptance": sweep_report.get("acceptance"),
        }
        if isolated is not None:
            detail["isolated_subprocess"] = isolated
        return detail

    def autotune_step() -> dict[str, Any]:
        if skip_autotune or skip_profile_decode_step:
            reason = (
                "skip_profile_decode_step requested; "
                "decode-step autotune requires profile-decode-step"
                if skip_profile_decode_step
                else "skip_autotune requested"
            )
            return {
                "status": "skipped",
                "reason": reason,
            }
        autotune_report = run_decode_step_autotune(
            program_dir=program_dir,
            model_path=None if dry_run else model_path,
            space=load_search_space(decode_step_search_space_path),
            out=paths["autotune_report"],
            layers=layer_count,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            metric=metric,
            candidates_dir=paths["autotune_candidates_dir"],
            dry_run=dry_run,
            device=device,
            device_id=device_id,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            tokenizer_module=tokenizer_module,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        dump_search_report(autotune_report, paths["autotune_report"])
        return {
            "status": (
                "dry_run"
                if dry_run
                else "pass"
                if autotune_report.get("best") is not None
                else "fail"
            ),
            "autotune_report": str(paths["autotune_report"]),
            "candidates_dir": str(paths["autotune_candidates_dir"]),
            "candidate_count": autotune_report["candidate_count"],
            "metric_direction": autotune_report.get("metric_direction"),
            "status_counts": autotune_report.get("status_counts", {}),
            "passed_candidate_count": autotune_report.get(
                "passed_candidate_count",
                0,
            ),
            "failed_candidate_count": autotune_report.get(
                "failed_candidate_count",
                0,
            ),
            "best": autotune_report.get("best", {}).get("id")
            if autotune_report.get("best") is not None
            else None,
            "best_reference_status": (
                autotune_report.get("best", {}).get("reference_status")
                if autotune_report.get("best") is not None
                else None
            ),
            "best_trace_status": (
                autotune_report.get("best", {}).get("trace_status")
                if autotune_report.get("best") is not None
                else None
            ),
            "best_parameter_source": (
                autotune_report.get("best", {}).get("parameter_source")
                if autotune_report.get("best") is not None
                else None
            ),
            "best_metric": (
                autotune_report.get("best", {}).get("metric")
                if autotune_report.get("best") is not None
                else None
            ),
            "reference_status_counts": autotune_report.get(
                "reference_status_counts",
                _candidate_reference_status_counts(autotune_report),
            ),
            "trace_status_counts": autotune_report.get(
                "trace_status_counts",
                {},
            ),
            "output_kind_counts": autotune_report.get(
                "output_kind_counts",
                {},
            ),
            "candidate_summaries": _autotune_candidate_summaries(
                autotune_report.get("candidates")
            ),
            "leaderboard": autotune_report.get("leaderboard", []),
            "best_candidate_summary": autotune_report.get(
                "best_candidate_summary"
            ),
            "knob_coverage": autotune_report.get("knob_coverage"),
            "default_search_space": decode_step_search_space_is_default,
            "all_knobs_varied": (
                autotune_report.get("knob_coverage") or {}
            ).get("all_knobs_varied"),
            "missing_varied_knobs": (
                autotune_report.get("knob_coverage") or {}
            ).get("missing_varied_knobs", []),
            "search_space": autotune_report.get("search_space"),
            "best_output_kind": (
                autotune_report.get("best", {}).get("output_kind")
                if autotune_report.get("best") is not None
                else None
            ),
            "dry_run": autotune_report["dry_run"],
        }

    step_actions = {
        "official_config_diff": official_config_diff_step,
        "materialize_parameters": materialize_step,
        "decode_shell": decode_shell_step,
        "attention_primitives": attention_primitives_step,
        "attention_layer": attention_layer_step,
        "single_layer_decode": single_layer_decode_step,
        "smoke_decode_step": smoke_step,
        "profile_decode_step": profile_step,
        "prompt_decode_loop": prompt_decode_loop_step,
        "generate_prefill_decode": generate_prefill_decode_step,
        "profile_generate": profile_generate_step,
        "generate_depth_sweep": generate_depth_sweep_step,
        "decode_depth_sweep": decode_depth_sweep_step,
        "decode_step_autotune": autotune_step,
    }

    for step in REAL_DECODE_VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            write_evidence_summary()
            return report

    acceptance = _real_decode_acceptance(
        report,
        require_trace=require_trace,
        require_official_config_match=require_official_config_match,
        require_full_depth=require_full_depth,
        require_program_runtime_shape=require_program_runtime_shape,
        require_batch32_decode_step=require_batch32_decode_step,
        require_model_end_to_end=require_model_end_to_end,
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        min_baseline_ratio=min_baseline_ratio,
        require_decode_shell_numeric_reference=(
            require_decode_shell_numeric_reference
        ),
    )
    report["acceptance"] = acceptance
    report["status"] = (
        "dry_run"
        if dry_run
        else "pass"
        if acceptance["passed"]
        else "acceptance_failed"
    )
    write_evidence_summary()
    return report


def _first_pending_real_decode_step(report: dict[str, Any]) -> str | None:
    results = report.get("results") or {}
    for step in REAL_DECODE_VALIDATION_STEPS:
        if results.get(step) == "pending":
            return step
    return None


def _real_decode_process_failure_looks_device_related(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> bool:
    if returncode in {-7, 135}:
        return True
    text = f"{stdout}\n{stderr}".lower()
    return (
        "bus error" in text
        or "failed to initialize fw" in text
        or "try resetting the board" in text
        or "non-existent physical address" in text
        or "libtt_metal" in text
        or "libtt-umd" in text
    )


def _coerce_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _runtime_step_status(
    runtime_report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if runtime_report.get("passed"):
        return "pass"
    return str(runtime_report.get("status", "fail"))


def _decode_depth_sweep_step_status(
    sweep_report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if sweep_report.get("passed"):
        return "pass"
    return str(sweep_report.get("status", "fail"))


def _generate_depth_sweep_step_status(
    sweep_report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if sweep_report.get("passed"):
        return "pass"
    return str(sweep_report.get("status", "fail"))


def _validation_depth_sweep_targets(
    *,
    layer_count: int,
    program_num_layers: int,
    require_full_depth: bool,
) -> list[int]:
    targets = [1]
    for depth in (2, 4):
        if depth <= layer_count:
            targets.append(depth)
    targets.append(layer_count)
    if require_full_depth:
        targets.append(program_num_layers)
    resolved: list[int] = []
    for depth in targets:
        depth = int(depth)
        if depth <= 0 or depth > program_num_layers:
            continue
        if depth not in resolved:
            resolved.append(depth)
    return resolved


def _attention_primitives_step_status(
    primitive_reports: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if all(report.get("passed") is True for report in primitive_reports):
        return "pass"
    for report in primitive_reports:
        if report.get("passed") is not True:
            return str(report.get("status", "fail"))
    return "fail"


def _primitive_runtime_status_counts(
    primitive_reports: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in primitive_reports:
        status = str(report.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _first_ttnn_environment(
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for report in reports:
        environment = report.get("ttnn_environment")
        if isinstance(environment, dict):
            return environment
    return None


def _load_program_config(program_dir: Path) -> dict[str, Any]:
    config_path = program_dir / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("num_layers", "batch_size", "max_cache_len"):
        if key not in config:
            raise ValueError(f"generated program config missing {key!r}")
    return config


def _resolve_runtime_dimension(
    name: str,
    *,
    requested: int | None,
    fallback: Any,
) -> int:
    value = fallback if requested is None else requested
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _safe_positive_int(
    name: str,
    value: Any,
    checks: list[dict[str, Any]],
) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        checks.append(
            {
                "name": f"runtime.{name}",
                "passed": False,
                "required": True,
                "observed": value,
                "expected": "positive integer",
            }
        )
        return None
    checks.append(
        {
            "name": f"runtime.{name}",
            "passed": resolved > 0,
            "required": True,
            "observed": resolved,
            "expected": "positive integer",
        }
    )
    return resolved if resolved > 0 else None


def _safe_runtime_dimension(
    name: str,
    *,
    requested: int | None,
    fallback: Any,
    checks: list[dict[str, Any]],
) -> int | None:
    try:
        resolved = _resolve_runtime_dimension(
            name,
            requested=requested,
            fallback=fallback,
        )
    except ValueError as exc:
        checks.append(
            {
                "name": f"runtime.{name}",
                "passed": False,
                "required": True,
                "observed": requested if requested is not None else fallback,
                "expected": "positive integer",
                "message": str(exc),
            }
        )
        return None
    checks.append(
        {
            "name": f"runtime.{name}",
            "passed": True,
            "required": True,
            "observed": resolved,
            "expected": "positive integer",
        }
    )
    return resolved


def _trace_summary(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    execute_samples = trace.get("execute_samples_ms")
    if isinstance(execute_samples, list):
        execute_sample_count = len(execute_samples)
    else:
        execute_sample_count = None
    summary = {
        "requested": trace.get("requested"),
        "status": trace.get("status"),
        "iterations": trace.get("iterations"),
        "capture_latency_ms": trace.get("capture_latency_ms"),
        "execute_latency_ms": trace.get("execute_latency_ms"),
        "execute_sample_count": execute_sample_count,
    }
    if isinstance(execute_samples, list):
        summary["execute_samples_ms"] = execute_samples
    if trace.get("error") is not None:
        summary["error"] = trace.get("error")
    return summary


def _required_materialized_tensor_paths(
    *,
    layer_count: int,
    lm_head_split_count: int | None,
) -> list[str]:
    paths = [
        "embedding.weight",
        "final_norm.weight",
        "lm_head.weight",
    ]
    for layer_id in range(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.q_proj.weight",
                f"layers.{layer_id}.attention.k_proj.weight",
                f"layers.{layer_id}.attention.v_proj.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    if lm_head_split_count is not None:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(int(lm_head_split_count))
        )
    return paths


def _required_tensorized_tensor_paths(
    *,
    layer_count: int,
    lm_head_split_count: int | None,
) -> list[str]:
    paths = [
        "embedding.weight",
        "final_norm.weight",
    ]
    for layer_id in range(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    if lm_head_split_count is not None:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(int(lm_head_split_count))
        )
    return paths


def _required_decode_shell_tensorized_tensor_paths(
    *,
    layer_count: int,
    lm_head_split_count: int | None,
) -> list[str]:
    paths = [
        "embedding.weight",
        "final_norm.weight",
    ]
    for layer_id in range(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    if lm_head_split_count is not None:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(int(lm_head_split_count))
        )
    return paths


def _expected_materialized_tensor_shapes(
    *,
    layer_count: Any,
    hidden_size: Any,
    intermediate_size: Any,
    vocab_size: Any,
    num_attention_heads: Any,
    num_kv_heads: Any,
    head_dim: Any,
    lm_head_splits: Any,
) -> dict[str, list[int]]:
    hidden = _safe_int(hidden_size)
    intermediate = _safe_int(intermediate_size)
    vocab = _safe_int(vocab_size)
    heads = _safe_int(num_attention_heads)
    kv_heads = _safe_int(num_kv_heads)
    dim = _safe_int(head_dim)
    layers = _safe_int(layer_count)
    if None in (hidden, intermediate, vocab, heads, kv_heads, dim, layers):
        return {}

    q_features = heads * dim
    kv_features = kv_heads * dim
    qkv_features = q_features + 2 * kv_features
    shapes: dict[str, list[int]] = {
        "embedding.weight": [vocab, hidden],
        "final_norm.weight": [hidden],
        "lm_head.weight": [vocab, hidden],
    }
    for layer_id in range(layers):
        shapes.update(
            {
                f"layers.{layer_id}.attention.q_proj.weight": [
                    q_features,
                    hidden,
                ],
                f"layers.{layer_id}.attention.k_proj.weight": [
                    kv_features,
                    hidden,
                ],
                f"layers.{layer_id}.attention.v_proj.weight": [
                    kv_features,
                    hidden,
                ],
                f"layers.{layer_id}.attention.o_proj.weight": [
                    hidden,
                    q_features,
                ],
                f"layers.{layer_id}.attention.wqkv_packed.weight": [
                    qkv_features,
                    hidden,
                ],
                f"layers.{layer_id}.mlp.gate_proj.weight": [
                    intermediate,
                    hidden,
                ],
                f"layers.{layer_id}.mlp.up_proj.weight": [
                    intermediate,
                    hidden,
                ],
                f"layers.{layer_id}.mlp.down_proj.weight": [
                    hidden,
                    intermediate,
                ],
                f"layers.{layer_id}.input_norm.weight": [hidden],
                f"layers.{layer_id}.post_attention_norm.weight": [hidden],
            }
        )

    if isinstance(lm_head_splits, list):
        for split in lm_head_splits:
            if not isinstance(split, dict):
                continue
            shard_id = _safe_int(split.get("shard_id"))
            vocab_start = _safe_int(split.get("vocab_start"))
            vocab_end = _safe_int(split.get("vocab_end"))
            if None in (shard_id, vocab_start, vocab_end):
                continue
            shapes[f"lm_head.splits.{shard_id}.weight"] = [
                vocab_end - vocab_start,
                hidden,
            ]
    return shapes


def _materialized_tensor_shape_mismatches(
    tensors: Any,
    expected_shapes: dict[str, list[int]],
) -> list[dict[str, Any]]:
    if not isinstance(tensors, dict):
        return []
    mismatches = []
    for path, expected in sorted(expected_shapes.items()):
        record = tensors.get(path)
        if not isinstance(record, dict):
            continue
        observed = _int_list(record.get("shape"))
        if observed != expected:
            mismatches.append(
                {
                    "path": path,
                    "observed": observed or None,
                    "expected": expected,
                }
            )
    return mismatches


def _materialization_key_tensors(
    tensors: dict[str, Any],
    required_tensor_paths: list[str],
) -> dict[str, Any]:
    key_paths = []
    for path in (
        "embedding.weight",
        "final_norm.weight",
        "layers.0.attention.wqkv_packed.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "lm_head.splits.0.weight",
    ):
        if path in required_tensor_paths:
            key_paths.append(path)
    return {
        path: tensors[path]
        for path in key_paths
        if path in tensors
    }


def _lm_head_source_reference_complete(materialize: Any) -> bool:
    observed = _lm_head_source_reference_observed(materialize)
    lm_head = observed.get("lm_head.weight")
    split0 = observed.get("lm_head.splits.0.weight")
    if not isinstance(lm_head, dict) or not isinstance(split0, dict):
        return False
    return (
        lm_head.get("materialization") == "metadata_reference"
        and lm_head.get("materialized") is False
        and split0.get("source_read") == "sliced_tensor"
    )


def _lm_head_source_reference_observed(materialize: Any) -> dict[str, Any]:
    if not isinstance(materialize, dict):
        return {}
    key_tensors = materialize.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return {}
    observed: dict[str, Any] = {}
    for path in ("lm_head.weight", "lm_head.splits.0.weight"):
        tensor = key_tensors.get(path)
        if not isinstance(tensor, dict):
            continue
        observed[path] = {
            "shape": tensor.get("shape"),
            "materialization": tensor.get("materialization"),
            "materialized": tensor.get("materialized"),
            "source_read": tensor.get("source_read"),
        }
    return observed


def _decode_shell_numeric_reference_complete(
    decode_shell: Any,
    *,
    expected_pcc_threshold: Any,
) -> bool:
    if not isinstance(decode_shell, dict):
        return False
    threshold = decode_shell.get("pcc_threshold")
    return (
        decode_shell.get("numeric_reference_status") == "passed"
        and decode_shell.get("numeric_reference_kind") == "torch_decode_shell"
        and decode_shell.get("numeric_reference_passed") is True
        and _number_at_least(decode_shell.get("pcc"), threshold)
        and _numbers_equal(threshold, expected_pcc_threshold)
        and decode_shell.get("numeric_reference_failed_checks") == []
    )


def _decode_shell_numeric_reference_observed(
    decode_shell: Any,
) -> dict[str, Any]:
    if not isinstance(decode_shell, dict):
        return {}
    return {
        "status": decode_shell.get("numeric_reference_status"),
        "kind": decode_shell.get("numeric_reference_kind"),
        "passed": decode_shell.get("numeric_reference_passed"),
        "pcc": decode_shell.get("pcc"),
        "pcc_threshold": decode_shell.get("pcc_threshold"),
        "failed_checks": decode_shell.get(
            "numeric_reference_failed_checks",
            [],
        ),
    }


def _real_decode_acceptance_scope(
    report: dict[str, Any],
    acceptance: Any,
) -> dict[str, Any]:
    if report.get("dry_run"):
        return {
            "status": "dry_run",
            "accepted_real_weight_runtime": False,
            "require_full_decode_step": bool(
                report.get("require_full_decode_step")
            ),
            "require_model_end_to_end": bool(
                report.get("require_model_end_to_end")
            ),
            "require_official_performance_parity": bool(
                report.get("require_official_performance_parity")
            ),
            "full_decode_step_ready": False,
            "official_performance_parity_ready": False,
            "missing_for_full_decode_step": [
                "run validate-real-decode without --dry-run"
            ],
            "missing_for_official_performance_parity": [
                "accepted full decode-step evidence"
            ],
        }

    accepted = (
        report.get("status") == "pass"
        and isinstance(acceptance, dict)
        and acceptance.get("passed") is True
    )
    full_depth = (
        accepted
        and report.get("require_full_depth") is True
        and _acceptance_check_passed(acceptance, "validation.full_depth_layers")
        and _acceptance_check_passed(acceptance, "decode_depth_sweep.full_depth")
    )
    program_runtime_shape = (
        accepted
        and report.get("require_program_runtime_shape") is True
        and _acceptance_check_passed(acceptance, "validation.program_batch_size")
        and _acceptance_check_passed(acceptance, "validation.program_cache_len")
    )
    batch32_contract = (
        accepted
        and report.get("require_batch32_decode_step") is True
        and _acceptance_check_passed(acceptance, "decode_step_contract.batch32")
    )
    trace = (
        accepted
        and report.get("require_trace") is True
        and _acceptance_check_passed(acceptance, "single_layer_decode.trace_status")
        and _acceptance_check_passed(acceptance, "smoke_decode_step.trace_status")
        and _acceptance_check_passed(acceptance, "profile_decode_step.trace_status")
        and _acceptance_check_passed(acceptance, "profile_decode_step.trace_profile")
    )
    numeric_shell = (
        accepted
        and report.get("require_decode_shell_numeric_reference") is True
        and _acceptance_check_passed(acceptance, "decode_shell.numeric_reference")
    )
    baseline_ratio_floor = (
        accepted
        and report.get("min_baseline_ratio") is not None
        and _acceptance_check_passed(
            acceptance,
            "profile_decode_step.min_baseline_ratio",
        )
    )
    official_positive_floor = (
        accepted
        and report.get("require_official_performance_parity") is True
        and _acceptance_check_passed(
            acceptance,
            "profile_decode_step.official_min_baseline_ratio_positive",
        )
    )
    performance_floor = baseline_ratio_floor and (
        report.get("require_official_performance_parity") is not True
        or official_positive_floor
    )
    official_baseline = (
        accepted
        and report.get("require_official_performance_parity") is True
        and _acceptance_check_passed(
            acceptance,
            "profile_decode_step.official_baseline_reference",
        )
    )
    official_metric = (
        accepted
        and report.get("require_official_performance_parity") is True
        and _acceptance_check_passed(
            acceptance,
            "decode_step_autotune.metric",
        )
    )
    official_config_match = (
        accepted
        and report.get("require_official_config_match") is True
        and _acceptance_check_passed(
            acceptance,
            "official_config_diff.official_reference_format",
        )
        and _acceptance_check_passed(acceptance, "official_config_diff.match")
    )
    model_end_to_end = (
        accepted
        and report.get("require_model_end_to_end") is True
        and _acceptance_check_passed(
            acceptance,
            "model_end_to_end_readiness.ready",
        )
    )

    missing_full_decode = []
    if not accepted:
        missing_full_decode.append("accepted real-weight runtime gates")
    if not full_depth:
        missing_full_decode.append(
            "--require-full-depth with layers == generated program layers"
        )
    if not program_runtime_shape:
        missing_full_decode.append(
            "--require-program-runtime-shape at generated batch/cache shape"
        )
    if not batch32_contract:
        missing_full_decode.append("--require-batch32-decode-step")
    if not trace:
        missing_full_decode.append("--require-trace with captured/executed traces")
    if not numeric_shell:
        missing_full_decode.append("--require-decode-shell-numeric-reference")

    full_decode_ready = accepted and not missing_full_decode
    missing_parity = []
    if not full_decode_ready:
        missing_parity.append("accepted full decode-step evidence")
    if not model_end_to_end:
        missing_parity.append("accepted model end-to-end readiness")
    if not official_config_match:
        missing_parity.append("--require-official-config-match")
    if not performance_floor:
        missing_parity.append(
            "--baseline-reference plus --min-baseline-ratio"
        )
    elif not official_baseline:
        missing_parity.append("official Llama 3.1 8B batch32 baseline")
    if not official_metric:
        missing_parity.append(
            f"--metric {OFFICIAL_PERFORMANCE_PARITY_METRIC}"
        )

    official_performance_parity_ready = (
        full_decode_ready
        and model_end_to_end
        and official_config_match
        and official_baseline
        and official_metric
        and performance_floor
    )

    if not accepted:
        scope = "incomplete"
    elif official_performance_parity_ready:
        scope = "official_performance_parity"
    elif full_decode_ready:
        scope = "full_decode_step"
    else:
        scope = "bringup"

    return {
        "status": scope,
        "accepted_real_weight_runtime": accepted,
        "require_full_decode_step": bool(
            report.get("require_full_decode_step")
        ),
        "require_model_end_to_end": bool(
            report.get("require_model_end_to_end")
        ),
        "require_official_performance_parity": bool(
            report.get("require_official_performance_parity")
        ),
        "requested_layers": report.get("layers"),
        "generated_program_layers": report.get("program_num_layers"),
        "batch_size": report.get("batch_size"),
        "generated_program_batch_size": report.get("program_batch_size"),
        "cache_len": report.get("cache_len"),
        "generated_program_cache_len": report.get("program_cache_len"),
        "full_depth_proven": full_depth,
        "program_runtime_shape_proven": program_runtime_shape,
        "batch32_decode_contract_proven": batch32_contract,
        "trace_proven": trace,
        "decode_shell_numeric_reference_proven": numeric_shell,
        "model_end_to_end_proven": model_end_to_end,
        "official_config_match_proven": official_config_match,
        "official_performance_baseline_proven": official_baseline,
        "official_performance_metric_proven": official_metric,
        "official_positive_baseline_ratio_floor_proven": (
            official_positive_floor
            if report.get("require_official_performance_parity") is True
            else None
        ),
        "performance_baseline_ratio_proven": performance_floor,
        "full_decode_step_ready": full_decode_ready,
        "official_performance_parity_ready": (
            official_performance_parity_ready
        ),
        "missing_for_full_decode_step": missing_full_decode,
        "missing_for_official_performance_parity": missing_parity,
    }


def _model_end_to_end_readiness(
    report: dict[str, Any],
    acceptance_scope: dict[str, Any],
) -> dict[str, Any]:
    runtime_scope = _runtime_input_scope(report)
    accepted = bool(acceptance_scope.get("accepted_real_weight_runtime"))
    full_decode_ready = bool(
        acceptance_scope.get("full_decode_step_ready")
    )
    synthetic_inputs = bool(
        runtime_scope.get("uses_synthetic_runtime_inputs")
    )
    prompt_runtime_observed = any(
        _positive_scalar_count(value)
        for value in (
            runtime_scope.get("prompt_runtime_input_tensor_counts") or {}
        ).values()
    )
    decode_runtime_state_observed = any(
        _positive_scalar_count(value)
        for value in (
            runtime_scope.get("decode_runtime_state_input_tensor_counts")
            or {}
        ).values()
    )
    rotary_runtime_observed = any(
        _positive_scalar_count(value)
        for value in (
            runtime_scope.get("rotary_runtime_input_tensor_counts") or {}
        ).values()
    )
    kv_cache_runtime_observed = any(
        _positive_scalar_count(value)
        for value in (
            runtime_scope.get("kv_cache_runtime_input_tensor_counts") or {}
        ).values()
    )
    steps = report.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    prompt_loop = steps.get("prompt_decode_loop")
    if not isinstance(prompt_loop, dict):
        prompt_loop = {}
    generate_step = steps.get("generate_prefill_decode")
    if not isinstance(generate_step, dict):
        generate_step = {}
    decode_loop_runtime_owned = bool(
        report.get("decode_loop_runtime_owned")
        or prompt_loop.get("decode_loop_runtime_owned")
    )
    generate_prefill_decode_ready = _generate_prefill_decode_ready(
        generate_step
    )
    missing = []
    if report.get("dry_run"):
        missing.append("run validate-real-decode without --dry-run")
    if not accepted:
        missing.append("accepted real-weight runtime gates")
    if not full_decode_ready:
        missing.append("accepted full decode-step evidence")
    if synthetic_inputs:
        if (
            prompt_runtime_observed
            and decode_runtime_state_observed
            and rotary_runtime_observed
            and kv_cache_runtime_observed
        ):
            missing.append(
                "decode loop that owns prompt token ids, page table, cache "
                "position, rotary tensors, and KV cache beyond smoke/profile "
                "harnesses"
            )
        elif (
            prompt_runtime_observed
            and decode_runtime_state_observed
            and rotary_runtime_observed
        ):
            missing.append("real runtime input path for KV cache tensors")
            missing.append(
                "decode loop that owns prompt token ids, page table, cache "
                "position, rotary tensors, and KV cache beyond smoke/profile "
                "harnesses"
            )
        elif prompt_runtime_observed and decode_runtime_state_observed:
            missing.append(
                "real runtime input path for KV cache and rotary tensors"
            )
            missing.append(
                "decode loop that owns prompt token ids, page table, cache "
                "position, KV cache, and rotary tensors beyond smoke/profile "
                "harnesses"
            )
        elif prompt_runtime_observed:
            missing.append(
                "real runtime input path for page table, cache position, "
                "KV cache, and rotary tensors"
            )
            missing.append(
                "decode loop that owns prompt token ids plus page table, "
                "cache position, KV cache, and rotary tensors beyond "
                "smoke/profile harnesses"
            )
        else:
            missing.append(
                "real runtime input path for token ids, page table, cache "
                "position, KV cache, and rotary tensors"
            )
            missing.append(
                "tokenizer/prompt runner that owns the decode loop instead "
                "of smoke-generated inputs"
            )
    elif not decode_loop_runtime_owned:
        missing.append(
            "decode loop that owns prompt token ids, page table, cache "
            "position, rotary tensors, and KV cache beyond smoke/profile "
            "harnesses"
        )
    if not generate_prefill_decode_ready:
        missing.append(
            "prefill+decode generate evidence with kv_cache_source=prefill "
            "and generated token/text output"
        )

    ready = (
        accepted
        and full_decode_ready
        and not synthetic_inputs
        and decode_loop_runtime_owned
        and generate_prefill_decode_ready
    )
    if ready:
        status = "ready"
    elif report.get("dry_run"):
        status = "dry_run"
    elif synthetic_inputs:
        status = "synthetic_runtime_inputs"
    elif not full_decode_ready:
        status = "needs_full_decode_step"
    elif not decode_loop_runtime_owned:
        status = "needs_decode_loop_ownership"
    elif not generate_prefill_decode_ready:
        status = "needs_prefill_generate"
    else:
        status = "incomplete"

    return {
        "schema_version": 1,
        "status": status,
        "model_end_to_end_ready": ready,
        "accepted_real_weight_runtime": accepted,
        "full_decode_step_ready": full_decode_ready,
        "decode_loop_runtime_owned": decode_loop_runtime_owned,
        "generate_prefill_decode_ready": generate_prefill_decode_ready,
        "uses_synthetic_runtime_inputs": synthetic_inputs,
        "runtime_input_scope": runtime_scope,
        "missing_for_model_end_to_end": missing,
        "note": (
            "Full decode-step acceptance can prove generated decode-step "
            "structure, shape, trace, and profile evidence while still using "
            "synthetic runtime inputs. Model end-to-end readiness is reserved "
            "for a real tokenizer/prompt runtime that owns token ids, page "
            "tables, cache positions, KV cache, rotary tensors, and the "
            "decode loop."
        ),
    }


def _runtime_input_scope(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("dry_run"):
        return {
            "schema_version": 1,
            "status": "dry_run",
            "uses_synthetic_runtime_inputs": False,
            "runtime_input_sources": {},
            "synthetic_runtime_input_steps": [],
            "synthetic_runtime_input_tensor_counts": {},
            "prompt_runtime_input_tensor_counts": {},
            "decode_runtime_state_input_tensor_counts": {},
            "rotary_runtime_input_tensor_counts": {},
            "kv_cache_runtime_input_tensor_counts": {},
            "synthetic_rotary_tensor_counts": {},
            "depth_sweep_synthetic_record_count": 0,
        }

    steps = report.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    runtime_step_names = [
        "decode_shell",
        "single_layer_decode",
        "smoke_decode_step",
        "profile_decode_step",
        "prompt_decode_loop",
        "generate_prefill_decode",
        "profile_generate",
    ]
    sources: dict[str, Any] = {}
    runtime_counts: dict[str, Any] = {}
    prompt_counts: dict[str, Any] = {}
    runtime_state_counts: dict[str, Any] = {}
    rotary_runtime_counts: dict[str, Any] = {}
    kv_cache_runtime_counts: dict[str, Any] = {}
    rotary_counts: dict[str, Any] = {}
    synthetic_steps: list[str] = []
    for name in runtime_step_names:
        step = steps.get(name)
        if not isinstance(step, dict):
            continue
        source = step.get("input_source")
        if source is not None:
            sources[name] = source
        runtime_count = step.get("synthetic_runtime_input_tensor_count")
        if runtime_count is None:
            runtime_count = _step_synthetic_runtime_input_count(step)
        if runtime_count is None and source == "synthetic":
            runtime_count = step.get("runtime_input_tensor_count")
        rotary_count = step.get("synthetic_rotary_tensor_count")
        if rotary_count is None:
            rotary_count = _step_synthetic_rotary_tensor_count(step)
        prompt_count = step.get("prompt_runtime_input_tensor_count")
        if prompt_count is None:
            setup = step.get("parameter_setup")
            if isinstance(setup, dict):
                prompt_count = setup.get("prompt_runtime_input_tensor_count")
        runtime_state_count = step.get(
            "decode_runtime_state_input_tensor_count"
        )
        if runtime_state_count is None:
            setup = step.get("parameter_setup")
            if isinstance(setup, dict):
                runtime_state_count = setup.get(
                    "decode_runtime_state_input_tensor_count"
                )
        rotary_runtime_count = step.get("rotary_runtime_input_tensor_count")
        if rotary_runtime_count is None:
            setup = step.get("parameter_setup")
            if isinstance(setup, dict):
                rotary_runtime_count = setup.get(
                    "rotary_runtime_input_tensor_count"
                )
        kv_cache_runtime_count = step.get(
            "kv_cache_runtime_input_tensor_count"
        )
        if kv_cache_runtime_count is None:
            setup = step.get("parameter_setup")
            if isinstance(setup, dict):
                kv_cache_runtime_count = setup.get(
                    "kv_cache_runtime_input_tensor_count"
                )
        if runtime_count is not None:
            runtime_counts[name] = runtime_count
        if prompt_count is not None:
            prompt_counts[name] = prompt_count
        if runtime_state_count is not None:
            runtime_state_counts[name] = runtime_state_count
        if rotary_runtime_count is not None:
            rotary_runtime_counts[name] = rotary_runtime_count
        if kv_cache_runtime_count is not None:
            kv_cache_runtime_counts[name] = kv_cache_runtime_count
        if rotary_count is not None:
            rotary_counts[name] = rotary_count
        if (
            source == "synthetic"
            or _positive_scalar_count(runtime_count)
            or _positive_scalar_count(rotary_count)
        ):
            synthetic_steps.append(name)

    depth_sweep = steps.get("decode_depth_sweep")
    synthetic_depth_records = 0
    if isinstance(depth_sweep, dict):
        records = depth_sweep.get("records")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                if (
                    record.get("input_source") == "synthetic"
                    or _positive_scalar_count(
                        record.get("synthetic_runtime_input_tensor_count")
                    )
                    or _positive_scalar_count(
                        record.get("synthetic_rotary_tensor_count")
                    )
                ):
                    synthetic_depth_records += 1
            if synthetic_depth_records:
                synthetic_steps.append("decode_depth_sweep")

    synthetic_steps = sorted(set(synthetic_steps))
    uses_synthetic = bool(synthetic_steps or synthetic_depth_records)
    return {
        "schema_version": 1,
        "status": (
            "synthetic_runtime_inputs"
            if uses_synthetic
            else "no_synthetic_runtime_inputs_observed"
        ),
        "uses_synthetic_runtime_inputs": uses_synthetic,
        "runtime_input_sources": sources,
        "synthetic_runtime_input_steps": synthetic_steps,
        "synthetic_runtime_input_tensor_counts": runtime_counts,
        "prompt_runtime_input_tensor_counts": prompt_counts,
        "decode_runtime_state_input_tensor_counts": runtime_state_counts,
        "rotary_runtime_input_tensor_counts": rotary_runtime_counts,
        "kv_cache_runtime_input_tensor_counts": kv_cache_runtime_counts,
        "synthetic_rotary_tensor_counts": rotary_counts,
        "depth_sweep_synthetic_record_count": synthetic_depth_records,
    }


def _generate_prefill_decode_ready(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    return (
        step.get("status") == "pass"
        and step.get("prefill_status") == "passed"
        and step.get("kv_cache_source") == "prefill"
        and step.get("generate_runtime_owned") is True
        and step.get("decode_loop_runtime_owned") is True
        and _generated_tokens_present(step.get("generated_token_ids"))
        and step.get("generated_text_status") not in {
            None,
            "not_run",
            "dry_run",
            "error",
        }
    )


def _generated_tokens_present(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if isinstance(row, list) and row:
            return True
    return False


def _generate_prefill_decode_failure_diagnostics(
    generate_report: dict[str, Any],
) -> dict[str, Any] | None:
    if bool(generate_report.get("passed")):
        return None
    prefill = generate_report.get("prefill")
    if not isinstance(prefill, dict):
        prefill = {}
    contract = generate_report.get("end_to_end_contract")
    if not isinstance(contract, dict):
        contract = {}
    step_reports = generate_report.get("step_reports")
    if not isinstance(step_reports, list):
        step_reports = []
    failed_step = next(
        (
            step
            for step in step_reports
            if isinstance(step, dict) and not bool(step.get("passed"))
        ),
        None,
    )
    return {
        "status": generate_report.get("status"),
        "runtime_status": generate_report.get("runtime_status"),
        "error": generate_report.get("error"),
        "detail": generate_report.get("detail"),
        "model_semantics": generate_report.get("model_semantics"),
        "layers": generate_report.get("layers"),
        "layout": generate_report.get("layout"),
        "end_to_end_contract": {
            "status": contract.get("status"),
            "failed_checks": contract.get("failed_checks", []),
        },
        "prefill": {
            "status": generate_report.get("prefill_status"),
            "cache_population": _prefill_cache_population_diagnostics(
                prefill.get("cache_population", [])
            ),
        },
        "decode": {
            "decode_loop_runtime_owned": generate_report.get(
                "decode_loop_runtime_owned"
            ),
            "step_count": len(step_reports),
            "failed_step": _generate_failed_step_diagnostics(failed_step),
        },
    }


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


def _generate_failed_step_diagnostics(step: Any) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    reference = step.get("reference")
    if not isinstance(reference, dict):
        reference = {}
    return {
        "step_index": step.get("step_index"),
        "status": step.get("status"),
        "cache_position_value": step.get("cache_position_value"),
        "input_shapes": step.get("input_shapes"),
        "output_shapes": step.get("output_shapes"),
        "decode_runtime_state": step.get("decode_runtime_state"),
        "rotary_runtime_state": step.get("rotary_runtime_state"),
        "reference_status": reference.get("status"),
        "reference_failed_checks": reference.get("failed_checks", []),
        "observed_ops": reference.get("observed_ops"),
        "expected_ops": reference.get("expected_ops"),
        "error": step.get("error"),
    }


def _positive_scalar_count(value: Any) -> bool:
    numeric = _safe_int(value)
    return numeric is not None and numeric > 0


def _sum_present_counts(*values: Any) -> int | None:
    total = 0
    seen = False
    for value in values:
        numeric = _safe_int(value)
        if numeric is None:
            continue
        total += numeric
        seen = True
    return total if seen else None


def _real_decode_evidence_manifest(
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
    acceptance_scope = _real_decode_acceptance_scope(report, acceptance)
    model_end_to_end_readiness = _model_end_to_end_readiness(
        report,
        acceptance_scope,
    )

    return {
        "schema_version": 1,
        "status": status,
        "acceptance_scope": acceptance_scope,
        "model_end_to_end_readiness": model_end_to_end_readiness,
        "reproducibility": report.get("reproducibility"),
        "final_acceptance_plan": report.get("final_acceptance_plan"),
        "runtime_diagnostics": report.get("runtime_diagnostics")
        or _real_decode_runtime_diagnostics(report),
        "acceptance_gate_matrix": _final_acceptance_gate_matrix(
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
            "failed_steps": _step_names_with_status(
                results,
                failing=True,
            ),
            "skipped_steps": _step_names_with_status(
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
            _artifact_evidence(name, path)
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
            "decode_shell_tensorization": _tensorization_evidence(
                decode_shell
            ),
            "single_layer_tensorization": _tensorization_evidence(
                single_layer
            ),
            "smoke_tensorization": _tensorization_evidence(smoke),
            "profile_tensorization": _tensorization_evidence(profile),
            "prompt_decode_loop_tensorization": _tensorization_evidence(
                prompt_loop
            ),
            "generate_prefill_decode_tensorization": _tensorization_evidence(
                generate_step
            ),
            "profile_generate_tensorization": _tensorization_evidence(
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


def _kv_cache_contract_from_template_config(
    template_config: dict[str, Any],
    *,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    template = template_config.get("kv_cache_template")
    return {
        "template": template,
        "policy": "paged" if template == "paged_kv_cache" else None,
        "page_block_size": 32,
        "dtype": "bfloat8_b",
        "max_cache_len": cache_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _decode_step_contract(
    *,
    layer_count: int,
    batch_size: int,
    seq_len: int,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    page_block_size = _safe_int(kv_cache.get("page_block_size")) or 32
    page_count = max(1, (cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    physical_kv_cache_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    logical_kv_cache_shape = [
        batch_size,
        cache_len,
        num_kv_heads,
        head_dim,
    ]
    kv_policy = kv_cache.get("policy")
    kv_template = kv_cache.get("template")
    generation_template = generation.get("template")
    retain_logits = bool(generation.get("retain_logits", False))
    output_kind = "logits" if retain_logits else "token"
    return {
        "schema_version": 1,
        "source": "generated_program_config",
        "layers": layer_count,
        "batch_size": batch_size,
        "decode_seq_len": seq_len,
        "cache_len": cache_len,
        "token_input_shape": [batch_size, seq_len],
        "kv_cache_policy": kv_policy,
        "kv_cache_template": kv_template,
        "uses_paged_kv_cache": (
            kv_policy == "paged" or kv_template == "paged_kv_cache"
        ),
        "kv_page_block_size": page_block_size,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "page_table_shape": [batch_size, page_count],
        "cache_position_shape": [batch_size],
        "kv_cache_shape": physical_kv_cache_shape,
        "kv_cache_physical_shape": physical_kv_cache_shape,
        "kv_cache_logical_shape": logical_kv_cache_shape,
        "kv_cache_layer_ids": list(range(layer_count)),
        "generation_template": generation_template,
        "output_kind": output_kind,
        "accepted_output_kinds": ["token", "logits"],
    }


def _validate_direct_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    steps = report.get("steps", {})
    results = report.get("results", {})
    artifacts = report.get("artifacts", {})
    plan_diff = steps.get("plan_diff", {})
    tensorize = steps.get("tensorize_parameters_dry_run", {})
    decode_shell = steps.get("decode_shell_dry_run", {})
    attention_primitives = steps.get("attention_primitives_dry_run", {})
    attention_layer = steps.get("attention_layer_dry_run", {})
    single_layer = steps.get("single_layer_decode_dry_run", {})
    smoke = steps.get("decode_step_smoke_dry_run", {})
    profile = steps.get("decode_step_profile_dry_run", {})
    search = steps.get("search_dry_run", {})
    autotune = steps.get("decode_step_autotune_dry_run", {})
    package = steps.get("package_program", {})

    checks = [
        _acceptance_check(
            "validate_direct.steps",
            results == {step: "pass" for step in VALIDATION_STEPS},
            observed=results,
            expected={step: "pass" for step in VALIDATION_STEPS},
        ),
        _acceptance_check(
            "plan_diff.clean",
            plan_diff.get("missing_ops") == []
            and plan_diff.get("extra_ops") == []
            and plan_diff.get("order_mismatch") == [],
            observed={
                "missing_ops": plan_diff.get("missing_ops"),
                "extra_ops": plan_diff.get("extra_ops"),
                "order_mismatch": plan_diff.get("order_mismatch"),
            },
            expected={
                "missing_ops": [],
                "extra_ops": [],
                "order_mismatch": [],
            },
        ),
        _acceptance_check(
            "py_compile.artifacts",
            _paths_exist(steps.get("py_compile", {}).get("compiled")),
            observed=steps.get("py_compile", {}).get("compiled"),
            expected="compiled files exist",
        ),
        _acceptance_check(
            "official_config_diff.status",
            steps.get("official_config_diff", {}).get("diff_status")
            in {"match", "diff_found"},
            observed=steps.get("official_config_diff", {}).get(
                "diff_status"
            ),
            expected=["match", "diff_found"],
        ),
        _acceptance_check(
            "official_config_diff.official_required_fields",
            _official_required_field_coverage_complete(
                steps.get("official_config_diff", {}).get(
                    "official_required_field_coverage"
                )
            ),
            observed=_official_required_field_coverage_observed(
                steps.get("official_config_diff", {}).get(
                    "official_required_field_coverage"
                )
            ),
            expected="complete",
        ),
        _acceptance_check(
            "tensorize_parameters_dry_run.status",
            tensorize.get("dry_run") is True
            and _positive_number(tensorize.get("tensor_count"))
            and _contains_all(
                tensorize.get("roles"),
                ["embedding", "norm", "attention", "mlp", "lm_head"],
            ),
            observed={
                "dry_run": tensorize.get("dry_run"),
                "tensor_count": tensorize.get("tensor_count"),
                "roles": tensorize.get("roles"),
            },
            expected=["embedding", "norm", "attention", "mlp", "lm_head"],
        ),
        _acceptance_check(
            "decode_shell_dry_run.status",
            decode_shell.get("dry_run") is True
            and decode_shell.get("smoke_status") == "dry_run",
            observed={
                "dry_run": decode_shell.get("dry_run"),
                "smoke_status": decode_shell.get("smoke_status"),
            },
            expected="dry_run",
        ),
        _acceptance_check(
            "attention_primitives_dry_run.status",
            _attention_primitives_dry_run_complete(
                attention_primitives.get("reports")
            ),
            observed=_attention_primitives_dry_run_observed(
                attention_primitives.get("reports")
            ),
            expected=list(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_layer_dry_run.status",
            attention_layer.get("dry_run") is True
            and attention_layer.get("smoke_status") == "dry_run"
            and _positive_number(attention_layer.get("primitive_count")),
            observed={
                "dry_run": attention_layer.get("dry_run"),
                "smoke_status": attention_layer.get("smoke_status"),
                "primitive_count": attention_layer.get("primitive_count"),
            },
            expected="dry_run",
        ),
        _acceptance_check(
            "single_layer_decode_dry_run.status",
            single_layer.get("dry_run") is True
            and single_layer.get("smoke_status") == "dry_run"
            and _positive_number(single_layer.get("op_count")),
            observed={
                "dry_run": single_layer.get("dry_run"),
                "smoke_status": single_layer.get("smoke_status"),
                "op_count": single_layer.get("op_count"),
            },
            expected="dry_run",
        ),
        _acceptance_check(
            "decode_step_smoke_dry_run.status",
            smoke.get("dry_run") is True
            and smoke.get("smoke_status") == "dry_run"
            and smoke.get("trace_status") == "dry_run"
            and smoke.get("reference_status") == "dry_run"
            and _positive_number(smoke.get("op_count")),
            observed={
                "dry_run": smoke.get("dry_run"),
                "smoke_status": smoke.get("smoke_status"),
                "trace_status": smoke.get("trace_status"),
                "reference_status": smoke.get("reference_status"),
                "op_count": smoke.get("op_count"),
            },
            expected="dry_run",
        ),
        _acceptance_check(
            "decode_step_profile_dry_run.status",
            profile.get("dry_run") is True
            and profile.get("profile_status") == "dry_run"
            and profile.get("trace_status") == "dry_run"
            and profile.get("reference_status") == "dry_run"
            and _non_empty_string(profile.get("bottleneck")),
            observed={
                "dry_run": profile.get("dry_run"),
                "profile_status": profile.get("profile_status"),
                "trace_status": profile.get("trace_status"),
                "reference_status": profile.get("reference_status"),
                "bottleneck": profile.get("bottleneck"),
            },
            expected="dry_run",
        ),
        _acceptance_check(
            "search_dry_run.candidate_count",
            search.get("dry_run") is True
            and _positive_number(search.get("candidate_count")),
            observed={
                "dry_run": search.get("dry_run"),
                "candidate_count": search.get("candidate_count"),
            },
            minimum=1,
        ),
        _acceptance_check(
            "decode_step_autotune_dry_run.knob_coverage",
            _autotune_knob_coverage_complete(
                autotune.get("knob_coverage"),
                candidate_count=autotune.get("candidate_count"),
            ),
            observed=_autotune_knob_coverage_observed(
                autotune.get("knob_coverage")
            ),
            expected=list(DECODE_STEP_AUTOTUNE_KNOBS),
        ),
        _acceptance_check(
            "decode_step_autotune_dry_run.status_counts",
            autotune.get("dry_run") is True
            and autotune.get("trace_enabled") is True
            and _status_count_matches_total(
                autotune.get("status_counts"),
                "dry_run_planned",
                autotune.get("candidate_count"),
            ),
            observed={
                "dry_run": autotune.get("dry_run"),
                "trace_enabled": autotune.get("trace_enabled"),
                "status_counts": autotune.get("status_counts"),
                "candidate_count": autotune.get("candidate_count"),
            },
            expected={"dry_run_planned": autotune.get("candidate_count")},
        ),
        _acceptance_check(
            "package_program.manifest",
            _path_exists(
                (package.get("artifacts") or {}).get("manifest.json")
            )
            or _path_exists(Path(str(package.get("package_dir"))) / "manifest.json"),
            observed=package.get("artifacts"),
            expected="manifest.json",
        ),
        _acceptance_check(
            "validate_direct.artifacts",
            _required_validate_direct_artifacts_exist(artifacts),
            observed=_validate_direct_artifact_observed(artifacts),
            expected=[
                "semantic_json",
                "execution_plan",
                "plan_diff",
                "official_config_diff",
                "parameter_config",
                "program_dir",
                "tensorize_report",
                "decode_shell_report",
                "attention_layer_report",
                "single_layer_decode_report",
                "decode_step_smoke_report",
                "decode_step_profile_report",
                "search_report",
                "decode_step_autotune_report",
                "package_dir",
            ],
        ),
    ]
    if report.get("decode_step_search_space_is_default"):
        checks.append(
            _acceptance_check(
                "decode_step_autotune_dry_run.default_knob_variation",
                _autotune_default_knobs_varied(
                    autotune.get("knob_coverage")
                ),
                observed=_autotune_knob_variation_observed(
                    autotune.get("knob_coverage")
                ),
                expected=list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
        )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "check_count": len(checks),
        "failed_checks": [
            check["name"] for check in checks if not check["passed"]
        ],
        "checks": checks,
    }


def _real_decode_acceptance(
    report: dict[str, Any],
    *,
    require_trace: bool,
    require_official_config_match: bool,
    require_full_depth: bool,
    require_program_runtime_shape: bool,
    require_batch32_decode_step: bool,
    require_model_end_to_end: bool,
    min_tokens_per_second_per_user: float | None,
    baseline_tokens_per_second_per_user: float | None,
    min_baseline_ratio: float | None,
    require_decode_shell_numeric_reference: bool,
) -> dict[str, Any]:
    if report.get("dry_run"):
        return {
            "status": "dry_run",
            "passed": True,
            "require_official_config_match": require_official_config_match,
            "require_full_depth": require_full_depth,
            "require_program_runtime_shape": require_program_runtime_shape,
            "require_batch32_decode_step": require_batch32_decode_step,
            "require_full_decode_step": bool(
                report.get("require_full_decode_step")
            ),
            "require_model_end_to_end": bool(
                report.get("require_model_end_to_end")
            ),
            "require_official_performance_parity": bool(
                report.get("require_official_performance_parity")
            ),
            "require_trace": require_trace,
            "min_tokens_per_second_per_user": (
                min_tokens_per_second_per_user
            ),
            "baseline_tokens_per_second_per_user": (
                baseline_tokens_per_second_per_user
            ),
            "baseline_reference": report.get("baseline_reference"),
            "baseline_reference_entry": (
                _performance_baseline_entry_summary(
                    report.get("baseline_reference_entry")
                )
            ),
            "min_baseline_ratio": min_baseline_ratio,
            "require_decode_shell_numeric_reference": (
                require_decode_shell_numeric_reference
            ),
            "checks": [],
            "message": "Dry run only; runtime acceptance was not evaluated.",
        }

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
    depth_sweep = steps.get("decode_depth_sweep", {})
    autotune = steps.get("decode_step_autotune", {})
    decode_contract = report.get("decode_step_contract") or {}
    single_layer_tensorization = _step_tensorization_summary(single_layer)
    decode_shell_tensorization = _step_tensorization_summary(decode_shell)
    smoke_tensorization = _step_tensorization_summary(smoke)
    profile_tensorization = _step_tensorization_summary(profile)
    attention_primitives_environment = _step_ttnn_environment(
        attention_primitives
    )
    attention_layer_environment = _step_ttnn_environment(attention_layer)
    single_layer_environment = _step_ttnn_environment(single_layer)
    smoke_environment = _step_ttnn_environment(smoke)
    profile_environment = _step_ttnn_environment(profile)
    single_layer_trace = _step_trace_summary(single_layer)
    smoke_trace = _step_trace_summary(smoke)
    profile_trace = _step_trace_summary(profile)
    expected_layers = report.get("layers")
    expected_layer_ids = _expected_layer_ids(expected_layers)
    program_num_layers = report.get("program_num_layers")
    program_batch_size = report.get("program_batch_size")
    program_cache_len = report.get("program_cache_len")
    program_seq_len = report.get("program_seq_len")
    program_hidden_size = report.get("program_hidden_size")
    program_num_attention_heads = report.get("program_num_attention_heads")
    program_num_kv_heads = report.get("program_num_key_value_heads")
    program_head_dim = report.get("program_head_dim")
    expected_batch_size = report.get("batch_size")
    expected_cache_len = report.get("cache_len")
    expected_trace_iterations = report.get("trace_iterations")
    expected_output_kind = decode_contract.get("output_kind")
    lm_head_split_count = _safe_int(materialize.get("lm_head_split_count"))
    expected_token_input_shape = [expected_batch_size, 1]
    expected_page_table_shape = [
        expected_batch_size,
        decode_contract.get("page_count"),
    ]
    expected_cache_position_shape = [expected_batch_size]
    expected_max_num_blocks = (
        expected_batch_size * decode_contract.get("page_count")
        if isinstance(expected_batch_size, int)
        and isinstance(decode_contract.get("page_count"), int)
        else None
    )
    expected_kv_cache_shape = [
        expected_max_num_blocks,
        program_num_kv_heads,
        decode_contract.get("kv_page_block_size"),
        program_head_dim,
    ]
    expected_logical_kv_cache_shape = [
        expected_batch_size,
        expected_cache_len,
        program_num_kv_heads,
        program_head_dim,
    ]
    throughput = profile.get("throughput_summary") or {}
    throughput_baseline = _throughput_baseline_summary(report, profile)
    profile_section_latency = profile.get("section_latency_ms")
    profile_lm_head = profile.get("lm_head_profile")
    profile_layer_profiles = profile.get("layer_profiles")
    profile_bottleneck = profile.get("bottleneck_summary")
    profile_generate_status = profile_generate.get("status")
    profile_generate_evaluated = profile_generate_status not in {
        None,
        "skipped",
    }
    skip_autotune = bool(report.get("skip_autotune"))
    checks = [
        _acceptance_check(
            "official_config_diff.status",
            official_config_diff.get("status") == "pass",
            observed=official_config_diff.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "official_config_diff.diff_status",
            official_config_diff.get("diff_status")
            in {"match", "diff_found"},
            observed=official_config_diff.get("diff_status"),
            expected=["match", "diff_found"],
        ),
        _acceptance_check(
            "official_config_diff.issue_count",
            _nonnegative_number(official_config_diff.get("issue_count")),
            observed=official_config_diff.get("issue_count"),
            minimum=0,
        ),
        _acceptance_check(
            "official_config_diff.sections",
            _contains_all(
                official_config_diff.get("sections"),
                list(PARITY_SECTIONS),
            ),
            observed=official_config_diff.get("sections"),
            expected=list(PARITY_SECTIONS),
        ),
        _acceptance_check(
            "official_config_diff.gap_summary",
            _config_gap_summary_complete(
                official_config_diff.get("gap_summary")
            ),
            observed=_config_gap_summary_observed(
                official_config_diff.get("gap_summary")
            ),
            expected=list(PARITY_SECTIONS),
        ),
        _acceptance_check(
            "official_config_diff.official_required_fields",
            _official_required_field_coverage_complete(
                official_config_diff.get("official_required_field_coverage")
            ),
            observed=_official_required_field_coverage_observed(
                official_config_diff.get("official_required_field_coverage")
            ),
            expected="complete",
        ),
        _acceptance_check(
            "materialize_parameters.tensor_count",
            _positive_number(materialize.get("tensor_count")),
            observed=materialize.get("tensor_count"),
            minimum=1,
        ),
        _acceptance_check(
            "materialize_parameters.layer_ids",
            materialize.get("materialized_layer_ids") == expected_layer_ids,
            observed=materialize.get("materialized_layer_ids"),
            expected=expected_layer_ids,
        ),
        _acceptance_check(
            "materialize_parameters.lm_head_split_count",
            _positive_number(materialize.get("lm_head_split_count")),
            observed=materialize.get("lm_head_split_count"),
            minimum=1,
        ),
        _acceptance_check(
            "materialize_parameters.required_tensor_paths",
            materialize.get("missing_required_tensor_paths") == [],
            observed=materialize.get("missing_required_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "materialize_parameters.tensor_shapes",
            materialize.get("materialized_tensor_shape_mismatches") == [],
            observed=materialize.get("materialized_tensor_shape_mismatches"),
            expected=[],
        ),
        _acceptance_check(
            "materialize_parameters.lm_head_source_reference",
            _lm_head_source_reference_complete(materialize),
            observed=_lm_head_source_reference_observed(materialize),
            expected={
                "lm_head.weight.materialization": "metadata_reference",
                "lm_head.weight.materialized": False,
                "lm_head.splits.0.weight.source_read": "sliced_tensor",
            },
        ),
        _acceptance_check(
            "decode_step_contract.decode_seq_len",
            _int_equal(decode_contract.get("decode_seq_len"), 1),
            observed=decode_contract.get("decode_seq_len"),
            expected=1,
        ),
        _acceptance_check(
            "decode_step_contract.token_input_shape",
            _int_list(decode_contract.get("token_input_shape"))
            == expected_token_input_shape,
            observed=decode_contract.get("token_input_shape"),
            expected=expected_token_input_shape,
        ),
        _acceptance_check(
            "decode_step_contract.paged_kv_cache",
            decode_contract.get("uses_paged_kv_cache") is True,
            observed={
                "uses_paged_kv_cache": decode_contract.get(
                    "uses_paged_kv_cache"
                ),
                "kv_cache_policy": decode_contract.get("kv_cache_policy"),
                "kv_cache_template": decode_contract.get(
                    "kv_cache_template"
                ),
            },
            expected=True,
        ),
        _acceptance_check(
            "decode_step_contract.kv_page_block_size",
            _positive_number(decode_contract.get("kv_page_block_size")),
            observed=decode_contract.get("kv_page_block_size"),
            minimum=1,
        ),
        _acceptance_check(
            "decode_step_contract.max_num_blocks",
            _int_equal(
                decode_contract.get("max_num_blocks"),
                expected_max_num_blocks,
            ),
            observed=decode_contract.get("max_num_blocks"),
            expected=expected_max_num_blocks,
        ),
        _acceptance_check(
            "decode_step_contract.page_table_shape",
            _int_list(decode_contract.get("page_table_shape"))
            == expected_page_table_shape,
            observed=decode_contract.get("page_table_shape"),
            expected=expected_page_table_shape,
        ),
        _acceptance_check(
            "decode_step_contract.cache_position_shape",
            _int_list(decode_contract.get("cache_position_shape"))
            == expected_cache_position_shape,
            observed=decode_contract.get("cache_position_shape"),
            expected=expected_cache_position_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_shape",
            _int_list(decode_contract.get("kv_cache_shape"))
            == expected_kv_cache_shape,
            observed=decode_contract.get("kv_cache_shape"),
            expected=expected_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_physical_shape",
            _int_list(decode_contract.get("kv_cache_physical_shape"))
            == expected_kv_cache_shape,
            observed=decode_contract.get("kv_cache_physical_shape"),
            expected=expected_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_logical_shape",
            _int_list(decode_contract.get("kv_cache_logical_shape"))
            == expected_logical_kv_cache_shape,
            observed=decode_contract.get("kv_cache_logical_shape"),
            expected=expected_logical_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.output_kind",
            decode_contract.get("output_kind") in {"token", "logits"},
            observed=decode_contract.get("output_kind"),
            expected=["token", "logits"],
        ),
        _acceptance_check(
            "decode_shell.layers",
            _int_equal(decode_shell.get("layers"), expected_layers),
            observed=decode_shell.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "decode_shell.parameter_source",
            decode_shell.get("parameter_source") == "hf_model",
            observed=decode_shell.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "decode_shell.input_source",
            _runtime_input_source_supported(decode_shell),
            observed=decode_shell.get("input_source"),
            expected=["synthetic", "prompt_runtime"],
        ),
        _acceptance_check(
            "decode_shell.runtime_input_tensor_count",
            _decode_shell_runtime_inputs_accepted(decode_shell),
            observed={
                "input_source": decode_shell.get("input_source"),
                "runtime_input_tensor_count": decode_shell.get(
                    "runtime_input_tensor_count"
                ),
                "synthetic_runtime_input_tensor_count": decode_shell.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "prompt_runtime_input_tensor_count": decode_shell.get(
                    "prompt_runtime_input_tensor_count"
                ),
            },
            expected=(
                "synthetic token_ids or one prompt token_ids tensor with no "
                "synthetic shell runtime input"
            ),
        ),
        _acceptance_check(
            "decode_shell.runtime_status",
            decode_shell.get("runtime_status") == "passed",
            observed=decode_shell.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "decode_shell.reference_status",
            decode_shell.get("reference_status") == "passed",
            observed=decode_shell.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "decode_shell.reference_failed_checks",
            decode_shell.get("reference_failed_checks") == [],
            observed=decode_shell.get("reference_failed_checks"),
            expected=[],
        ),
        _acceptance_check(
            "decode_shell.observed_op_sequence",
            _observed_ops_cover_planned(
                decode_shell.get("reference_planned_ops"),
                decode_shell.get("reference_observed_ops"),
            ),
            observed=decode_shell.get("reference_observed_ops"),
            expected=decode_shell.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "decode_shell.tensorization_status",
            decode_shell_tensorization.get("status") == "pass",
            observed=decode_shell_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "decode_shell.tensorization_roles",
            decode_shell_tensorization.get("roles")
            == ["embedding", "norm", "mlp", "lm_head"],
            observed=decode_shell_tensorization.get("roles"),
            expected=["embedding", "norm", "mlp", "lm_head"],
        ),
        _acceptance_check(
            "decode_shell.required_tensorized_tensor_paths",
            decode_shell.get("missing_required_tensorized_tensor_paths") == [],
            observed=decode_shell.get("missing_required_tensorized_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "decode_shell.tensorized_physical_shapes",
            _tensorized_physical_shape_mismatches(
                decode_shell_tensorization
            )
            == [],
            observed=_tensorized_physical_shape_mismatches(
                decode_shell_tensorization
            ),
            expected=[],
        ),
        _acceptance_check(
            "decode_shell.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                decode_shell_tensorization,
                layer_count=expected_layers,
            ),
            observed=_embedding_norm_weight_transform_observed(
                decode_shell_tensorization,
                layer_count=expected_layers,
            ),
        ),
        _acceptance_check(
            "decode_shell.linear_weight_transforms",
            _decode_shell_linear_weight_transform_complete(
                decode_shell_tensorization,
                layer_count=expected_layers,
                split_count=materialize.get("lm_head_split_count"),
            ),
            observed=_decode_shell_linear_weight_transform_observed(
                decode_shell_tensorization,
                layer_count=expected_layers,
                split_count=materialize.get("lm_head_split_count"),
            ),
        ),
        _acceptance_check(
            "attention_primitives.primitive_count",
            _int_equal(
                attention_primitives.get("primitive_count"),
                len(ATTENTION_PRIMITIVES),
            ),
            observed=attention_primitives.get("primitive_count"),
            expected=len(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_primitives.primitive_sequence",
            attention_primitives.get("primitive_sequence")
            == list(ATTENTION_PRIMITIVES),
            observed=attention_primitives.get("primitive_sequence"),
            expected=list(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_primitives.status",
            attention_primitives.get("status") == "pass",
            observed=attention_primitives.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "attention_primitives.runtime_status_counts",
            attention_primitives.get("runtime_status_counts")
            == {"passed": len(ATTENTION_PRIMITIVES)},
            observed=attention_primitives.get("runtime_status_counts"),
            expected={"passed": len(ATTENTION_PRIMITIVES)},
        ),
        _acceptance_check(
            "attention_primitives.ttnn_module_available",
            attention_primitives_environment.get("module_available") is True,
            observed=attention_primitives_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "attention_primitives.ttnn_version",
            _ttnn_runtime_identity_available(
                attention_primitives_environment
            ),
            observed=_ttnn_runtime_identity_observed(
                attention_primitives_environment
            ),
            expected="non-empty version or importable source module path",
            required=True,
        ),
        _acceptance_check(
            "attention_primitives.tt_metal_git_commit",
            _non_empty_string(
                attention_primitives_environment.get("tt_metal_git_commit")
            ),
            observed=attention_primitives_environment.get(
                "tt_metal_git_commit"
            ),
            source=attention_primitives_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "attention_primitives.primitive_reports",
            _attention_primitive_reports_complete(
                attention_primitives.get("primitive_reports"),
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_heads=program_num_attention_heads,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
            ),
            observed=_attention_primitive_reports_observed(
                attention_primitives.get("primitive_reports")
            ),
            expected=list(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_layer.layer",
            _int_equal(attention_layer.get("layer"), 0),
            observed=attention_layer.get("layer"),
            expected=0,
        ),
        _acceptance_check(
            "attention_layer.batch_size",
            _int_equal(attention_layer.get("batch_size"), expected_batch_size),
            observed=attention_layer.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "attention_layer.cache_len",
            _int_equal(attention_layer.get("cache_len"), expected_cache_len),
            observed=attention_layer.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "attention_layer.hidden_size",
            _int_equal(attention_layer.get("hidden_size"), program_hidden_size),
            observed=attention_layer.get("hidden_size"),
            expected=program_hidden_size,
        ),
        _acceptance_check(
            "attention_layer.num_kv_heads",
            _int_equal(
                attention_layer.get("num_kv_heads"),
                program_num_kv_heads,
            ),
            observed=attention_layer.get("num_kv_heads"),
            expected=program_num_kv_heads,
        ),
        _acceptance_check(
            "attention_layer.head_dim",
            _int_equal(attention_layer.get("head_dim"), program_head_dim),
            observed=attention_layer.get("head_dim"),
            expected=program_head_dim,
        ),
        _acceptance_check(
            "attention_layer.latency_ms",
            _nonnegative_number(attention_layer.get("latency_ms")),
            observed=attention_layer.get("latency_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "attention_layer.tensor_conversion_count",
            _positive_number(attention_layer.get("tensor_conversion_count")),
            observed=attention_layer.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "attention_layer.memory_config_conversion_count",
            _positive_number(
                attention_layer.get("memory_config_conversion_count")
            ),
            observed=attention_layer.get("memory_config_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "attention_layer.runtime_status",
            attention_layer.get("runtime_status") == "passed",
            observed=attention_layer.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "attention_layer.ttnn_module_available",
            attention_layer_environment.get("module_available") is True,
            observed=attention_layer_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "attention_layer.ttnn_version",
            _ttnn_runtime_identity_available(attention_layer_environment),
            observed=_ttnn_runtime_identity_observed(
                attention_layer_environment
            ),
            expected="non-empty version or importable source module path",
            required=True,
        ),
        _acceptance_check(
            "attention_layer.tt_metal_git_commit",
            _non_empty_string(
                attention_layer_environment.get("tt_metal_git_commit")
            ),
            observed=attention_layer_environment.get(
                "tt_metal_git_commit"
            ),
            source=attention_layer_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "attention_layer.reference_status",
            attention_layer.get("reference_status") == "passed",
            observed=attention_layer.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "attention_layer.reference_failed_checks",
            attention_layer.get("reference_failed_checks") == [],
            observed=attention_layer.get("reference_failed_checks"),
            expected=[],
        ),
        _acceptance_check(
            "attention_layer.primitive_sequence",
            attention_layer.get("primitive_sequence")
            == list(ATTENTION_LAYER_OPS),
            observed=attention_layer.get("primitive_sequence"),
            expected=list(ATTENTION_LAYER_OPS),
        ),
        _acceptance_check(
            "attention_layer.primitive_reports",
            _attention_layer_primitive_reports_complete(
                attention_layer.get("primitive_reports")
            ),
            observed=_attention_layer_primitive_reports_observed(
                attention_layer.get("primitive_reports")
            ),
            expected=list(ATTENTION_LAYER_OPS),
        ),
        _acceptance_check(
            "attention_layer.output_shapes",
            _attention_layer_output_shapes_complete(
                attention_layer.get("output_shapes"),
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_attention_layer_output_shape_observed(
                attention_layer.get("output_shapes")
            ),
            expected=_expected_attention_layer_output_shape_summary(
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "attention_layer.observed_op_sequence",
            _observed_ops_cover_planned(
                attention_layer.get("reference_planned_observed_ops"),
                attention_layer.get("reference_observed_ops"),
            ),
            observed=attention_layer.get("reference_observed_ops"),
            expected=attention_layer.get("reference_planned_observed_ops"),
        ),
        _acceptance_check(
            "single_layer_decode.parameter_source",
            single_layer.get("parameter_source") == "hf_model",
            observed=single_layer.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "single_layer_decode.input_source",
            _runtime_input_source_supported(single_layer),
            observed=single_layer.get("input_source"),
            expected=["synthetic", "prompt_runtime"],
        ),
        _acceptance_check(
            "single_layer_decode.synthetic_runtime_inputs",
            _synthetic_runtime_inputs_accepted(single_layer),
            observed=single_layer.get(
                "synthetic_runtime_input_tensor_count"
            ),
            input_source=single_layer.get("input_source"),
        ),
        _acceptance_check(
            "single_layer_decode.runtime_inputs",
            _decode_runtime_inputs_complete(
                single_layer,
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_runtime_input_observed(single_layer),
            expected=_expected_decode_runtime_input_summary(
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "single_layer_decode.layers",
            _int_equal(single_layer.get("layers"), 1),
            observed=single_layer.get("layers"),
            expected=1,
        ),
        _acceptance_check(
            "single_layer_decode.batch_size",
            _int_equal(single_layer.get("batch_size"), expected_batch_size),
            observed=single_layer.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "single_layer_decode.cache_len",
            _int_equal(single_layer.get("cache_len"), expected_cache_len),
            observed=single_layer.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "single_layer_decode.tensor_conversion_count",
            _positive_number(single_layer.get("tensor_conversion_count")),
            observed=single_layer.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.runtime_status",
            single_layer.get("runtime_status") == "passed",
            observed=single_layer.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "single_layer_decode.ttnn_module_available",
            single_layer_environment.get("module_available") is True,
            observed=single_layer_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "single_layer_decode.ttnn_version",
            _ttnn_runtime_identity_available(single_layer_environment),
            observed=_ttnn_runtime_identity_observed(
                single_layer_environment
            ),
            expected="non-empty version or importable source module path",
            required=True,
        ),
        _acceptance_check(
            "single_layer_decode.tt_metal_git_commit",
            _non_empty_string(
                single_layer_environment.get("tt_metal_git_commit")
            ),
            observed=single_layer_environment.get("tt_metal_git_commit"),
            source=single_layer_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_status",
            single_layer_tensorization.get("status") == "pass",
            observed=single_layer_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_roles",
            _contains_all(
                single_layer_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=single_layer_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "single_layer_decode.required_tensorized_tensor_paths",
            single_layer.get("missing_required_tensorized_tensor_paths")
            == [],
            observed=single_layer.get(
                "missing_required_tensorized_tensor_paths"
            ),
            expected=[],
        ),
        _acceptance_check(
            "single_layer_decode.tensorized_physical_shapes",
            _tensorized_physical_shape_mismatches(
                single_layer_tensorization
            )
            == [],
            observed=_tensorized_physical_shape_mismatches(
                single_layer_tensorization
            ),
            expected=[],
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_memory_configs",
            _positive_count(
                single_layer_tensorization.get("memory_config_counts")
            ),
            observed=single_layer_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_ttnn_memory_configs",
            _positive_count(
                single_layer_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=single_layer_tensorization.get(
                "ttnn_memory_config_counts"
            ),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                single_layer_tensorization,
                layer_count=1,
            ),
            observed=_embedding_norm_weight_transform_observed(
                single_layer_tensorization,
                layer_count=1,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(1),
            },
        ),
        _acceptance_check(
            "single_layer_decode.linear_weight_transforms",
            _linear_weight_transform_complete(
                single_layer_tensorization,
                layer_count=1,
            ),
            observed=_linear_weight_transform_observed(
                single_layer_tensorization,
                layer_count=1,
            ),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "paths": _linear_weight_transform_paths(1),
            },
        ),
        _acceptance_check(
            "single_layer_decode.lm_head_transform",
            _lm_head_transform_complete(
                single_layer_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(single_layer_tensorization),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "single_layer_decode.reference_status",
            single_layer.get("reference_status") == "passed",
            observed=single_layer.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "single_layer_decode.reference_failed_checks",
            single_layer.get("reference_failed_checks") == [],
            observed=single_layer.get("reference_failed_checks"),
            expected=[],
        ),
        _acceptance_check(
            "single_layer_decode.output_shapes",
            _decode_output_shapes_complete(
                single_layer.get("output_shapes"),
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                single_layer.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "single_layer_decode.observed_op_sequence",
            _observed_ops_cover_planned(
                single_layer.get("reference_planned_ops"),
                single_layer.get("reference_observed_ops"),
            ),
            observed=single_layer.get("reference_observed_ops"),
            expected=single_layer.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "smoke_decode_step.parameter_source",
            smoke.get("parameter_source") == "hf_model",
            observed=smoke.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "smoke_decode_step.input_source",
            _runtime_input_source_supported(smoke),
            observed=smoke.get("input_source"),
            expected=["synthetic", "prompt_runtime"],
        ),
        _acceptance_check(
            "smoke_decode_step.synthetic_runtime_inputs",
            _synthetic_runtime_inputs_accepted(smoke),
            observed=smoke.get("synthetic_runtime_input_tensor_count"),
            input_source=smoke.get("input_source"),
        ),
        _acceptance_check(
            "smoke_decode_step.runtime_inputs",
            _decode_runtime_inputs_complete(
                smoke,
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_runtime_input_observed(smoke),
            expected=_expected_decode_runtime_input_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "smoke_decode_step.layers",
            _int_equal(smoke.get("layers"), expected_layers),
            observed=smoke.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "smoke_decode_step.batch_size",
            _int_equal(smoke.get("batch_size"), expected_batch_size),
            observed=smoke.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "smoke_decode_step.cache_len",
            _int_equal(smoke.get("cache_len"), expected_cache_len),
            observed=smoke.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "smoke_decode_step.tensor_conversion_count",
            _positive_number(smoke.get("tensor_conversion_count")),
            observed=smoke.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.runtime_status",
            smoke.get("runtime_status") == "passed",
            observed=smoke.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "smoke_decode_step.ttnn_module_available",
            smoke_environment.get("module_available") is True,
            observed=smoke_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "smoke_decode_step.ttnn_version",
            _ttnn_runtime_identity_available(smoke_environment),
            observed=_ttnn_runtime_identity_observed(smoke_environment),
            expected="non-empty version or importable source module path",
            required=True,
        ),
        _acceptance_check(
            "smoke_decode_step.tt_metal_git_commit",
            _non_empty_string(
                smoke_environment.get("tt_metal_git_commit")
            ),
            observed=smoke_environment.get("tt_metal_git_commit"),
            source=smoke_environment.get("tt_metal_git_commit_source"),
            required=True,
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_status",
            smoke_tensorization.get("status") == "pass",
            observed=smoke_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_roles",
            _contains_all(
                smoke_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=smoke_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "smoke_decode_step.required_tensorized_tensor_paths",
            smoke.get("missing_required_tensorized_tensor_paths") == [],
            observed=smoke.get("missing_required_tensorized_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "smoke_decode_step.tensorized_physical_shapes",
            _tensorized_physical_shape_mismatches(smoke_tensorization) == [],
            observed=_tensorized_physical_shape_mismatches(
                smoke_tensorization
            ),
            expected=[],
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_memory_configs",
            _positive_count(smoke_tensorization.get("memory_config_counts")),
            observed=smoke_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_ttnn_memory_configs",
            _positive_count(
                smoke_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=smoke_tensorization.get("ttnn_memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            observed=_embedding_norm_weight_transform_observed(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(
                    expected_layers
                ),
            },
        ),
        _acceptance_check(
            "smoke_decode_step.linear_weight_transforms",
            _linear_weight_transform_complete(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            observed=_linear_weight_transform_observed(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "paths": _linear_weight_transform_paths(expected_layers),
            },
        ),
        _acceptance_check(
            "smoke_decode_step.lm_head_transform",
            _lm_head_transform_complete(
                smoke_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(smoke_tensorization),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "smoke_decode_step.reference_status",
            smoke.get("reference_status") == "passed",
            observed=smoke.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "smoke_decode_step.reference_failed_checks",
            smoke.get("reference_failed_checks") == [],
            observed=smoke.get("reference_failed_checks"),
            expected=[],
        ),
        _acceptance_check(
            "smoke_decode_step.output_shapes",
            _decode_output_shapes_complete(
                smoke.get("output_shapes"),
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                smoke.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "smoke_decode_step.observed_op_sequence",
            _observed_ops_cover_planned(
                smoke.get("reference_planned_ops"),
                smoke.get("reference_observed_ops"),
            ),
            observed=smoke.get("reference_observed_ops"),
            expected=smoke.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "profile_decode_step.parameter_source",
            profile.get("parameter_source") == "hf_model",
            observed=profile.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "profile_decode_step.input_source",
            _runtime_input_source_supported(profile),
            observed=profile.get("input_source"),
            expected=["synthetic", "prompt_runtime"],
        ),
        _acceptance_check(
            "profile_decode_step.synthetic_runtime_inputs",
            _synthetic_runtime_inputs_accepted(profile),
            observed=profile.get("synthetic_runtime_input_tensor_count"),
            input_source=profile.get("input_source"),
        ),
        _acceptance_check(
            "profile_decode_step.runtime_inputs",
            _decode_runtime_inputs_complete(
                profile,
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_runtime_input_observed(profile),
            expected=_expected_decode_runtime_input_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "profile_decode_step.layers",
            _int_equal(profile.get("layers"), expected_layers),
            observed=profile.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "profile_decode_step.batch_size",
            _int_equal(profile.get("batch_size"), expected_batch_size),
            observed=profile.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "profile_decode_step.cache_len",
            _int_equal(profile.get("cache_len"), expected_cache_len),
            observed=profile.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "profile_decode_step.tensor_conversion_count",
            _positive_number(profile.get("tensor_conversion_count")),
            observed=profile.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.tensor_conversion_ms",
            _nonnegative_number(profile.get("tensor_conversion_ms")),
            observed=profile.get("tensor_conversion_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.runtime_status",
            profile.get("runtime_status") == "profiled",
            observed=profile.get("runtime_status"),
            expected="profiled",
        ),
        _acceptance_check(
            "profile_decode_step.ttnn_module_available",
            profile_environment.get("module_available") is True,
            observed=profile_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "profile_decode_step.ttnn_version",
            _ttnn_runtime_identity_available(profile_environment),
            observed=_ttnn_runtime_identity_observed(profile_environment),
            expected="non-empty version or importable source module path",
            required=True,
        ),
        _acceptance_check(
            "profile_decode_step.tt_metal_git_commit",
            _non_empty_string(
                profile_environment.get("tt_metal_git_commit")
            ),
            observed=profile_environment.get("tt_metal_git_commit"),
            source=profile_environment.get("tt_metal_git_commit_source"),
            required=True,
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_status",
            profile_tensorization.get("status") == "pass",
            observed=profile_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_roles",
            _contains_all(
                profile_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=profile_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "profile_decode_step.required_tensorized_tensor_paths",
            profile.get("missing_required_tensorized_tensor_paths") == [],
            observed=profile.get("missing_required_tensorized_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "profile_decode_step.tensorized_physical_shapes",
            _tensorized_physical_shape_mismatches(profile_tensorization)
            == [],
            observed=_tensorized_physical_shape_mismatches(
                profile_tensorization
            ),
            expected=[],
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_memory_configs",
            _positive_count(profile_tensorization.get("memory_config_counts")),
            observed=profile_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_ttnn_memory_configs",
            _positive_count(
                profile_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=profile_tensorization.get("ttnn_memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            observed=_embedding_norm_weight_transform_observed(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(
                    expected_layers
                ),
            },
        ),
        _acceptance_check(
            "profile_decode_step.linear_weight_transforms",
            _linear_weight_transform_complete(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            observed=_linear_weight_transform_observed(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "paths": _linear_weight_transform_paths(expected_layers),
            },
        ),
        _acceptance_check(
            "profile_decode_step.lm_head_transform",
            _lm_head_transform_complete(
                profile_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(profile_tensorization),
            expected={
                "transform": LINEAR_WEIGHT_TRANSFORM,
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "profile_decode_step.reference_status",
            profile.get("reference_status") == "passed",
            observed=profile.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "profile_decode_step.reference_failed_checks",
            profile.get("reference_failed_checks") == [],
            observed=profile.get("reference_failed_checks"),
            expected=[],
        ),
        _acceptance_check(
            "profile_decode_step.output_shapes",
            _decode_output_shapes_complete(
                profile.get("output_shapes"),
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                profile.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "profile_decode_step.lm_head_profile",
            _lm_head_profile_complete(
                profile_lm_head,
                output_kind=expected_output_kind,
            ),
            observed=_lm_head_profile_observed(profile_lm_head),
            expected={
                "split_count": "positive",
                "lm_head_ms": "nonnegative",
                "argmax_ms": "nonnegative",
                "argmax_status": (
                    "skipped"
                    if expected_output_kind == "logits"
                    else "profiled"
                ),
            },
        ),
        _acceptance_check(
            "profile_decode_step.observed_op_sequence",
            _observed_ops_cover_planned(
                profile.get("reference_planned_ops"),
                profile.get("reference_observed_ops"),
            ),
            observed=profile.get("reference_observed_ops"),
            expected=profile.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "profile_decode_step.section_latency_ms",
            _has_nonnegative_fields(
                profile_section_latency,
                PROFILE_SECTION_LATENCY_KEYS,
            ),
            observed=_field_keys(profile_section_latency),
            expected=list(PROFILE_SECTION_LATENCY_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.layer_profile_count",
            _layer_profile_ids(profile_layer_profiles) == expected_layer_ids,
            observed=_layer_profile_ids(profile_layer_profiles),
            expected=expected_layer_ids,
        ),
        _acceptance_check(
            "profile_decode_step.layer_profile_sections",
            _layer_profiles_have_nonnegative_fields(
                profile_layer_profiles,
                PROFILE_LAYER_LATENCY_KEYS,
            ),
            observed=_layer_profile_field_keys(profile_layer_profiles),
            expected=list(PROFILE_LAYER_LATENCY_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.bottleneck_summary",
            _bottleneck_summary_complete(profile_bottleneck),
            observed=_bottleneck_summary_observed(profile_bottleneck),
            expected=list(PROFILE_BOTTLENECK_SECTION_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.throughput_status",
            throughput.get("status") == "measured",
            observed=throughput.get("status"),
            expected="measured",
        ),
        _acceptance_check(
            "profile_decode_step.latency_ms",
            _positive_number(throughput.get("latency_ms")),
            observed=throughput.get("latency_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.tokens_per_second_per_user",
            _positive_number(
                throughput.get("tokens_per_second_per_user")
            ),
            observed=throughput.get("tokens_per_second_per_user"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.aggregate_tokens_per_second",
            _positive_number(
                throughput.get("aggregate_tokens_per_second")
            ),
            observed=throughput.get("aggregate_tokens_per_second"),
            minimum=0,
        ),
    ]
    if profile_generate_evaluated:
        checks.extend(
            [
                _acceptance_check(
                    "profile_generate.full_generated_model_can_run",
                    profile_generate.get("generate_passed") is True,
                    observed={
                        "status": profile_generate.get("status"),
                        "runtime_status": profile_generate.get(
                            "runtime_status"
                        ),
                        "generate_status": profile_generate.get(
                            "generate_status"
                        ),
                        "generate_passed": profile_generate.get(
                            "generate_passed"
                        ),
                    },
                    expected="generate_passed=true",
                ),
                _acceptance_check(
                    "profile_generate.tokens_per_second_per_user_positive",
                    _positive_number(
                        profile_generate.get("tokens_per_second_per_user")
                    ),
                    observed=profile_generate.get(
                        "tokens_per_second_per_user"
                    ),
                    minimum=0,
                ),
                _acceptance_check(
                    "profile_generate.aggregate_tokens_per_second_positive",
                    _positive_number(
                        profile_generate.get("aggregate_tokens_per_second")
                    ),
                    observed=profile_generate.get(
                        "aggregate_tokens_per_second"
                    ),
                    minimum=0,
                ),
                _acceptance_check(
                    "profile_generate.no_official_parity_claim",
                    (
                        profile_generate.get(
                            "official_performance_parity_claimed"
                        )
                        is False
                    ),
                    observed=profile_generate.get(
                        "official_performance_parity_claimed"
                    ),
                    expected=False,
                ),
                _acceptance_check(
                    "profile_generate.profile_fields",
                    (
                        _contains_all(
                            _field_keys(profile_generate.get("sections")),
                            list(PROFILE_GENERATE_SECTION_KEYS),
                        )
                        and isinstance(
                            profile_generate.get("per_layer"),
                            dict,
                        )
                    ),
                    observed={
                        "sections": _field_keys(
                            profile_generate.get("sections")
                        ),
                        "per_layer": _field_keys(
                            profile_generate.get("per_layer")
                        ),
                    },
                    expected={
                        "sections": list(PROFILE_GENERATE_SECTION_KEYS),
                        "per_layer": "dict",
                    },
                ),
                _acceptance_check(
                    "profile_generate.performance_milestones",
                    _profile_generate_milestones_complete(
                        profile_generate.get("performance_milestones")
                    ),
                    observed=[
                        milestone.get("id")
                        for milestone in (
                            (
                                profile_generate.get(
                                    "performance_milestones"
                                )
                                or {}
                            ).get("milestones", [])
                        )
                        if isinstance(milestone, dict)
                    ],
                    expected=list(PROFILE_GENERATE_MILESTONE_IDS),
                ),
            ]
        )
    checks.extend(
        [
            _acceptance_check(
                "decode_depth_sweep.status",
                depth_sweep.get("status") == "pass",
                observed=depth_sweep.get("status"),
                expected="pass",
            ),
            _acceptance_check(
                "decode_depth_sweep.acceptance",
                (depth_sweep.get("acceptance") or {}).get("passed") is True,
                observed=depth_sweep.get("acceptance"),
                expected="passed",
            ),
            _acceptance_check(
                "decode_depth_sweep.requested_depth",
                _int_list_contains(depth_sweep.get("depths"), expected_layers),
                observed=depth_sweep.get("depths"),
                expected=expected_layers,
            ),
            _acceptance_check(
                "decode_depth_sweep.passed_depth_count",
                _int_equal(
                    depth_sweep.get("passed_depth_count"),
                    depth_sweep.get("depth_count"),
                ),
                observed=depth_sweep.get("passed_depth_count"),
                expected=depth_sweep.get("depth_count"),
            ),
            _acceptance_check(
                "decode_depth_sweep.records",
                _decode_depth_sweep_records_complete(
                    depth_sweep.get("records"),
                    expected_depths=depth_sweep.get("depths"),
                    batch_size=expected_batch_size,
                    cache_len=expected_cache_len,
                    seq_len=program_seq_len,
                    vocab_size=report.get("program_vocab_size"),
                    num_kv_heads=program_num_kv_heads,
                    head_dim=program_head_dim,
                    output_kind=expected_output_kind,
                    page_block_size=decode_contract.get("kv_page_block_size"),
                    require_trace=require_trace,
                    trace_iterations=expected_trace_iterations,
                ),
                observed=_decode_depth_sweep_records_observed(
                    depth_sweep.get("records")
                ),
                expected={
                    "depths": depth_sweep.get("depths"),
                    "batch_size": expected_batch_size,
                    "cache_len": expected_cache_len,
                    "reference_status": "passed",
                    "throughput_status": "measured",
                    "trace_status": (
                        "captured_and_executed" if require_trace else None
                    ),
                },
            ),
        ]
    )
    if require_decode_shell_numeric_reference:
        checks.append(
            _acceptance_check(
                "decode_shell.numeric_reference",
                _decode_shell_numeric_reference_complete(
                    decode_shell,
                    expected_pcc_threshold=report.get(
                        "decode_shell_pcc_threshold"
                    ),
                ),
                observed=_decode_shell_numeric_reference_observed(
                    decode_shell
                ),
                expected={
                    "status": "passed",
                    "kind": "torch_decode_shell",
                    "passed": True,
                    "pcc": f">= {decode_shell.get('pcc_threshold')}",
                    "pcc_threshold": report.get(
                        "decode_shell_pcc_threshold"
                    ),
                    "failed_checks": [],
                },
            )
        )
    if require_official_config_match:
        checks.append(
            _acceptance_check(
                "official_config_diff.official_reference_format",
                official_config_diff.get("official_source_format")
                == "normalized_parity_config",
                observed={
                    "source_format": official_config_diff.get(
                        "official_source_format"
                    ),
                    "source": official_config_diff.get("official_source"),
                },
                expected="normalized_parity_config",
            )
        )
        checks.append(
            _acceptance_check(
                "official_config_diff.match",
                official_config_diff.get("diff_status") == "match",
                observed=official_config_diff.get("diff_status"),
                expected="match",
                issue_count=official_config_diff.get("issue_count"),
            )
        )
    if require_full_depth:
        checks.extend(
            [
                _acceptance_check(
                    "validation.full_depth_layers",
                    _int_equal(expected_layers, program_num_layers),
                    observed=expected_layers,
                    expected=program_num_layers,
                ),
                _acceptance_check(
                    "decode_depth_sweep.full_depth",
                    depth_sweep.get("covered_full_depth") is True,
                    observed=depth_sweep.get("max_depth"),
                    expected=program_num_layers,
                ),
            ]
        )
    if require_program_runtime_shape:
        checks.extend(
            [
                _acceptance_check(
                    "validation.program_batch_size",
                    _int_equal(expected_batch_size, program_batch_size),
                    observed=expected_batch_size,
                    expected=program_batch_size,
                ),
                _acceptance_check(
                    "validation.program_cache_len",
                    _int_equal(expected_cache_len, program_cache_len),
                    observed=expected_cache_len,
                    expected=program_cache_len,
                ),
            ]
        )
    if require_batch32_decode_step:
        checks.append(
            _acceptance_check(
                "decode_step_contract.batch32",
                _int_equal(decode_contract.get("batch_size"), 32),
                observed=decode_contract.get("batch_size"),
                expected=32,
            )
        )
    if require_model_end_to_end:
        runtime_scope = _runtime_input_scope(report)
        decode_loop_runtime_owned = bool(
            report.get("decode_loop_runtime_owned")
            or prompt_loop.get("decode_loop_runtime_owned")
        )
        generate_prefill_decode_ready = _generate_prefill_decode_ready(
            generate_step
        )
        checks.append(
            _acceptance_check(
                "model_end_to_end_readiness.ready",
                (
                    not runtime_scope.get("uses_synthetic_runtime_inputs")
                    and decode_loop_runtime_owned
                    and generate_prefill_decode_ready
                ),
                observed={
                    "status": runtime_scope.get("status"),
                    "synthetic_runtime_input_steps": runtime_scope.get(
                        "synthetic_runtime_input_steps"
                    ),
                    "runtime_input_sources": runtime_scope.get(
                        "runtime_input_sources"
                    ),
                    "decode_loop_runtime_owned": (
                        decode_loop_runtime_owned
                    ),
                    "generate_prefill_decode_ready": (
                        generate_prefill_decode_ready
                    ),
                    "prefill_status": generate_step.get("prefill_status"),
                    "kv_cache_source": generate_step.get("kv_cache_source"),
                    "generated_text_status": generate_step.get(
                        "generated_text_status"
                    ),
                },
                expected=(
                    "no synthetic runtime inputs, prompt decode loop "
                    "ownership, and prefill+decode generate evidence"
                ),
            )
        )
    if require_trace:
        checks.extend(
            [
                _acceptance_check(
                    "single_layer_decode.trace_status",
                    single_layer.get("trace_status")
                    == "captured_and_executed",
                    observed=single_layer.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "single_layer_decode.trace_iterations",
                    _int_equal(
                        single_layer_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=single_layer_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "single_layer_decode.trace_execute_sample_count",
                    _int_equal(
                        single_layer_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=single_layer_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "smoke_decode_step.trace_status",
                    smoke.get("trace_status") == "captured_and_executed",
                    observed=smoke.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "smoke_decode_step.trace_iterations",
                    _int_equal(
                        smoke_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=smoke_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "smoke_decode_step.trace_execute_sample_count",
                    _int_equal(
                        smoke_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=smoke_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "profile_decode_step.trace_status",
                    profile.get("trace_status") == "captured_and_executed",
                    observed=profile.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "profile_decode_step.trace_iterations",
                    _int_equal(
                        profile_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=profile_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "profile_decode_step.trace_execute_sample_count",
                    _int_equal(
                        profile_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=profile_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "profile_decode_step.trace_profile",
                    _trace_profile_complete(
                        profile_trace,
                        throughput,
                        expected_iterations=expected_trace_iterations,
                    ),
                    observed=_trace_profile_observed(
                        profile_trace,
                        throughput,
                    ),
                    expected={
                        "status": "captured_and_executed",
                        "iterations": expected_trace_iterations,
                        "execute_samples_ms": "positive",
                        "capture_latency_ms": "nonnegative",
                        "execute_latency_ms": "positive",
                        "trace_execute_mean_ms": "positive",
                        "trace_execute_tokens_per_second_per_user": (
                            "positive"
                        ),
                        "trace_execute_aggregate_tokens_per_second": (
                            "positive"
                        ),
                    },
                ),
                _acceptance_check(
                    (
                        "profile_decode_step."
                        "trace_execute_tokens_per_second_per_user"
                    ),
                    _positive_number(
                        throughput.get(
                            "trace_execute_tokens_per_second_per_user"
                        )
                    ),
                    observed=throughput.get(
                        "trace_execute_tokens_per_second_per_user"
                    ),
                    minimum=0,
                ),
            ]
        )

    if min_tokens_per_second_per_user is not None:
        observed = throughput.get("tokens_per_second_per_user")
        checks.append(
            _acceptance_check(
                "profile_decode_step.min_tokens_per_second_per_user",
                _number_at_least(observed, min_tokens_per_second_per_user),
                observed=observed,
                minimum=min_tokens_per_second_per_user,
            )
        )

    if baseline_tokens_per_second_per_user is not None:
        checks.append(
            _acceptance_check(
                "profile_decode_step.baseline_tokens_per_second_per_user",
                _positive_number(baseline_tokens_per_second_per_user),
                observed=baseline_tokens_per_second_per_user,
                minimum=0,
            )
        )
    if report.get("baseline_reference") is not None:
        baseline_entry = report.get("baseline_reference_entry")
        checks.append(
            _acceptance_check(
                "profile_decode_step.baseline_reference",
                (
                    _performance_baseline_entry_complete(baseline_entry)
                    and _numbers_equal(
                        baseline_tokens_per_second_per_user,
                        (
                            baseline_entry or {}
                        ).get("decode_tokens_per_second_per_user"),
                    )
                ),
                observed=_performance_baseline_entry_summary(baseline_entry),
                expected=report.get("baseline_reference"),
            )
        )
    if report.get("require_official_performance_parity"):
        checks.append(
            _acceptance_check(
                "profile_decode_step.official_baseline_reference",
                _official_performance_baseline_entry_complete(
                    report.get("baseline_reference_entry")
                ),
                observed=_performance_baseline_entry_summary(
                    report.get("baseline_reference_entry")
                ),
                expected={
                    "role": "official_8b_target",
                    "model": "Llama 3.1 8B",
                    "batch_size": 32,
                },
            )
        )
        checks.append(
            _acceptance_check(
                "profile_decode_step.official_min_baseline_ratio_positive",
                _positive_number(min_baseline_ratio),
                observed=min_baseline_ratio,
                expected="> 0.0",
            )
        )
        checks.append(
            _acceptance_check(
                "decode_step_autotune.metric",
                autotune.get("metric") == OFFICIAL_PERFORMANCE_PARITY_METRIC,
                observed=autotune.get("metric"),
                expected=OFFICIAL_PERFORMANCE_PARITY_METRIC,
            )
        )
    if min_baseline_ratio is not None:
        checks.append(
            _acceptance_check(
                "profile_decode_step.min_baseline_ratio",
                _number_at_least(
                    throughput_baseline.get("ratio"),
                    min_baseline_ratio,
                ),
                observed=throughput_baseline.get("ratio"),
                minimum=min_baseline_ratio,
                baseline=baseline_tokens_per_second_per_user,
                tokens_per_second_per_user=throughput.get(
                    "tokens_per_second_per_user"
                ),
            )
        )

    if not skip_autotune:
        checks.extend(
            [
                _acceptance_check(
                    "decode_step_autotune.status",
                    autotune.get("status") == "pass",
                    observed=autotune.get("status"),
                    expected="pass",
                ),
                _acceptance_check(
                    "decode_step_autotune.candidate_count",
                    _positive_number(autotune.get("candidate_count")),
                    observed=autotune.get("candidate_count"),
                    minimum=1,
                ),
                _acceptance_check(
                    "decode_step_autotune.knob_coverage",
                    _autotune_knob_coverage_complete(
                        autotune.get("knob_coverage"),
                        candidate_count=autotune.get("candidate_count"),
                    ),
                    observed=_autotune_knob_coverage_observed(
                        autotune.get("knob_coverage")
                    ),
                    expected=list(DECODE_STEP_AUTOTUNE_KNOBS),
                ),
                _acceptance_check(
                    "decode_step_autotune.output_kind_counts",
                    _autotune_output_kind_counts_complete(
                        autotune.get("output_kind_counts"),
                        autotune.get("knob_coverage"),
                    ),
                    observed=_autotune_output_kind_counts_observed(
                        autotune.get("output_kind_counts"),
                        autotune.get("knob_coverage"),
                    ),
                    expected="output kinds implied by generation_template",
                ),
                _acceptance_check(
                    "decode_step_autotune.candidates",
                    _autotune_candidates_complete(
                        autotune.get("candidate_summaries"),
                        candidate_count=autotune.get("candidate_count"),
                        layer_count=expected_layers,
                        batch_size=expected_batch_size,
                        seq_len=program_seq_len,
                        cache_len=expected_cache_len,
                        vocab_size=report.get("program_vocab_size"),
                        num_kv_heads=program_num_kv_heads,
                        head_dim=program_head_dim,
                        page_block_size=decode_contract.get(
                            "kv_page_block_size"
                        ),
                        out_dir=report.get("out_dir"),
                        require_trace=require_trace,
                    ),
                    observed=_autotune_candidates_observed(
                        autotune.get("candidate_summaries")
                    ),
                    expected={
                        "candidate_count": autotune.get("candidate_count"),
                        "status": "profiled",
                        "passed": True,
                        "parameter_source": "hf_model",
                        "reference_status": "passed",
                        "trace_status": (
                            "captured_and_executed"
                            if require_trace
                            else None
                        ),
                    },
                ),
                _acceptance_check(
                    "decode_step_autotune.leaderboard",
                    _autotune_leaderboard_complete(
                        autotune.get("leaderboard"),
                        candidate_count=autotune.get("candidate_count"),
                        candidate_summaries=autotune.get(
                            "candidate_summaries"
                        ),
                        best=autotune.get("best"),
                        require_trace=require_trace,
                    ),
                    observed=_autotune_leaderboard_observed(
                        autotune.get("leaderboard")
                    ),
                    expected={
                        "candidate_count": autotune.get("candidate_count"),
                        "best": autotune.get("best"),
                    },
                ),
                _acceptance_check(
                    "decode_step_autotune.best_candidate_summary",
                    _autotune_best_candidate_summary_complete(
                        autotune.get("best_candidate_summary"),
                        best=autotune.get("best"),
                        require_trace=require_trace,
                    ),
                    observed=_autotune_best_candidate_summary_observed(
                        autotune.get("best_candidate_summary")
                    ),
                    expected={
                        "best": autotune.get("best"),
                        "parameter_source": "hf_model",
                        "reference_status": "passed",
                    },
                ),
                _acceptance_check(
                    "decode_step_autotune.passed_candidate_count",
                    _positive_number(autotune.get("passed_candidate_count")),
                    observed=autotune.get("passed_candidate_count"),
                    minimum=1,
                ),
                _acceptance_check(
                    "decode_step_autotune.best",
                    _non_empty_string(autotune.get("best")),
                    observed=autotune.get("best"),
                    required=True,
                ),
                _acceptance_check(
                    "decode_step_autotune.best_reference_status",
                    autotune.get("best_reference_status") == "passed",
                    observed=autotune.get("best_reference_status"),
                    expected="passed",
                ),
                _acceptance_check(
                    "decode_step_autotune.best_parameter_source",
                    autotune.get("best_parameter_source") == "hf_model",
                    observed=autotune.get("best_parameter_source"),
                    expected="hf_model",
                ),
                _acceptance_check(
                    "decode_step_autotune.best_metric",
                    _nonnegative_number(autotune.get("best_metric")),
                    observed=autotune.get("best_metric"),
                    minimum=0,
                ),
            ]
        )
        if report.get("decode_step_search_space_is_default"):
            checks.append(
                _acceptance_check(
                    "decode_step_autotune.default_knob_variation",
                    _autotune_default_knobs_varied(
                        autotune.get("knob_coverage")
                    ),
                    observed=_autotune_knob_variation_observed(
                        autotune.get("knob_coverage")
                    ),
                    expected=list(DECODE_STEP_AUTOTUNE_KNOBS),
                )
            )
        if require_trace:
            checks.append(
                _acceptance_check(
                    "decode_step_autotune.best_trace_status",
                    autotune.get("best_trace_status")
                    == "captured_and_executed",
                    observed=autotune.get("best_trace_status"),
                    expected="captured_and_executed",
                )
            )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "require_official_config_match": require_official_config_match,
        "require_full_depth": require_full_depth,
        "require_program_runtime_shape": require_program_runtime_shape,
        "require_batch32_decode_step": require_batch32_decode_step,
        "require_full_decode_step": bool(report.get("require_full_decode_step")),
        "require_model_end_to_end": bool(
            report.get("require_model_end_to_end")
        ),
        "require_official_performance_parity": bool(
            report.get("require_official_performance_parity")
        ),
        "require_trace": require_trace,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "baseline_tokens_per_second_per_user": (
            baseline_tokens_per_second_per_user
        ),
        "baseline_reference": report.get("baseline_reference"),
        "baseline_reference_entry": _performance_baseline_entry_summary(
            report.get("baseline_reference_entry")
        ),
        "min_baseline_ratio": min_baseline_ratio,
        "throughput_baseline": throughput_baseline,
        "require_decode_shell_numeric_reference": (
            require_decode_shell_numeric_reference
        ),
        "checks": checks,
    }


def _attention_primitives_dry_run_complete(reports: Any) -> bool:
    if not isinstance(reports, dict):
        return False
    if set(reports) != set(ATTENTION_PRIMITIVES):
        return False
    for primitive in ATTENTION_PRIMITIVES:
        report = reports.get(primitive)
        if not isinstance(report, dict):
            return False
        if report.get("status") != "dry_run" or report.get("dry_run") is not True:
            return False
        if not _path_exists(report.get("report")):
            return False
    return True


def _attention_primitives_dry_run_observed(
    reports: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(reports, dict):
        return {}
    observed = {}
    for primitive, report in sorted(reports.items()):
        if not isinstance(report, dict):
            continue
        observed[str(primitive)] = {
            "status": report.get("status"),
            "dry_run": report.get("dry_run"),
            "report_exists": _path_exists(report.get("report")),
        }
    return observed


def _required_validate_direct_artifacts_exist(artifacts: Any) -> bool:
    required = [
        "semantic_json",
        "execution_plan",
        "plan_diff",
        "official_config_diff",
        "parameter_config",
        "program_dir",
        "tensorize_report",
        "decode_shell_report",
        "attention_layer_report",
        "single_layer_decode_report",
        "decode_step_smoke_report",
        "decode_step_profile_report",
        "search_report",
        "decode_step_autotune_report",
        "package_dir",
    ]
    if not isinstance(artifacts, dict):
        return False
    return all(_path_exists(artifacts.get(name)) for name in required)


def _validate_direct_artifact_observed(
    artifacts: Any,
) -> dict[str, bool]:
    if not isinstance(artifacts, dict):
        return {}
    return {
        str(name): _path_exists(path)
        for name, path in sorted(artifacts.items())
    }


def _config_gap_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    status = summary.get("status")
    if status not in {"match", "diff_found"}:
        return False
    if not _int_equal(summary.get("section_count"), len(PARITY_SECTIONS)):
        return False
    counts = summary.get("issue_counts_by_section")
    if not isinstance(counts, dict):
        return False
    if set(counts) != set(PARITY_SECTIONS):
        return False
    try:
        issue_count = int(summary.get("issue_count"))
        section_counts = {
            section: int(counts[section])
            for section in PARITY_SECTIONS
        }
    except (TypeError, ValueError):
        return False
    if issue_count < 0 or any(count < 0 for count in section_counts.values()):
        return False
    if sum(section_counts.values()) != issue_count:
        return False
    sections_with_issues = summary.get("sections_with_issues")
    if not isinstance(sections_with_issues, list):
        return False
    if set(sections_with_issues) - set(PARITY_SECTIONS):
        return False
    expected_sections = [
        section
        for section in PARITY_SECTIONS
        if section_counts[section] > 0
    ]
    if sections_with_issues != expected_sections:
        return False
    top_issue_paths = summary.get("top_issue_paths")
    if not isinstance(top_issue_paths, list):
        return False
    if issue_count == 0:
        return status == "match" and sections_with_issues == [] and top_issue_paths == []
    return status == "diff_found" and bool(top_issue_paths) and all(
        _config_gap_issue_complete(issue) for issue in top_issue_paths
    )


def _official_required_field_coverage_complete(coverage: Any) -> bool:
    if not isinstance(coverage, dict):
        return False
    try:
        required = int(coverage.get("required_field_count"))
        present = int(coverage.get("present_required_count"))
        missing = int(coverage.get("missing_required_count"))
    except (TypeError, ValueError):
        return False
    missing_paths = coverage.get("missing_required_paths")
    missing_sections = coverage.get("sections_missing_required_fields")
    return (
        coverage.get("status") == "complete"
        and required > 0
        and present == required
        and missing == 0
        and missing_paths == []
        and missing_sections == []
    )


def _official_required_field_coverage_observed(
    coverage: Any,
) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "status": coverage.get("status"),
        "required_field_count": coverage.get("required_field_count"),
        "present_required_count": coverage.get("present_required_count"),
        "missing_required_count": coverage.get("missing_required_count"),
        "missing_required_paths": coverage.get("missing_required_paths", []),
        "sections_missing_required_fields": coverage.get(
            "sections_missing_required_fields",
            [],
        ),
    }


def _config_gap_issue_complete(issue: Any) -> bool:
    if not isinstance(issue, dict):
        return False
    return (
        issue.get("kind") in {"missing", "mismatch", "extra"}
        and issue.get("section") in PARITY_SECTIONS
        and _non_empty_string(issue.get("path"))
    )


def _config_gap_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    counts = summary.get("issue_counts_by_section")
    top_issue_paths = summary.get("top_issue_paths")
    return {
        "status": summary.get("status"),
        "issue_count": summary.get("issue_count"),
        "section_count": summary.get("section_count"),
        "sections_with_issues": summary.get("sections_with_issues"),
        "issue_count_sections": _field_keys(counts),
        "top_issue_count": len(top_issue_paths)
        if isinstance(top_issue_paths, list)
        else None,
    }


def _lm_head_profile_complete(
    profile: Any,
    *,
    output_kind: Any = "token",
) -> bool:
    if not isinstance(profile, dict):
        return False
    expected_argmax_status = (
        "skipped" if output_kind == "logits" else "profiled"
    )
    return (
        _positive_number(profile.get("split_count"))
        and _nonnegative_number(profile.get("lm_head_ms"))
        and _nonnegative_number(profile.get("argmax_ms"))
        and profile.get("argmax_status") == expected_argmax_status
    )


def _lm_head_profile_observed(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "split_count": profile.get("split_count"),
        "lm_head_ms": profile.get("lm_head_ms"),
        "argmax_ms": profile.get("argmax_ms"),
        "argmax_status": profile.get("argmax_status"),
    }


def _lm_head_transform_complete(
    tensorization: Any,
    split_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_count = _safe_int(split_count)
    if expected_count is None or expected_count <= 0:
        return False
    counts = tensorization.get("transform_counts")
    if not isinstance(counts, dict):
        return False
    observed_count = _safe_int(counts.get(LINEAR_WEIGHT_TRANSFORM))
    if observed_count is None or observed_count < expected_count:
        return False
    expected_paths = {
        f"lm_head.splits.{shard_id}.weight"
        for shard_id in range(expected_count)
    }
    transformed_paths = _transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if transformed_paths and not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    split0 = key_tensors.get("lm_head.splits.0.weight")
    if not isinstance(split0, dict):
        return False
    shape = split0.get("shape")
    return (
        split0.get("transform") == LINEAR_WEIGHT_TRANSFORM
        and isinstance(shape, list)
        and len(shape) == 4
    )


def _embedding_norm_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    paths = _embedding_norm_weight_transform_paths(layer_count)
    embedding_paths = set(paths["embedding"])
    norm_paths = set(paths["norm"])
    if not embedding_paths or not norm_paths:
        return False
    embedding_transformed = _transformed_tensor_paths(
        tensorization,
        "reshape_embedding_weight_4d",
    )
    norm_transformed = _transformed_tensor_paths(
        tensorization,
        "reshape_norm_weight_4d",
    )
    if not embedding_paths.issubset(embedding_transformed):
        return False
    if not norm_paths.issubset(norm_transformed):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "embedding.weight",
        "layers.0.input_norm.weight",
        "layers.0.post_attention_norm.weight",
        "final_norm.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def _embedding_norm_weight_transform_paths(layer_count: Any) -> dict[str, list[str]]:
    norm_paths = ["final_norm.weight"]
    for layer_id in _expected_layer_ids(layer_count):
        norm_paths.extend(
            [
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    return {
        "embedding": ["embedding.weight"],
        "norm": norm_paths,
    }


def _embedding_norm_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = _embedding_norm_weight_transform_paths(layer_count)
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths["embedding"] + expected_paths["norm"]:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "embedding_transformed_paths": sorted(
            _transformed_tensor_paths(
                tensorization,
                "reshape_embedding_weight_4d",
            )
        ),
        "norm_transformed_paths": sorted(
            _transformed_tensor_paths(
                tensorization,
                "reshape_norm_weight_4d",
            )
        ),
        "key_tensors": key_observed,
    }


def _linear_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_paths = set(_linear_weight_transform_paths(layer_count))
    if not expected_paths:
        return False
    transformed_paths = _transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "layers.0.attention.wqkv_packed.weight",
        "layers.0.attention.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            if tensor.get("transform") != LINEAR_WEIGHT_TRANSFORM:
                return False
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def _linear_weight_transform_paths(layer_count: Any) -> list[str]:
    paths: list[str] = []
    for layer_id in _expected_layer_ids(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
            ]
        )
    return paths


def _decode_shell_linear_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
    split_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_paths = set(
        _decode_shell_linear_weight_transform_paths(
            layer_count,
            split_count,
        )
    )
    if not expected_paths:
        return False
    transformed_paths = _transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.splits.0.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            if tensor.get("transform") != LINEAR_WEIGHT_TRANSFORM:
                return False
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def _decode_shell_linear_weight_transform_paths(
    layer_count: Any,
    split_count: Any,
) -> list[str]:
    paths: list[str] = []
    for layer_id in _expected_layer_ids(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
            ]
        )
    split_count_int = _safe_int(split_count)
    if split_count_int is not None and split_count_int > 0:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(split_count_int)
        )
    return paths


def _transformed_tensor_paths(
    tensorization: dict[str, Any],
    transform: str,
) -> set[str]:
    paths_by_kind = tensorization.get("transform_paths_by_kind")
    if isinstance(paths_by_kind, dict):
        paths = paths_by_kind.get(transform, [])
        if isinstance(paths, list):
            return {str(path) for path in paths}
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return set()
    return {
        str(path)
        for path, tensor in key_tensors.items()
        if isinstance(tensor, dict) and tensor.get("transform") == transform
    }


def _decode_shell_linear_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
    split_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = _decode_shell_linear_weight_transform_paths(
        layer_count,
        split_count,
    )
    transformed_paths = sorted(
        _transformed_tensor_paths(tensorization, LINEAR_WEIGHT_TRANSFORM)
    )
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "transformed_paths": transformed_paths,
        "key_tensors": key_observed,
    }


def _linear_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = _linear_weight_transform_paths(layer_count)
    transformed_paths = sorted(
        _transformed_tensor_paths(tensorization, LINEAR_WEIGHT_TRANSFORM)
    )
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "transformed_paths": transformed_paths,
        "key_tensors": key_observed,
    }


def _lm_head_transform_observed(tensorization: Any) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    key_tensors = tensorization.get("key_tensors")
    split0 = {}
    if isinstance(key_tensors, dict):
        maybe_split0 = key_tensors.get("lm_head.splits.0.weight")
        if isinstance(maybe_split0, dict):
            split0 = {
                "transform": maybe_split0.get("transform"),
                "source_shape": maybe_split0.get("source_shape"),
                "shape": maybe_split0.get("shape"),
            }
    return {
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
        "lm_head.splits.0.weight": split0,
    }


def _tensorized_physical_shape_mismatches(
    tensorization: Any,
) -> list[dict[str, Any]]:
    if not isinstance(tensorization, dict):
        return []
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return []
    mismatches = []
    for path, tensor in sorted(key_tensors.items()):
        if not isinstance(tensor, dict):
            continue
        transform = tensor.get("transform")
        if transform not in _TENSORIZED_PHYSICAL_SHAPE_TRANSFORMS:
            continue
        expected = _expected_tensorized_physical_shape(tensor)
        observed = _int_list(tensor.get("shape"))
        source_shape = _int_list(tensor.get("source_shape"))
        if expected is None:
            mismatches.append(
                {
                    "path": path,
                    "transform": transform,
                    "source_shape": source_shape or None,
                    "observed": observed or None,
                    "expected": "known transform/source shape",
                }
            )
        elif observed != expected:
            mismatches.append(
                {
                    "path": path,
                    "transform": transform,
                    "source_shape": source_shape or None,
                    "observed": observed or None,
                    "expected": expected,
                }
            )
    return mismatches


_TENSORIZED_PHYSICAL_SHAPE_TRANSFORMS = {
    LINEAR_WEIGHT_TRANSFORM,
    "reshape_embedding_weight_4d",
    "reshape_norm_weight_4d",
}


def _expected_tensorized_physical_shape(
    tensor: dict[str, Any],
) -> list[int] | None:
    transform = tensor.get("transform")
    source_shape = _int_list(tensor.get("source_shape"))
    if transform == LINEAR_WEIGHT_TRANSFORM:
        if len(source_shape) != 2:
            return None
        return [1, 1, source_shape[1], source_shape[0]]
    if transform == "reshape_embedding_weight_4d":
        if len(source_shape) != 2:
            return None
        return [1, 1, source_shape[0], source_shape[1]]
    if transform == "reshape_norm_weight_4d":
        if len(source_shape) != 1:
            return None
        hidden = source_shape[0]
        if hidden <= 0:
            return None
        if hidden % 32 == 0:
            return [1, 1, hidden // 32, 32]
        return [1, 1, 1, hidden]
    return None


def _trace_profile_complete(
    trace: Any,
    throughput: Any,
    *,
    expected_iterations: Any,
) -> bool:
    if not isinstance(trace, dict) or not isinstance(throughput, dict):
        return False
    samples = trace.get("execute_samples_ms")
    return (
        trace.get("status") == "captured_and_executed"
        and _int_equal(trace.get("iterations"), expected_iterations)
        and _trace_samples_complete(samples, expected_iterations)
        and _nonnegative_number(trace.get("capture_latency_ms"))
        and _positive_number(trace.get("execute_latency_ms"))
        and _positive_number(throughput.get("trace_execute_mean_ms"))
        and _positive_number(
            throughput.get("trace_execute_tokens_per_second_per_user")
        )
        and _positive_number(
            throughput.get("trace_execute_aggregate_tokens_per_second")
        )
        and _int_equal(throughput.get("trace_iterations"), expected_iterations)
    )


def _trace_samples_complete(samples: Any, expected_iterations: Any) -> bool:
    if not isinstance(samples, list):
        return False
    try:
        expected_count = int(expected_iterations)
    except (TypeError, ValueError):
        return False
    return (
        len(samples) == expected_count
        and bool(samples)
        and all(_positive_number(sample) for sample in samples)
    )


def _trace_profile_observed(
    trace: Any,
    throughput: Any,
) -> dict[str, Any]:
    trace_dict = trace if isinstance(trace, dict) else {}
    throughput_dict = throughput if isinstance(throughput, dict) else {}
    return {
        "status": trace_dict.get("status"),
        "iterations": trace_dict.get("iterations"),
        "execute_sample_count": trace_dict.get("execute_sample_count"),
        "capture_latency_ms": trace_dict.get("capture_latency_ms"),
        "execute_latency_ms": trace_dict.get("execute_latency_ms"),
        "execute_samples_ms": trace_dict.get("execute_samples_ms"),
        "trace_execute_mean_ms": throughput_dict.get(
            "trace_execute_mean_ms"
        ),
        "trace_execute_tokens_per_second_per_user": throughput_dict.get(
            "trace_execute_tokens_per_second_per_user"
        ),
        "trace_execute_aggregate_tokens_per_second": throughput_dict.get(
            "trace_execute_aggregate_tokens_per_second"
        ),
        "trace_iterations": throughput_dict.get("trace_iterations"),
    }


def _autotune_knob_coverage_complete(
    coverage: Any,
    *,
    candidate_count: Any,
) -> bool:
    if not isinstance(coverage, dict):
        return False
    coverage_count = coverage.get("candidate_count")
    if not _positive_number(coverage_count):
        return False
    if candidate_count is not None and not _int_equal(
        coverage_count,
        candidate_count,
    ):
        return False
    if not _contains_all(
        coverage.get("knobs"),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    values = coverage.get("values")
    value_counts = coverage.get("value_counts")
    if not isinstance(values, dict) or not isinstance(value_counts, dict):
        return False
    for knob in DECODE_STEP_AUTOTUNE_KNOBS:
        knob_values = values.get(knob)
        if not isinstance(knob_values, list) or not knob_values:
            return False
        counts = value_counts.get(knob)
        if not isinstance(counts, dict) or not counts:
            return False
        try:
            count_total = sum(int(count) for count in counts.values())
        except (TypeError, ValueError):
            return False
        if not _int_equal(count_total, coverage_count):
            return False
    return True


def _autotune_knob_coverage_observed(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "knobs": coverage.get("knobs"),
        "candidate_count": coverage.get("candidate_count"),
        "values": coverage.get("values"),
        "varied_knobs": coverage.get("varied_knobs"),
        "missing_varied_knobs": _autotune_missing_varied_knobs(coverage),
        "all_knobs_varied": _autotune_default_knobs_varied(coverage),
    }


def _autotune_default_knobs_varied(coverage: Any) -> bool:
    return _autotune_missing_varied_knobs(coverage) == []


def _autotune_knob_variation_observed(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "varied_knobs": coverage.get("varied_knobs"),
        "missing_varied_knobs": _autotune_missing_varied_knobs(coverage),
        "all_knobs_varied": _autotune_default_knobs_varied(coverage),
    }


def _autotune_missing_varied_knobs(coverage: Any) -> list[str]:
    if not isinstance(coverage, dict):
        return list(DECODE_STEP_AUTOTUNE_KNOBS)
    missing = coverage.get("missing_varied_knobs")
    if isinstance(missing, list):
        return [str(knob) for knob in missing]
    varied = coverage.get("varied_knobs")
    if not isinstance(varied, list):
        return list(DECODE_STEP_AUTOTUNE_KNOBS)
    return [
        knob
        for knob in DECODE_STEP_AUTOTUNE_KNOBS
        if knob not in varied
    ]


def _autotune_output_kind_counts_complete(
    counts: Any,
    coverage: Any,
) -> bool:
    if not isinstance(counts, dict) or not isinstance(coverage, dict):
        return False
    expected_kinds = _autotune_expected_output_kinds(coverage)
    if not expected_kinds:
        return True
    for kind in expected_kinds:
        if not _positive_number(counts.get(kind)):
            return False
    return True


def _autotune_output_kind_counts_observed(
    counts: Any,
    coverage: Any,
) -> dict[str, Any]:
    return {
        "counts": counts if isinstance(counts, dict) else {},
        "expected_output_kinds": _autotune_expected_output_kinds(coverage),
    }


def _autotune_candidate_summaries(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    summaries = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        summaries.append(
            {
                "id": candidate.get("id"),
                "config": candidate.get("config"),
                "model": candidate.get("model"),
                "profile_metadata": candidate.get("profile_metadata", []),
                "profile_report": candidate.get("profile_report"),
                "knobs": candidate.get("knobs"),
                "status": candidate.get("status"),
                "passed": candidate.get("passed"),
                "metric": candidate.get("metric"),
                "output_kind": candidate.get("output_kind"),
                "output_shapes": candidate.get("output_shapes"),
                "lm_head_profile": candidate.get("lm_head_profile"),
                "throughput_summary": candidate.get("throughput_summary"),
                "bottleneck_summary": candidate.get("bottleneck_summary"),
                "parameter_source": candidate.get("parameter_source"),
                "trace_status": candidate.get("trace_status"),
                "reference_status": candidate.get("reference_status"),
                "reference_kind": candidate.get("reference_kind"),
                "reference_failed_checks": candidate.get(
                    "reference_failed_checks",
                    [],
                ),
                "error": candidate.get("error"),
            }
        )
    return summaries


def _autotune_candidates_complete(
    candidates: Any,
    *,
    candidate_count: Any,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
    out_dir: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(candidates, list) or not candidates:
        return False
    if not _int_equal(len(candidates), candidate_count):
        return False
    candidate_ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False
        candidate_id = candidate.get("id")
        if not _non_empty_string(candidate_id):
            return False
        candidate_ids.append(candidate_id)
        if not _autotune_candidate_complete(
            candidate,
            layer_count=layer_count,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            out_dir=out_dir,
            require_trace=require_trace,
        ):
            return False
    return len(candidate_ids) == len(set(candidate_ids))


def _autotune_candidate_complete(
    candidate: dict[str, Any],
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
    out_dir: Any,
    require_trace: bool,
) -> bool:
    output_kind = candidate.get("output_kind")
    if output_kind not in {"token", "logits"}:
        return False
    knobs = candidate.get("knobs")
    if not isinstance(knobs, dict) or not _contains_all(
        list(knobs),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    if not (
        candidate.get("status") == "profiled"
        and candidate.get("passed") is True
        and candidate.get("parameter_source") == "hf_model"
        and candidate.get("reference_status") == "passed"
        and candidate.get("reference_failed_checks") == []
        and candidate.get("error") is None
        and _nonnegative_number(candidate.get("metric"))
        and _path_exists_relative_to(candidate.get("config"), out_dir)
        and _path_exists_relative_to(candidate.get("model"), out_dir)
        and _path_exists_relative_to(candidate.get("profile_report"), out_dir)
        and _paths_exist_relative_to(candidate.get("profile_metadata"), out_dir)
        and _lm_head_profile_complete(
            candidate.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and _throughput_summary_complete(candidate.get("throughput_summary"))
        and _bottleneck_summary_complete(candidate.get("bottleneck_summary"))
        and _decode_output_shapes_complete(
            candidate.get("output_shapes"),
            layer_count=layer_count,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
        )
    ):
        return False
    if require_trace:
        return candidate.get("trace_status") == "captured_and_executed"
    return True


def _autotune_candidates_observed(
    candidates: Any,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    observed = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        observed.append(
            {
                "id": candidate.get("id"),
                "status": candidate.get("status"),
                "passed": candidate.get("passed"),
                "metric": candidate.get("metric"),
                "output_kind": candidate.get("output_kind"),
                "parameter_source": candidate.get("parameter_source"),
                "trace_status": candidate.get("trace_status"),
                "reference_status": candidate.get("reference_status"),
                "reference_failed_checks": candidate.get(
                    "reference_failed_checks"
                ),
                "error": candidate.get("error"),
                "knobs": candidate.get("knobs"),
                "output_shapes": _decode_output_shape_observed(
                    candidate.get("output_shapes")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    candidate.get("lm_head_profile")
                ),
                "throughput": _throughput_summary_observed(
                    candidate.get("throughput_summary")
                ),
                "bottleneck": _bottleneck_summary_observed(
                    candidate.get("bottleneck_summary")
                ),
            }
        )
    return observed


def _autotune_leaderboard_complete(
    leaderboard: Any,
    *,
    candidate_count: Any,
    candidate_summaries: Any,
    best: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(leaderboard, list) or not leaderboard:
        return False
    expected_count = _safe_int(candidate_count)
    if expected_count is None or expected_count <= 0:
        return False
    if len(leaderboard) != expected_count:
        return False
    expected_ids = _autotune_candidate_ids(candidate_summaries)
    if len(expected_ids) != expected_count:
        return False
    observed_ids = []
    for expected_rank, entry in enumerate(leaderboard, start=1):
        if not isinstance(entry, dict):
            return False
        if not _int_equal(entry.get("rank"), expected_rank):
            return False
        if not _autotune_leaderboard_entry_complete(
            entry,
            require_trace=require_trace,
        ):
            return False
        observed_ids.append(str(entry.get("candidate_id")))
    if sorted(observed_ids) != sorted(expected_ids):
        return False
    if _non_empty_string(best):
        first = leaderboard[0]
        if first.get("candidate_id") != best:
            return False
    return len(observed_ids) == len(set(observed_ids))


def _autotune_leaderboard_entry_complete(
    entry: dict[str, Any],
    *,
    require_trace: bool,
) -> bool:
    output_kind = entry.get("output_kind")
    if output_kind not in {"token", "logits"}:
        return False
    knobs = entry.get("knobs")
    if not isinstance(knobs, dict) or not _contains_all(
        list(knobs),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    if not (
        _non_empty_string(entry.get("candidate_id"))
        and entry.get("status") == "profiled"
        and entry.get("passed") is True
        and entry.get("parameter_source") == "hf_model"
        and entry.get("reference_status") == "passed"
        and entry.get("error") is None
        and _nonnegative_number(entry.get("metric_value"))
        and _non_empty_string(entry.get("profile_report"))
        and _throughput_summary_complete(entry.get("throughput_summary"))
        and _lm_head_profile_complete(
            entry.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and _bottleneck_summary_complete(entry.get("bottleneck_summary"))
    ):
        return False
    if require_trace:
        return entry.get("trace_status") == "captured_and_executed"
    return True


def _autotune_best_candidate_summary_complete(
    summary: Any,
    *,
    best: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(summary, dict) or not _non_empty_string(best):
        return False
    return (
        summary.get("candidate_id") == best
        and _int_equal(summary.get("rank"), 1)
        and _autotune_leaderboard_entry_complete(
            summary,
            require_trace=require_trace,
        )
        and _non_empty_string(summary.get("config"))
        and _non_empty_string(summary.get("model"))
        and isinstance(summary.get("profile_metadata"), list)
    )


def _autotune_candidate_ids(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        return []
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return []
        candidate_id = candidate.get("id") or candidate.get("candidate_id")
        if not _non_empty_string(candidate_id):
            return []
        ids.append(str(candidate_id))
    return ids


def _autotune_leaderboard_observed(
    leaderboard: Any,
) -> list[dict[str, Any]]:
    if not isinstance(leaderboard, list):
        return []
    observed = []
    for entry in leaderboard:
        if not isinstance(entry, dict):
            continue
        observed.append(
            {
                "rank": entry.get("rank"),
                "candidate_id": entry.get("candidate_id"),
                "status": entry.get("status"),
                "passed": entry.get("passed"),
                "metric": entry.get("metric"),
                "metric_value": entry.get("metric_value"),
                "parameter_source": entry.get("parameter_source"),
                "trace_status": entry.get("trace_status"),
                "reference_status": entry.get("reference_status"),
                "output_kind": entry.get("output_kind"),
                "throughput": _throughput_summary_observed(
                    entry.get("throughput_summary")
                ),
                "bottleneck": _bottleneck_summary_observed(
                    entry.get("bottleneck_summary")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    entry.get("lm_head_profile")
                ),
            }
        )
    return observed


def _autotune_best_candidate_summary_observed(
    summary: Any,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    observed = _autotune_leaderboard_observed([summary])
    if not observed:
        return {}
    result = observed[0]
    result["config"] = summary.get("config")
    result["model"] = summary.get("model")
    metadata = summary.get("profile_metadata")
    result["profile_metadata_count"] = (
        len(metadata) if isinstance(metadata, list) else None
    )
    return result


def _throughput_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    return (
        summary.get("status") == "measured"
        and _positive_number(summary.get("latency_ms"))
        and _positive_number(summary.get("tokens_per_second_per_user"))
        and _positive_number(summary.get("aggregate_tokens_per_second"))
    )


def _throughput_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "status": summary.get("status"),
        "latency_ms": summary.get("latency_ms"),
        "tokens_per_second_per_user": summary.get(
            "tokens_per_second_per_user"
        ),
        "aggregate_tokens_per_second": summary.get(
            "aggregate_tokens_per_second"
        ),
        "trace_execute_tokens_per_second_per_user": summary.get(
            "trace_execute_tokens_per_second_per_user"
        ),
    }


def _autotune_expected_output_kinds(coverage: Any) -> list[str]:
    if not isinstance(coverage, dict):
        return []
    values = coverage.get("values")
    if not isinstance(values, dict):
        return []
    generation_templates = values.get("generation_template")
    if not isinstance(generation_templates, list):
        return []
    kinds = []
    for template in generation_templates:
        kind = "token" if template == "device_argmax_greedy" else "logits"
        if kind not in kinds:
            kinds.append(kind)
    return kinds


def _layer_profile_ids(layer_profiles: Any) -> list[int]:
    if not isinstance(layer_profiles, list):
        return []
    layer_ids = []
    for profile in layer_profiles:
        if not isinstance(profile, dict):
            return []
        try:
            layer_ids.append(int(profile["layer_id"]))
        except (KeyError, TypeError, ValueError):
            return []
    return layer_ids


def _layer_profiles_have_nonnegative_fields(
    layer_profiles: Any,
    fields: tuple[str, ...],
) -> bool:
    if not isinstance(layer_profiles, list) or not layer_profiles:
        return False
    return all(
        isinstance(profile, dict)
        and _has_nonnegative_fields(profile, fields)
        for profile in layer_profiles
    )


def _layer_profile_field_keys(layer_profiles: Any) -> list[list[str]]:
    if not isinstance(layer_profiles, list):
        return []
    return [
        _field_keys(profile)
        for profile in layer_profiles
        if isinstance(profile, dict)
    ]


def _bottleneck_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    sections = summary.get("sections_ms")
    max_section = summary.get("max_section")
    return (
        _non_empty_string(max_section)
        and isinstance(sections, dict)
        and str(max_section) in sections
        and _nonnegative_number(summary.get("max_section_ms"))
        and _has_nonnegative_fields(
            sections,
            PROFILE_BOTTLENECK_SECTION_KEYS,
        )
    )


def _bottleneck_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    sections = summary.get("sections_ms")
    return {
        "max_section": summary.get("max_section"),
        "max_section_ms": summary.get("max_section_ms"),
        "sections_ms": _field_keys(sections),
    }


def _observed_ops_cover_planned(planned_ops: Any, observed_ops: Any) -> bool:
    if not isinstance(planned_ops, list) or not isinstance(observed_ops, list):
        return False
    planned_index = 0
    for observed in observed_ops:
        if (
            planned_index < len(planned_ops)
            and str(observed) == str(planned_ops[planned_index])
        ):
            planned_index += 1
    return planned_index == len(planned_ops)


def _expected_layer_ids(layers: Any) -> list[int]:
    try:
        layer_count = int(layers)
    except (TypeError, ValueError):
        return []
    if layer_count <= 0:
        return []
    return list(range(layer_count))


def _step_tensorization_summary(step: dict[str, Any]) -> dict[str, Any]:
    setup = step.get("parameter_setup") or {}
    tensorization = setup.get("tensorization") or {}
    return tensorization if isinstance(tensorization, dict) else {}


def _step_synthetic_runtime_input_count(step: dict[str, Any]) -> Any:
    setup = step.get("parameter_setup") or {}
    if not isinstance(setup, dict):
        return None
    return setup.get("synthetic_runtime_input_tensor_count")


def _step_synthetic_rotary_tensor_count(step: dict[str, Any]) -> Any:
    setup = step.get("parameter_setup") or {}
    if not isinstance(setup, dict):
        return None
    return setup.get("synthetic_rotary_tensor_count")


def _decode_runtime_inputs_complete(
    step: Any,
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
) -> bool:
    if not isinstance(step, dict):
        return False
    expected = _expected_decode_runtime_input_summary(
        layer_count=layer_count,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    input_shapes = step.get("input_shapes")
    kv_cache = step.get("kv_cache")
    if not isinstance(input_shapes, dict) or not isinstance(kv_cache, dict):
        return False
    input_source = step.get("input_source")
    expected_synthetic_runtime_count = expected[
        "synthetic_runtime_input_tensor_count"
    ]
    expected_prompt_runtime_count = 0
    expected_decode_runtime_state_count = 0
    expected_rotary_runtime_count = 0
    expected_kv_cache_runtime_count = 0
    expected_synthetic_rotary_count = expected[
        "synthetic_rotary_tensor_count"
    ]
    if input_source == "prompt_runtime":
        expected_synthetic_runtime_count = 0
        expected_prompt_runtime_count = 1
        expected_decode_runtime_state_count = 2
        expected_rotary_runtime_count = _expected_prompt_rotary_runtime_count(
            step,
            fallback=expected_synthetic_rotary_count,
        )
        expected_kv_cache_runtime_count = expected[
            "kv_cache_runtime_input_tensor_count"
        ]
        expected_synthetic_rotary_count = 0
    return (
        _int_list(input_shapes.get("token_ids"))
        == expected["token_ids"]
        and _int_list(input_shapes.get("page_table"))
        == expected["page_table"]
        and _int_list(input_shapes.get("cache_position"))
        == expected["cache_position"]
        and _int_list(input_shapes.get("key_cache"))
        == expected["kv_cache_shape"]
        and _int_list(input_shapes.get("value_cache"))
        == expected["kv_cache_shape"]
        and _int_list(kv_cache.get("physical_shape"))
        == expected["kv_cache_shape"]
        and _int_list(kv_cache.get("logical_shape"))
        == expected["kv_cache_logical_shape"]
        and _int_equal(kv_cache.get("page_block_size"), page_block_size)
        and _int_equal(kv_cache.get("page_count"), expected["page_count"])
        and _int_equal(
            kv_cache.get("max_num_blocks"),
            expected["max_num_blocks"],
        )
        and _runtime_input_source_supported(step)
        and _int_equal(
            step.get("synthetic_runtime_input_tensor_count"),
            expected_synthetic_runtime_count,
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("prompt_runtime_input_tensor_count"),
                expected_prompt_runtime_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("decode_runtime_state_input_tensor_count"),
                expected_decode_runtime_state_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("rotary_runtime_input_tensor_count"),
                expected_rotary_runtime_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("kv_cache_runtime_input_tensor_count"),
                expected_kv_cache_runtime_count,
            )
        )
        and _int_equal(
            step.get("synthetic_rotary_tensor_count"),
            expected_synthetic_rotary_count,
        )
    )


def _expected_prompt_rotary_runtime_count(
    step: dict[str, Any],
    *,
    fallback: Any,
) -> Any:
    rotary_state = step.get("rotary_runtime_state")
    if not isinstance(rotary_state, dict):
        return fallback
    tensor_count = _safe_int(rotary_state.get("tensor_count"))
    if rotary_state.get("shared_across_layers") is True:
        return tensor_count if tensor_count is not None else 3
    return tensor_count if tensor_count is not None else fallback


def _runtime_input_source_supported(step: Any) -> bool:
    return (
        isinstance(step, dict)
        and step.get("input_source") in {"synthetic", "prompt_runtime"}
    )


def _synthetic_runtime_inputs_accepted(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("input_source") == "prompt_runtime":
        return _nonnegative_number(
            step.get("synthetic_runtime_input_tensor_count")
        )
    return _positive_number(step.get("synthetic_runtime_input_tensor_count"))


def _decode_shell_runtime_inputs_accepted(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    source = step.get("input_source")
    if source == "prompt_runtime":
        return (
            _int_equal(step.get("synthetic_runtime_input_tensor_count"), 0)
            and _int_equal(step.get("runtime_input_tensor_count"), 0)
            and _int_equal(step.get("prompt_runtime_input_tensor_count"), 1)
        )
    if source == "synthetic":
        return _positive_number(step.get("runtime_input_tensor_count"))
    return False


def _expected_decode_runtime_input_summary(
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    seq = _safe_int(seq_len)
    cache = _safe_int(cache_len)
    layers = _safe_int(layer_count)
    block = _safe_int(page_block_size) or 32
    page_count = None
    max_num_blocks = None
    if batch is not None and cache is not None and block > 0:
        page_count = max(1, (cache + block - 1) // block)
        max_num_blocks = batch * page_count
    kv_shape = _paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    logical_shape = (
        [batch, cache, _safe_int(num_kv_heads), _safe_int(head_dim)]
        if None
        not in (
            batch,
            cache,
            _safe_int(num_kv_heads),
            _safe_int(head_dim),
        )
        else []
    )
    return {
        "token_ids": [batch, seq]
        if None not in (batch, seq)
        else [],
        "page_table": [batch, page_count]
        if None not in (batch, page_count)
        else [],
        "cache_position": [batch] if batch is not None else [],
        "kv_cache_shape": kv_shape,
        "kv_cache_logical_shape": logical_shape,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "synthetic_runtime_input_tensor_count": (
            3 + 2 * layers if layers is not None else None
        ),
        "kv_cache_runtime_input_tensor_count": (
            2 * layers if layers is not None else None
        ),
        "synthetic_rotary_tensor_count": (
            3 * layers if layers is not None else None
        ),
    }


def _decode_runtime_input_observed(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    input_shapes = step.get("input_shapes")
    kv_cache = step.get("kv_cache")
    return {
        "input_source": step.get("input_source"),
        "synthetic_runtime_input_tensor_count": step.get(
            "synthetic_runtime_input_tensor_count"
        ),
        "prompt_runtime_input_tensor_count": step.get(
            "prompt_runtime_input_tensor_count"
        ),
        "prompt_tokenization": step.get("prompt_tokenization"),
        "decode_runtime_state_input_tensor_count": step.get(
            "decode_runtime_state_input_tensor_count"
        ),
        "decode_runtime_state": step.get("decode_runtime_state"),
        "rotary_runtime_input_tensor_count": step.get(
            "rotary_runtime_input_tensor_count"
        ),
        "rotary_runtime_state": step.get("rotary_runtime_state"),
        "kv_cache_runtime_input_tensor_count": step.get(
            "kv_cache_runtime_input_tensor_count"
        ),
        "kv_cache_runtime_state": step.get("kv_cache_runtime_state"),
        "synthetic_rotary_tensor_count": step.get(
            "synthetic_rotary_tensor_count"
        ),
        "input_shapes": {
            "token_ids": _int_list((input_shapes or {}).get("token_ids")),
            "page_table": _int_list((input_shapes or {}).get("page_table")),
            "cache_position": _int_list(
                (input_shapes or {}).get("cache_position")
            ),
            "key_cache": _int_list((input_shapes or {}).get("key_cache")),
            "value_cache": _int_list(
                (input_shapes or {}).get("value_cache")
            ),
        }
        if isinstance(input_shapes, dict)
        else {},
        "kv_cache": {
            "page_block_size": (kv_cache or {}).get("page_block_size"),
            "page_count": (kv_cache or {}).get("page_count"),
            "max_num_blocks": (kv_cache or {}).get("max_num_blocks"),
            "physical_shape": _int_list(
                (kv_cache or {}).get("physical_shape")
            ),
            "logical_shape": _int_list((kv_cache or {}).get("logical_shape")),
        }
        if isinstance(kv_cache, dict)
        else {},
    }


def _decode_output_shapes_complete(
    output_shapes: Any,
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> bool:
    expected = _expected_decode_output_shape_summary(
        layer_count=layer_count,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=cache_len,
        vocab_size=vocab_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        output_kind=output_kind,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    output_kind = expected["output_kind"]
    output_shape = _int_list(output_shapes.get(output_kind))
    if output_kind == "token":
        if output_shape not in expected["accepted_output_shapes"]:
            return False
    elif output_shape != expected["output_shape"]:
        return False
    if _int_list(output_shapes.get("key_cache")) != expected["kv_cache_shape"]:
        return False
    if _int_list(output_shapes.get("value_cache")) != expected["kv_cache_shape"]:
        return False
    layers = output_shapes.get("kv_cache_layers")
    if not isinstance(layers, list):
        return False
    if len(layers) != len(expected["kv_cache_layer_ids"]):
        return False
    observed_layer_ids = []
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        try:
            layer_id = int(layer["layer_id"])
        except (KeyError, TypeError, ValueError):
            return False
        observed_layer_ids.append(layer_id)
        if _int_list(layer.get("key_cache")) != expected["kv_cache_shape"]:
            return False
        if _int_list(layer.get("value_cache")) != expected["kv_cache_shape"]:
            return False
    return observed_layer_ids == expected["kv_cache_layer_ids"]


def _attention_layer_output_shapes_complete(
    output_shapes: Any,
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> bool:
    expected = _expected_attention_layer_output_shape_summary(
        batch_size=batch_size,
        cache_len=cache_len,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    return (
        _int_list(output_shapes.get("attention_output"))
        == expected["attention_output"]
        and _int_list(output_shapes.get("key_cache"))
        == expected["kv_cache_shape"]
        and _int_list(output_shapes.get("value_cache"))
        == expected["kv_cache_shape"]
    )


def _expected_attention_layer_output_shape_summary(
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    hidden = _safe_int(hidden_size)
    attention_output = (
        [1, 1, batch, hidden] if None not in (batch, hidden) else []
    )
    kv_shape = _paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    return {
        "attention_output": attention_output,
        "kv_cache_shape": kv_shape,
    }


def _attention_layer_output_shape_observed(
    output_shapes: Any,
) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    return {
        "attention_output": _int_list(
            output_shapes.get("attention_output")
        ),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
    }


def _paged_kv_cache_shape(
    *,
    batch_size: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> list[int]:
    batch = _safe_int(batch_size)
    cache = _safe_int(cache_len)
    kv_heads = _safe_int(num_kv_heads)
    dim = _safe_int(head_dim)
    block = _safe_int(page_block_size) or 32
    if None in (batch, cache, kv_heads, dim) or cache < 0 or block <= 0:
        return []
    pages_per_user = max(1, (cache + block - 1) // block)
    return [batch * pages_per_user, kv_heads, block, dim]


def _attention_primitive_reports_complete(
    reports: Any,
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_heads: Any,
    num_kv_heads: Any,
    head_dim: Any,
) -> bool:
    if not isinstance(reports, list):
        return False
    observed_sequence = [
        report.get("primitive")
        for report in reports
        if isinstance(report, dict)
    ]
    if observed_sequence != list(ATTENTION_PRIMITIVES):
        return False
    if len(reports) != len(ATTENTION_PRIMITIVES):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        if not _non_empty_string(report.get("report")):
            return False
        if report.get("status") != "passed":
            return False
        if report.get("error") is not None:
            return False
        if not _nonnegative_number(report.get("latency_ms")):
            return False
        if not _non_empty_string(report.get("dtype")):
            return False
        if not _non_empty_string(report.get("layout")):
            return False
        if "memory_config" not in report:
            return False
        if not _int_equal(report.get("batch_size"), batch_size):
            return False
        if not _int_equal(report.get("hidden_size"), hidden_size):
            return False
        if not _int_equal(report.get("num_heads"), num_heads):
            return False
        if not _int_equal(report.get("num_kv_heads"), num_kv_heads):
            return False
        if not _int_equal(report.get("head_dim"), head_dim):
            return False
        if not _int_equal(report.get("max_cache_len"), cache_len):
            return False
        if not _shape_dict_has_int_lists(report.get("input_shapes")):
            return False
        expected_shapes = report.get("expected_output_shapes")
        output_shapes = report.get("output_shapes")
        if not _shape_dict_has_int_lists(expected_shapes):
            return False
        if not isinstance(output_shapes, dict):
            return False
        for name, expected_shape in expected_shapes.items():
            if _int_list(output_shapes.get(name)) != _int_list(
                expected_shape
            ):
                return False
        reference = report.get("reference") or {}
        if reference.get("status") != "passed":
            return False
        if not _observed_ops_cover_planned(
            reference.get("planned_ops"),
            reference.get("observed_ops"),
        ):
            return False
        environment = report.get("ttnn_environment") or {}
        if not isinstance(environment, dict):
            return False
        if environment.get("module_available") is not True:
            return False
        if not _ttnn_runtime_identity_available(environment):
            return False
        if not _non_empty_string(environment.get("tt_metal_git_commit")):
            return False
    return True


def _attention_primitive_reports_observed(
    reports: Any,
) -> list[dict[str, Any]]:
    if not isinstance(reports, list):
        return []
    observed = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        input_shapes = report.get("input_shapes")
        output_shapes = report.get("output_shapes")
        expected_shapes = report.get("expected_output_shapes")
        reference = report.get("reference") or {}
        environment = report.get("ttnn_environment") or {}
        observed.append(
            {
                "primitive": report.get("primitive"),
                "status": report.get("status"),
                "latency_ms": report.get("latency_ms"),
                "error": report.get("error"),
                "input_shape_keys": _field_keys(input_shapes),
                "expected_output_shape_keys": _field_keys(expected_shapes),
                "output_shape_keys": _field_keys(output_shapes),
                "dtype": report.get("dtype"),
                "layout": report.get("layout"),
                "memory_config": report.get("memory_config"),
                "reference_status": reference.get("status"),
                "planned_ops": reference.get("planned_ops"),
                "observed_ops": reference.get("observed_ops"),
                "ttnn_version": environment.get("version")
                if isinstance(environment, dict)
                else None,
                "ttnn_module_file": environment.get("module_file")
                if isinstance(environment, dict)
                else None,
                "tt_metal_git_commit": environment.get(
                    "tt_metal_git_commit"
                )
                if isinstance(environment, dict)
                else None,
            }
        )
    return observed


def _attention_layer_primitive_reports_complete(reports: Any) -> bool:
    if not isinstance(reports, list):
        return False
    observed_sequence = [
        report.get("primitive")
        for report in reports
        if isinstance(report, dict)
    ]
    if observed_sequence != list(ATTENTION_LAYER_OPS):
        return False
    if len(reports) != len(ATTENTION_LAYER_OPS):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        if report.get("status") != "passed":
            return False
        if report.get("error") is not None:
            return False
        if not _nonnegative_number(report.get("latency_ms")):
            return False
        if not _non_empty_string(report.get("dtype")):
            return False
        if not _non_empty_string(report.get("layout")):
            return False
        if "memory_config" not in report:
            return False
        if not _shape_dict_has_int_lists(report.get("input_shapes")):
            return False
        expected_shapes = report.get("expected_output_shapes")
        output_shapes = report.get("output_shapes")
        if not _shape_dict_has_int_lists(expected_shapes):
            return False
        if not isinstance(output_shapes, dict):
            return False
        for name, expected_shape in expected_shapes.items():
            if _int_list(output_shapes.get(name)) != _int_list(
                expected_shape
            ):
                return False
    return True


def _attention_layer_primitive_reports_observed(
    reports: Any,
) -> list[dict[str, Any]]:
    if not isinstance(reports, list):
        return []
    observed = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        input_shapes = report.get("input_shapes")
        output_shapes = report.get("output_shapes")
        expected_shapes = report.get("expected_output_shapes")
        observed.append(
            {
                "primitive": report.get("primitive"),
                "status": report.get("status"),
                "latency_ms": report.get("latency_ms"),
                "error": report.get("error"),
                "input_shape_keys": _field_keys(input_shapes),
                "expected_output_shape_keys": _field_keys(expected_shapes),
                "output_shape_keys": _field_keys(output_shapes),
                "dtype": report.get("dtype"),
                "layout": report.get("layout"),
                "memory_config": report.get("memory_config"),
            }
        )
    return observed


def _shape_dict_has_int_lists(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(_int_list(shape) for shape in value.values())


def _expected_decode_output_shape_summary(
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    seq = _safe_int(seq_len)
    vocab = _safe_int(vocab_size)
    layers = _safe_int(layer_count)
    token_shape = [batch, seq] if batch is not None and seq is not None else []
    token_vector = [batch] if batch is not None else []
    normalized_output_kind = (
        "logits" if output_kind == "logits" else "token"
    )
    logits_shape = (
        [batch, seq, vocab]
        if None not in (batch, seq, vocab)
        else []
    )
    output_shape = logits_shape if normalized_output_kind == "logits" else token_shape
    kv_shape = _paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    layer_ids = list(range(layers)) if layers is not None and layers > 0 else []
    return {
        "output_kind": normalized_output_kind,
        "output_shape": output_shape,
        "accepted_output_shapes": [
            shape for shape in (token_shape, token_vector) if shape
        ]
        if normalized_output_kind == "token"
        else [output_shape],
        "kv_cache_shape": kv_shape,
        "kv_cache_layer_ids": layer_ids,
    }


def _decode_output_shape_observed(output_shapes: Any) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    layers = output_shapes.get("kv_cache_layers")
    return {
        "output_kind": "logits" if "logits" in output_shapes else "token",
        "token": _int_list(output_shapes.get("token")),
        "logits": _int_list(output_shapes.get("logits")),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
        "kv_cache_layer_ids": [
            _safe_int(layer.get("layer_id"))
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
        "kv_cache_layer_shapes": [
            {
                "layer_id": _safe_int(layer.get("layer_id")),
                "key_cache": _int_list(layer.get("key_cache")),
                "value_cache": _int_list(layer.get("value_cache")),
            }
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
    }


def _decode_depth_sweep_records_complete(
    records: Any,
    *,
    expected_depths: Any,
    batch_size: Any,
    cache_len: Any,
    seq_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any,
    page_block_size: Any,
    require_trace: bool,
    trace_iterations: Any,
) -> bool:
    if not isinstance(records, list) or not records:
        return False
    depths = _int_list(expected_depths)
    if len(records) != len(depths):
        return False
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return False
        depth = depths[index]
        if not _decode_depth_sweep_record_complete(
            record,
            depth=depth,
            batch_size=batch_size,
            cache_len=cache_len,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
            require_trace=require_trace,
            trace_iterations=trace_iterations,
        ):
            return False
    return True


def _decode_depth_sweep_record_complete(
    record: dict[str, Any],
    *,
    depth: int,
    batch_size: Any,
    cache_len: Any,
    seq_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any,
    page_block_size: Any,
    require_trace: bool,
    trace_iterations: Any,
) -> bool:
    throughput = record.get("throughput_summary")
    if not isinstance(throughput, dict):
        return False
    if not (
        _int_equal(record.get("depth"), depth)
        and record.get("status") == "profiled"
        and record.get("passed") is True
        and _non_empty_string(record.get("profile_report"))
        and _int_equal(record.get("layers"), depth)
        and _int_equal(record.get("batch_size"), batch_size)
        and _int_equal(record.get("cache_len"), cache_len)
        and record.get("parameter_source") == "hf_model"
        and _runtime_input_source_supported(record)
        and _positive_number(record.get("latency_ms"))
        and _positive_number(record.get("tensor_conversion_count"))
        and _nonnegative_number(record.get("tensor_conversion_ms"))
        and _int_equal(record.get("layer_profile_count"), depth)
        and record.get("layer_profile_ids") == list(range(depth))
        and _has_nonnegative_fields(
            record.get("section_latency_ms"),
            PROFILE_SECTION_LATENCY_KEYS,
        )
        and _layer_profile_ids(record.get("layer_profiles"))
        == list(range(depth))
        and _layer_profiles_have_nonnegative_fields(
            record.get("layer_profiles"),
            PROFILE_LAYER_LATENCY_KEYS,
        )
        and _lm_head_profile_complete(
            record.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and record.get("reference_status") == "passed"
        and record.get("reference_failed_checks") == []
        and throughput.get("status") == "measured"
        and _positive_number(record.get("tokens_per_second_per_user"))
        and _positive_number(record.get("aggregate_tokens_per_second"))
        and _positive_number(throughput.get("tokens_per_second_per_user"))
        and _positive_number(throughput.get("aggregate_tokens_per_second"))
        and _decode_runtime_inputs_complete(
            record,
            layer_count=depth,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
        )
        and _bottleneck_summary_complete(record.get("bottleneck_summary"))
        and _decode_output_shapes_complete(
            record.get("output_shapes"),
            layer_count=depth,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
        )
    ):
        return False
    if require_trace:
        return (
            record.get("trace_status") == "captured_and_executed"
            and _int_equal(record.get("trace_iterations"), trace_iterations)
        )
    return True


def _decode_depth_sweep_records_observed(
    records: Any,
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    observed = []
    for record in records:
        if not isinstance(record, dict):
            continue
        throughput = record.get("throughput_summary")
        bottleneck = record.get("bottleneck_summary")
        observed.append(
            {
                "depth": record.get("depth"),
                "status": record.get("status"),
                "passed": record.get("passed"),
                "layers": record.get("layers"),
                "batch_size": record.get("batch_size"),
                "cache_len": record.get("cache_len"),
                "parameter_source": record.get("parameter_source"),
                "input_source": record.get("input_source"),
                "latency_ms": record.get("latency_ms"),
                "tensor_conversion_count": record.get(
                    "tensor_conversion_count"
                ),
                "layer_profile_count": record.get("layer_profile_count"),
                "layer_profile_ids": record.get("layer_profile_ids"),
                "section_latency_ms": _field_keys(
                    record.get("section_latency_ms")
                ),
                "layer_profile_fields": _layer_profile_field_keys(
                    record.get("layer_profiles")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    record.get("lm_head_profile")
                ),
                "reference_status": record.get("reference_status"),
                "reference_failed_checks": record.get(
                    "reference_failed_checks"
                ),
                "throughput_status": throughput.get("status")
                if isinstance(throughput, dict)
                else None,
                "tokens_per_second_per_user": record.get(
                    "tokens_per_second_per_user"
                ),
                "aggregate_tokens_per_second": record.get(
                    "aggregate_tokens_per_second"
                ),
                "trace_status": record.get("trace_status"),
                "trace_iterations": record.get("trace_iterations"),
                "output_shapes": _decode_output_shape_observed(
                    record.get("output_shapes")
                ),
                "runtime_inputs": _decode_runtime_input_observed(record),
                "bottleneck": _bottleneck_summary_observed(bottleneck),
            }
        )
    return observed


def _tensorized_tensor_paths(tensorization: dict[str, Any]) -> list[str]:
    paths = tensorization.get("tensor_paths")
    if isinstance(paths, list):
        return sorted({str(path) for path in paths if path is not None})

    paths = tensorization.get("key_paths")
    if isinstance(paths, list):
        return sorted({str(path) for path in paths if path is not None})

    key_tensors = tensorization.get("key_tensors")
    if isinstance(key_tensors, dict):
        return sorted(str(path) for path in key_tensors)

    return []


def _step_ttnn_environment(step: dict[str, Any]) -> dict[str, Any]:
    environment = step.get("ttnn_environment") or {}
    return environment if isinstance(environment, dict) else {}


def _ttnn_runtime_identity_available(environment: Any) -> bool:
    if not isinstance(environment, dict):
        return False
    return _non_empty_string(environment.get("version")) or _non_empty_string(
        environment.get("module_file")
    )


def _ttnn_runtime_identity_observed(environment: Any) -> dict[str, Any]:
    if not isinstance(environment, dict):
        return {}
    return {
        "version": environment.get("version"),
        "module_file": environment.get("module_file"),
    }


def _step_trace_summary(step: dict[str, Any]) -> dict[str, Any]:
    trace = step.get("trace") or {}
    return trace if isinstance(trace, dict) else {}


def _require(value: Any, step: str) -> None:
    if value is None:
        raise RuntimeError(f"validate-direct requires {step} to pass first")


def _mark_remaining_skipped(
    report: dict[str, Any],
    failed_step: str,
    steps: tuple[str, ...] = VALIDATION_STEPS,
) -> None:
    seen_failed = False
    for step in steps:
        if step == failed_step:
            seen_failed = True
            continue
        if not seen_failed:
            continue
        if report["results"].get(step) == "pending":
            report["results"][step] = "skipped"
            report["steps"][step] = {
                "status": "skipped",
                "reason": f"blocked by failed step: {failed_step}",
            }
