from __future__ import annotations

import gc
import json
import importlib
import py_compile
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
from .reports.attention import (
    attention_layer_primitive_reports_complete as _attention_layer_primitive_reports_complete,
    attention_layer_primitive_reports_observed as _attention_layer_primitive_reports_observed,
    attention_primitive_reports_complete as _attention_primitive_reports_complete,
    attention_primitive_reports_observed as _attention_primitive_reports_observed,
    attention_primitives_dry_run_complete as _attention_primitives_dry_run_complete,
    attention_primitives_dry_run_observed as _attention_primitives_dry_run_observed,
)
from .reports.autotune import (
    autotune_best_candidate_summary_complete as _autotune_best_candidate_summary_complete,
    autotune_best_candidate_summary_observed as _autotune_best_candidate_summary_observed,
    autotune_candidate_complete as _autotune_candidate_complete,
    autotune_candidate_ids as _autotune_candidate_ids,
    autotune_candidate_summaries as _autotune_candidate_summaries,
    autotune_candidates_complete as _autotune_candidates_complete,
    autotune_candidates_observed as _autotune_candidates_observed,
    autotune_default_knobs_varied as _autotune_default_knobs_varied,
    autotune_expected_output_kinds as _autotune_expected_output_kinds,
    autotune_knob_coverage_complete as _autotune_knob_coverage_complete,
    autotune_knob_coverage_observed as _autotune_knob_coverage_observed,
    autotune_knob_variation_observed as _autotune_knob_variation_observed,
    autotune_leaderboard_complete as _autotune_leaderboard_complete,
    autotune_leaderboard_entry_complete as _autotune_leaderboard_entry_complete,
    autotune_leaderboard_observed as _autotune_leaderboard_observed,
    autotune_missing_varied_knobs as _autotune_missing_varied_knobs,
    autotune_output_kind_counts_complete as _autotune_output_kind_counts_complete,
    autotune_output_kind_counts_observed as _autotune_output_kind_counts_observed,
)
from .reports.depth import (
    PROFILE_LAYER_LATENCY_KEYS,
    PROFILE_SECTION_LATENCY_KEYS,
    decode_depth_sweep_record_complete as _decode_depth_sweep_record_complete,
    decode_depth_sweep_records_complete as _decode_depth_sweep_records_complete,
    decode_depth_sweep_records_observed as _decode_depth_sweep_records_observed,
)
from .reports.evidence import (
    artifact_evidence as _artifact_evidence,
    artifact_index as _artifact_index,
    candidate_reference_status_counts as _candidate_reference_status_counts,
    dump_validation_report,
    real_decode_cli_args as _real_decode_cli_args,
    real_decode_evidence_manifest as _real_decode_evidence_manifest,
    real_decode_reproducibility as _real_decode_reproducibility,
    reference_summary as _reference_summary,
    shell_command as _shell_command,
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
from .reports.profiling import (
    PROFILE_BOTTLENECK_SECTION_KEYS,
    bottleneck_summary_complete as _bottleneck_summary_complete,
    bottleneck_summary_observed as _bottleneck_summary_observed,
    layer_profile_field_keys as _layer_profile_field_keys,
    layer_profile_ids as _layer_profile_ids,
    layer_profiles_have_nonnegative_fields as _layer_profiles_have_nonnegative_fields,
    lm_head_profile_complete as _lm_head_profile_complete,
    lm_head_profile_observed as _lm_head_profile_observed,
    step_trace_summary as _step_trace_summary,
    throughput_summary_complete as _throughput_summary_complete,
    throughput_summary_observed as _throughput_summary_observed,
    trace_profile_complete as _trace_profile_complete,
    trace_profile_observed as _trace_profile_observed,
    trace_samples_complete as _trace_samples_complete,
)
from .reports.runtime import (
    attention_layer_output_shape_observed as _attention_layer_output_shape_observed,
    attention_layer_output_shapes_complete as _attention_layer_output_shapes_complete,
    decode_runtime_input_observed as _decode_runtime_input_observed,
    decode_runtime_inputs_complete as _decode_runtime_inputs_complete,
    decode_shell_numeric_reference_complete as _decode_shell_numeric_reference_complete,
    decode_shell_numeric_reference_observed as _decode_shell_numeric_reference_observed,
    decode_shell_runtime_inputs_accepted as _decode_shell_runtime_inputs_accepted,
    decode_output_shape_observed as _decode_output_shape_observed,
    decode_output_shapes_complete as _decode_output_shapes_complete,
    expected_attention_layer_output_shape_summary as _expected_attention_layer_output_shape_summary,
    expected_decode_runtime_input_summary as _expected_decode_runtime_input_summary,
    expected_decode_output_shape_summary as _expected_decode_output_shape_summary,
    expected_prompt_rotary_runtime_count as _expected_prompt_rotary_runtime_count,
    observed_ops_cover_planned as _observed_ops_cover_planned,
    paged_kv_cache_shape as _paged_kv_cache_shape,
    runtime_input_source_supported as _runtime_input_source_supported,
    shape_dict_has_int_lists as _shape_dict_has_int_lists,
    step_ttnn_environment as _step_ttnn_environment,
    synthetic_runtime_inputs_accepted as _synthetic_runtime_inputs_accepted,
    ttnn_runtime_identity_available as _ttnn_runtime_identity_available,
    ttnn_runtime_identity_observed as _ttnn_runtime_identity_observed,
)
from .reports.schema import (
    acceptance_check as _acceptance_check,
    contains_all as _contains_all,
    decode_step_contract as _decode_step_contract,
    expected_layer_ids as _expected_layer_ids,
    field_keys as _field_keys,
    has_nonnegative_fields as _has_nonnegative_fields,
    int_equal as _int_equal,
    int_list as _int_list,
    int_list_contains as _int_list_contains,
    kv_cache_contract_from_template_config as _kv_cache_contract_from_template_config,
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
from .reports.tensorization import (
    decode_shell_linear_weight_transform_complete as _decode_shell_linear_weight_transform_complete,
    decode_shell_linear_weight_transform_observed as _decode_shell_linear_weight_transform_observed,
    decode_shell_linear_weight_transform_paths as _decode_shell_linear_weight_transform_paths,
    embedding_norm_weight_transform_complete as _embedding_norm_weight_transform_complete,
    embedding_norm_weight_transform_observed as _embedding_norm_weight_transform_observed,
    embedding_norm_weight_transform_paths as _embedding_norm_weight_transform_paths,
    expected_tensorized_physical_shape as _expected_tensorized_physical_shape,
    linear_weight_transform_complete as _linear_weight_transform_complete,
    linear_weight_transform_observed as _linear_weight_transform_observed,
    linear_weight_transform_paths as _linear_weight_transform_paths,
    lm_head_source_reference_complete as _lm_head_source_reference_complete,
    lm_head_source_reference_observed as _lm_head_source_reference_observed,
    lm_head_transform_complete as _lm_head_transform_complete,
    lm_head_transform_observed as _lm_head_transform_observed,
    step_tensorization_summary as _step_tensorization_summary,
    tensorized_physical_shape_mismatches as _tensorized_physical_shape_mismatches,
    transformed_tensor_paths as _transformed_tensor_paths,
)
from .reports.validation import (
    REAL_DECODE_VALIDATION_STEPS,
    acceptance_check_passed as _acceptance_check_passed,
    final_acceptance_gate_matrix as _final_acceptance_gate_matrix,
    generate_prefill_decode_ready as _generate_prefill_decode_ready,
    model_end_to_end_readiness as _model_end_to_end_readiness,
    positive_scalar_count as _positive_scalar_count,
    real_decode_acceptance as _real_decode_acceptance,
    real_decode_acceptance_scope as _real_decode_acceptance_scope,
    real_decode_final_acceptance_plan as _real_decode_final_acceptance_plan,
    runtime_input_scope as _runtime_input_scope,
    step_synthetic_rotary_tensor_count as _step_synthetic_rotary_tensor_count,
    step_synthetic_runtime_input_count as _step_synthetic_runtime_input_count,
    validate_direct_acceptance as _validate_direct_acceptance,
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
