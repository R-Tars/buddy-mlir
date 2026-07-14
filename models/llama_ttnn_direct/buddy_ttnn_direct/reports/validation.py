from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codegen.ttnn_tensorizer import LINEAR_WEIGHT_TRANSFORM
from ..runtime.plans import DECODE_PARAMETER_ROLES
from .attention import (
    ATTENTION_LAYER_OPS,
    ATTENTION_PRIMITIVES,
    attention_layer_primitive_reports_complete as _attention_layer_primitive_reports_complete,
    attention_layer_primitive_reports_observed as _attention_layer_primitive_reports_observed,
    attention_primitive_reports_complete as _attention_primitive_reports_complete,
    attention_primitive_reports_observed as _attention_primitive_reports_observed,
    attention_primitives_dry_run_complete as _attention_primitives_dry_run_complete,
    attention_primitives_dry_run_observed as _attention_primitives_dry_run_observed,
)
from .artifacts import (
    required_validate_direct_artifacts_exist as _required_validate_direct_artifacts_exist,
    validate_direct_artifact_observed as _validate_direct_artifact_observed,
)
from .autotune import (
    DECODE_STEP_AUTOTUNE_KNOBS,
    autotune_best_candidate_summary_complete as _autotune_best_candidate_summary_complete,
    autotune_best_candidate_summary_observed as _autotune_best_candidate_summary_observed,
    autotune_candidates_complete as _autotune_candidates_complete,
    autotune_candidates_observed as _autotune_candidates_observed,
    autotune_default_knobs_varied as _autotune_default_knobs_varied,
    autotune_knob_coverage_complete as _autotune_knob_coverage_complete,
    autotune_knob_coverage_observed as _autotune_knob_coverage_observed,
    autotune_knob_variation_observed as _autotune_knob_variation_observed,
    autotune_leaderboard_complete as _autotune_leaderboard_complete,
    autotune_leaderboard_observed as _autotune_leaderboard_observed,
    autotune_output_kind_counts_complete as _autotune_output_kind_counts_complete,
    autotune_output_kind_counts_observed as _autotune_output_kind_counts_observed,
)
from .config import (
    PARITY_SECTIONS,
    config_gap_summary_complete as _config_gap_summary_complete,
    config_gap_summary_observed as _config_gap_summary_observed,
    official_required_field_coverage_complete as _official_required_field_coverage_complete,
    official_required_field_coverage_observed as _official_required_field_coverage_observed,
)
from .depth import (
    PROFILE_LAYER_LATENCY_KEYS,
    PROFILE_SECTION_LATENCY_KEYS,
    decode_depth_sweep_records_complete as _decode_depth_sweep_records_complete,
    decode_depth_sweep_records_observed as _decode_depth_sweep_records_observed,
)
from .performance import (
    OFFICIAL_PERFORMANCE_PARITY_METRIC,
    PROFILE_GENERATE_MILESTONE_IDS,
    official_performance_baseline_entry_complete as _official_performance_baseline_entry_complete,
    performance_baseline_entry_complete as _performance_baseline_entry_complete,
    performance_baseline_entry_summary as _performance_baseline_entry_summary,
    profile_generate_milestones_complete as _profile_generate_milestones_complete,
    throughput_baseline_summary as _throughput_baseline_summary,
)
from .profiling import (
    PROFILE_BOTTLENECK_SECTION_KEYS,
    PROFILE_GENERATE_SECTION_KEYS,
    bottleneck_summary_complete as _bottleneck_summary_complete,
    bottleneck_summary_observed as _bottleneck_summary_observed,
    layer_profile_field_keys as _layer_profile_field_keys,
    layer_profile_ids as _layer_profile_ids,
    layer_profiles_have_nonnegative_fields as _layer_profiles_have_nonnegative_fields,
    lm_head_profile_complete as _lm_head_profile_complete,
    lm_head_profile_observed as _lm_head_profile_observed,
    step_trace_summary as _step_trace_summary,
    trace_profile_complete as _trace_profile_complete,
    trace_profile_observed as _trace_profile_observed,
)
from .runtime import (
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
    observed_ops_cover_planned as _observed_ops_cover_planned,
    runtime_input_source_supported as _runtime_input_source_supported,
    step_ttnn_environment as _step_ttnn_environment,
    synthetic_runtime_inputs_accepted as _synthetic_runtime_inputs_accepted,
    ttnn_runtime_identity_available as _ttnn_runtime_identity_available,
    ttnn_runtime_identity_observed as _ttnn_runtime_identity_observed,
)
from .schema import (
    acceptance_check as _acceptance_check,
    contains_all as _contains_all,
    expected_layer_ids as _expected_layer_ids,
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
    paths_exist as _paths_exist,
    positive_count as _positive_count,
    positive_number as _positive_number,
    safe_int as _safe_int,
    status_count_matches_total as _status_count_matches_total,
)
from .tensorization import (
    decode_shell_linear_weight_transform_complete as _decode_shell_linear_weight_transform_complete,
    decode_shell_linear_weight_transform_observed as _decode_shell_linear_weight_transform_observed,
    embedding_norm_weight_transform_complete as _embedding_norm_weight_transform_complete,
    embedding_norm_weight_transform_observed as _embedding_norm_weight_transform_observed,
    embedding_norm_weight_transform_paths as _embedding_norm_weight_transform_paths,
    linear_weight_transform_complete as _linear_weight_transform_complete,
    linear_weight_transform_observed as _linear_weight_transform_observed,
    linear_weight_transform_paths as _linear_weight_transform_paths,
    lm_head_source_reference_complete as _lm_head_source_reference_complete,
    lm_head_source_reference_observed as _lm_head_source_reference_observed,
    lm_head_transform_complete as _lm_head_transform_complete,
    lm_head_transform_observed as _lm_head_transform_observed,
    step_tensorization_summary as _step_tensorization_summary,
    tensorized_physical_shape_mismatches as _tensorized_physical_shape_mismatches,
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


def validate_dryrun(
    *,
    artifacts: Any,
    generate: Any,
    profile: Any,
) -> dict[str, Any]:
    artifact_report = artifacts if isinstance(artifacts, dict) else {}
    generate_report = generate if isinstance(generate, dict) else {}
    profile_report = profile if isinstance(profile, dict) else {}
    checks = [
        _acceptance_check(
            "validate.program_artifacts",
            artifact_report.get("passed") is True,
            observed=artifact_report.get("files", {}),
        ),
        _acceptance_check(
            "validate.generate_dryrun",
            generate_report.get("passed") is True
            and generate_report.get("status") == "dry_run"
            and generate_report.get("dry_run") is True,
            observed=generate_report.get("status"),
            expected="dry_run",
        ),
        _acceptance_check(
            "validate.profile_dryrun",
            profile_report.get("passed") is True
            and profile_report.get("status") == "dry_run"
            and profile_report.get("dry_run") is True,
            observed=profile_report.get("status"),
            expected="dry_run",
        ),
    ]
    return _product_validation_result("dryrun", checks)


def validate_functional(
    *,
    artifacts: Any,
    generate: Any,
) -> dict[str, Any]:
    artifact_report = artifacts if isinstance(artifacts, dict) else {}
    generate_report = generate if isinstance(generate, dict) else {}
    checks = [
        _acceptance_check(
            "validate.program_artifacts",
            artifact_report.get("passed") is True,
            observed=artifact_report.get("files", {}),
        ),
        _acceptance_check(
            "validate.generate",
            generate_report.get("passed") is True
            and generate_report.get("status") == "passed",
            observed=generate_report.get("status"),
            expected="passed",
        ),
        _acceptance_check(
            "validate.prefill",
            generate_report.get("prefill_status") == "passed",
            observed=generate_report.get("prefill_status"),
            expected="passed",
        ),
        _acceptance_check(
            "validate.model_semantics",
            generate_report.get("model_semantics")
            == "prompt_conditioned_prefill_decode",
            observed=generate_report.get("model_semantics"),
            expected="prompt_conditioned_prefill_decode",
        ),
        _acceptance_check(
            "validate.kv_cache_source",
            generate_report.get("kv_cache_source") == "prefill",
            observed=generate_report.get("kv_cache_source"),
            expected="prefill",
        ),
        _acceptance_check(
            "validate.generated_text",
            _non_empty_string(generate_report.get("generated_text"))
            and generate_report.get("generated_text_status")
            not in {None, "not_run", "error"},
            observed={
                "status": generate_report.get("generated_text_status"),
                "text": generate_report.get("generated_text"),
            },
            expected="non-empty generated text",
        ),
    ]
    return _product_validation_result("functional", checks)


def validate_device(
    *,
    artifacts: Any,
    generate: Any,
    require_full_depth: bool = False,
    expected_device: Any = None,
    expected_device_id: Any = None,
) -> dict[str, Any]:
    generate_report = generate if isinstance(generate, dict) else {}
    functional = validate_functional(
        artifacts=artifacts,
        generate=generate_report,
    )
    environment = generate_report.get("ttnn_environment")
    observed_device = generate_report.get("device")
    observed_device_id = _safe_int(generate_report.get("device_id"))
    device_target_matches = (
        _non_empty_string(observed_device)
        and observed_device_id is not None
        and observed_device_id >= 0
        and (
            expected_device is None
            or str(observed_device) == str(expected_device)
        )
        and (
            expected_device_id is None
            or _int_equal(observed_device_id, expected_device_id)
        )
    )
    checks = list(functional["checks"])
    checks.extend(
        [
            _acceptance_check(
                "validate.device_identity",
                _ttnn_runtime_identity_available(environment),
                observed=_ttnn_runtime_identity_observed(environment),
                expected="available TTNN runtime identity",
            ),
            _acceptance_check(
                "validate.device_target",
                device_target_matches,
                observed={
                    "device": observed_device,
                    "device_id": observed_device_id,
                },
                expected={
                    "device": expected_device or "non-empty",
                    "device_id": expected_device_id
                    if expected_device_id is not None
                    else ">= 0",
                },
            ),
        ]
    )
    if require_full_depth:
        checks.append(
            _acceptance_check(
                "validate.full_depth",
                _int_equal(
                    generate_report.get("layers"),
                    generate_report.get("program_num_layers"),
                ),
                observed=generate_report.get("layers"),
                expected=generate_report.get("program_num_layers"),
            )
        )
    return _product_validation_result("device", checks)


def validate_performance(
    *,
    artifacts: Any,
    profile: Any,
    require_official_parity: bool = False,
    require_full_depth: bool = False,
) -> dict[str, Any]:
    artifact_report = artifacts if isinstance(artifacts, dict) else {}
    profile_report = profile if isinstance(profile, dict) else {}
    sections = profile_report.get("sections")
    section_keys = _field_keys(sections)
    checks = [
        _acceptance_check(
            "validate.program_artifacts",
            artifact_report.get("passed") is True,
            observed=artifact_report.get("files", {}),
        ),
        _acceptance_check(
            "validate.profile",
            profile_report.get("passed") is True
            and profile_report.get("status") == "profiled",
            observed=profile_report.get("status"),
            expected="profiled",
        ),
        _acceptance_check(
            "validate.positive_throughput",
            _positive_number(
                profile_report.get("tokens_per_second_per_user")
            ),
            observed=profile_report.get("tokens_per_second_per_user"),
            expected="> 0",
        ),
        _acceptance_check(
            "validate.profile_sections",
            isinstance(sections, dict)
            and set(PROFILE_GENERATE_SECTION_KEYS).issubset(sections),
            observed=section_keys,
            expected=list(PROFILE_GENERATE_SECTION_KEYS),
        ),
    ]
    if require_official_parity:
        checks.append(
            _acceptance_check(
                "validate.official_performance_parity",
                profile_report.get("official_performance_parity_claimed")
                is True,
                observed=profile_report.get(
                    "official_performance_parity_claimed"
                ),
                expected=True,
            )
        )
    if require_full_depth:
        checks.append(
            _acceptance_check(
                "validate.full_depth",
                _int_equal(
                    profile_report.get("layers"),
                    profile_report.get("program_num_layers"),
                ),
                observed=profile_report.get("layers"),
                expected=profile_report.get("program_num_layers"),
            )
        )
    return _product_validation_result("performance", checks)


def _product_validation_result(
    suite: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = all(check.get("passed") is True for check in checks)
    return {
        "schema_version": 1,
        "command": "validate",
        "suite": suite,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "check_count": len(checks),
        "failed_checks": [
            str(check.get("name"))
            for check in checks
            if check.get("passed") is not True
        ],
        "checks": checks,
    }


def acceptance_check_passed(acceptance: Any, name: str) -> bool:
    if not isinstance(acceptance, dict):
        return False
    checks = acceptance.get("checks")
    if not isinstance(checks, list):
        return False
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check.get("passed") is True
    return False


def final_acceptance_gate_matrix(
    report: dict[str, Any],
    acceptance: Any,
) -> dict[str, Any]:
    plan = report.get("final_acceptance_plan")
    if not isinstance(plan, dict):
        plan = {}
    checks = {}
    if isinstance(acceptance, dict):
        for check in acceptance.get("checks", []):
            if isinstance(check, dict) and isinstance(check.get("name"), str):
                checks[check["name"]] = check

    groups = {
        "full_decode_step": plan.get("full_decode_step_gate_names", []),
        "model_end_to_end": plan.get("model_end_to_end_gate_names", []),
        "official_performance_parity": plan.get(
            "official_performance_parity_gate_names",
            [],
        ),
        "optional": plan.get("optional_gate_names", []),
    }
    gate_entries = []
    for group, names in groups.items():
        if not isinstance(names, list):
            names = []
        for name in names:
            if not isinstance(name, str):
                continue
            check = checks.get(name)
            if check is None:
                status = "missing"
                passed = False
                details = {}
            else:
                passed = check.get("passed") is True
                status = "passed" if passed else "failed"
                details = {
                    key: value
                    for key, value in check.items()
                    if key not in {"name", "passed"}
                }
            gate_entries.append(
                {
                    "group": group,
                    "name": name,
                    "status": status,
                    "passed": passed,
                    "details": details,
                }
            )

    passed_gates = [
        gate["name"] for gate in gate_entries if gate["status"] == "passed"
    ]
    failed_gates = [
        gate["name"] for gate in gate_entries if gate["status"] == "failed"
    ]
    missing_gates = [
        gate["name"] for gate in gate_entries if gate["status"] == "missing"
    ]
    return {
        "schema_version": 1,
        "target_scope": plan.get("target_scope"),
        "planned_gate_count": len(gate_entries),
        "passed_gate_count": len(passed_gates),
        "failed_gate_count": len(failed_gates),
        "missing_gate_count": len(missing_gates),
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "missing_gates": missing_gates,
        "gates": gate_entries,
    }


def step_synthetic_runtime_input_count(step: dict[str, Any]) -> Any:
    setup = step.get("parameter_setup") or {}
    if not isinstance(setup, dict):
        return None
    return setup.get("synthetic_runtime_input_tensor_count")


def step_synthetic_rotary_tensor_count(step: dict[str, Any]) -> Any:
    setup = step.get("parameter_setup") or {}
    if not isinstance(setup, dict):
        return None
    return setup.get("synthetic_rotary_tensor_count")


def positive_scalar_count(value: Any) -> bool:
    numeric = _safe_int(value)
    return numeric is not None and numeric > 0


def _generated_tokens_present(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if isinstance(row, list) and row:
            return True
    return False


def generate_prefill_decode_ready(step: Any) -> bool:
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


def runtime_input_scope(report: dict[str, Any]) -> dict[str, Any]:
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
            runtime_count = step_synthetic_runtime_input_count(step)
        if runtime_count is None and source == "synthetic":
            runtime_count = step.get("runtime_input_tensor_count")
        rotary_count = step.get("synthetic_rotary_tensor_count")
        if rotary_count is None:
            rotary_count = step_synthetic_rotary_tensor_count(step)
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
            or positive_scalar_count(runtime_count)
            or positive_scalar_count(rotary_count)
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
                    or positive_scalar_count(
                        record.get("synthetic_runtime_input_tensor_count")
                    )
                    or positive_scalar_count(
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


def model_end_to_end_readiness(
    report: dict[str, Any],
    acceptance_scope: dict[str, Any],
) -> dict[str, Any]:
    runtime_scope = runtime_input_scope(report)
    accepted = bool(acceptance_scope.get("accepted_real_weight_runtime"))
    full_decode_ready = bool(
        acceptance_scope.get("full_decode_step_ready")
    )
    synthetic_inputs = bool(
        runtime_scope.get("uses_synthetic_runtime_inputs")
    )
    prompt_runtime_observed = any(
        positive_scalar_count(value)
        for value in (
            runtime_scope.get("prompt_runtime_input_tensor_counts") or {}
        ).values()
    )
    decode_runtime_state_observed = any(
        positive_scalar_count(value)
        for value in (
            runtime_scope.get("decode_runtime_state_input_tensor_counts")
            or {}
        ).values()
    )
    rotary_runtime_observed = any(
        positive_scalar_count(value)
        for value in (
            runtime_scope.get("rotary_runtime_input_tensor_counts") or {}
        ).values()
    )
    kv_cache_runtime_observed = any(
        positive_scalar_count(value)
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
    generate_ready = generate_prefill_decode_ready(generate_step)
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
    if not generate_ready:
        missing.append(
            "prefill+decode generate evidence with kv_cache_source=prefill "
            "and generated token/text output"
        )

    ready = (
        accepted
        and full_decode_ready
        and not synthetic_inputs
        and decode_loop_runtime_owned
        and generate_ready
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
    elif not generate_ready:
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
        "generate_prefill_decode_ready": generate_ready,
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


def real_decode_final_acceptance_plan(
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


def real_decode_acceptance_scope(
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
        and acceptance_check_passed(acceptance, "validation.full_depth_layers")
        and acceptance_check_passed(acceptance, "decode_depth_sweep.full_depth")
    )
    program_runtime_shape = (
        accepted
        and report.get("require_program_runtime_shape") is True
        and acceptance_check_passed(acceptance, "validation.program_batch_size")
        and acceptance_check_passed(acceptance, "validation.program_cache_len")
    )
    batch32_contract = (
        accepted
        and report.get("require_batch32_decode_step") is True
        and acceptance_check_passed(acceptance, "decode_step_contract.batch32")
    )
    trace = (
        accepted
        and report.get("require_trace") is True
        and acceptance_check_passed(acceptance, "single_layer_decode.trace_status")
        and acceptance_check_passed(acceptance, "smoke_decode_step.trace_status")
        and acceptance_check_passed(acceptance, "profile_decode_step.trace_status")
        and acceptance_check_passed(acceptance, "profile_decode_step.trace_profile")
    )
    numeric_shell = (
        accepted
        and report.get("require_decode_shell_numeric_reference") is True
        and acceptance_check_passed(acceptance, "decode_shell.numeric_reference")
    )
    baseline_ratio_floor = (
        accepted
        and report.get("min_baseline_ratio") is not None
        and acceptance_check_passed(
            acceptance,
            "profile_decode_step.min_baseline_ratio",
        )
    )
    official_positive_floor = (
        accepted
        and report.get("require_official_performance_parity") is True
        and acceptance_check_passed(
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
        and acceptance_check_passed(
            acceptance,
            "profile_decode_step.official_baseline_reference",
        )
    )
    official_metric = (
        accepted
        and report.get("require_official_performance_parity") is True
        and acceptance_check_passed(
            acceptance,
            "decode_step_autotune.metric",
        )
    )
    official_config_match = (
        accepted
        and report.get("require_official_config_match") is True
        and acceptance_check_passed(
            acceptance,
            "official_config_diff.official_reference_format",
        )
        and acceptance_check_passed(acceptance, "official_config_diff.match")
    )
    model_end_to_end = (
        accepted
        and report.get("require_model_end_to_end") is True
        and acceptance_check_passed(
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


def validate_direct_acceptance(report: dict[str, Any]) -> dict[str, Any]:
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


def real_decode_acceptance(
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
        runtime_scope = runtime_input_scope(report)
        decode_loop_runtime_owned = bool(
            report.get("decode_loop_runtime_owned")
            or prompt_loop.get("decode_loop_runtime_owned")
        )
        generate_ready = generate_prefill_decode_ready(generate_step)
        checks.append(
            _acceptance_check(
                "model_end_to_end_readiness.ready",
                (
                    not runtime_scope.get("uses_synthetic_runtime_inputs")
                    and decode_loop_runtime_owned
                    and generate_ready
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
                        generate_ready
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
