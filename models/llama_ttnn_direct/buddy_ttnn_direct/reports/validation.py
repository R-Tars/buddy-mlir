from __future__ import annotations

from pathlib import Path
from typing import Any


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

def validate_direct_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    from .. import validation

    VALIDATION_STEPS = validation.VALIDATION_STEPS
    ATTENTION_PRIMITIVES = validation.ATTENTION_PRIMITIVES
    DECODE_STEP_AUTOTUNE_KNOBS = validation.DECODE_STEP_AUTOTUNE_KNOBS
    _acceptance_check = validation._acceptance_check
    _paths_exist = validation._paths_exist
    _official_required_field_coverage_complete = validation._official_required_field_coverage_complete
    _official_required_field_coverage_observed = validation._official_required_field_coverage_observed
    _positive_number = validation._positive_number
    _contains_all = validation._contains_all
    _attention_primitives_dry_run_complete = validation._attention_primitives_dry_run_complete
    _attention_primitives_dry_run_observed = validation._attention_primitives_dry_run_observed
    _non_empty_string = validation._non_empty_string
    _autotune_knob_coverage_complete = validation._autotune_knob_coverage_complete
    _autotune_knob_coverage_observed = validation._autotune_knob_coverage_observed
    _status_count_matches_total = validation._status_count_matches_total
    _path_exists = validation._path_exists
    _required_validate_direct_artifacts_exist = validation._required_validate_direct_artifacts_exist
    _validate_direct_artifact_observed = validation._validate_direct_artifact_observed
    _autotune_default_knobs_varied = validation._autotune_default_knobs_varied
    _autotune_knob_variation_observed = validation._autotune_knob_variation_observed

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
