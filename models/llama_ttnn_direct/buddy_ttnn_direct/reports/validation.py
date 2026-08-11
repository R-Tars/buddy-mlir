from __future__ import annotations

from typing import Any

from .profiling import PROFILE_GENERATE_SECTION_KEYS
from .runtime import (
    ttnn_runtime_identity_available as _ttnn_runtime_identity_available,
    ttnn_runtime_identity_observed as _ttnn_runtime_identity_observed,
)
from .schema import (
    acceptance_check as _acceptance_check,
    field_keys as _field_keys,
    int_equal as _int_equal,
    non_empty_string as _non_empty_string,
    positive_number as _positive_number,
    safe_int as _safe_int,
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
