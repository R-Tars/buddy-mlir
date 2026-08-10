from __future__ import annotations

from typing import Any

from .schema import path_exists as _path_exists


REQUIRED_VALIDATE_DIRECT_ARTIFACTS = (
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
    "package_dir",
)


def required_validate_direct_artifacts_exist(artifacts: Any) -> bool:
    if not isinstance(artifacts, dict):
        return False
    return all(
        _path_exists(artifacts.get(name))
        for name in REQUIRED_VALIDATE_DIRECT_ARTIFACTS
    )


def validate_direct_artifact_observed(
    artifacts: Any,
) -> dict[str, bool]:
    if not isinstance(artifacts, dict):
        return {}
    return {
        str(name): _path_exists(path)
        for name, path in sorted(artifacts.items())
    }
