from __future__ import annotations

from typing import Any


def real_decode_runtime_diagnostics(
    report: dict[str, Any],
) -> dict[str, Any]:
    process_environment = report.get("tenstorrent_process_environment") or {}
    if process_environment.get("status") == "busy" and (
        report.get("guard_device_busy") is True
        or report.get("status") == "device_busy"
    ):
        return {
            "status": "device_busy",
            "device_busy": True,
            "device_reset_recommended": bool(
                process_environment.get("reset_in_progress")
            ),
            "findings": [
                {
                    "kind": "tenstorrent_device_busy",
                    "conflict_count": process_environment.get(
                        "conflict_count", 0
                    ),
                    "reset_in_progress": process_environment.get(
                        "reset_in_progress"
                    ),
                    "conflicts": process_environment.get("conflicts", []),
                }
            ],
            "recommended_action": (
                "Wait for an exclusive P150A window or stop external "
                "Tenstorrent reset/example jobs, then rerun "
                "validate-real-decode."
            ),
        }
    runtime_health = report.get("tenstorrent_runtime_health") or {}
    if runtime_health.get("status") in {"fail", "timeout", "error"} and (
        report.get("guard_device_health") is True
        or report.get("status") == "device_unhealthy"
    ):
        return {
            "status": "device_unhealthy",
            "device_busy": False,
            "device_reset_recommended": True,
            "findings": [
                {
                    "kind": "tenstorrent_runtime_health_failed",
                    "health_status": runtime_health.get("status"),
                    "device_id": runtime_health.get("device_id"),
                    "returncode": runtime_health.get("returncode"),
                    "timeout_seconds": runtime_health.get("timeout_seconds"),
                    "stdout": runtime_health.get("stdout"),
                    "stderr": runtime_health.get("stderr"),
                    "error": runtime_health.get("error"),
                }
            ],
            "recommended_action": (
                "Confirm no other jobs are using the board, reset the target "
                "device, then rerun validate-real-decode with "
                "--guard-device-busy --guard-device-health."
            ),
        }
    findings = runtime_error_findings(report.get("steps") or {})
    reset_recommended = any(
        finding["kind"] == "tenstorrent_firmware_init_failed"
        for finding in findings
    )
    return {
        "status": (
            "device_reset_recommended" if reset_recommended else "none"
        ),
        "device_reset_recommended": reset_recommended,
        "findings": findings,
        "recommended_action": (
            "Confirm no other jobs are using the board, reset the target "
            "device with tt-smi -r /dev/tenstorrent/<device_id>, then rerun "
            "validate-real-decode."
            if reset_recommended
            else None
        ),
    }


def runtime_error_findings(value: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in walk_strings(value):
        normalized = text.lower()
        if (
            "failed to initialize fw" in normalized
            or "try resetting the board" in normalized
            or ("timeout" in normalized and "physical cores" in normalized)
        ):
            findings.append(
                {
                    "kind": "tenstorrent_firmware_init_failed",
                    "path": path,
                    "message_excerpt": diagnostic_excerpt(text),
                }
            )
    return findings


def walk_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path or "$", value)]
    if isinstance(value, dict):
        values: list[tuple[str, str]] = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            values.extend(walk_strings(item, child))
        return values
    if isinstance(value, list):
        values = []
        for index, item in enumerate(value):
            child = f"{path}[{index}]" if path else f"[{index}]"
            values.extend(walk_strings(item, child))
        return values
    return []


def diagnostic_excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
