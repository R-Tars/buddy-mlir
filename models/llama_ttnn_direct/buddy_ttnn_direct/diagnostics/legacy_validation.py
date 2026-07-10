from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _validation_api() -> Any:
    from .. import validation

    return validation


def recover_real_decode_process_failure(
    *,
    out_dir: str | Path,
    returncode: int,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Complete real-decode evidence after an isolated runner crash."""
    validation = _validation_api()
    steps = validation.REAL_DECODE_VALIDATION_STEPS
    root = Path(out_dir)
    report_path = root / "real_decode_validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    else:
        report = {
            "schema_version": 1,
            "command": "validate-real-decode",
            "status": "running",
            "out_dir": str(root),
            "dry_run": False,
            "guard_device_health": True,
            "results": {step: "pending" for step in steps},
            "steps": {},
            "artifacts": {
                "evidence_manifest": str(
                    root / "real_decode_evidence_manifest.json"
                ),
                "report": str(report_path),
            },
        }

    stdout_text = validation._coerce_process_text(stdout)
    stderr_text = validation._coerce_process_text(stderr)
    status = (
        "device_unhealthy"
        if validation._real_decode_process_failure_looks_device_related(
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )
        else "fail"
    )
    failed_step = validation._first_pending_real_decode_step(report) or "process"
    message = (
        "Isolated validate-real-decode subprocess exited before completing "
        f"{failed_step}."
    )
    if status == "device_unhealthy":
        message = (
            "Isolated validate-real-decode subprocess exited with a "
            "Tenstorrent runtime/device failure before evidence completion."
        )
        report["tenstorrent_runtime_health"] = {
            "status": "fail",
            "device_id": report.get("device_id"),
            "returncode": returncode,
            "stdout": diagnostic_excerpt(stdout_text, limit=2000),
            "stderr": diagnostic_excerpt(stderr_text, limit=2000),
        }

    report["status"] = status
    report["message"] = message
    report["subprocess_failure"] = {
        "status": status,
        "returncode": returncode,
        "signal": -returncode if returncode < 0 else None,
        "command": command or [],
        "stdout": diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": diagnostic_excerpt(stderr_text, limit=2000),
    }
    if failed_step in steps:
        report["results"][failed_step] = status
        report["steps"][failed_step] = {
            "status": status,
            "error": {
                "type": "SubprocessFailure",
                "returncode": returncode,
                "message": message,
                "stdout": diagnostic_excerpt(stdout_text),
                "stderr": diagnostic_excerpt(stderr_text),
            },
        }
        validation._mark_remaining_skipped(report, failed_step, steps)
    else:
        report["steps"]["process"] = {
            "status": status,
            "error": report["subprocess_failure"],
        }

    paths = {
        name: Path(path)
        for name, path in (report.get("artifacts") or {}).items()
        if isinstance(path, str)
    }
    paths.setdefault(
        "evidence_manifest",
        root / "real_decode_evidence_manifest.json",
    )
    paths.setdefault("report", report_path)
    report["artifacts"] = {name: str(path) for name, path in paths.items()}
    report["runtime_diagnostics"] = real_decode_runtime_diagnostics(report)
    evidence = validation._real_decode_evidence_manifest(report, paths)
    validation._write_json(paths["evidence_manifest"], evidence)
    report["evidence"] = {
        "status": evidence["status"],
        "manifest": str(paths["evidence_manifest"]),
        "artifact_count": len(evidence["artifacts"]),
        "acceptance_scope": evidence.get("acceptance_scope", {}),
        "model_end_to_end_readiness": evidence.get(
            "model_end_to_end_readiness",
            {},
        ),
        "failed_acceptance_checks": evidence["acceptance"]["failed_checks"],
    }
    validation._write_json(report_path, report)
    return report


def recover_real_decode_process_timeout(
    *,
    out_dir: str | Path,
    timeout_seconds: float,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Complete real-decode evidence after an isolated runner timeout."""
    validation = _validation_api()
    steps = validation.REAL_DECODE_VALIDATION_STEPS
    root = Path(out_dir)
    report_path = root / "real_decode_validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    else:
        report = {
            "schema_version": 1,
            "command": "validate-real-decode",
            "status": "running",
            "out_dir": str(root),
            "dry_run": False,
            "guard_device_health": True,
            "results": {step: "pending" for step in steps},
            "steps": {},
            "artifacts": {
                "evidence_manifest": str(
                    root / "real_decode_evidence_manifest.json"
                ),
                "report": str(report_path),
            },
        }

    stdout_text = validation._coerce_process_text(stdout)
    stderr_text = validation._coerce_process_text(stderr)
    status = "device_unhealthy"
    failed_step = validation._first_pending_real_decode_step(report) or "process"
    message = (
        "Isolated validate-real-decode subprocess timed out before "
        f"completing {failed_step}."
    )
    report["status"] = status
    report["message"] = message
    report["tenstorrent_runtime_health"] = {
        "status": "timeout",
        "device_id": report.get("device_id"),
        "timeout_seconds": timeout_seconds,
        "stdout": diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": diagnostic_excerpt(stderr_text, limit=2000),
    }
    report["subprocess_timeout"] = {
        "status": status,
        "timeout_seconds": timeout_seconds,
        "command": command or [],
        "stdout": diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": diagnostic_excerpt(stderr_text, limit=2000),
    }
    if failed_step in steps:
        report["results"][failed_step] = status
        report["steps"][failed_step] = {
            "status": status,
            "error": {
                "type": "SubprocessTimeout",
                "timeout_seconds": timeout_seconds,
                "message": message,
                "stdout": diagnostic_excerpt(stdout_text),
                "stderr": diagnostic_excerpt(stderr_text),
            },
        }
        validation._mark_remaining_skipped(report, failed_step, steps)
    else:
        report["steps"]["process"] = {
            "status": status,
            "error": report["subprocess_timeout"],
        }

    paths = {
        name: Path(path)
        for name, path in (report.get("artifacts") or {}).items()
        if isinstance(path, str)
    }
    paths.setdefault(
        "evidence_manifest",
        root / "real_decode_evidence_manifest.json",
    )
    paths.setdefault("report", report_path)
    report["artifacts"] = {name: str(path) for name, path in paths.items()}
    report["runtime_diagnostics"] = real_decode_runtime_diagnostics(report)
    evidence = validation._real_decode_evidence_manifest(report, paths)
    validation._write_json(paths["evidence_manifest"], evidence)
    report["evidence"] = {
        "status": evidence["status"],
        "manifest": str(paths["evidence_manifest"]),
        "artifact_count": len(evidence["artifacts"]),
        "acceptance_scope": evidence.get("acceptance_scope", {}),
        "model_end_to_end_readiness": evidence.get(
            "model_end_to_end_readiness",
            {},
        ),
        "failed_acceptance_checks": evidence["acceptance"]["failed_checks"],
    }
    validation._write_json(report_path, report)
    return report


def real_decode_runtime_diagnostics(
    report: dict[str, Any],
) -> dict[str, Any]:
    process_environment = report.get("tenstorrent_process_environment") or {}
    if (
        process_environment.get("status") == "busy"
        and (
            report.get("guard_device_busy") is True
            or report.get("status") == "device_busy"
        )
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
                        "conflict_count",
                        0,
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
    if (
        runtime_health.get("status") in {"fail", "timeout", "error"}
        and (
            report.get("guard_device_health") is True
            or report.get("status") == "device_unhealthy"
        )
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
    reset_findings = [
        finding
        for finding in findings
        if finding["kind"] == "tenstorrent_firmware_init_failed"
    ]
    device_reset_recommended = bool(reset_findings)
    return {
        "status": (
            "device_reset_recommended"
            if device_reset_recommended
            else "none"
        ),
        "device_reset_recommended": device_reset_recommended,
        "findings": findings,
        "recommended_action": (
            "Confirm no other jobs are using the board, reset the target "
            "device with tt-smi -r /dev/tenstorrent/<device_id>, then rerun "
            "validate-real-decode."
            if device_reset_recommended
            else None
        ),
    }


def tenstorrent_device_preflight_diagnostics_from_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    return tenstorrent_device_preflight_diagnostics(
        ttnn_environment=report.get("ttnn_environment"),
        setup_environment=report.get("tenstorrent_setup_environment"),
        device_environment=report.get("tenstorrent_device_environment"),
        process_environment=report.get("tenstorrent_process_environment"),
        runtime_health=report.get("tenstorrent_runtime_health"),
        guard_device_busy=bool(report.get("guard_device_busy")),
        guard_device_health=bool(report.get("guard_device_health")),
    )


def tenstorrent_device_preflight_diagnostics(
    *,
    ttnn_environment: dict[str, Any] | None,
    setup_environment: dict[str, Any] | None,
    device_environment: dict[str, Any] | None,
    process_environment: dict[str, Any] | None,
    runtime_health: dict[str, Any] | None,
    guard_device_busy: bool,
    guard_device_health: bool,
) -> dict[str, Any]:
    ttnn_environment = ttnn_environment or {}
    setup_environment = setup_environment or {}
    device_environment = device_environment or {}
    process_environment = process_environment or {}
    runtime_health = runtime_health or {}
    findings: list[dict[str, Any]] = []

    setup_ttnn_available = setup_environment.get("ttnn_module_available")
    ttnn_available = (
        ttnn_environment.get("module_available") is True
        or setup_ttnn_available is True
    )
    if not ttnn_available:
        findings.append(
            {
                "kind": "ttnn_module_unavailable",
                "module_file": ttnn_environment.get("module_file"),
                "python_executable": setup_environment.get(
                    "python_executable"
                ),
            }
        )

    if setup_environment.get("reference_doc_available") is not True:
        findings.append(
            {
                "kind": "tenstorrent_environment_doc_missing",
                "reference_doc": setup_environment.get("reference_doc"),
            }
        )

    if device_environment.get("device_available") is not True:
        findings.append(
            {
                "kind": "tenstorrent_device_not_visible",
                "device_nodes": device_environment.get("device_nodes", []),
                "filesystem_entries": device_environment.get(
                    "filesystem_entries",
                    [],
                ),
                "driver_loaded": device_environment.get("driver_loaded"),
                "tt_smi": device_environment.get("tt_smi"),
            }
        )

    if guard_device_busy and process_environment.get("status") == "busy":
        findings.append(
            {
                "kind": "tenstorrent_device_busy",
                "conflict_count": process_environment.get(
                    "conflict_count",
                    0,
                ),
                "reset_in_progress": process_environment.get(
                    "reset_in_progress"
                ),
                "conflicts": process_environment.get("conflicts", []),
            }
        )

    if (
        guard_device_health
        and runtime_health.get("status") in {"fail", "timeout", "error"}
    ):
        findings.append(
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
        )

    status = "ready"
    if any(
        finding["kind"] == "tenstorrent_runtime_health_failed"
        for finding in findings
    ):
        status = "device_unhealthy"
    elif any(
        finding["kind"] == "tenstorrent_device_busy"
        for finding in findings
    ):
        status = "device_busy"
    elif any(
        finding["kind"] == "tenstorrent_device_not_visible"
        for finding in findings
    ):
        status = "device_not_visible"
    elif any(
        finding["kind"] == "ttnn_module_unavailable"
        for finding in findings
    ):
        status = "environment_incomplete"

    return {
        "status": status,
        "ready": status == "ready",
        "device_available": device_environment.get("device_available"),
        "device_node_count": device_environment.get("device_node_count"),
        "device_nodes": device_environment.get("device_nodes", []),
        "driver_loaded": device_environment.get("driver_loaded"),
        "process_status": process_environment.get("status"),
        "process_conflict_count": process_environment.get("conflict_count"),
        "runtime_health_status": runtime_health.get("status", "not_checked"),
        "guard_device_busy": guard_device_busy,
        "guard_device_health": guard_device_health,
        "reference_doc": setup_environment.get("reference_doc"),
        "reference_doc_available": setup_environment.get(
            "reference_doc_available"
        ),
        "recommended_probe_commands": setup_environment.get(
            "recommended_probe_commands",
            [],
        ),
        "recommended_action": tenstorrent_preflight_recommended_action(
            status
        ),
        "findings": findings,
    }


def tenstorrent_preflight_recommended_action(status: str) -> str:
    if status == "ready":
        return (
            "Run validate-real-decode during an exclusive P150A window; keep "
            "--guard-device-busy and --guard-device-health enabled for "
            "bring-up evidence."
        )
    if status == "device_busy":
        return (
            "Wait for the current Tenstorrent workload/reset to finish, then "
            "rerun preflight and validate-real-decode in an exclusive board "
            "window."
        )
    if status == "device_unhealthy":
        return (
            "Confirm the board is idle, run the Tenstorrent environment smoke "
            "probes including python -m ttrt query, reset the target device if "
            "needed, then rerun validate-real-decode."
        )
    if status == "device_not_visible":
        return (
            "Reactivate the Tenstorrent environment from "
            "docs/TenstorrentEnvironment.md, verify /dev/tenstorrent* nodes "
            "and python -m ttrt query, then rerun preflight."
        )
    return (
        "Reactivate the Tenstorrent Python/runtime environment from "
        "docs/TenstorrentEnvironment.md before running real decode."
    )


def runtime_error_findings(value: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in walk_strings(value):
        normalized = text.lower()
        if (
            "failed to initialize fw" in normalized
            or "try resetting the board" in normalized
            or (
                "timeout" in normalized
                and "physical cores" in normalized
            )
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
        items: list[tuple[str, str]] = []
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            items.extend(walk_strings(nested, nested_path))
        return items
    if isinstance(value, list):
        items = []
        for index, nested in enumerate(value):
            nested_path = f"{path}[{index}]" if path else f"[{index}]"
            items.extend(walk_strings(nested, nested_path))
        return items
    return []


def diagnostic_excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
