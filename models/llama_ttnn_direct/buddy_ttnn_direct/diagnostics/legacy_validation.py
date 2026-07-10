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
            "stdout": validation._diagnostic_excerpt(stdout_text, limit=2000),
            "stderr": validation._diagnostic_excerpt(stderr_text, limit=2000),
        }

    report["status"] = status
    report["message"] = message
    report["subprocess_failure"] = {
        "status": status,
        "returncode": returncode,
        "signal": -returncode if returncode < 0 else None,
        "command": command or [],
        "stdout": validation._diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": validation._diagnostic_excerpt(stderr_text, limit=2000),
    }
    if failed_step in steps:
        report["results"][failed_step] = status
        report["steps"][failed_step] = {
            "status": status,
            "error": {
                "type": "SubprocessFailure",
                "returncode": returncode,
                "message": message,
                "stdout": validation._diagnostic_excerpt(stdout_text),
                "stderr": validation._diagnostic_excerpt(stderr_text),
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
    report["runtime_diagnostics"] = (
        validation._real_decode_runtime_diagnostics(report)
    )
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
        "stdout": validation._diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": validation._diagnostic_excerpt(stderr_text, limit=2000),
    }
    report["subprocess_timeout"] = {
        "status": status,
        "timeout_seconds": timeout_seconds,
        "command": command or [],
        "stdout": validation._diagnostic_excerpt(stdout_text, limit=2000),
        "stderr": validation._diagnostic_excerpt(stderr_text, limit=2000),
    }
    if failed_step in steps:
        report["results"][failed_step] = status
        report["steps"][failed_step] = {
            "status": status,
            "error": {
                "type": "SubprocessTimeout",
                "timeout_seconds": timeout_seconds,
                "message": message,
                "stdout": validation._diagnostic_excerpt(stdout_text),
                "stderr": validation._diagnostic_excerpt(stderr_text),
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
    report["runtime_diagnostics"] = (
        validation._real_decode_runtime_diagnostics(report)
    )
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
