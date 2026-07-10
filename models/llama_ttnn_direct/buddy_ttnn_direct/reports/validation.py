from __future__ import annotations

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
