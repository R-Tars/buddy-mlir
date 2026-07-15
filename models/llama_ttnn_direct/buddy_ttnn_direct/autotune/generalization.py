from __future__ import annotations

import copy
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .confirmation import ConfirmationPolicy
from .schema import canonical_json, sha256_json
from .search import (
    SEARCH_SCHEMA_VERSION,
    SearchBudget,
    SearchCallbacks,
    SearchProposalGroup,
    atomic_write_json,
    run_hierarchical_search,
)
from .space import SearchSpaceConfig

GENERALIZATION_SCHEMA_VERSION = 1
_SEED_POLICIES = {"generic", "baseline_only"}
_FORBIDDEN_CHALLENGER_SOURCES = {
    "official_hand_seed",
    "manual_official_seed",
    "official_winner_injected",
}
_TARGET_FIELDS = (
    "batch_size",
    "prefill_len",
    "cache_len",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


class GeneralizationError(ValueError):
    """Raised when a cross-workload campaign is not comparable."""


@dataclass(frozen=True)
class GeneralizationWorkload:
    name: str
    model_id: str
    base_space: SearchSpaceConfig = field(repr=False)
    proposal_groups: tuple[SearchProposalGroup, ...] = field(repr=False)
    callbacks: SearchCallbacks = field(repr=False)
    run_identity: str
    _template_config_json: str = field(repr=False)
    _target_json: str = field(repr=False)
    seed_policy: str = "baseline_only"
    existing_search_report: Path | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.model_id or not self.run_identity:
            raise GeneralizationError(
                "workload name, model_id, and run_identity must be non-empty"
            )
        if self.seed_policy not in _SEED_POLICIES:
            raise GeneralizationError(
                f"seed_policy must be one of {sorted(_SEED_POLICIES)}"
            )
        template = _decode_object(self._template_config_json, "template config")
        if not template:
            raise GeneralizationError("template config must be non-empty")
        target = _decode_object(self._target_json, "workload target")
        missing = [name for name in _TARGET_FIELDS if name not in target]
        if missing:
            raise GeneralizationError(
                "workload target is missing: " + ", ".join(missing)
            )
        for name in _TARGET_FIELDS:
            try:
                value = int(target[name])
            except (TypeError, ValueError) as exc:
                raise GeneralizationError(
                    f"workload target {name} must be an integer"
                ) from exc
            if value <= 0:
                raise GeneralizationError(f"workload target {name} must be positive")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        model_id: str,
        base_space: SearchSpaceConfig,
        proposal_groups: Sequence[SearchProposalGroup],
        template_config: Mapping[str, Any],
        target: Mapping[str, Any],
        callbacks: SearchCallbacks | None = None,
        run_identity: str,
        seed_policy: str = "baseline_only",
        existing_search_report: str | Path | None = None,
    ) -> "GeneralizationWorkload":
        return cls(
            name=str(name),
            model_id=str(model_id),
            base_space=base_space,
            proposal_groups=tuple(proposal_groups),
            callbacks=callbacks or SearchCallbacks(),
            run_identity=str(run_identity),
            _template_config_json=canonical_json(template_config),
            _target_json=canonical_json(target),
            seed_policy=str(seed_policy),
            existing_search_report=(
                Path(existing_search_report)
                if existing_search_report is not None
                else None
            ),
        )

    @property
    def template_config(self) -> dict[str, Any]:
        return _decode_object(self._template_config_json, "template config")

    @property
    def target(self) -> dict[str, Any]:
        return _decode_object(self._target_json, "workload target")

    @property
    def shape_sha256(self) -> str:
        return sha256_json(
            {
                "model_id": self.model_id,
                "target": self.target,
            }
        )

    @property
    def workload_sha256(self) -> str:
        return sha256_json(self.identity_payload())

    @property
    def group_signature(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": group.stage,
                "name": group.name,
            }
            for group in self.proposal_groups
        ]

    def identity_payload(self) -> dict[str, Any]:
        existing = None
        if self.existing_search_report is not None:
            existing_report = _read_json(self.existing_search_report)
            existing = {
                "path": str(self.existing_search_report),
                "sha256": (
                    sha256_json(existing_report)
                    if isinstance(existing_report, Mapping)
                    else None
                ),
            }
        return {
            "name": self.name,
            "model_id": self.model_id,
            "target": self.target,
            "shape_sha256": self.shape_sha256,
            "base_space_sha256": sha256_json(self.base_space.to_dict()),
            "proposal_groups": [group.to_dict() for group in self.proposal_groups],
            "group_signature": self.group_signature,
            "template_config_sha256": sha256_json(self.template_config),
            "run_identity": self.run_identity,
            "seed_policy": self.seed_policy,
            "existing_search_report": existing,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "target": self.target,
            "shape_sha256": self.shape_sha256,
            "workload_sha256": self.workload_sha256,
            "base_space_sha256": sha256_json(self.base_space.to_dict()),
            "proposal_group_signature": self.group_signature,
            "proposal_count": sum(
                len(group.proposals) for group in self.proposal_groups
            ),
            "seed_policy": self.seed_policy,
            "official_seed_role": "incumbent_baseline_only",
            "existing_search_report": (
                str(self.existing_search_report)
                if self.existing_search_report is not None
                else None
            ),
        }


@dataclass(frozen=True)
class GeneralizationPolicy:
    minimum_model_count: int = 2
    minimum_workloads_per_model: int = 3

    def __post_init__(self) -> None:
        if self.minimum_model_count < 2:
            raise GeneralizationError("minimum_model_count must be at least two")
        if self.minimum_workloads_per_model < 3:
            raise GeneralizationError(
                "minimum_workloads_per_model must be at least three"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_model_count": self.minimum_model_count,
            "minimum_workloads_per_model": self.minimum_workloads_per_model,
            "acceptance": "2_models_or_1_model_x_3_workloads",
        }


def run_generalization_suite(
    *,
    workloads: Sequence[GeneralizationWorkload],
    out_dir: str | Path,
    budget: SearchBudget | None = None,
    confirmation_policy: ConfirmationPolicy | None = None,
    policy: GeneralizationPolicy | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "generalization_report.json"
    active_budget = budget or SearchBudget()
    active_confirmation = confirmation_policy or ConfirmationPolicy()
    active_policy = policy or GeneralizationPolicy()
    cases = tuple(workloads)
    report: dict[str, Any] = {
        "schema_version": GENERALIZATION_SCHEMA_VERSION,
        "stage": "semantic-autotune-generalization",
        "algorithm": "hierarchical_constrained_beam_search",
        "search_schema_version": SEARCH_SCHEMA_VERSION,
        "status": "running",
        "passed": False,
        "dry_run": bool(dry_run),
        "report_path": str(report_path),
        "budget_per_workload": active_budget.to_dict(),
        "confirmation_policy": active_confirmation.to_dict(),
        "generalization_policy": active_policy.to_dict(),
        "resume": {"requested": bool(resume), "completed_report_reused": False},
        "workloads": [],
        "active_workload": None,
        "acceptance": None,
        "failure_report_written": True,
    }
    try:
        if not cases:
            raise GeneralizationError("generalization suite requires workloads")
        _validate_workloads(cases)
        suite_fingerprint = sha256_json(
            {
                "schema_version": GENERALIZATION_SCHEMA_VERSION,
                "workloads": [case.identity_payload() for case in cases],
                "budget": active_budget.algorithm_identity(),
                "confirmation_policy": active_confirmation.to_dict(),
                "policy": active_policy.to_dict(),
                "dry_run": bool(dry_run),
            }
        )
        report["suite_fingerprint"] = suite_fingerprint
        completed = _read_json(report_path) if resume else None
        if (
            isinstance(completed, dict)
            and completed.get("suite_fingerprint") == suite_fingerprint
            and completed.get("status") in {"passed", "dry_run"}
        ):
            completed.setdefault("resume", {})["requested"] = True
            completed["resume"]["completed_report_reused"] = True
            atomic_write_json(report_path, completed)
            return completed

        atomic_write_json(report_path, report)
        for case in cases:
            report["active_workload"] = case.name
            atomic_write_json(report_path, report)
            if case.existing_search_report is not None:
                search_report = _load_existing_search_report(case)
                source = "existing_matched_campaign"
            else:
                case_dir = destination / "workloads" / _slug(case.name)
                search_report = run_hierarchical_search(
                    base_space=case.base_space,
                    proposal_groups=case.proposal_groups,
                    template_config=case.template_config,
                    out_dir=case_dir,
                    callbacks=case.callbacks,
                    budget=active_budget,
                    confirmation_policy=active_confirmation,
                    resume=resume,
                    dry_run=dry_run,
                    run_identity=case.run_identity,
                )
                source = "executed"
            record = {
                **case.to_dict(),
                "status": search_report.get("status"),
                "passed": bool(search_report.get("passed")),
                "source": source,
                "search_report": str(search_report.get("report_path", "")),
                "search_fingerprint": search_report.get("search_fingerprint"),
                "algorithm": search_report.get("algorithm"),
                "cartesian_exhaustive_search": search_report.get(
                    "cartesian_exhaustive_search"
                ),
                "budget_usage": copy.deepcopy(search_report.get("budget_usage") or {}),
                "selected_candidate": copy.deepcopy(
                    search_report.get("selected_candidate")
                ),
                "performance": _performance_summary(search_report),
                "artifacts": copy.deepcopy(search_report.get("artifacts") or {}),
                "search_acceptance": copy.deepcopy(search_report.get("acceptance")),
                "error": copy.deepcopy(search_report.get("error")),
            }
            report["workloads"].append(record)
            report["active_workload"] = None
            atomic_write_json(report_path, report)

        acceptance = _suite_acceptance(
            cases=cases,
            records=report["workloads"],
            policy=active_policy,
        )
        report.update(
            {
                "status": (
                    "dry_run"
                    if dry_run and acceptance["passed"]
                    else ("passed" if acceptance["passed"] else "failed")
                ),
                "passed": acceptance["passed"],
                "acceptance": acceptance,
                "summary": _suite_summary(report["workloads"]),
                "active_workload": None,
            }
        )
        atomic_write_json(report_path, report)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "passed": False,
                "active_workload": None,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "failure_report_written": True,
            }
        )
        atomic_write_json(report_path, report)
        return report


def _validate_workloads(cases: Sequence[GeneralizationWorkload]) -> None:
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise GeneralizationError("workload names must be unique")
    shapes = [case.shape_sha256 for case in cases]
    if len(set(shapes)) != len(shapes):
        raise GeneralizationError("workload model/target shapes must be distinct")
    signature = cases[0].group_signature
    for case in cases:
        if case.group_signature != signature:
            raise GeneralizationError(
                "all workloads must use the same template/op/layout group structure"
            )
        for group in case.proposal_groups:
            for proposal in group.proposals:
                if proposal.source in _FORBIDDEN_CHALLENGER_SOURCES:
                    raise GeneralizationError(
                        f"workload {case.name!r} injects forbidden challenger "
                        f"source {proposal.source!r}"
                    )


def _load_existing_search_report(case: GeneralizationWorkload) -> dict[str, Any]:
    path = case.existing_search_report
    if path is None:
        raise GeneralizationError("existing report path is missing")
    report = _read_json(path)
    if not isinstance(report, dict):
        raise GeneralizationError(
            f"existing search report is missing or malformed: {path}"
        )
    if report.get("algorithm") != "hierarchical_constrained_beam_search":
        raise GeneralizationError(
            f"existing report for {case.name!r} uses a different algorithm"
        )
    frozen = report.get("frozen_campaign_identity") or {}
    observed_target = frozen.get("target") or {}
    for key in ("batch_size", "cache_len", "prefill_len"):
        if int(observed_target.get(key, -1)) != int(case.target[key]):
            raise GeneralizationError(
                f"existing report target mismatch for {case.name}.{key}"
            )
    return report


def _suite_acceptance(
    *,
    cases: Sequence[GeneralizationWorkload],
    records: Sequence[Mapping[str, Any]],
    policy: GeneralizationPolicy,
) -> dict[str, Any]:
    model_workloads: dict[str, int] = {}
    for case in cases:
        model_workloads[case.model_id] = model_workloads.get(case.model_id, 0) + 1
    publication_breadth = bool(
        len(model_workloads) >= policy.minimum_model_count
        or any(
            count >= policy.minimum_workloads_per_model
            for count in model_workloads.values()
        )
    )
    final_configs = [
        (record.get("artifacts") or {}).get("best_config") for record in records
    ]
    checks = {
        "same_hierarchical_algorithm": all(
            record.get("algorithm") == "hierarchical_constrained_beam_search"
            for record in records
        ),
        "different_shape_fingerprints": len(
            {record.get("shape_sha256") for record in records}
        )
        == len(records),
        "official_seed_is_baseline_only": all(
            case.seed_policy in _SEED_POLICIES for case in cases
        ),
        "no_cartesian_exhaustive_search": all(
            record.get("cartesian_exhaustive_search") is False for record in records
        ),
        "all_workloads_passed": all(bool(record.get("passed")) for record in records),
        "all_final_configs_buildable": all(
            value and Path(str(value)).is_file() for value in final_configs
        ),
        "publication_breadth_2_models_or_1x3_workloads": publication_breadth,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "model_count": len(model_workloads),
        "workload_count": len(records),
        "workloads_per_model": model_workloads,
    }


def _performance_summary(search_report: Mapping[str, Any]) -> dict[str, Any]:
    confirmation = search_report.get("matched_ab_confirmation") or {}
    arms = confirmation.get("arms") or {}
    return {
        "metric": (confirmation.get("policy") or {}).get("metric"),
        "incumbent_median": (arms.get("incumbent") or {}).get("median"),
        "challenger_median": (arms.get("challenger") or {}).get("median"),
        "incumbent_cv": (arms.get("incumbent") or {}).get("cv"),
        "challenger_cv": (arms.get("challenger") or {}).get("cv"),
        "relative_improvement": confirmation.get("relative_improvement"),
        "promotion": copy.deepcopy(confirmation.get("promotion")),
    }


def _suite_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "workload_count": len(records),
        "passed_workload_count": sum(1 for record in records if record.get("passed")),
        "total_candidate_evaluations": sum(
            int((record.get("budget_usage") or {}).get("evaluation_count", 0))
            for record in records
        ),
        "total_device_minutes": sum(
            float((record.get("budget_usage") or {}).get("device_minutes", 0.0))
            for record in records
        ),
    }


def _decode_object(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise GeneralizationError(f"{label} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise GeneralizationError(f"{label} must be an object")
    return decoded


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _slug(value: str) -> str:
    return (
        "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in value
        ).strip("-.")
        or "workload"
    )
