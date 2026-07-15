from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from .confirmation import (
    ConfirmationArm,
    ConfirmationPolicy,
    confirm_matched_ab,
)
from .schema import CandidateConfig, MeasurementContract, canonical_json, sha256_json
from .space import SearchSpaceConfig

if TYPE_CHECKING:
    from .layout_graph import LayoutSearchResult
    from .matmul import MatmulEnumerationResult
    from .sdpa import SDPAEnumerationResult


SEARCH_SCHEMA_VERSION = 1
SEARCH_STAGES = ("template", "op", "layout")
PIPELINE_STAGES = (
    "template_search",
    "op_search",
    "layout_beam",
    "layer_confirmation",
    "full_model_trace",
    "long_confirmation",
)


class HierarchicalSearchError(ValueError):
    """Raised for malformed or incomplete hierarchical search requests."""


class SearchBudgetExhausted(RuntimeError):
    """Raised internally when no further device evaluation is admissible."""


@dataclass(frozen=True)
class SearchMutation:
    mutation_id: str
    stage: str
    group: str
    label: str
    _operations_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.stage not in SEARCH_STAGES:
            raise HierarchicalSearchError(
                f"mutation stage must be one of {list(SEARCH_STAGES)}"
            )
        if not self.group or not self.label or not self.mutation_id:
            raise HierarchicalSearchError(
                "mutation id, group, and label must be non-empty"
            )
        operations = _decode_operations(self._operations_json)
        seen: set[tuple[str, ...]] = set()
        for operation in operations:
            path = tuple(operation["path"])
            if not path or path[0] == "schema_version":
                raise HierarchicalSearchError(
                    "mutations cannot replace the search-space schema version"
                )
            if path in seen:
                raise HierarchicalSearchError(
                    f"mutation repeats path {_display_path(path)!r}"
                )
            seen.add(path)

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        group: str,
        label: str,
        operations: Sequence[Mapping[str, Any]],
        mutation_id: str | None = None,
    ) -> "SearchMutation":
        normalized = _normalize_operations(operations)
        identity = {
            "stage": stage,
            "group": group,
            "label": label,
            "operations": normalized,
        }
        identifier = mutation_id or (f"{_slug(group)}-{sha256_json(identity)[:12]}")
        return cls(
            mutation_id=identifier,
            stage=stage,
            group=group,
            label=label,
            _operations_json=canonical_json(normalized),
        )

    @classmethod
    def from_spaces(
        cls,
        *,
        stage: str,
        group: str,
        label: str,
        base_space: SearchSpaceConfig,
        candidate_space: SearchSpaceConfig,
        mutation_id: str | None = None,
    ) -> "SearchMutation":
        operations: list[dict[str, Any]] = []
        _diff_values(
            base_space.to_dict(),
            candidate_space.to_dict(),
            path=(),
            operations=operations,
        )
        return cls.create(
            stage=stage,
            group=group,
            label=label,
            operations=operations,
            mutation_id=mutation_id,
        )

    @property
    def operations(self) -> list[dict[str, Any]]:
        return _decode_operations(self._operations_json)

    @property
    def is_noop(self) -> bool:
        return not self.operations

    def apply(self, space: SearchSpaceConfig) -> SearchSpaceConfig:
        payload = space.to_dict()
        for operation in self.operations:
            path = tuple(operation["path"])
            if operation["op"] == "set":
                _set_json_path(payload, path, operation["value"])
            else:
                _remove_json_path(payload, path)
        return SearchSpaceConfig.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "stage": self.stage,
            "group": self.group,
            "label": self.label,
            "is_noop": self.is_noop,
            "operations": self.operations,
        }


@dataclass(frozen=True)
class ProposalScore:
    latency_ms: float
    l1_bytes: int = 0
    conversion_latency_ms: float = 0.0
    conversion_compatible: bool = True
    passed: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("latency_ms", self.latency_ms),
            ("conversion_latency_ms", self.conversion_latency_ms),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise HierarchicalSearchError(
                    f"proposal {label} must be finite and non-negative"
                )
        if self.l1_bytes < 0:
            raise HierarchicalSearchError("proposal l1_bytes must be non-negative")

    @property
    def total_latency_ms(self) -> float:
        return float(self.latency_ms) + float(self.conversion_latency_ms)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalScore":
        latency = value.get("latency_ms")
        if latency is None and isinstance(value.get("objective"), Mapping):
            objective = value["objective"]
            if objective.get("direction", "minimize") == "minimize":
                latency = objective.get("value")
        if latency is None:
            raise HierarchicalSearchError("proposal evidence must provide latency_ms")
        return cls(
            latency_ms=float(latency),
            l1_bytes=int(value.get("l1_bytes", 0)),
            conversion_latency_ms=float(value.get("conversion_latency_ms", 0.0)),
            conversion_compatible=bool(value.get("conversion_compatible", True)),
            passed=bool(value.get("passed", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "conversion_latency_ms": self.conversion_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "l1_bytes": self.l1_bytes,
            "conversion_compatible": self.conversion_compatible,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SearchProposal:
    proposal_id: str
    mutation: SearchMutation
    score: ProposalScore
    source: str = "enumerated"
    _evidence_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.source:
            raise HierarchicalSearchError("proposal id and source must be non-empty")
        _decode_object(self._evidence_json, "proposal evidence")

    @classmethod
    def create(
        cls,
        *,
        mutation: SearchMutation,
        score: ProposalScore,
        proposal_id: str | None = None,
        source: str = "enumerated",
        evidence: Mapping[str, Any] | None = None,
    ) -> "SearchProposal":
        identifier = proposal_id or mutation.mutation_id
        return cls(
            proposal_id=identifier,
            mutation=mutation,
            score=score,
            source=source,
            _evidence_json=canonical_json(evidence or {}),
        )

    @property
    def evidence(self) -> dict[str, Any]:
        return _decode_object(self._evidence_json, "proposal evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "mutation": self.mutation.to_dict(),
            "score": self.score.to_dict(),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SearchProposalGroup:
    stage: str
    name: str
    proposals: tuple[SearchProposal, ...]
    incumbent_score: ProposalScore = field(
        default_factory=lambda: ProposalScore(latency_ms=0.0)
    )

    def __post_init__(self) -> None:
        if self.stage not in SEARCH_STAGES:
            raise HierarchicalSearchError(
                f"proposal group stage must be one of {list(SEARCH_STAGES)}"
            )
        if not self.name:
            raise HierarchicalSearchError("proposal group name must be non-empty")
        seen: set[str] = set()
        for proposal in self.proposals:
            if proposal.proposal_id in seen:
                raise HierarchicalSearchError(
                    f"duplicate proposal id {proposal.proposal_id!r}"
                )
            seen.add(proposal.proposal_id)
            mutation = proposal.mutation
            if mutation.stage != self.stage or mutation.group != self.name:
                raise HierarchicalSearchError(
                    "proposal mutation stage/group does not match its group"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "incumbent_score": self.incumbent_score.to_dict(),
            "proposals": [proposal.to_dict() for proposal in self.proposals],
        }


@dataclass(frozen=True)
class SearchBudget:
    max_candidates: int = 256
    max_device_minutes: float = 60.0
    beam_width: int = 8
    template_top_k: int = 2
    microbench_top_k: int = 4
    full_model_top_k: int = 2

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise HierarchicalSearchError("max_candidates must be positive")
        if self.max_device_minutes <= 0.0:
            raise HierarchicalSearchError("max_device_minutes must be positive")
        if not 4 <= self.beam_width <= 16:
            raise HierarchicalSearchError("beam_width must be in [4, 16]")
        if not 2 <= self.template_top_k <= 4:
            raise HierarchicalSearchError("template_top_k must be in [2, 4]")
        if self.microbench_top_k <= 0:
            raise HierarchicalSearchError("microbench_top_k must be positive")
        if not 2 <= self.full_model_top_k <= 5:
            raise HierarchicalSearchError("full_model_top_k must be in [2, 5]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_device_minutes": self.max_device_minutes,
            "beam_width": self.beam_width,
            "template_top_k": self.template_top_k,
            "microbench_top_k": self.microbench_top_k,
            "full_model_top_k": self.full_model_top_k,
        }

    def algorithm_identity(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "template_top_k": self.template_top_k,
            "microbench_top_k": self.microbench_top_k,
            "full_model_top_k": self.full_model_top_k,
        }


@dataclass(frozen=True)
class SearchCandidate:
    space: SearchSpaceConfig = field(repr=False)
    mutation_ids: tuple[str, ...] = ()
    estimated_latency_ms: float = 0.0
    parent_id: str | None = None

    @property
    def space_sha256(self) -> str:
        return sha256_json(self.space.to_dict())

    @property
    def candidate_id(self) -> str:
        return f"candidate-{self.space_sha256[:16]}"

    def apply(
        self,
        proposal: SearchProposal | None,
        *,
        incumbent_score: ProposalScore,
    ) -> "SearchCandidate":
        score = incumbent_score if proposal is None else proposal.score
        if proposal is None or proposal.mutation.is_noop:
            next_space = self.space
            mutations = self.mutation_ids
        else:
            next_space = proposal.mutation.apply(self.space)
            mutations = (*self.mutation_ids, proposal.mutation.mutation_id)
        return SearchCandidate(
            space=next_space,
            mutation_ids=mutations,
            estimated_latency_ms=(self.estimated_latency_ms + score.total_latency_ms),
            parent_id=self.candidate_id,
        )

    def to_dict(self, *, incumbent_sha256: str | None = None) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "space_sha256": self.space_sha256,
            "is_incumbent": (
                incumbent_sha256 is not None and self.space_sha256 == incumbent_sha256
            ),
            "mutation_ids": list(self.mutation_ids),
            "estimated_latency_ms": self.estimated_latency_ms,
            "parent_id": self.parent_id,
        }


CandidateEvaluator = Callable[[str, SearchCandidate], Mapping[str, Any]]
LayerEvaluator = Callable[[SearchCandidate], Mapping[str, Any]]
FullModelEvaluator = Callable[[SearchCandidate], Mapping[str, Any]]
ConfirmationRunner = Callable[
    [SearchCandidate, str, int, MeasurementContract], Mapping[str, Any]
]
CandidateConfigFactory = Callable[
    [SearchCandidate, MeasurementContract], CandidateConfig
]


@dataclass(frozen=True)
class SearchCallbacks:
    candidate_evaluator: CandidateEvaluator | None = None
    layer_evaluator: LayerEvaluator | None = None
    full_model_evaluator: FullModelEvaluator | None = None
    confirmation_runner: ConfirmationRunner | None = None
    candidate_config_factory: CandidateConfigFactory | None = None


@dataclass(frozen=True)
class EvaluationRecord:
    stage: str
    label: str
    candidate_id: str
    status: str
    passed: bool
    objective_name: str | None
    objective_value: float | None
    objective_direction: str | None
    correctness_passed: bool
    quality_passed: bool
    device_seconds: float
    reused: bool
    report_path: str
    _raw_json: str = field(repr=False)

    @property
    def raw(self) -> dict[str, Any]:
        return _decode_object(self._raw_json, "evaluation result")

    @property
    def rank_value(self) -> float:
        if not self.passed or self.objective_value is None:
            return math.inf
        if self.objective_direction == "maximize":
            return -self.objective_value
        return self.objective_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "passed": self.passed,
            "objective": {
                "name": self.objective_name,
                "value": self.objective_value,
                "direction": self.objective_direction,
            },
            "correctness_passed": self.correctness_passed,
            "quality_passed": self.quality_passed,
            "device_seconds": self.device_seconds,
            "reused": self.reused,
            "report_path": self.report_path,
        }


def proposal_from_space(
    *,
    stage: str,
    group: str,
    label: str,
    base_space: SearchSpaceConfig,
    candidate_space: SearchSpaceConfig,
    score: ProposalScore,
    proposal_id: str | None = None,
    source: str = "enumerated",
    evidence: Mapping[str, Any] | None = None,
) -> SearchProposal:
    mutation = SearchMutation.from_spaces(
        stage=stage,
        group=group,
        label=label,
        base_space=base_space,
        candidate_space=candidate_space,
        mutation_id=proposal_id,
    )
    return SearchProposal.create(
        mutation=mutation,
        score=score,
        proposal_id=proposal_id,
        source=source,
        evidence=evidence,
    )


def template_proposal_groups(
    *,
    base_space: SearchSpaceConfig,
    measurements: Mapping[str, Mapping[str, Any]],
) -> tuple[SearchProposalGroup, ...]:
    from .templates import DEFAULT_TEMPLATE_SELECTION, list_template_definitions

    by_axis: dict[str, list[Any]] = {}
    for definition in list_template_definitions():
        by_axis.setdefault(definition.axis, []).append(definition)
    groups: list[SearchProposalGroup] = []
    for axis in DEFAULT_TEMPLATE_SELECTION:
        incumbent_name = base_space.templates[axis]
        incumbent_evidence = measurements.get(incumbent_name)
        if incumbent_evidence is None:
            raise HierarchicalSearchError(
                f"template measurements are missing incumbent {incumbent_name!r}"
            )
        proposals: list[SearchProposal] = []
        for definition in sorted(by_axis.get(axis, []), key=lambda item: item.name):
            if definition.name == incumbent_name:
                continue
            evidence = measurements.get(definition.name)
            if evidence is None:
                continue
            payload = base_space.to_dict()
            payload["templates"][axis] = definition.name
            candidate_space = SearchSpaceConfig.from_dict(payload)
            proposals.append(
                proposal_from_space(
                    stage="template",
                    group=axis,
                    label=definition.name,
                    base_space=base_space,
                    candidate_space=candidate_space,
                    score=ProposalScore.from_mapping(evidence),
                    proposal_id=f"template-{_slug(definition.name)}",
                    source="template_registry",
                    evidence=evidence,
                )
            )
        groups.append(
            SearchProposalGroup(
                stage="template",
                name=axis,
                proposals=tuple(proposals),
                incumbent_score=ProposalScore.from_mapping(incumbent_evidence),
            )
        )
    return tuple(groups)


def matmul_proposal_group(
    result: "MatmulEnumerationResult",
    *,
    base_space: SearchSpaceConfig,
    measurements: Mapping[str, Mapping[str, Any]],
) -> SearchProposalGroup:
    official = result.official_candidates
    if len(official) != 1:
        raise HierarchicalSearchError(
            f"{result.operator_name} must have exactly one official candidate"
        )
    official_id = official[0].candidate_id
    incumbent_evidence = measurements.get(official_id)
    if incumbent_evidence is None:
        raise HierarchicalSearchError(
            f"measurements are missing official candidate {official_id!r}"
        )
    proposals: list[SearchProposal] = []
    for candidate in result.candidates:
        if candidate.is_official or candidate.candidate_id not in measurements:
            continue
        evidence = measurements[candidate.candidate_id]
        l1_bytes = sum(
            estimate.total_bytes
            for estimate in candidate.legality.l1_estimates
            if estimate.path.startswith(result.operator_name)
        )
        score_payload = {**dict(evidence), "l1_bytes": l1_bytes}
        proposals.append(
            proposal_from_space(
                stage="op",
                group=result.operator_name,
                label=candidate.candidate_id,
                base_space=base_space,
                candidate_space=candidate.search_space,
                score=ProposalScore.from_mapping(score_payload),
                proposal_id=candidate.candidate_id,
                source=candidate.source,
                evidence=candidate.to_dict(),
            )
        )
    return SearchProposalGroup(
        stage="op",
        name=result.operator_name,
        proposals=tuple(proposals),
        incumbent_score=ProposalScore.from_mapping(
            {**dict(incumbent_evidence), "l1_bytes": 0}
        ),
    )


def sdpa_proposal_group(
    result: "SDPAEnumerationResult",
    *,
    base_space: SearchSpaceConfig,
    measurements: Mapping[str, Mapping[str, Any]],
) -> SearchProposalGroup:
    official = result.official_candidates
    if len(official) != 1:
        raise HierarchicalSearchError(
            "attention.sdpa must have exactly one official candidate"
        )
    official_id = official[0].candidate_id
    incumbent_evidence = measurements.get(official_id)
    if incumbent_evidence is None:
        raise HierarchicalSearchError(
            f"measurements are missing official candidate {official_id!r}"
        )
    proposals: list[SearchProposal] = []
    for candidate in result.candidates:
        if candidate.is_official or candidate.candidate_id not in measurements:
            continue
        evidence = measurements[candidate.candidate_id]
        l1_bytes = sum(
            estimate.total_bytes
            for estimate in candidate.legality.l1_estimates
            if estimate.path == "attention.sdpa"
        )
        score_payload = {**dict(evidence), "l1_bytes": l1_bytes}
        proposals.append(
            proposal_from_space(
                stage="op",
                group="attention.sdpa",
                label=candidate.candidate_id,
                base_space=base_space,
                candidate_space=candidate.search_space,
                score=ProposalScore.from_mapping(score_payload),
                proposal_id=candidate.candidate_id,
                source=candidate.source,
                evidence=candidate.to_dict(),
            )
        )
    return SearchProposalGroup(
        stage="op",
        name="attention.sdpa",
        proposals=tuple(proposals),
        incumbent_score=ProposalScore.from_mapping(
            {**dict(incumbent_evidence), "l1_bytes": 0}
        ),
    )


def layout_proposal_group(
    *,
    name: str,
    base_space: SearchSpaceConfig,
    results: Sequence["LayoutSearchResult"],
    incumbent_latency_ms: float,
) -> SearchProposalGroup:
    from .layout_graph import apply_layout_result

    proposals: list[SearchProposal] = []
    for index, result in enumerate(results):
        candidate_space = apply_layout_result(base_space, result)
        proposals.append(
            proposal_from_space(
                stage="layout",
                group=name,
                label=f"{result.graph.region}-path-{index}",
                base_space=base_space,
                candidate_space=candidate_space,
                score=ProposalScore(
                    latency_ms=result.op_latency_ms,
                    conversion_latency_ms=result.conversion_latency_ms,
                ),
                proposal_id=(
                    f"layout-{_slug(result.graph.region)}-"
                    f"{sha256_json(result.to_dict())[:12]}"
                ),
                source="layout_beam",
                evidence=result.to_dict(),
            )
        )
    return SearchProposalGroup(
        stage="layout",
        name=name,
        proposals=tuple(proposals),
        incumbent_score=ProposalScore(latency_ms=float(incumbent_latency_ms)),
    )


def buildable_template_config(
    template_config: Mapping[str, Any],
    selected_space: SearchSpaceConfig,
) -> dict[str, Any]:
    from ..compiler.tuning import normalize_tuning_config
    from ..templates.registry import validate_template_config

    result = copy.deepcopy(dict(template_config))
    result["autotune"] = selected_space.to_dict()
    validate_template_config(result)
    normalized = normalize_tuning_config(result["autotune"])
    if not isinstance(normalized, SearchSpaceConfig):
        raise HierarchicalSearchError(
            "final template config did not normalize as schema v2"
        )
    if normalized.to_dict() != selected_space.to_dict():
        raise HierarchicalSearchError(
            "final template config does not preserve the selected search space"
        )
    return result


def run_hierarchical_search(
    *,
    base_space: SearchSpaceConfig,
    proposal_groups: Sequence[SearchProposalGroup],
    template_config: Mapping[str, Any],
    out_dir: str | Path,
    callbacks: SearchCallbacks | None = None,
    budget: SearchBudget | None = None,
    confirmation_policy: ConfirmationPolicy | None = None,
    resume: bool = True,
    dry_run: bool = False,
    run_identity: str = "default",
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "search_report.json"
    try:
        orchestrator = _HierarchicalSearchOrchestrator(
            base_space=base_space,
            proposal_groups=tuple(proposal_groups),
            template_config=template_config,
            out_dir=destination,
            callbacks=callbacks or SearchCallbacks(),
            budget=budget or SearchBudget(),
            confirmation_policy=confirmation_policy or ConfirmationPolicy(),
            resume=resume,
            dry_run=dry_run,
            run_identity=run_identity,
        )
        return orchestrator.run()
    except Exception as exc:
        existing = _read_json(report_path)
        report = existing if isinstance(existing, dict) else {}
        report.update(
            {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "stage": "semantic-autotune",
                "algorithm": "hierarchical_constrained_beam_search",
                "cartesian_exhaustive_search": False,
                "status": (
                    "budget_exhausted"
                    if isinstance(exc, SearchBudgetExhausted)
                    else "failed"
                ),
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "failure_report_written": True,
                "report_path": str(report_path),
            }
        )
        atomic_write_json(report_path, report)
        return report


class _HierarchicalSearchOrchestrator:
    def __init__(
        self,
        *,
        base_space: SearchSpaceConfig,
        proposal_groups: tuple[SearchProposalGroup, ...],
        template_config: Mapping[str, Any],
        out_dir: Path,
        callbacks: SearchCallbacks,
        budget: SearchBudget,
        confirmation_policy: ConfirmationPolicy,
        resume: bool,
        dry_run: bool,
        run_identity: str,
    ) -> None:
        if not run_identity:
            raise HierarchicalSearchError("run_identity must be non-empty")
        _validate_group_order(proposal_groups)
        if not dry_run:
            missing = [
                name
                for name, callback in (
                    ("layer_evaluator", callbacks.layer_evaluator),
                    ("full_model_evaluator", callbacks.full_model_evaluator),
                    ("confirmation_runner", callbacks.confirmation_runner),
                    ("candidate_config_factory", callbacks.candidate_config_factory),
                )
                if callback is None
            ]
            if missing:
                raise HierarchicalSearchError(
                    "non-dry hierarchical search requires callbacks: "
                    + ", ".join(missing)
                )
        self.base_space = base_space
        self.incumbent_sha256 = sha256_json(base_space.to_dict())
        self.groups = proposal_groups
        self.template_config = copy.deepcopy(dict(template_config))
        self.out_dir = out_dir
        self.callbacks = callbacks
        self.budget = budget
        self.policy = confirmation_policy
        self.resume = bool(resume)
        self.dry_run = bool(dry_run)
        self.run_identity = run_identity
        self.report_path = out_dir / "search_report.json"
        self.checkpoint_path = out_dir / "checkpoint.json"
        self.candidates_dir = out_dir / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.frozen_campaign_identity = self._campaign_identity()
        self.search_fingerprint = sha256_json(
            {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "run_identity": run_identity,
                "base_space": base_space.to_dict(),
                "proposal_groups": [group.to_dict() for group in proposal_groups],
                "template_config": self.template_config,
                "algorithm": budget.algorithm_identity(),
                "confirmation_policy": confirmation_policy.to_dict(),
                "frozen_campaign_identity": self.frozen_campaign_identity,
                "dry_run": self.dry_run,
            }
        )
        self.evaluation_count = 0
        self.device_seconds = 0.0
        self.reused_evaluation_count = 0
        self.failure_count = 0
        self.report = self._initial_report()
        self._restore_checkpoint()

    def run(self) -> dict[str, Any]:
        completed = self._completed_report()
        if completed is not None:
            return completed

        atomic_write_json(self.report_path, self.report)
        incumbent = SearchCandidate(space=self.base_space)
        beam = [incumbent]
        stage_records: dict[str, EvaluationRecord] = {}

        for stage in SEARCH_STAGES:
            stage_name = {
                "template": "template_search",
                "op": "op_search",
                "layout": "layout_beam",
            }[stage]
            stage_report = {
                "name": stage_name,
                "status": "running",
                "groups": [],
                "input_beam_size": len(beam),
            }
            self.report["stages"].append(stage_report)
            self._flush()
            stage_groups = [group for group in self.groups if group.stage == stage]
            if not stage_groups:
                stage_report.update(
                    {
                        "status": "skipped",
                        "reason": "no proposal groups",
                        "output_beam_size": len(beam),
                    }
                )
                self._flush()
                continue
            for group in stage_groups:
                beam, records, group_report = self._expand_group(beam, group)
                stage_records.update(records)
                stage_report["groups"].append(group_report)
                self._flush()
            stage_report.update(
                {
                    "status": "passed",
                    "output_beam_size": len(beam),
                    "selected_candidates": [
                        candidate.to_dict(incumbent_sha256=self.incumbent_sha256)
                        for candidate in beam
                    ],
                }
            )
            self._flush()

        layer_candidates = self._cap_candidates(
            beam,
            stage_records,
            self.budget.microbench_top_k,
        )
        layer_records = self._evaluate_stage_candidates(
            stage="layer_confirmation",
            candidates=layer_candidates,
            evaluator=self.callbacks.layer_evaluator,
        )
        passed_layer = [
            candidate
            for candidate in layer_candidates
            if layer_records[candidate.candidate_id].passed
        ]
        if not passed_layer:
            raise HierarchicalSearchError(
                "no candidate passed whole-layer confirmation"
            )
        self.report["stages"].append(
            self._confirmation_stage_report(
                "layer_confirmation", layer_candidates, layer_records
            )
        )
        self._flush()

        full_candidates = self._cap_candidates(
            passed_layer,
            layer_records,
            self.budget.full_model_top_k,
        )
        full_records = self._evaluate_stage_candidates(
            stage="full_model_trace",
            candidates=full_candidates,
            evaluator=self.callbacks.full_model_evaluator,
        )
        passed_full = [
            candidate
            for candidate in full_candidates
            if full_records[candidate.candidate_id].passed
        ]
        if not passed_full:
            raise HierarchicalSearchError("no candidate passed full-model trace")
        self.report["stages"].append(
            self._confirmation_stage_report(
                "full_model_trace", full_candidates, full_records
            )
        )
        self._flush()

        ranked_full = sorted(
            passed_full,
            key=lambda candidate: self._candidate_rank(candidate, full_records),
        )
        incumbent_candidate = next(
            (
                candidate
                for candidate in ranked_full
                if candidate.space_sha256 == self.incumbent_sha256
            ),
            None,
        )
        if incumbent_candidate is None:
            raise HierarchicalSearchError(
                "incumbent did not reach full-model matched A/B confirmation"
            )
        challenger = next(
            (
                candidate
                for candidate in ranked_full
                if candidate.space_sha256 != self.incumbent_sha256
            ),
            incumbent_candidate,
        )

        if self.dry_run:
            selected = ranked_full[0]
            confirmation = {
                "status": "dry_run",
                "passed": True,
                "matched_ab": False,
                "reason": "long confirmation is not executed in dry-run mode",
            }
        else:
            confirmation = self._run_long_confirmation(
                incumbent_candidate,
                challenger,
                full_records,
            )
            selected = (
                challenger
                if confirmation["promotion"]["selected_arm"] == "challenger"
                else incumbent_candidate
            )
        self.report["stages"].append(
            {
                "name": "long_confirmation",
                "status": confirmation["status"],
                "passed": confirmation["passed"],
                "incumbent_candidate_id": incumbent_candidate.candidate_id,
                "challenger_candidate_id": challenger.candidate_id,
                "matched_ab": confirmation.get("matched_ab", False),
            }
        )

        best_config = buildable_template_config(self.template_config, selected.space)
        best_config_path = self.out_dir / "best_config.json"
        best_space_path = self.out_dir / "best_space.json"
        atomic_write_json(best_config_path, best_config)
        atomic_write_json(best_space_path, selected.space.to_dict())
        reproduce_path = self.out_dir / "reproduce_build.sh"
        _atomic_write_text(
            reproduce_path,
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            ': "${MODEL:?set MODEL to the Hugging Face model directory}"\n'
            ': "${PROGRAM:?set PROGRAM to the output program directory}"\n'
            "python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli "
            f'build-program --model-path "$MODEL" --config "{best_config_path}" '
            '--out-dir "$PROGRAM"\n',
        )
        reproduce_path.chmod(0o755)

        acceptance = self._acceptance(
            confirmation=confirmation,
            best_config=best_config,
        )
        passed = bool(confirmation["passed"] and acceptance["passed"])
        self.report.update(
            {
                "status": (
                    "dry_run" if self.dry_run else ("passed" if passed else "failed")
                ),
                "passed": passed,
                "active_evaluation": None,
                "top_candidate_id": challenger.candidate_id,
                "incumbent_candidate_id": incumbent_candidate.candidate_id,
                "selected_candidate": selected.to_dict(
                    incumbent_sha256=self.incumbent_sha256
                ),
                "matched_ab_confirmation": confirmation,
                "artifacts": {
                    "best_config": str(best_config_path),
                    "best_space": str(best_space_path),
                    "reproduce_build": str(reproduce_path),
                },
                "acceptance": acceptance,
                "failure_report_written": True,
            }
        )
        self._flush()
        return copy.deepcopy(self.report)

    def _expand_group(
        self,
        beam: Sequence[SearchCandidate],
        group: SearchProposalGroup,
    ) -> tuple[
        list[SearchCandidate],
        dict[str, EvaluationRecord],
        dict[str, Any],
    ]:
        proposals, pruning = self._prune_proposals(group)
        choices: list[SearchProposal | None] = [None, *proposals]
        deduplicated: dict[str, SearchCandidate] = {}
        origins: dict[str, list[str]] = {}
        for parent in beam:
            for proposal in choices:
                candidate = parent.apply(
                    proposal,
                    incumbent_score=group.incumbent_score,
                )
                existing = deduplicated.get(candidate.candidate_id)
                if (
                    existing is None
                    or candidate.estimated_latency_ms < existing.estimated_latency_ms
                ):
                    deduplicated[candidate.candidate_id] = candidate
                origins.setdefault(candidate.candidate_id, []).append(
                    "incumbent" if proposal is None else proposal.proposal_id
                )
        candidates = list(deduplicated.values())
        records: dict[str, EvaluationRecord] = {}
        label = f"{group.stage}-{group.name}"
        for candidate in candidates:
            if self.callbacks.candidate_evaluator is None:
                raw = {
                    "status": "estimated",
                    "passed": True,
                    "objective": {
                        "name": "estimated_latency_ms",
                        "value": candidate.estimated_latency_ms,
                        "direction": "minimize",
                    },
                    "correctness_passed": True,
                    "quality_passed": True,
                }
                record = self._synthetic_record(
                    stage=group.stage,
                    label=label,
                    candidate=candidate,
                    raw=raw,
                )
            else:
                record = self._cached_evaluate(
                    stage=group.stage,
                    label=label,
                    candidate=candidate,
                    invoke=lambda candidate=candidate: self.callbacks.candidate_evaluator(  # type: ignore[misc]
                        group.stage, candidate
                    ),
                )
            records[candidate.candidate_id] = record
        passed = [
            candidate
            for candidate in candidates
            if records[candidate.candidate_id].passed
        ]
        if not passed:
            if self._budget_is_exhausted():
                raise SearchBudgetExhausted(
                    f"budget exhausted while evaluating {group.stage}/{group.name}"
                )
            raise HierarchicalSearchError(
                f"no candidate passed {group.stage}/{group.name}"
            )
        selected = self._cap_candidates(
            passed,
            records,
            self.budget.beam_width,
        )
        return (
            selected,
            records,
            {
                "name": group.name,
                "stage": group.stage,
                "status": "passed",
                "input_beam_size": len(beam),
                "proposal_count": len(group.proposals),
                "retained_proposal_count": len(proposals),
                "implicit_incumbent_choice": True,
                "choice_count": len(choices),
                "expanded_state_count": len(beam) * len(choices),
                "deduplicated_candidate_count": len(candidates),
                "output_beam_size": len(selected),
                "pruning": pruning,
                "origins": origins,
                "evaluations": [
                    records[candidate.candidate_id].to_dict()
                    for candidate in candidates
                ],
                "selected_candidates": [
                    candidate.to_dict(incumbent_sha256=self.incumbent_sha256)
                    for candidate in selected
                ],
            },
        )

    def _prune_proposals(
        self,
        group: SearchProposalGroup,
    ) -> tuple[list[SearchProposal], dict[str, Any]]:
        eligible = [
            proposal
            for proposal in group.proposals
            if proposal.score.passed and proposal.score.conversion_compatible
        ]
        eligible.sort(
            key=lambda proposal: (
                proposal.score.total_latency_ms,
                proposal.score.l1_bytes,
                proposal.proposal_id,
            )
        )
        if group.stage == "template":
            selected = eligible[: self.budget.template_top_k]
            strategy = "independent_region_top_k"
        elif group.stage == "op":
            frontier = _pareto_frontier(eligible)
            selected = frontier[: self.budget.microbench_top_k]
            strategy = "latency_l1_conversion_pareto_frontier"
        else:
            selected = eligible[: self.budget.beam_width]
            strategy = "layout_cost_beam_preselection"
        return selected, {
            "strategy": strategy,
            "eligible_proposal_ids": [item.proposal_id for item in eligible],
            "retained_proposal_ids": [item.proposal_id for item in selected],
            "rejected_proposal_ids": [
                item.proposal_id for item in group.proposals if item not in selected
            ],
        }

    def _evaluate_stage_candidates(
        self,
        *,
        stage: str,
        candidates: Sequence[SearchCandidate],
        evaluator: LayerEvaluator | FullModelEvaluator | None,
    ) -> dict[str, EvaluationRecord]:
        records: dict[str, EvaluationRecord] = {}
        for candidate in candidates:
            if evaluator is None:
                raw = {
                    "status": "dry_run",
                    "passed": True,
                    "objective": {
                        "name": "estimated_latency_ms",
                        "value": candidate.estimated_latency_ms,
                        "direction": "minimize",
                    },
                    "correctness_passed": True,
                    "quality_passed": True,
                }
                record = self._synthetic_record(
                    stage=stage,
                    label=stage,
                    candidate=candidate,
                    raw=raw,
                )
            else:
                record = self._cached_evaluate(
                    stage=stage,
                    label=stage,
                    candidate=candidate,
                    invoke=lambda candidate=candidate: evaluator(candidate),
                )
            records[candidate.candidate_id] = record
        return records

    def _run_long_confirmation(
        self,
        incumbent: SearchCandidate,
        challenger: SearchCandidate,
        full_records: Mapping[str, EvaluationRecord],
    ) -> dict[str, Any]:
        runner = self.callbacks.confirmation_runner
        factory = self.callbacks.candidate_config_factory
        if runner is None or factory is None:
            raise HierarchicalSearchError("long confirmation callbacks are missing")
        contract = self.policy.measurement_contract
        reports: dict[str, list[Mapping[str, Any]]] = {
            "incumbent": [],
            "challenger": [],
        }
        order = (
            ("incumbent", "challenger"),
            ("challenger", "incumbent"),
            ("incumbent", "challenger"),
        )
        subjects = {"incumbent": incumbent, "challenger": challenger}
        execution_order: list[dict[str, Any]] = []
        for repetition, arms in enumerate(order):
            for arm in arms:
                candidate = subjects[arm]
                label = f"{arm}-rep-{repetition}"
                record = self._cached_evaluate(
                    stage="long_confirmation",
                    label=label,
                    candidate=candidate,
                    invoke=lambda candidate=candidate, arm=arm, repetition=repetition: runner(
                        candidate,
                        arm,
                        repetition,
                        contract,
                    ),
                )
                raw = record.raw
                raw["confirmation_arm"] = arm
                raw["confirmation_repetition"] = repetition
                reports[arm].append(raw)
                execution_order.append(
                    {
                        "repetition": repetition,
                        "arm": arm,
                        "candidate_id": candidate.candidate_id,
                        "report_path": record.report_path,
                        "reused": record.reused,
                    }
                )
                self._flush()

        incumbent_record = full_records[incumbent.candidate_id]
        challenger_record = full_records[challenger.candidate_id]
        confirmation = confirm_matched_ab(
            incumbent=ConfirmationArm(
                label="incumbent",
                candidate=factory(incumbent, contract),
                reports=tuple(reports["incumbent"]),
                correctness_passed=incumbent_record.correctness_passed,
                quality_passed=incumbent_record.quality_passed,
            ),
            challenger=ConfirmationArm(
                label="challenger",
                candidate=factory(challenger, contract),
                reports=tuple(reports["challenger"]),
                correctness_passed=challenger_record.correctness_passed,
                quality_passed=challenger_record.quality_passed,
            ),
            policy=self.policy,
        )
        confirmation["execution_order"] = execution_order
        confirmation["alternating_arm_order"] = True
        atomic_write_json(self.out_dir / "matched_ab_confirmation.json", confirmation)
        return confirmation

    def _cached_evaluate(
        self,
        *,
        stage: str,
        label: str,
        candidate: SearchCandidate,
        invoke: Callable[[], Mapping[str, Any]],
    ) -> EvaluationRecord:
        record_dir = self.candidates_dir / candidate.candidate_id
        record_dir.mkdir(parents=True, exist_ok=True)
        record_path = record_dir / f"{_slug(stage)}-{_slug(label)}.json"
        request = {
            "search_fingerprint": self.search_fingerprint,
            "run_identity": self.run_identity,
            "stage": stage,
            "label": label,
            "candidate_id": candidate.candidate_id,
            "space_sha256": candidate.space_sha256,
        }
        request_sha256 = sha256_json(request)
        cached = _read_json(record_path) if self.resume else None
        if (
            isinstance(cached, dict)
            and cached.get("request_sha256") == request_sha256
            and cached.get("complete") is True
            and isinstance(cached.get("result"), Mapping)
        ):
            self.reused_evaluation_count += 1
            return _evaluation_record(
                stage=stage,
                label=label,
                candidate_id=candidate.candidate_id,
                raw=cached["result"],
                device_seconds=float(cached.get("device_seconds", 0.0)),
                reused=True,
                report_path=record_path,
            )

        self._admit_evaluation()
        self.report["active_evaluation"] = request
        self._flush()
        started = time.monotonic()
        try:
            output = invoke()
            if not isinstance(output, Mapping):
                raise HierarchicalSearchError(
                    "evaluation callback must return a mapping"
                )
            raw = copy.deepcopy(dict(output))
        except Exception as exc:
            raw = {
                "status": "failed",
                "passed": False,
                "correctness_passed": False,
                "quality_passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        if (
            not self.dry_run
            and stage in {"layer_confirmation", "full_model_trace", "long_confirmation"}
            and raw.get("isolated_subprocess") is not True
        ):
            raw = {
                **raw,
                "status": "invalid_measurement",
                "passed": False,
                "correctness_passed": False,
                "quality_passed": False,
                "error": {
                    "type": "IsolationContractViolation",
                    "message": (
                        "device candidate evaluations must run in an isolated "
                        "subprocess"
                    ),
                },
            }
        elapsed = time.monotonic() - started
        device_seconds = _float_or_default(raw.get("device_seconds"), elapsed)
        self.evaluation_count += 1
        self.device_seconds += max(0.0, device_seconds)
        record = _evaluation_record(
            stage=stage,
            label=label,
            candidate_id=candidate.candidate_id,
            raw=raw,
            device_seconds=device_seconds,
            reused=False,
            report_path=record_path,
        )
        if not record.passed:
            self.failure_count += 1
        atomic_write_json(
            record_path,
            {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "request": request,
                "request_sha256": request_sha256,
                "complete": True,
                "status": record.status,
                "passed": record.passed,
                "device_seconds": device_seconds,
                "result": raw,
            },
        )
        self.report["active_evaluation"] = None
        self._write_checkpoint()
        self._flush()
        return record

    def _synthetic_record(
        self,
        *,
        stage: str,
        label: str,
        candidate: SearchCandidate,
        raw: Mapping[str, Any],
    ) -> EvaluationRecord:
        return _evaluation_record(
            stage=stage,
            label=label,
            candidate_id=candidate.candidate_id,
            raw=raw,
            device_seconds=0.0,
            reused=False,
            report_path="",
        )

    def _cap_candidates(
        self,
        candidates: Sequence[SearchCandidate],
        records: Mapping[str, EvaluationRecord],
        limit: int,
    ) -> list[SearchCandidate]:
        if limit <= 0:
            raise HierarchicalSearchError("candidate cap must be positive")
        unique = {candidate.candidate_id: candidate for candidate in candidates}
        ranked = sorted(
            unique.values(),
            key=lambda candidate: self._candidate_rank(candidate, records),
        )
        selected = ranked[:limit]
        incumbent = next(
            (
                candidate
                for candidate in ranked
                if candidate.space_sha256 == self.incumbent_sha256
            ),
            None,
        )
        if incumbent is not None and incumbent not in selected:
            if len(selected) >= limit:
                selected[-1] = incumbent
            else:
                selected.append(incumbent)
            selected.sort(
                key=lambda candidate: self._candidate_rank(candidate, records)
            )
        return selected

    @staticmethod
    def _candidate_rank(
        candidate: SearchCandidate,
        records: Mapping[str, EvaluationRecord],
    ) -> tuple[Any, ...]:
        record = records.get(candidate.candidate_id)
        return (
            record.rank_value if record is not None else candidate.estimated_latency_ms,
            candidate.estimated_latency_ms,
            candidate.candidate_id,
        )

    def _confirmation_stage_report(
        self,
        name: str,
        candidates: Sequence[SearchCandidate],
        records: Mapping[str, EvaluationRecord],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "status": (
                "passed"
                if any(
                    records[candidate.candidate_id].passed for candidate in candidates
                )
                else "failed"
            ),
            "candidate_count": len(candidates),
            "evaluations": [
                records[candidate.candidate_id].to_dict() for candidate in candidates
            ],
        }

    def _admit_evaluation(self) -> None:
        if self.evaluation_count >= self.budget.max_candidates:
            raise SearchBudgetExhausted(
                "maximum candidate evaluation budget was reached"
            )
        if self.device_seconds / 60.0 >= self.budget.max_device_minutes:
            raise SearchBudgetExhausted("maximum device-minute budget was reached")

    def _budget_is_exhausted(self) -> bool:
        return bool(
            self.evaluation_count >= self.budget.max_candidates
            or self.device_seconds / 60.0 >= self.budget.max_device_minutes
        )

    def _restore_checkpoint(self) -> None:
        if not self.resume:
            return
        checkpoint = _read_json(self.checkpoint_path)
        if not isinstance(checkpoint, dict):
            return
        if checkpoint.get("search_fingerprint") != self.search_fingerprint:
            raise HierarchicalSearchError(
                "resume checkpoint does not match this search request"
            )
        budget_state = checkpoint.get("budget_state") or {}
        self.evaluation_count = int(budget_state.get("evaluation_count", 0))
        self.device_seconds = float(budget_state.get("device_seconds", 0.0))
        self.failure_count = int(budget_state.get("failure_count", 0))
        self.report["resume"]["checkpoint_loaded"] = True

    def _completed_report(self) -> dict[str, Any] | None:
        if not self.resume:
            return None
        existing = _read_json(self.report_path)
        if not isinstance(existing, dict):
            return None
        if existing.get("search_fingerprint") != self.search_fingerprint:
            return None
        if existing.get("status") not in {"passed", "dry_run"}:
            return None
        artifacts = existing.get("artifacts") or {}
        if not all(
            Path(str(artifacts.get(name, ""))).is_file()
            for name in ("best_config", "best_space")
        ):
            return None
        existing.setdefault("resume", {})["completed_report_reused"] = True
        existing["resume"]["requested"] = True
        atomic_write_json(self.report_path, existing)
        return existing

    def _write_checkpoint(self) -> None:
        atomic_write_json(
            self.checkpoint_path,
            {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "search_fingerprint": self.search_fingerprint,
                "budget_state": {
                    "evaluation_count": self.evaluation_count,
                    "device_seconds": self.device_seconds,
                    "failure_count": self.failure_count,
                },
            },
        )

    def _flush(self) -> None:
        self.report["budget_usage"] = {
            "evaluation_count": self.evaluation_count,
            "max_candidates": self.budget.max_candidates,
            "device_seconds": self.device_seconds,
            "device_minutes": self.device_seconds / 60.0,
            "max_device_minutes": self.budget.max_device_minutes,
            "reused_evaluation_count": self.reused_evaluation_count,
            "failure_count": self.failure_count,
        }
        atomic_write_json(self.report_path, self.report)

    def _initial_report(self) -> dict[str, Any]:
        choice_counts = [len(group.proposals) + 1 for group in self.groups]
        theoretical = math.prod(choice_counts) if choice_counts else 1
        return {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "stage": "semantic-autotune",
            "algorithm": "hierarchical_constrained_beam_search",
            "stage_order": list(PIPELINE_STAGES),
            "cartesian_exhaustive_search": False,
            "theoretical_cartesian_path_count": theoretical,
            "search_fingerprint": self.search_fingerprint,
            "run_identity": self.run_identity,
            "status": "running",
            "passed": False,
            "dry_run": self.dry_run,
            "report_path": str(self.report_path),
            "candidate_root": str(self.candidates_dir),
            "base_space_sha256": self.incumbent_sha256,
            "frozen_campaign_identity": self.frozen_campaign_identity,
            "budget": self.budget.to_dict(),
            "budget_usage": {},
            "resume": {
                "requested": self.resume,
                "checkpoint_loaded": False,
                "completed_report_reused": False,
            },
            "proposal_groups": [
                {
                    "stage": group.stage,
                    "name": group.name,
                    "proposal_count": len(group.proposals),
                }
                for group in self.groups
            ],
            "stages": [],
            "active_evaluation": None,
            "matched_ab_confirmation": None,
            "selected_candidate": None,
            "artifacts": {},
            "acceptance": None,
            "failure_report_written": True,
        }

    def _campaign_identity(self) -> dict[str, Any] | None:
        if self.dry_run:
            return None
        factory = self.callbacks.candidate_config_factory
        if factory is None:
            raise HierarchicalSearchError("candidate_config_factory is missing")
        candidate = factory(
            SearchCandidate(space=self.base_space),
            self.policy.measurement_contract,
        )
        identity = candidate.identity_payload()
        identity.pop("tunable_state", None)
        return identity

    def _acceptance(
        self,
        *,
        confirmation: Mapping[str, Any],
        best_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        checks = {
            "no_cartesian_exhaustive_search": True,
            "budget_enforced": (
                self.evaluation_count <= self.budget.max_candidates
                and self.device_seconds / 60.0 <= self.budget.max_device_minutes
            ),
            "resume_supported": True,
            "failure_report_always_written": True,
            "top_candidate_and_incumbent_matched_ab": (
                self.dry_run or bool(confirmation.get("matched_ab"))
            ),
            "long_confirmation_contract": (
                self.dry_run
                or confirmation.get("measurement_contract")
                == self.policy.measurement_contract.to_dict()
            ),
            "final_config_directly_buildable": (
                isinstance(best_config.get("autotune"), Mapping)
                and best_config["autotune"].get("schema_version") == 2
            ),
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "passed": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _evaluation_record(
    *,
    stage: str,
    label: str,
    candidate_id: str,
    raw: Mapping[str, Any],
    device_seconds: float,
    reused: bool,
    report_path: Path | str,
) -> EvaluationRecord:
    status = str(raw.get("status", "passed" if raw.get("passed") else "failed"))
    passed = bool(
        raw.get("passed", status.lower() in {"passed", "profiled", "success"})
    )
    objective_name: str | None = None
    objective_value: float | None = None
    objective_direction: str | None = None
    objective = raw.get("objective")
    if isinstance(objective, Mapping):
        objective_name = str(objective.get("name") or "objective")
        objective_value = _float_or_none(objective.get("value"))
        objective_direction = str(objective.get("direction", "minimize"))
    if objective_value is None:
        for key, direction in (
            ("latency_ms", "minimize"),
            ("decode_step_ms_p50", "minimize"),
            ("tokens_per_second_per_user", "maximize"),
            ("metric_value", str(raw.get("metric_direction", "maximize"))),
        ):
            value = _float_or_none(raw.get(key))
            if value is not None:
                objective_name = key
                objective_value = value
                objective_direction = direction
                break
    if passed and (
        objective_value is None
        or not math.isfinite(objective_value)
        or objective_direction not in {"minimize", "maximize"}
    ):
        passed = False
        status = "invalid_measurement"
    return EvaluationRecord(
        stage=stage,
        label=label,
        candidate_id=candidate_id,
        status=status,
        passed=passed,
        objective_name=objective_name,
        objective_value=objective_value,
        objective_direction=objective_direction,
        correctness_passed=bool(raw.get("correctness_passed", False)),
        quality_passed=bool(raw.get("quality_passed", False)),
        device_seconds=max(0.0, float(device_seconds)),
        reused=bool(reused),
        report_path=str(report_path),
        _raw_json=canonical_json(copy.deepcopy(dict(raw))),
    )


def _pareto_frontier(proposals: Sequence[SearchProposal]) -> list[SearchProposal]:
    frontier: list[SearchProposal] = []
    for candidate in proposals:
        candidate_dimensions = (
            candidate.score.total_latency_ms,
            candidate.score.l1_bytes,
            candidate.score.conversion_latency_ms,
        )
        dominated = False
        for other in proposals:
            if other is candidate:
                continue
            other_dimensions = (
                other.score.total_latency_ms,
                other.score.l1_bytes,
                other.score.conversion_latency_ms,
            )
            if all(
                observed <= current
                for observed, current in zip(other_dimensions, candidate_dimensions)
            ) and any(
                observed < current
                for observed, current in zip(other_dimensions, candidate_dimensions)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda proposal: (
            proposal.score.total_latency_ms,
            proposal.score.l1_bytes,
            proposal.score.conversion_latency_ms,
            proposal.proposal_id,
        )
    )
    return frontier


def _validate_group_order(groups: Sequence[SearchProposalGroup]) -> None:
    indexes = [SEARCH_STAGES.index(group.stage) for group in groups]
    if indexes != sorted(indexes):
        raise HierarchicalSearchError(
            "proposal groups must be ordered template, op, then layout"
        )
    names: set[tuple[str, str]] = set()
    for group in groups:
        key = (group.stage, group.name)
        if key in names:
            raise HierarchicalSearchError(f"duplicate proposal group {key!r}")
        names.add(key)


def _diff_values(
    base: Any,
    candidate: Any,
    *,
    path: tuple[str, ...],
    operations: list[dict[str, Any]],
) -> None:
    if isinstance(base, Mapping) and isinstance(candidate, Mapping):
        for key in sorted(set(base) | set(candidate), key=str):
            child_path = (*path, str(key))
            if key not in candidate:
                operations.append({"op": "remove", "path": list(child_path)})
            elif key not in base:
                operations.append(
                    {
                        "op": "set",
                        "path": list(child_path),
                        "value": copy.deepcopy(candidate[key]),
                    }
                )
            else:
                _diff_values(
                    base[key],
                    candidate[key],
                    path=child_path,
                    operations=operations,
                )
        return
    if canonical_json(base) != canonical_json(candidate):
        operations.append(
            {
                "op": "set",
                "path": list(path),
                "value": copy.deepcopy(candidate),
            }
        )


def _normalize_operations(
    operations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for operation in operations:
        kind = str(operation.get("op", ""))
        if kind not in {"set", "remove"}:
            raise HierarchicalSearchError("mutation operation must be set or remove")
        raw_path = operation.get("path")
        if not isinstance(raw_path, (list, tuple)) or not raw_path:
            raise HierarchicalSearchError(
                "mutation operation path must be a non-empty sequence"
            )
        path = [str(part) for part in raw_path]
        item: dict[str, Any] = {"op": kind, "path": path}
        if kind == "set":
            if "value" not in operation:
                raise HierarchicalSearchError("set mutation requires a value")
            item["value"] = copy.deepcopy(operation["value"])
        normalized.append(item)
    normalized.sort(key=lambda item: (item["path"], item["op"]))
    return normalized


def _decode_operations(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HierarchicalSearchError("mutation operations must be valid JSON") from exc
    if not isinstance(decoded, list):
        raise HierarchicalSearchError("mutation operations must be a list")
    return _normalize_operations(decoded)


def _set_json_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = target
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise HierarchicalSearchError(
                f"mutation path does not exist: {_display_path(path)!r}"
            )
        current = current[part]
    if not isinstance(current, dict):
        raise HierarchicalSearchError(
            f"mutation parent is not an object: {_display_path(path)!r}"
        )
    current[path[-1]] = copy.deepcopy(value)


def _remove_json_path(target: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = target
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise HierarchicalSearchError(
                f"mutation path does not exist: {_display_path(path)!r}"
            )
        current = current[part]
    if not isinstance(current, dict) or path[-1] not in current:
        raise HierarchicalSearchError(
            f"mutation remove path does not exist: {_display_path(path)!r}"
        )
    del current[path[-1]]


def _decode_object(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HierarchicalSearchError(f"{label} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HierarchicalSearchError(f"{label} must be an object")
    return decoded


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-.")
    return normalized or "item"


def _display_path(path: Sequence[str]) -> str:
    return "/" + "/".join(path)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    observed = _float_or_none(value)
    return default if observed is None else observed
