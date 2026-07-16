from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .legality import (
    DEFAULT_L1_SAFETY_FACTOR,
    DeviceDescriptor,
    LegalityReport,
    MatmulWorkload,
    WorkloadSpec,
    validate_candidate,
)
from .measurement import MeasurementCandidate
from .schema import PrecisionContract, canonical_json, sha256_json
from .space import MatmulProgramConfig, SearchSpaceConfig

MATMUL_ENUMERATOR_SCHEMA_VERSION = 1
MATMUL_OPERATORS = (
    "attention.qkv",
    "attention.o_proj",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
    "lm_head.shards",
)
_DRAM_SHARDED_RUNTIME_KIND = "ttnn_matmul_dram_sharded_program_config"


class MatmulEnumerationError(ValueError):
    """Raised when an operator cannot be represented by the enumerator."""


@dataclass(frozen=True)
class MatmulCandidate:
    candidate_id: str
    operator_name: str
    programs: tuple[MatmulProgramConfig, ...]
    worker_core_counts: tuple[int, ...]
    source: str
    is_official: bool
    search_space: SearchSpaceConfig = field(repr=False)
    legality: LegalityReport = field(repr=False)
    _derivation_json: str = field(default="{}", repr=False)

    @property
    def derivation(self) -> dict[str, Any]:
        value = _decode_object(self._derivation_json)
        return value

    @property
    def program_family(self) -> str:
        families = {program.program_family for program in self.programs}
        if len(families) != 1:
            raise MatmulEnumerationError(
                f"candidate {self.candidate_id} mixes program families"
            )
        return next(iter(families))

    def to_dict(self) -> dict[str, Any]:
        estimates = [
            estimate.to_dict()
            for estimate in self.legality.l1_estimates
            if estimate.path.startswith(self.operator_name)
        ]
        return {
            "candidate_id": self.candidate_id,
            "operator": self.operator_name,
            "source": self.source,
            "is_official": self.is_official,
            "program_family": self.program_family,
            "programs": [program.to_dict() for program in self.programs],
            "worker_core_counts": list(self.worker_core_counts),
            "derivation": self.derivation,
            "space_sha256": sha256_json(self.search_space.to_dict()),
            "legality": {
                "status": self.legality.status,
                "passed": self.legality.passed,
                "error_class": self.legality.error_class,
                "issues": [issue.to_dict() for issue in self.legality.issues],
                "l1_estimates": estimates,
            },
        }


@dataclass(frozen=True)
class MatmulEnumerationResult:
    operator_name: str
    workloads: tuple[MatmulWorkload, ...]
    candidates: tuple[MatmulCandidate, ...]
    rejected: tuple[dict[str, Any], ...]
    precision_contract_hash: str
    official_program_sha256: str
    schema_version: int = MATMUL_ENUMERATOR_SCHEMA_VERSION

    @property
    def official_candidates(self) -> tuple[MatmulCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate.is_official
        )

    @property
    def status(self) -> str:
        if len(self.official_candidates) != 1:
            return "official_missing"
        if len(self.candidates) < 2:
            return "insufficient_legal_candidates"
        return "passed"

    def candidate(self, candidate_id: str) -> MatmulCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise MatmulEnumerationError(f"unknown candidate: {candidate_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "operator": self.operator_name,
            "workloads": [workload.to_dict() for workload in self.workloads],
            "precision_contract_hash": self.precision_contract_hash,
            "official_program_sha256": self.official_program_sha256,
            "official_candidate_ids": [
                candidate.candidate_id for candidate in self.official_candidates
            ],
            "legal_candidate_count": len(self.candidates),
            "rejected_candidate_count": len(self.rejected),
            "program_families": sorted(
                {candidate.program_family for candidate in self.candidates}
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [copy.deepcopy(item) for item in self.rejected],
        }


def enumerate_matmul_programs(
    *,
    operator_name: str,
    base_space: SearchSpaceConfig,
    workload: WorkloadSpec,
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    safety_factor: float = DEFAULT_L1_SAFETY_FACTOR,
    max_proposals: int = 64,
) -> MatmulEnumerationResult:
    if operator_name not in MATMUL_OPERATORS:
        raise MatmulEnumerationError(
            f"operator must be one of {list(MATMUL_OPERATORS)}"
        )
    if max_proposals < 2:
        raise MatmulEnumerationError("max_proposals must be at least two")
    operator = base_space.operators.get(operator_name)
    if not isinstance(operator, Mapping) or operator.get("kind") != "matmul":
        raise MatmulEnumerationError(
            f"search space has no matmul operator {operator_name!r}"
        )
    raw_programs = operator.get("programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        raise MatmulEnumerationError(
            f"operator {operator_name!r} has no official programs"
        )
    official_programs = tuple(
        MatmulProgramConfig.from_dict(program) for program in raw_programs
    )
    workloads = _operator_workloads(workload, operator_name)
    if len(official_programs) != len(workloads):
        raise MatmulEnumerationError(
            f"operator {operator_name!r} has {len(official_programs)} programs "
            f"for {len(workloads)} workloads"
        )
    official_hash = sha256_json(
        [program.to_runtime_descriptor() for program in official_programs]
    )
    proposals = _program_proposals(
        workloads=workloads,
        official_programs=official_programs,
        device=device,
        max_proposals=max_proposals,
    )

    legal: list[MatmulCandidate] = []
    rejected: list[dict[str, Any]] = []
    for source, programs, derivation in proposals:
        candidate_space = _replace_programs(
            base_space,
            operator_name=operator_name,
            programs=programs,
        )
        program_payload = [program.to_dict() for program in programs]
        candidate_id = _candidate_id(
            operator_name,
            program_payload,
            workloads=workloads,
            device=device,
        )
        report = validate_candidate(
            candidate_space,
            workload,
            device,
            candidate_id=candidate_id,
            safety_factor=safety_factor,
        )
        official = canonical_json(program_payload) == canonical_json(
            [program.to_dict() for program in official_programs]
        )
        worker_counts = tuple(
            _worker_core_count(shape, program)
            for shape, program in zip(workloads, programs)
        )
        if report.passed:
            legal.append(
                MatmulCandidate(
                    candidate_id=candidate_id,
                    operator_name=operator_name,
                    programs=programs,
                    worker_core_counts=worker_counts,
                    source=source,
                    is_official=official,
                    search_space=candidate_space,
                    legality=report,
                    _derivation_json=canonical_json(derivation),
                )
            )
        else:
            target_estimates = [
                estimate.to_dict()
                for estimate in report.l1_estimates
                if estimate.path.startswith(operator_name)
            ]
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_official": official,
                    "programs": program_payload,
                    "worker_core_counts": list(worker_counts),
                    "derivation": copy.deepcopy(derivation),
                    "legality": {
                        "status": report.status,
                        "passed": report.passed,
                        "error_class": report.error_class,
                        "issues": [issue.to_dict() for issue in report.issues],
                        "l1_estimates": target_estimates,
                    },
                }
            )

    result = MatmulEnumerationResult(
        operator_name=operator_name,
        workloads=workloads,
        candidates=tuple(legal),
        rejected=tuple(rejected),
        precision_contract_hash=precision_contract.hash,
        official_program_sha256=official_hash,
    )
    if len(result.official_candidates) != 1:
        raise MatmulEnumerationError(
            f"official config for {operator_name!r} did not survive legality"
        )
    return result


def enumerate_all_matmul_programs(
    *,
    base_space: SearchSpaceConfig,
    workload: WorkloadSpec,
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    safety_factor: float = DEFAULT_L1_SAFETY_FACTOR,
    max_proposals_per_operator: int = 64,
) -> dict[str, Any]:
    results = [
        enumerate_matmul_programs(
            operator_name=operator_name,
            base_space=base_space,
            workload=workload,
            device=device,
            precision_contract=precision_contract,
            safety_factor=safety_factor,
            max_proposals=max_proposals_per_operator,
        )
        for operator_name in MATMUL_OPERATORS
    ]
    passed = all(result.status == "passed" for result in results)
    return {
        "schema_version": MATMUL_ENUMERATOR_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "precision_contract_hash": precision_contract.hash,
        "device": device.to_dict(),
        "operator_count": len(results),
        "legal_candidate_count": sum(len(result.candidates) for result in results),
        "rejected_candidate_count": sum(len(result.rejected) for result in results),
        "operators": {result.operator_name: result.to_dict() for result in results},
    }


def select_matmul_microbenchmark_winner(
    enumeration: MatmulEnumerationResult,
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    statistic: str = "p50",
) -> dict[str, Any]:
    if statistic not in {"mean", "p50", "p90"}:
        raise MatmulEnumerationError(
            "winner statistic must be one of 'mean', 'p50', or 'p90'"
        )
    ranked: list[dict[str, Any]] = []
    for candidate in enumeration.candidates:
        measurement = measurements.get(candidate.candidate_id)
        if not isinstance(measurement, Mapping):
            continue
        statistics = measurement.get("statistics")
        if measurement.get("status") != "passed" or not isinstance(statistics, Mapping):
            continue
        value = statistics.get(statistic)
        if value is None:
            continue
        score = float(value)
        if not math.isfinite(score) or score <= 0:
            continue
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "is_official": candidate.is_official,
                "program_family": candidate.program_family,
                "latency_ms": score,
                "statistic": statistic,
                "measurement_cache_key": ((measurement.get("cache") or {}).get("key")),
            }
        )
    ranked.sort(key=lambda item: (item["latency_ms"], item["candidate_id"]))
    winner = ranked[0] if ranked else None
    return {
        "schema_version": MATMUL_ENUMERATOR_SCHEMA_VERSION,
        "status": "selected" if winner is not None else "no_measurements",
        "operator": enumeration.operator_name,
        "selection_scope": "representative_microbenchmark_only",
        "promotion_allowed": False,
        "statistic": statistic,
        "winner": copy.deepcopy(winner),
        "ranked": ranked,
        "missing_measurement_candidate_ids": [
            candidate.candidate_id
            for candidate in enumeration.candidates
            if candidate.candidate_id not in measurements
        ],
    }


def rank_matmul_measurement_candidates(
    enumeration: MatmulEnumerationResult,
) -> tuple[MeasurementCandidate, ...]:
    """Rank legal MatMul candidates before spending device measurements."""

    ranked: list[MeasurementCandidate] = []
    for candidate in enumeration.candidates:
        workload_scores: list[dict[str, Any]] = []
        analytical_score = 0.0
        for workload, program, worker_count in zip(
            enumeration.workloads,
            candidate.programs,
            candidate.worker_core_counts,
        ):
            m_tiles = max(1, math.ceil(workload.m / 32))
            k_tiles = max(1, math.ceil(workload.k / 32))
            n_tiles = max(1, math.ceil(workload.n / 32))
            active_cores = max(1, min(int(worker_count), n_tiles))
            in0_block_w = max(1, int(program.parameters.get("in0_block_w", 1)))
            compute_tiles_per_core = (
                workload.batch_count * m_tiles * k_tiles * n_tiles / active_cores
            )
            k_block_count = math.ceil(k_tiles / in0_block_w)
            score = compute_tiles_per_core + 0.05 * k_block_count
            analytical_score += score
            workload_scores.append(
                {
                    "name": workload.name,
                    "m_tiles": m_tiles,
                    "k_tiles": k_tiles,
                    "n_tiles": n_tiles,
                    "active_cores": active_cores,
                    "in0_block_w": in0_block_w,
                    "compute_tiles_per_core": compute_tiles_per_core,
                    "k_block_count": k_block_count,
                    "score": score,
                }
            )
        l1_bytes = sum(
            estimate.total_bytes
            for estimate in candidate.legality.l1_estimates
            if estimate.path.startswith(enumeration.operator_name)
        )
        ranked.append(
            MeasurementCandidate.create(
                candidate_id=candidate.candidate_id,
                operator_name=enumeration.operator_name,
                candidate_kind="matmul",
                analytical_score=analytical_score,
                l1_bytes=l1_bytes,
                source=candidate.source,
                is_incumbent=candidate.is_official,
                metadata={
                    "ranking_model": "tile_work_per_active_core_plus_k_block_overhead",
                    "workloads": workload_scores,
                    "program_family": candidate.program_family,
                    "candidate": candidate.to_dict(),
                },
            )
        )
    ranked.sort(
        key=lambda item: (
            item.analytical_score,
            item.l1_bytes,
            not item.is_incumbent,
            item.candidate_id,
        )
    )
    return tuple(ranked)


def _program_proposals(
    *,
    workloads: tuple[MatmulWorkload, ...],
    official_programs: tuple[MatmulProgramConfig, ...],
    device: DeviceDescriptor,
    max_proposals: int,
) -> list[tuple[str, tuple[MatmulProgramConfig, ...], dict[str, Any]]]:
    proposals: list[tuple[str, tuple[MatmulProgramConfig, ...], dict[str, Any]]] = [
        (
            "official",
            official_programs,
            {
                "rule": "frozen_official_config",
                "shape_derived": False,
            },
        )
    ]
    if not all(_supports_dram_sharded(shape) for shape in workloads):
        return proposals

    common_k_tiles = math.gcd(*(shape.k // 32 for shape in workloads))
    in0_blocks = _bounded_divisors(common_k_tiles, upper_bound=32)
    in0_blocks.update(
        int(program.parameters["in0_block_w"])
        for program in official_programs
        if "in0_block_w" in program.parameters
    )
    worker_targets = _worker_core_targets(device)
    worker_targets.update(
        _worker_core_count(shape, program)
        for shape, program in zip(workloads, official_programs)
    )
    seen = {canonical_json([program.to_dict() for program in official_programs])}
    for in0_block_w in sorted(in0_blocks):
        for worker_target in sorted(worker_targets):
            programs = tuple(
                MatmulProgramConfig.from_runtime_descriptor(
                    {
                        "kind": _DRAM_SHARDED_RUNTIME_KIND,
                        "in0_block_w": in0_block_w,
                        "per_core_M": 1,
                        "per_core_N": max(
                            1, math.ceil((shape.n // 32) / worker_target)
                        ),
                    }
                )
                for shape in workloads
            )
            key = canonical_json([program.to_dict() for program in programs])
            if key in seen:
                continue
            seen.add(key)
            proposals.append(
                (
                    "shape_derived",
                    programs,
                    {
                        "rule": "tile_divisibility_and_worker_budget",
                        "shape_derived": True,
                        "k_tiles": [shape.k // 32 for shape in workloads],
                        "n_tiles": [shape.n // 32 for shape in workloads],
                        "in0_block_w": in0_block_w,
                        "requested_worker_cores": worker_target,
                        "device_worker_core_limit": device.worker_core_count,
                        "device_compute_grid": device.compute_grid.to_list(),
                    },
                )
            )
            if len(proposals) >= max_proposals:
                return proposals
    return proposals


def _replace_programs(
    base_space: SearchSpaceConfig,
    *,
    operator_name: str,
    programs: Sequence[MatmulProgramConfig],
) -> SearchSpaceConfig:
    payload = base_space.to_dict()
    operator = payload["operators"].get(operator_name)
    if not isinstance(operator, dict):
        raise MatmulEnumerationError(f"operator disappeared: {operator_name}")
    operator["programs"] = [program.to_dict() for program in programs]
    return SearchSpaceConfig.from_dict(payload)


def _operator_workloads(
    workload: WorkloadSpec,
    operator_name: str,
) -> tuple[MatmulWorkload, ...]:
    if operator_name == "lm_head.shards":
        matches = tuple(
            shape
            for shape in workload.matmuls
            if shape.name.startswith("lm_head.shards[")
        )
    else:
        matches = tuple(
            shape for shape in workload.matmuls if shape.name == operator_name
        )
    if not matches:
        raise MatmulEnumerationError(
            f"no representative workload for {operator_name!r}"
        )
    return matches


def _supports_dram_sharded(workload: MatmulWorkload) -> bool:
    return (
        workload.input_memory.layout == "width_sharded"
        and workload.output_memory.layout == "width_sharded"
        and workload.weight_memory.buffer == "dram"
        and workload.weight_memory.layout == "width_sharded"
        and workload.m == 32
        and workload.batch_count == 1
    )


def _worker_core_targets(device: DeviceDescriptor) -> set[int]:
    dram_width = int(device.dram_grid_width or 1)
    targets = {
        dram_width * multiplier
        for multiplier in (1, 2, 4, 8)
        if dram_width * multiplier <= device.worker_core_count
    }
    targets.add(min(device.worker_core_count, dram_width * device.compute_grid.y))
    return {target for target in targets if target > 0}


def _worker_core_count(
    workload: MatmulWorkload,
    program: MatmulProgramConfig,
) -> int:
    per_core_n = int(program.parameters.get("per_core_n", 1))
    return math.ceil((workload.n // 32) / per_core_n)


def _bounded_divisors(value: int, *, upper_bound: int) -> set[int]:
    return {
        divisor
        for divisor in range(1, min(value, upper_bound) + 1)
        if value % divisor == 0
    }


def _candidate_id(
    operator_name: str,
    programs: Sequence[Mapping[str, Any]],
    *,
    workloads: Sequence[MatmulWorkload],
    device: DeviceDescriptor,
) -> str:
    slug = "".join(
        character if character.isalnum() else "-" for character in operator_name.lower()
    ).strip("-")
    identity = {
        "operator": operator_name,
        "programs": list(programs),
        "workloads": [workload.to_dict() for workload in workloads],
        "device": device.to_dict(),
    }
    return f"{slug}-{sha256_json(identity)[:12]}"


def _decode_object(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise MatmulEnumerationError("candidate derivation must be an object")
    return payload
