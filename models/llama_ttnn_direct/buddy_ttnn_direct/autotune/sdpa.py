from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Mapping

from .legality import (
    DEFAULT_L1_SAFETY_FACTOR,
    DeviceDescriptor,
    LegalityReport,
    SDPAWorkload,
    WorkloadSpec,
    validate_candidate,
)
from .schema import PrecisionContract, canonical_json, sha256_json
from .space import CoreGrid, MemoryConfig, SDPAProgramConfig, SearchSpaceConfig

SDPA_ENUMERATOR_SCHEMA_VERSION = 1
SDPA_OPERATOR = "attention.sdpa"
_PROGRAM_GRID_PATH = "attention.sdpa_program_config.core_grid"
_KERNEL_OUTPUT_MEMORY_PATH = "attention.sdpa_kernel_output_memory_config"


class SDPAEnumerationError(ValueError):
    """Raised when the SDPA search space cannot be enumerated safely."""


@dataclass(frozen=True)
class SDPACandidate:
    candidate_id: str
    program: SDPAProgramConfig
    kernel_output_memory: MemoryConfig
    post_sdpa_output_memory: MemoryConfig
    source: str
    is_official: bool
    search_space: SearchSpaceConfig = field(repr=False)
    legality: LegalityReport = field(repr=False)
    _derivation_json: str = field(default="{}", repr=False)

    @property
    def derivation(self) -> dict[str, Any]:
        value = json.loads(self._derivation_json)
        if not isinstance(value, dict):
            raise SDPAEnumerationError("candidate derivation must be an object")
        return value

    def runtime_config(self) -> dict[str, Any]:
        return {
            "program_config": self.program.to_runtime_descriptor(),
            "kernel_output_memory_config": (
                self.kernel_output_memory.to_runtime_descriptor()
            ),
            "post_sdpa_output_memory_config": (
                self.post_sdpa_output_memory.to_runtime_descriptor()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        estimates = [
            estimate.to_dict()
            for estimate in self.legality.l1_estimates
            if estimate.path == SDPA_OPERATOR
        ]
        return {
            "candidate_id": self.candidate_id,
            "operator": SDPA_OPERATOR,
            "source": self.source,
            "is_official": self.is_official,
            "program": self.program.to_dict(),
            "kernel_output_memory": self.kernel_output_memory.to_dict(),
            "post_sdpa_output_memory": self.post_sdpa_output_memory.to_dict(),
            "runtime_config": self.runtime_config(),
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
class SDPAEnumerationResult:
    workload: SDPAWorkload
    candidates: tuple[SDPACandidate, ...]
    rejected: tuple[dict[str, Any], ...]
    precision_contract_hash: str
    official_config_sha256: str
    proposal_count: int
    schema_version: int = SDPA_ENUMERATOR_SCHEMA_VERSION

    @property
    def official_candidates(self) -> tuple[SDPACandidate, ...]:
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

    def candidate(self, candidate_id: str) -> SDPACandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise SDPAEnumerationError(f"unknown candidate: {candidate_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "operator": SDPA_OPERATOR,
            "workload": self.workload.to_dict(),
            "precision_contract_hash": self.precision_contract_hash,
            "official_config_sha256": self.official_config_sha256,
            "official_candidate_ids": [
                candidate.candidate_id for candidate in self.official_candidates
            ],
            "proposal_count": self.proposal_count,
            "legal_candidate_count": len(self.candidates),
            "rejected_candidate_count": len(self.rejected),
            "searched_dimensions": {
                "grid": [
                    list(grid)
                    for grid in sorted(
                        {
                            tuple(candidate.program.grid.to_list())
                            for candidate in self.candidates
                        }
                    )
                ],
                "sub_core_grids": _unique_values(
                    candidate.program.sub_core_grids for candidate in self.candidates
                ),
                "q_chunk_size": sorted(
                    {candidate.program.q_chunk_size for candidate in self.candidates}
                ),
                "k_chunk_size": sorted(
                    {candidate.program.k_chunk_size for candidate in self.candidates}
                ),
                "max_cores_per_head_batch": sorted(
                    {
                        candidate.program.max_cores_per_head_batch
                        for candidate in self.candidates
                    }
                ),
                "kernel_output_memory": _unique_values(
                    candidate.kernel_output_memory.to_dict()
                    for candidate in self.candidates
                ),
            },
            "frozen_dimensions": {
                "exp_approx_mode": self.official_candidates[0].program.exp_approx_mode,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [copy.deepcopy(item) for item in self.rejected],
        }


def enumerate_sdpa_programs(
    *,
    base_space: SearchSpaceConfig,
    workload: WorkloadSpec,
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    safety_factor: float = DEFAULT_L1_SAFETY_FACTOR,
    max_proposals: int = 256,
) -> SDPAEnumerationResult:
    if max_proposals < 2:
        raise SDPAEnumerationError("max_proposals must be at least two")
    operator = base_space.operators.get(SDPA_OPERATOR)
    if not isinstance(operator, Mapping) or operator.get("kind") != "sdpa":
        raise SDPAEnumerationError("search space has no attention.sdpa operator")
    official_program = SDPAProgramConfig.from_dict(operator)
    official_kernel_memory = _required_memory(
        operator,
        "kernel_output_memory",
    )
    official_post_memory = _required_memory(operator, "output_memory")
    official_identity = _config_identity(
        official_program,
        official_kernel_memory,
        official_post_memory,
    )
    official_hash = sha256_json(official_identity)
    proposals = _program_proposals(
        official_program=official_program,
        official_kernel_memory=official_kernel_memory,
        official_post_memory=official_post_memory,
        workload=workload.sdpa,
        device=device,
        max_proposals=max_proposals,
    )

    legal: list[SDPACandidate] = []
    rejected: list[dict[str, Any]] = []
    for source, program, kernel_memory, post_memory, derivation in proposals:
        candidate_space = _replace_sdpa(
            base_space,
            program=program,
            kernel_output_memory=kernel_memory,
        )
        identity = _config_identity(program, kernel_memory, post_memory)
        candidate_id = _candidate_id(
            identity,
            workload=workload.sdpa,
            device=device,
        )
        report = validate_candidate(
            candidate_space,
            workload,
            device,
            candidate_id=candidate_id,
            safety_factor=safety_factor,
        )
        is_official = canonical_json(identity) == canonical_json(official_identity)
        if report.passed:
            legal.append(
                SDPACandidate(
                    candidate_id=candidate_id,
                    program=program,
                    kernel_output_memory=kernel_memory,
                    post_sdpa_output_memory=post_memory,
                    source=source,
                    is_official=is_official,
                    search_space=candidate_space,
                    legality=report,
                    _derivation_json=canonical_json(derivation),
                )
            )
        else:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_official": is_official,
                    **identity,
                    "derivation": copy.deepcopy(derivation),
                    "legality": {
                        "status": report.status,
                        "passed": report.passed,
                        "error_class": report.error_class,
                        "issues": [issue.to_dict() for issue in report.issues],
                        "l1_estimates": [
                            estimate.to_dict()
                            for estimate in report.l1_estimates
                            if estimate.path == SDPA_OPERATOR
                        ],
                    },
                }
            )

    result = SDPAEnumerationResult(
        workload=workload.sdpa,
        candidates=tuple(legal),
        rejected=tuple(rejected),
        precision_contract_hash=precision_contract.hash,
        official_config_sha256=official_hash,
        proposal_count=len(proposals),
    )
    if len(result.official_candidates) != 1:
        raise SDPAEnumerationError(
            "the exact official SDPA configuration did not survive enumeration"
        )
    return result


def _program_proposals(
    *,
    official_program: SDPAProgramConfig,
    official_kernel_memory: MemoryConfig,
    official_post_memory: MemoryConfig,
    workload: SDPAWorkload,
    device: DeviceDescriptor,
    max_proposals: int,
) -> list[
    tuple[
        str,
        SDPAProgramConfig,
        MemoryConfig,
        MemoryConfig,
        dict[str, Any],
    ]
]:
    grids = _grid_options(official_program.grid, device)
    q_chunks = _ordered_unique([official_program.q_chunk_size, 0, 32, 64])
    k_chunks = _k_chunk_options(official_program.k_chunk_size, workload.cache_len)
    max_core_options = _ordered_unique(
        [official_program.max_cores_per_head_batch, 4, 8, 16, 32]
    )
    kernel_memories = _memory_options(
        official_kernel_memory,
        official_post_memory,
    )
    official_key = canonical_json(
        _config_identity(
            official_program,
            official_kernel_memory,
            official_post_memory,
        )
    )

    ranked: list[
        tuple[
            tuple[Any, ...],
            SDPAProgramConfig,
            MemoryConfig,
            int,
            dict[str, Any],
        ]
    ] = []
    for grid in grids:
        for sub_core_grids in _sub_core_options(grid, device):
            for q_chunk, k_chunk, max_cores, kernel_memory in itertools.product(
                q_chunks,
                k_chunks,
                max_core_options,
                kernel_memories,
            ):
                active_cores = grid.x * grid.y
                if max_cores > active_cores:
                    continue
                program = SDPAProgramConfig.from_dict(
                    {
                        **official_program.to_dict(),
                        "grid": grid.to_list(),
                        "sub_core_grids": sub_core_grids,
                        "q_chunk_size": q_chunk,
                        "k_chunk_size": k_chunk,
                        "max_cores_per_head_batch": max_cores,
                    }
                )
                identity = _config_identity(
                    program,
                    kernel_memory,
                    official_post_memory,
                )
                changed = _changed_dimensions(
                    program,
                    kernel_memory,
                    official_program=official_program,
                    official_kernel_memory=official_kernel_memory,
                )
                rank = (
                    len(changed),
                    grid.x * grid.y,
                    grid.x,
                    grid.y,
                    canonical_json(sub_core_grids),
                    q_chunk,
                    k_chunk,
                    max_cores,
                    canonical_json(kernel_memory.to_dict()),
                )
                ranked.append(
                    (
                        rank,
                        program,
                        kernel_memory,
                        len(changed),
                        {
                            "rule": "bounded_pairwise_shape_device_search",
                            "changed_dimensions": changed,
                            "cache_len": workload.cache_len,
                            "batch_size": workload.batch_size,
                            "num_heads": workload.num_heads,
                            "num_kv_heads": workload.num_kv_heads,
                            "head_dim": workload.head_dim,
                            "device_compute_grid": device.compute_grid.to_list(),
                            "device_worker_core_limit": device.worker_core_count,
                        },
                    )
                )

    ranked.sort(key=lambda item: item[0])
    proposals = []
    seen: set[str] = set()
    for _, program, kernel_memory, changed_count, derivation in ranked:
        identity = _config_identity(
            program,
            kernel_memory,
            official_post_memory,
        )
        key = canonical_json(identity)
        if key in seen:
            continue
        seen.add(key)
        source = (
            "official"
            if key == official_key
            else (
                "one_axis"
                if changed_count == 1
                else "pairwise" if changed_count == 2 else "multi_axis"
            )
        )
        proposals.append(
            (
                source,
                program,
                kernel_memory,
                official_post_memory,
                {
                    **derivation,
                    "frozen_exp_approx_mode": official_program.exp_approx_mode,
                },
            )
        )
        if len(proposals) >= max_proposals:
            break
    return proposals


def _grid_options(
    official: CoreGrid,
    device: DeviceDescriptor,
) -> list[CoreGrid]:
    values = [
        official,
        *(CoreGrid(x, y) for x, y in ((8, 4), (8, 8), (10, 8), (8, 10), (10, 10))),
    ]
    result: list[CoreGrid] = []
    seen: set[tuple[int, int]] = set()
    for grid in values:
        key = (grid.x, grid.y)
        if key in seen:
            continue
        if grid.x > device.compute_grid.x or grid.y > device.compute_grid.y:
            continue
        if grid.x * grid.y > device.worker_core_count:
            continue
        seen.add(key)
        result.append(grid)
    return result


def _sub_core_options(
    grid: CoreGrid,
    device: DeviceDescriptor,
) -> list[list[list[int]] | None]:
    width = grid.x
    height = grid.y
    placements: list[list[list[int]] | None] = [None]
    placements.append([[0, 0, width - 1, height - 1]])
    if width < device.compute_grid.x:
        placements.append([[1, 0, width, height - 1]])
    if height < device.compute_grid.y:
        placements.append([[0, 1, width - 1, height]])
    if height > 1:
        split = height // 2
        placements.append(
            [
                [0, 0, width - 1, split - 1],
                [0, split, width - 1, height - 1],
            ]
        )
    return _ordered_unique(placements)


def _k_chunk_options(official: int, cache_len: int) -> list[int]:
    values = [official, 0, 32, 64, 128]
    if cache_len >= 512:
        values.append(256)
    if cache_len >= 1024:
        values.append(512)
    return _ordered_unique(
        value for value in values if value == 0 or value <= cache_len
    )


def _memory_options(
    official_kernel: MemoryConfig,
    official_post: MemoryConfig,
) -> list[MemoryConfig]:
    values = [
        official_kernel,
        MemoryConfig.named("L1_MEMORY_CONFIG"),
        official_post,
    ]
    result: list[MemoryConfig] = []
    seen: set[str] = set()
    for memory in values:
        key = canonical_json(memory.to_dict())
        if key in seen:
            continue
        seen.add(key)
        result.append(memory)
    return result


def _changed_dimensions(
    program: SDPAProgramConfig,
    kernel_memory: MemoryConfig,
    *,
    official_program: SDPAProgramConfig,
    official_kernel_memory: MemoryConfig,
) -> list[str]:
    dimensions = (
        ("grid", program.grid.to_list(), official_program.grid.to_list()),
        (
            "sub_core_grids",
            program.sub_core_grids,
            official_program.sub_core_grids,
        ),
        ("q_chunk_size", program.q_chunk_size, official_program.q_chunk_size),
        ("k_chunk_size", program.k_chunk_size, official_program.k_chunk_size),
        (
            "max_cores_per_head_batch",
            program.max_cores_per_head_batch,
            official_program.max_cores_per_head_batch,
        ),
        (
            "kernel_output_memory",
            kernel_memory.to_dict(),
            official_kernel_memory.to_dict(),
        ),
    )
    return [
        name
        for name, value, official in dimensions
        if canonical_json(value) != canonical_json(official)
    ]


def _replace_sdpa(
    base_space: SearchSpaceConfig,
    *,
    program: SDPAProgramConfig,
    kernel_output_memory: MemoryConfig,
) -> SearchSpaceConfig:
    payload = base_space.to_dict()
    operator = payload["operators"].get(SDPA_OPERATOR)
    if not isinstance(operator, dict):
        raise SDPAEnumerationError("attention.sdpa disappeared from search space")
    post_memory = copy.deepcopy(operator.get("output_memory"))
    operator.clear()
    operator.update(program.to_dict())
    operator["kernel_output_memory"] = kernel_output_memory.to_dict()
    operator["output_memory"] = post_memory
    if _PROGRAM_GRID_PATH in payload["core_grids"]:
        payload["core_grids"][_PROGRAM_GRID_PATH] = program.grid.to_list()
    if _KERNEL_OUTPUT_MEMORY_PATH in payload["memory_configs"]:
        payload["memory_configs"][
            _KERNEL_OUTPUT_MEMORY_PATH
        ] = kernel_output_memory.to_dict()
    edge = payload["edges"].get("sdpa_to_concat_heads")
    if isinstance(edge, dict):
        consumer = edge.get("consumer_input_memory")
        producer = kernel_output_memory.to_dict()
        edge["producer_output_memory"] = producer
        edge["conversion"] = (
            "none"
            if isinstance(consumer, Mapping)
            and canonical_json(producer) == canonical_json(consumer)
            else "explicit"
        )
    return SearchSpaceConfig.from_dict(payload)


def _required_memory(operator: Mapping[str, Any], key: str) -> MemoryConfig:
    value = operator.get(key)
    if not isinstance(value, Mapping):
        raise SDPAEnumerationError(f"attention.sdpa has no {key}")
    return MemoryConfig.from_dict(value)


def _config_identity(
    program: SDPAProgramConfig,
    kernel_memory: MemoryConfig,
    post_memory: MemoryConfig,
) -> dict[str, Any]:
    return {
        "program": program.to_dict(),
        "kernel_output_memory": kernel_memory.to_dict(),
        "post_sdpa_output_memory": post_memory.to_dict(),
    }


def _candidate_id(
    identity: Mapping[str, Any],
    *,
    workload: SDPAWorkload,
    device: DeviceDescriptor,
) -> str:
    payload = {
        "operator": SDPA_OPERATOR,
        "config": copy.deepcopy(dict(identity)),
        "workload": workload.to_dict(),
        "device": device.to_dict(),
    }
    return f"attention-sdpa-{sha256_json(payload)[:12]}"


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result = []
    seen: set[str] = set()
    for value in values:
        key = canonical_json(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _unique_values(values: Iterable[Any]) -> list[Any]:
    result: dict[str, Any] = {}
    for value in values:
        result.setdefault(canonical_json(value), copy.deepcopy(value))
    return [result[key] for key in sorted(result)]
