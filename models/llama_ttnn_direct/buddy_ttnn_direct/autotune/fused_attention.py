from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .legality import (
    DeviceDescriptor,
    LegalityReport,
    WorkloadSpec,
    validate_candidate,
)
from .measurement import MeasurementCandidate
from .schema import PrecisionContract, canonical_json, sha256_json
from .space import SearchSpaceConfig
from .templates import (
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    KV_UPDATE_AXIS,
    ROPE_AXIS,
    apply_template_selection,
)

FUSED_ATTENTION_SCHEMA_VERSION = 1
FUSED_ATTENTION_REGION = "attention.fused_qk_kv_region"
FUSED_ATTENTION_REGION_PROMOTION_GAIN = 0.01

_INCUMBENT_OPERATIONS = (
    "linear.qkv_packed",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding.q",
    "rotary_embedding.k",
    "paged_update_cache.k",
    "paged_update_cache.v",
)
_FUSED_OPERATIONS = (
    "linear.qkv_packed",
    "nlp_create_qkv_heads_decode.disjoint_qk",
    "rotary_embedding_llama_fused_qk",
    "paged_fused_update_cache.kv",
)


class FusedAttentionError(ValueError):
    """Raised when a fused attention layout candidate is inconsistent."""


@dataclass(frozen=True)
class FusedAttentionLayoutCandidate:
    candidate_id: str
    name: str
    qkv_program_config: Mapping[str, Any]
    qkv_output_memory_config: Mapping[str, Any]
    producer_output_layout: Mapping[str, Any]
    q_core_ranges: tuple[tuple[int, int, int, int], ...]
    k_core_ranges: tuple[tuple[int, int, int, int], ...]
    v_core_ranges: tuple[tuple[int, int, int, int], ...]
    persistent_core_ranges: tuple[tuple[int, int, int, int], ...]
    operation_sequence: tuple[str, ...]
    added_conversion_count: int
    removed_conversion_count: int
    fused_operation_count: int
    operation_count_reduction: int
    legality: LegalityReport = field(repr=False)
    _runtime_config_json: str = field(repr=False)

    @property
    def runtime_config(self) -> dict[str, Any]:
        value = json.loads(self._runtime_config_json)
        if not isinstance(value, dict):
            raise FusedAttentionError("candidate runtime config is not an object")
        return value

    @property
    def zero_hot_path_conversions(self) -> bool:
        return self.added_conversion_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "operator": FUSED_ATTENTION_REGION,
            "status": "static_legal",
            "producer": {
                "qkv_linear": {
                    "api": "ttnn.linear",
                    "program_config": copy.deepcopy(
                        dict(self.qkv_program_config)
                    ),
                    "output_memory_config": copy.deepcopy(
                        dict(self.qkv_output_memory_config)
                    ),
                },
                "create_heads": {
                    "api": "ttnn.experimental.nlp_create_qkv_heads_decode",
                    "direct_layout": True,
                    "overlap_qk_coregrid": False,
                    "output_layout": copy.deepcopy(
                        dict(self.producer_output_layout)
                    ),
                },
            },
            "core_placement": {
                "q": [list(item) for item in self.q_core_ranges],
                "k": [list(item) for item in self.k_core_ranges],
                "v": [list(item) for item in self.v_core_ranges],
                "persistent_rope": [
                    list(item) for item in self.persistent_core_ranges
                ],
            },
            "templates": {
                ROPE_AXIS: FUSED_QK_ROPE,
                KV_UPDATE_AXIS: FUSED_PAGED_UPDATE,
            },
            "operation_sequence": list(self.operation_sequence),
            "fused_operation_count": self.fused_operation_count,
            "operation_count_reduction": self.operation_count_reduction,
            "added_conversion_count": self.added_conversion_count,
            "removed_conversion_count": self.removed_conversion_count,
            "zero_hot_path_conversion_target": self.zero_hot_path_conversions,
            "legality": self.legality.to_dict(),
        }


@dataclass(frozen=True)
class FusedAttentionEnumerationResult:
    incumbent_candidate_id: str
    candidates: tuple[FusedAttentionLayoutCandidate, ...]
    rejected: tuple[dict[str, Any], ...]
    precision_contract_hash: str
    schema_version: int = FUSED_ATTENTION_SCHEMA_VERSION

    @property
    def status(self) -> str:
        return "passed" if self.candidates else "conversion_bound"

    def candidate(self, candidate_id: str) -> FusedAttentionLayoutCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise FusedAttentionError(
            f"unknown fused attention candidate: {candidate_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "operator": FUSED_ATTENTION_REGION,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "precision_contract_hash": self.precision_contract_hash,
            "search_unit": [
                "qkv_linear_output_sharding",
                "create_heads_output_layout",
                "q_k_core_placement",
                "persistent_rope_constant_placement",
                "fused_qk_rope",
                "cache_update_input_layout",
                "fused_kv_update",
            ],
            "zero_hot_path_conversion_target": True,
            "legal_candidate_count": len(self.candidates),
            "conversion_bound_count": sum(
                item.get("status") == "conversion_bound"
                for item in self.rejected
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [copy.deepcopy(item) for item in self.rejected],
        }


def enumerate_fused_attention_layout_candidates(
    *,
    runtime_config: Mapping[str, Any],
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    producer_supports_disjoint_qk: bool = True,
) -> FusedAttentionEnumerationResult:
    identity_base = {
        "operator": FUSED_ATTENTION_REGION,
        "precision_contract_hash": precision_contract.hash,
        "batch_size": int(runtime_config.get("batch_size", 0)),
        "num_attention_heads": int(
            runtime_config.get("num_attention_heads", 0)
        ),
        "num_key_value_heads": int(
            runtime_config.get("num_key_value_heads", 0)
        ),
        "head_dim": int(runtime_config.get("head_dim", 0)),
    }
    incumbent_id = "attention-fused-region-incumbent-" + sha256_json(
        {**identity_base, "operations": _INCUMBENT_OPERATIONS}
    )[:12]

    rejected = [_naive_fused_conversion_bound(identity_base)]
    constraints = _producer_constraints(
        runtime_config,
        device,
        producer_supports_disjoint_qk=producer_supports_disjoint_qk,
    )
    if constraints:
        rejected.append(
            {
                "candidate_id": "attention-fused-layout-"
                + sha256_json({**identity_base, "mode": "direct"})[:12],
                "name": "zero_conversion_fused_qk_kv",
                "status": "conversion_bound",
                "error_class": "unsupported_layout",
                "reasons": constraints,
                "enter_full_model": False,
            }
        )
        return FusedAttentionEnumerationResult(
            incumbent_candidate_id=incumbent_id,
            candidates=(),
            rejected=tuple(rejected),
            precision_contract_hash=precision_contract.hash,
        )

    configured = _zero_conversion_runtime_config(runtime_config)
    candidate_identity = {
        **identity_base,
        "mode": "zero_conversion_fused_qk_kv",
        "layout": _layout_identity(configured),
        "operations": _FUSED_OPERATIONS,
    }
    candidate_id = "attention-fused-layout-" + sha256_json(
        candidate_identity
    )[:12]
    space = SearchSpaceConfig.from_runtime_config(configured)
    workload = WorkloadSpec.from_runtime_config(configured)
    legality = validate_candidate(
        space,
        workload,
        device,
        candidate_id=candidate_id,
    )
    cross_op_issues = _cross_op_layout_issues(configured)
    if not legality.passed or cross_op_issues:
        rejected.append(
            {
                "candidate_id": candidate_id,
                "name": "zero_conversion_fused_qk_kv",
                "status": "static_illegal",
                "error_class": legality.error_class or "unsupported_layout",
                "legality": legality.to_dict(),
                "cross_op_issues": cross_op_issues,
                "enter_full_model": False,
            }
        )
        candidates: tuple[FusedAttentionLayoutCandidate, ...] = ()
    else:
        attention = configured["attention"]
        candidates = (
            FusedAttentionLayoutCandidate(
                candidate_id=candidate_id,
                name="zero_conversion_fused_qk_kv",
                qkv_program_config=copy.deepcopy(
                    attention["qkv_program_config"]
                ),
                qkv_output_memory_config=copy.deepcopy(
                    attention["qkv_output_memory_config"]
                ),
                producer_output_layout=copy.deepcopy(
                    attention["qkv_heads_memory_config"]
                ),
                q_core_ranges=_ranges(
                    attention["fused_q_memory_config"]
                ),
                k_core_ranges=_ranges(
                    attention["fused_k_memory_config"]
                ),
                v_core_ranges=_ranges(
                    attention["fused_cache_value_memory_config"]
                ),
                persistent_core_ranges=_ranges(
                    attention["fused_rope_cos_sin_memory_config"]
                ),
                operation_sequence=_FUSED_OPERATIONS,
                added_conversion_count=0,
                removed_conversion_count=3,
                fused_operation_count=2,
                operation_count_reduction=(
                    len(_INCUMBENT_OPERATIONS) - len(_FUSED_OPERATIONS)
                ),
                legality=legality,
                _runtime_config_json=canonical_json(configured),
            ),
        )
    return FusedAttentionEnumerationResult(
        incumbent_candidate_id=incumbent_id,
        candidates=candidates,
        rejected=tuple(rejected),
        precision_contract_hash=precision_contract.hash,
    )


def apply_fused_attention_layout_candidate(
    runtime_config: Mapping[str, Any],
    candidate: FusedAttentionLayoutCandidate,
) -> dict[str, Any]:
    configured = candidate.runtime_config
    source_identity = {
        "batch_size": int(runtime_config.get("batch_size", 0)),
        "num_attention_heads": int(
            runtime_config.get("num_attention_heads", 0)
        ),
        "num_key_value_heads": int(
            runtime_config.get("num_key_value_heads", 0)
        ),
        "head_dim": int(runtime_config.get("head_dim", 0)),
    }
    candidate_identity = {
        name: int(configured.get(name, 0)) for name in source_identity
    }
    if source_identity != candidate_identity:
        raise FusedAttentionError(
            "fused attention candidate does not match runtime dimensions"
        )
    result = copy.deepcopy(configured)
    result.setdefault("autotune", {})["fused_attention_layout"] = {
        "schema_version": FUSED_ATTENTION_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "zero_hot_path_conversions": candidate.zero_hot_path_conversions,
        "operation_count_reduction": candidate.operation_count_reduction,
        "removed_conversion_count": candidate.removed_conversion_count,
    }
    return result


def rank_fused_attention_region_candidates(
    enumeration: FusedAttentionEnumerationResult,
) -> tuple[MeasurementCandidate, ...]:
    ranked = [
        MeasurementCandidate.create(
            candidate_id=enumeration.incumbent_candidate_id,
            operator_name=FUSED_ATTENTION_REGION,
            candidate_kind="fused_attention_region",
            analytical_score=2.0,
            l1_bytes=0,
            source="incumbent_separate_rope_kv_update",
            is_incumbent=True,
            metadata={
                "mode": "incumbent",
                "operation_sequence": list(_INCUMBENT_OPERATIONS),
                "fused_operation_count": 0,
                "operation_count_reduction": 0,
                "added_conversion_count": 0,
                "removed_conversion_count": 0,
            },
        )
    ]
    for candidate in enumeration.candidates:
        ranked.append(
            MeasurementCandidate.create(
                candidate_id=candidate.candidate_id,
                operator_name=FUSED_ATTENTION_REGION,
                candidate_kind="fused_attention_region",
                analytical_score=float(candidate.added_conversion_count),
                l1_bytes=5 * 2_048,
                source="fused_template_producer_layout_cosearch",
                is_incumbent=False,
                metadata={
                    "mode": "challenger",
                    "candidate": candidate.to_dict(),
                },
            )
        )
    return tuple(ranked)


def select_fused_attention_region_winner(
    enumeration: FusedAttentionEnumerationResult,
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    statistic: str = "p50",
) -> dict[str, Any]:
    incumbent_latency = _measurement_latency(
        measurements.get(enumeration.incumbent_candidate_id), statistic
    )
    ranked = []
    for candidate in enumeration.candidates:
        latency = _measurement_latency(
            measurements.get(candidate.candidate_id), statistic
        )
        if latency is None:
            continue
        ranked.append(
            {
                **candidate.to_dict(),
                "latency_ms": latency,
            }
        )
    ranked.sort(key=lambda item: (item["latency_ms"], item["candidate_id"]))
    winner = ranked[0] if ranked else None
    gain = None
    if incumbent_latency is not None and winner is not None:
        gain = (
            incumbent_latency - float(winner["latency_ms"])
        ) / incumbent_latency
    gate_passed = (
        gain is not None and gain >= FUSED_ATTENTION_REGION_PROMOTION_GAIN
    )
    return {
        "schema_version": FUSED_ATTENTION_SCHEMA_VERSION,
        "status": (
            "selected"
            if incumbent_latency is not None and winner is not None
            else "incomplete"
        ),
        "operator": FUSED_ATTENTION_REGION,
        "statistic": statistic,
        "incumbent": {
            "candidate_id": enumeration.incumbent_candidate_id,
            "latency_ms": incumbent_latency,
            "operation_sequence": list(_INCUMBENT_OPERATIONS),
            "operation_count": len(_INCUMBENT_OPERATIONS),
            "fused_operation_count": 0,
            "added_conversion_count": 0,
        },
        "winner": copy.deepcopy(winner),
        "ranked_challengers": ranked,
        "net_attention_subregion_gain": gain,
        "region_promotion_threshold": FUSED_ATTENTION_REGION_PROMOTION_GAIN,
        "region_gate_passed": gate_passed,
        "retained_candidate_id": winner["candidate_id"] if gate_passed else None,
        "enter_full_model": gate_passed,
    }


def build_fused_attention_region_payload(
    *,
    enumeration: FusedAttentionEnumerationResult,
    candidate_id: str,
    runtime_config: Mapping[str, Any],
    device_id: int = 0,
    physical_cache_len: int | None = None,
) -> dict[str, Any]:
    if not enumeration.candidates:
        raise FusedAttentionError(
            "fused attention region payload requires a legal challenger"
        )
    challenger = enumeration.candidates[0]
    if candidate_id == enumeration.incumbent_candidate_id:
        mode = "incumbent"
        operation_sequence = _INCUMBENT_OPERATIONS
        accounting = {
            "fused_operation_count": 0,
            "operation_count_reduction": 0,
            "added_conversion_count": 0,
            "removed_conversion_count": 0,
        }
    else:
        challenger = enumeration.candidate(candidate_id)
        mode = "challenger"
        operation_sequence = challenger.operation_sequence
        accounting = {
            "fused_operation_count": challenger.fused_operation_count,
            "operation_count_reduction": challenger.operation_count_reduction,
            "added_conversion_count": challenger.added_conversion_count,
            "removed_conversion_count": challenger.removed_conversion_count,
        }
    configured = challenger.runtime_config
    attention = configured["attention"]
    incumbent_attention = runtime_config.get("attention")
    if not isinstance(incumbent_attention, Mapping):
        raise FusedAttentionError("runtime attention config is missing")
    workload = WorkloadSpec.from_runtime_config(configured)
    qkv_workload = next(
        (item for item in workload.matmuls if item.name == "attention.qkv"),
        None,
    )
    if qkv_workload is None:
        raise FusedAttentionError("attention QKV workload is missing")
    batch = int(configured["batch_size"])
    num_heads = int(configured["num_attention_heads"])
    num_kv_heads = int(configured["num_key_value_heads"])
    head_dim = int(configured["head_dim"])
    cache_len = int(
        physical_cache_len
        if physical_cache_len is not None
        else configured["max_cache_len"]
    )
    if cache_len <= 0 or cache_len % 32:
        raise FusedAttentionError(
            "fused attention region cache length must be a positive multiple of 32"
        )
    kv_cache = configured.get("kv_cache")
    cache_dtype = (
        kv_cache.get("dtype")
        if isinstance(kv_cache, Mapping)
        else "bfloat8_b"
    )
    if isinstance(cache_dtype, Mapping):
        cache_dtype = cache_dtype.get("name", "bfloat8_b")
    return {
        "device_id": int(device_id),
        "mode": mode,
        "batch_size": batch,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "physical_cache_len": cache_len,
        "page_block_size": 32,
        "active_position": min(127, cache_len - 1),
        "hidden_size": int(configured["hidden_size"]),
        "hidden_input_memory": (
            qkv_workload.input_memory.to_runtime_descriptor()
        ),
        "weight_memory": qkv_workload.weight_memory.to_runtime_descriptor(),
        "qkv_output_memory": (
            qkv_workload.output_memory.to_runtime_descriptor()
        ),
        "incumbent_qkv_program_config": copy.deepcopy(
            incumbent_attention["qkv_program_config"]
        ),
        "challenger_qkv_program_config": copy.deepcopy(
            attention["qkv_program_config"]
        ),
        "input_dtype": qkv_workload.input_dtype,
        "weight_dtype": qkv_workload.weight_dtype,
        "output_dtype": qkv_workload.output_dtype,
        "incumbent_heads_memory": _height_sharded_config(
            [[0, 0, 7, 3]], [batch, head_dim]
        ),
        "incumbent_rope_cos_sin_memory": _height_sharded_config(
            [[0, 0, 7, 3]], [32, head_dim]
        ),
        "incumbent_rope_transform_memory": _height_sharded_config(
            [[0, 0, 7, 3]], [32, 32]
        ),
        "fused_heads_memory": copy.deepcopy(
            attention["qkv_heads_memory_config"]
        ),
        "fused_rope_cos_sin_memory": copy.deepcopy(
            attention["fused_rope_cos_sin_memory_config"]
        ),
        "fused_rope_transform_memory": copy.deepcopy(
            attention["fused_rope_transform_memory_config"]
        ),
        "compute_kernel_config": copy.deepcopy(
            attention["qkv_compute_kernel_config"]
        ),
        "cache_dtype": str(cache_dtype),
        "rope": copy.deepcopy(dict(configured.get("rotary") or {})),
        "operation_sequence": list(operation_sequence),
        **accounting,
    }


def _zero_conversion_runtime_config(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    space = SearchSpaceConfig.from_runtime_config(runtime_config)
    templates = dict(space.templates)
    templates[ROPE_AXIS] = FUSED_QK_ROPE
    templates[KV_UPDATE_AXIS] = FUSED_PAGED_UPDATE
    result = apply_template_selection(runtime_config, templates)
    attention = result.setdefault("attention", {})
    first = [[0, 0, 7, 3]]
    second = [[0, 4, 7, 7]]
    combined = [*first, *second]
    head_dim = int(result["head_dim"])
    qkv_program = attention.get("qkv_program_config")
    if not isinstance(qkv_program, Mapping):
        raise FusedAttentionError("attention QKV program config is missing")
    qkv_program = copy.deepcopy(dict(qkv_program))
    qkv_program["per_core_N"] = head_dim // 32
    attention.update(
        {
            "qkv_program_config": qkv_program,
            "qkv_heads_memory_config": _height_sharded_config(
                [[0, 0, 7, 7]], [32, head_dim]
            ),
            "qkv_heads_overlap_qk_coregrid": False,
            "fused_q_memory_config": _height_sharded_config(
                first, [32, head_dim]
            ),
            "fused_k_memory_config": _height_sharded_config(
                second, [32, head_dim]
            ),
            "fused_rope_cos_sin_memory_config": _height_sharded_config(
                combined, [32, head_dim]
            ),
            "fused_rope_transform_memory_config": _height_sharded_config(
                combined, [32, 32]
            ),
            "fused_cache_key_memory_config": _height_sharded_config(
                second, [32, head_dim]
            ),
            "fused_cache_value_memory_config": _height_sharded_config(
                first, [32, head_dim]
            ),
        }
    )
    return result


def _height_sharded_config(
    core_ranges: Sequence[Sequence[int]], shard_shape: Sequence[int]
) -> dict[str, Any]:
    ranges = [[int(value) for value in item] for item in core_ranges]
    core_count = sum(
        (x1 - x0 + 1) * (y1 - y0 + 1)
        for x0, y0, x1, y1 in ranges
    )
    return {
        "kind": "ttnn_sharded_memory_config",
        "strategy": "height",
        "core_grid": [8, core_count // 8],
        "core_ranges": ranges,
        "shard_shape": [int(value) for value in shard_shape],
        "orientation": "row_major",
    }


def _producer_constraints(
    runtime_config: Mapping[str, Any],
    device: DeviceDescriptor,
    *,
    producer_supports_disjoint_qk: bool,
) -> list[dict[str, Any]]:
    constraints = []
    if not producer_supports_disjoint_qk:
        constraints.append(
            {
                "code": "CREATE_HEADS_DISJOINT_QK_API_UNAVAILABLE",
                "path": "attention.qkv_heads_overlap_qk_coregrid",
                "message": (
                    "create-heads producer cannot emit non-overlapping Q/K "
                    "layouts; fused candidate would require hot conversions"
                ),
            }
        )
    expected = {
        "batch_size": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    for name, value in expected.items():
        observed = int(runtime_config.get(name, 0))
        if observed != value:
            constraints.append(
                {
                    "code": "FUSED_ATTENTION_WORKLOAD_UNSUPPORTED",
                    "path": name,
                    "message": f"expected {name}={value}, observed {observed}",
                }
            )
    if device.compute_grid.x < 8 or device.compute_grid.y < 8:
        constraints.append(
            {
                "code": "FUSED_ATTENTION_CORE_GRID_UNAVAILABLE",
                "path": "device.compute_grid",
                "message": "zero-conversion fused attention requires an 8x8 grid",
            }
        )
    attention = runtime_config.get("attention")
    qkv_program = (
        attention.get("qkv_program_config")
        if isinstance(attention, Mapping)
        else None
    )
    if not isinstance(qkv_program, Mapping):
        constraints.append(
            {
                "code": "QKV_PRODUCER_PROGRAM_MISSING",
                "path": "attention.qkv_program_config",
                "message": "QKV producer program config is required",
            }
        )
        return constraints
    if qkv_program.get("kind") != "ttnn_matmul_dram_sharded_program_config":
        constraints.append(
            {
                "code": "QKV_PRODUCER_PROGRAM_UNSUPPORTED",
                "path": "attention.qkv_program_config.kind",
                "message": (
                    "direct fused-compatible QKV output requires the "
                    "DRAM-sharded matmul producer"
                ),
            }
        )
    head_dim = int(runtime_config.get("head_dim", 0))
    qkv_width = (
        int(runtime_config.get("num_attention_heads", 0))
        + 2 * int(runtime_config.get("num_key_value_heads", 0))
    ) * head_dim
    if head_dim <= 0 or head_dim % 32:
        constraints.append(
            {
                "code": "QKV_HEAD_DIM_NOT_TILE_ALIGNED",
                "path": "head_dim",
                "message": "disjoint create-heads requires tile-aligned heads",
            }
        )
    else:
        producer_shard_width = head_dim
        producer_core_count = (
            qkv_width // producer_shard_width if producer_shard_width else 0
        )
        available_cores = device.compute_grid.x * device.compute_grid.y
        if qkv_width % producer_shard_width or producer_core_count > available_cores:
            constraints.append(
                {
                    "code": "QKV_DIRECT_OUTPUT_GRID_UNAVAILABLE",
                    "path": "attention.qkv_program_config.per_core_N",
                    "message": (
                        "QKV producer cannot emit one whole-head shard per core "
                        f"within the {available_cores}-core device grid"
                    ),
                }
            )
    return constraints


def _cross_op_layout_issues(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    attention = config.get("attention")
    if not isinstance(attention, Mapping):
        return [{"code": "ATTENTION_CONFIG_MISSING"}]
    q = _ranges(attention["fused_q_memory_config"])
    k = _ranges(attention["fused_k_memory_config"])
    cache_k = _ranges(attention["fused_cache_key_memory_config"])
    cache_v = _ranges(attention["fused_cache_value_memory_config"])
    issues = []
    if q == k:
        issues.append({"code": "FUSED_QK_CORE_OVERLAP"})
    if k != cache_k:
        issues.append({"code": "ROPE_K_TO_CACHE_K_CONVERSION_REQUIRED"})
    if q != cache_v:
        issues.append({"code": "CREATE_HEADS_V_TO_CACHE_V_CONVERSION_REQUIRED"})
    if attention.get("qkv_heads_overlap_qk_coregrid") is not False:
        issues.append({"code": "CREATE_HEADS_OVERLAPPING_OUTPUTS"})
    return issues


def _layout_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    attention = config["attention"]
    return {
        name: copy.deepcopy(attention[name])
        for name in (
            "qkv_program_config",
            "qkv_output_memory_config",
            "qkv_heads_memory_config",
            "qkv_heads_overlap_qk_coregrid",
            "fused_q_memory_config",
            "fused_k_memory_config",
            "fused_rope_cos_sin_memory_config",
            "fused_rope_transform_memory_config",
            "fused_cache_key_memory_config",
            "fused_cache_value_memory_config",
        )
    }


def _ranges(value: Mapping[str, Any]) -> tuple[tuple[int, int, int, int], ...]:
    ranges = value.get("core_ranges")
    if not isinstance(ranges, list):
        return ()
    return tuple(tuple(int(component) for component in item) for item in ranges)


def _naive_fused_conversion_bound(
    identity_base: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": "attention-fused-layout-"
        + sha256_json({**identity_base, "mode": "naive_fused"})[:12],
        "name": "fused_templates_with_official_producer_layout",
        "status": "conversion_bound",
        "error_class": "unsupported_layout",
        "producer_direct_layout": False,
        "added_conversion_count": 3,
        "removed_conversion_count": 0,
        "zero_hot_path_conversion_target": False,
        "reasons": [
            {
                "code": "PRODUCER_LAYOUT_REQUIRES_HOT_CONVERSIONS",
                "message": (
                    "overlapping create-heads outputs require K relocation for "
                    "fused RoPE and K/V relocation for fused cache update"
                ),
            }
        ],
        "enter_full_model": False,
    }


def _measurement_latency(
    measurement: Mapping[str, Any] | None, statistic: str
) -> float | None:
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("status") != "passed"
    ):
        return None
    statistics = measurement.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get(statistic) is None:
        return None
    value = float(statistics[statistic])
    return value if math.isfinite(value) and value > 0 else None
