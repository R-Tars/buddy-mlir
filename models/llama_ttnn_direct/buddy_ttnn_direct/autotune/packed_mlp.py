from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .legality import DeviceDescriptor, WorkloadSpec
from .matmul import (
    PACKED_GATE_UP_OPERATOR,
    MatmulCandidate,
    MatmulEnumerationResult,
    enumerate_matmul_programs,
    rank_matmul_measurement_candidates,
)
from .measurement import MeasurementCandidate
from .schema import PrecisionContract, canonical_json, sha256_json
from .space import MemoryConfig, SearchSpaceConfig
from .templates import (
    ACTIVATION_AXIS,
    GATE_UP_AXIS,
    MUL_FUSED_SILU,
    PACKED_GATE_UP,
    apply_template_axis_updates,
)
from .transfer import DEFAULT_LAYER_GROUP, OVERRIDE_LAYER_GROUP

PACKED_MLP_SCHEMA_VERSION = 1
PACKED_REGION_PROMOTION_GAIN = 0.03
PACKED_REGION_PREFERRED_GAIN = 0.05
FULL_MODEL_PROMOTION_GAIN = 0.01

_LAYER_GROUPS = {
    0: DEFAULT_LAYER_GROUP,
    31: OVERRIDE_LAYER_GROUP,
}
_SPLIT_STRATEGIES = ("split", "slice")
_SPLIT_LAYOUT_PAIRS = (("L1_MEMORY_CONFIG", "L1_MEMORY_CONFIG"),)
_PACKED_RUNTIME_FIELDS = (
    "packed_gate_up_program_config",
    "packed_gate_up_output_memory_config",
    "packed_gate_up_split_strategy",
    "packed_gate_up_split_output_memory_config",
    "packed_gate_up_mul_input_memory_config",
    "packed_gate_up_mul_conversion",
)


class PackedMLPError(ValueError):
    """Raised when a packed gate/up candidate or report is malformed."""


@dataclass(frozen=True)
class PackedGateUpCandidate:
    candidate_id: str
    layer_group: str
    representative_layer: int
    matmul_candidate: MatmulCandidate = field(repr=False)
    split_strategy: str
    split_output_memory: MemoryConfig
    mul_input_memory: MemoryConfig
    requires_mul_conversion: bool

    @property
    def operator_name(self) -> str:
        return PACKED_GATE_UP_OPERATOR

    @property
    def program_family(self) -> str:
        return self.matmul_candidate.program_family

    @property
    def is_official_program(self) -> bool:
        return self.matmul_candidate.is_official

    def to_dict(self) -> dict[str, Any]:
        operator = self.matmul_candidate.search_space.operators[PACKED_GATE_UP_OPERATOR]
        return {
            "candidate_id": self.candidate_id,
            "operator": self.operator_name,
            "layer_group": self.layer_group,
            "representative_layer": self.representative_layer,
            "program_family": self.program_family,
            "is_official_program": self.is_official_program,
            "matmul_candidate_id": self.matmul_candidate.candidate_id,
            "programs": [
                program.to_dict() for program in self.matmul_candidate.programs
            ],
            "worker_core_counts": list(self.matmul_candidate.worker_core_counts),
            "weight_memory": copy.deepcopy(operator.get("weight_memory")),
            "linear_output_memory": copy.deepcopy(operator.get("output_memory")),
            "split_strategy": self.split_strategy,
            "split_output_memory": self.split_output_memory.to_dict(),
            "mul_input_memory": self.mul_input_memory.to_dict(),
            "requires_mul_conversion": self.requires_mul_conversion,
            "hot_path_conversion_count": (2 if self.requires_mul_conversion else 0),
            "legality": {
                "status": self.matmul_candidate.legality.status,
                "passed": self.matmul_candidate.legality.passed,
                "issues": [
                    issue.to_dict() for issue in self.matmul_candidate.legality.issues
                ],
            },
        }


@dataclass(frozen=True)
class PackedGateUpEnumerationResult:
    layer_group: str
    representative_layer: int
    incumbent_candidate_id: str
    matmul: MatmulEnumerationResult = field(repr=False)
    candidates: tuple[PackedGateUpCandidate, ...]
    rejected: tuple[dict[str, Any], ...]
    precision_contract_hash: str
    schema_version: int = PACKED_MLP_SCHEMA_VERSION

    @property
    def status(self) -> str:
        if not self.candidates:
            return "no_legal_candidates"
        if not any(item.is_official_program for item in self.candidates):
            return "official_program_missing"
        return "passed"

    def candidate(self, candidate_id: str) -> PackedGateUpCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise PackedMLPError(f"unknown packed gate/up candidate: {candidate_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "operator": PACKED_GATE_UP_OPERATOR,
            "layer_group": self.layer_group,
            "representative_layer": self.representative_layer,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "precision_contract_hash": self.precision_contract_hash,
            "matmul_enumeration": self.matmul.to_dict(),
            "legal_candidate_count": len(self.candidates),
            "rejected_candidate_count": len(self.rejected),
            "search_field_coverage": _search_field_coverage(self.candidates),
            "attempted_layouts": {
                "linear_output": [
                    "width_sharded",
                    "interleaved",
                    "height_sharded",
                    "block_sharded",
                ],
                "split_output": ["width_sharded", "interleaved"],
                "mul_input": ["width_sharded", "interleaved"],
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [copy.deepcopy(item) for item in self.rejected],
        }


def enumerate_packed_gate_up_candidates(
    *,
    runtime_config: Mapping[str, Any],
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    representative_layer: int,
    max_matmul_proposals: int = 64,
) -> PackedGateUpEnumerationResult:
    layer_group = _layer_group(representative_layer)
    configured = _packed_runtime_config(runtime_config)
    space = SearchSpaceConfig.from_runtime_config(configured)
    workload = WorkloadSpec.from_runtime_config(
        configured,
        representative_layer=representative_layer,
    )
    matmul = enumerate_matmul_programs(
        operator_name=PACKED_GATE_UP_OPERATOR,
        base_space=space,
        workload=workload,
        device=device,
        precision_contract=precision_contract,
        max_proposals=max_matmul_proposals,
    )

    candidates: list[PackedGateUpCandidate] = []
    for matmul_candidate in matmul.candidates:
        for split_strategy in _SPLIT_STRATEGIES:
            for split_name, mul_name in _SPLIT_LAYOUT_PAIRS:
                split_memory = MemoryConfig.named(split_name)
                mul_memory = MemoryConfig.named(mul_name)
                requires_conversion = split_memory.to_dict() != mul_memory.to_dict()
                identity = {
                    "operator": PACKED_GATE_UP_OPERATOR,
                    "layer_group": layer_group,
                    "representative_layer": representative_layer,
                    "matmul_candidate_id": matmul_candidate.candidate_id,
                    "split_strategy": split_strategy,
                    "split_output_memory": split_memory.to_dict(),
                    "mul_input_memory": mul_memory.to_dict(),
                    "requires_mul_conversion": requires_conversion,
                    "precision_contract_hash": precision_contract.hash,
                }
                candidates.append(
                    PackedGateUpCandidate(
                        candidate_id=(
                            "mlp-gate-up-packed-" + sha256_json(identity)[:12]
                        ),
                        layer_group=layer_group,
                        representative_layer=representative_layer,
                        matmul_candidate=matmul_candidate,
                        split_strategy=split_strategy,
                        split_output_memory=split_memory,
                        mul_input_memory=mul_memory,
                        requires_mul_conversion=requires_conversion,
                    )
                )

    candidates.sort(key=lambda item: item.candidate_id)
    rejected = tuple(
        [copy.deepcopy(item) for item in matmul.rejected]
        + _unsupported_layout_attempts(layer_group)
    )
    incumbent_id = (
        "mlp-gate-up-incumbent-"
        + sha256_json(
            {
                "layer_group": layer_group,
                "representative_layer": representative_layer,
                "precision_contract_hash": precision_contract.hash,
                "operation_sequence": ["linear.gate", "linear.up", "mul_silu"],
            }
        )[:12]
    )
    return PackedGateUpEnumerationResult(
        layer_group=layer_group,
        representative_layer=representative_layer,
        incumbent_candidate_id=incumbent_id,
        matmul=matmul,
        candidates=tuple(candidates),
        rejected=rejected,
        precision_contract_hash=precision_contract.hash,
    )


def apply_packed_gate_up_candidate(
    runtime_config: Mapping[str, Any],
    candidate: PackedGateUpCandidate,
) -> dict[str, Any]:
    configured = _packed_runtime_config(runtime_config)
    result = candidate.matmul_candidate.search_space.apply_operator_to_runtime_config(
        configured,
        PACKED_GATE_UP_OPERATOR,
    )
    mlp = result.setdefault("mlp", {})
    mlp["packed_gate_up_split_strategy"] = candidate.split_strategy
    mlp["packed_gate_up_split_output_memory_config"] = (
        candidate.split_output_memory.to_runtime_descriptor()
    )
    mlp["packed_gate_up_mul_input_memory_config"] = (
        candidate.mul_input_memory.to_runtime_descriptor()
    )
    mlp["packed_gate_up_mul_conversion"] = candidate.requires_mul_conversion
    return result


def apply_packed_gate_up_layer_group_candidates(
    runtime_config: Mapping[str, Any],
    candidates: Mapping[str, PackedGateUpCandidate],
) -> dict[str, Any]:
    expected = {DEFAULT_LAYER_GROUP, OVERRIDE_LAYER_GROUP}
    if set(candidates) != expected:
        raise PackedMLPError(
            "packed gate/up full-model config requires default and override groups"
        )
    default = candidates[DEFAULT_LAYER_GROUP]
    override = candidates[OVERRIDE_LAYER_GROUP]
    if default.layer_group != DEFAULT_LAYER_GROUP or default.representative_layer != 0:
        raise PackedMLPError("default packed gate/up candidate must represent layer 0")
    if (
        override.layer_group != OVERRIDE_LAYER_GROUP
        or override.representative_layer != 31
    ):
        raise PackedMLPError(
            "override packed gate/up candidate must represent layer 31"
        )

    result = apply_packed_gate_up_candidate(runtime_config, default)
    override_result = apply_packed_gate_up_candidate(runtime_config, override)
    default_weight = _nested_get(
        result, "parameter_config.weight_memory_config.mlp_gate_up"
    )
    override_weight = _nested_get(
        override_result, "parameter_config.weight_memory_config.mlp_gate_up"
    )
    if canonical_json(default_weight) != canonical_json(override_weight):
        raise PackedMLPError(
            "packed gate/up layer groups must use one offline weight memory layout"
        )

    mlp = result.setdefault("mlp", {})
    layer_overrides = mlp.setdefault("layer_overrides", {})
    layer_31 = layer_overrides.setdefault("31", {})
    source = override_result["mlp"]
    for field_name in _PACKED_RUNTIME_FIELDS:
        layer_31[field_name] = copy.deepcopy(source.get(field_name))
    result.setdefault("autotune", {})["packed_gate_up_layer_groups"] = {
        DEFAULT_LAYER_GROUP: default.to_dict(),
        OVERRIDE_LAYER_GROUP: override.to_dict(),
    }
    return result


def rank_packed_gate_up_region_candidates(
    enumeration: PackedGateUpEnumerationResult,
) -> tuple[MeasurementCandidate, ...]:
    operator_group = f"{PACKED_GATE_UP_OPERATOR}.{enumeration.layer_group}"
    matmul_rank = {
        item.candidate_id: item
        for item in rank_matmul_measurement_candidates(enumeration.matmul)
    }
    ranked = [
        MeasurementCandidate.create(
            candidate_id=enumeration.incumbent_candidate_id,
            operator_name=operator_group,
            candidate_kind="packed_mlp_region",
            analytical_score=0.0,
            l1_bytes=0,
            source="incumbent_separate_gate_up",
            is_incumbent=True,
            metadata={
                "operator": PACKED_GATE_UP_OPERATOR,
                "layer_group": enumeration.layer_group,
                "representative_layer": enumeration.representative_layer,
                "mode": "incumbent",
                "operation_sequence": [
                    "linear.gate",
                    "linear.up",
                    "mul_silu",
                ],
            },
        )
    ]
    for candidate in enumeration.candidates:
        base = matmul_rank[candidate.matmul_candidate.candidate_id]
        split_penalty = 0.0 if candidate.split_strategy == "split" else 0.05
        conversion_penalty = 10.0 if candidate.requires_mul_conversion else 0.0
        ranked.append(
            MeasurementCandidate.create(
                candidate_id=candidate.candidate_id,
                operator_name=operator_group,
                candidate_kind="packed_mlp_region",
                analytical_score=(
                    base.analytical_score + split_penalty + conversion_penalty
                ),
                l1_bytes=base.l1_bytes,
                source="packed_gate_up_independent_search",
                is_incumbent=False,
                metadata={
                    "operator": PACKED_GATE_UP_OPERATOR,
                    "layer_group": enumeration.layer_group,
                    "representative_layer": enumeration.representative_layer,
                    "mode": "challenger",
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


def select_packed_gate_up_region_winner(
    enumeration: PackedGateUpEnumerationResult,
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    statistic: str = "p50",
) -> dict[str, Any]:
    incumbent = _measurement_latency(
        measurements.get(enumeration.incumbent_candidate_id), statistic
    )
    ranked: list[dict[str, Any]] = []
    for candidate in enumeration.candidates:
        latency = _measurement_latency(
            measurements.get(candidate.candidate_id), statistic
        )
        if latency is None:
            continue
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "latency_ms": latency,
                "program_family": candidate.program_family,
                "split_strategy": candidate.split_strategy,
                "split_output_layout": candidate.split_output_memory.layout,
                "mul_input_layout": candidate.mul_input_memory.layout,
                "requires_mul_conversion": candidate.requires_mul_conversion,
            }
        )
    ranked.sort(key=lambda item: (item["latency_ms"], item["candidate_id"]))
    winner = ranked[0] if ranked else None
    gain = None
    if incumbent is not None and winner is not None:
        gain = (incumbent - float(winner["latency_ms"])) / incumbent
    gate_passed = gain is not None and gain >= PACKED_REGION_PROMOTION_GAIN
    return {
        "schema_version": PACKED_MLP_SCHEMA_VERSION,
        "status": (
            "selected" if incumbent is not None and winner is not None else "incomplete"
        ),
        "operator": PACKED_GATE_UP_OPERATOR,
        "layer_group": enumeration.layer_group,
        "representative_layer": enumeration.representative_layer,
        "statistic": statistic,
        "incumbent": {
            "candidate_id": enumeration.incumbent_candidate_id,
            "latency_ms": incumbent,
            "operation_sequence": ["linear.gate", "linear.up", "mul_silu"],
        },
        "winner": copy.deepcopy(winner),
        "ranked_challengers": ranked,
        "representative_region_gain": gain,
        "region_promotion_threshold": PACKED_REGION_PROMOTION_GAIN,
        "preferred_gain": PACKED_REGION_PREFERRED_GAIN,
        "region_gate_passed": gate_passed,
        "enter_full_model": gate_passed,
        "default_promotion_allowed": False,
    }


def build_packed_gate_up_phase_report(
    selections: Sequence[Mapping[str, Any]],
    *,
    full_model_gain: float | None = None,
) -> dict[str, Any]:
    by_group = {str(item.get("layer_group")): dict(item) for item in selections}
    expected = {DEFAULT_LAYER_GROUP, OVERRIDE_LAYER_GROUP}
    if set(by_group) != expected:
        raise PackedMLPError(
            "packed gate/up report requires layer 0 and layer 31 selections"
        )
    complete = all(item.get("status") == "selected" for item in by_group.values())
    region_passed = complete and all(
        item.get("region_gate_passed") is True for item in by_group.values()
    )
    weighted_incumbent = _weighted_group_latency(by_group, "incumbent")
    weighted_challenger = _weighted_group_latency(by_group, "winner")
    weighted_gain = None
    if weighted_incumbent is not None and weighted_challenger is not None:
        weighted_gain = (weighted_incumbent - weighted_challenger) / weighted_incumbent
    full_model_passed = (
        full_model_gain is not None and full_model_gain >= FULL_MODEL_PROMOTION_GAIN
    )
    return {
        "schema_version": PACKED_MLP_SCHEMA_VERSION,
        "stage": "packed-gate-up-independent-tuning",
        "status": "passed" if region_passed else "not_promoted",
        "operator": PACKED_GATE_UP_OPERATOR,
        "layer_groups": by_group,
        "weighted_region": {
            "layer_weights": {
                DEFAULT_LAYER_GROUP: 31,
                OVERRIDE_LAYER_GROUP: 1,
            },
            "incumbent_latency_ms": weighted_incumbent,
            "challenger_latency_ms": weighted_challenger,
            "gain": weighted_gain,
        },
        "enter_full_model": region_passed,
        "full_model_gain": full_model_gain,
        "full_model_promotion_threshold": FULL_MODEL_PROMOTION_GAIN,
        "default_promotion_allowed": region_passed and full_model_passed,
    }


def _packed_runtime_config(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    return apply_template_axis_updates(
        runtime_config,
        {
            GATE_UP_AXIS: PACKED_GATE_UP,
            ACTIVATION_AXIS: MUL_FUSED_SILU,
        },
    )


def _layer_group(representative_layer: int) -> str:
    try:
        return _LAYER_GROUPS[representative_layer]
    except KeyError as exc:
        raise PackedMLPError(
            "packed gate/up tuning only accepts representative layers 0 and 31"
        ) from exc


def _measurement_latency(
    measurement: Mapping[str, Any] | None,
    statistic: str,
) -> float | None:
    if not isinstance(measurement, Mapping) or measurement.get("status") != "passed":
        return None
    statistics = measurement.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get(statistic) is None:
        return None
    value = float(statistics[statistic])
    return value if math.isfinite(value) and value > 0 else None


def _weighted_group_latency(
    groups: Mapping[str, Mapping[str, Any]], key: str
) -> float | None:
    weights = {DEFAULT_LAYER_GROUP: 31, OVERRIDE_LAYER_GROUP: 1}
    total = 0.0
    for group, weight in weights.items():
        item = groups[group].get(key)
        if not isinstance(item, Mapping) or item.get("latency_ms") is None:
            return None
        total += float(item["latency_ms"]) * weight
    return total


def _search_field_coverage(
    candidates: Sequence[PackedGateUpCandidate],
) -> dict[str, list[Any]]:
    def values(getter):
        return sorted(
            {getter(candidate) for candidate in candidates},
            key=canonical_json,
        )

    return {
        "program_family": values(lambda item: item.program_family),
        "in0_block_w": values(
            lambda item: int(
                item.matmul_candidate.programs[0].parameters["in0_block_w"]
            )
        ),
        "per_core_N": values(
            lambda item: int(item.matmul_candidate.programs[0].parameters["per_core_n"])
        ),
        "grid": values(
            lambda item: canonical_json(
                item.matmul_candidate.programs[0].compute_grid.to_list()
                if item.matmul_candidate.programs[0].compute_grid is not None
                else None
            )
        ),
        "worker_cores": values(
            lambda item: item.matmul_candidate.worker_core_counts[0]
        ),
        "weight_sharding": values(
            lambda item: (
                item.matmul_candidate.search_space.operators[
                    PACKED_GATE_UP_OPERATOR
                ].get("weight_memory")
                or {}
            ).get("layout")
        ),
        "output_sharding": values(
            lambda item: (
                item.matmul_candidate.search_space.operators[
                    PACKED_GATE_UP_OPERATOR
                ].get("output_memory")
                or {}
            ).get("layout")
        ),
        "split_strategy": values(lambda item: item.split_strategy),
        "split_output_layout": values(lambda item: item.split_output_memory.layout),
        "mul_input_layout": values(lambda item: item.mul_input_memory.layout),
    }


def _unsupported_layout_attempts(layer_group: str) -> list[dict[str, Any]]:
    linear_rejections = [
        {
            "operator": PACKED_GATE_UP_OPERATOR,
            "layer_group": layer_group,
            "field": "linear_output_memory",
            "value": layout,
            "status": "static_rejected",
            "error_class": "unsupported_layout",
            "reason": (
                "the surviving DRAM-sharded and 1D multicast decode families "
                "require width-sharded packed linear output"
            ),
        }
        for layout in ("interleaved", "height_sharded", "block_sharded")
    ]
    split_rejections = [
        {
            "operator": PACKED_GATE_UP_OPERATOR,
            "layer_group": layer_group,
            "field": "split_output_memory",
            "value": "width_sharded",
            "split_strategy": strategy,
            "status": "static_rejected",
            "error_class": "unsupported_layout",
            "reason": (
                "the pinned TTNN split/slice API requires an explicit half-width "
                "shard spec; its named width-sharded memory config raises "
                "'bad optional access' and is not a legal production candidate"
            ),
        }
        for strategy in _SPLIT_STRATEGIES
    ]
    mul_rejection = {
        "operator": PACKED_GATE_UP_OPERATOR,
        "layer_group": layer_group,
        "field": "mul_input_memory",
        "value": "width_sharded",
        "status": "static_rejected",
        "error_class": "unsupported_layout",
        "reason": (
            "a width-sharded mul input would require the same unavailable "
            "explicit split shard spec or two additional hot-path conversions"
        ),
    }
    return [*linear_rejections, *split_rejections, mul_rejection]


def _nested_get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(component)
    return current
