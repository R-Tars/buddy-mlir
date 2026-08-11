from __future__ import annotations

import copy
import importlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .legality import DeviceDescriptor
from .schema import canonical_json, sha256_json
from .space import MemoryConfig, SearchSpaceConfig

LAYOUT_GRAPH_SCHEMA_VERSION = 1

ATTENTION_REGION = "attention"
MLP_REGION = "mlp"
LM_HEAD_REGION = "lm_head"
LLAMA_LAYOUT_REGIONS = (ATTENTION_REGION, MLP_REGION, LM_HEAD_REGION)

ATTENTION_DAG_NODES = (
    "attention.rms_norm",
    "attention.qkv",
    "attention.create_heads",
    "attention.rope",
    "attention.cache_update",
    "attention.sdpa",
    "attention.concat_heads",
    "attention.o_proj",
)
MLP_DAG_NODES = (
    "mlp.rms_norm",
    "mlp.gate",
    "mlp.up",
    "mlp.mul_silu",
    "mlp.down",
)
LM_HEAD_DAG_NODES = (
    "lm_head.final_norm",
    "lm_head.shards",
    "lm_head.concat",
    "lm_head.argmax",
)

MIN_BEAM_WIDTH = 4
MAX_BEAM_WIDTH = 16


class LayoutGraphError(ValueError):
    """Raised when a layout graph or its measurement evidence is invalid."""


@dataclass(frozen=True)
class LayoutMeasurement:
    value_ms: float
    metric: str
    source_kind: str
    source_report_sha256: str
    sample_count: int | None = None
    _metadata_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value_ms) or self.value_ms < 0:
            raise LayoutGraphError("layout cost must be a finite non-negative value")
        if not self.metric or not self.source_kind or not self.source_report_sha256:
            raise LayoutGraphError("layout cost must retain metric and source evidence")
        if self.sample_count is not None and self.sample_count <= 0:
            raise LayoutGraphError("sample_count must be positive when present")
        _json_object(self._metadata_json, "measurement metadata")

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
        *,
        metric: str = "p50",
        source_kind: str | None = None,
    ) -> "LayoutMeasurement":
        payload = copy.deepcopy(dict(report))
        status = str(payload.get("status", ""))
        passed = bool(payload.get("passed", status in {"passed", "profiled"}))
        if not passed or status not in {"passed", "profiled"}:
            raise LayoutGraphError("only passed measurement reports may provide costs")

        statistics = payload.get("statistics")
        latency = payload.get("latency_ms")
        value: Any = None
        sample_count: int | None = None
        detected_kind = source_kind
        if isinstance(statistics, Mapping):
            value = statistics.get(metric)
            raw_count = statistics.get("sample_count")
            sample_count = int(raw_count) if raw_count is not None else None
            detected_kind = detected_kind or "microbenchmark"
        elif isinstance(latency, Mapping):
            value = latency.get(metric)
            if value is None and metric == "p50":
                value = latency.get("median", latency.get("mean"))
            detected_kind = detected_kind or "profile"
        elif latency is not None:
            value = latency
            detected_kind = detected_kind or "profile"
        elif payload.get("decode_step_ms_p50") is not None:
            value = {
                "p50": payload.get("decode_step_ms_p50"),
                "mean": payload.get("decode_step_ms_mean"),
                "min": payload.get("decode_step_ms_min"),
                "max": payload.get("decode_step_ms_max"),
            }.get(metric)
            sample_count = len(payload.get("decode_step_ms_samples") or []) or None
            detected_kind = detected_kind or "steady_decode_profile"
        if value is None:
            raise LayoutGraphError(f"measurement report has no {metric!r} latency")
        return cls(
            value_ms=float(value),
            metric=metric,
            source_kind=detected_kind or "report",
            source_report_sha256=sha256_json(payload),
            sample_count=sample_count,
            _metadata_json=canonical_json(
                {
                    "status": status,
                    "candidate_fingerprint": payload.get("candidate_fingerprint"),
                    "target": payload.get("target"),
                }
            ),
        )

    @classmethod
    def structural_zero(cls, label: str) -> "LayoutMeasurement":
        evidence = {"kind": "structural_zero", "label": str(label)}
        return cls(
            value_ms=0.0,
            metric="structural_cost",
            source_kind="structural_estimate",
            source_report_sha256=sha256_json(evidence),
            _metadata_json=canonical_json(evidence),
        )

    @property
    def measured(self) -> bool:
        return self.source_kind != "structural_estimate"

    @property
    def metadata(self) -> dict[str, Any]:
        return _json_object(self._metadata_json, "measurement metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_ms": self.value_ms,
            "metric": self.metric,
            "source_kind": self.source_kind,
            "source_report_sha256": self.source_report_sha256,
            "sample_count": self.sample_count,
            "measured": self.measured,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LayoutNodeCandidate:
    node: str
    candidate_id: str
    inputs: tuple[tuple[str, MemoryConfig], ...]
    outputs: tuple[tuple[str, MemoryConfig], ...]
    latency: LayoutMeasurement
    memory_updates: tuple[tuple[str, MemoryConfig], ...] = ()
    source: str = "enumerated"
    eligible: bool = True
    eligibility_reason: str | None = None
    _validation_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if not self.node or not self.candidate_id or not self.source:
            raise LayoutGraphError("layout candidates need node, id, and source")
        _validate_named_memories(self.inputs, f"{self.candidate_id}.inputs")
        _validate_named_memories(self.outputs, f"{self.candidate_id}.outputs")
        _validate_named_memories(
            self.memory_updates,
            f"{self.candidate_id}.memory_updates",
        )
        if not self.eligible and not self.eligibility_reason:
            raise LayoutGraphError("ineligible candidates must retain a reason")
        _json_object(self._validation_json, "candidate validation evidence")

    @classmethod
    def create(
        cls,
        *,
        node: str,
        candidate_id: str,
        inputs: Mapping[str, MemoryConfig],
        outputs: Mapping[str, MemoryConfig],
        latency: LayoutMeasurement,
        memory_updates: Mapping[str, MemoryConfig] | None = None,
        source: str = "enumerated",
        eligible: bool = True,
        eligibility_reason: str | None = None,
        validation_evidence: Mapping[str, Any] | None = None,
    ) -> "LayoutNodeCandidate":
        return cls(
            node=str(node),
            candidate_id=str(candidate_id),
            inputs=_memory_items(inputs),
            outputs=_memory_items(outputs),
            latency=latency,
            memory_updates=_memory_items(memory_updates or {}),
            source=str(source),
            eligible=bool(eligible),
            eligibility_reason=eligibility_reason,
            _validation_json=canonical_json(validation_evidence or {}),
        )

    def input_memory(self, port: str) -> MemoryConfig:
        return _memory_at(self.inputs, port, self.candidate_id)

    def output_memory(self, port: str) -> MemoryConfig:
        return _memory_at(self.outputs, port, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "candidate_id": self.candidate_id,
            "source": self.source,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "validation_evidence": _json_object(
                self._validation_json,
                "candidate validation evidence",
            ),
            "inputs": {name: memory.to_dict() for name, memory in self.inputs},
            "outputs": {name: memory.to_dict() for name, memory in self.outputs},
            "latency": self.latency.to_dict(),
            "memory_updates": {
                path: memory.to_dict() for path, memory in self.memory_updates
            },
        }


@dataclass(frozen=True)
class LayoutEdge:
    name: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    tensor_shape: tuple[int, ...]
    incumbent_conversion: bool
    allow_explicit_conversion: bool = True
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not all(
            (
                self.name,
                self.source_node,
                self.source_port,
                self.target_node,
                self.target_port,
            )
        ):
            raise LayoutGraphError(
                "layout edge names, nodes, and ports must be non-empty"
            )
        if len(self.tensor_shape) < 2 or any(value <= 0 for value in self.tensor_shape):
            raise LayoutGraphError(
                "layout edge tensor shapes must be positive and rank >= 2"
            )
        if self.source_node == self.target_node:
            raise LayoutGraphError("layout edges cannot be self loops")
        if self.multiplicity <= 0:
            raise LayoutGraphError("layout edge multiplicity must be positive")

    @property
    def flattened_shape(self) -> tuple[int, int]:
        return (math.prod(self.tensor_shape[:-1]), self.tensor_shape[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": {"node": self.source_node, "port": self.source_port},
            "target": {"node": self.target_node, "port": self.target_port},
            "tensor_shape": list(self.tensor_shape),
            "flattened_shape": list(self.flattened_shape),
            "incumbent_conversion": self.incumbent_conversion,
            "allow_explicit_conversion": self.allow_explicit_conversion,
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True)
class LayoutLegalityIssue:
    code: str
    path: str
    message: str
    _details_json: str = field(default="{}", repr=False)

    @property
    def details(self) -> dict[str, Any]:
        return _json_object(self._details_json, "legality issue details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "details": self.details,
        }


def validate_layout_memory(
    memory: MemoryConfig,
    *,
    tensor_shape: Sequence[int],
    device: DeviceDescriptor,
    path: str = "memory",
) -> tuple[LayoutLegalityIssue, ...]:
    shape = tuple(int(value) for value in tensor_shape)
    if len(shape) < 2 or any(value <= 0 for value in shape):
        raise LayoutGraphError("layout tensor shape must be positive and rank >= 2")
    height, width = math.prod(shape[:-1]), shape[-1]
    issues: list[LayoutLegalityIssue] = []

    def issue(code: str, message: str, **details: Any) -> None:
        issues.append(
            LayoutLegalityIssue(
                code=code,
                path=path,
                message=message,
                _details_json=canonical_json(details),
            )
        )

    shard_fields = (memory.grid, memory.shard_shape, memory.orientation)
    if memory.layout == "interleaved":
        if any(value is not None for value in shard_fields):
            issue(
                "INTERLEAVED_HAS_SHARD_SPEC",
                "interleaved memory cannot carry grid, shard shape, or orientation",
            )
        return tuple(issues)

    if memory.buffer != "l1":
        issue(
            "ACTIVATION_SHARD_NOT_L1",
            "cross-op activation sharding is only legal in L1",
            buffer=memory.buffer,
        )

    if memory.runtime_kind == "ttnn_memory_config":
        if memory.runtime_name not in {
            "L1_WIDTH_SHARDED_MEMORY_CONFIG",
            "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
            "L1_BLOCK_SHARDED_MEMORY_CONFIG",
        }:
            issue(
                "UNKNOWN_NAMED_SHARD",
                "named sharded memory must be a supported TTNN built-in",
                runtime_name=memory.runtime_name,
            )
        if any(value is not None for value in shard_fields):
            issue(
                "NAMED_SHARD_HAS_EXPLICIT_SPEC",
                "named sharded memory cannot also carry an explicit shard spec",
            )
        return tuple(issues)

    if memory.runtime_kind != "ttnn_sharded_memory_config":
        issue(
            "UNSUPPORTED_SHARDED_RUNTIME_KIND",
            "activation sharding requires a TTNN sharded memory descriptor",
            runtime_kind=memory.runtime_kind,
        )
        return tuple(issues)
    if memory.grid is None or memory.shard_shape is None:
        issue(
            "SHARDED_SPEC_INCOMPLETE",
            "explicit sharding requires both grid and shard shape",
        )
        return tuple(issues)
    if (
        memory.grid.x > device.compute_grid.x
        or memory.grid.y > device.compute_grid.y
        or memory.grid.x * memory.grid.y > device.worker_core_count
    ):
        issue(
            "SHARD_GRID_OUT_OF_BOUNDS",
            "shard grid exceeds the device worker grid",
            grid=memory.grid.to_list(),
            device_grid=device.compute_grid.to_list(),
            worker_core_count=device.worker_core_count,
        )
    if memory.orientation is not None and memory.orientation.lower() not in {
        "row_major",
        "col_major",
    }:
        issue(
            "SHARD_ORIENTATION_UNSUPPORTED",
            "shard orientation must be row_major or col_major",
            orientation=memory.orientation,
        )

    shard_h, shard_w = memory.shard_shape
    core_count = memory.grid.x * memory.grid.y
    if memory.layout == "width_sharded":
        covers = shard_h == height and shard_w * core_count == width
    elif memory.layout == "height_sharded":
        covers = shard_w == width and shard_h * core_count == height
    else:
        covers = shard_h * shard_w * core_count == height * width
    if not covers:
        issue(
            "SHARD_SHAPE_MISMATCH",
            "shards do not exactly cover the padded tensor",
            tensor_shape=[height, width],
            shard_shape=list(memory.shard_shape),
            core_count=core_count,
            layout=memory.layout,
        )
    return tuple(issues)


@dataclass(frozen=True)
class ConversionCost:
    edge_name: str
    producer_memory_sha256: str
    consumer_memory_sha256: str
    measurement: LayoutMeasurement

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_name": self.edge_name,
            "producer_memory_sha256": self.producer_memory_sha256,
            "consumer_memory_sha256": self.consumer_memory_sha256,
            "measurement": self.measurement.to_dict(),
        }


class ConversionCostTable:
    def __init__(self, entries: Iterable[ConversionCost] = ()) -> None:
        self._entries: dict[tuple[str, str, str], ConversionCost] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: ConversionCost) -> None:
        if not entry.measurement.measured:
            raise LayoutGraphError("conversion costs must come from measured reports")
        key = (
            entry.edge_name,
            entry.producer_memory_sha256,
            entry.consumer_memory_sha256,
        )
        existing = self._entries.get(key)
        if existing is not None and existing != entry:
            raise LayoutGraphError("conflicting conversion measurements for one edge")
        self._entries[key] = entry

    def record_report(
        self,
        *,
        edge_name: str,
        producer_memory: MemoryConfig,
        consumer_memory: MemoryConfig,
        report: Mapping[str, Any],
        metric: str = "p50",
    ) -> ConversionCost:
        entry = ConversionCost(
            edge_name=str(edge_name),
            producer_memory_sha256=memory_fingerprint(producer_memory),
            consumer_memory_sha256=memory_fingerprint(consumer_memory),
            measurement=LayoutMeasurement.from_report(
                report,
                metric=metric,
                source_kind="conversion_microbenchmark",
            ),
        )
        self.add(entry)
        return entry

    def lookup(
        self,
        edge_name: str,
        producer_memory: MemoryConfig,
        consumer_memory: MemoryConfig,
    ) -> ConversionCost | None:
        return self._entries.get(
            (
                edge_name,
                memory_fingerprint(producer_memory),
                memory_fingerprint(consumer_memory),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAYOUT_GRAPH_SCHEMA_VERSION,
            "entry_count": len(self._entries),
            "entries": [self._entries[key].to_dict() for key in sorted(self._entries)],
        }


@dataclass(frozen=True)
class LayoutRegionGraph:
    region: str
    candidates: tuple[LayoutNodeCandidate, ...]
    edges: tuple[LayoutEdge, ...]

    def __post_init__(self) -> None:
        if not self.region or not self.candidates:
            raise LayoutGraphError("layout region graphs need a name and candidates")
        grouped = self.candidates_by_node
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise LayoutGraphError("layout candidate ids must be globally unique")
        edge_names = [edge.name for edge in self.edges]
        if len(set(edge_names)) != len(edge_names):
            raise LayoutGraphError("layout edge names must be unique")
        for edge in self.edges:
            if edge.source_node not in grouped or edge.target_node not in grouped:
                raise LayoutGraphError(f"edge {edge.name!r} references an unknown node")
            for candidate in grouped[edge.source_node]:
                candidate.output_memory(edge.source_port)
            for candidate in grouped[edge.target_node]:
                candidate.input_memory(edge.target_port)
        self.topological_order()

    @property
    def candidates_by_node(self) -> dict[str, tuple[LayoutNodeCandidate, ...]]:
        grouped: dict[str, list[LayoutNodeCandidate]] = defaultdict(list)
        for candidate in self.candidates:
            grouped[candidate.node].append(candidate)
        return {
            node: tuple(sorted(values, key=lambda item: item.candidate_id))
            for node, values in grouped.items()
        }

    def topological_order(self) -> tuple[str, ...]:
        nodes = sorted(self.candidates_by_node)
        indegree = {node: 0 for node in nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            outgoing[edge.source_node].append(edge.target_node)
            indegree[edge.target_node] += 1
        ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        ordered: list[str] = []
        while ready:
            node = ready.popleft()
            ordered.append(node)
            for target in sorted(outgoing[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(ordered) != len(nodes):
            raise LayoutGraphError(f"layout region {self.region!r} contains a cycle")
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAYOUT_GRAPH_SCHEMA_VERSION,
            "region": self.region,
            "topological_order": list(self.topological_order()),
            "node_count": len(self.candidates_by_node),
            "candidate_count": len(self.candidates),
            "edges": [edge.to_dict() for edge in self.edges],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class _BeamState:
    selections: tuple[tuple[str, LayoutNodeCandidate], ...]
    op_latency_ms: float
    conversion_latency_ms: float
    conversions: tuple[dict[str, Any], ...]

    @property
    def total_latency_ms(self) -> float:
        return self.op_latency_ms + self.conversion_latency_ms

    @property
    def selected(self) -> dict[str, LayoutNodeCandidate]:
        return dict(self.selections)


@dataclass(frozen=True)
class LayoutSearchResult:
    graph: LayoutRegionGraph = field(repr=False)
    beam_width: int
    selected_candidates: tuple[LayoutNodeCandidate, ...]
    op_latency_ms: float
    conversion_latency_ms: float
    conversions: tuple[dict[str, Any], ...]
    rejected_transitions: tuple[dict[str, Any], ...]
    expanded_state_count: int
    peak_beam_size: int
    cartesian_path_count: int

    @property
    def total_latency_ms(self) -> float:
        return self.op_latency_ms + self.conversion_latency_ms

    @property
    def removed_conversions(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for item in self.conversions
            if item["action"] == "removed"
        ]

    @property
    def retained_conversions(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for item in self.conversions
            if item["action"] == "retained"
        ]

    @property
    def already_elided_conversions(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for item in self.conversions
            if item["action"] == "already_elided"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAYOUT_GRAPH_SCHEMA_VERSION,
            "status": "passed",
            "algorithm": "topological_constrained_beam_search",
            "cartesian_exhaustive_search": False,
            "region": self.graph.region,
            "beam_width": self.beam_width,
            "topological_order": list(self.graph.topological_order()),
            "selected_candidate_ids": [
                candidate.candidate_id for candidate in self.selected_candidates
            ],
            "selected_candidates": [
                candidate.to_dict() for candidate in self.selected_candidates
            ],
            "cost_ms": {
                "operators": self.op_latency_ms,
                "conversions": self.conversion_latency_ms,
                "total": self.total_latency_ms,
            },
            "removed_conversions": self.removed_conversions,
            "retained_conversions": self.retained_conversions,
            "already_elided_conversions": self.already_elided_conversions,
            "all_edge_decisions": [copy.deepcopy(item) for item in self.conversions],
            "legality": {
                "selected_path_legal": True,
                "illegal_or_unmeasured_transition_count": len(
                    self.rejected_transitions
                ),
                "rejected_transitions": [
                    copy.deepcopy(item) for item in self.rejected_transitions
                ],
            },
            "search_statistics": {
                "expanded_state_count": self.expanded_state_count,
                "peak_beam_size": self.peak_beam_size,
                "cartesian_path_count": self.cartesian_path_count,
            },
            "graph_sha256": sha256_json(self.graph.to_dict()),
        }


def search_layout_graph(
    graph: LayoutRegionGraph,
    *,
    conversion_costs: ConversionCostTable,
    device: DeviceDescriptor,
    beam_width: int = 8,
    require_measured_op_costs: bool = False,
) -> LayoutSearchResult:
    if not MIN_BEAM_WIDTH <= beam_width <= MAX_BEAM_WIDTH:
        raise LayoutGraphError(
            f"beam_width must be in [{MIN_BEAM_WIDTH}, {MAX_BEAM_WIDTH}]"
        )
    grouped = graph.candidates_by_node
    if require_measured_op_costs:
        unmeasured = sorted(
            candidate.candidate_id
            for candidate in graph.candidates
            if candidate.eligible and not candidate.latency.measured
        )
        if unmeasured:
            raise LayoutGraphError(
                "layout search requires measured operator costs; missing: "
                + ", ".join(unmeasured)
            )
    incoming: dict[str, list[LayoutEdge]] = defaultdict(list)
    connected: dict[str, list[tuple[LayoutEdge, str]]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_node].append(edge)
        connected[edge.source_node].append((edge, "output"))
        connected[edge.target_node].append((edge, "input"))

    legality: dict[str, tuple[LayoutLegalityIssue, ...]] = {}
    for candidate in graph.candidates:
        issues: list[LayoutLegalityIssue] = []
        seen: set[tuple[str, str]] = set()
        for edge, direction in connected[candidate.node]:
            port = edge.source_port if direction == "output" else edge.target_port
            key = (direction, port)
            if key in seen:
                continue
            seen.add(key)
            memory = (
                candidate.output_memory(port)
                if direction == "output"
                else candidate.input_memory(port)
            )
            issues.extend(
                validate_layout_memory(
                    memory,
                    tensor_shape=edge.tensor_shape,
                    device=device,
                    path=f"{candidate.candidate_id}.{direction}s.{port}",
                )
            )
        legality[candidate.candidate_id] = tuple(issues)

    rejected: list[dict[str, Any]] = []
    states = [
        _BeamState(
            selections=(),
            op_latency_ms=0.0,
            conversion_latency_ms=0.0,
            conversions=(),
        )
    ]
    expanded = 0
    peak_beam = 1
    for node in graph.topological_order():
        expanded_states: list[_BeamState] = []
        for state in states:
            selected = state.selected
            for candidate in grouped[node]:
                expanded += 1
                if not candidate.eligible:
                    rejected.append(
                        {
                            "node": node,
                            "candidate_id": candidate.candidate_id,
                            "reason": "candidate_validation_failed",
                            "detail": candidate.eligibility_reason,
                            "validation_evidence": _json_object(
                                candidate._validation_json,
                                "candidate validation evidence",
                            ),
                        }
                    )
                    continue
                candidate_issues = legality[candidate.candidate_id]
                if candidate_issues:
                    rejected.append(
                        {
                            "node": node,
                            "candidate_id": candidate.candidate_id,
                            "reason": "illegal_sharding",
                            "issues": [issue.to_dict() for issue in candidate_issues],
                        }
                    )
                    continue
                decisions: list[dict[str, Any]] = []
                conversion_latency = 0.0
                compatible = True
                for edge in sorted(incoming[node], key=lambda item: item.name):
                    producer = selected[edge.source_node]
                    producer_memory = producer.output_memory(edge.source_port)
                    consumer_memory = candidate.input_memory(edge.target_port)
                    decision = _transition_decision(
                        edge,
                        producer_memory,
                        consumer_memory,
                        conversion_costs,
                    )
                    if decision is None:
                        compatible = False
                        rejected.append(
                            {
                                "edge": edge.name,
                                "producer_candidate_id": producer.candidate_id,
                                "consumer_candidate_id": candidate.candidate_id,
                                "reason": (
                                    "explicit_conversion_disabled"
                                    if not edge.allow_explicit_conversion
                                    else "conversion_cost_not_measured"
                                ),
                                "producer_memory": producer_memory.to_dict(),
                                "consumer_memory": consumer_memory.to_dict(),
                            }
                        )
                        break
                    decisions.append(decision)
                    conversion_latency += float(decision["latency_ms"])
                if not compatible:
                    continue
                expanded_states.append(
                    _BeamState(
                        selections=(*state.selections, (node, candidate)),
                        op_latency_ms=(
                            state.op_latency_ms + candidate.latency.value_ms
                        ),
                        conversion_latency_ms=(
                            state.conversion_latency_ms + conversion_latency
                        ),
                        conversions=(*state.conversions, *decisions),
                    )
                )
        if not expanded_states:
            raise LayoutGraphError(
                f"layout search has no compatible path after node {node!r}"
            )
        expanded_states.sort(key=_beam_rank)
        states = expanded_states[:beam_width]
        peak_beam = max(peak_beam, len(states))

    winner = min(states, key=_beam_rank)
    selected = winner.selected
    ordered_candidates = tuple(selected[node] for node in graph.topological_order())
    cartesian_paths = math.prod(len(grouped[node]) for node in grouped)
    return LayoutSearchResult(
        graph=graph,
        beam_width=beam_width,
        selected_candidates=ordered_candidates,
        op_latency_ms=winner.op_latency_ms,
        conversion_latency_ms=winner.conversion_latency_ms,
        conversions=winner.conversions,
        rejected_transitions=tuple(_unique_dicts(rejected)),
        expanded_state_count=expanded,
        peak_beam_size=peak_beam,
        cartesian_path_count=cartesian_paths,
    )


def apply_layout_result(
    base_space: SearchSpaceConfig,
    result: LayoutSearchResult,
) -> SearchSpaceConfig:
    payload = base_space.to_dict()
    updates: dict[str, MemoryConfig] = {}
    for candidate in result.selected_candidates:
        for path, memory in candidate.memory_updates:
            existing = updates.get(path)
            if existing is not None and canonical_json(
                existing.to_dict()
            ) != canonical_json(memory.to_dict()):
                raise LayoutGraphError(f"selected candidates conflict at {path!r}")
            updates[path] = memory
    for path, memory in updates.items():
        payload["memory_configs"][path] = memory.to_dict()
        _sync_operator_memory(payload, path, memory)

    for edge in payload["edges"].values():
        producer_path = edge.get("producer_path")
        consumer_path = edge.get("consumer_path")
        producer = payload["memory_configs"].get(producer_path)
        consumer = payload["memory_configs"].get(consumer_path)
        if isinstance(producer, Mapping) and isinstance(consumer, Mapping):
            edge["producer_output_memory"] = copy.deepcopy(producer)
            edge["consumer_input_memory"] = copy.deepcopy(consumer)
            edge["conversion"] = (
                "none"
                if canonical_json(producer) == canonical_json(consumer)
                else "explicit"
            )
    return SearchSpaceConfig.from_dict(payload)


def confirm_whole_layer_no_regression(
    *,
    search_result: LayoutSearchResult,
    incumbent_report: Mapping[str, Any],
    challenger_report: Mapping[str, Any],
    metric: str = "p50",
    max_regression_fraction: float = 0.0,
) -> dict[str, Any]:
    if max_regression_fraction < 0:
        raise LayoutGraphError("max_regression_fraction must be non-negative")
    incumbent = LayoutMeasurement.from_report(
        incumbent_report,
        metric=metric,
        source_kind="whole_layer_incumbent",
    )
    challenger_error = None
    try:
        challenger = LayoutMeasurement.from_report(
            challenger_report,
            metric=metric,
            source_kind="whole_layer_challenger",
        )
    except LayoutGraphError as err:
        challenger = None
        challenger_error = str(err)
    limit = incumbent.value_ms * (1.0 + max_regression_fraction)
    promote = challenger is not None and challenger.value_ms <= limit
    selected_latency = (
        challenger.value_ms
        if promote and challenger is not None
        else incumbent.value_ms
    )
    return {
        "schema_version": LAYOUT_GRAPH_SCHEMA_VERSION,
        "status": "passed",
        "gate": "whole_layer_no_regression",
        "region": search_result.graph.region,
        "search_winner_candidate_ids": [
            candidate.candidate_id for candidate in search_result.selected_candidates
        ],
        "decision": "promote_challenger" if promote else "retain_incumbent",
        "challenger_promoted": promote,
        "incumbent": incumbent.to_dict(),
        "challenger": (
            challenger.to_dict()
            if challenger is not None
            else {
                "status": "invalid",
                "error": challenger_error,
                "report_status": challenger_report.get("status"),
                "report_error": challenger_report.get("error"),
            }
        ),
        "max_regression_fraction": max_regression_fraction,
        "acceptance_limit_ms": limit,
        "selected_latency_ms": selected_latency,
        "no_regression": selected_latency <= limit,
        "incumbent_report_sha256": sha256_json(dict(incumbent_report)),
        "challenger_report_sha256": sha256_json(dict(challenger_report)),
    }


def write_layout_search_report(
    out: str | Path,
    result: LayoutSearchResult,
    *,
    conversion_costs: ConversionCostTable | None = None,
    whole_layer_confirmation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = result.to_dict()
    if conversion_costs is not None:
        report["conversion_cost_measurements"] = conversion_costs.to_dict()
    if whole_layer_confirmation is not None:
        report["whole_layer_confirmation"] = copy.deepcopy(
            dict(whole_layer_confirmation)
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def layout_conversion_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """Measure one real TTNN memory-layout conversion in an isolated worker."""
    from ..runtime.config_runtime import realize_ttnn_config
    from ..runtime.device import managed_ttnn_device
    from .microbench import make_worker_response

    payload = request.get("payload")
    candidate = request.get("candidate")
    if not isinstance(payload, Mapping) or not isinstance(candidate, Mapping):
        raise LayoutGraphError("conversion worker request is missing payload/candidate")
    measurement = candidate.get("measurement_contract")
    if not isinstance(measurement, Mapping):
        raise LayoutGraphError("conversion worker has no measurement contract")
    source_descriptor = payload.get("source_memory")
    target_descriptor = payload.get("target_memory")
    shape_value = payload.get("shape")
    if not isinstance(source_descriptor, Mapping) or not isinstance(
        target_descriptor, Mapping
    ):
        raise LayoutGraphError("conversion worker memory descriptors must be objects")
    if not isinstance(shape_value, list) or len(shape_value) < 2:
        raise LayoutGraphError("conversion worker shape must be a rank >= 2 list")
    shape = tuple(int(value) for value in shape_value)
    if any(value <= 0 for value in shape):
        raise LayoutGraphError("conversion worker shape dimensions must be positive")
    source_schema = MemoryConfig.from_runtime_descriptor(source_descriptor)
    target_schema = MemoryConfig.from_runtime_descriptor(target_descriptor)
    if memory_fingerprint(source_schema) == memory_fingerprint(target_schema):
        raise LayoutGraphError("conversion worker requires distinct memory layouts")

    warmup = int(measurement["warmup"])
    iterations = int(measurement["iterations"])
    device_id = int(payload.get("device_id", 0))
    ttnn = importlib.import_module("ttnn")
    torch = importlib.import_module("torch")
    source_memory = realize_ttnn_config(dict(source_descriptor), ttnn)
    target_memory = realize_ttnn_config(dict(target_descriptor), ttnn)
    dtype = getattr(ttnn, str(payload.get("dtype", "bfloat16")))
    layout = getattr(ttnn, str(payload.get("layout", "TILE_LAYOUT")))
    torch.manual_seed(int(payload.get("seed", 0)))
    host_tensor = torch.randn(shape, dtype=torch.bfloat16)

    samples: list[float] = []
    with managed_ttnn_device(ttnn, device_id) as device:
        source = ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=layout,
            memory_config=source_memory,
            device=device,
        )
        _synchronize(ttnn, device)
        for _ in range(warmup):
            converted = ttnn.to_memory_config(
                source,
                memory_config=target_memory,
            )
            _synchronize(ttnn, device)
            _deallocate(ttnn, converted)
        for _ in range(iterations):
            start = time.perf_counter()
            converted = ttnn.to_memory_config(
                source,
                memory_config=target_memory,
            )
            _synchronize(ttnn, device)
            samples.append((time.perf_counter() - start) * 1000.0)
            _deallocate(ttnn, converted)
        count = getattr(device, "num_program_cache_entries", None)
        program_cache_count = int(count()) if callable(count) else None
        _deallocate(ttnn, source)

    return make_worker_response(
        request,
        samples,
        program_cache_count=program_cache_count,
        trace_capture_count=0,
        new_tensor_allocations=iterations,
        metadata={
            "worker_kind": "ttnn_layout_conversion",
            "shape": list(shape),
            "source_memory": source_schema.to_dict(),
            "target_memory": target_schema.to_dict(),
            "synchronized_each_iteration": True,
        },
    )


def build_llama_layout_graphs(
    base_space: SearchSpaceConfig,
    *,
    batch_size: int = 32,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    cache_len: int = 1024,
    vocab_size: int = 128256,
    lm_head_split_count: int = 8,
    op_measurements: Mapping[str, LayoutMeasurement | Mapping[str, Any]] | None = None,
) -> tuple[LayoutRegionGraph, ...]:
    dimensions = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "cache_len": cache_len,
        "vocab_size": vocab_size,
        "lm_head_split_count": lm_head_split_count,
    }
    if any(value <= 0 for value in dimensions.values()):
        raise LayoutGraphError("Llama layout graph dimensions must be positive")
    if hidden_size != num_heads * head_dim:
        raise LayoutGraphError("hidden_size must equal num_heads * head_dim")
    measurements = op_measurements or {}
    return (
        _build_attention_graph(base_space, dimensions, measurements),
        _build_mlp_graph(base_space, dimensions, measurements),
        _build_lm_head_graph(base_space, dimensions, measurements),
    )


def _build_attention_graph(
    space: SearchSpaceConfig,
    dimensions: Mapping[str, int],
    measurements: Mapping[str, LayoutMeasurement | Mapping[str, Any]],
) -> LayoutRegionGraph:
    batch = dimensions["batch_size"]
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_heads"]
    kv_heads = dimensions["num_kv_heads"]
    head_dim = dimensions["head_dim"]
    cache_len = dimensions["cache_len"]
    qkv_width = (heads + 2 * kv_heads) * head_dim

    norm = _space_memory(
        space,
        "rms_norm.attention.output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    qkv = _space_memory(
        space,
        "attention.qkv_output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    heads_memory = _space_memory(
        space,
        "attention.qkv_heads_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    q_memory = _space_memory(
        space,
        "attention.fused_q_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    k_memory = _space_memory(
        space,
        "attention.fused_k_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    v_memory = _space_memory(
        space,
        "attention.fused_cache_value_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    cache_k_input = _space_memory(
        space,
        "attention.fused_cache_key_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    cache_v_input = _space_memory(
        space,
        "attention.fused_cache_value_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    cache_memory = MemoryConfig.named("DRAM_MEMORY_CONFIG")
    kernel_output = _space_memory(
        space,
        "attention.sdpa_kernel_output_memory_config",
        "DRAM_MEMORY_CONFIG",
    )
    concat_input = _space_memory(
        space,
        "attention.concat_heads_input_memory_config",
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
    )
    hidden_after_concat = MemoryConfig.named("L1_WIDTH_SHARDED_MEMORY_CONFIG")
    o_output = _space_memory(
        space,
        "attention.o_proj_output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )

    candidates = [
        _candidate(
            "attention.rms_norm",
            inputs={"hidden": norm},
            outputs={"hidden": norm},
            measurements=measurements,
        ),
        _candidate(
            "attention.qkv",
            inputs={"hidden": norm},
            outputs={"qkv": qkv},
            measurements=measurements,
        ),
        _candidate(
            "attention.create_heads",
            inputs={"qkv": qkv},
            outputs={"qk": heads_memory, "v": v_memory},
            measurements=measurements,
        ),
        _candidate(
            "attention.rope",
            inputs={"qk": heads_memory},
            outputs={"q": q_memory, "k": k_memory},
            measurements=measurements,
        ),
        _candidate(
            "attention.cache_update",
            inputs={"k": cache_k_input, "v": cache_v_input},
            outputs={"cache": cache_memory},
            measurements=measurements,
        ),
        _candidate(
            "attention.sdpa",
            candidate_id="attention.sdpa.incumbent",
            inputs={"q": q_memory, "cache": cache_memory},
            outputs={"attention": kernel_output},
            measurements=measurements,
            source="incumbent",
        ),
        _candidate(
            "attention.concat_heads",
            inputs={"attention": concat_input},
            outputs={"hidden": hidden_after_concat},
            measurements=measurements,
        ),
        _candidate(
            "attention.o_proj",
            inputs={"hidden": hidden_after_concat},
            outputs={"hidden": o_output},
            measurements=measurements,
        ),
    ]
    if memory_fingerprint(kernel_output) != memory_fingerprint(concat_input):
        candidates.append(
            _candidate(
                "attention.sdpa",
                candidate_id="attention.sdpa.direct_concat_layout",
                inputs={"q": q_memory, "cache": cache_memory},
                outputs={"attention": concat_input},
                memory_updates={
                    "attention.sdpa_kernel_output_memory_config": concat_input,
                },
                measurements=measurements,
                source="producer_compatible",
                eligible=(heads == kv_heads or concat_input.layout == "interleaved"),
                eligibility_reason=(
                    None
                    if heads == kv_heads or concat_input.layout == "interleaved"
                    else "TTNN SDPA decode does not support sharded output for GQA"
                ),
                validation_evidence={
                    "kind": "active_ttnn_runtime_constraint",
                    "code": "SDPA_GQA_SHARDED_OUTPUT_UNSUPPORTED",
                    "num_heads": heads,
                    "num_kv_heads": kv_heads,
                    "output_layout": concat_input.layout,
                },
            )
        )

    query_head_shape = (batch * heads, head_dim)
    qk_head_shape = _physical_sharded_shape(
        heads_memory,
        fallback=query_head_shape,
    )
    # TTNN decode pads K/V to one 32-row shard per user even under GQA.
    kv_head_shape = (batch * heads, head_dim)
    return LayoutRegionGraph(
        region=ATTENTION_REGION,
        candidates=tuple(candidates),
        edges=(
            LayoutEdge(
                "rms_norm_attention_to_qkv",
                "attention.rms_norm",
                "hidden",
                "attention.qkv",
                "hidden",
                (batch, hidden),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "qkv_to_create_heads",
                "attention.qkv",
                "qkv",
                "attention.create_heads",
                "qkv",
                (batch, qkv_width),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "create_heads_to_rope",
                "attention.create_heads",
                "qk",
                "attention.rope",
                "qk",
                qk_head_shape,
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "rope_k_to_cache_update",
                "attention.rope",
                "k",
                "attention.cache_update",
                "k",
                kv_head_shape,
                incumbent_conversion=(
                    memory_fingerprint(k_memory)
                    != memory_fingerprint(cache_k_input)
                ),
            ),
            LayoutEdge(
                "v_to_cache_update",
                "attention.create_heads",
                "v",
                "attention.cache_update",
                "v",
                kv_head_shape,
                incumbent_conversion=(
                    memory_fingerprint(v_memory)
                    != memory_fingerprint(cache_v_input)
                ),
            ),
            LayoutEdge(
                "rope_q_to_sdpa",
                "attention.rope",
                "q",
                "attention.sdpa",
                "q",
                query_head_shape,
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "cache_update_to_sdpa",
                "attention.cache_update",
                "cache",
                "attention.sdpa",
                "cache",
                (batch * kv_heads, cache_len, head_dim),
                incumbent_conversion=False,
                allow_explicit_conversion=False,
            ),
            LayoutEdge(
                "sdpa_to_concat_heads",
                "attention.sdpa",
                "attention",
                "attention.concat_heads",
                "attention",
                query_head_shape,
                incumbent_conversion=(
                    memory_fingerprint(kernel_output)
                    != memory_fingerprint(concat_input)
                ),
            ),
            LayoutEdge(
                "concat_heads_to_o_proj",
                "attention.concat_heads",
                "hidden",
                "attention.o_proj",
                "hidden",
                (batch, hidden),
                incumbent_conversion=False,
            ),
        ),
    )


def _build_mlp_graph(
    space: SearchSpaceConfig,
    dimensions: Mapping[str, int],
    measurements: Mapping[str, LayoutMeasurement | Mapping[str, Any]],
) -> LayoutRegionGraph:
    batch = dimensions["batch_size"]
    hidden = dimensions["hidden_size"]
    intermediate = dimensions["intermediate_size"]
    norm = _space_memory(
        space,
        "rms_norm.mlp.output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    gate = _space_memory(
        space,
        "mlp.gate_output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    up = _space_memory(
        space,
        "mlp.up_output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    down = _space_memory(
        space,
        "mlp.down_output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    candidates = (
        _candidate(
            "mlp.rms_norm",
            inputs={"hidden": norm},
            outputs={"hidden": norm},
            measurements=measurements,
        ),
        _candidate(
            "mlp.gate",
            inputs={"hidden": norm},
            outputs={"gate": gate},
            measurements=measurements,
        ),
        _candidate(
            "mlp.up",
            inputs={"hidden": norm},
            outputs={"up": up},
            measurements=measurements,
        ),
        _candidate(
            "mlp.mul_silu",
            inputs={"gate": gate, "up": up},
            outputs={"activation": gate},
            measurements=measurements,
        ),
        _candidate(
            "mlp.down",
            inputs={"activation": gate},
            outputs={"hidden": down},
            measurements=measurements,
        ),
    )
    return LayoutRegionGraph(
        region=MLP_REGION,
        candidates=candidates,
        edges=(
            LayoutEdge(
                "rms_norm_mlp_to_gate",
                "mlp.rms_norm",
                "hidden",
                "mlp.gate",
                "hidden",
                (batch, hidden),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "rms_norm_mlp_to_up",
                "mlp.rms_norm",
                "hidden",
                "mlp.up",
                "hidden",
                (batch, hidden),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "gate_to_mul_silu",
                "mlp.gate",
                "gate",
                "mlp.mul_silu",
                "gate",
                (batch, intermediate),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "up_to_mul_silu",
                "mlp.up",
                "up",
                "mlp.mul_silu",
                "up",
                (batch, intermediate),
                incumbent_conversion=False,
            ),
            LayoutEdge(
                "mul_silu_to_down",
                "mlp.mul_silu",
                "activation",
                "mlp.down",
                "activation",
                (batch, intermediate),
                incumbent_conversion=False,
            ),
        ),
    )


def _build_lm_head_graph(
    space: SearchSpaceConfig,
    dimensions: Mapping[str, int],
    measurements: Mapping[str, LayoutMeasurement | Mapping[str, Any]],
) -> LayoutRegionGraph:
    batch = dimensions["batch_size"]
    hidden = dimensions["hidden_size"]
    vocab = dimensions["vocab_size"]
    split_count = dimensions["lm_head_split_count"]
    if vocab % split_count:
        raise LayoutGraphError("vocab_size must be divisible by lm_head_split_count")
    shard_width = vocab // split_count
    final_norm = _space_memory(
        space,
        "rms_norm.final.output_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    lm_input = _space_memory(
        space,
        "lm_head.input_memory_config",
        "L1_WIDTH_SHARDED_MEMORY_CONFIG",
    )
    shard_output = _space_memory(
        space,
        "lm_head.shard_output_memory_config",
        "L1_MEMORY_CONFIG",
    )
    concat = _space_memory(
        space,
        "lm_head.concat_memory_config",
        "L1_MEMORY_CONFIG",
    )
    candidates = (
        _candidate(
            "lm_head.final_norm",
            inputs={"hidden": final_norm},
            outputs={"hidden": final_norm},
            measurements=measurements,
        ),
        _candidate(
            "lm_head.shards",
            inputs={"hidden": lm_input},
            outputs={"logit_shards": shard_output},
            measurements=measurements,
        ),
        _candidate(
            "lm_head.concat",
            inputs={"logit_shards": concat},
            outputs={"logits": concat},
            measurements=measurements,
        ),
        _candidate(
            "lm_head.argmax",
            inputs={"logits": concat},
            outputs={"token": concat},
            measurements=measurements,
        ),
    )
    return LayoutRegionGraph(
        region=LM_HEAD_REGION,
        candidates=candidates,
        edges=(
            LayoutEdge(
                "final_norm_to_lm_head",
                "lm_head.final_norm",
                "hidden",
                "lm_head.shards",
                "hidden",
                (batch, hidden),
                incumbent_conversion=(
                    memory_fingerprint(final_norm) != memory_fingerprint(lm_input)
                ),
            ),
            LayoutEdge(
                "lm_head_shards_to_concat",
                "lm_head.shards",
                "logit_shards",
                "lm_head.concat",
                "logit_shards",
                (batch, shard_width),
                incumbent_conversion=(
                    memory_fingerprint(shard_output) != memory_fingerprint(concat)
                ),
                multiplicity=split_count,
            ),
            LayoutEdge(
                "lm_head_concat_to_argmax",
                "lm_head.concat",
                "logits",
                "lm_head.argmax",
                "logits",
                (batch, vocab),
                incumbent_conversion=False,
            ),
        ),
    )


def _candidate(
    node: str,
    *,
    inputs: Mapping[str, MemoryConfig],
    outputs: Mapping[str, MemoryConfig],
    measurements: Mapping[str, LayoutMeasurement | Mapping[str, Any]],
    candidate_id: str | None = None,
    memory_updates: Mapping[str, MemoryConfig] | None = None,
    source: str = "incumbent",
    eligible: bool = True,
    eligibility_reason: str | None = None,
    validation_evidence: Mapping[str, Any] | None = None,
) -> LayoutNodeCandidate:
    candidate_id = candidate_id or f"{node}.incumbent"
    value = measurements.get(candidate_id, measurements.get(node))
    if value is None:
        latency = LayoutMeasurement.structural_zero(candidate_id)
    elif isinstance(value, LayoutMeasurement):
        latency = value
    elif isinstance(value, Mapping):
        latency = LayoutMeasurement.from_report(value)
    else:
        raise LayoutGraphError(
            f"measurement for {candidate_id!r} must be a report or LayoutMeasurement"
        )
    return LayoutNodeCandidate.create(
        node=node,
        candidate_id=candidate_id,
        inputs=inputs,
        outputs=outputs,
        latency=latency,
        memory_updates=memory_updates,
        source=source,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        validation_evidence=validation_evidence,
    )


def _space_memory(
    space: SearchSpaceConfig,
    path: str,
    fallback_name: str,
) -> MemoryConfig:
    descriptor = space.memory_configs.get(path)
    if isinstance(descriptor, Mapping):
        return MemoryConfig.from_dict(descriptor)
    return MemoryConfig.named(fallback_name)


def _physical_sharded_shape(
    memory: MemoryConfig,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    if memory.grid is None or memory.shard_shape is None:
        return fallback
    core_count = memory.grid.x * memory.grid.y
    shard_height, shard_width = memory.shard_shape
    if memory.layout == "height_sharded":
        return (shard_height * core_count, shard_width)
    if memory.layout == "width_sharded":
        return (shard_height, shard_width * core_count)
    return fallback


def memory_fingerprint(memory: MemoryConfig) -> str:
    descriptor = memory.to_runtime_descriptor()
    if descriptor.get("kind") == "ttnn_sharded_memory_config":
        descriptor.setdefault("orientation", "row_major")
    return sha256_json(descriptor)


def _synchronize(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)


def _deallocate(ttnn: Any, tensor: Any) -> None:
    deallocate = getattr(ttnn, "deallocate", None)
    if not callable(deallocate):
        return
    try:
        deallocate(tensor, force=True)
    except TypeError:
        deallocate(tensor)


def _transition_decision(
    edge: LayoutEdge,
    producer: MemoryConfig,
    consumer: MemoryConfig,
    costs: ConversionCostTable,
) -> dict[str, Any] | None:
    exact = memory_fingerprint(producer) == memory_fingerprint(consumer)
    if exact:
        return {
            "edge": edge.name,
            "action": "removed" if edge.incumbent_conversion else "already_elided",
            "conversion": "none",
            "latency_ms": 0.0,
            "multiplicity": edge.multiplicity,
            "single_conversion_latency_ms": 0.0,
            "producer_memory": producer.to_dict(),
            "consumer_memory": consumer.to_dict(),
            "measurement": None,
        }
    if not edge.allow_explicit_conversion:
        return None
    measured = costs.lookup(edge.name, producer, consumer)
    if measured is None:
        return None
    return {
        "edge": edge.name,
        "action": "retained",
        "conversion": "explicit",
        "latency_ms": measured.measurement.value_ms * edge.multiplicity,
        "multiplicity": edge.multiplicity,
        "single_conversion_latency_ms": measured.measurement.value_ms,
        "producer_memory": producer.to_dict(),
        "consumer_memory": consumer.to_dict(),
        "measurement": measured.measurement.to_dict(),
    }


def _sync_operator_memory(
    payload: dict[str, Any],
    path: str,
    memory: MemoryConfig,
) -> None:
    operator_key: tuple[str, str] | None = None
    if path == "attention.sdpa_kernel_output_memory_config":
        operator_key = ("attention.sdpa", "kernel_output_memory")
    elif path == "attention.sdpa_output_memory_config":
        operator_key = ("attention.sdpa", "output_memory")
    elif path == "attention.qkv_output_memory_config":
        operator_key = ("attention.qkv", "output_memory")
    elif path == "attention.o_proj_output_memory_config":
        operator_key = ("attention.o_proj", "output_memory")
    elif path == "mlp.gate_output_memory_config":
        operator_key = ("mlp.gate", "output_memory")
    elif path == "mlp.up_output_memory_config":
        operator_key = ("mlp.up", "output_memory")
    elif path == "mlp.down_output_memory_config":
        operator_key = ("mlp.down", "output_memory")
    elif path == "lm_head.shard_output_memory_config":
        operator_key = ("lm_head.shards", "output_memory")
    if operator_key is None:
        return
    operator = payload["operators"].get(operator_key[0])
    if isinstance(operator, dict):
        operator[operator_key[1]] = memory.to_dict()


def _beam_rank(state: _BeamState) -> tuple[Any, ...]:
    return (
        state.total_latency_ms,
        state.conversion_latency_ms,
        tuple(candidate.candidate_id for _, candidate in state.selections),
    )


def _memory_items(
    values: Mapping[str, MemoryConfig],
) -> tuple[tuple[str, MemoryConfig], ...]:
    return tuple(sorted((str(name), memory) for name, memory in values.items()))


def _validate_named_memories(
    values: Sequence[tuple[str, MemoryConfig]],
    label: str,
) -> None:
    names = [name for name, _ in values]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise LayoutGraphError(f"{label} must use unique non-empty names")
    if not all(isinstance(memory, MemoryConfig) for _, memory in values):
        raise LayoutGraphError(f"{label} values must be MemoryConfig objects")


def _memory_at(
    values: Sequence[tuple[str, MemoryConfig]],
    name: str,
    candidate_id: str,
) -> MemoryConfig:
    for candidate_name, memory in values:
        if candidate_name == name:
            return memory
    raise LayoutGraphError(f"candidate {candidate_id!r} has no memory port {name!r}")


def _unique_dicts(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        result.setdefault(canonical_json(value), copy.deepcopy(value))
    return [result[key] for key in sorted(result)]


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LayoutGraphError(f"{label} must be valid JSON") from exc
    if not isinstance(result, dict):
        raise LayoutGraphError(f"{label} must be a JSON object")
    return result
