from __future__ import annotations

import copy
import importlib
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .layout_graph import (
    ATTENTION_REGION,
    LM_HEAD_REGION,
    MLP_REGION,
    ConversionCostTable,
    LayoutGraphError,
    LayoutMeasurement,
    LayoutRegionGraph,
    LayoutSearchResult,
    memory_fingerprint,
    validate_layout_memory,
)
from .legality import DeviceDescriptor
from .microbench import make_worker_response
from .schema import canonical_json, sha256_json
from .space import MemoryConfig

LAYOUT_CAMPAIGN_SCHEMA_VERSION = 1

DRAM_INTERLEAVED = "dram_interleaved"
L1_INTERLEAVED = "l1_interleaved"
L1_WIDTH_SHARDED = "l1_width_sharded"
L1_HEIGHT_SHARDED = "l1_height_sharded"
L1_BLOCK_SHARDED = "l1_block_sharded"
LAYOUT_MEMORY_CLASSES = (
    DRAM_INTERLEAVED,
    L1_INTERLEAVED,
    L1_WIDTH_SHARDED,
    L1_HEIGHT_SHARDED,
    L1_BLOCK_SHARDED,
)

REQUIRED_LLAMA_LAYOUT_EDGES = {
    ATTENTION_REGION: (
        "rms_norm_attention_to_qkv",
        "qkv_to_create_heads",
        "create_heads_to_rope",
        "rope_q_to_sdpa",
        "rope_k_to_cache_update",
        "v_to_cache_update",
        "sdpa_to_concat_heads",
        "concat_heads_to_o_proj",
    ),
    MLP_REGION: (
        "rms_norm_mlp_to_gate",
        "rms_norm_mlp_to_up",
        "gate_to_mul_silu",
        "up_to_mul_silu",
        "mul_silu_to_down",
    ),
    LM_HEAD_REGION: (
        "final_norm_to_lm_head",
        "lm_head_shards_to_concat",
        "lm_head_concat_to_argmax",
    ),
}

_PROFILE_REGIONS_BY_NODE = {
    "attention.rms_norm": (("attention_rmsnorm",), "per_layer"),
    "attention.qkv": (("qkv_linear",), "per_layer"),
    "attention.create_heads": (("create_heads",), "per_layer"),
    "attention.rope": (("rope",), "per_layer"),
    "attention.cache_update": (("kv_update",), "per_layer"),
    "attention.sdpa": (("sdpa",), "per_layer"),
    "attention.concat_heads": (("concat_heads",), "per_layer"),
    "attention.o_proj": (("o_projection",), "per_layer"),
    "mlp.rms_norm": (("mlp_rmsnorm",), "per_layer"),
    "mlp.gate": (("gate_linear",), "per_layer"),
    "mlp.up": (("up_linear",), "per_layer"),
    "mlp.mul_silu": (("silu_mul",), "per_layer"),
    "mlp.down": (("down_linear",), "per_layer"),
    "lm_head.final_norm": (("final_norm",), "per_decode"),
    "lm_head.shards": (("lm_head_shards",), "per_decode"),
    "lm_head.concat": (
        ("lm_head_concat", "untilize"),
        "per_decode",
    ),
    "lm_head.argmax": (("argmax",), "per_decode"),
}


@dataclass(frozen=True)
class LayoutConversionProbe:
    probe_id: str
    tensor_shape: tuple[int, ...]
    source_class: str
    target_class: str
    source_memory: MemoryConfig
    target_memory: MemoryConfig
    edge_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.probe_id or not self.edge_names:
            raise LayoutGraphError("layout conversion probes need an id and edges")
        if self.source_class == self.target_class:
            raise LayoutGraphError("conversion probes require distinct layouts")
        if self.source_class not in LAYOUT_MEMORY_CLASSES:
            raise LayoutGraphError(f"unknown source layout: {self.source_class}")
        if self.target_class not in LAYOUT_MEMORY_CLASSES:
            raise LayoutGraphError(f"unknown target layout: {self.target_class}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "tensor_shape": list(self.tensor_shape),
            "flattened_shape": [
                math.prod(self.tensor_shape[:-1]),
                self.tensor_shape[-1],
            ],
            "source_class": self.source_class,
            "target_class": self.target_class,
            "source_memory": self.source_memory.to_dict(),
            "source_runtime_descriptor": (
                self.source_memory.to_runtime_descriptor()
            ),
            "target_memory": self.target_memory.to_dict(),
            "target_runtime_descriptor": (
                self.target_memory.to_runtime_descriptor()
            ),
            "edge_names": list(self.edge_names),
        }


def build_profiled_layout_op_measurements(
    profiler_report: Mapping[str, Any],
    *,
    num_layers: int = 32,
) -> dict[str, LayoutMeasurement]:
    if num_layers <= 0:
        raise LayoutGraphError("num_layers must be positive")
    if profiler_report.get("passed") is not True:
        raise LayoutGraphError("device profiler report did not pass")
    raw_regions = profiler_report.get("regions")
    if not isinstance(raw_regions, list):
        raise LayoutGraphError("device profiler report has no regions")
    regions: dict[str, Mapping[str, Any]] = {}
    for value in raw_regions:
        if not isinstance(value, Mapping) or not value.get("region"):
            continue
        name = str(value["region"])
        if name in regions:
            raise LayoutGraphError(f"duplicate profiler region: {name}")
        regions[name] = value

    result: dict[str, LayoutMeasurement] = {}
    for node, (region_names, scale) in _PROFILE_REGIONS_BY_NODE.items():
        missing = [name for name in region_names if name not in regions]
        if missing:
            raise LayoutGraphError(
                f"profiler cost for {node!r} is missing regions: {missing}"
            )
        aggregate_ms = sum(
            _finite_nonnegative(
                regions[name].get("device_kernel_latency_ms"),
                f"regions.{name}.device_kernel_latency_ms",
            )
            for name in region_names
        )
        value_ms = aggregate_ms / num_layers if scale == "per_layer" else aggregate_ms
        evidence = {
            "status": "profiled",
            "passed": True,
            "latency_ms": {
                "mean": value_ms,
                "p50": value_ms,
                "p90": value_ms,
            },
            "target": {
                "node": node,
                "source_regions": list(region_names),
                "normalization": scale,
                "num_layers": num_layers,
            },
            "source": {
                "stage": profiler_report.get("stage"),
                "schema_version": profiler_report.get("schema_version"),
                "report_sha256": sha256_json(dict(profiler_report)),
            },
        }
        result[node] = LayoutMeasurement.from_report(
            evidence,
            source_kind="device_profiler_region",
        )
    return result


def build_profiled_op_cost_report(
    measurements: Mapping[str, LayoutMeasurement],
) -> dict[str, Any]:
    expected = set(_PROFILE_REGIONS_BY_NODE)
    missing = sorted(expected - set(measurements))
    unmeasured = sorted(
        name
        for name, measurement in measurements.items()
        if name in expected and not measurement.measured
    )
    return {
        "schema_version": LAYOUT_CAMPAIGN_SCHEMA_VERSION,
        "status": "passed" if not missing and not unmeasured else "failed",
        "passed": not missing and not unmeasured,
        "required_node_count": len(expected),
        "measured_node_count": len(expected) - len(missing) - len(unmeasured),
        "missing_nodes": missing,
        "unmeasured_nodes": unmeasured,
        "costs": {
            name: measurements[name].to_dict()
            for name in sorted(expected & set(measurements))
        },
    }


def canonical_layout_memories(
    tensor_shape: Sequence[int],
    *,
    device: DeviceDescriptor,
) -> dict[str, MemoryConfig]:
    shape = tuple(int(value) for value in tensor_shape)
    if len(shape) < 2 or any(value <= 0 for value in shape):
        raise LayoutGraphError("layout matrix shape must be positive and rank >= 2")
    height = math.prod(shape[:-1])
    width = shape[-1]
    if height % 32 or width % 32:
        raise LayoutGraphError(
            "layout matrix tensors must have tile-aligned flattened dimensions"
        )
    height_tiles = height // 32
    width_tiles = width // 32
    worker_limit = min(
        device.worker_core_count,
        device.compute_grid.x * device.compute_grid.y,
    )

    width_cores = _largest_grid_divisor(
        width_tiles,
        worker_limit,
        device=device,
    )
    height_cores = _largest_grid_divisor(
        height_tiles,
        worker_limit,
        device=device,
    )
    width_grid = _rectangle_for_cores(width_cores, device)
    height_grid = _rectangle_for_cores(height_cores, device)
    block_x, block_y = _block_grid(height_tiles, width_tiles, device)

    result = {
        DRAM_INTERLEAVED: MemoryConfig.named("DRAM_MEMORY_CONFIG"),
        L1_INTERLEAVED: MemoryConfig.named("L1_MEMORY_CONFIG"),
        L1_WIDTH_SHARDED: _explicit_sharded_memory(
            "width",
            width_grid,
            (height, width // width_cores),
        ),
        L1_HEIGHT_SHARDED: _explicit_sharded_memory(
            "height",
            height_grid,
            (height // height_cores, width),
        ),
        L1_BLOCK_SHARDED: _explicit_sharded_memory(
            "block",
            (block_x, block_y),
            (height // block_y, width // block_x),
        ),
    }
    for name, memory in result.items():
        issues = validate_layout_memory(
            memory,
            tensor_shape=shape,
            device=device,
            path=name,
        )
        if issues:
            raise LayoutGraphError(
                f"generated layout {name!r} is illegal: "
                + canonical_json([item.to_dict() for item in issues])
            )
    return result


def build_layout_conversion_probes(
    graphs: Iterable[LayoutRegionGraph],
    *,
    device: DeviceDescriptor,
) -> tuple[LayoutConversionProbe, ...]:
    graph_by_region = {graph.region: graph for graph in graphs}
    required_edges = _required_edges(graph_by_region)
    shape_edges: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for edge in required_edges:
        shape_edges[edge.tensor_shape].append(edge.name)

    probes = []
    for shape, edge_names in sorted(shape_edges.items()):
        memories = canonical_layout_memories(shape, device=device)
        for source_class in LAYOUT_MEMORY_CLASSES:
            for target_class in LAYOUT_MEMORY_CLASSES:
                if source_class == target_class:
                    continue
                source = memories[source_class]
                target = memories[target_class]
                identity = {
                    "shape": list(shape),
                    "source": source.to_runtime_descriptor(),
                    "target": target.to_runtime_descriptor(),
                }
                probes.append(
                    LayoutConversionProbe(
                        probe_id="layout-conversion-" + sha256_json(identity)[:16],
                        tensor_shape=shape,
                        source_class=source_class,
                        target_class=target_class,
                        source_memory=source,
                        target_memory=target,
                        edge_names=tuple(sorted(edge_names)),
                    )
                )
    return tuple(probes)


def group_layout_conversion_probes(
    probes: Iterable[LayoutConversionProbe],
) -> tuple[tuple[LayoutConversionProbe, ...], ...]:
    grouped: dict[tuple[int, ...], list[LayoutConversionProbe]] = defaultdict(list)
    for probe in probes:
        grouped[probe.tensor_shape].append(probe)
    return tuple(
        tuple(sorted(grouped[shape], key=lambda item: item.probe_id))
        for shape in sorted(grouped)
    )


def build_layout_conversion_payload(
    probes: Sequence[LayoutConversionProbe],
    *,
    device_id: int = 0,
    dtype: str = "bfloat16",
    layout: str = "TILE_LAYOUT",
    seed: int = 0,
) -> dict[str, Any]:
    if not probes:
        raise LayoutGraphError("layout conversion payload needs at least one probe")
    shapes = {probe.tensor_shape for probe in probes}
    if len(shapes) != 1:
        raise LayoutGraphError("one layout conversion payload must use one shape")
    return {
        "device_id": int(device_id),
        "dtype": str(dtype),
        "layout": str(layout),
        "seed": int(seed),
        "shape": list(next(iter(shapes))),
        "probes": [probe.to_dict() for probe in probes],
    }


def layout_conversion_matrix_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    from ..runtime.config_runtime import realize_ttnn_config
    from ..smoke_mlp import _managed_ttnn_device

    payload = _mapping(request.get("payload"), "payload")
    candidate = _mapping(request.get("candidate"), "candidate")
    measurement = _mapping(
        candidate.get("measurement_contract"),
        "measurement_contract",
    )
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise LayoutGraphError("conversion matrix worker has no probes")
    warmup = int(measurement["warmup"])
    iterations = int(measurement["iterations"])
    if warmup < 0 or iterations <= 0:
        raise LayoutGraphError("invalid conversion matrix measurement contract")

    ttnn = importlib.import_module("ttnn")
    torch = importlib.import_module("torch")
    device_id = int(payload.get("device_id", 0))
    dtype = getattr(ttnn, str(payload.get("dtype", "bfloat16")))
    layout = getattr(ttnn, str(payload.get("layout", "TILE_LAYOUT")))
    torch.manual_seed(int(payload.get("seed", 0)))

    transition_reports: list[dict[str, Any]] = []
    successful_samples: list[list[float]] = []
    with _managed_ttnn_device(ttnn, device_id) as device:
        for raw_probe in raw_probes:
            if not isinstance(raw_probe, Mapping):
                raise LayoutGraphError("conversion matrix probe must be an object")
            probe = dict(raw_probe)
            try:
                report = _measure_layout_transition(
                    probe,
                    ttnn=ttnn,
                    torch=torch,
                    device=device,
                    dtype=dtype,
                    layout=layout,
                    warmup=warmup,
                    iterations=iterations,
                    realize_ttnn_config=realize_ttnn_config,
                )
            except Exception as exc:  # Hardware legality is part of the matrix.
                report = {
                    "probe_id": str(probe.get("probe_id", "unknown")),
                    "status": "runtime_rejected",
                    "passed": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "samples": [],
                    "statistics": None,
                }
            transition_reports.append(report)
            statistics_report = report.get("statistics")
            if report.get("passed") and isinstance(statistics_report, Mapping):
                successful_samples.append(
                    [float(value) for value in report.get("samples") or []]
                )

        count = getattr(device, "num_program_cache_entries", None)
        program_cache_count = int(count()) if callable(count) else None

    aggregate_samples = [
        sum(samples[index] for samples in successful_samples)
        for index in range(iterations)
    ]
    return make_worker_response(
        request,
        aggregate_samples,
        program_cache_count=program_cache_count,
        trace_capture_count=0,
        new_tensor_allocations=iterations * len(raw_probes),
        metadata={
            "worker_kind": "ttnn_layout_conversion_matrix",
            "device_id": device_id,
            "transition_count": len(transition_reports),
            "successful_transition_count": sum(
                item.get("passed") is True for item in transition_reports
            ),
            "synchronized_each_iteration": True,
            "transition_reports": transition_reports,
        },
    )


def collect_layout_conversion_costs(
    probes: Iterable[LayoutConversionProbe],
    benchmark_reports: Iterable[Mapping[str, Any]],
) -> tuple[ConversionCostTable, dict[str, Any]]:
    probe_by_id = {probe.probe_id: probe for probe in probes}
    observed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    batch_count = 0
    for benchmark in benchmark_reports:
        batch_count += 1
        repetitions = benchmark.get("repetitions")
        if not isinstance(repetitions, list):
            continue
        for repetition in repetitions:
            if not isinstance(repetition, Mapping):
                continue
            metadata = repetition.get("metadata")
            transitions = (
                metadata.get("transition_reports")
                if isinstance(metadata, Mapping)
                else None
            )
            if not isinstance(transitions, list):
                continue
            for transition in transitions:
                if not isinstance(transition, Mapping):
                    continue
                probe_id = str(transition.get("probe_id", ""))
                if probe_id in probe_by_id:
                    observed[probe_id].append(transition)

    table = ConversionCostTable()
    successful = []
    failures = []
    missing = []
    for probe_id, probe in sorted(probe_by_id.items()):
        reports = observed.get(probe_id, [])
        samples = [
            float(sample)
            for report in reports
            if report.get("passed") is True
            for sample in (report.get("samples") or [])
        ]
        if not reports:
            missing.append(probe.to_dict())
            continue
        if not samples:
            failures.append(
                {
                    **probe.to_dict(),
                    "attempts": [copy.deepcopy(dict(item)) for item in reports],
                }
            )
            continue
        report = {
            "status": "passed",
            "passed": True,
            "statistics": _sample_statistics(samples),
            "target": {
                "probe_id": probe.probe_id,
                "shape": list(probe.tensor_shape),
                "source_class": probe.source_class,
                "target_class": probe.target_class,
                "edge_names": list(probe.edge_names),
            },
            "hardware_evidence": [
                copy.deepcopy(dict(item))
                for item in reports
                if item.get("passed") is True
            ],
        }
        for edge_name in probe.edge_names:
            table.record_report(
                edge_name=edge_name,
                producer_memory=probe.source_memory,
                consumer_memory=probe.target_memory,
                report=report,
            )
        successful.append(
            {
                **probe.to_dict(),
                "statistics": report["statistics"],
                "edge_cost_entry_count": len(probe.edge_names),
            }
        )

    attempted_pairs = {
        (item.source_class, item.target_class)
        for item in probe_by_id.values()
        if item.probe_id in observed
    }
    expected_pairs = {
        (source, target)
        for source in LAYOUT_MEMORY_CLASSES
        for target in LAYOUT_MEMORY_CLASSES
        if source != target
    }
    matrix_report = {
        "schema_version": LAYOUT_CAMPAIGN_SCHEMA_VERSION,
        "status": (
            "passed"
            if not missing and attempted_pairs == expected_pairs
            else "failed"
        ),
        "passed": not missing and attempted_pairs == expected_pairs,
        "layout_classes": list(LAYOUT_MEMORY_CLASSES),
        "directed_layout_pair_count": len(expected_pairs),
        "measured_layout_pair_count": len(attempted_pairs),
        "shape_count": len({probe.tensor_shape for probe in probe_by_id.values()}),
        "batch_measurement_count": batch_count,
        "requested_transition_count": len(probe_by_id),
        "attempted_transition_count": len(observed),
        "successful_transition_count": len(successful),
        "runtime_rejected_transition_count": len(failures),
        "missing_transition_count": len(missing),
        "edge_cost_entry_count": table.to_dict()["entry_count"],
        "successful_transitions": successful,
        "runtime_rejected_transitions": failures,
        "missing_transitions": missing,
        "conversion_cost_table": table.to_dict(),
    }
    return table, matrix_report


def build_phase7_acceptance_report(
    *,
    graphs: Iterable[LayoutRegionGraph],
    search_results: Mapping[str, LayoutSearchResult],
    op_cost_report: Mapping[str, Any],
    conversion_matrix_report: Mapping[str, Any],
    whole_layer_report: Mapping[str, Any],
    full_model_report: Mapping[str, Any],
) -> dict[str, Any]:
    graph_by_region = {graph.region: graph for graph in graphs}
    _required_edges(graph_by_region)
    missing_searches = sorted(set(REQUIRED_LLAMA_LAYOUT_EDGES) - set(search_results))
    removed = [
        copy.deepcopy(item)
        for region in sorted(search_results)
        for item in search_results[region].removed_conversions
    ]
    retained = [
        copy.deepcopy(item)
        for region in sorted(search_results)
        for item in search_results[region].retained_conversions
    ]
    searches_measured = all(
        candidate.latency.measured
        for result in search_results.values()
        for candidate in result.selected_candidates
    )
    passed = all(
        (
            not missing_searches,
            searches_measured,
            op_cost_report.get("passed") is True,
            conversion_matrix_report.get("passed") is True,
            whole_layer_report.get("passed") is True,
            full_model_report.get("passed") is True,
        )
    )
    return {
        "schema_version": LAYOUT_CAMPAIGN_SCHEMA_VERSION,
        "phase": 7,
        "name": "full_cross_op_layout_graph",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "phase_completed": passed,
        "required_edges": {
            region: list(names)
            for region, names in REQUIRED_LLAMA_LAYOUT_EDGES.items()
        },
        "missing_region_searches": missing_searches,
        "beam_widths": {
            region: result.beam_width
            for region, result in sorted(search_results.items())
        },
        "measured_op_costs": copy.deepcopy(dict(op_cost_report)),
        "number_of_measured_transitions": conversion_matrix_report.get(
            "successful_transition_count"
        ),
        "conversion_matrix": copy.deepcopy(dict(conversion_matrix_report)),
        "removed_conversions": removed,
        "retained_conversions": retained,
        "whole_layer_latency": copy.deepcopy(dict(whole_layer_report)),
        "full_model_latency": copy.deepcopy(dict(full_model_report)),
        "searches": {
            region: result.to_dict()
            for region, result in sorted(search_results.items())
        },
    }


def _required_edges(
    graph_by_region: Mapping[str, LayoutRegionGraph],
) -> list[Any]:
    missing_regions = sorted(set(REQUIRED_LLAMA_LAYOUT_EDGES) - set(graph_by_region))
    if missing_regions:
        raise LayoutGraphError(f"missing layout graph regions: {missing_regions}")
    selected = []
    for region, required_names in REQUIRED_LLAMA_LAYOUT_EDGES.items():
        edges = {edge.name: edge for edge in graph_by_region[region].edges}
        missing = sorted(set(required_names) - set(edges))
        if missing:
            raise LayoutGraphError(
                f"layout graph {region!r} is missing required edges: {missing}"
            )
        selected.extend(edges[name] for name in required_names)
    return selected


def _largest_grid_divisor(
    tile_count: int,
    worker_limit: int,
    *,
    device: DeviceDescriptor,
) -> int:
    for count in range(min(tile_count, worker_limit), 0, -1):
        if tile_count % count == 0 and _has_rectangle(count, device):
            return count
    return 1


def _has_rectangle(count: int, device: DeviceDescriptor) -> bool:
    return any(
        count % x == 0 and count // x <= device.compute_grid.y
        for x in range(1, min(count, device.compute_grid.x) + 1)
    )


def _rectangle_for_cores(
    count: int,
    device: DeviceDescriptor,
) -> tuple[int, int]:
    for x in range(min(count, device.compute_grid.x), 0, -1):
        if count % x == 0 and count // x <= device.compute_grid.y:
            return (x, count // x)
    raise LayoutGraphError(f"cannot place {count} cores on device grid")


def _block_grid(
    height_tiles: int,
    width_tiles: int,
    device: DeviceDescriptor,
) -> tuple[int, int]:
    candidates = [
        (x, y)
        for x in range(1, min(width_tiles, device.compute_grid.x) + 1)
        for y in range(1, min(height_tiles, device.compute_grid.y) + 1)
        if width_tiles % x == 0
        and height_tiles % y == 0
        and x * y <= device.worker_core_count
    ]
    if not candidates:
        raise LayoutGraphError("no exact block-sharded grid covers tensor")
    return max(candidates, key=lambda value: (value[0] * value[1], value[1], value[0]))


def _explicit_sharded_memory(
    strategy: str,
    grid: Sequence[int],
    shard_shape: Sequence[int],
) -> MemoryConfig:
    return MemoryConfig.from_runtime_descriptor(
        {
            "kind": "ttnn_sharded_memory_config",
            "strategy": strategy,
            "core_grid": [int(value) for value in grid],
            "shard_shape": [int(value) for value in shard_shape],
            "orientation": "row_major",
        }
    )


def _measure_layout_transition(
    probe: Mapping[str, Any],
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype: Any,
    layout: Any,
    warmup: int,
    iterations: int,
    realize_ttnn_config: Any,
) -> dict[str, Any]:
    probe_id = str(probe.get("probe_id", ""))
    shape_value = probe.get("tensor_shape")
    source_descriptor = probe.get("source_runtime_descriptor")
    target_descriptor = probe.get("target_runtime_descriptor")
    if not probe_id:
        raise LayoutGraphError("conversion probe has no probe_id")
    if not isinstance(shape_value, list) or len(shape_value) < 2:
        raise LayoutGraphError("conversion probe has no valid tensor_shape")
    if not isinstance(source_descriptor, Mapping) or not isinstance(
        target_descriptor,
        Mapping,
    ):
        raise LayoutGraphError("conversion probe memory descriptor is missing")
    shape = tuple(int(value) for value in shape_value)
    source_memory = realize_ttnn_config(dict(source_descriptor), ttnn)
    target_memory = realize_ttnn_config(dict(target_descriptor), ttnn)
    host = torch.randn(shape, dtype=torch.bfloat16)
    source = None
    converted = None
    samples: list[float] = []
    try:
        source = ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            memory_config=source_memory,
            device=device,
        )
        _synchronize(ttnn, device)
        converted = ttnn.to_memory_config(source, memory_config=target_memory)
        _synchronize(ttnn, device)
        observed = ttnn.to_torch(converted)
        correctness = {
            "passed": bool(torch.equal(observed, host)),
            "source_shape": list(host.shape),
            "observed_shape": list(observed.shape),
        }
        if not correctness["passed"]:
            raise LayoutGraphError("layout conversion correctness gate failed")
        _deallocate(ttnn, converted)
        converted = None
        for _ in range(warmup):
            converted = ttnn.to_memory_config(source, memory_config=target_memory)
            _synchronize(ttnn, device)
            _deallocate(ttnn, converted)
            converted = None
        for _ in range(iterations):
            start = time.perf_counter_ns()
            converted = ttnn.to_memory_config(source, memory_config=target_memory)
            _synchronize(ttnn, device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
            _deallocate(ttnn, converted)
            converted = None
    finally:
        if converted is not None:
            _deallocate(ttnn, converted)
        if source is not None:
            _deallocate(ttnn, source)
    return {
        "probe_id": probe_id,
        "status": "passed",
        "passed": True,
        "tensor_shape": list(shape),
        "source_class": probe.get("source_class"),
        "target_class": probe.get("target_class"),
        "source_memory_sha256": memory_fingerprint(
            MemoryConfig.from_runtime_descriptor(source_descriptor)
        ),
        "target_memory_sha256": memory_fingerprint(
            MemoryConfig.from_runtime_descriptor(target_descriptor)
        ),
        "samples": samples,
        "statistics": _sample_statistics(samples),
        "correctness": correctness,
    }


def _sample_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise LayoutGraphError(
            "layout conversion samples must be finite and non-negative"
        )
    ordered = sorted(values)
    return {
        "sample_count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": _percentile(ordered, 0.9),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _finite_nonnegative(value: Any, path: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise LayoutGraphError(f"{path} must be finite and non-negative")
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayoutGraphError(f"{label} must be an object")
    return value


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
