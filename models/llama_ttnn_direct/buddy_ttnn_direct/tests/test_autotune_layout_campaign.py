from __future__ import annotations

import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.layout_campaign import (
    LAYOUT_MEMORY_CLASSES,
    REQUIRED_LLAMA_LAYOUT_EDGES,
    build_layout_conversion_payload,
    build_layout_conversion_probes,
    build_phase7_acceptance_report,
    build_profiled_layout_op_measurements,
    build_profiled_op_cost_report,
    canonical_layout_memories,
    collect_layout_conversion_costs,
    group_layout_conversion_probes,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.layout_graph import (
    ConversionCostTable,
    LayoutGraphError,
    build_llama_layout_graphs,
    search_layout_graph,
    validate_layout_memory,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.legality import (
    DeviceDescriptor,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.space import (
    MemoryConfig,
    SearchSpaceConfig,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.templates import (
    DEFAULT_TEMPLATE_SELECTION,
)


class ProfiledLayoutCostTest(unittest.TestCase):
    def test_profiler_regions_become_measured_per_invocation_costs(self) -> None:
        report = _profiler_report()
        measurements = build_profiled_layout_op_measurements(report)

        self.assertEqual(len(measurements), 17)
        self.assertTrue(all(item.measured for item in measurements.values()))
        self.assertEqual(measurements["attention.qkv"].value_ms, 1.0)
        self.assertEqual(measurements["mlp.down"].value_ms, 1.0)
        self.assertEqual(measurements["lm_head.concat"].value_ms, 64.0)
        self.assertTrue(build_profiled_op_cost_report(measurements)["passed"])

    def test_failed_profiler_report_is_rejected(self) -> None:
        report = _profiler_report()
        report["passed"] = False
        with self.assertRaisesRegex(LayoutGraphError, "did not pass"):
            build_profiled_layout_op_measurements(report)


class LayoutConversionCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = DeviceDescriptor.p150a()
        self.measurements = build_profiled_layout_op_measurements(
            _profiler_report()
        )
        self.graphs = build_llama_layout_graphs(
            _llama_space(),
            op_measurements=self.measurements,
        )
        self.probes = build_layout_conversion_probes(
            self.graphs,
            device=self.device,
        )

    def test_canonical_five_layouts_exactly_cover_each_real_shape(self) -> None:
        for shape in ((32, 4096), (1024, 128), (32, 128256)):
            memories = canonical_layout_memories(shape, device=self.device)
            self.assertEqual(tuple(memories), LAYOUT_MEMORY_CLASSES)
            for name, memory in memories.items():
                self.assertEqual(
                    validate_layout_memory(
                        memory,
                        tensor_shape=shape,
                        device=self.device,
                        path=name,
                    ),
                    (),
                )

    def test_required_edges_deduplicate_by_physical_shape(self) -> None:
        self.assertEqual(len(self.probes), 6 * 20)
        groups = group_layout_conversion_probes(self.probes)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(len(group) == 20 for group in groups))
        payload = build_layout_conversion_payload(groups[0])
        self.assertEqual(len(payload["probes"]), 20)

        observed_edges = {
            edge_name
            for probe in self.probes
            for edge_name in probe.edge_names
        }
        expected_edges = {
            edge_name
            for edge_names in REQUIRED_LLAMA_LAYOUT_EDGES.values()
            for edge_name in edge_names
        }
        self.assertEqual(observed_edges, expected_edges)

    def test_measured_matrix_feeds_strict_beam_search_and_acceptance(self) -> None:
        benchmark_reports = _successful_benchmark_reports(self.probes)
        costs, matrix = collect_layout_conversion_costs(
            self.probes,
            benchmark_reports,
        )

        self.assertTrue(matrix["passed"])
        self.assertEqual(matrix["successful_transition_count"], 120)
        self.assertEqual(matrix["edge_cost_entry_count"], 16 * 20)
        self.assertIsInstance(costs, ConversionCostTable)

        results = {
            graph.region: search_layout_graph(
                graph,
                conversion_costs=costs,
                device=self.device,
                beam_width=8,
                require_measured_op_costs=True,
            )
            for graph in self.graphs
        }
        acceptance = build_phase7_acceptance_report(
            graphs=self.graphs,
            search_results=results,
            op_cost_report=build_profiled_op_cost_report(self.measurements),
            conversion_matrix_report=matrix,
            whole_layer_report={"passed": True, "p50_ms": 1.0},
            full_model_report={"passed": True, "p50_ms": 29.0},
        )
        self.assertTrue(acceptance["phase_completed"])
        self.assertEqual(acceptance["number_of_measured_transitions"], 120)
        self.assertEqual(
            set(acceptance["beam_widths"]),
            set(REQUIRED_LLAMA_LAYOUT_EDGES),
        )

    def test_missing_transition_is_reported_and_fails_matrix(self) -> None:
        reports = _successful_benchmark_reports(self.probes[:-1])
        _, matrix = collect_layout_conversion_costs(self.probes, reports)
        self.assertFalse(matrix["passed"])
        self.assertEqual(matrix["missing_transition_count"], 1)

    def test_fused_qk_and_gqa_cache_edges_use_physical_shard_shapes(self) -> None:
        payload = _llama_space().to_dict()
        payload["memory_configs"].update(
            {
                "attention.qkv_heads_memory_config": _explicit_memory(
                    "height", [8, 8], [32, 128]
                ).to_dict(),
                "attention.fused_q_memory_config": _explicit_memory(
                    "height", [8, 4], [32, 128]
                ).to_dict(),
                "attention.fused_k_memory_config": _explicit_memory(
                    "height", [8, 4], [32, 128]
                ).to_dict(),
                "attention.fused_cache_key_memory_config": _explicit_memory(
                    "height", [8, 4], [32, 128]
                ).to_dict(),
                "attention.fused_cache_value_memory_config": _explicit_memory(
                    "height", [8, 4], [32, 128]
                ).to_dict(),
            }
        )
        attention = build_llama_layout_graphs(
            SearchSpaceConfig.from_dict(payload)
        )[0]
        edges = {edge.name: edge for edge in attention.edges}

        self.assertEqual(
            edges["create_heads_to_rope"].tensor_shape,
            (2048, 128),
        )
        self.assertEqual(
            edges["rope_k_to_cache_update"].tensor_shape,
            (1024, 128),
        )
        self.assertEqual(
            edges["v_to_cache_update"].tensor_shape,
            (1024, 128),
        )


class StrictMeasuredSearchTest(unittest.TestCase):
    def test_structural_operator_cost_is_rejected(self) -> None:
        graph = build_llama_layout_graphs(_llama_space())[1]
        with self.assertRaisesRegex(LayoutGraphError, "measured operator costs"):
            search_layout_graph(
                graph,
                conversion_costs=ConversionCostTable(),
                device=DeviceDescriptor.p150a(),
                beam_width=4,
                require_measured_op_costs=True,
            )


def _successful_benchmark_reports(probes):
    return [
        {
            "status": "passed",
            "passed": True,
            "repetitions": [
                {
                    "status": "passed",
                    "metadata": {
                        "transition_reports": [
                            {
                                "probe_id": probe.probe_id,
                                "status": "passed",
                                "passed": True,
                                "samples": [0.01, 0.02, 0.015],
                                "statistics": {"p50": 0.015},
                            }
                            for probe in group
                        ]
                    },
                }
            ],
        }
        for group in group_layout_conversion_probes(probes)
    ]


def _profiler_report() -> dict:
    region_names = {
        "attention_rmsnorm",
        "qkv_linear",
        "create_heads",
        "rope",
        "kv_update",
        "sdpa",
        "concat_heads",
        "o_projection",
        "mlp_rmsnorm",
        "gate_linear",
        "up_linear",
        "silu_mul",
        "down_linear",
        "final_norm",
        "lm_head_shards",
        "lm_head_concat",
        "untilize",
        "argmax",
    }
    return {
        "schema_version": 1,
        "stage": "autotune-profiler-audit",
        "status": "passed",
        "passed": True,
        "regions": [
            {
                "region": name,
                "device_kernel_latency_ms": 32.0,
                "op_count": 32,
            }
            for name in sorted(region_names)
        ],
    }


def _llama_space() -> SearchSpaceConfig:
    width_32 = _explicit_memory("width", [8, 4], [32, 128])
    width_64 = _explicit_memory("width", [8, 8], [32, 64])
    height_32 = _explicit_memory("height", [8, 4], [32, 128])
    dram = MemoryConfig.named("DRAM_MEMORY_CONFIG")
    l1 = MemoryConfig.named("L1_MEMORY_CONFIG")
    named_width = MemoryConfig.named("L1_WIDTH_SHARDED_MEMORY_CONFIG")
    memories = {
        "rms_norm.attention.output_memory_config": width_32,
        "attention.qkv_output_memory_config": named_width,
        "attention.qkv_heads_memory_config": height_32,
        "attention.sdpa_kernel_output_memory_config": dram,
        "attention.sdpa_output_memory_config": height_32,
        "attention.concat_heads_input_memory_config": height_32,
        "attention.o_proj_output_memory_config": named_width,
        "rms_norm.mlp.output_memory_config": width_64,
        "mlp.gate_output_memory_config": named_width,
        "mlp.up_output_memory_config": named_width,
        "mlp.down_output_memory_config": named_width,
        "rms_norm.final.output_memory_config": width_64,
        "lm_head.input_memory_config": width_64,
        "lm_head.shard_output_memory_config": l1,
        "lm_head.concat_memory_config": l1,
    }
    return SearchSpaceConfig.create(
        templates={
            **DEFAULT_TEMPLATE_SELECTION,
            "lm_head": "official_split_force_argmax",
        },
        operators={},
        memory_configs={
            path: memory.to_dict() for path, memory in memories.items()
        },
        core_grids={},
        extra_program_configs={},
        edges={},
    )


def _explicit_memory(
    strategy: str,
    grid: list[int],
    shard_shape: list[int],
) -> MemoryConfig:
    return MemoryConfig.from_runtime_descriptor(
        {
            "kind": "ttnn_sharded_memory_config",
            "strategy": strategy,
            "core_grid": grid,
            "shard_shape": shard_shape,
            "orientation": "row_major",
        }
    )


if __name__ == "__main__":
    unittest.main()
