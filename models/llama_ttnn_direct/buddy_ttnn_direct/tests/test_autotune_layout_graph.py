from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.layout_graph import (
    ATTENTION_DAG_NODES,
    ATTENTION_REGION,
    LM_HEAD_DAG_NODES,
    LM_HEAD_REGION,
    MLP_DAG_NODES,
    MLP_REGION,
    ConversionCostTable,
    LayoutEdge,
    LayoutGraphError,
    LayoutMeasurement,
    LayoutNodeCandidate,
    LayoutRegionGraph,
    apply_layout_result,
    build_llama_layout_graphs,
    confirm_whole_layer_no_regression,
    search_layout_graph,
    validate_layout_memory,
    write_layout_search_report,
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


class LayoutGraphSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = DeviceDescriptor.p150a()
        self.dram = MemoryConfig.named("DRAM_MEMORY_CONFIG")
        self.height = _explicit_memory("height", [8, 4], [32, 128])

    def test_measured_conversion_and_direct_output_compete_in_beam(self) -> None:
        graph = LayoutRegionGraph(
            region=ATTENTION_REGION,
            candidates=(
                _node(
                    "producer",
                    "producer.dram",
                    outputs={"out": self.dram},
                    latency_ms=1.0,
                ),
                _node(
                    "producer",
                    "producer.direct",
                    outputs={"out": self.height},
                    latency_ms=1.1,
                ),
                _node(
                    "consumer",
                    "consumer.height",
                    inputs={"in": self.height},
                    outputs={"out": self.height},
                    latency_ms=1.0,
                ),
            ),
            edges=(
                LayoutEdge(
                    "producer_to_consumer",
                    "producer",
                    "out",
                    "consumer",
                    "in",
                    (1024, 128),
                    incumbent_conversion=True,
                ),
            ),
        )
        costs = ConversionCostTable()
        costs.record_report(
            edge_name="producer_to_consumer",
            producer_memory=self.dram,
            consumer_memory=self.height,
            report=_microbench_report(0.4),
        )

        result = search_layout_graph(
            graph,
            conversion_costs=costs,
            device=self.device,
            beam_width=4,
        )

        self.assertEqual(
            [candidate.candidate_id for candidate in result.selected_candidates],
            ["producer.direct", "consumer.height"],
        )
        self.assertAlmostEqual(result.total_latency_ms, 2.1)
        self.assertEqual(
            [item["edge"] for item in result.removed_conversions],
            ["producer_to_consumer"],
        )
        self.assertEqual(result.retained_conversions, [])
        self.assertEqual(costs.to_dict()["entry_count"], 1)

    def test_unmeasured_transition_and_illegal_shard_are_pruned(self) -> None:
        invalid = _explicit_memory("height", [8, 4], [31, 128])
        graph = LayoutRegionGraph(
            region=ATTENTION_REGION,
            candidates=(
                _node(
                    "producer",
                    "producer.unmeasured",
                    outputs={"out": self.dram},
                    latency_ms=0.1,
                ),
                _node(
                    "producer",
                    "producer.illegal",
                    outputs={"out": invalid},
                    latency_ms=0.01,
                ),
                _node(
                    "producer",
                    "producer.legal",
                    outputs={"out": self.height},
                    latency_ms=1.0,
                ),
                _node(
                    "consumer",
                    "consumer",
                    inputs={"in": self.height},
                    outputs={"out": self.height},
                    latency_ms=1.0,
                ),
            ),
            edges=(
                LayoutEdge(
                    "edge",
                    "producer",
                    "out",
                    "consumer",
                    "in",
                    (1024, 128),
                    incumbent_conversion=True,
                ),
            ),
        )

        result = search_layout_graph(
            graph,
            conversion_costs=ConversionCostTable(),
            device=self.device,
            beam_width=4,
        )

        self.assertEqual(result.selected_candidates[0].candidate_id, "producer.legal")
        reasons = {item["reason"] for item in result.rejected_transitions}
        self.assertEqual(
            reasons,
            {"illegal_sharding", "conversion_cost_not_measured"},
        )
        report = result.to_dict()
        self.assertTrue(report["legality"]["selected_path_legal"])
        self.assertFalse(report["cartesian_exhaustive_search"])

    def test_beam_expansion_is_bounded_below_cartesian_product(self) -> None:
        candidates = []
        edges = []
        for node_index in range(4):
            node = f"node{node_index}"
            for candidate_index in range(5):
                candidates.append(
                    _node(
                        node,
                        f"{node}.candidate{candidate_index}",
                        inputs={"in": self.dram} if node_index else {},
                        outputs={"out": self.dram},
                        latency_ms=float(candidate_index + 1),
                    )
                )
            if node_index:
                edges.append(
                    LayoutEdge(
                        f"edge{node_index}",
                        f"node{node_index - 1}",
                        "out",
                        node,
                        "in",
                        (32, 4096),
                        incumbent_conversion=False,
                    )
                )
        result = search_layout_graph(
            LayoutRegionGraph(
                region=MLP_REGION,
                candidates=tuple(candidates),
                edges=tuple(edges),
            ),
            conversion_costs=ConversionCostTable(),
            device=self.device,
            beam_width=4,
        )
        self.assertEqual(result.cartesian_path_count, 625)
        self.assertLess(result.expanded_state_count, result.cartesian_path_count)
        self.assertLessEqual(result.peak_beam_size, 4)

    def test_invalid_beam_width_is_rejected(self) -> None:
        graph = LayoutRegionGraph(
            region=MLP_REGION,
            candidates=(_node("only", "only", outputs={"out": self.dram}),),
            edges=(),
        )
        with self.assertRaisesRegex(LayoutGraphError, "beam_width"):
            search_layout_graph(
                graph,
                conversion_costs=ConversionCostTable(),
                device=self.device,
                beam_width=3,
            )


class LlamaLayoutGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.space = _llama_space()
        self.graphs = {
            graph.region: graph for graph in build_llama_layout_graphs(self.space)
        }
        self.costs = ConversionCostTable()
        self.costs.record_report(
            edge_name="sdpa_to_concat_heads",
            producer_memory=MemoryConfig.from_dict(
                self.space.memory_configs["attention.sdpa_kernel_output_memory_config"]
            ),
            consumer_memory=MemoryConfig.from_dict(
                self.space.memory_configs["attention.concat_heads_input_memory_config"]
            ),
            report=_microbench_report(0.03),
        )

    def test_three_region_dags_match_the_execution_structure(self) -> None:
        self.assertEqual(
            set(self.graphs),
            {ATTENTION_REGION, MLP_REGION, LM_HEAD_REGION},
        )
        self.assertEqual(
            self.graphs[ATTENTION_REGION].topological_order(),
            ATTENTION_DAG_NODES,
        )
        self.assertEqual(self.graphs[MLP_REGION].topological_order(), MLP_DAG_NODES)
        self.assertEqual(
            self.graphs[LM_HEAD_REGION].topological_order(),
            LM_HEAD_DAG_NODES,
        )
        attention_ids = {
            candidate.candidate_id
            for candidate in self.graphs[ATTENTION_REGION].candidates
        }
        self.assertIn("attention.sdpa.incumbent", attention_ids)
        self.assertIn("attention.sdpa.direct_concat_layout", attention_ids)

    def test_gqa_rejects_sharded_sdpa_output_and_retains_conversion(self) -> None:
        graph = self.graphs[ATTENTION_REGION]
        result = search_layout_graph(
            graph,
            conversion_costs=self.costs,
            device=DeviceDescriptor.p150a(),
            beam_width=8,
        )
        selected = {candidate.candidate_id for candidate in result.selected_candidates}
        self.assertIn("attention.sdpa.incumbent", selected)
        self.assertNotIn("attention.sdpa.direct_concat_layout", selected)
        self.assertEqual(result.removed_conversions, [])
        self.assertEqual(
            [item["edge"] for item in result.retained_conversions],
            ["sdpa_to_concat_heads"],
        )
        rejected = {
            item.get("candidate_id"): item
            for item in result.rejected_transitions
            if item.get("reason") == "candidate_validation_failed"
        }
        self.assertEqual(
            rejected["attention.sdpa.direct_concat_layout"]["validation_evidence"][
                "code"
            ],
            "SDPA_GQA_SHARDED_OUTPUT_UNSUPPORTED",
        )

        tuned = apply_layout_result(self.space, result)
        kernel = tuned.memory_configs["attention.sdpa_kernel_output_memory_config"]
        concat = tuned.memory_configs["attention.concat_heads_input_memory_config"]
        self.assertNotEqual(kernel, concat)
        self.assertEqual(
            tuned.edges["sdpa_to_concat_heads"]["conversion"],
            "explicit",
        )

    def test_whole_layer_gate_promotes_or_falls_back_without_regression(self) -> None:
        result = search_layout_graph(
            self.graphs[ATTENTION_REGION],
            conversion_costs=self.costs,
            device=DeviceDescriptor.p150a(),
            beam_width=8,
        )
        promoted = confirm_whole_layer_no_regression(
            search_result=result,
            incumbent_report=_profile_report(10.0),
            challenger_report=_profile_report(9.5),
        )
        self.assertEqual(promoted["decision"], "promote_challenger")
        self.assertTrue(promoted["no_regression"])

        fallback = confirm_whole_layer_no_regression(
            search_result=result,
            incumbent_report=_profile_report(10.0),
            challenger_report=_profile_report(10.1),
        )
        self.assertEqual(fallback["decision"], "retain_incumbent")
        self.assertEqual(fallback["selected_latency_ms"], 10.0)
        self.assertTrue(fallback["no_regression"])

        failed = confirm_whole_layer_no_regression(
            search_result=result,
            incumbent_report=_profile_report(10.0),
            challenger_report={
                "status": "runtime_error",
                "passed": False,
                "error": "sharded output unsupported",
            },
        )
        self.assertEqual(failed["decision"], "retain_incumbent")
        self.assertEqual(failed["challenger"]["status"], "invalid")
        self.assertTrue(failed["no_regression"])

    def test_written_report_names_removed_and_retained_conversions(self) -> None:
        result = search_layout_graph(
            self.graphs[ATTENTION_REGION],
            conversion_costs=self.costs,
            device=DeviceDescriptor.p150a(),
            beam_width=8,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report = write_layout_search_report(Path(tmpdir) / "layout.json", result)
        self.assertIn("removed_conversions", report)
        self.assertIn("retained_conversions", report)
        self.assertIsInstance(report["removed_conversions"], list)
        self.assertIsInstance(report["retained_conversions"], list)
        self.assertEqual(
            [item["edge"] for item in report["retained_conversions"]],
            ["sdpa_to_concat_heads"],
        )


class LayoutLegalityTest(unittest.TestCase):
    def test_explicit_shard_must_exactly_cover_tensor(self) -> None:
        valid = _explicit_memory("width", [8, 4], [32, 128])
        invalid = _explicit_memory("width", [8, 4], [32, 127])
        self.assertEqual(
            validate_layout_memory(
                valid,
                tensor_shape=(32, 4096),
                device=DeviceDescriptor.p150a(),
            ),
            (),
        )
        issues = validate_layout_memory(
            invalid,
            tensor_shape=(32, 4096),
            device=DeviceDescriptor.p150a(),
        )
        self.assertEqual([issue.code for issue in issues], ["SHARD_SHAPE_MISMATCH"])


def _node(
    node: str,
    candidate_id: str,
    *,
    inputs: dict[str, MemoryConfig] | None = None,
    outputs: dict[str, MemoryConfig] | None = None,
    latency_ms: float = 0.0,
) -> LayoutNodeCandidate:
    return LayoutNodeCandidate.create(
        node=node,
        candidate_id=candidate_id,
        inputs=inputs or {},
        outputs=outputs or {},
        latency=LayoutMeasurement.from_report(_microbench_report(latency_ms)),
    )


def _microbench_report(latency_ms: float) -> dict:
    return {
        "status": "passed",
        "passed": True,
        "statistics": {
            "sample_count": 8,
            "mean": latency_ms,
            "p50": latency_ms,
            "p90": latency_ms,
        },
    }


def _profile_report(latency_ms: float) -> dict:
    return {
        "status": "profiled",
        "passed": True,
        "latency_ms": {
            "mean": latency_ms,
            "p50": latency_ms,
            "p90": latency_ms,
        },
    }


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
    templates = {
        **DEFAULT_TEMPLATE_SELECTION,
        "lm_head": "official_split_force_argmax",
    }
    edges = {
        "sdpa_to_concat_heads": {
            "producer_path": "attention.sdpa_kernel_output_memory_config",
            "consumer_path": "attention.concat_heads_input_memory_config",
            "producer_output_memory": dram.to_dict(),
            "consumer_input_memory": height_32.to_dict(),
            "conversion": "explicit",
        },
        "final_norm_to_lm_head": {
            "producer_path": "rms_norm.final.output_memory_config",
            "consumer_path": "lm_head.input_memory_config",
            "producer_output_memory": width_64.to_dict(),
            "consumer_input_memory": width_64.to_dict(),
            "conversion": "none",
        },
        "lm_head_shards_to_concat": {
            "producer_path": "lm_head.shard_output_memory_config",
            "consumer_path": "lm_head.concat_memory_config",
            "producer_output_memory": l1.to_dict(),
            "consumer_input_memory": l1.to_dict(),
            "conversion": "none",
        },
    }
    return SearchSpaceConfig.create(
        templates=templates,
        operators={},
        memory_configs={path: memory.to_dict() for path, memory in memories.items()},
        core_grids={},
        extra_program_configs={},
        edges=edges,
    )


if __name__ == "__main__":
    unittest.main()
