from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.execution_graph_diff import (
    compare_execution_graphs,
    load_execution_graph,
    run_execution_graph_diff,
)


class ExecutionGraphDiffTest(unittest.TestCase):
    def test_loads_current_python_io_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "buddy.json"
            graph.write_text("[]\n")
            graph.with_suffix(".python_io.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "ttnn.to_layout",
                            "arguments": {
                                "0": (
                                    "ttnn.Tensor(shape=Shape([1, 1, 32, 4096]), "
                                    "dtype=DataType::BFLOAT16, layout=Layout::TILE, "
                                    "memory_config=MemoryConfig(L1))"
                                )
                            },
                            "input_tensor_ids": [4],
                            "output_tensor_ids": [5],
                        },
                        {
                            "name": "ttnn.paged_scaled_dot_product_attention_decode",
                            "arguments": {},
                        },
                    ]
                )
            )

            capture = load_execution_graph(graph)

        self.assertEqual(capture["format"], "ttnn-python-io-sidecar")
        self.assertEqual(
            [op["canonical_name"] for op in capture["operations"]],
            ["to_layout", "sdpa"],
        )
        first = capture["operations"][0]
        self.assertEqual(first["input_tensor_ids"], [4])
        self.assertEqual(first["output_tensor_ids"], [5])
        self.assertEqual(first["metadata"]["shapes"], [[1, 1, 32, 4096]])
        self.assertEqual(first["metadata"]["dtypes"], ["BFLOAT16"])
        self.assertEqual(first["metadata"]["layouts"], ["TILE"])

    def test_loads_release_raw_graph_at_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "official.json"
            graph.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "raw_graph": [
                            {"node_type": "capture_start"},
                            {
                                "node_type": "function_start",
                                "params": {"name": "ttnn::reshape"},
                                "arguments": ["Shape([1, 1, 32, 4096])"],
                            },
                            {
                                "node_type": "function_start",
                                "params": {"name": "ttnn::prim::reshape"},
                                "arguments": [],
                            },
                            {
                                "node_type": "function_end",
                                "params": {"name": "ttnn::prim::reshape"},
                            },
                            {
                                "node_type": "function_end",
                                "params": {"name": "ttnn::reshape"},
                            },
                            {
                                "node_type": "function_start",
                                "params": {"name": "ttnn::argmax"},
                                "arguments": [],
                            },
                            {
                                "node_type": "function_end",
                                "params": {"name": "ttnn::argmax"},
                            },
                            {"node_type": "capture_end"},
                        ],
                    }
                )
            )

            capture = load_execution_graph(graph)

        self.assertEqual(capture["format"], "ttnn-raw-call-graph")
        self.assertEqual(
            [op["canonical_name"] for op in capture["operations"]],
            ["reshape", "argmax"],
        )

    def test_compare_reports_focus_count_delta(self) -> None:
        operation = lambda name: {  # noqa: E731
            "name": name,
            "canonical_name": name,
            "input_tensor_ids": [],
            "output_tensor_ids": [],
            "metadata": {},
        }
        comparison = compare_execution_graphs(
            {
                "format": "buddy",
                "operations": [operation("reshape"), operation("reshape")],
            },
            {
                "format": "official",
                "operations": [operation("reshape")],
            },
        )
        reshape = next(
            item for item in comparison["focus_diff"] if item["operation"] == "reshape"
        )
        self.assertEqual(reshape["buddy_count"], 2)
        self.assertEqual(reshape["official_count"], 1)
        self.assertEqual(reshape["count_delta"], 1)

    def test_dry_run_plans_trace_and_persistent_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "program"
            program.mkdir()
            (program / "config.json").write_text(json.dumps({"num_layers": 32}))
            official = root / "official"
            official.mkdir()
            model = root / "model"
            model.mkdir()
            output = root / "report.json"

            report = run_execution_graph_diff(
                out=output,
                buddy_program=program,
                official_tt_metal_root=official,
                model_path=model,
                dry_run=True,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "dry_run")
        buddy = next(
            plan for plan in report["planned_runs"] if plan["implementation"] == "buddy"
        )
        self.assertIn("trace", buddy["command"])
        self.assertIn("persistent", buddy["command"])


if __name__ == "__main__":
    unittest.main()
