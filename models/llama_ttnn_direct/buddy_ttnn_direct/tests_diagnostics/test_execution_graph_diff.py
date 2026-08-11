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


def _raw_event(
    node_type: str,
    name: str | None = None,
    arguments: list[str] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {"node_type": node_type}
    if name is not None:
        event["params"] = {"name": name}
    if arguments is not None:
        event["arguments"] = arguments
    return event


def _operation(name: str) -> dict[str, object]:
    return {
        "name": name,
        "canonical_name": name,
        "input_tensor_ids": [],
        "output_tensor_ids": [],
        "metadata": {},
    }


class ExecutionGraphDiffTest(unittest.TestCase):
    def test_loads_current_python_io_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "buddy.json"
            graph.write_text("[]\n")
            graph.with_suffix(".python_io.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "ttnn.to_layout",
                            "arguments": {
                                "0": "ttnn.Tensor(shape=Shape([1, 1, 32, 4096]), "
                                "dtype=DataType::BFLOAT16, layout=Layout::TILE, "
                                "memory_config=MemoryConfig(L1))"
                            },
                            "input_tensor_ids": [4],
                            "output_tensor_ids": [5],
                        },
                        {"name": "ttnn.paged_scaled_dot_product_attention_decode", "arguments": {}},
                    ]
                )
            )
            capture = load_execution_graph(graph)

        self.assertEqual(capture["format"], "ttnn-python-io-sidecar")
        self.assertEqual(
            [op["canonical_name"] for op in capture["operations"]], ["to_layout", "sdpa"]
        )
        first = capture["operations"][0]
        self.assertEqual((first["input_tensor_ids"], first["output_tensor_ids"]), ([4], [5]))
        self.assertEqual(
            {key: first["metadata"][key] for key in ("shapes", "dtypes", "layouts")},
            {"shapes": [[1, 1, 32, 4096]], "dtypes": ["BFLOAT16"], "layouts": ["TILE"]},
        )

    def test_loads_release_raw_graph_at_top_level(self) -> None:
        events = [
            _raw_event("capture_start"),
            _raw_event("function_start", "ttnn::reshape", ["Shape([1, 1, 32, 4096])"]),
            _raw_event("function_start", "ttnn::prim::reshape", []),
            _raw_event("function_end", "ttnn::prim::reshape"),
            _raw_event("function_end", "ttnn::reshape"),
            _raw_event("function_start", "ttnn::argmax", []),
            _raw_event("function_end", "ttnn::argmax"),
            _raw_event("capture_end"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "official.json"
            graph.write_text(json.dumps({"schema_version": 1, "raw_graph": events}))
            capture = load_execution_graph(graph)
        self.assertEqual(capture["format"], "ttnn-raw-call-graph")
        self.assertEqual(
            [op["canonical_name"] for op in capture["operations"]], ["reshape", "argmax"]
        )

    def test_compare_reports_focus_count_delta(self) -> None:
        comparison = compare_execution_graphs(
            {"format": "buddy", "operations": [_operation("reshape")] * 2},
            {"format": "official", "operations": [_operation("reshape")]},
        )
        reshape = next(
            item for item in comparison["focus_diff"] if item["operation"] == "reshape"
        )
        self.assertEqual(
            {key: reshape[key] for key in ("buddy_count", "official_count", "count_delta")},
            {"buddy_count": 2, "official_count": 1, "count_delta": 1},
        )

    def test_dry_run_plans_trace_and_persistent_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program, official, model = root / "program", root / "official", root / "model"
            for path in (program, official, model):
                path.mkdir()
            (program / "config.json").write_text(json.dumps({"num_layers": 32}))
            report = run_execution_graph_diff(
                out=root / "report.json",
                buddy_program=program,
                official_tt_metal_root=official,
                model_path=model,
                dry_run=True,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "dry_run")
        buddy = next(item for item in report["planned_runs"] if item["implementation"] == "buddy")
        self.assertIn("trace", buddy["command"])
        self.assertIn("persistent", buddy["command"])


if __name__ == "__main__":
    unittest.main()
