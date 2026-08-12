from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.execution_graph_diff import (
    compare_execution_graphs,
    load_execution_graph,
    run_execution_graph_diff,
    _planned_runs,
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
    return {"name": name, "canonical_name": name, "input_tensor_ids": [],
            "output_tensor_ids": [], "metadata": {}}

class ExecutionGraphDiffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.program, self.official, self.model = (
            self.root / name for name in ("program", "official", "model")
        )
        for path in (self.program, self.official, self.model):
            path.mkdir()
        (self.program / "config.json").write_text(json.dumps({"num_layers": 32}))

    def plans(self) -> list[dict[str, object]]:
        return _planned_runs(
            runs_root=self.root / "runs", program_root=self.program,
            official_root=self.official, model_root=self.model,
            tokenizer_root=self.root / "tokenizer", prompts_path=self.root / "prompts.json",
            official_python=self.root / "python", layer_count=32, batch_size=32,
            prefill_len=256, cache_len=1024, page_block_size=32,
            device="p150a", device_id=0,
        )

    def test_loads_current_python_io_sidecar(self) -> None:
        graph = self.root / "buddy.json"
        graph.write_text("[]\n")
        graph.with_suffix(".python_io.json").write_text(json.dumps([
            {
                "name": "ttnn.to_layout",
                "arguments": {"0": "ttnn.Tensor(shape=Shape([1, 1, 32, 4096]), "
                              "dtype=DataType::BFLOAT16, layout=Layout::TILE, "
                              "memory_config=MemoryConfig(L1))"},
                "input_tensor_ids": [4], "output_tensor_ids": [5],
            },
            {"name": "ttnn.paged_scaled_dot_product_attention_decode", "arguments": {}},
        ]))
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
        graph = self.root / "official.json"
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
        report = run_execution_graph_diff(
            out=self.root / "report.json", buddy_program=self.program,
            official_tt_metal_root=self.official, model_path=self.model, dry_run=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "dry_run")
        buddy = next(item for item in report["planned_runs"] if item["implementation"] == "buddy")
        self.assertIn("trace", buddy["command"])
        self.assertIn("persistent", buddy["command"])

    def test_official_plan_exposes_source_ttnn_package(self) -> None:
        plans = self.plans()
        official = next(item for item in plans if item["implementation"] == "official")
        python_path = str(official["environment"]["PYTHONPATH"]).split(os.pathsep)
        self.assertIn(str((self.official / "ttnn").resolve()), python_path)
        trace_index = official["command"].index("--enable_trace")
        self.assertEqual(trace_index, len(official["command"]) - 1)

        buddy = next(item for item in plans if item["implementation"] == "buddy")
        buddy_environment = buddy["environment"]
        buddy_python_path = buddy_environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(buddy_environment["TT_METAL_HOME"], str(self.official))
        self.assertEqual(buddy_environment["TT_METAL_BUILD_HOME"], str(self.official / "build"))
        self.assertIn(str(self.official / "ttnn"), buddy_python_path)
        self.assertIn(str(self.official / "tt_eager"), buddy_python_path)
        self.assertIn(str(Path(__file__).resolve().parents[4] / "build-ttmlir" / "python_packages"),
                      buddy_python_path)

    def test_real_run_stages_prompts_outside_pytest_search_tree(self) -> None:
        source_prompts = self.root / "external-checkout/models/prompts.json"
        source_prompts.parent.mkdir(parents=True)
        source_prompts.write_text("[]\n")
        observed: list[object] = []

        def runner(command, cwd, environment, log_path, timeout, address_limit):
            observed.append((command, cwd, environment, log_path, timeout, address_limit))
            return 1

        run_execution_graph_diff(
            out=self.root / "report.json", buddy_program=self.program,
            official_tt_metal_root=self.official, model_path=self.model,
            input_prompts=source_prompts, command_runner=runner,
        )
        command = observed[0][0]
        self.assertIn(str(self.root / "report_runs/inputs/input_prompts.json"), command)
        self.assertNotIn(str(source_prompts), command)

if __name__ == "__main__":
    unittest.main()
