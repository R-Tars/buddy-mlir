from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.search.generate_depth_sweep import (
    _generate_record,
    run_generate_depth_sweep,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_generate import (
    _make_generate_fake_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_single_layer_decode import (
    _fake_tokenizer_module,
)


class GenerateDepthSweepTest(unittest.TestCase):
    def test_cli_generate_depth_sweep_dry_run_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_depth_sweep.json"
            reports_dir = root / "generate_depth_reports"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
                        "--model-path",
                        str(model_dir),
                        "--config",
                        str(config_json),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )

            exit_code = main(
                [
                    "generate-depth-sweep",
                    "--program-dir",
                    str(program_dir),
                    "--depths",
                    "1,2,full",
                    "--reports-dir",
                    str(reports_dir),
                    "--max-new-tokens",
                    "3",
                    "--prefill-len",
                    "8",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "generate-depth-sweep")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["program_num_layers"], 2)
            self.assertEqual(report["depths"], [1, 2])
            self.assertEqual(report["max_new_tokens"], 3)
            self.assertEqual(report["prefill_len"], 8)
            self.assertEqual(report["status_counts"], {"dry_run": 2})
            self.assertEqual(report["prefill_status_counts"], {"dry_run": 2})
            self.assertEqual(
                report["generated_text_status_counts"],
                {"not_run": 2},
            )
            self.assertEqual(
                report["model_semantics_counts"],
                {"prompt_conditioned_prefill_decode": 2},
            )
            self.assertEqual(report["failed_depths"], [])
            self.assertEqual(report["failed_depth_diagnostics"], [])
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["acceptance"]["required_depths"], [1, 2])
            self.assertFalse(report["acceptance"]["require_full_depth"])
            for depth in (1, 2):
                path = reports_dir / f"generate_depth_{depth}.json"
                self.assertTrue(path.is_file())
                payload = json.loads(path.read_text())
                self.assertEqual(payload["mode"], "generate")
                self.assertEqual(payload["status"], "dry_run")
                self.assertEqual(
                    payload["model_semantics"],
                    "prompt_conditioned_prefill_decode",
                )
                self.assertEqual(payload["parameter_tensorization_count_per_generate"], 1)
                self.assertEqual(payload["parameter_tensorization_count_per_decode_step"], 0)
                self.assertFalse(payload["kv_cache_reinitialized_per_step"])

    def test_generate_depth_sweep_runs_fake_generate_depths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_depth_sweep.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
                        "--model-path",
                        str(model_dir),
                        "--config",
                        str(config_json),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )

            with _fake_torch_and_safetensors():
                report = run_generate_depth_sweep(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    depths=[1, "full"],
                    max_new_tokens=3,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    ttnn_module=_make_generate_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["passed"])
            self.assertEqual(report["depths"], [1, 2])
            self.assertEqual(report["status_counts"], {"passed": 2})
            self.assertEqual(report["prefill_status_counts"], {"passed": 2})
            self.assertEqual(
                report["generated_text_status_counts"],
                {"fallback": 2},
            )
            self.assertEqual(
                report["model_semantics_counts"],
                {"prompt_conditioned_prefill_decode": 2},
            )
            self.assertEqual(report["failed_depths"], [])
            self.assertEqual(report["failed_depth_diagnostics"], [])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            depth_one, depth_two = report["records"]
            self.assertEqual(
                depth_one["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertEqual(depth_one["end_to_end_failed_checks"], [])
            self.assertIsNone(depth_one["failure_diagnostics"])
            self.assertEqual(depth_one["generated_token_count_by_user"], [3, 3])
            self.assertEqual(depth_two["generated_token_count_by_user"], [3, 3])
            self.assertEqual(
                depth_two["generated_token_budget"][
                    "total_planned_generated_tokens"
                ],
                3,
            )
            self.assertEqual(
                depth_two["generated_token_budget"]["decode_loop_token_count"],
                2,
            )
            self.assertEqual(depth_two["prefill_status"], "passed")
            self.assertEqual(depth_two["kv_cache_source"], "prefill")
            self.assertTrue(depth_two["decode_loop_runtime_owned"])
            self.assertEqual(depth_two["runtime_owner"], "TTNNDirectRuntimeContext")
            self.assertEqual(depth_two["parameter_tensorization_count_per_generate"], 1)
            self.assertEqual(depth_two["parameter_tensorization_count_per_decode_step"], 0)
            self.assertFalse(depth_two["kv_cache_reinitialized_per_step"])
            self.assertEqual(
                depth_two["runtime_context"]["class"],
                "TTNNDirectRuntimeContext",
            )
            self.assertEqual(
                depth_two["generated_text"],
                "<tok:23> <tok:23> <tok:23>",
            )
            self.assertEqual(len(depth_two["prefill_cache_population"]), 2)
            self.assertTrue(Path(depth_two["generate_report"]).is_file())
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_generate_depth_sweep_writes_skipped_depth_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = root / "program"
            reports_dir = root / "depth_reports"
            report_json = root / "generate_depth_sweep.json"
            program_dir.mkdir()
            (program_dir / "config.json").write_text(
                json.dumps({"num_layers": 2}) + "\n"
            )

            def fake_generate(*, out, layers, **_kwargs):
                payload = {
                    "schema_version": 1,
                    "command": "generate",
                    "mode": "generate",
                    "status": "reference_mismatch",
                    "passed": False,
                    "layers": layers,
                    "batch_size": 2,
                    "cache_len": 16,
                    "prefill_len": 8,
                    "max_new_tokens": 3,
                    "model_semantics": "prompt_conditioned_prefill_decode",
                    "prefill_status": "passed",
                    "kv_cache_source": "prefill",
                    "decode_loop_runtime_owned": False,
                    "generate_runtime_owned": False,
                    "generated_text_status": "not_run",
                    "generated_token_ids": [],
                    "throughput_summary": {},
                    "end_to_end_contract": {
                        "status": "failed",
                        "failed_checks": ["generate.decode_loop_runtime_owned"],
                    },
                    "prefill": {
                        "status": "passed",
                        "cache_population": [
                            {
                                "layer_id": 0,
                                "status": "filled",
                                "write_policy": "paged_fill_cache_per_user",
                                "update_shape_layout": "batch_heads_seq_head_dim",
                                "key_update_shape": [1, 2, 8, 4],
                                "value_update_shape": [1, 2, 8, 4],
                            }
                        ],
                    },
                    "step_reports": [
                        {
                            "step_index": 0,
                            "status": "reference_mismatch",
                            "passed": False,
                            "output_shapes": {"token": [1, 1, 2]},
                            "reference": {
                                "status": "failed",
                                "failed_checks": ["output.token"],
                            },
                        }
                    ],
                    "error": "generate structural reference mismatch",
                }
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(payload) + "\n")
                return payload

            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.search."
                "generate_depth_sweep.run_generate",
                side_effect=fake_generate,
            ) as run_generate_mock:
                report = run_generate_depth_sweep(
                    out=report_json,
                    program_dir=program_dir,
                    depths=[1, 2],
                    reports_dir=reports_dir,
                    max_new_tokens=3,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                )

            self.assertEqual(run_generate_mock.call_count, 1)
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["passed"])
            self.assertEqual(report["status_counts"], {"reference_mismatch": 1, "skipped": 1})
            self.assertEqual(report["failed_depths"], [1])
            self.assertIn(
                "generate_depth_sweep.depth_1_passed",
                report["acceptance"]["failed_checks"],
            )
            self.assertNotIn(
                "generate_depth_sweep.report_files",
                report["acceptance"]["failed_checks"],
            )
            depth_one, depth_two = report["records"]
            self.assertTrue(depth_one["generate_report_exists"])
            self.assertTrue(depth_two["generate_report_exists"])
            self.assertEqual(depth_two["status"], "skipped")
            self.assertEqual(
                depth_two["reason"],
                "blocked by an earlier depth failure",
            )
            skipped_payload = json.loads(
                Path(depth_two["generate_report"]).read_text()
            )
            self.assertEqual(skipped_payload["mode"], "generate")
            self.assertEqual(skipped_payload["status"], "skipped")
            self.assertEqual(skipped_payload["layers"], 2)
            self.assertEqual(
                skipped_payload["reason"],
                "blocked by an earlier depth failure",
            )
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_generate_depth_sweep_writes_exception_depth_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = root / "program"
            reports_dir = root / "depth_reports"
            report_json = root / "generate_depth_sweep.json"
            program_dir.mkdir()
            (program_dir / "config.json").write_text(
                json.dumps({"num_layers": 2}) + "\n"
            )

            def fake_generate(**_kwargs):
                raise RuntimeError("synthetic depth failure")

            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.search."
                "generate_depth_sweep.run_generate",
                side_effect=fake_generate,
            ) as run_generate_mock:
                report = run_generate_depth_sweep(
                    out=report_json,
                    program_dir=program_dir,
                    depths=[1, 2],
                    reports_dir=reports_dir,
                    max_new_tokens=3,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                )

            self.assertEqual(run_generate_mock.call_count, 1)
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["passed"])
            self.assertEqual(report["status_counts"], {"fail": 1, "skipped": 1})
            self.assertEqual(report["failed_depths"], [1])
            self.assertNotIn(
                "generate_depth_sweep.report_files",
                report["acceptance"]["failed_checks"],
            )
            depth_one, depth_two = report["records"]
            self.assertTrue(depth_one["generate_report_exists"])
            self.assertTrue(depth_two["generate_report_exists"])
            self.assertEqual(depth_one["status"], "fail")
            self.assertEqual(depth_two["status"], "skipped")
            diagnostics = report["failed_depth_diagnostics"][0]
            self.assertEqual(diagnostics["depth"], 1)
            self.assertEqual(diagnostics["status"], "fail")
            self.assertEqual(diagnostics["generate_report"], depth_one["generate_report"])
            self.assertEqual(diagnostics["error"]["type"], "RuntimeError")
            exception_payload = json.loads(
                Path(depth_one["generate_report"]).read_text()
            )
            self.assertEqual(exception_payload["status"], "fail")
            self.assertEqual(exception_payload["error"]["type"], "RuntimeError")
            self.assertIn("synthetic depth failure", exception_payload["error"]["message"])
            skipped_payload = json.loads(
                Path(depth_two["generate_report"]).read_text()
            )
            self.assertEqual(skipped_payload["status"], "skipped")
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_generate_record_reports_failure_diagnostics(self) -> None:
        generate = {
            "status": "reference_mismatch",
            "passed": False,
            "layers": 2,
            "batch_size": 2,
            "cache_len": 16,
            "prefill_len": 8,
            "max_new_tokens": 3,
            "layout": "tile",
            "model_semantics": "prompt_conditioned_prefill_decode",
            "prefill_status": "passed",
            "kv_cache_source": "prefill",
            "decode_loop_runtime_owned": False,
            "generate_runtime_owned": False,
            "generated_text_status": "not_run",
            "generated_token_ids": [],
            "throughput_summary": {},
            "end_to_end_contract": {
                "status": "failed",
                "failed_checks": ["generate.decode_loop_runtime_owned"],
            },
            "prefill": {
                "cache_population": [
                    {
                        "layer_id": 1,
                        "status": "filled",
                        "write_policy": "paged_fill_cache_per_user",
                        "update_shape_layout": "batch_heads_seq_head_dim",
                        "key_update_shape": [1, 2, 8, 4],
                        "value_update_shape": [1, 2, 8, 4],
                        "key_cache_shape": [2, 2, 16, 4],
                        "value_cache_shape": [2, 2, 16, 4],
                        "page_table_shape": [2, 1],
                        "planned_user_count": 2,
                        "filled_user_count": 2,
                    }
                ],
            },
            "step_reports": [
                {
                    "step_index": 0,
                    "status": "reference_mismatch",
                    "passed": False,
                    "cache_position_value": 8,
                    "input_shapes": {
                        "token_ids": [2, 1],
                        "page_table": [2, 1],
                    },
                    "output_shapes": {"token": [2, 1]},
                    "decode_runtime_state": {"cache_position_value": 8},
                    "rotary_runtime_state": {"cos_shape": [2, 1, 64]},
                    "reference": {
                        "status": "failed",
                        "failed_checks": ["decode.output_shape"],
                        "observed_ops": ["paged_scaled_dot_product_attention_decode"],
                        "expected_ops": ["paged_scaled_dot_product_attention_decode"],
                    },
                    "error": "decode output shape mismatch",
                }
            ],
            "error": "generate reference mismatch",
        }

        record = _generate_record(
            depth=2,
            generate=generate,
            report_path=Path("/tmp/generate_depth_2.json"),
        )

        self.assertFalse(record["passed"])
        self.assertEqual(
            record["model_semantics"],
            "prompt_conditioned_prefill_decode",
        )
        self.assertEqual(
            record["end_to_end_failed_checks"],
            ["generate.decode_loop_runtime_owned"],
        )
        diagnostics = record["failure_diagnostics"]
        self.assertEqual(diagnostics["depth"], 2)
        self.assertEqual(
            diagnostics["generate_report"],
            "/tmp/generate_depth_2.json",
        )
        self.assertEqual(diagnostics["error"], "generate reference mismatch")
        self.assertEqual(
            diagnostics["prefill"]["cache_population"][0][
                "update_shape_layout"
            ],
            "batch_heads_seq_head_dim",
        )
        self.assertEqual(
            diagnostics["prefill"]["cache_population"][0]["key_update_shape"],
            [1, 2, 8, 4],
        )
        failed_step = diagnostics["decode"]["failed_step"]
        self.assertEqual(failed_step["step_index"], 0)
        self.assertEqual(failed_step["input_shapes"]["token_ids"], [2, 1])
        self.assertEqual(failed_step["output_shapes"]["token"], [2, 1])
        self.assertEqual(
            failed_step["reference_failed_checks"],
            ["decode.output_shape"],
        )
        self.assertEqual(
            failed_step["observed_ops"],
            ["paged_scaled_dot_product_attention_decode"],
        )
        self.assertEqual(failed_step["error"], "decode output shape mismatch")


if __name__ == "__main__":
    unittest.main()
