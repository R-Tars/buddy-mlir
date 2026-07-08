from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.search.generate_depth_sweep import (
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
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_single_layer_decode import (
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
            self.assertEqual(report["failed_depths"], [])
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["acceptance"]["required_depths"], [1, 2])
            self.assertFalse(report["acceptance"]["require_full_depth"])
            for depth in (1, 2):
                path = reports_dir / f"generate_depth_{depth}.json"
                self.assertTrue(path.is_file())
                payload = json.loads(path.read_text())
                self.assertEqual(payload["mode"], "generate")
                self.assertEqual(payload["status"], "dry_run")
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
            self.assertEqual(report["failed_depths"], [])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            depth_one, depth_two = report["records"]
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
            self.assertEqual(depth_two["parameter_tensorization_count_per_generate"], 1)
            self.assertEqual(depth_two["parameter_tensorization_count_per_decode_step"], 0)
            self.assertFalse(depth_two["kv_cache_reinitialized_per_step"])
            self.assertEqual(
                depth_two["runtime_context"]["class"],
                "TTNNDirectRuntimeContext",
            )
            self.assertEqual(
                depth_two["generated_text"],
                "<tok:17> <tok:23> <tok:23>",
            )
            self.assertEqual(len(depth_two["prefill_cache_population"]), 2)
            self.assertTrue(Path(depth_two["generate_report"]).is_file())
            self.assertEqual(json.loads(report_json.read_text()), report)


if __name__ == "__main__":
    unittest.main()
