from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.generate import (
    run_profile_generate,
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


class ProfileGenerateTest(unittest.TestCase):
    def test_cli_profile_generate_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_profile_report.json"
            generate_report_json = root / "underlying_generate_report.json"
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
                    "profile-generate",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "3",
                    "--prefill-len",
                    "8",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--generate-report",
                    str(generate_report_json),
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "profile-generate")
            self.assertEqual(report["mode"], "profile-generate")
            self.assertEqual(
                report["template"],
                "prefill_then_decode_generate_profile",
            )
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["generate_report"], str(generate_report_json))
            self.assertTrue(generate_report_json.is_file())
            self.assertIsNone(report["prefill_ms"])
            self.assertIsNone(report["decode_step_ms_mean"])
            self.assertIsNone(report["tokens_per_second_per_user"])
            self.assertFalse(report["official_performance_parity_claimed"])
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["acceptance"]["failed_checks"], [])

    def test_profile_generate_runs_fake_generate_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_profile_report.json"
            generate_report_json = root / "generate_report.json"
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
                report = run_profile_generate(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    max_new_tokens=3,
                    layers=1,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    generate_report=generate_report_json,
                    ttnn_module=_make_generate_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "profiled")
            self.assertTrue(report["passed"])
            self.assertEqual(report["generate_status"], "passed")
            self.assertEqual(report["prefill_status"], "passed")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(report["parameter_source"], "hf_model")
            self.assertEqual(report["input_source"], "prompt_prefill")
            self.assertTrue(report["generate_runtime_owned"])
            self.assertTrue(report["decode_loop_runtime_owned"])
            self.assertIsNotNone(report["prefill_ms"])
            self.assertIsNotNone(report["decode_step_ms_mean"])
            self.assertGreater(report["tokens_per_second_per_user"], 0.0)
            self.assertGreater(report["aggregate_tokens_per_second"], 0.0)
            self.assertEqual(report["synthetic_runtime_input_tensor_count"], 0)
            self.assertEqual(report["synthetic_rotary_tensor_count"], 0)
            self.assertEqual(report["synthetic_kv_cache_tensor_count"], 0)
            self.assertEqual(
                report["parameter_setup"][
                    "parameter_tensorization_count_per_generate"
                ],
                1,
            )
            self.assertEqual(report["generated_token_count_by_user"], [3, 3])
            self.assertFalse(report["official_performance_parity_claimed"])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            self.assertEqual(
                report["sections"]["host_copy_ms"]["host_roundtrip_present"],
                True,
            )
            self.assertEqual(report["per_layer"]["status"], "unavailable")
            self.assertTrue(generate_report_json.is_file())
            self.assertEqual(json.loads(report_json.read_text()), report)


if __name__ == "__main__":
    unittest.main()
