from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.generate import (
    run_profile_decode_steady,
    run_profile_generate,
    run_profile_prefill_steady,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_generate_dryrun import (
    _make_generate_fake_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
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


class ProfileGenerateTest(unittest.TestCase):
    def test_cli_profile_prefill_steady_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prefill_steady_profile.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build",
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
                    "profile",
                    "--mode",
                    "prefill-steady",
                    "--program-dir",
                    str(program_dir),
                    "--prefill-len",
                    "8",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--warmup",
                    "2",
                    "--iterations",
                    "4",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "profile")
            self.assertEqual(report["mode"], "prefill-steady")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["warmup"], 2)
            self.assertEqual(report["iterations"], 4)
            self.assertEqual(report["prefill_execution_mode"], "eager")
            references = report["official_references"]
            self.assertEqual(
                references["release_tag"],
                "v0.64.0-dev20251030",
            )
            self.assertEqual(
                references["release_commit"],
                "b76035fbdac81d8f9974976471dc60fc005e1bfb",
            )
            self.assertEqual(
                references["matched_tt_metal_commit"],
                "61e690c25202111b52cbc1fbc9148b6524070c6f",
            )

    def test_prefill_trace_candidate_fails_before_device_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prefill_trace_profile.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build",
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

            report = run_profile_prefill_steady(
                out=report_json,
                program_dir=program_dir,
                prefill_execution_mode="trace",
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "unsupported_prefill_trace")
            self.assertIn("residual add", report["error"])
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_cli_profile_decode_steady_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_steady_profile.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build",
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
                    "profile",
                    "--mode",
                    "decode-steady",
                    "--program-dir",
                    str(program_dir),
                    "--prefill-len",
                    "8",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--warmup",
                    "2",
                    "--iterations",
                    "4",
                    "--after-prefill",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "profile")
            self.assertEqual(report["mode"], "decode-steady")
            self.assertEqual(
                report["template"],
                "post_prefill_decode_steady_profile",
            )
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertTrue(report["after_prefill"])
            self.assertEqual(report["layers"], 2)
            self.assertEqual(report["warmup"], 2)
            self.assertEqual(report["iterations"], 4)
            self.assertTrue(report["compile_excluded_by_warmup"])
            self.assertIsNone(report["prefill_ms"])
            self.assertIsNone(report["decode_step_ms_p50"])
            self.assertIsNone(report["decode_step_ms_mean"])
            self.assertIsNone(report["tokens_per_second_per_user"])
            self.assertEqual(report["host_copy_profile"]["status"], "not_run")
            self.assertEqual(report["section_profile"]["status"], "not_run")
            self.assertFalse(
                report["timing_scope"]["per_op_section_profiler_installed"]
            )
            self.assertTrue(
                report["timing_scope"]["runtime_metadata_in_timed_region"]
            )
            self.assertFalse(report["official_performance_parity_claimed"])

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
                        "build",
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
                    "profile",
                    "--program-dir",
                    str(program_dir),
                    "--mode",
                    "generate",
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
            self.assertIsNone(report["host_copy_ms"])
            self.assertEqual(report["host_copy_profile"]["status"], "not_run")
            self.assertEqual(report["section_profile"]["status"], "not_run")
            self.assertIsNone(report["tokens_per_second_per_user"])
            self.assertFalse(report["official_performance_parity_claimed"])
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            milestones = report["performance_milestones"]
            self.assertTrue(milestones["dry_run"])
            self.assertEqual(
                milestones["official_reference"][
                    "tokens_per_second_per_user"
                ],
                33.1,
            )
            self.assertEqual(
                [entry["id"] for entry in milestones["milestones"]],
                ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
            )
            self.assertFalse(
                any(entry["passed"] for entry in milestones["milestones"])
            )

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
                        "build",
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
            self.assertEqual(report["program_num_layers"], 2)
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(
                report["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertEqual(report["parameter_source"], "hf_model")
            self.assertEqual(report["input_source"], "prompt_prefill")
            self.assertEqual(report["runtime_owner"], "TTNNDirectRuntimeContext")
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
                report["end_to_end_contract"]["status"],
                "passed",
            )
            self.assertEqual(
                report["end_to_end_contract"]["failed_checks"],
                [],
            )
            self.assertEqual(
                report["sections"]["host_copy_ms"]["host_roundtrip_present"],
                False,
            )
            self.assertEqual(
                report["sections"]["host_copy_ms"][
                    "runtime_host_roundtrip_present"
                ],
                False,
            )
            self.assertEqual(
                report["sections"]["host_copy_ms"]["status"],
                "measured",
            )
            self.assertGreaterEqual(report["sections"]["host_copy_ms"]["value_ms"], 0.0)
            self.assertEqual(report["host_copy_profile"]["status"], "measured")
            self.assertEqual(report["section_profile"]["status"], "measured")
            self.assertGreaterEqual(report["host_copy_ms"], 0.0)
            self.assertGreaterEqual(
                report["prefill_first_token_materialization_ms"],
                0.0,
            )
            self.assertEqual(len(report["decode_token_materialization_ms_samples"]), 2)
            for name in (
                "embedding_ms",
                "prefill_attention_ms",
                "decode_attention_ms",
                "mlp_ms",
                "lm_head_ms",
                "argmax_ms",
                "host_copy_ms",
            ):
                self.assertEqual(report["sections"][name]["status"], "measured")
                self.assertGreaterEqual(
                    report["sections"][name]["value_ms"],
                    0.0,
                )
            self.assertEqual(report["per_layer"]["status"], "measured")
            self.assertEqual(len(report["per_layer"]["prefill"]), 1)
            self.assertEqual(len(report["per_layer"]["decode"]), 1)
            milestones = report["performance_milestones"]
            by_id = {entry["id"]: entry for entry in milestones["milestones"]}
            self.assertFalse(milestones["dry_run"])
            self.assertEqual(
                [entry["id"] for entry in milestones["milestones"]],
                ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
            )
            self.assertTrue(by_id["M0"]["passed"])
            self.assertFalse(by_id["M1"]["passed"])
            self.assertEqual(by_id["M1"]["reason"], "not_full_depth_profile")
            self.assertFalse(by_id["M2"]["passed"])
            self.assertEqual(by_id["M2"]["reason"], "requires_batch32_profile")
            self.assertEqual(milestones["highest_passed"], "M0")
            self.assertTrue(generate_report_json.is_file())
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_profile_decode_steady_runs_repeated_post_prefill_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_steady_profile.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build",
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

            fake_ttnn = _make_generate_fake_ttnn()
            with _fake_torch_and_safetensors():
                report = run_profile_decode_steady(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    layers=1,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    warmup=2,
                    iterations=4,
                    ttnn_module=fake_ttnn,
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "profiled")
            self.assertTrue(report["passed"])
            self.assertEqual(report["prefill_status"], "passed")
            self.assertGreaterEqual(report["prefill_ms"], 0.0)
            self.assertEqual(len(report["warmup_step_ms_samples"]), 2)
            self.assertEqual(len(report["decode_step_ms_samples"]), 4)
            self.assertGreater(report["decode_step_ms_p50"], 0.0)
            self.assertGreater(report["decode_step_ms_mean"], 0.0)
            self.assertGreater(report["tokens_per_second_per_user"], 0.0)
            self.assertGreater(report["aggregate_tokens_per_second"], 0.0)
            self.assertEqual(
                report["throughput_summary"]["measured_tokens_per_user"],
                4,
            )
            self.assertEqual(
                report["throughput_summary"]["measured_aggregate_tokens"],
                8,
            )
            self.assertEqual(report["cache_position_start"], 5)
            self.assertEqual(report["cache_position_end"], 8)
            self.assertEqual(report["runtime_context"]["decode_step_count"], 6)
            self.assertEqual(
                report["runtime_context"]["decode_token_update_count"],
                7,
            )
            self.assertEqual(
                report["runtime_context"]["page_table_update_count"],
                6,
            )
            self.assertGreater(
                report["decode_runtime_tensor_conversion_count"],
                0,
            )
            self.assertFalse(
                report["host_copy_profile"]["runtime_host_roundtrip_present"]
            )
            self.assertEqual(report["section_profile"]["status"], "not_run")
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            decode_ops = [
                call
                for call in fake_ttnn.calls
                if call["op"] == "paged_scaled_dot_product_attention_decode"
            ]
            self.assertEqual(len(decode_ops), 6)
            self.assertEqual(json.loads(report_json.read_text()), report)


if __name__ == "__main__":
    unittest.main()
