from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics import decode_depth_sweep as decode_sweep
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.generate_depth_sweep import (
    _generate_record,
    run_generate_depth_sweep,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import (
    _fake_tokenizer_module,
    _fake_torch,
    _make_fake_ttnn,
    _make_generate_fake_ttnn,
    _write_fake_model_config,
    _write_template_config,
)


class _SweepFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model = self.root / "model"
        self.template = self.root / "template.json"
        self.program = self.root / "program"
        _write_fake_model_config(self.model)
        _write_template_config(self.template)

    def build(self, *, weights: bool = False) -> None:
        if weights:
            _write_fake_model_weights(self.model, _fake_weight_specs())
        self.assertEqual(main(["build", "--model-path", str(self.model), "--config", str(self.template), "--out-dir", str(self.program)]), 0)

    def minimal_program(self) -> None:
        self.program.mkdir()
        (self.program / "config.json").write_text('{"num_layers": 2}\n')


class DecodeDepthSweepTest(_SweepFixture):
    def test_resolve_decode_depths_defaults_to_review_progression(self) -> None:
        cases = ((None, 2, [1, 2]), (None, 6, [1, 2, 4, 6]), ("1,2,full", 2, [1, 2]))
        for value, layers, expected in cases:
            with self.subTest(value=value, layers=layers):
                self.assertEqual(decode_sweep.resolve_decode_depths(value, program_num_layers=layers), expected)
        with self.assertRaisesRegex(ValueError, "num_layers"):
            decode_sweep.resolve_decode_depths("1,4", program_num_layers=2)

    def test_cli_decode_depth_sweep_dry_run_writes_report(self) -> None:
        self.build()
        out = self.root / "depth_sweep.json"
        self.assertEqual(main(["diagnose", "--stage", "depth-sweep", "--program-dir", str(self.program), "--depths", "1,2,full", "--batch-size", "2", "--cache-len", "16", "--dry-run", "--out", str(out)]), 0)
        report = json.loads(out.read_text())
        self.assertEqual(
            {key: report[key] for key in ("schema_version", "command", "status", "passed", "program_num_layers", "depths", "covered_full_depth", "status_counts", "passed_depth_count", "failed_depths")},
            {"schema_version": 1, "command": "decode-depth-sweep", "status": "dry_run", "passed": True, "program_num_layers": 2, "depths": [1, 2], "covered_full_depth": True, "status_counts": {"dry_run": 2}, "passed_depth_count": 2, "failed_depths": []},
        )
        self.assertTrue(report["acceptance"]["passed"] and all(item["passed"] for item in report["acceptance"]["checks"]))
        self.assertEqual([item["layer_profile_ids"] for item in report["records"]], [[0], [0, 1]])
        self.assertEqual([item["layer_id"] for item in report["records"][1]["layer_profiles"]], [0, 1])
        self.assertIn("embedding_ms", report["records"][0]["section_latency_ms"])
        self.assertIn("argmax_status", report["records"][0]["lm_head_profile"])
        self.assertIn("decode_depth_sweep.profile_breakdown", [item["name"] for item in report["acceptance"]["checks"]])
        self.assertTrue(all((self.root / "depth_sweep_profiles" / f"profile_depth_{depth}.json").is_file() for depth in (1, 2)))

    def test_run_decode_depth_sweep_profiles_fake_real_weights(self) -> None:
        self.build(weights=True)
        with _fake_torch_and_safetensors():
            report = decode_sweep.run_decode_depth_sweep(
                out=self.root / "depth.json", program_dir=self.program, model_path=self.model,
                depths=[1, "full"], batch_size=2, cache_len=16, device="p150a",
                trace=True, trace_iterations=2, ttnn_module=_make_fake_ttnn(), torch_module=_fake_torch(),
            )
        self.assertEqual({key: report[key] for key in ("status", "passed", "depths", "status_counts", "reference_status_counts", "trace_status_counts", "failed_depths")}, {
            "status": "pass", "passed": True, "depths": [1, 2], "status_counts": {"profiled": 2},
            "reference_status_counts": {"passed": 2}, "trace_status_counts": {"captured_and_executed": 2}, "failed_depths": [],
        })
        one, two = report["records"]
        self.assertEqual((one["layer_profile_ids"], two["layer_profile_ids"]), ([0], [0, 1]))
        self.assertEqual([item["layer_id"] for item in two["layer_profiles"]], [0, 1])
        self.assertEqual([item["layer_id"] for item in two["output_shapes"]["kv_cache_layers"]], [0, 1])
        self.assertEqual((two["lm_head_profile"]["argmax_status"], two["output_shapes"]["token"], two["trace_iterations"]), ("profiled", [2, 1], 2))
        self.assertGreater(two["tokens_per_second_per_user"], 0.0)
        self.assertEqual(report["acceptance"]["failed_checks"], [])

    def test_run_decode_depth_sweep_accepts_partial_depth_when_allowed(self) -> None:
        self.build()
        report = decode_sweep.run_decode_depth_sweep(out=self.root / "depth.json", program_dir=self.program, depths=[1], batch_size=2, cache_len=16, dry_run=True, require_full_depth=False)
        self.assertEqual((report["status"], report["passed"], report["covered_full_depth"], report["require_full_depth"], report["depths"]), ("dry_run", True, False, False, [1]))
        self.assertEqual(report["acceptance"]["failed_checks"], [])
        self.assertNotIn("decode_depth_sweep.full_depth", [item["name"] for item in report["acceptance"]["checks"]])

    def test_decode_depth_sweep_reports_profile_failure(self) -> None:
        self.build(weights=True)
        original = decode_sweep.profile_decode_step

        def fail_depth_two(*args, **kwargs):
            profile = original(*args, **kwargs)
            if kwargs["layers"] == 2:
                profile.update(passed=False, status="reference_mismatch", error="forced depth two mismatch")
                profile["reference"]["status"] = "failed"
                Path(kwargs["out"]).write_text(json.dumps(profile) + "\n")
            return profile

        with patch.object(decode_sweep, "profile_decode_step", side_effect=fail_depth_two), _fake_torch_and_safetensors():
            report = decode_sweep.run_decode_depth_sweep(
                out=self.root / "depth.json", program_dir=self.program, model_path=self.model,
                depths=[1, 2], batch_size=2, cache_len=16, device="p150a",
                ttnn_module=_make_fake_ttnn(), torch_module=_fake_torch(),
            )
        self.assertEqual((report["status"], report["passed"], report["failed_depths"], report["status_counts"]["reference_mismatch"]), ("fail", False, [2], 1))
        self.assertIn("decode_depth_sweep.all_depths_passed", report["acceptance"]["failed_checks"])
        self.assertEqual(report["records"][1]["error"], "forced depth two mismatch")

    def test_decode_depth_sweep_fails_without_profile_breakdown(self) -> None:
        self.build(weights=True)
        original = decode_sweep.profile_decode_step

        def omit_breakdown(*args, **kwargs):
            profile = original(*args, **kwargs)
            profile.pop("lm_head_profile", None)
            Path(kwargs["out"]).write_text(json.dumps(profile) + "\n")
            return profile

        with patch.object(decode_sweep, "profile_decode_step", side_effect=omit_breakdown), _fake_torch_and_safetensors():
            report = decode_sweep.run_decode_depth_sweep(
                out=self.root / "depth.json", program_dir=self.program, model_path=self.model,
                depths=[1], batch_size=2, cache_len=16, device="p150a",
                ttnn_module=_make_fake_ttnn(), torch_module=_fake_torch(), require_full_depth=False,
            )
        self.assertEqual((report["status"], report["passed"]), ("fail", False))
        self.assertIn("decode_depth_sweep.profile_breakdown", report["acceptance"]["failed_checks"])
        check = next(item for item in report["acceptance"]["checks"] if item["name"] == "decode_depth_sweep.profile_breakdown")
        self.assertEqual(check["observed"][0]["lm_head_profile"], {})


class GenerateDepthSweepTest(_SweepFixture):
    def test_cli_generate_depth_sweep_dry_run_writes_reports(self) -> None:
        self.build()
        out, reports = self.root / "generate.json", self.root / "reports"
        self.assertEqual(main([
            "diagnose", "--stage", "generate-depth-sweep", "--program-dir", str(self.program),
            "--depths", "1,2,full", "--reports-dir", str(reports), "--max-new-tokens", "3",
            "--prefill-len", "8", "--batch-size", "2", "--cache-len", "16", "--dry-run", "--out", str(out),
        ]), 0)
        report = json.loads(out.read_text())
        expected = {
            "command": "generate-depth-sweep", "status": "dry_run", "passed": True,
            "program_num_layers": 2, "depths": [1, 2], "max_new_tokens": 3, "prefill_len": 8,
            "status_counts": {"dry_run": 2}, "prefill_status_counts": {"dry_run": 2},
            "generated_text_status_counts": {"not_run": 2},
            "model_semantics_counts": {"prompt_conditioned_prefill_decode": 2},
            "failed_depths": [], "failed_depth_diagnostics": [],
        }
        self.assertEqual({key: report[key] for key in expected}, expected)
        self.assertTrue(report["acceptance"]["passed"])
        self.assertEqual((report["acceptance"]["required_depths"], report["acceptance"]["require_full_depth"]), ([1, 2], False))
        for depth in (1, 2):
            payload = json.loads((reports / f"generate_depth_{depth}.json").read_text())
            self.assertEqual((payload["mode"], payload["status"], payload["model_semantics"]), ("generate", "dry_run", "prompt_conditioned_prefill_decode"))
            self.assertEqual((payload["parameter_tensorization_count_per_generate"], payload["parameter_tensorization_count_per_decode_step"], payload["kv_cache_reinitialized_per_step"]), (1, 0, False))

    def test_generate_depth_sweep_runs_fake_generate_depths(self) -> None:
        self.build(weights=True)
        with _fake_torch_and_safetensors():
            report = run_generate_depth_sweep(
                out=self.root / "generate.json", program_dir=self.program, model_path=self.model,
                prompt="hello tenstorrent", tokenizer_path=self.model, tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                depths=[1, "full"], max_new_tokens=3, prefill_len=8, batch_size=2, cache_len=16,
                device="p150a", ttnn_module=_make_generate_fake_ttnn(), torch_module=_fake_torch(),
            )
        expected = {
            "status": "pass", "passed": True, "depths": [1, 2], "status_counts": {"passed": 2},
            "prefill_status_counts": {"passed": 2}, "generated_text_status_counts": {"fallback": 2},
            "model_semantics_counts": {"prompt_conditioned_prefill_decode": 2}, "failed_depths": [],
            "failed_depth_diagnostics": [],
        }
        self.assertEqual({key: report[key] for key in expected}, expected)
        self.assertEqual(report["acceptance"]["failed_checks"], [])
        one, two = report["records"]
        self.assertEqual((one["end_to_end_failed_checks"], one["failure_diagnostics"]), ([], None))
        self.assertEqual((one["generated_token_count_by_user"], two["generated_token_count_by_user"]), ([3, 3], [3, 3]))
        self.assertEqual((two["generated_token_budget"]["total_planned_generated_tokens"], two["generated_token_budget"]["decode_loop_token_count"]), (3, 2))
        self.assertEqual((two["prefill_status"], two["kv_cache_source"], two["decode_loop_runtime_owned"], two["runtime_owner"]), ("passed", "prefill", True, "TTNNDirectRuntimeContext"))
        self.assertEqual((two["parameter_tensorization_count_per_generate"], two["parameter_tensorization_count_per_decode_step"], two["kv_cache_reinitialized_per_step"]), (1, 0, False))
        self.assertEqual((two["runtime_context"]["class"], two["generated_text"], len(two["prefill_cache_population"])), ("TTNNDirectRuntimeContext", "<tok:23> <tok:23> <tok:23>", 2))
        self.assertTrue(Path(two["generate_report"]).is_file())
        self.assertEqual(json.loads((self.root / "generate.json").read_text()), report)

    def test_generate_depth_sweep_writes_skipped_depth_reports(self) -> None:
        self.minimal_program()
        reports = self.root / "reports"

        def fake_generate(*, out, layers, **_kwargs):
            payload = _failed_generate(layers)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(payload) + "\n")
            return payload

        with patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.generate_depth_sweep.run_generate", side_effect=fake_generate) as mocked:
            report = run_generate_depth_sweep(out=self.root / "generate.json", program_dir=self.program, depths=[1, 2], reports_dir=reports, max_new_tokens=3, prefill_len=8, batch_size=2, cache_len=16, device="p150a")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual((report["status"], report["passed"], report["status_counts"], report["failed_depths"]), ("fail", False, {"reference_mismatch": 1, "skipped": 1}, [1]))
        self.assertIn("generate_depth_sweep.depth_1_passed", report["acceptance"]["failed_checks"])
        self.assertNotIn("generate_depth_sweep.report_files", report["acceptance"]["failed_checks"])
        one, two = report["records"]
        self.assertTrue(one["generate_report_exists"] and two["generate_report_exists"])
        self.assertEqual((two["status"], two["reason"]), ("skipped", "blocked by an earlier depth failure"))
        skipped = json.loads(Path(two["generate_report"]).read_text())
        self.assertEqual((skipped["mode"], skipped["status"], skipped["layers"], skipped["reason"]), ("generate", "skipped", 2, "blocked by an earlier depth failure"))

    def test_generate_depth_sweep_writes_exception_depth_reports(self) -> None:
        self.minimal_program()
        with patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.generate_depth_sweep.run_generate", side_effect=RuntimeError("synthetic depth failure")) as mocked:
            report = run_generate_depth_sweep(out=self.root / "generate.json", program_dir=self.program, depths=[1, 2], reports_dir=self.root / "reports", max_new_tokens=3, prefill_len=8, batch_size=2, cache_len=16, device="p150a")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual((report["status"], report["passed"], report["status_counts"], report["failed_depths"]), ("fail", False, {"fail": 1, "skipped": 1}, [1]))
        self.assertNotIn("generate_depth_sweep.report_files", report["acceptance"]["failed_checks"])
        one, two = report["records"]
        self.assertTrue(one["generate_report_exists"] and two["generate_report_exists"])
        self.assertEqual((one["status"], two["status"]), ("fail", "skipped"))
        diagnostics = report["failed_depth_diagnostics"][0]
        self.assertEqual((diagnostics["depth"], diagnostics["status"], diagnostics["generate_report"], diagnostics["error"]["type"]), (1, "fail", one["generate_report"], "RuntimeError"))
        exception = json.loads(Path(one["generate_report"]).read_text())
        self.assertEqual((exception["status"], exception["error"]["type"]), ("fail", "RuntimeError"))
        self.assertIn("synthetic depth failure", exception["error"]["message"])
        self.assertEqual(json.loads(Path(two["generate_report"]).read_text())["status"], "skipped")

    def test_generate_record_reports_failure_diagnostics(self) -> None:
        record = _generate_record(depth=2, generate=_failed_generate(2, detailed=True), report_path=Path("/tmp/generate_depth_2.json"))
        self.assertFalse(record["passed"])
        self.assertEqual((record["model_semantics"], record["end_to_end_failed_checks"]), ("prompt_conditioned_prefill_decode", ["generate.decode_loop_runtime_owned"]))
        diagnostics = record["failure_diagnostics"]
        self.assertEqual((diagnostics["depth"], diagnostics["generate_report"], diagnostics["error"]), (2, "/tmp/generate_depth_2.json", "generate reference mismatch"))
        cache = diagnostics["prefill"]["cache_population"][0]
        self.assertEqual((cache["update_shape_layout"], cache["key_update_shape"]), ("batch_heads_seq_head_dim", [1, 2, 8, 4]))
        step = diagnostics["decode"]["failed_step"]
        self.assertEqual((step["step_index"], step["input_shapes"]["token_ids"], step["output_shapes"]["token"]), (0, [2, 1], [2, 1]))
        self.assertEqual((step["reference_failed_checks"], step["observed_ops"], step["error"]), (["decode.output_shape"], ["paged_scaled_dot_product_attention_decode"], "decode output shape mismatch"))


def _failed_generate(layers: int, *, detailed: bool = False) -> dict[str, object]:
    cache = {
        "layer_id": layers - 1, "status": "filled", "write_policy": "paged_fill_cache_per_user",
        "update_shape_layout": "batch_heads_seq_head_dim", "key_update_shape": [1, 2, 8, 4],
        "value_update_shape": [1, 2, 8, 4],
    }
    if detailed:
        cache.update(key_cache_shape=[2, 2, 16, 4], value_cache_shape=[2, 2, 16, 4], page_table_shape=[2, 1], planned_user_count=2, filled_user_count=2)
    step = {
        "step_index": 0, "status": "reference_mismatch", "passed": False,
        "output_shapes": {"token": [2, 1] if detailed else [1, 1, 2]},
        "reference": {"status": "failed", "failed_checks": ["decode.output_shape"] if detailed else ["output.token"]},
    }
    if detailed:
        step.update(cache_position_value=8, input_shapes={"token_ids": [2, 1], "page_table": [2, 1]}, decode_runtime_state={"cache_position_value": 8}, rotary_runtime_state={"cos_shape": [2, 1, 64]}, error="decode output shape mismatch")
        step["reference"].update(observed_ops=["paged_scaled_dot_product_attention_decode"], expected_ops=["paged_scaled_dot_product_attention_decode"])
    return {
        "schema_version": 1, "command": "generate", "mode": "generate", "status": "reference_mismatch",
        "passed": False, "layers": layers, "batch_size": 2, "cache_len": 16, "prefill_len": 8,
        "max_new_tokens": 3, "layout": "tile", "model_semantics": "prompt_conditioned_prefill_decode",
        "prefill_status": "passed", "kv_cache_source": "prefill", "decode_loop_runtime_owned": False,
        "generate_runtime_owned": False, "generated_text_status": "not_run", "generated_token_ids": [],
        "throughput_summary": {}, "end_to_end_contract": {"status": "failed", "failed_checks": ["generate.decode_loop_runtime_owned"]},
        "prefill": {"status": "passed", "cache_population": [cache]}, "step_reports": [step],
        "error": "generate reference mismatch" if detailed else "generate structural reference mismatch",
    }
