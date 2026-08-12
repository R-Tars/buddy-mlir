from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity import (
    _collect_metadata,
    _execute_planned_run,
    _finalize_report,
    _latency_statistics,
    _load_resumable_artifact,
    _official_execution_features,
    _parse_buddy_report,
    _parse_official_log,
    _planned_commands,
    _resumable_runs,
    parse_official_accuracy_samples,
    run_benchmark_parity,
)

def _run(profile: str, tpsu: float) -> dict[str, object]:
    return {
        "profile": profile,
        "passed": True,
        "tokens_per_second_per_user": tpsu,
        "first_decode_tokens_per_second_per_user": tpsu + 0.5,
        "decode_step_ms_samples": [1000.0 / tpsu] * 4,
    }

def _resume_contract() -> dict[str, object]:
    return {
        "buddy_program": "/program", "official_tt_metal_root": "/official",
        "model_path": "/model", "tokenizer_path": "/model", "input_prompts": "/prompts.json",
        "official_python": "/python", "official_release_root": None,
        "official_release_python": None, "official_release_runtime_root": None,
        "device": "p150a", "device_id": 0, "batch_size": 32,
        "requested_prefill_len": 128, "cache_len": 1024, "page_block_size": 32,
        "warmup": 5, "iterations": 100, "repetitions": 1,
    }

class BenchmarkParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_parse_official_accuracy_samples(self) -> None:
        path = self.write("official.log", "BUDDY_ACCURACY_SAMPLE token_iteration=1 predicted_token=22\nBUDDY_ACCURACY_SAMPLE token_iteration=0 predicted_token=11\n")
        self.assertEqual(parse_official_accuracy_samples(path), [11, 22])

    def test_latency_statistics_reports_required_distribution(self) -> None:
        result = _latency_statistics([{"passed": True, "decode_step_ms_samples": [10.0, 20.0, 30.0, 40.0]}])
        self.assertEqual({key: result[key] for key in ("sample_count", "mean", "p50", "p90", "min", "max")}, {
            "sample_count": 4, "mean": 25.0, "p50": 25.0, "p90": 37.0, "min": 10.0, "max": 40.0,
        })
        self.assertIsNotNone(result["stdev"])

    def test_official_execution_features_record_disabled_prefetcher(self) -> None:
        current, release = self.root / "current", self.root / "release"
        self.write("current/models/tt_transformers/demo/conftest.py", "use_prefetcher = False\n")
        self.write("release/models/tt_transformers/demo/conftest.py", "enable_trace = True\n")
        features = _official_execution_features(official_root=current, release_root=release)
        profile = features["official-greedy"]
        self.assertEqual({key: profile[key] for key in ("use_prefetcher", "prefetcher_supported_by_source", "global_cb_active", "global_cb", "sub_device_id", "trace")}, {
            "use_prefetcher": False, "prefetcher_supported_by_source": True, "global_cb_active": False,
            "global_cb": None, "sub_device_id": None, "trace": True,
        })
        self.assertFalse(features["official-release-demo"]["prefetcher_supported_by_source"])
        self.assertEqual(features["official-release-demo"]["sampling_mode"], "force argmax (temperature=0) in the release demo")

    def test_parse_official_exact_samples(self) -> None:
        path = self.write("official.log", "\n".join(f"BUDDY_PARITY_SAMPLE token_iteration={i} duration_ms={30 + i / 10:.6f}" for i in range(1, 8)) + "\n")
        parsed = _parse_official_log(path, warmup=2, iterations=5)
        self.assertEqual((parsed["passed"], parsed["sample_source"], parsed["warmup_step_ms_samples"], parsed["decode_step_ms_samples"]), (True, "benchmark_profiler_exact", [30.1, 30.2], [30.3, 30.4, 30.5, 30.6, 30.7]))
        self.assertAlmostEqual(parsed["tokens_per_second_per_user"], 1000.0 / 30.5)

    def test_parse_official_rounded_log_fallback(self) -> None:
        path = self.write("official.log", "Iteration 0: 900ms @ 1.1 tok/s/user\nIteration 1: 31ms @ 32.3 tok/s/user\nIteration 2: 30ms @ 33.3 tok/s/user\n")
        parsed = _parse_official_log(path, warmup=1, iterations=1)
        self.assertEqual((parsed["passed"], parsed["sample_source"], parsed["warmup_step_ms_samples"], parsed["decode_step_ms_samples"]), (True, "official_debug_log_rounded_ms", [31.0], [30.0]))

    def test_parse_buddy_report_checks_requested_sample_counts(self) -> None:
        path = self.write("profile.json", json.dumps({"passed": True, "warmup_step_ms_samples": [40.0, 39.0], "decode_step_ms_samples": [35.0, 34.0, 33.0]}))
        parsed = _parse_buddy_report(path, warmup=2, iterations=3)
        self.assertEqual((parsed["passed"], parsed["decode_step_ms_mean"], parsed["sample_source"]), (True, 34.0, "buddy_profile_json"))

    def test_finalize_uses_greedy_official_as_primary(self) -> None:
        report = {
            "official_local_runs": [_run("official-demo", value) for value in (34.0, 34.1, 33.9)] + [_run("official-greedy", value) for value in (33.0, 33.1, 32.9)],
            "buddy_local_runs": [_run("buddy-greedy", value) for value in (31.4, 31.5, 31.6)],
            "repetitions": 3, "iterations": 4, "same_tt_metal_commit": True,
        }
        _finalize_report(report)
        self.assertEqual((report["passed"], report["official_local_median_tpsu"], report["buddy_local_median_tpsu"], report["official_local_primary_profile"]), (True, 33.0, 31.5, "official-greedy"))
        self.assertAlmostEqual(report["buddy_ratio_of_local_official"], 31.5 / 33.0)
        self.assertEqual(report["decode_latency_statistics_ms"]["buddy-greedy"]["sample_count"], 12)

    def test_finalize_keeps_release_reference_out_of_primary_ratio(self) -> None:
        report = {
            "official_local_runs": [_run("official-demo", value) for value in (21.0, 21.1, 20.9)] + [_run("official-greedy", value) for value in (21.6, 21.7, 21.8)],
            "official_release_runs": [_run("official-release-demo", value) for value in (33.0, 33.1, 33.2)],
            "official_release": {"commit": "b76035fbdac81d8f9974976471dc60fc005e1bfb", "matches_external_release_commit": True},
            "buddy_local_runs": [_run("buddy-greedy", value) for value in (27.2, 27.3, 27.4)],
            "repetitions": 3, "iterations": 4, "same_tt_metal_commit": True,
        }
        _finalize_report(report)
        self.assertEqual((report["passed"], report["official_local_median_tpsu"], report["official_release_median_tpsu"]), (True, 21.7, 33.1))
        self.assertAlmostEqual(report["buddy_ratio_of_local_official"], 27.3 / 21.7)
        self.assertAlmostEqual(report["buddy_ratio_of_release_official"], 27.3 / 33.1)
        self.assertFalse(report["semantic_match"]["release_reference_directly_comparable_to_buddy"])

    def test_mismatched_local_commit_suppresses_accepted_ratio(self) -> None:
        report = {
            "official_local_runs": [_run("official-demo", 21.0), _run("official-greedy", 22.0)],
            "buddy_local_runs": [_run("buddy-greedy", 35.0)],
            "repetitions": 1, "iterations": 4, "same_tt_metal_commit": True,
            "same_runtime_commit_comparable": False,
        }
        _finalize_report(report)
        self.assertIsNone(report["buddy_ratio_of_local_official"])
        self.assertFalse(report["passed"])
        self.assertIn("same-runtime-commit-comparable", report["acceptance"]["failed_checks"])

    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity._token_lengths", return_value=[16])
    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity._current_main_commit", return_value="C")
    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity.git_commit_ancestry", return_value={"contains_ancestor": False, "exit_status": 1, "error": None})
    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity._git_value")
    def test_collect_metadata_records_mismatched_runtime_provenance(
        self, git_value: object, _ancestry: object, _main: object, _tokens: object
    ) -> None:
        program, official, model, prompts = (
            self.root / "program", self.root / "official", self.root / "model", self.root / "prompts.json"
        )
        for path in (program, official, model):
            path.mkdir()
        (program / "config.json").write_text('{"num_layers": 32}')
        prompts.write_text('["prompt"]')
        python = self.write("python", "")
        git_value.side_effect = ["C", "A", "B"]
        context = {
            "program_root": program, "official_root": official, "model_root": model,
            "tokenizer_root": model, "prompts_path": prompts, "official_python": python,
            "release_root": None, "release_python": None, "release_runtime_root": None,
            "batch_size": 1, "prefill_len": 128, "cache_len": 1024,
            "repetitions": 1, "warmup": 1, "iterations": 3, "device": "p150a",
        }
        with patch.dict("os.environ", {"TT_METAL_HOME": str(self.root / "buddy-runtime")}):
            metadata = _collect_metadata(context)
        self.assertFalse(metadata["same_tt_metal_commit"])
        self.assertFalse(metadata["same_runtime_commit_comparable"])
        self.assertEqual(
            metadata["official_reference_provenance"]["same_runtime_commit"]["reason"],
            "version_mismatch",
        )

    def test_failure_is_written_before_return(self) -> None:
        out = self.root / "parity.json"
        report = run_benchmark_parity(out=out, buddy_program=self.root / "missing-program", official_tt_metal_root=self.root / "missing-tt-metal", model_path=self.root / "missing-model", dry_run=True)
        self.assertEqual((report["passed"], report["status"], out.is_file(), json.loads(out.read_text())["error"]["type"]), (False, "failed", True, "ValueError"))

    def test_dry_run_plans_buddy_full_trace(self) -> None:
        with patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity._git_value", return_value="fake-commit"):
            plans = _planned_commands(
                runs_root=self.root / "runs", tensor_cache_root=self.root / "cache", program_root=self.root / "program",
                official_root=self.root / "official", model_root=self.root / "model", tokenizer_root=self.root / "tokenizer",
                prompts_path=self.root / "prompts.json", official_python=self.root / "python", release_root=None,
                release_python=None, release_runtime_root=None, layer_count=32, batch_size=32, effective_prefill_len=256,
                cache_len=1024, page_block_size=32, warmup=5, iterations=50, repetitions=1, device="p150a", device_id=0,
            )
        command = next(item["command"] for item in plans if item["implementation"] == "buddy")
        self.assertIn("persistent", command)
        self.assertIn("trace", command)

    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity.run_benchmark_parity")
    def test_cli_forwards_benchmark_arguments(self, run_mock: object) -> None:
        run_mock.return_value = {"status": "passed", "passed": True}
        args = [
            "diagnose", "--stage", "benchmark-parity", "--buddy-program", str(self.root / "program"),
            "--official-tt-metal-root", str(self.root / "tt-metal"), "--official-python", str(self.root / "official-python"),
            "--official-release-root", str(self.root / "release-root"), "--official-release-python", str(self.root / "release-python"),
            "--official-release-runtime-root", str(self.root / "release-runtime-root"), "--model-path", str(self.root / "model"),
            "--input-prompts", str(self.root / "prompts.json"), "--batch-size", "32", "--prefill-len", "128",
            "--cache-len", "1024", "--warmup", "5", "--iterations", "50", "--repetitions", "3", "--out", str(self.root / "parity.json"),
        ]
        self.assertEqual(main(args), 0)
        kwargs = run_mock.call_args.kwargs
        expected = {"batch_size": 32, "prefill_len": 128, "iterations": 50, "repetitions": 3, "official_release_root": self.root / "release-root", "official_release_python": self.root / "release-python", "official_release_runtime_root": self.root / "release-runtime-root"}
        self.assertEqual({key: kwargs[key] for key in expected}, expected)

    def test_release_source_build_uses_separate_runtime_root(self) -> None:
        model, runtime = self.root / "model-source", self.root / "runtime"
        python, run_dir = self.root / "env/bin/python", self.root / "run"
        for path in (model, runtime / "ttnn", runtime / "tt_eager", runtime / "build/lib", python.parent):
            path.mkdir(parents=True, exist_ok=True)
        python.write_text("")
        captured: dict[str, object] = {}

        def runner(_command, cwd, environment, log_path, _timeout, _limit):
            captured.update(cwd=cwd, environment=environment)
            log_path.write_text("BUDDY_PARITY_SAMPLE token_iteration=1 duration_ms=30.000000\n")
            return 0

        result = _execute_planned_run(
            plan={"implementation": "official", "profile": "official-release-demo", "comparison_scope": "release-reference", "repetition": 1, "run_dir": str(run_dir), "log_path": str(run_dir / "run.log"), "command": [str(python), "-m", "pytest"], "model_path": str(self.root / "model"), "official_root": str(model), "runtime_root": str(runtime), "runtime_mode": "source-build", "tensor_cache_path": str(self.root / "cache")},
            page_block_size=32, cache_len=1024, warmup=0, iterations=1, runner=runner, timeout_seconds=10, address_space_limit_bytes=None,
        )
        env = captured["environment"]
        self.assertTrue(result["passed"])
        self.assertEqual((captured["cwd"], env["TT_METAL_HOME"], env["TT_METAL_RUNTIME_ROOT"]), (model, str(runtime), str(runtime)))
        self.assertIn(str(runtime / "ttnn"), env["PYTHONPATH"])
        self.assertIn(str(runtime / "build/lib"), env["LD_LIBRARY_PATH"])

    def test_resumable_runs_only_reuses_matching_passes(self) -> None:
        current, command = _resume_contract(), ["python", "benchmark"]
        plans = [{"profile": "official-demo", "repetition": 1, "command": command}]
        previous = {**current, "official_release_runs": [], "official_local_runs": [{"profile": "official-demo", "repetition": 1, "command": command, "passed": True}], "buddy_local_runs": []}
        self.assertEqual(set(_resumable_runs(previous, current_report=current, plans=plans)), {("official-demo", 1)})
        previous["iterations"] = 50
        self.assertEqual(_resumable_runs(previous, current_report=current, plans=plans), {})

    def test_resumable_artifact_requires_successful_junit_and_samples(self) -> None:
        run_dir = self.root
        self.write("pytest.xml", '<testsuites><testsuite errors="0" failures="0" tests="1" /></testsuites>')
        log = self.write("run.log", "BUDDY_PARITY_SAMPLE token_iteration=1 duration_ms=30.0\n")
        plan = {"implementation": "official", "profile": "official-demo", "comparison_scope": "same-commit-local", "repetition": 1, "run_dir": str(run_dir), "log_path": str(log), "command": ["python", "benchmark"], "model_path": "/model", "official_root": "/official", "runtime_root": "/official", "runtime_mode": "source-build", "tensor_cache_path": "/cache"}
        resumed = _load_resumable_artifact(plan, warmup=0, iterations=1)
        self.assertIsNotNone(resumed)
        self.assertTrue(resumed["passed"] and (run_dir / "run_contract.json").is_file())
        self.write("pytest.xml", '<testsuites><testsuite errors="1" failures="0" tests="1" /></testsuites>')
        (run_dir / "run_contract.json").unlink()
        self.assertIsNone(_load_resumable_artifact(plan, warmup=0, iterations=1))
