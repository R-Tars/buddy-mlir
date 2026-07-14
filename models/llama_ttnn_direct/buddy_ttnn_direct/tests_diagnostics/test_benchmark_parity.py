from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity import (
    _finalize_report,
    _execute_planned_run,
    _latency_statistics,
    _load_resumable_artifact,
    _official_execution_features,
    _planned_commands,
    _parse_buddy_report,
    _parse_official_log,
    _resumable_runs,
    parse_official_accuracy_samples,
    run_benchmark_parity,
)


class BenchmarkParityTest(unittest.TestCase):
    def test_parse_official_accuracy_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "official.log"
            log_path.write_text(
                "BUDDY_ACCURACY_SAMPLE token_iteration=1 predicted_token=22\n"
                "BUDDY_ACCURACY_SAMPLE token_iteration=0 predicted_token=11\n"
            )
            self.assertEqual(
                parse_official_accuracy_samples(log_path),
                [11, 22],
            )

    def test_latency_statistics_reports_required_distribution(self) -> None:
        statistics = _latency_statistics(
            [
                {
                    "passed": True,
                    "decode_step_ms_samples": [10.0, 20.0, 30.0, 40.0],
                }
            ]
        )
        self.assertEqual(statistics["sample_count"], 4)
        self.assertEqual(statistics["mean"], 25.0)
        self.assertEqual(statistics["p50"], 25.0)
        self.assertEqual(statistics["p90"], 37.0)
        self.assertEqual(statistics["min"], 10.0)
        self.assertEqual(statistics["max"], 40.0)
        self.assertIsNotNone(statistics["stdev"])

    def test_official_execution_features_record_disabled_prefetcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = root / "current"
            release = root / "release"
            current_source = current / "models/tt_transformers/demo/conftest.py"
            release_source = release / "models/tt_transformers/demo/conftest.py"
            current_source.parent.mkdir(parents=True)
            release_source.parent.mkdir(parents=True)
            current_source.write_text("use_prefetcher = False\n")
            release_source.write_text("enable_trace = True\n")

            features = _official_execution_features(
                official_root=current,
                release_root=release,
            )

        current_profile = features["official-greedy"]
        self.assertFalse(current_profile["use_prefetcher"])
        self.assertTrue(current_profile["prefetcher_supported_by_source"])
        self.assertFalse(current_profile["global_cb_active"])
        self.assertIsNone(current_profile["global_cb"])
        self.assertIsNone(current_profile["sub_device_id"])
        self.assertTrue(current_profile["trace"])
        release_profile = features["official-release-demo"]
        self.assertFalse(release_profile["prefetcher_supported_by_source"])
        self.assertEqual(
            release_profile["sampling_mode"],
            "force argmax (temperature=0) in the release demo",
        )

    def test_parse_official_exact_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "official.log"
            log_path.write_text(
                "\n".join(
                    f"BUDDY_PARITY_SAMPLE token_iteration={iteration} "
                    f"duration_ms={30.0 + iteration / 10:.6f}"
                    for iteration in range(1, 8)
                )
                + "\n"
            )

            parsed = _parse_official_log(log_path, warmup=2, iterations=5)

            self.assertTrue(parsed["passed"])
            self.assertEqual(parsed["sample_source"], "benchmark_profiler_exact")
            self.assertEqual(parsed["warmup_step_ms_samples"], [30.1, 30.2])
            self.assertEqual(
                parsed["decode_step_ms_samples"],
                [30.3, 30.4, 30.5, 30.6, 30.7],
            )
            self.assertAlmostEqual(
                parsed["tokens_per_second_per_user"],
                1000.0 / 30.5,
            )

    def test_parse_official_rounded_log_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "official.log"
            log_path.write_text(
                "Iteration 0: 900ms @ 1.1 tok/s/user\n"
                "Iteration 1: 31ms @ 32.3 tok/s/user\n"
                "Iteration 2: 30ms @ 33.3 tok/s/user\n"
            )

            parsed = _parse_official_log(log_path, warmup=1, iterations=1)

            self.assertTrue(parsed["passed"])
            self.assertEqual(
                parsed["sample_source"],
                "official_debug_log_rounded_ms",
            )
            self.assertEqual(parsed["warmup_step_ms_samples"], [31.0])
            self.assertEqual(parsed["decode_step_ms_samples"], [30.0])

    def test_parse_buddy_report_checks_requested_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "profile.json"
            report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "warmup_step_ms_samples": [40.0, 39.0],
                        "decode_step_ms_samples": [35.0, 34.0, 33.0],
                    }
                )
            )

            parsed = _parse_buddy_report(
                report_path,
                warmup=2,
                iterations=3,
            )

            self.assertTrue(parsed["passed"])
            self.assertAlmostEqual(parsed["decode_step_ms_mean"], 34.0)
            self.assertEqual(parsed["sample_source"], "buddy_profile_json")

    def test_finalize_uses_greedy_official_as_primary(self) -> None:
        def run(profile: str, tpsu: float) -> dict[str, object]:
            return {
                "profile": profile,
                "passed": True,
                "tokens_per_second_per_user": tpsu,
                "decode_step_ms_samples": [30.0] * 4,
            }

        report = {
            "official_local_runs": [
                run("official-demo", value) for value in (34.0, 34.1, 33.9)
            ]
            + [run("official-greedy", value) for value in (33.0, 33.1, 32.9)],
            "buddy_local_runs": [
                run("buddy-greedy", value) for value in (31.4, 31.5, 31.6)
            ],
            "repetitions": 3,
            "iterations": 4,
            "same_tt_metal_commit": True,
        }

        _finalize_report(report)

        self.assertTrue(report["passed"])
        self.assertEqual(report["official_local_median_tpsu"], 33.0)
        self.assertEqual(report["buddy_local_median_tpsu"], 31.5)
        self.assertAlmostEqual(
            report["buddy_ratio_of_local_official"],
            31.5 / 33.0,
        )
        self.assertEqual(
            report["official_local_primary_profile"],
            "official-greedy",
        )
        self.assertEqual(
            report["decode_latency_statistics_ms"]["buddy-greedy"]["sample_count"],
            12,
        )

    def test_finalize_keeps_release_reference_out_of_primary_ratio(self) -> None:
        def run(profile: str, tpsu: float) -> dict[str, object]:
            return {
                "profile": profile,
                "passed": True,
                "tokens_per_second_per_user": tpsu,
                "first_decode_tokens_per_second_per_user": tpsu + 0.5,
                "decode_step_ms_samples": [1000.0 / tpsu] * 4,
            }

        report = {
            "official_local_runs": [
                run("official-demo", value) for value in (21.0, 21.1, 20.9)
            ]
            + [run("official-greedy", value) for value in (21.6, 21.7, 21.8)],
            "official_release_runs": [
                run("official-release-demo", value) for value in (33.0, 33.1, 33.2)
            ],
            "official_release": {
                "commit": ("b76035fbdac81d8f9974976471dc60fc005e1bfb"),
                "matches_external_release_commit": True,
            },
            "buddy_local_runs": [
                run("buddy-greedy", value) for value in (27.2, 27.3, 27.4)
            ],
            "repetitions": 3,
            "iterations": 4,
            "same_tt_metal_commit": True,
        }

        _finalize_report(report)

        self.assertTrue(report["passed"])
        self.assertEqual(report["official_local_median_tpsu"], 21.7)
        self.assertEqual(report["official_release_median_tpsu"], 33.1)
        self.assertAlmostEqual(
            report["buddy_ratio_of_local_official"],
            27.3 / 21.7,
        )
        self.assertAlmostEqual(
            report["buddy_ratio_of_release_official"],
            27.3 / 33.1,
        )
        self.assertFalse(
            report["semantic_match"]["release_reference_directly_comparable_to_buddy"]
        )

    def test_failure_is_written_before_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "parity.json"

            report = run_benchmark_parity(
                out=report_path,
                buddy_program=root / "missing-program",
                official_tt_metal_root=root / "missing-tt-metal",
                model_path=root / "missing-model",
                dry_run=True,
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report_path.is_file())
            on_disk = json.loads(report_path.read_text())
            self.assertEqual(on_disk["error"]["type"], "ValueError")

    def test_dry_run_plans_buddy_full_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                "benchmark_parity._git_value",
                return_value="fake-commit",
            ):
                plans = _planned_commands(
                    runs_root=root / "runs",
                    tensor_cache_root=root / "cache",
                    program_root=root / "program",
                    official_root=root / "official",
                    model_root=root / "model",
                    tokenizer_root=root / "tokenizer",
                    prompts_path=root / "prompts.json",
                    official_python=root / "python",
                    release_root=None,
                    release_python=None,
                    release_runtime_root=None,
                    layer_count=32,
                    batch_size=32,
                    effective_prefill_len=256,
                    cache_len=1024,
                    page_block_size=32,
                    warmup=5,
                    iterations=50,
                    repetitions=1,
                    device="p150a",
                    device_id=0,
                )

        buddy = next(plan for plan in plans if plan["implementation"] == "buddy")
        self.assertIn("persistent", buddy["command"])
        self.assertIn("trace", buddy["command"])

    @patch(
        "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
        "benchmark_parity.run_benchmark_parity"
    )
    def test_cli_forwards_benchmark_arguments(self, run_mock: object) -> None:
        run_mock.return_value = {"status": "passed", "passed": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "parity.json"

            exit_code = main(
                [
                    "diagnose",
                    "--stage",
                    "benchmark-parity",
                    "--buddy-program",
                    str(root / "program"),
                    "--official-tt-metal-root",
                    str(root / "tt-metal"),
                    "--official-python",
                    str(root / "official-python"),
                    "--official-release-root",
                    str(root / "release-root"),
                    "--official-release-python",
                    str(root / "release-python"),
                    "--official-release-runtime-root",
                    str(root / "release-runtime-root"),
                    "--model-path",
                    str(root / "model"),
                    "--input-prompts",
                    str(root / "prompts.json"),
                    "--batch-size",
                    "32",
                    "--prefill-len",
                    "128",
                    "--cache-len",
                    "1024",
                    "--warmup",
                    "5",
                    "--iterations",
                    "50",
                    "--repetitions",
                    "3",
                    "--out",
                    str(report_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["batch_size"], 32)
        self.assertEqual(kwargs["prefill_len"], 128)
        self.assertEqual(kwargs["iterations"], 50)
        self.assertEqual(kwargs["repetitions"], 3)
        self.assertEqual(kwargs["official_release_root"], root / "release-root")
        self.assertEqual(
            kwargs["official_release_python"],
            root / "release-python",
        )
        self.assertEqual(
            kwargs["official_release_runtime_root"],
            root / "release-runtime-root",
        )

    def test_release_source_build_uses_separate_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "clean-model-source"
            runtime_root = root / "same-commit-runtime"
            release_python = root / "release-env" / "bin" / "python"
            run_dir = root / "run"
            for path in (
                model_root,
                runtime_root / "ttnn",
                runtime_root / "tt_eager",
                runtime_root / "build" / "lib",
                release_python.parent,
            ):
                path.mkdir(parents=True, exist_ok=True)
            release_python.write_text("")
            captured: dict[str, object] = {}

            def runner(
                _command: object,
                cwd: Path,
                environment: dict[str, str],
                log_path: Path,
                _timeout: object,
                _limit: object,
            ) -> int:
                captured.update({"cwd": cwd, "environment": environment})
                log_path.write_text(
                    "BUDDY_PARITY_SAMPLE token_iteration=1 " "duration_ms=30.000000\n"
                )
                return 0

            result = _execute_planned_run(
                plan={
                    "implementation": "official",
                    "profile": "official-release-demo",
                    "comparison_scope": "release-reference",
                    "repetition": 1,
                    "run_dir": str(run_dir),
                    "log_path": str(run_dir / "run.log"),
                    "command": [str(release_python), "-m", "pytest"],
                    "model_path": str(root / "model"),
                    "official_root": str(model_root),
                    "runtime_root": str(runtime_root),
                    "runtime_mode": "source-build",
                    "tensor_cache_path": str(root / "cache"),
                },
                page_block_size=32,
                cache_len=1024,
                warmup=0,
                iterations=1,
                runner=runner,
                timeout_seconds=10,
                address_space_limit_bytes=None,
            )

        environment = captured["environment"]
        self.assertTrue(result["passed"])
        self.assertEqual(captured["cwd"], model_root)
        self.assertEqual(environment["TT_METAL_HOME"], str(runtime_root))
        self.assertEqual(
            environment["TT_METAL_RUNTIME_ROOT"],
            str(runtime_root),
        )
        self.assertIn(str(runtime_root / "ttnn"), environment["PYTHONPATH"])
        self.assertIn(
            str(runtime_root / "build" / "lib"),
            environment["LD_LIBRARY_PATH"],
        )

    def test_resumable_runs_only_reuses_matching_passes(self) -> None:
        current = {
            "buddy_program": "/program",
            "official_tt_metal_root": "/official",
            "model_path": "/model",
            "tokenizer_path": "/model",
            "input_prompts": "/prompts.json",
            "official_python": "/python",
            "official_release_root": None,
            "official_release_python": None,
            "official_release_runtime_root": None,
            "device": "p150a",
            "device_id": 0,
            "batch_size": 32,
            "requested_prefill_len": 128,
            "cache_len": 1024,
            "page_block_size": 32,
            "warmup": 5,
            "iterations": 100,
            "repetitions": 1,
        }
        command = ["python", "benchmark"]
        plans = [
            {
                "profile": "official-demo",
                "repetition": 1,
                "command": command,
            }
        ]
        previous = dict(current)
        previous.update(
            {
                "official_release_runs": [],
                "official_local_runs": [
                    {
                        "profile": "official-demo",
                        "repetition": 1,
                        "command": command,
                        "passed": True,
                    }
                ],
                "buddy_local_runs": [],
            }
        )

        resumed = _resumable_runs(
            previous,
            current_report=current,
            plans=plans,
        )
        self.assertEqual(set(resumed), {("official-demo", 1)})

        previous["iterations"] = 50
        self.assertEqual(
            _resumable_runs(
                previous,
                current_report=current,
                plans=plans,
            ),
            {},
        )

    def test_resumable_artifact_requires_successful_junit_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            log_path = run_dir / "run.log"
            (run_dir / "pytest.xml").write_text(
                '<testsuites><testsuite errors="0" failures="0" '
                'tests="1" /></testsuites>'
            )
            log_path.write_text(
                "BUDDY_PARITY_SAMPLE token_iteration=1 duration_ms=30.0\n"
            )
            plan = {
                "implementation": "official",
                "profile": "official-demo",
                "comparison_scope": "same-commit-local",
                "repetition": 1,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "command": ["python", "benchmark"],
                "model_path": "/model",
                "official_root": "/official",
                "runtime_root": "/official",
                "runtime_mode": "source-build",
                "tensor_cache_path": "/cache",
            }

            resumed = _load_resumable_artifact(
                plan,
                warmup=0,
                iterations=1,
            )
            self.assertIsNotNone(resumed)
            self.assertTrue(resumed["passed"])
            self.assertTrue((run_dir / "run_contract.json").is_file())

            (run_dir / "pytest.xml").write_text(
                '<testsuites><testsuite errors="1" failures="0" '
                'tests="1" /></testsuites>'
            )
            (run_dir / "run_contract.json").unlink()
            self.assertIsNone(
                _load_resumable_artifact(
                    plan,
                    warmup=0,
                    iterations=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
