from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity import (
    _finalize_report,
    _parse_buddy_report,
    _parse_official_log,
    run_benchmark_parity,
)


class BenchmarkParityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
