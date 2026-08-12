from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics import official_support
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.official_support import (
    FORCE_ARGMAX_FIX_COMMIT,
    git_commit_ancestry,
    measurement_provenance,
    reference_provenance,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.process_support import (
    git_value,
    run_logged_command,
    sha256_file,
)

class OfficialSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_parsers_keep_exact_samples_and_accuracy_order(self) -> None:
        log = self.root / "run.log"
        log.write_text(
            "BUDDY_PARITY_SAMPLE token_iteration=1 duration_ms=20.000000\n"
            "BUDDY_PARITY_SAMPLE token_iteration=0 duration_ms=21.000000\n"
            "BUDDY_ACCURACY_SAMPLE token_iteration=1 predicted_token=22\n"
            "BUDDY_ACCURACY_SAMPLE token_iteration=0 predicted_token=11\n"
        )
        latency = official_support.parse_official_latency_report(log, warmup=1, iterations=1)
        self.assertEqual(latency["decode_step_ms_samples"], [20.0])
        self.assertEqual(official_support.parse_official_accuracy_samples(log), [11, 22])

    def test_page_params_command_and_environment_are_canonical(self) -> None:
        self.assertEqual(
            official_support.encode_page_params(1024, 32),
            '{"page_block_size":32,"page_max_num_blocks_per_dp":1024}',
        )
        command = official_support.official_pytest_command(
            "/env/bin/python", "selector", ["--batch_size", "32"]
        )
        self.assertEqual(command[:8], [
            "/env/bin/python", "-m", "pytest", "-s", "-q",
            "models/tt_transformers/demo/simple_text_demo.py", "-k", "selector",
        ])
        self.assertIn("-p", command)
        polluted = {"PYTEST_PLUGINS": "external.conftest", "PYTEST_ADDOPTS": "-p external",
                    "LLAMA_DIR": "stale"}
        with patch.dict(os.environ, polluted):
            source_environment = official_support.official_source_environment(
                Path("/metal"), Path("/model"), Path("/cache"), "official-greedy", 1024, 32
            )
        self.assertTrue(all(name not in source_environment
                            for name in ("PYTEST_PLUGINS", "PYTEST_ADDOPTS", "LLAMA_DIR")))
        self.assertEqual(source_environment["MESH_DEVICE"], "P150")
        self.assertEqual(source_environment[official_support.PYTEST_PROFILE_ENV], "official-greedy")

    def test_plugin_selects_profile_and_accuracy_hooks(self) -> None:
        class Config:
            option = types.SimpleNamespace()

        profiler = type("BenchmarkProfiler", (), {"end": lambda *_args, **_kwargs: None})
        benchmarking = types.ModuleType("models.perf.benchmarking_utils")
        benchmarking.BenchmarkProfiler = profiler
        environment = {
            official_support.PYTEST_PLUGIN_ENABLED_ENV: "1",
            official_support.PYTEST_PROFILE_ENV: "official-greedy",
            official_support.PYTEST_PAGE_PARAMS_ENV: '{"page_block_size":32}',
        }
        with patch.dict(os.environ, environment, clear=False), patch.dict(
            sys.modules, {"models.perf.benchmarking_utils": benchmarking}
        ):
            official_support.pytest_configure(Config())
        self.assertEqual(Config.option.page_params, {"page_block_size": 32})
        self.assertTrue(Config.option.enable_trace)
        self.assertEqual(Config.option.stop_at_eos, 0)
        self.assertEqual(Config.option.sampling_params["temperature"], 0)

    def test_graph_capture_wrapper_records_operations_without_graph_state(self) -> None:
        class Device:
            def num_program_cache_entries(self) -> int:
                return 7

        class FastOperation:
            def __init__(self, name: str) -> None:
                self.python_fully_qualified_name = name

            def __call__(self, value: object, **_kwargs: object) -> object:
                return value

        events: list[str] = []
        fake_ttnn = types.ModuleType("ttnn")
        decorators = types.ModuleType("ttnn.decorators")
        decorators.FastOperation = FastOperation

        def begin_trace(*_args: object, **_kwargs: object) -> str:
            events.append("trace-begin")
            return "trace-1"

        def end_trace(*_args: object, **_kwargs: object) -> None:
            events.append("trace-end")

        fake_ttnn.begin_trace_capture = begin_trace
        fake_ttnn.end_trace_capture = end_trace
        modules = {"ttnn": fake_ttnn, "ttnn.decorators": decorators}
        with patch.dict(sys.modules, modules):
            official_support._install_official_trace_graph_capture(self.root)
            trace = fake_ttnn.begin_trace_capture(Device())
            self.assertEqual(FastOperation("ttnn.reshape")("tensor", shape=[1, 32]), "tensor")
            fake_ttnn.end_trace_capture(Device(), trace)
            manifest = json.loads((self.root / "manifest.json").read_text())
            records = json.loads((self.root / "trace_000.python_io.json").read_text())
        self.assertEqual(manifest["captures"][0]["program_cache_entries_before"], 7)
        self.assertEqual(manifest["captures"][0]["program_cache_entries_after"], 7)
        self.assertEqual(manifest["captures"][0]["operation_count"], 1)
        self.assertEqual(records[0]["name"], "ttnn.reshape")
        self.assertEqual(records[0]["arguments"]["positional_types"], ["tensor"])
        self.assertEqual(events, ["trace-begin", "trace-end"])

    def test_graph_capture_wrapper_restores_state_after_trace_failure(self) -> None:
        class FastOperation:
            python_fully_qualified_name = "ttnn.reshape"

            def __call__(self, value: object) -> object:
                return value

        fake_ttnn = types.ModuleType("ttnn")
        decorators = types.ModuleType("ttnn.decorators")
        decorators.FastOperation = FastOperation
        fake_ttnn.begin_trace_capture = lambda *_args, **_kwargs: "trace-1"

        def fail_end(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("trace end failed")

        fake_ttnn.end_trace_capture = fail_end
        modules = {"ttnn": fake_ttnn, "ttnn.decorators": decorators}
        with patch.dict(sys.modules, modules):
            official_support._install_official_trace_graph_capture(self.root)
            trace = fake_ttnn.begin_trace_capture(object())
            with self.assertRaisesRegex(RuntimeError, "trace end failed"):
                fake_ttnn.end_trace_capture(object(), trace)
            self.assertFalse((self.root / "manifest.json").exists())
            second_trace = fake_ttnn.begin_trace_capture(object())
            self.assertEqual(second_trace, "trace-1")

    def test_process_runner_combines_output_and_identity_helpers(self) -> None:
        log = self.root / "nested/run.log"
        code = "import sys; print('stdout'); print('stderr', file=sys.stderr)"
        self.assertEqual(run_logged_command(
            [sys.executable, "-c", code], self.root, dict(os.environ), log, 10, None
        ), 0)
        self.assertEqual(set(log.read_text().splitlines()), {"stdout", "stderr"})
        payload = self.root / "payload.bin"
        payload.write_bytes(b"payload")
        self.assertEqual(len(sha256_file(payload) or ""), 64)
        self.assertEqual(len(git_value(Path.cwd(), "rev-parse", "HEAD")), 40)

    def provenance(self, *, buddy: str = "A", local: str = "A", current: str | None = "B", fix: bool | None = False) -> dict[str, object]:
        return reference_provenance(
            buddy_sha=buddy, local_official_sha=local, current_main_sha=current,
            pinned_release_sha="R", pinned_release_tpsu=33.1,
            local_force_argmax_fix_present=fix,
            current_main_force_argmax_fix_present=True,
        )

    def test_same_runtime_commit_does_not_imply_current_main(self) -> None:
        report = self.provenance()
        self.assertTrue(report["same_runtime_commit_comparable"])
        self.assertFalse(report["current_main_comparable"])

    def test_current_main_comparable_requires_same_sha(self) -> None:
        self.assertFalse(self.provenance(local="A", current="B")["current_main_comparable"])
        self.assertTrue(self.provenance(buddy="B", local="B", current="B")["current_main_comparable"])

    @patch("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.official_support.subprocess.run")
    def test_force_argmax_fix_ancestry_is_reported(self, run_mock: object) -> None:
        run_mock.return_value.returncode = 1
        run_mock.return_value.stderr = ""
        report = git_commit_ancestry(self.root)
        self.assertEqual((report["ancestor"], report["contains_ancestor"], report["exit_status"]),
                         (FORCE_ARGMAX_FIX_COMMIT, False, 1))

    def test_old_local_commit_is_labeled_same_runtime_only(self) -> None:
        report = self.provenance(fix=False)
        scope = report["official_reference_provenance"]["same_runtime_commit"]
        self.assertEqual(scope["comparison_scope"], "same-runtime-commit")
        self.assertFalse(scope["contains_single_chip_force_argmax_fix"])

    def test_short_harness_measurement_is_not_formal_baseline(self) -> None:
        smoke = measurement_provenance(repetitions=1, warmup=1, iterations=3)
        formal = measurement_provenance(repetitions=3, warmup=5, iterations=100)
        self.assertEqual((smoke["classification"], smoke["not_a_performance_baseline"]),
                         ("harness_smoke", True))
        self.assertEqual((formal["classification"], formal["baseline_comparable"]),
                         ("formal_product_performance", True))

    def test_reference_provenance_has_local_release_current_scopes(self) -> None:
        scopes = self.provenance()["official_reference_provenance"]
        self.assertEqual(set(scopes), {"same_runtime_commit", "pinned_release", "current_main"})
        self.assertEqual(scopes["pinned_release"]["comparison_scope"],
                         "cross-version-release-reference")

if __name__ == "__main__":
    unittest.main()
