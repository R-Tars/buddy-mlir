from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import (
    build_parser,
    main,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.correctness.artifacts import (
    kv_cache_snapshot,
    tensor_snapshot,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.correctness.hf_reference import (
    load_hf_reference,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.correctness.metrics import (
    compare_snapshots,
    compare_top_token,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.correctness.run import (
    run_correctness,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.profiling import (
    PROFILE_GENERATE_SECTION_KEYS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.observations import (
    TTNNObservationCollector,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.validation import (
    validate_device,
    validate_dryrun,
    validate_functional,
    validate_performance,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)


class ProductCliTest(unittest.TestCase):
    def test_provided_hf_reference_must_match_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            config_path = model_dir / "config.json"
            config_path.write_text('{"model_type": "llama"}\n')
            prompt = "hello"
            snapshot = {
                "sample_count": 1,
                "values": [0.0],
                "sha256": "0" * 64,
            }
            report = {
                "schema_version": 1,
                "kind": "hf_llama_correctness_reference",
                "status": "captured",
                "passed": True,
                "model_config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "layers": 1,
                "prefill_len": 8,
                "dtype": "bfloat16",
                "input_token_ids": [1, 2],
                "checkpoints": {
                    "prefill.layer.0.hidden": snapshot,
                    "prefill.layer.0.key_cache": snapshot,
                    "prefill.layer.0.value_cache": snapshot,
                    "prefill.final_hidden": snapshot,
                    "prefill.logits": snapshot,
                },
            }
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps(report))

            loaded = load_hf_reference(
                reference_path,
                model_path=model_dir,
                prompt=prompt,
                layers=1,
                prefill_len=8,
                dtype="bfloat16",
            )
            self.assertEqual(loaded["input_token_ids"], [1, 2])
            with self.assertRaisesRegex(ValueError, "prompt digest"):
                load_hf_reference(
                    reference_path,
                    model_path=model_dir,
                    prompt="different",
                    layers=1,
                    prefill_len=8,
                    dtype="bfloat16",
                )

    def test_correctness_runner_compares_captured_artifacts(self) -> None:
        import torch

        hidden = tensor_snapshot("prefill.layer.0.hidden", torch.arange(8.0))
        logits = tensor_snapshot("prefill.logits", torch.arange(16.0))
        reference = {
            "schema_version": 1,
            "kind": "hf_llama_correctness_reference",
            "status": "captured",
            "passed": True,
            "top_token": 15,
            "checkpoints": {
                "prefill.layer.0.hidden": hidden,
                "prefill.logits": logits,
            },
        }

        def fake_generate(**kwargs):
            kwargs["observer"].checkpoints.update(reference["checkpoints"])
            return {
                "passed": True,
                "status": "passed",
                "generated_token_ids": [[15]],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.correctness.run.capture_hf_reference",
                return_value=reference,
            ), patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.correctness.run.run_generate",
                side_effect=fake_generate,
            ):
                report = run_correctness(
                    out_dir=tmpdir,
                    program_dir="/tmp/program",
                    model_path="/tmp/model",
                    tokenizer_path="/tmp/model",
                    prompt="hello",
                    layers=1,
                    prefill_len=8,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    checks=("top_token", "logits_pcc", "hidden_pcc"),
                    ttnn_module=types.SimpleNamespace(),
                    torch_module=torch,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["comparisons"]), 3)
            self.assertEqual(report["failed_checks"], [])
            self.assertTrue((Path(tmpdir) / "hf_reference.json").is_file())
            self.assertTrue((Path(tmpdir) / "ttnn_observations.json").is_file())
            self.assertTrue((Path(tmpdir) / "validation_report.json").is_file())

    def test_correctness_observer_samples_hidden_logits_and_paged_kv(self) -> None:
        import torch

        class FakeOps:
            @staticmethod
            def select_sequence_position(tensor, position, **_kwargs):
                return tensor[:, position : position + 1, :]

            @staticmethod
            def slice_batch_user(tensor, user_id, **_kwargs):
                return tensor[user_id : user_id + 1]

        def slice_op(tensor, starts, ends, steps):
            return tensor[
                tuple(
                    slice(start, end, step)
                    for start, end, step in zip(
                        starts,
                        ends,
                        steps,
                        strict=True,
                    )
                )
            ]

        ttnn = types.SimpleNamespace(to_torch=lambda tensor: tensor, slice=slice_op)
        collector = TTNNObservationCollector(ttnn=ttnn, torch=torch)
        hidden = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8)
        logits = torch.arange(2 * 1 * 16, dtype=torch.float32).reshape(2, 1, 16)
        collector.observe(
            "prefill.layer.0.hidden",
            hidden,
            ops=FakeOps(),
            valid_seq_len=4,
        )
        collector.observe("prefill.logits", logits, ops=FakeOps())
        cache = torch.arange(4 * 2 * 4 * 8, dtype=torch.float32).reshape(
            4,
            2,
            4,
            8,
        )
        collector.observe_prefill_kv_cache(
            [types.SimpleNamespace(k=cache, v=cache + 1)],
            effective_token_count=6,
        )

        report = collector.to_report()
        self.assertEqual(report["checkpoint_count"], 4)
        self.assertEqual(
            report["checkpoints"]["prefill.layer.0.hidden"]["values"],
            hidden[0, 3].tolist(),
        )
        self.assertEqual(
            report["checkpoints"]["prefill.logits"]["sample_count"],
            16,
        )
        self.assertEqual(
            report["checkpoints"]["prefill.layer.0.key_cache"][
                "logical_shape"
            ],
            [1, 2, 6, 8],
        )

    def test_correctness_snapshots_and_metrics_are_deterministic(self) -> None:
        import torch

        reference = tensor_snapshot(
            "hidden",
            torch.tensor([1.0, 2.0, 4.0, 8.0]),
        )
        identical = tensor_snapshot(
            "hidden",
            torch.tensor([1.0, 2.0, 4.0, 8.0]),
        )
        shifted = tensor_snapshot(
            "hidden",
            torch.tensor([1.1, 2.1, 4.1, 8.1]),
        )

        self.assertEqual(reference["sha256"], identical["sha256"])
        exact = compare_snapshots(
            identical,
            reference,
            pcc_threshold=0.999,
            atol=0.0,
        )
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["pcc"], 1.0)
        self.assertEqual(exact["max_abs_error"], 0.0)

        tolerance_failure = compare_snapshots(
            shifted,
            reference,
            pcc_threshold=0.999,
            atol=0.05,
        )
        self.assertFalse(tolerance_failure["passed"])
        self.assertGreaterEqual(tolerance_failure["pcc"], 0.999)
        self.assertTrue(compare_top_token(42, 42)["passed"])
        self.assertFalse(compare_top_token(42, 7)["passed"])

    def test_kv_cache_snapshot_records_reproducible_coordinates(self) -> None:
        import torch

        cache = torch.arange(2 * 4 * 10 * 16, dtype=torch.float32).reshape(
            2,
            4,
            10,
            16,
        )
        snapshot = kv_cache_snapshot(
            "prefill.layer.0.key_cache",
            cache,
            max_heads=2,
            max_positions=3,
            max_channels=4,
        )

        self.assertEqual(snapshot["logical_shape"], [2, 4, 10, 16])
        self.assertEqual(snapshot["sample_shape"], [2, 3, 4])
        self.assertEqual(snapshot["sample_count"], 24)
        self.assertEqual(
            snapshot["sample_policy"],
            {
                "kind": "kv_coordinates",
                "batch_id": 0,
                "head_ids": [0, 3],
                "position_ids": [0, 4, 9],
                "channel_ids": [0, 5, 10, 15],
            },
        )

    def test_top_level_help_shows_product_commands_only(self) -> None:
        help_text = build_parser().format_help()

        for command in (
            "build",
            "generate",
            "profile",
            "validate",
            "inspect",
            "diagnose",
        ):
            self.assertIn(command, help_text)
        for legacy in (
            "build-program",
            "profile-generate",
            "validate-direct",
            "smoke-prefill",
            "generate-depth-sweep",
            "search",
        ):
            self.assertNotIn(legacy, help_text)

    def test_build_alias_writes_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

            exit_code = main(
                [
                    "build",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(program_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((program_dir / "model.py").is_file())
            self.assertTrue((program_dir / "run_decode.py").is_file())

    def test_profile_alias_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            report_json = root / "profile.json"
            underlying_generate = root / "profile_generate.json"

            exit_code = main(
                [
                    "profile",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "2",
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
                    str(underlying_generate),
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "profile-generate")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])

    def test_validate_dryrun_program_dir_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            out_dir = root / "validate"

            exit_code = main(
                [
                    "validate",
                    "--suite",
                    "dryrun",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "2",
                    "--prefill-len",
                    "8",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertEqual(report["command"], "validate")
            self.assertEqual(report["suite"], "dryrun")
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["failed_checks"], [])
            self.assertTrue((out_dir / "generate_dryrun.json").is_file())
            self.assertTrue((out_dir / "profile_dryrun.json").is_file())

    def test_validate_correctness_wires_product_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            out_dir = root / "correctness"
            result = {
                "schema_version": 1,
                "command": "validate",
                "suite": "correctness",
                "status": "pass",
                "passed": True,
                "runtime_status": "passed",
                "comparisons": [],
                "failed_checks": [],
            }
            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.correctness.run.run_correctness",
                return_value=result,
            ) as runner:
                exit_code = main(
                    [
                        "validate",
                        "--suite",
                        "correctness",
                        "--program-dir",
                        str(program_dir),
                        "--model-path",
                        str(root / "fake_model"),
                        "--prompt",
                        "hello",
                        "--layers",
                        "1",
                        "--batch-size",
                        "2",
                        "--prefill-len",
                        "8",
                        "--cache-len",
                        "16",
                        "--check",
                        "top_token,hidden_pcc",
                        "--out-dir",
                        str(out_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                runner.call_args.kwargs["checks"],
                ("top_token", "hidden_pcc"),
            )
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(report["artifacts"]["passed"])

    def test_product_validation_suites_use_compact_reports(self) -> None:
        artifacts = {"passed": True, "files": {"model.py": True}}
        dryrun = {"passed": True, "status": "dry_run", "dry_run": True}
        self.assertTrue(
            validate_dryrun(
                artifacts=artifacts,
                generate=dryrun,
                profile=dryrun,
            )["passed"]
        )

        generate = self._functional_generate_report()
        self.assertTrue(
            validate_functional(
                artifacts=artifacts,
                generate=generate,
            )["passed"]
        )
        self.assertTrue(
            validate_device(
                artifacts=artifacts,
                generate=generate,
                require_full_depth=True,
                expected_device="p150a",
                expected_device_id=0,
            )["passed"]
        )

        profile = {
            "passed": True,
            "status": "profiled",
            "tokens_per_second_per_user": 1.0,
            "layers": 2,
            "program_num_layers": 2,
            "sections": {
                name: {"status": "measured"}
                for name in PROFILE_GENERATE_SECTION_KEYS
            },
        }
        self.assertTrue(
            validate_performance(
                artifacts=artifacts,
                profile=profile,
                require_full_depth=True,
            )["passed"]
        )

        generate["generated_text"] = ""
        failed = validate_functional(
            artifacts=artifacts,
            generate=generate,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failed_checks"], ["validate.generated_text"])

    def test_validate_functional_uses_product_generate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            out_dir = root / "validate_functional"
            generate = self._functional_generate_report()

            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli.run_generate",
                return_value=generate,
            ) as run_generate:
                exit_code = main(
                    [
                        "validate",
                        "--suite",
                        "functional",
                        "--program-dir",
                        str(program_dir),
                        "--model-path",
                        str(root / "fake_model"),
                        "--prompt",
                        "hello",
                        "--out-dir",
                        str(out_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            run_generate.assert_called_once()
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertEqual(report["suite"], "functional")
            self.assertEqual(report["status"], "pass")
            self.assertNotIn("steps", report)

    def test_inspect_writes_program_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            report_json = root / "inspect.json"

            exit_code = main(
                [
                    "inspect",
                    "--program-dir",
                    str(program_dir),
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "inspect")
            self.assertTrue(report["passed"])
            self.assertEqual(report["artifacts"]["missing"], [])

    def test_diagnose_mlp_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = Path(tmpdir) / "diagnose_mlp.json"

            exit_code = main(
                [
                    "diagnose",
                    "--stage",
                    "mlp",
                    "--batch-size",
                    "2",
                    "--hidden-size",
                    "16",
                    "--intermediate-size",
                    "32",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])

    def _build_program(self, root: Path) -> Path:
        model_dir = root / "fake_model"
        config_json = root / "template_config.json"
        program_dir = root / "program"
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
        return program_dir

    @staticmethod
    def _functional_generate_report() -> dict[str, object]:
        return {
            "passed": True,
            "status": "passed",
            "prefill_status": "passed",
            "model_semantics": "prompt_conditioned_prefill_decode",
            "kv_cache_source": "prefill",
            "generated_text": "hello from ttnn",
            "generated_text_status": "decoded",
            "device": "p150a",
            "device_id": 0,
            "layers": 2,
            "program_num_layers": 2,
            "ttnn_environment": {
                "version": "test",
                "module_file": "/tmp/ttnn.py",
            },
        }


if __name__ == "__main__":
    unittest.main()
