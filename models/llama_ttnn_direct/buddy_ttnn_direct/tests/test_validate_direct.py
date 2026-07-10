from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct import (
    validation as validation_module,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_diff import (
    PARITY_SECTIONS,
    REQUIRED_PARITY_PATHS,
    build_config_parity_view,
    default_official_config_path,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.ttnn_tensorizer import (
    LINEAR_WEIGHT_TRANSFORM,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.evidence import (
    artifact_evidence,
    artifact_index,
    step_names_with_status,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.performance import (
    performance_gap_summary,
    resolve_performance_baseline as resolve_report_performance_baseline,
    throughput_baseline_summary,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.schema import (
    acceptance_check,
    int_list,
    path_exists,
    positive_number,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.search.decode_step_autotune import (
    DECODE_STEP_AUTOTUNE_KNOBS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_layer import (
    ATTENTION_LAYER_OPS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.validation import (
    REAL_DECODE_VALIDATION_STEPS,
    VALIDATION_STEPS,
    preflight_real_decode,
    recover_real_decode_process_failure,
    recover_real_decode_process_timeout,
    validate_real_decode,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_single_layer_decode import (
    _fake_tokenizer_module,
    _make_fake_ttnn,
)


class ValidateDirectTest(unittest.TestCase):
    def test_validation_evidence_helpers_reexport_compatibly(self) -> None:
        self.assertIs(validation_module._artifact_index, artifact_index)
        self.assertIs(validation_module._artifact_evidence, artifact_evidence)
        self.assertIs(
            validation_module._step_names_with_status,
            step_names_with_status,
        )
        self.assertEqual(
            validation_module._step_names_with_status(
                {
                    "ok": "pass",
                    "pending": "pending",
                    "bad": "runtime_error",
                },
                failing=True,
            ),
            ["bad"],
        )

    def test_validation_schema_helpers_reexport_compatibly(self) -> None:
        self.assertIs(validation_module._acceptance_check, acceptance_check)
        self.assertIs(validation_module._path_exists, path_exists)
        self.assertIs(validation_module._positive_number, positive_number)
        self.assertIs(validation_module._int_list, int_list)
        self.assertEqual(
            validation_module._acceptance_check(
                "schema.sample",
                True,
                observed="ok",
            ),
            {
                "name": "schema.sample",
                "passed": True,
                "observed": "ok",
            },
        )
        self.assertEqual(validation_module._int_list(["1", 2]), [1, 2])

    def test_performance_baseline_reference_resolves(self) -> None:
        baseline = validation_module.resolve_performance_baseline(
            "tt_metal_official_llama31_8b_b32"
        )
        report_baseline = resolve_report_performance_baseline(
            "tt_metal_official_llama31_8b_b32"
        )
        self.assertEqual(baseline["model"], "Llama 3.1 8B")
        self.assertEqual(baseline["role"], "official_8b_target")
        self.assertEqual(baseline["batch_size"], 32)
        self.assertEqual(
            baseline["decode_tokens_per_second_per_user"],
            33.1,
        )
        self.assertEqual(report_baseline, baseline)
        self.assertTrue(baseline["baseline_file"].endswith(
            "performance_baselines.json"
        ))

    def test_validation_performance_summary_helpers_reexport_compatibly(self) -> None:
        self.assertIs(
            validation_module._throughput_baseline_summary,
            throughput_baseline_summary,
        )
        self.assertIs(
            validation_module._performance_gap_summary,
            performance_gap_summary,
        )
        report = {
            "baseline_tokens_per_second_per_user": 10.0,
            "baseline_reference": "sample",
            "min_baseline_ratio": 0.5,
            "baseline_reference_entry": {
                "id": "sample",
                "role": "official_8b_target",
            },
        }
        profile = {
            "throughput_summary": {
                "status": "measured",
                "tokens_per_second_per_user": 6.0,
            },
            "bottleneck_summary": {
                "max_section": "argmax",
                "max_section_ms": 3.0,
                "sections_ms": {"argmax": 3.0, "decode": 1.0},
            },
        }

        throughput = validation_module._throughput_baseline_summary(
            report,
            profile,
        )
        gap = validation_module._performance_gap_summary(report, profile)

        self.assertEqual(throughput["ratio"], 0.6)
        self.assertTrue(throughput["passed"])
        self.assertEqual(gap["shortfall_to_baseline"], 4.0)
        self.assertEqual(gap["bottleneck"]["max_section_share"], 0.75)

    def test_decode_runtime_inputs_accept_shared_prompt_rotary(self) -> None:
        step = {
            "input_source": "prompt_runtime",
            "input_shapes": {
                "token_ids": [32, 1],
                "page_table": [32, 32],
                "cache_position": [32],
                "key_cache": [1024, 8, 32, 128],
                "value_cache": [1024, 8, 32, 128],
            },
            "kv_cache": {
                "physical_shape": [1024, 8, 32, 128],
                "logical_shape": [32, 1024, 8, 128],
                "page_block_size": 32,
                "page_count": 32,
                "max_num_blocks": 1024,
            },
            "synthetic_runtime_input_tensor_count": 0,
            "prompt_runtime_input_tensor_count": 1,
            "decode_runtime_state_input_tensor_count": 2,
            "rotary_runtime_input_tensor_count": 3,
            "rotary_runtime_state": {
                "shared_across_layers": True,
                "tensor_count": 3,
            },
            "kv_cache_runtime_input_tensor_count": 64,
            "synthetic_rotary_tensor_count": 0,
        }

        self.assertTrue(
            validation_module._decode_runtime_inputs_complete(
                step,
                layer_count=32,
                batch_size=32,
                seq_len=1,
                cache_len=1024,
                num_kv_heads=8,
                head_dim=128,
                page_block_size=32,
            )
        )

    def test_validate_real_decode_official_performance_parity_requires_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "baseline_reference"):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    require_official_performance_parity=True,
                    min_baseline_ratio=0.1,
                )
            with self.assertRaisesRegex(ValueError, "min_baseline_ratio"):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    require_official_performance_parity=True,
                    baseline_reference="tt_metal_official_llama31_8b_b32",
                )
            with self.assertRaisesRegex(
                ValueError,
                "positive min_baseline_ratio",
            ):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    require_official_performance_parity=True,
                    baseline_reference="tt_metal_official_llama31_8b_b32",
                    min_baseline_ratio=0.0,
                )
            with self.assertRaisesRegex(ValueError, "skip_autotune"):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    require_official_performance_parity=True,
                    baseline_reference="tt_metal_official_llama31_8b_b32",
                    min_baseline_ratio=0.1,
                    skip_autotune=True,
                )
            with self.assertRaisesRegex(
                ValueError,
                "tokens_per_second_per_user",
            ):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    require_official_performance_parity=True,
                    baseline_reference="tt_metal_official_llama31_8b_b32",
                    min_baseline_ratio=0.1,
                    metric="latency_ms",
                )
            with self.assertRaisesRegex(
                ValueError,
                "min_tokens_per_second_per_user",
            ):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    min_tokens_per_second_per_user=-1.0,
                )
            with self.assertRaisesRegex(
                ValueError,
                "decode_shell_pcc_threshold",
            ):
                validate_real_decode(
                    program_dir=root / "missing_program",
                    model_path=root / "missing_model",
                    out_dir=root / "validate_real",
                    decode_shell_pcc_threshold=1.5,
                )

    def test_real_decode_scope_marks_official_performance_parity_ready(
        self,
    ) -> None:
        report = {
            "dry_run": False,
            "status": "pass",
            "require_full_decode_step": True,
            "require_model_end_to_end": True,
            "require_official_performance_parity": True,
            "require_full_depth": True,
            "require_program_runtime_shape": True,
            "require_batch32_decode_step": True,
            "require_trace": True,
            "require_decode_shell_numeric_reference": True,
            "require_official_config_match": True,
            "min_baseline_ratio": 0.1,
            "layers": 32,
            "program_num_layers": 32,
            "batch_size": 32,
            "program_batch_size": 32,
            "cache_len": 1024,
            "program_cache_len": 1024,
        }
        acceptance = {
            "passed": True,
            "checks": [
                {"name": "validation.full_depth_layers", "passed": True},
                {"name": "decode_depth_sweep.full_depth", "passed": True},
                {"name": "validation.program_batch_size", "passed": True},
                {"name": "validation.program_cache_len", "passed": True},
                {"name": "decode_step_contract.batch32", "passed": True},
                {"name": "single_layer_decode.trace_status", "passed": True},
                {"name": "smoke_decode_step.trace_status", "passed": True},
                {"name": "profile_decode_step.trace_status", "passed": True},
                {"name": "profile_decode_step.trace_profile", "passed": True},
                {
                    "name": "decode_shell.numeric_reference",
                    "passed": True,
                },
                {
                    "name": (
                        "official_config_diff.official_reference_format"
                    ),
                    "passed": True,
                },
                {"name": "official_config_diff.match", "passed": True},
                {
                    "name": "model_end_to_end_readiness.ready",
                    "passed": True,
                },
                {
                    "name": "profile_decode_step.min_baseline_ratio",
                    "passed": True,
                },
                {
                    "name": (
                        "profile_decode_step."
                        "official_min_baseline_ratio_positive"
                    ),
                    "passed": True,
                },
                {
                    "name": (
                        "profile_decode_step.official_baseline_reference"
                    ),
                    "passed": True,
                },
                {
                    "name": "decode_step_autotune.metric",
                    "passed": True,
                },
            ],
        }

        scope = validation_module._real_decode_acceptance_scope(
            report,
            acceptance,
        )

        self.assertEqual(scope["status"], "official_performance_parity")
        self.assertTrue(scope["require_official_performance_parity"])
        self.assertTrue(scope["full_decode_step_ready"])
        self.assertTrue(scope["model_end_to_end_proven"])
        self.assertTrue(
            scope["official_positive_baseline_ratio_floor_proven"]
        )
        self.assertTrue(scope["official_performance_parity_ready"])
        self.assertEqual(scope["missing_for_full_decode_step"], [])
        self.assertEqual(
            scope["missing_for_official_performance_parity"],
            [],
        )

    def test_real_decode_scope_requires_e2e_for_official_performance_parity(
        self,
    ) -> None:
        report = {
            "dry_run": False,
            "status": "pass",
            "require_full_decode_step": True,
            "require_model_end_to_end": True,
            "require_official_performance_parity": True,
            "require_full_depth": True,
            "require_program_runtime_shape": True,
            "require_batch32_decode_step": True,
            "require_trace": True,
            "require_decode_shell_numeric_reference": True,
            "require_official_config_match": True,
            "min_baseline_ratio": 0.1,
            "layers": 32,
            "program_num_layers": 32,
            "batch_size": 32,
            "program_batch_size": 32,
            "cache_len": 1024,
            "program_cache_len": 1024,
        }
        acceptance = {
            "passed": True,
            "checks": [
                {"name": "validation.full_depth_layers", "passed": True},
                {"name": "decode_depth_sweep.full_depth", "passed": True},
                {"name": "validation.program_batch_size", "passed": True},
                {"name": "validation.program_cache_len", "passed": True},
                {"name": "decode_step_contract.batch32", "passed": True},
                {"name": "single_layer_decode.trace_status", "passed": True},
                {"name": "smoke_decode_step.trace_status", "passed": True},
                {"name": "profile_decode_step.trace_status", "passed": True},
                {"name": "profile_decode_step.trace_profile", "passed": True},
                {
                    "name": "decode_shell.numeric_reference",
                    "passed": True,
                },
                {
                    "name": (
                        "official_config_diff.official_reference_format"
                    ),
                    "passed": True,
                },
                {"name": "official_config_diff.match", "passed": True},
                {
                    "name": "profile_decode_step.min_baseline_ratio",
                    "passed": True,
                },
            ],
        }

        scope = validation_module._real_decode_acceptance_scope(
            report,
            acceptance,
        )

        self.assertEqual(scope["status"], "full_decode_step")
        self.assertTrue(scope["full_decode_step_ready"])
        self.assertFalse(scope["model_end_to_end_proven"])
        self.assertFalse(scope["official_performance_parity_ready"])
        self.assertIn(
            "accepted model end-to-end readiness",
            scope["missing_for_official_performance_parity"],
        )

    def test_preflight_real_decode_passes_with_official_parity_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                max_new_tokens=3,
                prefill_len=9,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                min_tokens_per_second_per_user=1.0,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                decode_shell_pcc_threshold=0.99,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
                device_environment=_fake_tenstorrent_device_environment(),
            )

            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["ready_to_run"])
            self.assertTrue(report_json.is_file())
            self.assertEqual(
                report["device_preflight_diagnostics"]["status"],
                "ready",
            )
            self.assertTrue(
                report["device_preflight_diagnostics"]["ready"]
            )
            self.assertTrue(
                report["requirements"]["require_full_decode_step"]
            )
            self.assertTrue(
                report["requirements"]["require_model_end_to_end"]
            )
            self.assertTrue(
                report["requirements"][
                    "require_official_performance_parity"
                ]
            )
            self.assertTrue(
                report["requirements"]["require_official_config_match"]
            )
            self.assertEqual(report["official_config_diff"]["status"], "match")

    def test_preflight_real_decode_accepts_source_ttnn_without_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            fake_ttnn = _make_fake_ttnn()
            delattr(fake_ttnn, "__version__")
            fake_ttnn.__file__ = "/fake/ttnn/__init__.py"
            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                max_new_tokens=3,
                prefill_len=9,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                min_tokens_per_second_per_user=1.0,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                decode_shell_pcc_threshold=0.99,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=fake_ttnn,
                device_environment=_fake_tenstorrent_device_environment(),
            )

            self.assertEqual(report["status"], "pass")
            version_check = _check_by_name(report, "ttnn.version")
            self.assertTrue(version_check["passed"])
            self.assertIsNone(version_check["observed"]["version"])
            self.assertEqual(
                version_check["observed"]["module_file"],
                "/fake/ttnn/__init__.py",
            )
            device_check = _check_by_name(
                report,
                "tenstorrent.device_available",
            )
            self.assertTrue(device_check["passed"])
            self.assertEqual(
                report["official_config_diff"]["official_source_format"],
                "normalized_parity_config",
            )
            self.assertEqual(report["decode_step_contract"]["batch_size"], 32)
            self.assertEqual(
                report["decode_step_contract"]["decode_seq_len"],
                1,
            )
            self.assertEqual(report["max_new_tokens"], 3)
            self.assertEqual(report["prefill_len"], 9)
            self.assertEqual(
                report["baseline_reference_entry"]["model"],
                "Llama 3.1 8B",
            )
            self.assertEqual(report["min_tokens_per_second_per_user"], 1.0)
            self.assertEqual(report["decode_shell_pcc_threshold"], 0.99)
            self.assertEqual(report["metric"], "tokens_per_second_per_user")
            self.assertTrue(report["prompt_runtime_requested"])
            self.assertEqual(report["prompt_char_count"], len("hello tenstorrent"))
            self.assertEqual(report["tokenizer_path"], str(model_dir))
            self.assertEqual(report["effective_tokenizer_path"], str(model_dir))
            plan = report["final_acceptance_plan"]
            self.assertEqual(
                plan["target_scope"],
                "official_performance_parity",
            )
            self.assertEqual(plan["metric"], "tokens_per_second_per_user")
            self.assertIn(
                "decode_shell.numeric_reference",
                plan["full_decode_step_gate_names"],
            )
            self.assertIn(
                "model_end_to_end_readiness.ready",
                plan["model_end_to_end_gate_names"],
            )
            self.assertIn(
                "official_config_diff.match",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "official_config_diff.official_reference_format",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "profile_decode_step.official_baseline_reference",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                (
                    "profile_decode_step."
                    "official_min_baseline_ratio_positive"
                ),
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "decode_step_autotune.status",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "decode_step_autotune.metric",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "decode_step_autotune.best_candidate_summary",
                plan["official_performance_parity_gate_names"],
            )
            self.assertIn(
                "--require-official-performance-parity",
                plan["requested_acceptance_flags"],
            )
            self.assertIn(
                "--require-model-end-to-end",
                plan["requested_acceptance_flags"],
            )
            self.assertEqual(
                plan["thresholds"]["decode_shell_pcc_threshold"],
                0.99,
            )
            self.assertEqual(
                plan["baseline"]["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            repro = report["reproducibility"]
            self.assertIn(
                "--preflight-only",
                repro["preflight_cli_args"],
            )
            self.assertNotIn(
                "--preflight-only",
                repro["final_validation_cli_args"],
            )
            self.assertIn(
                "validate-real-decode",
                repro["final_validation_cli_command"],
            )
            self.assertIn(
                "--min-tokens-per-second-per-user",
                repro["final_validation_cli_args"],
            )
            self.assertIn("--metric", repro["final_validation_cli_args"])
            self.assertIn(
                "tokens_per_second_per_user",
                repro["final_validation_cli_args"],
            )
            self.assertIn("--max-new-tokens", repro["preflight_cli_args"])
            self.assertIn("3", repro["preflight_cli_args"])
            self.assertIn("--prefill-len", repro["preflight_cli_args"])
            self.assertIn("9", repro["preflight_cli_args"])
            self.assertIn("--max-new-tokens", repro["final_validation_cli_args"])
            self.assertIn("3", repro["final_validation_cli_args"])
            self.assertIn("--prefill-len", repro["final_validation_cli_args"])
            self.assertIn("9", repro["final_validation_cli_args"])
            self.assertIn("--prompt", repro["final_validation_cli_args"])
            self.assertIn(
                "hello tenstorrent",
                repro["final_validation_cli_args"],
            )
            self.assertIn(
                "--tokenizer-path",
                repro["final_validation_cli_args"],
            )
            self.assertIn(str(model_dir), repro["final_validation_cli_args"])
            self.assertIn("1.0", repro["final_validation_cli_args"])
            self.assertIn(
                "--decode-shell-pcc-threshold",
                repro["final_validation_cli_args"],
            )
            self.assertIn("0.99", repro["final_validation_cli_args"])
            self.assertEqual(
                report["ttnn_environment"]["tt_metal_git_commit"],
                "fake-tt-metal",
            )
            self.assertEqual(report["failed_checks"], [])

    def test_preflight_real_decode_fails_without_visible_device(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                min_tokens_per_second_per_user=1.0,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                decode_shell_pcc_threshold=0.99,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
                device_environment={
                    "device_available": False,
                    "device_nodes": [],
                    "filesystem_entries": [],
                    "driver_loaded": True,
                    "tt_smi_path": "/usr/bin/tt-smi",
                    "tt_smi": {
                        "status": "fail",
                        "returncode": 1,
                        "stdout": "No Tenstorrent devices detected",
                        "stderr": "",
                    },
                },
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "tenstorrent.device_available",
                report["failed_checks"],
            )
            device_check = _check_by_name(
                report,
                "tenstorrent.device_available",
            )
            self.assertFalse(device_check["passed"])
            self.assertEqual(
                report["tenstorrent_device_environment"]["device_nodes"],
                [],
            )
            diagnostics = report["device_preflight_diagnostics"]
            self.assertEqual(diagnostics["status"], "device_not_visible")
            self.assertFalse(diagnostics["ready"])
            self.assertIn(
                "python -m ttrt query",
                diagnostics["recommended_action"],
            )
            self.assertIn(
                "tenstorrent_device_not_visible",
                [finding["kind"] for finding in diagnostics["findings"]],
            )
            self.assertTrue(diagnostics["recommended_probe_commands"])

    def test_preflight_rejects_generated_config_as_official_reference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "real_decode_preflight_report.json"
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

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=program_dir / "config.json",
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "official_config.reference_format",
                report["failed_checks"],
            )
            self.assertEqual(report["official_config_diff"]["status"], "match")
            self.assertEqual(
                report["official_config_diff"]["official_source_format"],
                "generated_ttnn_direct_config",
            )

    def test_preflight_rejects_zero_official_performance_ratio(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.0,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "requirements.official_min_baseline_ratio_positive",
                report["failed_checks"],
            )

    def test_preflight_rejects_skipped_official_performance_autotune(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                skip_autotune=True,
                require_official_performance_parity=True,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "requirements.official_performance_autotune_enabled",
                report["failed_checks"],
            )

    def test_preflight_rejects_latency_metric_for_official_performance_parity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="latency_ms",
                require_official_performance_parity=True,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertEqual(report["metric"], "latency_ms")
            self.assertIn(
                "requirements.official_performance_metric",
                report["failed_checks"],
            )

    def test_preflight_rejects_non_8b_official_performance_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            report_json = root / "real_decode_preflight_report.json"
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
            _write_official_parity_from_program(program_dir, official_json)

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=official_json,
                layers=2,
                batch_size=32,
                cache_len=1024,
                trace_iterations=10,
                metric="tokens_per_second_per_user",
                require_official_performance_parity=True,
                baseline_reference="tt_metal_official_llama32_3b_b32",
                min_baseline_ratio=0.1,
                prompt="hello tenstorrent",
                tokenizer_path=model_dir,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "baseline_reference.official_performance_target",
                report["failed_checks"],
            )
            self.assertEqual(
                report["baseline_reference_entry"]["role"],
                "official_3b_target",
            )

    def test_preflight_real_decode_reports_missing_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "real_decode_preflight_report.json"
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

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=program_dir / "config.json",
                layers=1,
                batch_size=32,
                cache_len=1024,
                require_full_decode_step=True,
                baseline_reference="tt_metal_official_llama31_8b_b32",
                min_baseline_ratio=0.1,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertIn(
                "model_weights.safetensors_present",
                report["failed_checks"],
            )
            self.assertTrue(report_json.is_file())

    def test_preflight_real_decode_requires_prompt_for_model_end_to_end(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "real_decode_preflight_report.json"
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

            report = preflight_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out=report_json,
                official_config_path=program_dir / "config.json",
                layers=1,
                batch_size=32,
                cache_len=1024,
                require_model_end_to_end=True,
                ttnn_module=_make_fake_ttnn(),
            )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["ready_to_run"])
            self.assertFalse(report["prompt_runtime_requested"])
            self.assertIn("prompt_runtime.prompt", report["failed_checks"])
            self.assertIn(
                "--require-model-end-to-end",
                report["reproducibility"]["final_validation_cli_args"],
            )
            self.assertNotIn(
                "--prompt",
                report["reproducibility"]["final_validation_cli_args"],
            )

    def test_cli_validate_real_decode_preflight_records_final_thresholds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            official_json = root / "official_parity_config.json"
            out_dir = root / "validate_real"
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
            _write_official_parity_from_program(program_dir, official_json)

            with patch.object(
                validation_module.importlib,
                "import_module",
                return_value=_make_fake_ttnn(),
            ), patch.object(
                validation_module,
                "collect_tenstorrent_device_environment",
                return_value=_fake_tenstorrent_device_environment(),
            ):
                exit_code = main(
                    [
                        "validate-real-decode",
                        "--program-dir",
                        str(program_dir),
                        "--model-path",
                        str(model_dir),
                        "--out-dir",
                        str(out_dir),
                        "--official-config",
                        str(official_json),
                        "--layers",
                        "2",
                        "--batch-size",
                        "32",
                        "--cache-len",
                        "1024",
                        "--prompt",
                        "hello tenstorrent",
                        "--tokenizer-path",
                        str(model_dir),
                        "--require-official-performance-parity",
                        "--metric",
                        "tokens_per_second_per_user",
                        "--min-tokens-per-second-per-user",
                        "1.25",
                        "--baseline-reference",
                        "tt_metal_official_llama31_8b_b32",
                        "--min-baseline-ratio",
                        "0.1",
                        "--decode-shell-pcc-threshold",
                        "0.98",
                        "--preflight-only",
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(
                (out_dir / "real_decode_preflight_report.json").read_text()
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["metric"], "tokens_per_second_per_user")
            self.assertEqual(report["min_tokens_per_second_per_user"], 1.25)
            self.assertEqual(report["decode_shell_pcc_threshold"], 0.98)
            self.assertEqual(report["min_baseline_ratio"], 0.1)
            self.assertIn(
                "--preflight-only",
                report["reproducibility"]["preflight_cli_args"],
            )
            self.assertNotIn(
                "--preflight-only",
                report["reproducibility"]["final_validation_cli_args"],
            )
            self.assertIn(
                "1.25",
                report["reproducibility"]["final_validation_cli_args"],
            )
            self.assertIn(
                "0.98",
                report["reproducibility"]["final_validation_cli_args"],
            )
            self.assertEqual(
                report["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )

    def test_cli_validate_direct_runs_all_device_free_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "validate"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

            exit_code = main(
                [
                    "validate-direct",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["command"], "validate-direct")
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["decode_step_search_space_is_default"])
            self.assertEqual(
                report["results"],
                {step: "pass" for step in VALIDATION_STEPS},
            )
            self.assertEqual(
                list(report["results"]),
                list(VALIDATION_STEPS),
            )
            self.assertEqual(report["acceptance"]["status"], "passed")
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            acceptance_check_names = [
                check["name"] for check in report["acceptance"]["checks"]
            ]
            self.assertIn("validate_direct.steps", acceptance_check_names)
            self.assertIn("plan_diff.clean", acceptance_check_names)
            self.assertIn("py_compile.artifacts", acceptance_check_names)
            self.assertIn(
                "official_config_diff.official_required_fields",
                acceptance_check_names,
            )
            self.assertIn(
                "tensorize_parameters_dry_run.status",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_primitives_dry_run.status",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune_dry_run.knob_coverage",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune_dry_run.status_counts",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune_dry_run.default_knob_variation",
                acceptance_check_names,
            )
            self.assertIn(
                "validate_direct.artifacts",
                acceptance_check_names,
            )

            self.assertTrue((out_dir / "semantic_graph.json").is_file())
            self.assertTrue((out_dir / "execution_plan.json").is_file())
            self.assertTrue((out_dir / "plan_diff.json").is_file())
            self.assertTrue((out_dir / "parameter_config.json").is_file())
            self.assertTrue(
                (out_dir / "offline_artifacts" / "weights_manifest.json").is_file()
            )
            self.assertTrue((out_dir / "program" / "model.py").is_file())
            self.assertTrue((out_dir / "program" / "run_decode.py").is_file())
            self.assertTrue((out_dir / "official_config_diff.json").is_file())
            self.assertTrue((out_dir / "tensorize_report.json").is_file())
            self.assertTrue((out_dir / "decode_shell_report.json").is_file())
            for primitive in ATTENTION_PRIMITIVES:
                self.assertTrue(
                    (out_dir / "attention_primitives" / f"{primitive}.json").is_file()
                )
            self.assertTrue((out_dir / "attention_layer_report.json").is_file())
            self.assertTrue((out_dir / "single_layer_decode_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_smoke_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_profile_report.json").is_file())
            self.assertTrue((out_dir / "package" / "manifest.json").is_file())
            self.assertTrue((out_dir / "search_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_autotune_report.json").is_file())

            diff = json.loads((out_dir / "plan_diff.json").read_text())
            self.assertEqual(diff["missing_ops"], [])
            self.assertEqual(diff["extra_ops"], [])
            self.assertEqual(diff["order_mismatch"], [])

            config_diff = json.loads((out_dir / "official_config_diff.json").read_text())
            self.assertEqual(config_diff["status"], "diff_found")
            self.assertGreater(config_diff["summary"]["issue_count"], 0)
            self.assertEqual(
                config_diff["required_field_coverage"]["official"]["status"],
                "complete",
            )
            self.assertEqual(
                config_diff["required_field_coverage"]["official"][
                    "missing_required_paths"
                ],
                [],
            )
            self.assertEqual(
                report["steps"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["status"],
                "complete",
            )
            self.assertEqual(
                report["steps"]["official_config_diff"][
                    "required_parity_fields"
                ],
                list(REQUIRED_PARITY_PATHS),
            )

            tensorize_report = json.loads((out_dir / "tensorize_report.json").read_text())
            self.assertTrue(tensorize_report["dry_run"])
            self.assertEqual(
                tensorize_report["roles"],
                ["embedding", "norm", "attention", "mlp", "lm_head"],
            )
            self.assertEqual(tensorize_report["tensor_count"], 17)

            decode_shell_report = json.loads(
                (out_dir / "decode_shell_report.json").read_text()
            )
            self.assertEqual(decode_shell_report["status"], "dry_run")
            self.assertTrue(decode_shell_report["dry_run"])

            attention_layer_report = json.loads(
                (out_dir / "attention_layer_report.json").read_text()
            )
            self.assertEqual(attention_layer_report["status"], "dry_run")
            self.assertEqual(
                len(attention_layer_report["primitive_reports"]),
                8,
            )

            single_layer_report = json.loads(
                (out_dir / "single_layer_decode_report.json").read_text()
            )
            self.assertEqual(single_layer_report["status"], "dry_run")
            self.assertEqual(single_layer_report["layers"], 1)

            decode_step_report = json.loads(
                (out_dir / "decode_step_smoke_report.json").read_text()
            )
            self.assertEqual(decode_step_report["status"], "dry_run")
            self.assertEqual(decode_step_report["layers"], 2)
            self.assertEqual(decode_step_report["trace"]["status"], "dry_run")

            profile_report = json.loads(
                (out_dir / "decode_step_profile_report.json").read_text()
            )
            self.assertEqual(profile_report["status"], "dry_run")
            self.assertEqual(profile_report["layers"], 2)

            search_report = json.loads((out_dir / "search_report.json").read_text())
            self.assertTrue(search_report["dry_run"])
            self.assertEqual(search_report["candidate_count"], 10)

            decode_step_autotune = json.loads(
                (out_dir / "decode_step_autotune_report.json").read_text()
            )
            self.assertTrue(decode_step_autotune["dry_run"])
            self.assertEqual(decode_step_autotune["candidate_count"], 32)
            self.assertEqual(
                decode_step_autotune["knob_coverage"]["knobs"],
                list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
            self.assertEqual(
                decode_step_autotune["knob_coverage"]["candidate_count"],
                32,
            )
            self.assertEqual(
                decode_step_autotune["knob_coverage"]["values"][
                    "attention_sdpa_output_memory_config"
                ],
                [None, "l1"],
            )
            self.assertEqual(
                decode_step_autotune["knob_coverage"]["values"][
                    "attention_concat_heads_output_memory_config"
                ],
                [None, "l1"],
            )
            self.assertEqual(
                decode_step_autotune["knob_coverage"]["varied_knobs"],
                list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
            self.assertEqual(
                decode_step_autotune["knob_coverage"][
                    "missing_varied_knobs"
                ],
                [],
            )
            self.assertTrue(
                decode_step_autotune["knob_coverage"]["all_knobs_varied"]
            )
            self.assertEqual(
                decode_step_autotune["status_counts"],
                {"dry_run_planned": 32},
            )

            self.assertEqual(
                report["steps"]["py_compile"]["compiled"],
                [
                    str(out_dir / "program" / "model.py"),
                    str(out_dir / "program" / "run_decode.py"),
                ],
            )
            self.assertEqual(
                report["steps"]["package_program"]["package_dir"],
                str(out_dir / "package"),
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"]["candidate_count"],
                32,
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"][
                    "knob_coverage"
                ]["knobs"],
                list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
            self.assertTrue(
                report["steps"]["decode_step_autotune_dry_run"][
                    "default_search_space"
                ]
            )
            self.assertTrue(
                report["steps"]["decode_step_autotune_dry_run"][
                    "all_knobs_varied"
                ]
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"][
                    "missing_varied_knobs"
                ],
                [],
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"][
                    "metric_direction"
                ],
                "minimize",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"][
                    "status_counts"
                ],
                {"dry_run_planned": 32},
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune_dry_run"][
                    "reference_status_counts"
                ],
                {},
            )
            self.assertEqual(
                report["steps"]["decode_step_smoke_dry_run"][
                    "reference_status"
                ],
                "dry_run",
            )
            self.assertEqual(
                report["steps"]["decode_step_profile_dry_run"][
                    "reference_status"
                ],
                "dry_run",
            )

    def test_validate_direct_fails_acceptance_on_default_autotune_variation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "validate"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

            original_autotune = validation_module.run_decode_step_autotune

            def autotune_without_default_variation(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs["space"] = {
                    "lm_head_split_count": [2],
                    "generation_template": [
                        "device_argmax_greedy",
                        "full_logits",
                    ],
                    "mlp_intermediate_dtype": [None],
                    "attention_sdpa_output_memory_config": [None],
                    "attention_concat_heads_output_memory_config": [None],
                }
                autotune = original_autotune(*args, **kwargs)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(autotune, indent=2) + "\n")
                return autotune

            with patch.object(
                validation_module,
                "run_decode_step_autotune",
                side_effect=autotune_without_default_variation,
            ):
                report = validation_module.validate_direct(
                    model_path=model_dir,
                    config_path=config_json,
                    out_dir=out_dir,
                )

            self.assertTrue(report["decode_step_search_space_is_default"])
            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(
                report["results"],
                {step: "pass" for step in VALIDATION_STEPS},
            )
            self.assertEqual(report["acceptance"]["status"], "failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_autotune_dry_run.default_knob_variation"],
            )
            self.assertEqual(
                failed_checks[0]["observed"]["missing_varied_knobs"],
                [
                    "lm_head_split_count",
                    "mlp_intermediate_dtype",
                    "attention_sdpa_output_memory_config",
                    "attention_concat_heads_output_memory_config",
                ],
            )
            persisted = json.loads(
                (out_dir / "validation_report.json").read_text()
            )
            self.assertEqual(persisted["status"], "acceptance_failed")
            self.assertEqual(
                persisted["acceptance"]["failed_checks"],
                ["decode_step_autotune_dry_run.default_knob_variation"],
            )

    def test_cli_validate_real_decode_dry_run_writes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                    "validate-real-decode",
                    "--program-dir",
                    str(program_dir),
                    "--model-path",
                    str(model_dir),
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--skip-autotune",
                    "--require-trace",
                    "--min-tokens-per-second-per-user",
                    "1.0",
                    "--require-decode-shell-numeric-reference",
                    "--dry-run",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(
                (out_dir / "real_decode_validation_report.json").read_text()
            )
            self.assertEqual(report["command"], "validate-real-decode")
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["program_batch_size"], 32)
            self.assertEqual(report["program_cache_len"], 1024)
            self.assertEqual(report["requested_batch_size"], 2)
            self.assertEqual(report["requested_cache_len"], 16)
            self.assertEqual(report["batch_size"], 2)
            self.assertEqual(report["cache_len"], 16)
            self.assertEqual(
                report["decode_step_contract"]["token_input_shape"],
                [2, 1],
            )
            self.assertEqual(
                report["decode_step_contract"]["kv_cache_policy"],
                "paged",
            )
            self.assertEqual(
                report["decode_step_contract"]["page_table_shape"],
                [2, 1],
            )
            self.assertEqual(
                report["decode_step_contract"]["max_num_blocks"],
                2,
            )
            self.assertEqual(
                report["decode_step_contract"]["kv_cache_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                report["decode_step_contract"]["kv_cache_physical_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                report["decode_step_contract"]["kv_cache_logical_shape"],
                [2, 16, 2, 4],
            )
            self.assertEqual(
                report["decode_step_contract"]["output_kind"],
                "token",
            )
            self.assertEqual(
                list(report["results"]),
                list(REAL_DECODE_VALIDATION_STEPS),
            )
            self.assertEqual(report["results"]["materialize_parameters"], "dry_run")
            self.assertEqual(report["results"]["official_config_diff"], "pass")
            self.assertEqual(report["results"]["decode_shell"], "dry_run")
            self.assertEqual(report["results"]["attention_primitives"], "dry_run")
            self.assertEqual(report["results"]["attention_layer"], "dry_run")
            self.assertEqual(report["results"]["single_layer_decode"], "dry_run")
            self.assertEqual(report["results"]["smoke_decode_step"], "dry_run")
            self.assertEqual(report["results"]["profile_decode_step"], "dry_run")
            self.assertEqual(
                report["results"]["generate_prefill_decode"],
                "dry_run",
            )
            self.assertEqual(report["results"]["profile_generate"], "dry_run")
            self.assertEqual(report["results"]["generate_depth_sweep"], "dry_run")
            self.assertEqual(report["results"]["decode_depth_sweep"], "dry_run")
            self.assertEqual(report["results"]["decode_step_autotune"], "skipped")
            self.assertIn("official_config", report)
            self.assertEqual(
                report["steps"]["official_config_diff"]["diff_status"],
                "diff_found",
            )
            self.assertGreaterEqual(
                report["steps"]["official_config_diff"]["issue_count"],
                0,
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["sections"],
                sorted(PARITY_SECTIONS),
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["gap_summary"][
                    "status"
                ],
                "diff_found",
            )
            self.assertEqual(
                report["steps"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["status"],
                "complete",
            )
            self.assertEqual(
                report["steps"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["missing_required_paths"],
                [],
            )
            self.assertEqual(
                report["steps"]["official_config_diff"][
                    "required_parity_fields"
                ],
                list(REQUIRED_PARITY_PATHS),
            )
            self.assertIn(
                "memory_config",
                report["steps"]["official_config_diff"]["gap_summary"][
                    "sections_with_issues"
                ],
            )
            self.assertEqual(report["acceptance"]["status"], "dry_run")
            self.assertTrue(report["acceptance"]["passed"])
            self.assertFalse(
                report["acceptance"]["require_official_config_match"]
            )
            self.assertFalse(report["acceptance"]["require_full_depth"])
            self.assertFalse(
                report["acceptance"]["require_program_runtime_shape"]
            )
            self.assertFalse(
                report["acceptance"]["require_batch32_decode_step"]
            )
            self.assertTrue(report["acceptance"]["require_trace"])
            self.assertTrue(
                report["acceptance"][
                    "require_decode_shell_numeric_reference"
                ]
            )
            self.assertEqual(
                report["acceptance"]["min_tokens_per_second_per_user"],
                1.0,
            )
            self.assertIsNone(
                report["acceptance"]["baseline_tokens_per_second_per_user"]
            )
            self.assertIsNone(report["acceptance"]["min_baseline_ratio"])
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["reference_status"],
                "dry_run",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["reference_status"],
                "dry_run",
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["ttnn_environment"][
                    "module_available"
                ],
                False,
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["reference_status"],
                "dry_run",
            )
            self.assertEqual(
                report["steps"]["decode_depth_sweep"]["depths"],
                [1],
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"]["depths"],
                [1],
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"]["status_counts"],
                {"dry_run": 1},
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"][
                    "model_semantics_counts"
                ],
                {"prompt_conditioned_prefill_decode": 1},
            )
            self.assertTrue(
                report["steps"]["generate_depth_sweep"]["acceptance"][
                    "passed"
                ]
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"][
                    "failed_depth_diagnostics"
                ],
                [],
            )
            self.assertEqual(
                report["steps"]["decode_depth_sweep"]["status_counts"],
                {"dry_run": 1},
            )
            self.assertTrue(
                report["steps"]["decode_depth_sweep"]["acceptance"]["passed"]
            )
            self.assertTrue(
                (out_dir / "parameter_materialization_report.json").is_file()
            )
            self.assertTrue((out_dir / "decode_shell_report.json").is_file())
            self.assertTrue((out_dir / "attention_layer_report.json").is_file())
            self.assertTrue((out_dir / "single_layer_decode_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_smoke_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_profile_report.json").is_file())
            self.assertTrue((out_dir / "generate_profile_report.json").is_file())
            self.assertTrue(
                (out_dir / "profile_generate_underlying_generate_report.json").is_file()
            )
            self.assertTrue((out_dir / "generate_depth_sweep_report.json").is_file())
            self.assertTrue((out_dir / "generate_depth_reports").is_dir())
            self.assertTrue((out_dir / "decode_depth_sweep_report.json").is_file())
            self.assertTrue((out_dir / "decode_depth_profiles").is_dir())
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "dry_run")
            self.assertEqual(evidence["validation"]["status"], "dry_run")
            self.assertEqual(evidence["validation"]["program_batch_size"], 32)
            self.assertEqual(evidence["validation"]["program_cache_len"], 1024)
            self.assertEqual(evidence["validation"]["batch_size"], 2)
            self.assertEqual(evidence["validation"]["cache_len"], 16)
            self.assertTrue(evidence["requirements"]["require_trace"])
            self.assertIsNone(
                evidence["requirements"][
                    "baseline_tokens_per_second_per_user"
                ]
            )
            self.assertIsNone(evidence["requirements"]["min_baseline_ratio"])
            self.assertFalse(
                evidence["requirements"]["require_official_config_match"]
            )
            self.assertFalse(evidence["requirements"]["require_full_depth"])
            self.assertFalse(
                evidence["requirements"]["require_program_runtime_shape"]
            )
            self.assertFalse(
                evidence["requirements"]["require_batch32_decode_step"]
            )
            self.assertEqual(
                evidence["decode_step_contract"]["token_input_shape"],
                [2, 1],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_physical_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_logical_shape"],
                [2, 16, 2, 4],
            )
            self.assertEqual(evidence["acceptance"]["status"], "dry_run")
            self.assertEqual(evidence["acceptance_scope"]["status"], "dry_run")
            self.assertFalse(
                evidence["acceptance_scope"]["full_decode_step_ready"]
            )
            self.assertIn(
                "run validate-real-decode without --dry-run",
                evidence["acceptance_scope"]["missing_for_full_decode_step"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"]["depths"],
                [1],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"]["depths"],
                [1],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_generate"]["status"],
                "dry_run",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_generate"][
                    "model_semantics"
                ],
                "prompt_conditioned_prefill_decode",
            )
            self.assertFalse(
                evidence["performance_evidence"]["profile_generate"][
                    "official_performance_parity_claimed"
                ]
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"][
                    "model_semantics_counts"
                ],
                {"prompt_conditioned_prefill_decode": 1},
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"][
                    "failed_depth_diagnostics"
                ],
                [],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "status_counts"
                ],
                {"dry_run": 1},
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "diff_status"
                ],
                "diff_found",
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "gap_summary"
                ]["status"],
                "diff_found",
            )
            self.assertIsNone(
                evidence["performance_evidence"]["throughput_baseline"][
                    "baseline"
                ]
            )
            artifact_names = {
                artifact["name"]: artifact for artifact in evidence["artifacts"]
            }
            self.assertTrue(artifact_names["report"]["exists"])
            self.assertTrue(artifact_names["official_config_diff"]["exists"])
            self.assertTrue(artifact_names["attention_layer_report"]["exists"])
            self.assertTrue(artifact_names["single_layer_decode_report"]["exists"])
            self.assertTrue(artifact_names["smoke_report"]["exists"])
            self.assertTrue(
                artifact_names["generate_depth_sweep_report"]["exists"]
            )
            self.assertTrue(
                artifact_names["profile_generate_report"]["exists"]
            )
            self.assertTrue(
                artifact_names["profile_generate_underlying_report"]["exists"]
            )
            self.assertTrue(
                artifact_names["generate_depth_reports_dir"]["exists"]
            )
            self.assertTrue(
                artifact_names["decode_depth_sweep_report"]["exists"]
            )
            self.assertTrue(
                artifact_names["decode_depth_profiles_dir"]["exists"]
            )
            self.assertFalse(artifact_names["autotune_report"]["exists"])
            self.assertEqual(
                report["evidence"]["manifest"],
                str(out_dir / "real_decode_evidence_manifest.json"),
            )
            self.assertEqual(
                report["evidence"]["acceptance_scope"]["status"],
                "dry_run",
            )

    def test_validate_real_decode_runs_fake_real_weight_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            space_json = root / "decode_step_space.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [2],
                        "generation_template": [
                            "device_argmax_greedy",
                            "full_logits",
                        ],
                        "mlp_intermediate_dtype": [None],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    }
                )
            )
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    decode_step_search_space_path=space_json,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    trace=True,
                    trace_iterations=2,
                    require_trace=True,
                    min_tokens_per_second_per_user=0.0,
                    baseline_reference=(
                        "tt_metal_official_llama31_8b_b32"
                    ),
                    min_baseline_ratio=0.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["program_num_layers"], 2)
            self.assertEqual(report["program_seq_len"], 1)
            self.assertEqual(report["program_hidden_size"], 16)
            self.assertEqual(report["program_num_attention_heads"], 4)
            self.assertEqual(report["program_num_key_value_heads"], 2)
            self.assertEqual(report["program_head_dim"], 4)
            self.assertEqual(report["requested_batch_size"], 2)
            self.assertEqual(report["requested_cache_len"], 16)
            self.assertEqual(report["batch_size"], 2)
            self.assertEqual(report["cache_len"], 16)
            self.assertEqual(report["program_generation"]["mode"], "greedy")
            self.assertEqual(report["program_kv_cache"]["policy"], "paged")
            self.assertFalse(report["decode_step_search_space_is_default"])
            self.assertEqual(
                report["program_kv_cache"]["template"],
                "paged_kv_cache",
            )
            self.assertEqual(report["program_kv_cache"]["page_block_size"], 32)
            self.assertEqual(
                report["decode_step_contract"],
                {
                    "schema_version": 1,
                    "source": "generated_program_config",
                    "layers": 1,
                    "batch_size": 2,
                    "decode_seq_len": 1,
                    "cache_len": 16,
                    "token_input_shape": [2, 1],
                    "kv_cache_policy": "paged",
                    "kv_cache_template": "paged_kv_cache",
                    "uses_paged_kv_cache": True,
                    "kv_page_block_size": 32,
                    "page_count": 1,
                    "max_num_blocks": 2,
                    "page_table_shape": [2, 1],
                    "cache_position_shape": [2],
                    "kv_cache_shape": [2, 2, 32, 4],
                    "kv_cache_physical_shape": [2, 2, 32, 4],
                    "kv_cache_logical_shape": [2, 16, 2, 4],
                    "kv_cache_layer_ids": [0],
                    "generation_template": "device_argmax_greedy",
                    "output_kind": "token",
                    "accepted_output_kinds": ["token", "logits"],
                },
            )
            self.assertEqual(report["evidence"]["status"], "accepted")
            self.assertEqual(
                report["final_acceptance_plan"]["target_scope"],
                "bringup",
            )
            self.assertIn(
                "decode_step_autotune",
                report["final_acceptance_plan"]["required_runtime_steps"],
            )
            self.assertEqual(
                report["final_acceptance_plan"]["optional_gate_names"],
                [
                    "profile_decode_step.min_tokens_per_second_per_user",
                    "profile_decode_step.baseline_tokens_per_second_per_user",
                    "profile_decode_step.baseline_reference",
                    "profile_decode_step.min_baseline_ratio",
                ],
            )
            repro = report["reproducibility"]
            self.assertIn(
                "validate-real-decode",
                repro["validation_cli_command"],
            )
            self.assertIn("--preflight-only", repro["preflight_cli_args"])
            self.assertNotIn("--preflight-only", repro["validation_cli_args"])
            self.assertIn("--trace", repro["validation_cli_args"])
            self.assertIn(
                "--baseline-reference",
                repro["validation_cli_args"],
            )
            self.assertEqual(
                repro["artifact_index"]["report"],
                str(out_dir / "real_decode_validation_report.json"),
            )
            self.assertEqual(
                repro["artifact_index"]["evidence_manifest"],
                str(out_dir / "real_decode_evidence_manifest.json"),
            )
            expected_results = {
                step: "pass" for step in REAL_DECODE_VALIDATION_STEPS
            }
            expected_results["prompt_decode_loop"] = "skipped"
            expected_results["generate_prefill_decode"] = "skipped"
            expected_results["profile_generate"] = "skipped"
            expected_results["generate_depth_sweep"] = "skipped"
            self.assertEqual(report["results"], expected_results)
            self.assertEqual(
                report["steps"]["materialize_parameters"]["tensor_count"],
                21,
            )
            self.assertEqual(
                len(
                    report["steps"]["materialize_parameters"][
                        "required_tensor_paths"
                    ]
                ),
                21,
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"][
                    "missing_required_tensor_paths"
                ],
                [],
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"][
                    "materialized_tensor_shape_mismatches"
                ],
                [],
            )
            self.assertIn(
                "layers.0.attention.wqkv_packed.weight",
                report["steps"]["materialize_parameters"]["key_tensors"],
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"]["key_tensors"][
                    "lm_head.weight"
                ]["materialization"],
                "metadata_reference",
            )
            self.assertFalse(
                report["steps"]["materialize_parameters"]["key_tensors"][
                    "lm_head.weight"
                ]["materialized"]
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"]["key_tensors"][
                    "lm_head.splits.0.weight"
                ]["source_read"],
                "sliced_tensor",
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"][
                    "materialized_layer_ids"
                ],
                [0],
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["input_source"],
                "synthetic",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["runtime_input_tensor_count"],
                1,
            )
            self.assertEqual(report["steps"]["decode_shell"]["layers"], 1)
            self.assertEqual(
                report["steps"]["decode_shell"]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["numeric_reference_status"],
                "not_run",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["parameter_setup"][
                    "tensorization"
                ]["roles"],
                ["embedding", "norm", "mlp", "lm_head"],
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["parameter_setup"][
                    "tensorization"
                ]["transform_counts"],
                {
                    "reshape_embedding_weight_4d": 1,
                    "reshape_norm_weight_4d": 3,
                    "transpose_2d_to_4d": 11,
                },
            )
            self.assertEqual(
                report["steps"]["decode_shell"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            primitive_step = report["steps"]["attention_primitives"]
            self.assertEqual(primitive_step["status"], "pass")
            self.assertEqual(
                primitive_step["runtime_status_counts"],
                {"passed": len(ATTENTION_PRIMITIVES)},
            )
            self.assertEqual(
                primitive_step["primitive_sequence"],
                list(ATTENTION_PRIMITIVES),
            )
            self.assertEqual(
                len(primitive_step["primitive_reports"]),
                len(ATTENTION_PRIMITIVES),
            )
            self.assertTrue(
                all(
                    primitive["status"] == "passed"
                    and primitive["error"] is None
                    and primitive["reference"]["status"] == "passed"
                    and primitive["latency_ms"] >= 0.0
                    for primitive in primitive_step["primitive_reports"]
                )
            )
            self.assertEqual(
                primitive_step["primitive_reports"][0]["output_shapes"]["qkv"],
                [1, 1, 2, 32],
            )
            self.assertEqual(
                primitive_step["ttnn_environment"]["version"],
                "fake-ttnn",
            )
            self.assertEqual(
                primitive_step["ttnn_environment"]["tt_metal_git_commit"],
                "fake-tt-metal",
            )
            attention_step = report["steps"]["attention_layer"]
            self.assertEqual(attention_step["runtime_status"], "passed")
            self.assertEqual(attention_step["layer"], 0)
            self.assertEqual(attention_step["batch_size"], 2)
            self.assertEqual(attention_step["cache_len"], 16)
            self.assertEqual(attention_step["hidden_size"], 16)
            self.assertEqual(attention_step["num_kv_heads"], 2)
            self.assertEqual(attention_step["head_dim"], 4)
            self.assertEqual(attention_step["primitive_count"], 8)
            self.assertEqual(
                attention_step["primitive_sequence"],
                list(ATTENTION_LAYER_OPS),
            )
            self.assertEqual(
                len(attention_step["primitive_reports"]),
                len(ATTENTION_LAYER_OPS),
            )
            self.assertTrue(
                all(
                    primitive["error"] is None
                    and primitive["latency_ms"] >= 0.0
                    for primitive in attention_step["primitive_reports"]
                )
            )
            self.assertEqual(
                attention_step["output_shapes"]["attention_output"],
                [1, 1, 2, 16],
            )
            self.assertEqual(
                attention_step["output_shapes"]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                attention_step["output_shapes"]["value_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(attention_step["tensor_conversion_count"], 10)
            self.assertEqual(
                attention_step["memory_config_conversion_count"],
                1,
            )
            self.assertEqual(attention_step["reference_status"], "passed")
            self.assertEqual(
                attention_step["ttnn_environment"]["version"],
                "fake-ttnn",
            )
            self.assertEqual(
                attention_step["ttnn_environment"]["tt_metal_git_commit"],
                "fake-tt-metal",
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["input_source"],
                "synthetic",
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(report["steps"]["single_layer_decode"]["layers"], 1)
            self.assertEqual(
                report["steps"]["single_layer_decode"]["batch_size"],
                2,
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["cache_len"],
                16,
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["trace"]["iterations"],
                2,
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )
            self.assertEqual(
                report["steps"]["single_layer_decode"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertIn(
                "qkv_linear",
                report["steps"]["single_layer_decode"]["reference_observed_ops"],
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["input_source"],
                "synthetic",
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(report["steps"]["smoke_decode_step"]["layers"], 1)
            self.assertEqual(report["steps"]["smoke_decode_step"]["batch_size"], 2)
            self.assertEqual(report["steps"]["smoke_decode_step"]["cache_len"], 16)
            self.assertEqual(
                report["steps"]["profile_decode_step"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["input_source"],
                "synthetic",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(report["steps"]["profile_decode_step"]["layers"], 1)
            self.assertEqual(report["steps"]["profile_decode_step"]["batch_size"], 2)
            self.assertEqual(report["steps"]["profile_decode_step"]["cache_len"], 16)
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["trace"]["iterations"],
                2,
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["trace"]["iterations"],
                2,
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )
            self.assertEqual(
                len(
                    report["steps"]["profile_decode_step"]["trace"][
                        "execute_samples_ms"
                    ]
                ),
                2,
            )
            self.assertTrue(
                all(
                    sample > 0.0
                    for sample in report["steps"]["profile_decode_step"][
                        "trace"
                    ]["execute_samples_ms"]
                )
            )
            depth_sweep = report["steps"]["decode_depth_sweep"]
            self.assertEqual(depth_sweep["status"], "pass")
            self.assertEqual(depth_sweep["depths"], [1])
            self.assertEqual(depth_sweep["depth_count"], 1)
            self.assertEqual(depth_sweep["max_depth"], 1)
            self.assertFalse(depth_sweep["covered_full_depth"])
            self.assertFalse(depth_sweep["require_full_depth"])
            self.assertEqual(depth_sweep["status_counts"], {"profiled": 1})
            self.assertEqual(
                depth_sweep["reference_status_counts"],
                {"passed": 1},
            )
            self.assertEqual(
                depth_sweep["trace_status_counts"],
                {"captured_and_executed": 1},
            )
            self.assertEqual(depth_sweep["passed_depth_count"], 1)
            self.assertEqual(depth_sweep["failed_depths"], [])
            self.assertTrue(depth_sweep["acceptance"]["passed"])
            self.assertEqual(
                depth_sweep["records"][0]["layer_profile_ids"],
                [0],
            )
            self.assertEqual(report["acceptance"]["status"], "passed")
            self.assertTrue(report["acceptance"]["passed"])
            self.assertTrue(report["acceptance"]["require_trace"])
            self.assertTrue(
                all(check["passed"] for check in report["acceptance"]["checks"])
            )
            acceptance_check_names = [
                check["name"] for check in report["acceptance"]["checks"]
            ]
            self.assertIn(
                "official_config_diff.status",
                acceptance_check_names,
            )
            self.assertIn(
                "official_config_diff.diff_status",
                acceptance_check_names,
            )
            self.assertIn(
                "official_config_diff.issue_count",
                acceptance_check_names,
            )
            self.assertIn(
                "official_config_diff.sections",
                acceptance_check_names,
            )
            self.assertIn(
                "official_config_diff.gap_summary",
                acceptance_check_names,
            )
            self.assertIn(
                "official_config_diff.official_required_fields",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.tensor_count",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.layer_ids",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.lm_head_split_count",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.required_tensor_paths",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.tensor_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.lm_head_source_reference",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.decode_seq_len",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.token_input_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.paged_kv_cache",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.kv_page_block_size",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.max_num_blocks",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.page_table_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.cache_position_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.kv_cache_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.kv_cache_physical_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.kv_cache_logical_shape",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_contract.output_kind",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.layers",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.runtime_status",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.input_source",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.runtime_input_tensor_count",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.observed_op_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.reference_failed_checks",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.tensorized_physical_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.runtime_status",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.batch_size",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.cache_len",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.tensor_conversion_count",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.memory_config_conversion_count",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.ttnn_version",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.tt_metal_git_commit",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.primitive_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.primitive_reports",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.output_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.observed_op_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "attention_layer.reference_failed_checks",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.parameter_source",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.input_source",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.synthetic_runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.runtime_status",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.required_tensorized_tensor_paths",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.tensorized_physical_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.output_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.observed_op_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.reference_failed_checks",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.batch_size",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.input_source",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.synthetic_runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.cache_len",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.tensor_conversion_count",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.runtime_status",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.ttnn_version",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.tt_metal_git_commit",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.tensorization_roles",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.required_tensorized_tensor_paths",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.tensorized_physical_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.tensorization_memory_configs",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.output_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.observed_op_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.reference_failed_checks",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensor_conversion_count",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensor_conversion_ms",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.runtime_status",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.input_source",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.synthetic_runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.runtime_inputs",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.batch_size",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.cache_len",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.ttnn_version",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tt_metal_git_commit",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensorization_roles",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.required_tensorized_tensor_paths",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensorized_physical_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.section_latency_ms",
                acceptance_check_names,
            )

            self.assertIn(
                "profile_decode_step.layer_profile_count",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.layer_profile_sections",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.bottleneck_summary",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.output_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.lm_head_profile",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.observed_op_sequence",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.reference_failed_checks",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensorization_ttnn_memory_configs",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.lm_head_transform",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.embedding_norm_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.linear_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.lm_head_transform",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.embedding_norm_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.linear_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.lm_head_transform",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.embedding_norm_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.linear_weight_transforms",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.trace_iterations",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.trace_iterations",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.trace_execute_sample_count",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.trace_iterations",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.trace_execute_sample_count",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.trace_profile",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.throughput_status",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.latency_ms",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tokens_per_second_per_user",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.aggregate_tokens_per_second",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.min_tokens_per_second_per_user",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.baseline_tokens_per_second_per_user",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.baseline_reference",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.min_baseline_ratio",
                acceptance_check_names,
            )
            self.assertIn(
                (
                    "profile_decode_step."
                    "trace_execute_tokens_per_second_per_user"
                ),
                acceptance_check_names,
            )
            self.assertIn(
                "decode_depth_sweep.status",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_depth_sweep.acceptance",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_depth_sweep.requested_depth",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_depth_sweep.passed_depth_count",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_depth_sweep.records",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.status",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.candidate_count",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.knob_coverage",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.output_kind_counts",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.candidates",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.leaderboard",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best_candidate_summary",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.passed_candidate_count",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best_reference_status",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best_parameter_source",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best_metric",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_step_autotune.best_trace_status",
                acceptance_check_names,
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["ttnn_environment"][
                    "version"
                ],
                "fake-ttnn",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["ttnn_environment"][
                    "tt_metal_git_commit"
                ],
                "fake-tt-metal",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["candidate_count"],
                2,
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["metric_direction"],
                "minimize",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["status_counts"],
                {"profiled": 2},
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["knob_coverage"][
                    "knobs"
                ],
                list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["knob_coverage"][
                    "values"
                ]["generation_template"],
                ["device_argmax_greedy", "full_logits"],
            )
            self.assertFalse(
                report["steps"]["decode_step_autotune"][
                    "default_search_space"
                ]
            )
            self.assertFalse(
                report["steps"]["decode_step_autotune"][
                    "all_knobs_varied"
                ]
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "missing_varied_knobs"
                ],
                [
                    "lm_head_split_count",
                    "mlp_intermediate_dtype",
                    "attention_sdpa_output_memory_config",
                    "attention_concat_heads_output_memory_config",
                ],
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["output_kind_counts"],
                {"token": 1, "logits": 1},
            )
            self.assertEqual(
                len(
                    report["steps"]["decode_step_autotune"][
                        "candidate_summaries"
                    ]
                ),
                2,
            )
            self.assertTrue(
                all(
                    candidate["reference_status"] == "passed"
                    for candidate in report["steps"]["decode_step_autotune"][
                        "candidate_summaries"
                    ]
                )
            )
            self.assertEqual(
                {
                    candidate["output_kind"]
                    for candidate in report["steps"]["decode_step_autotune"][
                        "candidate_summaries"
                    ]
                },
                {"token", "logits"},
            )
            self.assertEqual(
                len(report["steps"]["decode_step_autotune"]["leaderboard"]),
                2,
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["leaderboard"][0][
                    "candidate_id"
                ],
                report["steps"]["decode_step_autotune"][
                    "best_candidate_summary"
                ]["candidate_id"],
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "best_candidate_summary"
                ]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "best_candidate_summary"
                ]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "passed_candidate_count"
                ],
                2,
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "failed_candidate_count"
                ],
                0,
            )
            self.assertIsNotNone(report["steps"]["decode_step_autotune"]["best"])
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["best_reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["best_trace_status"],
                "captured_and_executed",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["best_parameter_source"],
                "hf_model",
            )
            self.assertIsNotNone(
                report["steps"]["decode_step_autotune"]["best_metric"],
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["reference_status_counts"],
                {"passed": 2},
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["trace_status_counts"],
                {"captured_and_executed": 2},
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["diff_status"],
                "diff_found",
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["sections"],
                sorted(PARITY_SECTIONS),
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["gap_summary"][
                    "status"
                ],
                "diff_found",
            )
            self.assertIn(
                "paged_attention",
                report["steps"]["official_config_diff"]["gap_summary"][
                    "sections_with_issues"
                ],
            )
            attention_report = json.loads(
                (out_dir / "attention_layer_report.json").read_text()
            )
            self.assertEqual(attention_report["status"], "passed")
            self.assertEqual(
                attention_report["op_sequence"],
                list(ATTENTION_LAYER_OPS),
            )
            self.assertEqual(
                attention_report["output_shapes"]["attention_output"],
                [1, 1, 2, 16],
            )
            self.assertEqual(
                attention_report["output_shapes"]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                attention_report["tensor_conversion_count"],
                10,
            )
            self.assertEqual(
                attention_report["memory_config_conversion_count"],
                1,
            )
            self.assertEqual(
                [
                    primitive["primitive"]
                    for primitive in attention_report["primitive_reports"]
                ],
                list(ATTENTION_LAYER_OPS),
            )
            self.assertEqual(
                attention_report["primitive_reports"][0][
                    "expected_output_shapes"
                ]["qkv"],
                [1, 1, 2, 32],
            )
            self.assertEqual(
                attention_report["primitive_reports"][-1]["output_shapes"][
                    "attention_output"
                ],
                [1, 1, 2, 16],
            )
            smoke_report = json.loads(
                (out_dir / "decode_step_smoke_report.json").read_text()
            )
            self.assertEqual(smoke_report["parameter_source"], "hf_model")
            self.assertEqual(smoke_report["input_source"], "synthetic")
            self.assertEqual(
                smoke_report["parameter_setup"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(smoke_report["output_shapes"]["token"], [2, 1])
            self.assertEqual(
                smoke_report["output_shapes"]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in smoke_report["output_shapes"][
                        "kv_cache_layers"
                    ]
                ],
                [0],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"]["tensor_count"],
                17,
            )
            self.assertIn(
                "layers.0.attention.wqkv_packed.weight",
                smoke_report["parameter_setup"]["tensorization"][
                    "tensor_paths"
                ],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "ttnn_memory_config_counts"
                ],
                {"ttnn.DRAM_MEMORY_CONFIG": 17},
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "transform_counts"
                ],
                {
                    "reshape_embedding_weight_4d": 1,
                    "reshape_norm_weight_4d": 3,
                    "transpose_2d_to_4d": 13,
                },
            )
            self.assertIn(
                "embedding.weight",
                smoke_report["parameter_setup"]["tensorization"][
                    "transform_paths_by_kind"
                ]["reshape_embedding_weight_4d"],
            )
            self.assertIn(
                "layers.0.input_norm.weight",
                smoke_report["parameter_setup"]["tensorization"][
                    "transform_paths_by_kind"
                ]["reshape_norm_weight_4d"],
            )
            self.assertIn(
                "layers.0.mlp.gate_proj.weight",
                smoke_report["parameter_setup"]["tensorization"][
                    "transform_paths_by_kind"
                ]["transpose_2d_to_4d"],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "key_tensors"
                ]["lm_head.splits.0.weight"]["transform"],
                "transpose_2d_to_4d",
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "key_tensors"
                ]["lm_head.splits.0.weight"]["source_shape"],
                [16, 16],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"][
                    "key_tensors"
                ]["lm_head.splits.0.weight"]["shape"],
                [1, 1, 16, 16],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"]["key_tensors"][
                    "embedding.weight"
                ]["shape"],
                [1, 1, 128, 16],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"]["key_tensors"][
                    "layers.0.input_norm.weight"
                ]["shape"],
                [1, 1, 1, 16],
            )
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"]["key_tensors"][
                    "embedding.weight"
                ]["ttnn_memory_config"],
                "ttnn.DRAM_MEMORY_CONFIG",
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["parameter_setup"][
                    "tensorization"
                ]["memory_config_counts"],
                {"dram": 17},
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertIn(
                "qkv_linear",
                report["steps"]["smoke_decode_step"]["reference_observed_ops"],
            )
            self.assertIn(
                "mlp_gate",
                report["steps"]["decode_shell"]["reference_observed_ops"],
            )
            profile_report = json.loads(
                (out_dir / "decode_step_profile_report.json").read_text()
            )
            self.assertEqual(profile_report["parameter_source"], "hf_model")
            self.assertEqual(profile_report["input_source"], "synthetic")
            self.assertEqual(
                profile_report["parameter_setup"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(profile_report["output_shapes"]["token"], [2, 1])
            self.assertEqual(
                profile_report["output_shapes"]["value_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in profile_report["output_shapes"][
                        "kv_cache_layers"
                    ]
                ],
                [0],
            )
            self.assertEqual(profile_report["trace"]["status"], "captured_and_executed")
            self.assertGreater(
                profile_report["lm_head_profile"]["split_count"],
                0,
            )
            self.assertGreaterEqual(
                profile_report["lm_head_profile"]["lm_head_ms"],
                0.0,
            )
            self.assertGreaterEqual(
                profile_report["lm_head_profile"]["argmax_ms"],
                0.0,
            )
            self.assertEqual(
                profile_report["lm_head_profile"]["argmax_status"],
                "profiled",
            )
            self.assertEqual(
                profile_report["parameter_setup"]["tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            self.assertEqual(
                profile_report["parameter_setup"]["tensorization"][
                    "transform_counts"
                ],
                {
                    "reshape_embedding_weight_4d": 1,
                    "reshape_norm_weight_4d": 3,
                    "transpose_2d_to_4d": 13,
                },
            )
            self.assertIn(
                "final_norm.weight",
                profile_report["parameter_setup"]["tensorization"][
                    "transform_paths_by_kind"
                ]["reshape_norm_weight_4d"],
            )
            self.assertIn(
                "layers.0.attention.wqkv_packed.weight",
                profile_report["parameter_setup"]["tensorization"][
                    "transform_paths_by_kind"
                ]["transpose_2d_to_4d"],
            )
            self.assertIn(
                "embedding_ms",
                profile_report["section_latency_ms"],
            )
            self.assertEqual(
                [layer["layer_id"] for layer in profile_report["layer_profiles"]],
                [0],
            )
            self.assertIn(
                "per_layer_attention_ms",
                profile_report["bottleneck_summary"]["sections_ms"],
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            profile_step = report["steps"]["profile_decode_step"]
            self.assertEqual(
                [layer["layer_id"] for layer in profile_step["layer_profiles"]],
                [0],
            )
            self.assertIn(
                "mlp_gate",
                profile_step["reference_observed_ops"],
            )
            self.assertEqual(
                profile_report["throughput_summary"]["status"],
                "measured",
            )
            self.assertGreater(
                profile_report["throughput_summary"][
                    "tokens_per_second_per_user"
                ],
                0.0,
            )
            self.assertGreater(
                profile_report["throughput_summary"][
                    "trace_execute_mean_ms"
                ],
                0.0,
            )
            self.assertGreater(
                profile_report["throughput_summary"][
                    "trace_execute_aggregate_tokens_per_second"
                ],
                0.0,
            )
            depth_sweep_report = json.loads(
                (out_dir / "decode_depth_sweep_report.json").read_text()
            )
            self.assertEqual(depth_sweep_report["status"], "pass")
            self.assertEqual(depth_sweep_report["depths"], [1])
            self.assertFalse(depth_sweep_report["covered_full_depth"])
            self.assertFalse(depth_sweep_report["require_full_depth"])
            self.assertEqual(
                depth_sweep_report["status_counts"],
                {"profiled": 1},
            )
            self.assertEqual(
                depth_sweep_report["records"][0]["layer_profile_ids"],
                [0],
            )
            self.assertTrue(
                (out_dir / "decode_depth_profiles" / "profile_depth_1.json")
                .is_file()
            )
            autotune_report = json.loads(
                (out_dir / "decode_step_autotune_report.json").read_text()
            )
            self.assertEqual(autotune_report["best"]["reference_status"], "passed")
            self.assertEqual(autotune_report["status_counts"], {"profiled": 2})
            self.assertEqual(
                autotune_report["trace_status_counts"],
                {"captured_and_executed": 2},
            )
            self.assertEqual(
                autotune_report["output_kind_counts"],
                {"token": 1, "logits": 1},
            )
            self.assertEqual(
                autotune_report["candidates"][0]["reference_status"],
                "passed",
            )
            self.assertEqual(
                autotune_report["best"]["parameter_setup"]["tensorization"][
                    "tensor_count"
                ],
                11,
            )
            self.assertEqual(
                autotune_report["best"]["parameter_setup"]["tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 11},
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "accepted")
            self.assertEqual(
                evidence["final_acceptance_plan"],
                report["final_acceptance_plan"],
            )
            self.assertEqual(
                evidence["reproducibility"]["validation_cli_args"],
                report["reproducibility"]["validation_cli_args"],
            )
            self.assertEqual(
                evidence["reproducibility"]["artifact_index"][
                    "profile_report"
                ],
                str(out_dir / "decode_step_profile_report.json"),
            )
            self.assertEqual(
                evidence["reproducibility"]["artifact_index"][
                    "autotune_report"
                ],
                str(out_dir / "decode_step_autotune_report.json"),
            )
            self.assertEqual(evidence["acceptance_scope"]["status"], "bringup")
            matrix = evidence["acceptance_gate_matrix"]
            self.assertEqual(matrix["target_scope"], "bringup")
            self.assertEqual(matrix["planned_gate_count"], 4)
            self.assertEqual(matrix["failed_gates"], [])
            self.assertEqual(matrix["missing_gates"], [])
            self.assertIn(
                "profile_decode_step.min_baseline_ratio",
                matrix["passed_gates"],
            )
            e2e = evidence["model_end_to_end_readiness"]
            self.assertEqual(e2e["status"], "synthetic_runtime_inputs")
            self.assertFalse(e2e["model_end_to_end_ready"])
            self.assertTrue(e2e["uses_synthetic_runtime_inputs"])
            self.assertIn(
                "single_layer_decode",
                e2e["runtime_input_scope"]["synthetic_runtime_input_steps"],
            )
            self.assertIn(
                "real runtime input path for token ids, page table, cache "
                "position, KV cache, and rotary tensors",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertTrue(
                evidence["acceptance_scope"]["accepted_real_weight_runtime"]
            )
            self.assertFalse(
                evidence["acceptance_scope"]["full_decode_step_ready"]
            )
            self.assertIn(
                "--require-full-depth with layers == generated program layers",
                evidence["acceptance_scope"]["missing_for_full_decode_step"],
            )
            self.assertIn(
                "--require-batch32-decode-step",
                evidence["acceptance_scope"]["missing_for_full_decode_step"],
            )
            self.assertEqual(
                report["evidence"]["acceptance_scope"]["status"],
                "bringup",
            )
            self.assertEqual(evidence["validation"]["batch_size"], 2)
            self.assertEqual(evidence["validation"]["cache_len"], 16)
            self.assertEqual(evidence["validation"]["program_seq_len"], 1)
            self.assertEqual(evidence["validation"]["program_hidden_size"], 16)
            self.assertEqual(
                evidence["validation"]["program_num_key_value_heads"],
                2,
            )
            self.assertEqual(evidence["validation"]["program_head_dim"], 4)
            self.assertEqual(
                evidence["validation"]["program_kv_cache"]["policy"],
                "paged",
            )
            self.assertFalse(
                evidence["validation"]["decode_step_search_space_is_default"]
            )
            self.assertFalse(
                evidence["requirements"]["require_batch32_decode_step"]
            )
            self.assertEqual(
                evidence["decode_step_contract"]["token_input_shape"],
                [2, 1],
            )
            self.assertTrue(
                evidence["decode_step_contract"]["uses_paged_kv_cache"]
            )
            self.assertEqual(
                evidence["decode_step_contract"]["page_table_shape"],
                [2, 1],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["cache_position_shape"],
                [2],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["max_num_blocks"],
                2,
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_physical_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["kv_cache_logical_shape"],
                [2, 16, 2, 4],
            )
            self.assertEqual(
                evidence["decode_step_contract"]["output_kind"],
                "token",
            )
            self.assertTrue(evidence["acceptance"]["passed"])
            self.assertEqual(evidence["acceptance"]["failed_checks"], [])
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "diff_status"
                ],
                "diff_found",
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "sections"
                ],
                sorted(PARITY_SECTIONS),
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "gap_summary"
                ]["status"],
                "diff_found",
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["status"],
                "complete",
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["missing_required_paths"],
                [],
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "required_parity_fields"
                ],
                list(REQUIRED_PARITY_PATHS),
            )
            self.assertIn(
                "memory_config",
                evidence["config_evidence"]["official_config_diff"][
                    "gap_summary"
                ]["sections_with_issues"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"]["tensor_count"],
                21,
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"][
                    "missing_required_tensor_paths"
                ],
                [],
            )
            self.assertIn(
                "embedding.weight",
                evidence["weight_evidence"]["materialization"]["key_tensors"],
            )
            self.assertIn(
                "lm_head.splits.0.weight",
                evidence["weight_evidence"]["materialization"]["key_tensors"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"][
                    "key_tensors"
                ]["lm_head.weight"]["materialization"],
                "metadata_reference",
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"][
                    "key_tensors"
                ]["lm_head.splits.0.weight"]["source_read"],
                "sliced_tensor",
            )
            self.assertEqual(
                evidence["weight_evidence"]["decode_shell_tensorization"][
                    "transform_counts"
                ],
                {
                    "reshape_embedding_weight_4d": 1,
                    "reshape_norm_weight_4d": 3,
                    "transpose_2d_to_4d": 11,
                },
            )
            self.assertEqual(
                evidence["weight_evidence"]["decode_shell_tensorization"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_counts"
                ],
                {
                    "reshape_embedding_weight_4d": 1,
                    "reshape_norm_weight_4d": 3,
                    "transpose_2d_to_4d": 13,
                },
            )
            self.assertIn(
                "embedding.weight",
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_paths_by_kind"
                ]["reshape_embedding_weight_4d"],
            )
            self.assertIn(
                "layers.0.mlp.down_proj.weight",
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_paths_by_kind"
                ]["transpose_2d_to_4d"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "key_tensors"
                ]["lm_head.splits.0.weight"]["transform"],
                "transpose_2d_to_4d",
            )
            self.assertEqual(
                evidence["weight_evidence"]["single_layer_tensorization"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertEqual(
                evidence["weight_evidence"]["profile_tensorization"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "trace_status"
                ],
                "captured_and_executed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "output_shapes"
                ]["token"],
                [2, 1],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "output_shapes"
                ]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "output_shapes"
                ]["value_cache"],
                [2, 2, 32, 4],
            )
            self.assertGreater(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["split_count"],
                0,
            )
            self.assertGreaterEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["lm_head_ms"],
                0.0,
            )
            self.assertGreaterEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["argmax_ms"],
                0.0,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["argmax_status"],
                "profiled",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"]["status"],
                "pass",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"]["depths"],
                [1],
            )
            self.assertFalse(
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "covered_full_depth"
                ]
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "passed_depth_count"
                ],
                1,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "records"
                ][0]["layer_profile_ids"],
                [0],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_shell"]["input_source"],
                "synthetic",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_shell"][
                    "runtime_input_tensor_count"
                ],
                1,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "runtime_status"
                ],
                "passed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"]["layer"],
                0,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "primitive_sequence"
                ],
                list(ATTENTION_LAYER_OPS),
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "primitive_count"
                ],
                len(ATTENTION_LAYER_OPS),
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "output_shapes"
                ]["attention_output"],
                [1, 1, 2, 16],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "output_shapes"
                ]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_primitives"][
                    "runtime_status_counts"
                ],
                {"passed": len(ATTENTION_PRIMITIVES)},
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_primitives"][
                    "primitive_sequence"
                ],
                list(ATTENTION_PRIMITIVES),
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_primitives"][
                    "primitive_reports"
                ][0]["layout"],
                "tile",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "tensor_conversion_count"
                ],
                10,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "memory_config_conversion_count"
                ],
                1,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "primitive_reports"
                ][0]["layout"],
                "tile",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "input_source"
                ],
                "synthetic",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "synthetic_rotary_tensor_count"
                ],
                3,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "input_shapes"
                ]["token_ids"],
                [2, 1],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "input_shapes"
                ]["page_table"],
                [2, 1],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "kv_cache"
                ]["physical_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "input_source"
                ],
                "synthetic",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "synthetic_rotary_tensor_count"
                ],
                3,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "input_source"
                ],
                "synthetic",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "synthetic_rotary_tensor_count"
                ],
                3,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "input_shapes"
                ]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"][
                    "runtime_status"
                ],
                "passed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["single_layer_decode"]["layers"],
                1,
            )
            self.assertIn(
                "qkv_linear",
                evidence["runtime_evidence"]["single_layer_decode"][
                    "reference_observed_ops"
                ],
            )
            self.assertGreater(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "latency_ms"
                ],
                0.0,
            )
            self.assertIn(
                "embedding_ms",
                evidence["runtime_evidence"]["profile_decode_step"][
                    "section_latency_ms"
                ],
            )
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in evidence["runtime_evidence"][
                        "profile_decode_step"
                    ]["layer_profiles"]
                ],
                [0],
            )
            self.assertIn(
                "per_layer_mlp_ms",
                evidence["runtime_evidence"]["profile_decode_step"][
                    "bottleneck_summary"
                ]["sections_ms"],
            )
            self.assertIn(
                "qkv_linear",
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "reference_observed_ops"
                ],
            )
            self.assertIn(
                "mlp_gate",
                evidence["runtime_evidence"]["decode_shell"][
                    "reference_observed_ops"
                ],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "throughput_summary"
                ]["status"],
                "measured",
            )
            self.assertGreater(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "throughput_summary"
                ]["tokens_per_second_per_user"],
                0.0,
            )
            baseline = evidence["performance_evidence"]["throughput_baseline"]
            self.assertEqual(baseline["baseline"], 33.1)
            self.assertEqual(
                baseline["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            self.assertEqual(
                baseline["baseline_reference_entry"]["model"],
                "Llama 3.1 8B",
            )
            self.assertEqual(
                evidence["requirements"]["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            self.assertEqual(baseline["min_ratio"], 0.0)
            self.assertGreater(baseline["ratio"], 0.0)
            self.assertTrue(baseline["passed"])
            gap = evidence["performance_evidence"][
                "performance_gap_summary"
            ]
            self.assertEqual(gap["baseline"], 33.1)
            self.assertEqual(
                gap["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            self.assertGreater(gap["ratio"], 0.0)
            self.assertTrue(gap["passed_min_ratio"])
            self.assertGreaterEqual(gap["shortfall_to_baseline"], 0.0)
            self.assertGreater(gap["required_speedup_to_baseline"], 0.0)
            self.assertIn(
                gap["bottleneck"]["max_section"],
                gap["bottleneck"]["sections_ms"],
            )
            self.assertGreaterEqual(
                gap["bottleneck"]["max_section_share"],
                0.0,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"]["trace"][
                    "iterations"
                ],
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"]["trace"][
                    "iterations"
                ],
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )
            self.assertEqual(
                len(
                    evidence["runtime_evidence"]["profile_decode_step"][
                        "trace"
                    ]["execute_samples_ms"]
                ),
                2,
            )
            self.assertTrue(
                all(
                    sample > 0.0
                    for sample in evidence["runtime_evidence"][
                        "profile_decode_step"
                    ]["trace"]["execute_samples_ms"]
                )
            )
            self.assertGreater(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "throughput_summary"
                ]["trace_execute_mean_ms"],
                0.0,
            )
            self.assertGreater(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "throughput_summary"
                ]["trace_execute_aggregate_tokens_per_second"],
                0.0,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "batch_size"
                ],
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "cache_len"
                ],
                16,
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "attention_primitives_ttnn_environment"
                ]["version"],
                "fake-ttnn",
            )
            self.assertEqual(
                evidence["device_evidence"]["profile_ttnn_environment"][
                    "version"
                ],
                "fake-ttnn",
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "attention_layer_ttnn_environment"
                ]["version"],
                "fake-ttnn",
            )
            self.assertEqual(
                evidence["device_evidence"]["profile_ttnn_environment"][
                    "tt_metal_git_commit"
                ],
                "fake-tt-metal",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_reference_status"
                ],
                "passed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_trace_status"
                ],
                "captured_and_executed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_parameter_source"
                ],
                "hf_model",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "knob_coverage"
                ]["knobs"],
                list(DECODE_STEP_AUTOTUNE_KNOBS),
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "knob_coverage"
                ]["values"]["lm_head_split_count"],
                [2],
            )
            self.assertFalse(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "default_search_space"
                ]
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "missing_varied_knobs"
                ],
                [
                    "lm_head_split_count",
                    "mlp_intermediate_dtype",
                    "attention_sdpa_output_memory_config",
                    "attention_concat_heads_output_memory_config",
                ],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "output_kind_counts"
                ],
                {"token": 1, "logits": 1},
            )
            self.assertEqual(
                len(
                    evidence["runtime_evidence"]["decode_step_autotune"][
                        "candidate_summaries"
                    ]
                ),
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "candidate_summaries"
                ][0]["reference_status"],
                "passed",
            )
            self.assertEqual(
                len(
                    evidence["runtime_evidence"]["decode_step_autotune"][
                        "leaderboard"
                    ]
                ),
                2,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "leaderboard"
                ][0]["candidate_id"],
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_candidate_summary"
                ]["candidate_id"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_candidate_summary"
                ]["reference_status"],
                "passed",
            )

    def test_validate_real_decode_accepts_source_ttnn_without_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            fake_ttnn = _make_fake_ttnn()
            delattr(fake_ttnn, "__version__")
            fake_ttnn.__file__ = "/fake/ttnn/__init__.py"
            with _fake_torch_and_safetensors():
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    min_tokens_per_second_per_user=0.0,
                    ttnn_module=fake_ttnn,
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            version_checks = {
                check["name"]: check
                for check in report["acceptance"]["checks"]
                if check["name"].endswith(".ttnn_version")
            }
            self.assertEqual(
                sorted(version_checks),
                [
                    "attention_layer.ttnn_version",
                    "attention_primitives.ttnn_version",
                    "profile_decode_step.ttnn_version",
                    "single_layer_decode.ttnn_version",
                    "smoke_decode_step.ttnn_version",
                ],
            )
            for check in version_checks.values():
                self.assertTrue(check["passed"])
                self.assertEqual(check["observed"]["version"], None)
                self.assertEqual(
                    check["observed"]["module_file"],
                    "/fake/ttnn/__init__.py",
                )
                self.assertEqual(
                    check["expected"],
                    "non-empty version or importable source module path",
                )
            self.assertEqual(
                [
                    check["name"]
                    for check in report["acceptance"]["checks"]
                    if not check["passed"]
                ],
                [],
            )

    def test_validate_real_decode_uses_prompt_runtime_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    min_tokens_per_second_per_user=0.0,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    ttnn_module=_make_fake_ttnn(with_to_torch=True),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["prompt_runtime_requested"])
            self.assertEqual(report["tokenizer_path"], str(model_dir))
            repro = report["reproducibility"]
            self.assertIn("--prompt", repro["validation_cli_args"])
            self.assertIn("hello tenstorrent", repro["validation_cli_args"])
            self.assertIn("--tokenizer-path", repro["validation_cli_args"])
            self.assertIn(str(model_dir), repro["validation_cli_args"])
            self.assertIn("--prompt", repro["preflight_cli_args"])
            self.assertIn("hello tenstorrent", repro["preflight_cli_args"])
            self.assertIn("--tokenizer-path", repro["preflight_cli_args"])
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["status"],
                "skipped",
            )
            prompt_runtime_steps = [
                "decode_shell",
                "single_layer_decode",
                "smoke_decode_step",
                "profile_decode_step",
            ]
            for step_name in prompt_runtime_steps:
                step = report["steps"][step_name]
                self.assertEqual(step["input_source"], "prompt_runtime")
                self.assertEqual(step["prompt_runtime_input_tensor_count"], 1)
                self.assertEqual(
                    step["prompt_tokenization"]["selected_token_id"],
                    42,
                )
                self.assertEqual(
                    step["prompt_tokenization"]["token_input_shape"],
                    [2, 1],
                )

            prompt_loop = report["steps"]["prompt_decode_loop"]
            self.assertEqual(prompt_loop["status"], "pass")
            self.assertTrue(prompt_loop["decode_loop_runtime_owned"])
            self.assertEqual(prompt_loop["input_source"], "prompt_decode_loop")
            self.assertEqual(prompt_loop["runtime_owner"], "prompt_decode_loop")
            self.assertEqual(prompt_loop["decode_steps"], 2)
            self.assertEqual(prompt_loop["prompt_runtime_input_tensor_count"], 1)
            self.assertEqual(
                prompt_loop["decode_runtime_state_input_tensor_count"],
                4,
            )
            self.assertEqual(prompt_loop["rotary_runtime_input_tensor_count"], 6)
            self.assertEqual(prompt_loop["kv_cache_runtime_input_tensor_count"], 2)
            self.assertEqual(
                prompt_loop["synthetic_runtime_input_tensor_count"],
                0,
            )
            self.assertEqual(prompt_loop["synthetic_rotary_tensor_count"], 0)

            generate_step = report["steps"]["generate_prefill_decode"]
            self.assertEqual(generate_step["status"], "pass")
            self.assertEqual(generate_step["prefill_status"], "passed")
            self.assertTrue(generate_step["generate_runtime_owned"])
            self.assertTrue(generate_step["decode_loop_runtime_owned"])
            self.assertEqual(generate_step["kv_cache_source"], "prefill")
            self.assertEqual(
                generate_step["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertEqual(generate_step["input_source"], "prompt_prefill")
            self.assertEqual(generate_step["prompt_runtime_input_tensor_count"], 1)
            self.assertEqual(
                generate_step["synthetic_runtime_input_tensor_count"],
                0,
            )
            self.assertEqual(generate_step["synthetic_rotary_tensor_count"], 0)
            self.assertEqual(generate_step["synthetic_kv_cache_tensor_count"], 0)
            self.assertEqual(
                generate_step["generated_token_budget"][
                    "total_planned_generated_tokens"
                ],
                2,
            )
            self.assertEqual(
                generate_step["generated_text_status"],
                "fallback",
            )
            self.assertEqual(
                generate_step["runtime_context"]["class"],
                "TTNNDirectRuntimeContext",
            )
            self.assertEqual(
                generate_step["decode_token_runtime_handoff"],
                "device_tensor_direct",
            )
            self.assertFalse(
                generate_step["decode_token_host_roundtrip_per_step"]
            )
            self.assertTrue(
                generate_step[
                    "host_token_materialization_for_reporting_only"
                ]
            )
            self.assertFalse(
                generate_step["runtime_context"][
                    "kv_cache_reinitialized_per_step"
                ]
            )
            profile_generate = report["steps"]["profile_generate"]
            self.assertEqual(profile_generate["status"], "pass")
            self.assertTrue(profile_generate["generate_passed"])
            self.assertEqual(
                profile_generate["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertGreater(
                profile_generate["tokens_per_second_per_user"],
                0.0,
            )
            profile_milestones = profile_generate["performance_milestones"]
            milestone_by_id = {
                entry["id"]: entry
                for entry in profile_milestones["milestones"]
            }
            self.assertEqual(
                list(milestone_by_id),
                ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
            )
            self.assertTrue(milestone_by_id["M0"]["passed"])
            self.assertFalse(milestone_by_id["M2"]["passed"])
            self.assertEqual(
                milestone_by_id["M2"]["reason"],
                "requires_batch32_profile",
            )
            self.assertFalse(
                profile_generate["official_performance_parity_claimed"]
            )
            self.assertEqual(
                profile_generate["decode_token_runtime_handoff"],
                "device_tensor_direct",
            )
            self.assertFalse(
                profile_generate["decode_token_host_roundtrip_per_step"]
            )
            acceptance_check_names = {
                check["name"] for check in report["acceptance"]["checks"]
            }
            self.assertIn(
                "profile_generate.full_generated_model_can_run",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_generate.tokens_per_second_per_user_positive",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_generate.no_official_parity_claim",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_generate.profile_fields",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_generate.performance_milestones",
                acceptance_check_names,
            )

            self.assertEqual(
                report["steps"]["decode_shell"]["runtime_input_tensor_count"],
                0,
            )
            self.assertEqual(
                report["steps"]["decode_shell"][
                    "synthetic_runtime_input_tensor_count"
                ],
                0,
            )
            for step_name in (
                "single_layer_decode",
                "smoke_decode_step",
                "profile_decode_step",
            ):
                self.assertEqual(
                    report["steps"][step_name][
                        "synthetic_runtime_input_tensor_count"
                    ],
                    0,
                )
                self.assertEqual(
                    report["steps"][step_name][
                        "decode_runtime_state_input_tensor_count"
                    ],
                    2,
                )
                self.assertEqual(
                    report["steps"][step_name][
                        "synthetic_rotary_tensor_count"
                    ],
                    0,
                )
                self.assertEqual(
                    report["steps"][step_name][
                        "rotary_runtime_input_tensor_count"
                    ],
                    3,
                )
                self.assertEqual(
                    report["steps"][step_name][
                        "kv_cache_runtime_input_tensor_count"
                    ],
                    2,
                )

            depth_sweep = report["steps"]["decode_depth_sweep"]
            self.assertEqual(depth_sweep["records"][0]["input_source"], "prompt_runtime")
            self.assertEqual(
                depth_sweep["records"][0]["prompt_runtime_input_tensor_count"],
                1,
            )
            self.assertEqual(
                depth_sweep["records"][0][
                    "synthetic_runtime_input_tensor_count"
                ],
                0,
            )
            self.assertEqual(
                depth_sweep["records"][0][
                    "decode_runtime_state_input_tensor_count"
                ],
                2,
            )
            self.assertEqual(
                depth_sweep["records"][0]["decode_runtime_state"][
                    "cache_position_value"
                ],
                2,
            )
            self.assertEqual(
                depth_sweep["records"][0][
                    "rotary_runtime_input_tensor_count"
                ],
                3,
            )
            self.assertEqual(
                depth_sweep["records"][0]["rotary_runtime_state"][
                    "tensor_count"
                ],
                3,
            )
            self.assertEqual(
                depth_sweep["records"][0][
                    "kv_cache_runtime_input_tensor_count"
                ],
                2,
            )
            self.assertEqual(
                depth_sweep["records"][0]["kv_cache_runtime_state"][
                    "tensor_count"
                ],
                2,
            )

            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(
                evidence["reproducibility"]["validation_cli_args"],
                repro["validation_cli_args"],
            )
            self.assertIn(
                "--prompt",
                evidence["reproducibility"]["validation_cli_args"],
            )
            self.assertIn(
                "--tokenizer-path",
                evidence["reproducibility"]["validation_cli_args"],
            )
            e2e = evidence["model_end_to_end_readiness"]
            scope = e2e["runtime_input_scope"]
            self.assertEqual(evidence["status"], "accepted")
            self.assertEqual(e2e["status"], "needs_full_decode_step")
            self.assertFalse(e2e["model_end_to_end_ready"])
            self.assertFalse(e2e["uses_synthetic_runtime_inputs"])
            self.assertTrue(e2e["decode_loop_runtime_owned"])
            self.assertTrue(e2e["generate_prefill_decode_ready"])
            self.assertEqual(
                scope["runtime_input_sources"],
                {
                    **{step: "prompt_runtime" for step in prompt_runtime_steps},
                    "prompt_decode_loop": "prompt_decode_loop",
                    "generate_prefill_decode": "prompt_prefill",
                    "profile_generate": "prompt_prefill",
                },
            )
            self.assertEqual(
                scope["prompt_runtime_input_tensor_counts"],
                {
                    **{step: 1 for step in prompt_runtime_steps},
                    "prompt_decode_loop": 1,
                    "generate_prefill_decode": 1,
                    "profile_generate": 1,
                },
            )
            self.assertEqual(
                scope["decode_runtime_state_input_tensor_counts"],
                {
                    "single_layer_decode": 2,
                    "smoke_decode_step": 2,
                    "profile_decode_step": 2,
                    "prompt_decode_loop": 4,
                    "generate_prefill_decode": 2,
                    "profile_generate": 2,
                },
            )
            self.assertEqual(
                scope["rotary_runtime_input_tensor_counts"],
                {
                    "single_layer_decode": 3,
                    "smoke_decode_step": 3,
                    "profile_decode_step": 3,
                    "prompt_decode_loop": 6,
                    "generate_prefill_decode": 6,
                    "profile_generate": 6,
                },
            )
            self.assertEqual(
                scope["kv_cache_runtime_input_tensor_counts"],
                {
                    "single_layer_decode": 2,
                    "smoke_decode_step": 2,
                    "profile_decode_step": 2,
                    "prompt_decode_loop": 2,
                    "generate_prefill_decode": 2,
                    "profile_generate": 2,
                },
            )
            self.assertEqual(
                scope["synthetic_runtime_input_tensor_counts"],
                {
                    "decode_shell": 0,
                    "single_layer_decode": 0,
                    "smoke_decode_step": 0,
                    "profile_decode_step": 0,
                    "prompt_decode_loop": 0,
                    "generate_prefill_decode": 0,
                    "profile_generate": 0,
                },
            )
            self.assertEqual(scope["synthetic_runtime_input_steps"], [])
            self.assertEqual(scope["depth_sweep_synthetic_record_count"], 0)
            evidence_profile_milestones = evidence["performance_evidence"][
                "profile_generate"
            ]["performance_milestones"]
            self.assertEqual(
                [
                    entry["id"]
                    for entry in evidence_profile_milestones["milestones"]
                ],
                ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
            )
            generate_milestones = evidence["performance_evidence"][
                "generate_milestones"
            ]
            self.assertEqual(generate_milestones["highest_passed"], "M0")
            self.assertEqual(
                generate_milestones["observed"]["generate_depth_sweep"][
                    "covered_full_depth"
                ],
                False,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_prefill_decode"][
                    "decode_token_runtime_handoff"
                ],
                "device_tensor_direct",
            )
            self.assertFalse(
                evidence["runtime_evidence"]["profile_generate"][
                    "decode_token_host_roundtrip_per_step"
                ]
            )
            self.assertNotIn(
                "decode loop that owns prompt token ids, page table, cache "
                "position, rotary tensors, and KV cache beyond "
                "smoke/profile harnesses",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertIn(
                "accepted full decode-step evidence",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertNotIn(
                "real runtime input path for KV cache tensors",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertNotIn(
                "real runtime input path for KV cache and rotary tensors",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertNotIn(
                "real runtime input path for page table, cache position, "
                "KV cache, and rotary tensors",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertNotIn(
                "real runtime input path for token ids, page table, cache "
                "position, KV cache, and rotary tensors",
                e2e["missing_for_model_end_to_end"],
            )

    def test_validate_real_decode_rejects_generate_profile_parity_claim(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            def fake_profile_generate(*args, **kwargs):
                out = Path(kwargs["out"])
                generate_report = Path(kwargs["generate_report"])
                generate_report.write_text("{}\n")
                profile = {
                    "schema_version": 1,
                    "command": "profile-generate",
                    "mode": "profile-generate",
                    "status": "profiled",
                    "passed": True,
                    "generate_status": "passed",
                    "generate_passed": True,
                    "prefill_status": "passed",
                    "kv_cache_source": "prefill",
                    "model_semantics": "prompt_conditioned_prefill_decode",
                    "parameter_source": "hf_model",
                    "input_source": "prompt_prefill",
                    "runtime_owner": "TTNNDirectRuntimeContext",
                    "generate_runtime_owned": True,
                    "decode_loop_runtime_owned": True,
                    "runtime_context": {
                        "class": "TTNNDirectRuntimeContext",
                        "status": "built",
                    },
                    "parameter_setup": {},
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "prefill_prompt_runtime_input_tensor_count": 1,
                    "prefill_rotary_runtime_input_tensor_count": 3,
                    "decode_runtime_state_input_tensor_count": 2,
                    "decode_rotary_runtime_input_tensor_count": 3,
                    "kv_cache_runtime_input_tensor_count": 2,
                    "generated_text_status": "fallback",
                    "generated_token_count_by_user": [2, 2],
                    "latency_ms": 2.0,
                    "prefill_ms": 1.0,
                    "decode_step_ms_mean": 1.0,
                    "decode_step_ms_samples": [1.0],
                    "host_copy_ms": 0.1,
                    "host_copy_profile": {
                        "status": "measured",
                        "total_ms": 0.1,
                    },
                    "section_profile": {"status": "measured"},
                    "sections": {
                        "prefill_ms": 1.0,
                        "decode_total_ms": 1.0,
                        "decode_step_ms_mean": 1.0,
                        "embedding_ms": {"status": "measured"},
                        "prefill_attention_ms": {"status": "measured"},
                        "decode_attention_ms": {"status": "measured"},
                        "mlp_ms": {"status": "measured"},
                        "lm_head_ms": {"status": "measured"},
                        "argmax_ms": {"status": "measured"},
                        "host_copy_ms": {"status": "measured"},
                    },
                    "per_layer": {"status": "measured"},
                    "tokens_per_second_per_user": 1.0,
                    "aggregate_tokens_per_second": 2.0,
                    "throughput_summary": {
                        "status": "measured",
                        "tokens_per_second_per_user": 1.0,
                        "aggregate_tokens_per_second": 2.0,
                    },
                    "performance_milestones": {
                        "schema_version": 1,
                        "basis": "PR-7 generate performance milestone ladder",
                        "dry_run": False,
                        "official_reference": {
                            "id": "tt_metal_official_llama31_8b_b32",
                            "tokens_per_second_per_user": 33.1,
                            "batch_size": 32,
                        },
                        "observed": {
                            "layers": 1,
                            "program_num_layers": 2,
                            "batch_size": 2,
                            "tokens_per_second_per_user": 1.0,
                        },
                        "milestones": [
                            {
                                "id": milestone_id,
                                "name": milestone_id,
                                "passed": milestone_id == "M0",
                            }
                            for milestone_id in (
                                "M0",
                                "M1",
                                "M2",
                                "M3",
                                "M4",
                                "M5",
                                "M6",
                            )
                        ],
                        "highest_passed": "M0",
                        "next_milestone": {"id": "M1", "name": "M1"},
                    },
                    "acceptance": {
                        "passed": True,
                        "failed_checks": [],
                    },
                    "official_performance_parity_claimed": True,
                    "ttnn_environment": {"module_available": True},
                }
                out.write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "run_profile_generate",
                side_effect=fake_profile_generate,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        prompt="hello tenstorrent",
                        tokenizer_path=model_dir,
                        tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(with_to_torch=True),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check["name"]
                for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                failed_checks,
                ["profile_generate.no_official_parity_claim"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_generate.no_official_parity_claim"],
            )

    def test_validate_real_decode_can_require_official_config_match(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    require_official_config_match=True,
                    min_tokens_per_second_per_user=0.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(
                report["steps"]["official_config_diff"]["diff_status"],
                "diff_found",
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["official_config_diff.match"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(
                evidence["requirements"]["require_official_config_match"]
            )
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["official_config_diff.match"],
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "diff_status"
                ],
                "diff_found",
            )

    def test_validate_real_decode_fails_on_incomplete_official_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            official_json = root / "incomplete_official_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            official = json.loads(default_official_config_path().read_text())
            official["parity_config"]["compute_fidelity"].pop("mlp")
            official_json.write_text(json.dumps(official))
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    official_config_path=official_json,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["official_config_diff.official_required_fields"],
            )
            self.assertEqual(
                failed_checks[0]["observed"]["status"],
                "incomplete",
            )
            self.assertIn(
                "compute_fidelity.mlp",
                failed_checks[0]["observed"]["missing_required_paths"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["official_config_diff.official_required_fields"],
            )
            self.assertEqual(
                evidence["config_evidence"]["official_config_diff"][
                    "official_required_field_coverage"
                ]["status"],
                "incomplete",
            )

    def test_validate_real_decode_scope_marks_full_decode_step_ready(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            e2e_out_dir = root / "validate_real_e2e"
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

            original_shell = validation_module.run_smoke_decode_shell

            def shell_with_numeric_reference(*args, **kwargs):
                shell = original_shell(*args, **kwargs)
                numeric = dict(
                    shell.get("reference", {}).get("numeric_reference") or {}
                )
                numeric.update(
                    {
                        "status": "passed",
                        "passed": True,
                        "kind": "torch_decode_shell",
                        "pcc": 1.0,
                        "pcc_threshold": kwargs.get("pcc_threshold", 0.99),
                        "checks": [],
                    }
                )
                reference = dict(shell.get("reference") or {})
                reference["numeric_reference"] = numeric
                shell["reference"] = reference
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(shell, indent=2) + "\n")
                return shell

            with patch.object(
                validation_module,
                "run_smoke_decode_shell",
                side_effect=shell_with_numeric_reference,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=2,
                        batch_size=32,
                        cache_len=1024,
                        device="p150a",
                        skip_autotune=True,
                        require_full_decode_step=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )
                    e2e_report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=e2e_out_dir,
                        layers=2,
                        batch_size=32,
                        cache_len=1024,
                        device="p150a",
                        skip_autotune=True,
                        require_model_end_to_end=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["require_full_decode_step"])
            self.assertTrue(report["require_full_depth"])
            self.assertTrue(report["require_program_runtime_shape"])
            self.assertTrue(report["require_batch32_decode_step"])
            self.assertTrue(report["trace_enabled"])
            self.assertTrue(report["require_trace"])
            self.assertTrue(
                report["require_decode_shell_numeric_reference"]
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            scope = evidence["acceptance_scope"]
            matrix = evidence["acceptance_gate_matrix"]
            e2e = evidence["model_end_to_end_readiness"]
            self.assertEqual(evidence["status"], "accepted")
            self.assertEqual(scope["status"], "full_decode_step")
            self.assertEqual(matrix["target_scope"], "full_decode_step")
            self.assertEqual(matrix["failed_gates"], [])
            self.assertEqual(matrix["missing_gates"], [])
            self.assertIn(
                "decode_step_contract.batch32",
                matrix["passed_gates"],
            )
            self.assertTrue(scope["require_full_decode_step"])
            self.assertTrue(scope["accepted_real_weight_runtime"])
            self.assertTrue(scope["full_depth_proven"])
            self.assertTrue(scope["program_runtime_shape_proven"])
            self.assertTrue(scope["batch32_decode_contract_proven"])
            self.assertTrue(scope["trace_proven"])
            self.assertTrue(scope["decode_shell_numeric_reference_proven"])
            self.assertTrue(scope["full_decode_step_ready"])
            self.assertEqual(e2e["status"], "synthetic_runtime_inputs")
            self.assertTrue(e2e["full_decode_step_ready"])
            self.assertFalse(e2e["model_end_to_end_ready"])
            self.assertIn(
                "tokenizer/prompt runner that owns the decode loop instead "
                "of smoke-generated inputs",
                e2e["missing_for_model_end_to_end"],
            )
            self.assertFalse(scope["official_performance_parity_ready"])
            self.assertEqual(scope["missing_for_full_decode_step"], [])
            self.assertEqual(
                scope["missing_for_official_performance_parity"],
                [
                    "accepted model end-to-end readiness",
                    "--require-official-config-match",
                    "--baseline-reference plus --min-baseline-ratio",
                    "--metric tokens_per_second_per_user",
                ],
            )
            self.assertEqual(
                report["evidence"]["acceptance_scope"]["status"],
                "full_decode_step",
            )
            self.assertEqual(e2e_report["status"], "acceptance_failed")
            self.assertTrue(e2e_report["require_full_decode_step"])
            self.assertTrue(e2e_report["require_model_end_to_end"])
            self.assertEqual(
                [
                    check["name"]
                    for check in e2e_report["acceptance"]["checks"]
                    if not check["passed"]
                ],
                ["model_end_to_end_readiness.ready"],
            )
            e2e_evidence = json.loads(
                (e2e_out_dir / "real_decode_evidence_manifest.json")
                .read_text()
            )
            self.assertEqual(e2e_evidence["status"], "incomplete")
            self.assertEqual(
                e2e_evidence["acceptance_gate_matrix"]["target_scope"],
                "model_end_to_end",
            )
            self.assertEqual(
                e2e_evidence["acceptance_gate_matrix"]["failed_gates"],
                ["model_end_to_end_readiness.ready"],
            )
            self.assertFalse(
                e2e_evidence["model_end_to_end_readiness"][
                    "model_end_to_end_ready"
                ]
            )

    def test_validate_real_decode_can_require_full_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    require_full_depth=True,
                    min_tokens_per_second_per_user=0.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["program_num_layers"], 2)
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["validation.full_depth_layers"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertTrue(evidence["requirements"]["require_full_depth"])
            self.assertEqual(evidence["validation"]["program_num_layers"], 2)
            self.assertEqual(evidence["validation"]["layers"], 1)
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["validation.full_depth_layers"],
            )

    def test_validate_real_decode_can_require_program_runtime_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    require_program_runtime_shape=True,
                    min_tokens_per_second_per_user=0.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["program_batch_size"], 32)
            self.assertEqual(report["program_cache_len"], 1024)
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                [
                    "validation.program_batch_size",
                    "validation.program_cache_len",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertTrue(
                evidence["requirements"]["require_program_runtime_shape"]
            )
            self.assertEqual(evidence["validation"]["program_batch_size"], 32)
            self.assertEqual(evidence["validation"]["program_cache_len"], 1024)
            self.assertEqual(evidence["validation"]["batch_size"], 2)
            self.assertEqual(evidence["validation"]["cache_len"], 16)
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                [
                    "validation.program_batch_size",
                    "validation.program_cache_len",
                ],
            )

    def test_validate_real_decode_batch32_gate_rejects_runtime_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    require_batch32_decode_step=True,
                    min_tokens_per_second_per_user=0.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertTrue(report["acceptance"]["require_batch32_decode_step"])
            self.assertEqual(report["decode_step_contract"]["batch_size"], 2)
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_contract.batch32"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(
                evidence["requirements"]["require_batch32_decode_step"]
            )
            self.assertEqual(evidence["decode_step_contract"]["batch_size"], 2)
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_step_contract.batch32"],
            )

    def test_validate_real_decode_fails_on_depth_sweep_record_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_sweep = validation_module.run_decode_depth_sweep

            def sweep_with_bad_record_reference(*args, **kwargs):
                sweep = original_sweep(*args, **kwargs)
                sweep["records"][0]["reference_status"] = "failed"
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(sweep, indent=2) + "\n")
                return sweep

            with patch.object(
                validation_module,
                "run_decode_depth_sweep",
                side_effect=sweep_with_bad_record_reference,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_depth_sweep"], "pass")
            self.assertTrue(
                report["steps"]["decode_depth_sweep"]["acceptance"][
                    "passed"
                ]
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_depth_sweep.records"],
            )
            self.assertEqual(
                failed_checks[0]["observed"][0]["reference_status"],
                "failed",
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_depth_sweep.records"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "records"
                ][0]["reference_status"],
                "failed",
            )

    def test_validate_real_decode_fails_on_depth_sweep_profile_breakdown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_sweep = validation_module.run_decode_depth_sweep

            def sweep_without_lm_head_profile(*args, **kwargs):
                sweep = original_sweep(*args, **kwargs)
                sweep["records"][0].pop("lm_head_profile", None)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(sweep, indent=2) + "\n")
                return sweep

            with patch.object(
                validation_module,
                "run_decode_depth_sweep",
                side_effect=sweep_without_lm_head_profile,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_depth_sweep.records"],
            )
            self.assertEqual(
                failed_checks[0]["observed"][0]["lm_head_profile"],
                {},
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_depth_sweep.records"],
            )
            self.assertNotIn(
                "lm_head_profile",
                evidence["runtime_evidence"]["decode_depth_sweep"][
                    "records"
                ][0],
            )

    def test_validate_real_decode_fails_on_autotune_best_reference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            space_json = root / "decode_step_space.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [2],
                        "generation_template": ["device_argmax_greedy"],
                        "mlp_intermediate_dtype": [None],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    }
                )
            )
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

            original_autotune = validation_module.run_decode_step_autotune

            def autotune_with_bad_best_reference(*args, **kwargs):
                autotune = original_autotune(*args, **kwargs)
                autotune["best"]["reference_status"] = "failed"
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(autotune, indent=2) + "\n")
                return autotune

            with patch.object(
                validation_module,
                "run_decode_step_autotune",
                side_effect=autotune_with_bad_best_reference,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        decode_step_search_space_path=space_json,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_step_autotune"], "pass")
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "best_reference_status"
                ],
                "failed",
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_autotune.best_reference_status"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_step_autotune.best_reference_status"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "best_reference_status"
                ],
                "failed",
            )

    def test_validate_real_decode_fails_on_autotune_candidate_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            space_json = root / "decode_step_space.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [2],
                        "generation_template": [
                            "device_argmax_greedy",
                            "full_logits",
                        ],
                        "mlp_intermediate_dtype": [None],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    }
                )
            )
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

            original_autotune = validation_module.run_decode_step_autotune

            def autotune_with_bad_candidate_reference(*args, **kwargs):
                autotune = original_autotune(*args, **kwargs)
                autotune["candidates"][0]["reference_status"] = "failed"
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(autotune, indent=2) + "\n")
                return autotune

            with patch.object(
                validation_module,
                "run_decode_step_autotune",
                side_effect=autotune_with_bad_candidate_reference,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        decode_step_search_space_path=space_json,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_step_autotune"], "pass")
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "candidate_summaries"
                ][0]["reference_status"],
                "failed",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "best_reference_status"
                ],
                "passed",
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_autotune.candidates"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_step_autotune.candidates"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "candidate_summaries"
                ][0]["reference_status"],
                "failed",
            )

    def test_validate_real_decode_fails_on_autotune_knob_coverage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
            space_json = root / "decode_step_space.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [2],
                        "generation_template": ["device_argmax_greedy"],
                        "mlp_intermediate_dtype": [None],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    }
                )
            )
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

            original_autotune = validation_module.run_decode_step_autotune

            def autotune_with_incomplete_knob_coverage(*args, **kwargs):
                autotune = original_autotune(*args, **kwargs)
                autotune["knob_coverage"] = {
                    "knobs": ["lm_head_split_count"],
                    "candidate_count": autotune["candidate_count"],
                    "values": {"lm_head_split_count": [2]},
                    "value_counts": {"lm_head_split_count": {"2": 1}},
                    "varied_knobs": [],
                }
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(autotune, indent=2) + "\n")
                return autotune

            with patch.object(
                validation_module,
                "run_decode_step_autotune",
                side_effect=autotune_with_incomplete_knob_coverage,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        decode_step_search_space_path=space_json,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_step_autotune"], "pass")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_autotune.knob_coverage"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_step_autotune.knob_coverage"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "knob_coverage"
                ]["knobs"],
                ["lm_head_split_count"],
            )

    def test_validate_real_decode_requires_default_autotune_variation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_autotune = validation_module.run_decode_step_autotune

            def autotune_without_default_variation(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs["space"] = {
                    "lm_head_split_count": [2],
                    "generation_template": [
                        "device_argmax_greedy",
                        "full_logits",
                    ],
                    "mlp_intermediate_dtype": [None],
                    "attention_sdpa_output_memory_config": [None],
                    "attention_concat_heads_output_memory_config": [None],
                }
                autotune = original_autotune(*args, **kwargs)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(autotune, indent=2) + "\n")
                return autotune

            with patch.object(
                validation_module,
                "run_decode_step_autotune",
                side_effect=autotune_without_default_variation,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertTrue(report["decode_step_search_space_is_default"])
            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_step_autotune.default_knob_variation"],
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "missing_varied_knobs"
                ],
                [
                    "lm_head_split_count",
                    "mlp_intermediate_dtype",
                    "attention_sdpa_output_memory_config",
                    "attention_concat_heads_output_memory_config",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_step_autotune.default_knob_variation"],
            )
            self.assertTrue(
                evidence["validation"]["decode_step_search_space_is_default"]
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_step_autotune"][
                    "missing_varied_knobs"
                ],
                [
                    "lm_head_split_count",
                    "mlp_intermediate_dtype",
                    "attention_sdpa_output_memory_config",
                    "attention_concat_heads_output_memory_config",
                ],
            )

    def test_validate_real_decode_fails_on_attention_layer_primitive_reports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_attention = validation_module.run_smoke_attention_layer

            def attention_with_bad_primitive_error(*args, **kwargs):
                report = original_attention(*args, **kwargs)
                report["primitive_reports"][0]["error"] = "api mismatch"
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_attention_layer",
                side_effect=attention_with_bad_primitive_error,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["attention_layer"], "pass")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["attention_layer.primitive_reports"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["attention_layer.primitive_reports"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["attention_layer"][
                    "primitive_reports"
                ][0]["error"],
                "api mismatch",
            )

    def test_validate_real_decode_writes_evidence_on_runtime_step_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            def shell_no_device(*args, **kwargs):
                report = {
                    "schema_version": 1,
                    "passed": False,
                    "status": "no_device",
                    "layers_requested": 1,
                    "parameter_source": None,
                    "input_source": None,
                    "runtime_input_tensor_count": 0,
                    "reference": {
                        "status": "not_run",
                        "kind": None,
                        "checks": [],
                    },
                    "error": "TTNN device is unavailable",
                }
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_shell",
                side_effect=shell_no_device,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                    )

            self.assertEqual(report["status"], "no_device")
            self.assertEqual(report["results"]["materialize_parameters"], "pass")
            self.assertEqual(report["results"]["decode_shell"], "no_device")
            self.assertEqual(report["results"]["attention_primitives"], "skipped")
            self.assertEqual(report["results"]["attention_layer"], "skipped")
            self.assertEqual(report["results"]["smoke_decode_step"], "skipped")
            self.assertEqual(report["evidence"]["status"], "incomplete")
            self.assertTrue(
                (out_dir / "real_decode_evidence_manifest.json").is_file()
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(evidence["validation"]["status"], "no_device")
            self.assertEqual(
                evidence["validation"]["results"]["decode_shell"],
                "no_device",
            )
            self.assertEqual(
                evidence["validation"]["failed_steps"],
                ["decode_shell"],
            )
            self.assertEqual(
                evidence["validation"]["skipped_steps"],
                [
                    "attention_primitives",
                    "attention_layer",
                    "single_layer_decode",
                    "smoke_decode_step",
                    "profile_decode_step",
                    "prompt_decode_loop",
                    "generate_prefill_decode",
                    "profile_generate",
                    "generate_depth_sweep",
                    "decode_depth_sweep",
                    "decode_step_autotune",
                ],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_shell"]["runtime_status"],
                "no_device",
            )
            artifact_names = {
                artifact["name"]: artifact for artifact in evidence["artifacts"]
            }
            self.assertTrue(artifact_names["report"]["exists"])
            self.assertTrue(artifact_names["decode_shell_report"]["exists"])
            self.assertFalse(artifact_names["attention_layer_report"]["exists"])
            self.assertFalse(artifact_names["smoke_report"]["exists"])

    def test_validate_real_decode_guard_reports_device_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            report = validate_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out_dir=out_dir,
                layers=1,
                batch_size=2,
                cache_len=16,
                device="p150a",
                skip_autotune=True,
                guard_device_busy=True,
                device_process_environment={
                    "status": "busy",
                    "conflict_count": 1,
                    "reset_in_progress": True,
                    "conflicts": [
                        {
                            "kind": "tt_smi_reset",
                            "user": "other",
                            "pid": 123,
                            "command": "tt-smi -r 0",
                        }
                    ],
                },
            )

            self.assertEqual(report["status"], "device_busy")
            self.assertEqual(
                report["results"]["official_config_diff"],
                "skipped",
            )
            self.assertEqual(
                report["steps"]["device_exclusive_check"]["status"],
                "device_busy",
            )
            self.assertEqual(
                report["runtime_diagnostics"]["status"],
                "device_busy",
            )
            self.assertTrue(
                (out_dir / "real_decode_evidence_manifest.json").is_file()
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["validation"]["status"], "device_busy")
            self.assertEqual(
                evidence["runtime_diagnostics"]["status"],
                "device_busy",
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "device_preflight_diagnostics"
                ]["status"],
                "device_busy",
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "tenstorrent_process_environment"
                ]["conflict_count"],
                1,
            )

    def test_validate_real_decode_guard_reports_device_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            report = validate_real_decode(
                program_dir=program_dir,
                model_path=model_dir,
                out_dir=out_dir,
                layers=1,
                batch_size=2,
                cache_len=16,
                device="p150a",
                skip_autotune=True,
                guard_device_health=True,
                device_process_environment={
                    "status": "idle",
                    "conflict_count": 0,
                    "reset_in_progress": False,
                    "conflicts": [],
                },
                device_health_environment={
                    "status": "fail",
                    "device_id": 0,
                    "returncode": 135,
                    "stdout": "",
                    "stderr": "Bus error (core dumped)",
                },
            )

            self.assertEqual(report["status"], "device_unhealthy")
            self.assertEqual(
                report["results"]["official_config_diff"],
                "skipped",
            )
            self.assertEqual(
                report["steps"]["device_health_check"]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                report["runtime_diagnostics"]["status"],
                "device_unhealthy",
            )
            self.assertTrue(
                report["runtime_diagnostics"]["device_reset_recommended"]
            )
            self.assertTrue(
                (out_dir / "real_decode_evidence_manifest.json").is_file()
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(
                evidence["validation"]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                evidence["runtime_diagnostics"]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "device_preflight_diagnostics"
                ]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                evidence["device_evidence"][
                    "tenstorrent_runtime_health"
                ]["status"],
                "fail",
            )

    def test_recover_real_decode_process_failure_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "validate_real"
            out_dir.mkdir()
            report_path = out_dir / "real_decode_validation_report.json"
            evidence_path = out_dir / "real_decode_evidence_manifest.json"
            results = {
                step: "pending" for step in REAL_DECODE_VALIDATION_STEPS
            }
            results["official_config_diff"] = "pass"
            results["materialize_parameters"] = "pass"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "command": "validate-real-decode",
                        "status": "running",
                        "out_dir": str(out_dir),
                        "dry_run": False,
                        "guard_device_health": True,
                        "device_id": 0,
                        "results": results,
                        "steps": {
                            "official_config_diff": {"status": "pass"},
                            "materialize_parameters": {"status": "pass"},
                        },
                        "artifacts": {
                            "report": str(report_path),
                            "evidence_manifest": str(evidence_path),
                        },
                    }
                )
            )

            report = recover_real_decode_process_failure(
                out_dir=out_dir,
                returncode=-7,
                stderr="Bus error in libtt_metal",
                command=["python", "-m", "validate-real-decode"],
            )

            self.assertEqual(report["status"], "device_unhealthy")
            self.assertEqual(
                report["results"]["decode_shell"],
                "device_unhealthy",
            )
            self.assertEqual(
                report["steps"]["decode_shell"]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                report["results"]["attention_primitives"],
                "skipped",
            )
            self.assertEqual(
                report["runtime_diagnostics"]["status"],
                "device_unhealthy",
            )
            self.assertTrue(evidence_path.is_file())
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(
                evidence["validation"]["status"],
                "device_unhealthy",
            )

    def test_recover_real_decode_process_timeout_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "validate_real"
            out_dir.mkdir()
            report_path = out_dir / "real_decode_validation_report.json"
            evidence_path = out_dir / "real_decode_evidence_manifest.json"
            results = {
                step: "pending" for step in REAL_DECODE_VALIDATION_STEPS
            }
            results["official_config_diff"] = "pass"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "command": "validate-real-decode",
                        "status": "running",
                        "out_dir": str(out_dir),
                        "dry_run": False,
                        "guard_device_health": True,
                        "device_id": 0,
                        "results": results,
                        "steps": {
                            "official_config_diff": {"status": "pass"},
                        },
                        "artifacts": {
                            "report": str(report_path),
                            "evidence_manifest": str(evidence_path),
                        },
                    }
                )
            )

            report = recover_real_decode_process_timeout(
                out_dir=out_dir,
                timeout_seconds=12.5,
                stdout="partial stdout",
                stderr="partial stderr",
                command=["python", "-m", "validate-real-decode"],
            )

            self.assertEqual(report["status"], "device_unhealthy")
            self.assertEqual(
                report["tenstorrent_runtime_health"]["status"],
                "timeout",
            )
            self.assertEqual(
                report["tenstorrent_runtime_health"]["timeout_seconds"],
                12.5,
            )
            self.assertEqual(
                report["results"]["materialize_parameters"],
                "device_unhealthy",
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"]["error"]["type"],
                "SubprocessTimeout",
            )
            self.assertEqual(
                report["results"]["decode_shell"],
                "skipped",
            )
            self.assertEqual(
                report["runtime_diagnostics"]["status"],
                "device_unhealthy",
            )
            self.assertTrue(evidence_path.is_file())
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(
                evidence["validation"]["status"],
                "device_unhealthy",
            )
            self.assertEqual(
                evidence["runtime_diagnostics"]["findings"][0][
                    "health_status"
                ],
                "timeout",
            )

    def test_validate_real_decode_cli_isolates_signal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "validate_real"
            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli."
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["python", "-m", "models...cli"],
                    -7,
                    stdout="",
                    stderr="Bus error in libtt_metal",
                ),
            ):
                self.assertEqual(
                    main(
                        [
                            "validate-real-decode",
                            "--program-dir",
                            str(root / "program"),
                            "--model-path",
                            str(root / "model"),
                            "--out-dir",
                            str(out_dir),
                            "--guard-device-health",
                        ]
                    ),
                    1,
                )

            report = json.loads(
                (out_dir / "real_decode_validation_report.json").read_text()
            )
            self.assertEqual(report["status"], "device_unhealthy")
            self.assertEqual(
                report["results"]["official_config_diff"],
                "device_unhealthy",
            )
            self.assertTrue(
                (out_dir / "real_decode_evidence_manifest.json").is_file()
            )

    def test_validate_real_decode_cli_isolates_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "validate_real"
            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct.cli."
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    ["python", "-m", "models...cli"],
                    timeout=9.0,
                    output="partial stdout",
                    stderr="partial stderr",
                ),
            ) as run_mock:
                self.assertEqual(
                    main(
                        [
                            "validate-real-decode",
                            "--program-dir",
                            str(root / "program"),
                            "--model-path",
                            str(root / "model"),
                            "--out-dir",
                            str(out_dir),
                            "--guard-device-health",
                            "--device-isolation-timeout-seconds",
                            "9",
                        ]
                    ),
                    1,
                )

            self.assertEqual(run_mock.call_args.kwargs["timeout"], 9.0)
            report = json.loads(
                (out_dir / "real_decode_validation_report.json").read_text()
            )
            self.assertEqual(report["status"], "device_unhealthy")
            self.assertEqual(
                report["tenstorrent_runtime_health"]["status"],
                "timeout",
            )
            self.assertEqual(
                report["results"]["official_config_diff"],
                "device_unhealthy",
            )
            self.assertEqual(
                report["steps"]["official_config_diff"]["error"]["type"],
                "SubprocessTimeout",
            )
            self.assertTrue(
                (out_dir / "real_decode_evidence_manifest.json").is_file()
            )

    def test_validate_real_decode_diagnoses_firmware_init_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            def shell_firmware_init_failure(*args, **kwargs):
                message = (
                    "RuntimeError: Device 0 init: failed to initialize FW! "
                    "Try resetting the board."
                )
                report = {
                    "schema_version": 1,
                    "passed": False,
                    "status": "runtime_error",
                    "layers_requested": 1,
                    "parameter_source": None,
                    "input_source": None,
                    "runtime_input_tensor_count": 0,
                    "reference": {
                        "status": "not_run",
                        "kind": None,
                        "checks": [],
                    },
                    "error": message,
                }
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_shell",
                side_effect=shell_firmware_init_failure,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                    )

            diagnostics = report["runtime_diagnostics"]
            self.assertEqual(diagnostics["status"], "device_reset_recommended")
            self.assertTrue(diagnostics["device_reset_recommended"])
            self.assertIn("tt-smi -r", diagnostics["recommended_action"])
            self.assertEqual(
                diagnostics["findings"][0]["kind"],
                "tenstorrent_firmware_init_failed",
            )
            self.assertIn(
                "decode_shell.error",
                diagnostics["findings"][0]["path"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(
                evidence["runtime_diagnostics"]["status"],
                "device_reset_recommended",
            )
            self.assertTrue(
                evidence["runtime_diagnostics"]["device_reset_recommended"]
            )

    def test_validate_real_decode_fails_acceptance_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    min_tokens_per_second_per_user=1.0e12,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_step_autotune"], "skipped")
            self.assertEqual(report["acceptance"]["status"], "failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.min_tokens_per_second_per_user"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.min_tokens_per_second_per_user"],
            )

    def test_validate_real_decode_fails_baseline_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    baseline_tokens_per_second_per_user=1.0e12,
                    min_baseline_ratio=1.0,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(report["results"]["decode_step_autotune"], "skipped")
            self.assertEqual(report["acceptance"]["status"], "failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.min_baseline_ratio"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.min_baseline_ratio"],
            )
            baseline = evidence["performance_evidence"]["throughput_baseline"]
            self.assertEqual(baseline["baseline"], 1.0e12)
            self.assertEqual(baseline["min_ratio"], 1.0)
            self.assertLess(baseline["ratio"], 1.0)
            self.assertFalse(baseline["passed"])
            matrix = evidence["acceptance_gate_matrix"]
            self.assertEqual(
                matrix["failed_gates"],
                ["profile_decode_step.min_baseline_ratio"],
            )
            self.assertEqual(matrix["missing_gates"], [])
            gap = evidence["performance_evidence"][
                "performance_gap_summary"
            ]
            self.assertEqual(gap["baseline"], 1.0e12)
            self.assertLess(gap["ratio"], 1.0)
            self.assertFalse(gap["passed_min_ratio"])
            self.assertGreater(gap["shortfall_to_min_ratio"], 0.0)
            self.assertGreater(gap["required_speedup_to_min_ratio"], 1.0)
            self.assertIn(
                gap["bottleneck"]["max_section"],
                gap["bottleneck"]["sections_ms"],
            )

    def test_validate_real_decode_fails_on_profile_runtime_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_with_bad_runtime_status(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["status"] = "runtime_error"
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_with_bad_runtime_status,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(
                report["steps"]["profile_decode_step"]["status"],
                "pass",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["runtime_status"],
                "runtime_error",
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.runtime_status"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.runtime_status"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "runtime_status"
                ],
                "runtime_error",
            )

    def test_validate_real_decode_skip_profile_continues_generate_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            def fake_generate(*args, **kwargs):
                out = Path(kwargs["out"])
                generate_report = {
                    "status": "passed",
                    "passed": True,
                    "runtime_status": "passed",
                    "prefill_status": "passed",
                    "generate_runtime_owned": True,
                    "decode_loop_runtime_owned": True,
                    "kv_cache_source": "prefill",
                    "model_semantics": "prompt_conditioned_prefill_decode",
                    "layers": kwargs["layers"],
                    "batch_size": kwargs["batch_size"],
                    "cache_len": kwargs["cache_len"],
                    "prefill_len": kwargs["prefill_len"],
                    "max_new_tokens": kwargs["max_new_tokens"],
                    "decode_steps": 2,
                    "generated_token_budget": {
                        "total_planned_generated_tokens": 3,
                    },
                    "parameter_source": "hf_model",
                    "input_source": "prompt_prefill",
                    "runtime_owner": "TTNNDirectRuntimeContext",
                    "generated_token_ids": [[17, 23, 23], [17, 23, 23]],
                    "generated_token_id_source": "runtime",
                    "token_materialization_status": "passed",
                    "generated_text": "<tok:17> <tok:23> <tok:23>",
                    "generated_text_by_user": [
                        "<tok:17> <tok:23> <tok:23>",
                        "<tok:17> <tok:23> <tok:23>",
                    ],
                    "generated_text_status": "fallback",
                    "generated_text_source": "tokenizer",
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "parameter_setup": {
                        "prefill_prompt_runtime_input_tensor_count": 1,
                        "decode_runtime_state_input_tensor_count": 2,
                        "prefill_rotary_runtime_input_tensor_count": 3,
                        "decode_rotary_runtime_input_tensor_count": 3,
                    },
                    "kv_cache_runtime_state": {
                        "tensor_count": 2,
                    },
                    "prefill": {
                        "status": "passed",
                        "cache_population": [
                            {
                                "layer_id": 0,
                                "status": "filled",
                                "write_policy": "paged_fill_cache_per_user",
                                "update_shape_layout": (
                                    "batch_heads_seq_head_dim"
                                ),
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
                    "prompt_tokenization": {"status": "tokenized"},
                    "prefill_tokenization": {"status": "tokenized"},
                    "runtime_context": {
                        "class": "TTNNDirectRuntimeContext",
                        "status": "built",
                    },
                    "end_to_end_contract": {
                        "status": "passed",
                        "failed_checks": [],
                    },
                    "host_copy_profile": {
                        "status": "measured",
                        "total_ms": 0.1,
                    },
                    "section_profile": {"status": "measured"},
                    "per_step_token_metadata": [],
                    "step_reports": [],
                    "output_shapes": {"token": [2, 1]},
                    "latency_ms": 1.0,
                    "throughput_summary": {
                        "status": "measured",
                        "tokens_per_second_per_user": 1.0,
                    },
                    "ttnn_environment": {
                        "module_available": True,
                    },
                    "trace": {"status": "disabled"},
                    "reference_status": "passed",
                    "reference_failed_checks": [],
                }
                out.write_text(json.dumps(generate_report, indent=2) + "\n")
                return generate_report

            def fake_generate_depth_sweep(*args, **kwargs):
                out = Path(kwargs["out"])
                reports_dir = Path(kwargs["reports_dir"])
                reports_dir.mkdir(parents=True, exist_ok=True)
                depth_report = reports_dir / "generate_depth_1.json"
                depth_report.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "passed": True,
                            "model_semantics": (
                                "prompt_conditioned_prefill_decode"
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                sweep_report = {
                    "schema_version": 1,
                    "command": "generate-depth-sweep",
                    "status": "pass",
                    "passed": True,
                    "dry_run": False,
                    "depths": [1],
                    "depth_count": 1,
                    "max_depth": 1,
                    "covered_full_depth": False,
                    "require_full_depth": False,
                    "status_counts": {"passed": 1},
                    "prefill_status_counts": {"passed": 1},
                    "generated_text_status_counts": {"fallback": 1},
                    "model_semantics_counts": {
                        "prompt_conditioned_prefill_decode": 1,
                    },
                    "passed_depth_count": 1,
                    "failed_depths": [],
                    "failed_depth_diagnostics": [],
                    "records": [
                        {
                            "depth": 1,
                            "status": "passed",
                            "passed": True,
                            "generate_report": str(depth_report),
                            "model_semantics": (
                                "prompt_conditioned_prefill_decode"
                            ),
                        }
                    ],
                    "acceptance": {
                        "status": "passed",
                        "passed": True,
                        "failed_checks": [],
                    },
                }
                out.write_text(json.dumps(sweep_report, indent=2) + "\n")
                return sweep_report

            with patch.object(
                validation_module,
                "run_generate",
                side_effect=fake_generate,
            ), patch.object(
                validation_module,
                "run_generate_depth_sweep",
                side_effect=fake_generate_depth_sweep,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        max_new_tokens=3,
                        prefill_len=8,
                        device="p150a",
                        prompt="hello tenstorrent",
                        tokenizer_path=model_dir,
                        tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                        skip_profile_decode_step=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertTrue(report["skip_profile_decode_step"])
            self.assertEqual(
                report["results"]["profile_decode_step"],
                "skipped",
            )
            self.assertEqual(
                report["results"]["generate_prefill_decode"],
                "pass",
            )
            self.assertEqual(
                report["results"]["profile_generate"],
                "skipped",
            )
            self.assertEqual(
                report["results"]["generate_depth_sweep"],
                "pass",
            )
            self.assertEqual(
                report["results"]["decode_depth_sweep"],
                "skipped",
            )
            self.assertEqual(
                report["results"]["decode_step_autotune"],
                "skipped",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["runtime_status"],
                "skipped",
            )
            self.assertEqual(
                report["steps"]["generate_prefill_decode"]["prefill_status"],
                "passed",
            )
            self.assertTrue(
                report["steps"]["generate_prefill_decode"][
                    "decode_loop_runtime_owned"
                ]
            )
            self.assertEqual(
                report["final_acceptance_plan"]["required_runtime_steps"],
                [
                    "official_config_diff",
                    "materialize_parameters",
                    "decode_shell",
                    "attention_primitives",
                    "attention_layer",
                    "single_layer_decode",
                    "smoke_decode_step",
                    "prompt_decode_loop",
                    "generate_prefill_decode",
                    "generate_depth_sweep",
                ],
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"]["status_counts"],
                {"passed": 1},
            )
            self.assertEqual(
                report["steps"]["generate_depth_sweep"][
                    "model_semantics_counts"
                ],
                {"prompt_conditioned_prefill_decode": 1},
            )
            profile_stub = json.loads(
                (out_dir / "decode_step_profile_report.json").read_text()
            )
            self.assertEqual(profile_stub["status"], "skipped")
            profile_generate_stub = json.loads(
                (out_dir / "generate_profile_report.json").read_text()
            )
            self.assertEqual(profile_generate_stub["status"], "skipped")
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "runtime_status"
                ],
                "skipped",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_generate"]["status"],
                "skipped",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_generate"]["reason"],
                "skip_profile_decode_step requested",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_prefill_decode"][
                    "prefill_status"
                ],
                "passed",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"][
                    "status"
                ],
                "pass",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"][
                    "model_semantics_counts"
                ],
                {"prompt_conditioned_prefill_decode": 1},
            )
            self.assertEqual(
                evidence["runtime_evidence"]["generate_depth_sweep"][
                    "failed_depth_diagnostics"
                ],
                [],
            )
            generate_evidence = evidence["runtime_evidence"][
                "generate_prefill_decode"
            ]
            self.assertEqual(generate_evidence["end_to_end_failed_checks"], [])
            self.assertEqual(
                generate_evidence["end_to_end_contract"]["status"],
                "passed",
            )
            self.assertEqual(
                generate_evidence[
                    "prefill_cache_population_diagnostics"
                ][0]["update_shape_layout"],
                "batch_heads_seq_head_dim",
            )
            self.assertEqual(
                generate_evidence[
                    "prefill_cache_population_diagnostics"
                ][0]["key_update_shape"],
                [1, 2, 8, 4],
            )
            self.assertEqual(
                generate_evidence["host_copy_profile"]["status"],
                "measured",
            )
            self.assertIsNone(generate_evidence["failure_diagnostics"])

    def test_generate_prefill_decode_failure_diagnostics_capture_shapes_ops(
        self,
    ) -> None:
        diagnostics = (
            validation_module._generate_prefill_decode_failure_diagnostics(
                {
                    "status": "reference_mismatch",
                    "passed": False,
                    "runtime_status": "reference_mismatch",
                    "error": "generate reference mismatch",
                    "detail": "decode output shape mismatch",
                    "model_semantics": "prompt_conditioned_prefill_decode",
                    "layers": 2,
                    "layout": "tile",
                    "prefill_status": "passed",
                    "decode_loop_runtime_owned": False,
                    "end_to_end_contract": {
                        "status": "failed",
                        "failed_checks": [
                            "generate.decode_loop_runtime_owned"
                        ],
                    },
                    "prefill": {
                        "cache_population": [
                            {
                                "layer_id": 1,
                                "status": "filled",
                                "write_policy": "paged_fill_cache_per_user",
                                "update_shape_layout": (
                                    "batch_heads_seq_head_dim"
                                ),
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
                            "decode_runtime_state": {
                                "cache_position_value": 8
                            },
                            "rotary_runtime_state": {
                                "cos_shape": [2, 1, 64]
                            },
                            "reference": {
                                "status": "failed",
                                "failed_checks": ["decode.output_shape"],
                                "observed_ops": [
                                    "paged_scaled_dot_product_attention_decode"
                                ],
                                "expected_ops": [
                                    "paged_scaled_dot_product_attention_decode"
                                ],
                            },
                            "error": "decode output shape mismatch",
                        }
                    ],
                }
            )
        )

        self.assertEqual(diagnostics["status"], "reference_mismatch")
        self.assertEqual(
            diagnostics["end_to_end_contract"]["failed_checks"],
            ["generate.decode_loop_runtime_owned"],
        )
        self.assertEqual(
            diagnostics["prefill"]["cache_population"][0][
                "update_shape_layout"
            ],
            "batch_heads_seq_head_dim",
        )
        self.assertEqual(
            diagnostics["prefill"]["cache_population"][0][
                "key_update_shape"
            ],
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
        self.assertEqual(
            failed_step["error"],
            "decode output shape mismatch",
        )

    def test_validate_real_decode_fails_without_profile_runtime_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_without_runtime_inputs(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["input_source"] = None
                setup = dict(profile.get("parameter_setup") or {})
                setup["synthetic_runtime_input_tensor_count"] = 0
                profile["parameter_setup"] = setup
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_without_runtime_inputs,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                [
                    "profile_decode_step.input_source",
                    "profile_decode_step.synthetic_runtime_inputs",
                    "profile_decode_step.runtime_inputs",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                [
                    "profile_decode_step.input_source",
                    "profile_decode_step.synthetic_runtime_inputs",
                    "profile_decode_step.runtime_inputs",
                ],
            )
            self.assertIsNone(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "input_source"
                ]
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "synthetic_runtime_input_tensor_count"
                ],
                0,
            )

    def test_validate_real_decode_fails_on_profile_runtime_input_shapes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_with_bad_runtime_shape(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                input_shapes = dict(profile.get("input_shapes") or {})
                input_shapes["page_table"] = [1, 1]
                profile["input_shapes"] = input_shapes
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_with_bad_runtime_shape,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.runtime_inputs"],
            )
            observed = failed_checks[0]["observed"]
            self.assertEqual(
                observed["input_shapes"]["page_table"],
                [1, 1],
            )
            self.assertEqual(
                failed_checks[0]["expected"]["page_table"],
                [2, 1],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.runtime_inputs"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "input_shapes"
                ]["page_table"],
                [1, 1],
            )

    def test_validate_real_decode_fails_on_profile_output_shapes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_with_bad_output_shapes(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                output_shapes = dict(profile.get("output_shapes") or {})
                output_shapes["key_cache"] = [2, 8, 2, 4]
                kv_layers = [
                    dict(layer)
                    for layer in output_shapes.get("kv_cache_layers", [])
                    if isinstance(layer, dict)
                ]
                if kv_layers:
                    kv_layers[0]["key_cache"] = [2, 8, 2, 4]
                output_shapes["kv_cache_layers"] = kv_layers
                profile["output_shapes"] = output_shapes
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_with_bad_output_shapes,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.output_shapes"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.output_shapes"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "output_shapes"
                ]["key_cache"],
                [2, 8, 2, 4],
            )

    def test_validate_real_decode_fails_without_lm_head_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_without_lm_head_profile(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["lm_head_profile"] = {
                    "split_count": 0,
                    "lm_head_ms": 0.0,
                    "argmax_ms": 0.0,
                    "argmax_status": "skipped",
                }
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_without_lm_head_profile,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.lm_head_profile"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.lm_head_profile"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["argmax_status"],
                "skipped",
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "lm_head_profile"
                ]["split_count"],
                0,
            )

    def test_validate_real_decode_fails_without_measured_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_without_throughput(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["throughput_summary"] = {
                    "status": "unavailable",
                    "latency_ms": None,
                    "tokens_per_second_per_user": None,
                    "aggregate_tokens_per_second": None,
                }
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_without_throughput,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                [
                    "profile_decode_step.throughput_status",
                    "profile_decode_step.latency_ms",
                    "profile_decode_step.tokens_per_second_per_user",
                    "profile_decode_step.aggregate_tokens_per_second",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                [
                    "profile_decode_step.throughput_status",
                    "profile_decode_step.latency_ms",
                    "profile_decode_step.tokens_per_second_per_user",
                    "profile_decode_step.aggregate_tokens_per_second",
                ],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "throughput_summary"
                ]["status"],
                "unavailable",
            )

    def test_validate_real_decode_fails_when_required_weight_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_materialize = (
                validation_module.materialize_parameters_from_program
            )
            missing_path = "layers.0.attention.wqkv_packed.weight"

            def materialize_without_required_path(*args, **kwargs):
                report = original_materialize(*args, **kwargs)
                report["tensors"].pop(missing_path)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "materialize_parameters_from_program",
                side_effect=materialize_without_required_path,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["materialize_parameters.required_tensor_paths"],
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"][
                    "missing_required_tensor_paths"
                ],
                [missing_path],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["materialize_parameters.required_tensor_paths"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"][
                    "missing_required_tensor_paths"
                ],
                [missing_path],
            )

    def test_validate_real_decode_fails_on_materialized_weight_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_materialize = (
                validation_module.materialize_parameters_from_program
            )
            bad_path = "layers.0.mlp.gate_proj.weight"

            def materialize_with_bad_shape(*args, **kwargs):
                report = original_materialize(*args, **kwargs)
                report["tensors"][bad_path]["shape"] = [31, 16]
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "materialize_parameters_from_program",
                side_effect=materialize_with_bad_shape,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["materialize_parameters.tensor_shapes"],
            )
            mismatches = report["steps"]["materialize_parameters"][
                "materialized_tensor_shape_mismatches"
            ]
            self.assertEqual(
                mismatches,
                [
                    {
                        "path": bad_path,
                        "observed": [31, 16],
                        "expected": [32, 16],
                    }
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["materialize_parameters.tensor_shapes"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["materialization"][
                    "materialized_tensor_shape_mismatches"
                ],
                mismatches,
            )

    def test_validate_real_decode_fails_when_tensorized_weight_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            missing_path = "layers.0.attention.wqkv_packed.weight"

            def smoke_without_required_tensorized_path(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                tensorization = report["parameter_setup"]["tensorization"]
                tensorization["tensor_paths"] = [
                    path
                    for path in tensorization["tensor_paths"]
                    if path != missing_path
                ]
                tensorization["key_paths"] = [
                    path
                    for path in tensorization["key_paths"]
                    if path != missing_path
                ]
                tensorization["key_tensors"].pop(missing_path, None)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_without_required_tensorized_path,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.required_tensorized_tensor_paths"],
            )
            self.assertEqual(
                report["steps"]["smoke_decode_step"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [missing_path],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.required_tensorized_tensor_paths"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "missing_required_tensorized_tensor_paths"
                ],
                [missing_path],
            )

    def test_validate_real_decode_fails_on_tensorized_physical_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            bad_path = "layers.0.attention.wqkv_packed.weight"

            def smoke_with_bad_tensorized_shape(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                tensorization = report["parameter_setup"]["tensorization"]
                tensorization["key_tensors"][bad_path]["shape"] = [
                    1,
                    1,
                    15,
                    32,
                ]
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_with_bad_tensorized_shape,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.tensorized_physical_shapes"],
            )
            self.assertEqual(
                failed_checks[0]["observed"],
                [
                    {
                        "path": bad_path,
                        "transform": LINEAR_WEIGHT_TRANSFORM,
                        "source_shape": [32, 16],
                        "observed": [1, 1, 15, 32],
                        "expected": [1, 1, 16, 32],
                    }
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.tensorized_physical_shapes"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "physical_shape_mismatches"
                ],
                failed_checks[0]["observed"],
            )

    def test_validate_real_decode_fails_without_lm_head_transform_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step

            def smoke_without_lm_head_transform(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                tensorization = report["parameter_setup"]["tensorization"]
                tensorization["transform_counts"] = {}
                tensorization["key_tensors"]["lm_head.splits.0.weight"].pop(
                    "transform",
                    None,
                )
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_without_lm_head_transform,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.lm_head_transform"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.lm_head_transform"],
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_counts"
                ],
                {},
            )

    def test_validate_real_decode_fails_without_linear_transform_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            missing_path = "layers.0.attention.wqkv_packed.weight"

            def smoke_without_linear_transform(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                tensorization = report["parameter_setup"]["tensorization"]
                paths = tensorization["transform_paths_by_kind"][
                    "transpose_2d_to_4d"
                ]
                tensorization["transform_paths_by_kind"][
                    "transpose_2d_to_4d"
                ] = [path for path in paths if path != missing_path]
                tensorization["key_tensors"][missing_path].pop(
                    "transform",
                    None,
                )
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_without_linear_transform,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.linear_weight_transforms"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.linear_weight_transforms"],
            )
            self.assertNotIn(
                missing_path,
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_paths_by_kind"
                ]["transpose_2d_to_4d"],
            )

    def test_validate_real_decode_fails_without_embedding_norm_transform_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            missing_path = "embedding.weight"

            def smoke_without_embedding_transform(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                tensorization = report["parameter_setup"]["tensorization"]
                paths = tensorization["transform_paths_by_kind"][
                    "reshape_embedding_weight_4d"
                ]
                tensorization["transform_paths_by_kind"][
                    "reshape_embedding_weight_4d"
                ] = [path for path in paths if path != missing_path]
                tensorization["key_tensors"][missing_path].pop(
                    "transform",
                    None,
                )
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_without_embedding_transform,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.embedding_norm_weight_transforms"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.embedding_norm_weight_transforms"],
            )
            self.assertNotIn(
                missing_path,
                evidence["weight_evidence"]["smoke_tensorization"][
                    "transform_paths_by_kind"
                ]["reshape_embedding_weight_4d"],
            )

    def test_validate_real_decode_fails_when_shell_observed_op_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_shell = validation_module.run_smoke_decode_shell
            missing_op = "mlp_gate"

            def shell_without_required_observed_op(*args, **kwargs):
                report = original_shell(*args, **kwargs)
                observed_ops = report["reference"]["observed_ops"]
                observed_ops.remove(missing_op)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_shell",
                side_effect=shell_without_required_observed_op,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_shell.observed_op_sequence"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_shell.observed_op_sequence"],
            )
            self.assertNotIn(
                missing_op,
                evidence["runtime_evidence"]["decode_shell"][
                    "reference_observed_ops"
                ],
            )

    def test_validate_real_decode_fails_when_observed_op_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            missing_op = "qkv_linear"

            def smoke_without_required_observed_op(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                observed_ops = report["reference"]["observed_ops"]
                observed_ops.remove(missing_op)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_without_required_observed_op,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.observed_op_sequence"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.observed_op_sequence"],
            )
            self.assertNotIn(
                missing_op,
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "reference_observed_ops"
                ],
            )

    def test_validate_real_decode_fails_on_reference_failed_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step
            failed_check_name = "forced_reference_mismatch"

            def smoke_with_failed_reference_check(*args, **kwargs):
                report = original_smoke(*args, **kwargs)
                report["reference"]["checks"].append(
                    {
                        "name": failed_check_name,
                        "passed": False,
                        "observed": "bad",
                        "expected": "good",
                    }
                )
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(report, indent=2) + "\n")
                return report

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_with_failed_reference_check,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.reference_failed_checks"],
            )
            self.assertEqual(
                failed_checks[0]["observed"],
                [failed_check_name],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.reference_failed_checks"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"][
                    "reference_failed_checks"
                ],
                [failed_check_name],
            )

    def test_validate_real_decode_fails_without_layer_profile_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_without_layer_profiles(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["layer_profiles"] = []
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_without_layer_profiles,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                [
                    "profile_decode_step.layer_profile_count",
                    "profile_decode_step.layer_profile_sections",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                [
                    "profile_decode_step.layer_profile_count",
                    "profile_decode_step.layer_profile_sections",
                ],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "layer_profiles"
                ],
                [],
            )

    def test_validate_real_decode_fails_on_runtime_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_with_wrong_batch_size(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile["batch_size"] = int(profile["batch_size"]) + 1
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_with_wrong_batch_size,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.batch_size"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.batch_size"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "batch_size"
                ],
                3,
            )

    def test_validate_real_decode_fails_on_trace_iteration_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_smoke = validation_module.run_smoke_decode_step

            def smoke_with_wrong_trace_iterations(*args, **kwargs):
                smoke = original_smoke(*args, **kwargs)
                smoke["trace"]["iterations"] = 1
                return smoke

            with patch.object(
                validation_module,
                "run_smoke_decode_step",
                side_effect=smoke_with_wrong_trace_iterations,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        trace=True,
                        trace_iterations=2,
                        require_trace=True,
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["smoke_decode_step.trace_iterations"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["smoke_decode_step.trace_iterations"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"]["trace"][
                    "iterations"
                ],
                1,
            )
            self.assertEqual(
                evidence["runtime_evidence"]["smoke_decode_step"]["trace"][
                    "execute_sample_count"
                ],
                2,
            )

    def test_validate_real_decode_fails_on_profile_trace_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_profile = validation_module.profile_decode_step

            def profile_with_zero_trace_samples(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                trace = dict(profile.get("trace") or {})
                trace["execute_samples_ms"] = [0.0, 0.0]
                trace["execute_latency_ms"] = 0.0
                profile["trace"] = trace
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                validation_module,
                "profile_decode_step",
                side_effect=profile_with_zero_trace_samples,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        trace=True,
                        trace_iterations=2,
                        require_trace=True,
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["profile_decode_step.trace_profile"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.trace_profile"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"]["trace"][
                    "execute_samples_ms"
                ],
                [0.0, 0.0],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"]["trace"][
                    "execute_latency_ms"
                ],
                0.0,
            )

    def test_validate_real_decode_fails_without_tt_metal_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            fake_ttnn = _make_fake_ttnn()
            delattr(fake_ttnn, "__tt_metal_commit__")
            with patch.dict("os.environ", {}, clear=True):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        ttnn_module=fake_ttnn,
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                [
                    "attention_primitives.tt_metal_git_commit",
                    "attention_primitives.primitive_reports",
                    "attention_layer.tt_metal_git_commit",
                    "single_layer_decode.tt_metal_git_commit",
                    "smoke_decode_step.tt_metal_git_commit",
                    "profile_decode_step.tt_metal_git_commit",
                ],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                [
                    "attention_primitives.tt_metal_git_commit",
                    "attention_primitives.primitive_reports",
                    "attention_layer.tt_metal_git_commit",
                    "single_layer_decode.tt_metal_git_commit",
                    "smoke_decode_step.tt_metal_git_commit",
                    "profile_decode_step.tt_metal_git_commit",
                ],
            )

    def test_validate_real_decode_fails_when_numeric_shell_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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
                report = validate_real_decode(
                    program_dir=program_dir,
                    model_path=model_dir,
                    out_dir=out_dir,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    skip_autotune=True,
                    min_tokens_per_second_per_user=0.0,
                    require_decode_shell_numeric_reference=True,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "acceptance_failed")
            self.assertEqual(
                report["steps"]["decode_shell"]["numeric_reference_status"],
                "not_run",
            )
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_shell.numeric_reference"],
            )
            self.assertEqual(
                failed_checks[0]["observed"]["status"],
                "not_run",
            )

    def test_validate_real_decode_fails_on_decode_shell_numeric_pcc(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            out_dir = root / "validate_real"
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

            original_shell = validation_module.run_smoke_decode_shell

            def shell_with_low_numeric_pcc(*args, **kwargs):
                shell = original_shell(*args, **kwargs)
                numeric = dict(
                    shell.get("reference", {}).get("numeric_reference") or {}
                )
                numeric.update(
                    {
                        "status": "passed",
                        "passed": True,
                        "kind": "torch_decode_shell",
                        "pcc": 0.25,
                        "pcc_threshold": 0.99,
                        "checks": [],
                    }
                )
                reference = dict(shell.get("reference") or {})
                reference["numeric_reference"] = numeric
                shell["reference"] = reference
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(shell, indent=2) + "\n")
                return shell

            with patch.object(
                validation_module,
                "run_smoke_decode_shell",
                side_effect=shell_with_low_numeric_pcc,
            ):
                with _fake_torch_and_safetensors():
                    report = validate_real_decode(
                        program_dir=program_dir,
                        model_path=model_dir,
                        out_dir=out_dir,
                        layers=1,
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        skip_autotune=True,
                        min_tokens_per_second_per_user=0.0,
                        require_decode_shell_numeric_reference=True,
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "acceptance_failed")
            failed_checks = [
                check for check in report["acceptance"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["decode_shell.numeric_reference"],
            )
            self.assertEqual(failed_checks[0]["observed"]["pcc"], 0.25)
            self.assertEqual(
                failed_checks[0]["observed"]["pcc_threshold"],
                0.99,
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["decode_shell.numeric_reference"],
            )
            self.assertEqual(
                evidence["runtime_evidence"]["decode_shell"]["pcc"],
                0.25,
            )

    def test_cli_validate_direct_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "bad_config.json"
            out_dir = root / "validate"
            _write_fake_model_config(model_dir)
            config_json.write_text("{}")

            exit_code = main(
                [
                    "validate-direct",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 1)
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["results"]["import_llama"], "fail")
            self.assertEqual(report["results"]["plan"], "skipped")
            self.assertEqual(
                report["steps"]["plan"]["reason"],
                "blocked by failed step: import_llama",
            )
            self.assertEqual(
                report["steps"]["import_llama"]["error"]["type"],
                "ValueError",
            )


def _check_by_name(report: dict[str, object], name: str) -> dict[str, object]:
    for check in report.get("checks", []):  # type: ignore[union-attr]
        if check.get("name") == name:
            return check
    raise AssertionError(f"missing check {name!r}")


def _fake_tenstorrent_device_environment() -> dict[str, object]:
    return {
        "device_available": True,
        "device_node_count": 1,
        "device_nodes": ["/dev/tenstorrent/0"],
        "filesystem_entries": ["/dev/tenstorrent/0"],
        "driver_loaded": True,
        "tt_smi_path": "/usr/bin/tt-smi",
        "tt_smi": {
            "status": "pass",
            "returncode": 0,
            "stdout": "Tenstorrent device 0",
            "stderr": "",
        },
    }


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-validate-direct",
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "tie_word_embeddings": False,
            }
        )
    )


def _write_template_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "device": "p150a",
                "model": "llama3.1-8b",
                "batch_size": 32,
                "decode_seq_len": 1,
                "prefill_seq_len": 128,
                "max_cache_len": 1024,
                "attention_template": "official_paged_attention_decode",
                "mlp_template": "official_gated_mlp_decode",
                "lm_head_template": "official_split_lm_head",
                "kv_cache_template": "paged_kv_cache",
                "generation_template": "device_argmax_greedy",
                "lm_head_split_count": 8,
                "dtype_recipe": "official_like_performance_seed",
            }
        )
    )


def _write_official_parity_from_program(program_dir: Path, path: Path) -> None:
    generated_config = json.loads((program_dir / "config.json").read_text())
    parity_view = build_config_parity_view(generated_config)
    path.write_text(
        json.dumps(
            {
                "model_name": generated_config.get("model_name"),
                "source": "unit_test_normalized_parity_reference",
                "parity_config": parity_view["parity_config"],
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
