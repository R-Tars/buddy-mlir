from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct import (
    validation as validation_module,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.validation import (
    REAL_DECODE_VALIDATION_STEPS,
    VALIDATION_STEPS,
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
    _make_fake_ttnn,
)


class ValidateDirectTest(unittest.TestCase):
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
            self.assertEqual(
                report["results"],
                {step: "pass" for step in VALIDATION_STEPS},
            )
            self.assertEqual(
                list(report["results"]),
                list(VALIDATION_STEPS),
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
            self.assertEqual(report["requested_batch_size"], 2)
            self.assertEqual(report["requested_cache_len"], 16)
            self.assertEqual(report["batch_size"], 2)
            self.assertEqual(report["cache_len"], 16)
            self.assertEqual(
                list(report["results"]),
                list(REAL_DECODE_VALIDATION_STEPS),
            )
            self.assertEqual(report["results"]["materialize_parameters"], "dry_run")
            self.assertEqual(report["results"]["decode_shell"], "dry_run")
            self.assertEqual(report["results"]["smoke_decode_step"], "dry_run")
            self.assertEqual(report["results"]["profile_decode_step"], "dry_run")
            self.assertEqual(report["results"]["decode_step_autotune"], "skipped")
            self.assertEqual(report["acceptance"]["status"], "dry_run")
            self.assertTrue(report["acceptance"]["passed"])
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
            self.assertTrue(
                (out_dir / "parameter_materialization_report.json").is_file()
            )
            self.assertTrue((out_dir / "decode_shell_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_smoke_report.json").is_file())
            self.assertTrue((out_dir / "decode_step_profile_report.json").is_file())
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "dry_run")
            self.assertEqual(evidence["validation"]["status"], "dry_run")
            self.assertEqual(evidence["validation"]["batch_size"], 2)
            self.assertEqual(evidence["validation"]["cache_len"], 16)
            self.assertTrue(evidence["requirements"]["require_trace"])
            self.assertEqual(evidence["acceptance"]["status"], "dry_run")
            artifact_names = {
                artifact["name"]: artifact for artifact in evidence["artifacts"]
            }
            self.assertTrue(artifact_names["report"]["exists"])
            self.assertTrue(artifact_names["smoke_report"]["exists"])
            self.assertFalse(artifact_names["autotune_report"]["exists"])
            self.assertEqual(
                report["evidence"]["manifest"],
                str(out_dir / "real_decode_evidence_manifest.json"),
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
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["program_num_layers"], 2)
            self.assertEqual(report["requested_batch_size"], 2)
            self.assertEqual(report["requested_cache_len"], 16)
            self.assertEqual(report["batch_size"], 2)
            self.assertEqual(report["cache_len"], 16)
            self.assertEqual(report["evidence"]["status"], "accepted")
            self.assertEqual(
                report["results"],
                {step: "pass" for step in REAL_DECODE_VALIDATION_STEPS},
            )
            self.assertEqual(
                report["steps"]["materialize_parameters"]["tensor_count"],
                21,
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
                report["steps"]["smoke_decode_step"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(report["steps"]["smoke_decode_step"]["layers"], 1)
            self.assertEqual(report["steps"]["smoke_decode_step"]["batch_size"], 2)
            self.assertEqual(report["steps"]["smoke_decode_step"]["cache_len"], 16)
            self.assertEqual(
                report["steps"]["profile_decode_step"]["parameter_source"],
                "hf_model",
            )
            self.assertEqual(report["steps"]["profile_decode_step"]["layers"], 1)
            self.assertEqual(report["steps"]["profile_decode_step"]["batch_size"], 2)
            self.assertEqual(report["steps"]["profile_decode_step"]["cache_len"], 16)
            self.assertEqual(
                report["steps"]["smoke_decode_step"]["reference_status"],
                "passed",
            )
            self.assertEqual(
                report["steps"]["profile_decode_step"]["reference_status"],
                "passed",
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
                "materialize_parameters.tensor_count",
                acceptance_check_names,
            )
            self.assertIn(
                "materialize_parameters.layer_ids",
                acceptance_check_names,
            )
            self.assertIn(
                "decode_shell.layers",
                acceptance_check_names,
            )
            self.assertIn(
                "smoke_decode_step.batch_size",
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
                "smoke_decode_step.tensorization_memory_configs",
                acceptance_check_names,
            )
            self.assertIn(
                "profile_decode_step.tensor_conversion_count",
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
                "profile_decode_step.tensorization_ttnn_memory_configs",
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
                1,
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["metric_direction"],
                "minimize",
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["status_counts"],
                {"profiled": 1},
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"][
                    "passed_candidate_count"
                ],
                1,
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
                report["steps"]["decode_step_autotune"]["reference_status_counts"],
                {"passed": 1},
            )
            self.assertEqual(
                report["steps"]["decode_step_autotune"]["trace_status_counts"],
                {"captured_and_executed": 1},
            )
            smoke_report = json.loads(
                (out_dir / "decode_step_smoke_report.json").read_text()
            )
            self.assertEqual(smoke_report["parameter_source"], "hf_model")
            self.assertEqual(
                smoke_report["parameter_setup"]["tensorization"]["tensor_count"],
                17,
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
            profile_report = json.loads(
                (out_dir / "decode_step_profile_report.json").read_text()
            )
            self.assertEqual(profile_report["parameter_source"], "hf_model")
            self.assertEqual(profile_report["trace"]["status"], "captured_and_executed")
            self.assertEqual(
                profile_report["parameter_setup"]["tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            autotune_report = json.loads(
                (out_dir / "decode_step_autotune_report.json").read_text()
            )
            self.assertEqual(autotune_report["best"]["reference_status"], "passed")
            self.assertEqual(autotune_report["status_counts"], {"profiled": 1})
            self.assertEqual(
                autotune_report["trace_status_counts"],
                {"captured_and_executed": 1},
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
            self.assertEqual(evidence["validation"]["batch_size"], 2)
            self.assertEqual(evidence["validation"]["cache_len"], 16)
            self.assertTrue(evidence["acceptance"]["passed"])
            self.assertEqual(evidence["acceptance"]["failed_checks"], [])
            self.assertEqual(
                evidence["weight_evidence"]["materialization"]["tensor_count"],
                21,
            )
            self.assertEqual(
                evidence["weight_evidence"]["smoke_tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            self.assertEqual(
                evidence["runtime_evidence"]["profile_decode_step"][
                    "trace_status"
                ],
                "captured_and_executed",
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
                evidence["device_evidence"]["profile_ttnn_environment"][
                    "version"
                ],
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
                ["profile_decode_step.tokens_per_second_per_user"],
            )
            evidence = json.loads(
                (out_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertEqual(evidence["status"], "incomplete")
            self.assertEqual(
                evidence["acceptance"]["failed_checks"],
                ["profile_decode_step.tokens_per_second_per_user"],
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
                ["decode_shell.numeric_reference_status"],
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


if __name__ == "__main__":
    unittest.main()
