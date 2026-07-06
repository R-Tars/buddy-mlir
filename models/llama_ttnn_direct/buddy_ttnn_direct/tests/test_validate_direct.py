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
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_diff import (
    PARITY_SECTIONS,
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
    def test_performance_baseline_reference_resolves(self) -> None:
        baseline = validation_module.resolve_performance_baseline(
            "tt_metal_official_llama31_8b_b32"
        )
        self.assertEqual(baseline["model"], "Llama 3.1 8B")
        self.assertEqual(baseline["batch_size"], 32)
        self.assertEqual(
            baseline["decode_tokens_per_second_per_user"],
            33.1,
        )
        self.assertTrue(baseline["baseline_file"].endswith(
            "performance_baselines.json"
        ))

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
            self.assertEqual(
                evidence["runtime_evidence"]["decode_depth_sweep"]["depths"],
                [1],
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
                report["results"],
                {step: "pass" for step in REAL_DECODE_VALIDATION_STEPS},
            )
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
                "single_layer_decode.output_shapes",
                acceptance_check_names,
            )
            self.assertIn(
                "single_layer_decode.observed_op_sequence",
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
