from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_diff import (
    build_config_parity_view,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.program import (
    PROGRAM_ARTIFACTS,
)


class BuildProgramTest(unittest.TestCase):
    def test_cli_build_program_writes_decode_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

            exit_code = main(
                [
                    "build-program",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                sorted(path.name for path in out_dir.iterdir()),
                sorted(PROGRAM_ARTIFACTS),
            )
            py_compile.compile(str(out_dir / "model.py"), doraise=True)
            py_compile.compile(str(out_dir / "run_decode.py"), doraise=True)

            generated_config = json.loads((out_dir / "config.json").read_text())
            self.assertEqual(generated_config["num_layers"], 2)
            self.assertEqual(generated_config["hidden_size"], 16)
            self.assertEqual(generated_config["intermediate_size"], 32)
            self.assertEqual(
                generated_config["kv_cache"]["template"],
                "paged_kv_cache",
            )
            self.assertEqual(generated_config["kv_cache"]["policy"], "paged")
            self.assertEqual(generated_config["kv_cache"]["page_block_size"], 32)
            self.assertEqual(generated_config["kv_cache"]["max_cache_len"], 1024)
            self.assertEqual(generated_config["kv_cache"]["num_kv_heads"], 2)
            self.assertEqual(generated_config["kv_cache"]["head_dim"], 4)

            semantic = json.loads((out_dir / "semantic_graph.json").read_text())
            self.assertEqual(semantic["model_name"], "fake-build-program")
            self.assertEqual(semantic["num_layers"], 2)

            plan = json.loads((out_dir / "execution_plan.json").read_text())
            self.assertEqual(plan["final"][-1], "device_argmax_greedy")

            weights_manifest = json.loads(
                (out_dir / "weights_manifest.json").read_text()
            )
            self.assertEqual(weights_manifest["model_name"], "fake-build-program")
            self.assertFalse(
                weights_manifest["metadata_policy"]["loads_tensor_payloads"]
            )
            self.assertIn(
                "model.layers.0.self_attn.q_proj.weight",
                weights_manifest["weights"],
            )

            source = (out_dir / "model.py").read_text()
            self.assertIn(
                "templates import ttnn_ops",
                source,
            )
            self.assertIn(
                "ttnn_ops.paged_sdpa_decode",
                source,
            )
            self.assertIn(
                "ttnn_ops.paged_update_cache",
                source,
            )

    def test_generated_run_decode_dry_run_prints_per_layer_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
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
                        str(out_dir),
                    ]
                ),
                0,
            )

            result = subprocess.run(
                [sys.executable, str(out_dir / "run_decode.py"), "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["num_layers"], 2)
            self.assertEqual(len(payload["layers"]), 2)
            self.assertEqual(payload["layers"][0]["layer_id"], 0)
            self.assertEqual(
                payload["layers"][0]["ops"],
                [
                    "rmsnorm.attn",
                    "linear.qkv_packed",
                    "nlp_create_qkv_heads_decode",
                    "rotary_embedding_decode",
                    "paged_update_cache",
                    "paged_scaled_dot_product_attention_decode",
                    "nlp_concat_heads_decode",
                    "linear.o_proj",
                    "residual_add",
                    "rmsnorm.mlp",
                    "linear.mlp_gate",
                    "linear.mlp_up",
                    "mul.silu",
                    "linear.mlp_down",
                    "residual_add",
                ],
            )
            self.assertEqual(
                payload["final_ops"],
                ["rmsnorm.final", "split_lm_head", "argmax_or_sampling"],
            )

    def test_generated_run_decode_wraps_smoke_profile_and_real_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
            smoke_report = root / "decode_step_smoke_report.json"
            profile_report = root / "decode_step_profile_report.json"
            loop_report = root / "prompt_decode_loop_report.json"
            preflight_dir = root / "real_decode_preflight"
            validate_dir = root / "real_decode_validation"
            official_json = root / "official_parity_config.json"
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
                        str(out_dir),
                    ]
                ),
                0,
            )
            _write_official_parity_from_program(out_dir, official_json)
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"")
            (out_dir / "ttnn.py").write_text(
                "\n".join(
                    [
                        '__version__ = "fake-ttnn"',
                        '__tt_metal_commit__ = "fake-tt-metal"',
                        "",
                    ]
                )
            )
            program_readme = (out_dir / "README.md").read_text()
            self.assertIn("--preflight-only", program_readme)
            self.assertIn("--min-tokens-per-second-per-user 1.0", program_readme)
            self.assertIn("--metric tokens_per_second_per_user", program_readme)
            self.assertIn("--decode-shell-pcc-threshold 0.99", program_readme)
            self.assertIn("--mode decode-loop", program_readme)
            self.assertIn("--max-new-tokens 2", program_readme)
            self.assertIn("--require-model-end-to-end", program_readme)
            self.assertIn('--prompt "Hello from TTNN Direct"', program_readme)

            smoke = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "smoke",
                    "--dry-run",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out",
                    str(smoke_report),
                ],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            smoke_summary = json.loads(smoke.stdout)
            self.assertEqual(smoke_summary["status"], "dry_run")
            self.assertEqual(smoke_summary["report"], str(smoke_report))
            self.assertEqual(
                json.loads(smoke_report.read_text())["template"],
                "generated_decode_step",
            )

            profile = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "profile",
                    "--dry-run",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out",
                    str(profile_report),
                ],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            profile_summary = json.loads(profile.stdout)
            self.assertEqual(profile_summary["status"], "dry_run")
            self.assertEqual(profile_summary["report"], str(profile_report))
            self.assertEqual(
                json.loads(profile_report.read_text())["template"],
                "generated_decode_step_profile",
            )

            loop = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "decode-loop",
                    "--dry-run",
                    "--max-new-tokens",
                    "3",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out",
                    str(loop_report),
                ],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            loop_summary = json.loads(loop.stdout)
            self.assertEqual(loop_summary["status"], "dry_run")
            self.assertEqual(loop_summary["report"], str(loop_report))
            self.assertEqual(
                json.loads(loop_report.read_text())["template"],
                "prompt_decode_loop",
            )
            loop_payload = json.loads(loop_report.read_text())
            self.assertEqual(loop_payload["decode_steps"], 3)
            self.assertEqual(loop_payload["max_new_tokens"], 3)
            self.assertEqual(loop_payload["prefill_status"], "not_run")
            self.assertEqual(
                loop_payload["kv_cache_source"],
                "empty_initialized",
            )
            self.assertEqual(loop_payload["generated_text_status"], "not_run")

            preflight = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "validate-real",
                    "--model-path",
                    str(model_dir),
                    "--official-config",
                    str(official_json),
                    "--prompt",
                    "hello tenstorrent",
                    "--tokenizer-path",
                    str(model_dir),
                    "--require-model-end-to-end",
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
                    "--layers",
                    "2",
                    "--batch-size",
                    "32",
                    "--cache-len",
                    "1024",
                    "--preflight-only",
                    "--out-dir",
                    str(preflight_dir),
                ],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            preflight_summary = json.loads(preflight.stdout)
            self.assertEqual(preflight_summary["status"], "pass")
            preflight_report = (
                preflight_dir / "real_decode_preflight_report.json"
            )
            self.assertEqual(preflight_summary["report"], str(preflight_report))
            preflight_payload = json.loads(preflight_report.read_text())
            self.assertEqual(preflight_payload["status"], "pass")
            self.assertEqual(
                preflight_payload["metric"],
                "tokens_per_second_per_user",
            )
            self.assertTrue(preflight_payload["prompt_runtime_requested"])
            self.assertEqual(
                preflight_payload["effective_tokenizer_path"],
                str(model_dir),
            )
            self.assertEqual(
                preflight_payload["min_tokens_per_second_per_user"],
                1.25,
            )
            self.assertEqual(
                preflight_payload["decode_shell_pcc_threshold"],
                0.98,
            )
            self.assertEqual(
                preflight_payload["ttnn_environment"]["version"],
                "fake-ttnn",
            )
            self.assertIn(
                "--prompt",
                preflight_payload["reproducibility"][
                    "final_validation_cli_args"
                ],
            )
            self.assertIn(
                "hello tenstorrent",
                preflight_payload["reproducibility"][
                    "final_validation_cli_args"
                ],
            )
            self.assertIn(
                "--tokenizer-path",
                preflight_payload["reproducibility"][
                    "final_validation_cli_args"
                ],
            )

            validation = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "validate-real",
                    "--dry-run",
                    "--require-official-performance-parity",
                    "--metric",
                    "tokens_per_second_per_user",
                    "--min-tokens-per-second-per-user",
                    "1.0",
                    "--baseline-reference",
                    "tt_metal_official_llama31_8b_b32",
                    "--min-baseline-ratio",
                    "0.1",
                    "--decode-shell-pcc-threshold",
                    "0.5",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out-dir",
                    str(validate_dir),
                ],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            validation_summary = json.loads(validation.stdout)
            self.assertEqual(validation_summary["status"], "dry_run")
            validation_report = validate_dir / "real_decode_validation_report.json"
            self.assertEqual(validation_summary["report"], str(validation_report))
            validation_payload = json.loads(validation_report.read_text())
            self.assertEqual(validation_payload["command"], "validate-real-decode")
            self.assertEqual(
                validation_payload["metric"],
                "tokens_per_second_per_user",
            )
            self.assertTrue(validation_payload["require_full_decode_step"])
            self.assertTrue(
                validation_payload["require_official_performance_parity"]
            )
            self.assertTrue(validation_payload["require_model_end_to_end"])
            self.assertTrue(validation_payload["require_trace"])
            self.assertTrue(
                validation_payload["require_official_config_match"]
            )
            self.assertTrue(validation_payload["require_full_depth"])
            self.assertTrue(
                validation_payload["require_program_runtime_shape"]
            )
            self.assertTrue(
                validation_payload["require_batch32_decode_step"]
            )
            self.assertEqual(
                validation_payload["min_tokens_per_second_per_user"],
                1.0,
            )
            self.assertEqual(
                validation_payload["baseline_tokens_per_second_per_user"],
                33.1,
            )
            self.assertEqual(
                validation_payload["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            self.assertEqual(
                validation_payload["baseline_reference_entry"]["model"],
                "Llama 3.1 8B",
            )
            self.assertEqual(validation_payload["min_baseline_ratio"], 0.1)
            self.assertEqual(
                validation_payload["decode_shell_pcc_threshold"],
                0.5,
            )
            self.assertTrue(
                validation_payload[
                    "require_decode_shell_numeric_reference"
                ]
            )
            self.assertEqual(validation_payload["acceptance"]["status"], "dry_run")
            self.assertTrue(validation_payload["acceptance"]["require_trace"])
            self.assertTrue(
                validation_payload["acceptance"][
                    "require_batch32_decode_step"
                ]
            )
            self.assertTrue(
                validation_payload["acceptance"]["require_full_decode_step"]
            )
            self.assertTrue(
                validation_payload["acceptance"][
                    "require_model_end_to_end"
                ]
            )
            self.assertTrue(
                validation_payload["acceptance"][
                    "require_official_performance_parity"
                ]
            )
            evidence = json.loads(
                (validate_dir / "real_decode_evidence_manifest.json").read_text()
            )
            self.assertTrue(evidence["requirements"]["require_full_decode_step"])
            self.assertTrue(evidence["requirements"]["require_model_end_to_end"])
            self.assertTrue(
                evidence["requirements"][
                    "require_official_performance_parity"
                ]
            )
            self.assertTrue(
                evidence["requirements"]["require_official_config_match"]
            )
            self.assertTrue(evidence["requirements"]["require_full_depth"])
            self.assertTrue(
                evidence["requirements"]["require_program_runtime_shape"]
            )
            self.assertTrue(
                evidence["requirements"]["require_batch32_decode_step"]
            )
            self.assertEqual(
                evidence["requirements"]["baseline_tokens_per_second_per_user"],
                33.1,
            )
            self.assertEqual(
                evidence["requirements"]["baseline_reference"],
                "tt_metal_official_llama31_8b_b32",
            )
            self.assertEqual(evidence["requirements"]["min_baseline_ratio"], 0.1)


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-build-program",
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
