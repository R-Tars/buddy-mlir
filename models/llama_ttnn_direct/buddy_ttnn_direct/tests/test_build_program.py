from __future__ import annotations

import inspect
import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.program import (
    PROGRAM_ARTIFACTS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import TTNNCompatOps


class BuildProgramTest(unittest.TestCase):
    def test_correctness_recipe_reaches_generated_kv_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            template = json.loads(config_json.read_text())
            template["dtype_recipe"] = "all_bf16_correctness"
            config_json.write_text(json.dumps(template))

            self.assertEqual(
                main(
                    [
                        "build",
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

            generated = json.loads((out_dir / "config.json").read_text())
            manifest = json.loads(
                (out_dir / "weights_manifest.json").read_text()
            )
            self.assertEqual(generated["kv_cache"]["dtype"], "bfloat16")
            self.assertEqual(
                manifest["config_summary"]["recipe"],
                "all_bf16_correctness",
            )

    def test_correctness_recipe_overrides_official_runtime_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            template = json.loads(config_json.read_text())
            template["dtype_recipe"] = "all_bf16_correctness"
            template["official_config_profile"] = "p150a_llama31_8b_b32_performance"
            config_json.write_text(json.dumps(template))

            self.assertEqual(
                main(
                    [
                        "build",
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

            generated = json.loads((out_dir / "config.json").read_text())
            runtime_dtype_paths = (
                ("attention", "qkv_output_dtype"),
                ("attention", "o_proj_output_dtype"),
                ("mlp", "intermediate_dtype"),
                ("mlp", "output_dtype"),
                ("lm_head", "output_dtype"),
            )
            for section, name in runtime_dtype_paths:
                self.assertEqual(
                    generated[section][name]["name"],
                    "bfloat16",
                )
            self.assertEqual(generated["kv_cache"]["dtype"], "bfloat16")

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
                    "build",
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
            self.assertEqual(
                generated_config["runtime_input_mode"],
                "persistent",
            )
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
            self.assertEqual(
                generated_config["lm_head"]["argmax_strategy"],
                "full_logits_untilize_multicore_argmax",
            )
            self.assertEqual(
                generated_config["rms_norm"]["input_memory_config"],
                "dram",
            )
            self.assertEqual(
                generated_config["rms_norm"]["output_memory_config"],
                "dram",
            )
            self.assertEqual(generated_config["rotary"]["theta"], 500000.0)
            self.assertIsNone(generated_config["rotary"]["scaling"])
            self.assertEqual(
                generated_config["rotary"]["max_position_embeddings"],
                1024,
            )

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
                "ttnn_compat.model_ops import",
                source,
            )
            self.assertIn(
                "self.ops.paged_sdpa_decode",
                source,
            )
            self.assertIn(
                "self.ops.paged_update_cache",
                source,
            )
            self.assertIn("def prefill_prompt", source)
            self.assertIn("valid_seq_len=None", source)
            self.assertIn("self.ops.select_sequence_position", source)
            self.assertIn(
                'op_name="reshape_prefill_selected_hidden"',
                source,
            )
            self.assertIn(
                'op_name="reshape_prefill_attention_for_residual"',
                source,
            )
            self.assertIn(
                'op_name="reshape_prefill_mlp_for_residual"',
                source,
            )
            self.assertIn("observer=None", source)
            self.assertIn('self._observe("prefill.final_hidden"', source)
            self.assertIn('self._observe(f"{stage}.logits"', source)
            self.assertIn("self.ops.local_argmax", source)
            self.assertIn("self.ops.global_argmax", source)
            self.assertIn("self.ops.force_argmax", source)
            self.assertIn("self.ops.scaled_dot_product_attention", source)
            self.assertIn("self.ops.fill_cache", source)
            self.assertIn("to_memory_config.{op_name}.input", source)

            compat_source = inspect.getsource(TTNNCompatOps)
            self.assertIn("ttnn_ops.paged_sdpa_decode", compat_source)
            self.assertIn("ttnn_ops.paged_update_cache", compat_source)
            self.assertIn(
                "ttnn_ops.scaled_dot_product_attention",
                compat_source,
            )
            self.assertIn("ttnn_ops.fill_cache", compat_source)
            self.assertIn("def normalize_decode_token", compat_source)
            self.assertIn("def resolve_memory_config", compat_source)
            self.assertIn("[0, shape[1] - 1]", compat_source)
            self.assertIn("[batch_size, shape[1]]", compat_source)

    def test_generated_run_decode_defaults_to_program_inspection(self) -> None:
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
                        "build",
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
            self.assertEqual(payload["command"], "inspect")
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["config"]["num_layers"], 2)
            self.assertEqual(len(payload["execution_plan"]["layers"]), 2)

    def test_generated_run_decode_forwards_product_and_legacy_modes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            out_dir = root / "program"
            smoke_report = root / "decode_step_smoke_report.json"
            profile_report = root / "generate_profile_report.json"
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
                        str(out_dir),
                    ]
                ),
                0,
            )

            runner = out_dir / "run_decode.py"
            source = runner.read_text()
            self.assertLessEqual(len(source.splitlines()), 180)
            for forbidden in (
                "from .smoke_",
                "import smoke_",
                "decode_loop",
                "profile_template",
                "legacy_validation",
            ):
                self.assertNotIn(forbidden, source)
            readme = (out_dir / "README.md").read_text()
            self.assertIn("python run_decode.py inspect", readme)
            self.assertIn("python run_decode.py generate", readme)
            self.assertIn("python run_decode.py profile", readme)
            self.assertIn("python run_decode.py validate", readme)
            self.assertIn("python run_decode.py diagnose", readme)

            inspect_result = subprocess.run(
                [sys.executable, str(runner), "--dry-run"],
                check=True,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            inspect_report = json.loads(inspect_result.stdout)
            self.assertEqual(inspect_report["command"], "inspect")
            self.assertTrue(inspect_report["passed"])
            self.assertEqual(inspect_report["config"]["num_layers"], 2)

            subprocess.run(
                [
                    sys.executable,
                    str(runner),
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
            smoke_payload = json.loads(smoke_report.read_text())
            self.assertEqual(smoke_payload["template"], "generated_decode_step")
            self.assertEqual(smoke_payload["status"], "dry_run")

            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "profile",
                    "--mode",
                    "generate",
                    "--dry-run",
                    "--max-new-tokens",
                    "3",
                    "--prefill-len",
                    "8",
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
            profile_payload = json.loads(profile_report.read_text())
            self.assertEqual(profile_payload["status"], "dry_run")
            self.assertEqual(profile_payload["mode"], "profile-generate")

            unknown = subprocess.run(
                [sys.executable, str(runner), "--mode", "removed-mode"],
                check=False,
                capture_output=True,
                cwd=out_dir,
                text=True,
            )
            self.assertEqual(unknown.returncode, 2)
            self.assertIn(
                "build, generate, profile, validate, inspect, diagnose",
                unknown.stderr,
            )



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


if __name__ == "__main__":
    unittest.main()
