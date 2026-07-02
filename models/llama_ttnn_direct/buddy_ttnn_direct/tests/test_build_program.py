from __future__ import annotations

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
            validate_dir = root / "real_decode_validation"
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

            validation = subprocess.run(
                [
                    sys.executable,
                    str(out_dir / "run_decode.py"),
                    "--mode",
                    "validate-real",
                    "--dry-run",
                    "--skip-autotune",
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
            self.assertEqual(
                json.loads(validation_report.read_text())["command"],
                "validate-real-decode",
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
