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
                "Attention template official_paged_attention_decode",
                source,
            )
            self.assertIn(
                "requires TTNN op wrapper implementation",
                source,
            )
            self.assertIn(
                "paged_scaled_dot_product_attention_decode",
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
