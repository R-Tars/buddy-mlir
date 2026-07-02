from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_layer import (
    ATTENTION_LAYER_OPS,
    run_smoke_attention_layer,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
    _fake_ttnn,
)


class SmokeAttentionLayerTest(unittest.TestCase):
    def test_cli_smoke_attention_layer_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "attention_layer_report.json"
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
                    "smoke-attention-layer",
                    "--program-dir",
                    str(program_dir),
                    "--layer",
                    "0",
                    "--device",
                    "p150a",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["op_sequence"], ATTENTION_LAYER_OPS)
            self.assertEqual(
                report["expected_output_shapes"]["attention_output"],
                [2, 1, 16],
            )
            self.assertEqual(report["tensor_conversion_count"], 10)
            self.assertEqual(report["memory_config_conversion_count"], 1)
            self.assertEqual(len(report["primitive_reports"]), len(ATTENTION_LAYER_OPS))
            first = report["primitive_reports"][0]
            self.assertEqual(first["primitive"], "qkv_linear")
            self.assertEqual(first["input_shapes"]["hidden"], [2, 1, 16])
            self.assertEqual(first["input_shapes"]["qkv_weight"], [16, 32])
            self.assertEqual(first["dtype"], "bfloat16")
            self.assertEqual(first["layout"], "tile")
            self.assertEqual(first["memory_config"], "default_or_l1")
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_run_smoke_attention_layer_executes_fake_full_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "attention_layer_report.json"
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

            fake_ttnn = _fake_ttnn()
            report = run_smoke_attention_layer(
                out=report_json,
                program_dir=program_dir,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                [item["primitive"] for item in report["primitive_reports"]],
                ATTENTION_LAYER_OPS,
            )
            self.assertEqual(
                report["output_shapes"]["attention_output"],
                [2, 1, 16],
            )
            self.assertEqual(report["output_shapes"]["key_cache"], [2, 16, 2, 4])
            self.assertEqual(report["tensor_conversion_count"], 10)
            self.assertEqual(report["memory_config_conversion_count"], 1)
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(report["reference"]["kind"], "structural_shape")
            self.assertTrue(
                all(check["passed"] for check in report["reference"]["checks"])
            )
            primitive_reports = report["primitive_reports"]
            self.assertEqual(
                primitive_reports[2]["input_shapes"]["query"],
                [2, 4, 1, 4],
            )
            self.assertEqual(
                primitive_reports[5]["input_shapes"]["value_cache"],
                [2, 16, 2, 4],
            )
            self.assertEqual(primitive_reports[5]["dtype"], "bfloat16")
            self.assertEqual(
                primitive_reports[5]["memory_config"],
                "ttnn.L1_MEMORY_CONFIG",
            )
            self.assertEqual(json.loads(report_json.read_text()), report)

            called_ops = [call["op"] for call in fake_ttnn.calls]
            self.assertIn("nlp_create_qkv_heads_decode", called_ops)
            self.assertIn("rotary_embedding_llama", called_ops)
            self.assertIn("paged_update_cache", called_ops)
            self.assertIn("paged_scaled_dot_product_attention_decode", called_ops)
            self.assertIn("nlp_concat_heads_decode", called_ops)

    def test_attention_layer_api_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "attention_layer_report.json"
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

            report = run_smoke_attention_layer(
                out=report_json,
                program_dir=program_dir,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_fake_ttnn(with_transformer=False),
                torch_module=_fake_torch(),
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "api_mismatch")
            self.assertIn(
                "paged_scaled_dot_product_attention_decode",
                report["error"],
            )


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-attention-layer",
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
