from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.validation import (
    VALIDATION_STEPS,
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
            self.assertTrue((out_dir / "package" / "manifest.json").is_file())
            self.assertTrue((out_dir / "search_report.json").is_file())

            diff = json.loads((out_dir / "plan_diff.json").read_text())
            self.assertEqual(diff["missing_ops"], [])
            self.assertEqual(diff["extra_ops"], [])
            self.assertEqual(diff["order_mismatch"], [])

            search_report = json.loads((out_dir / "search_report.json").read_text())
            self.assertTrue(search_report["dry_run"])
            self.assertEqual(search_report["candidate_count"], 10)

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
