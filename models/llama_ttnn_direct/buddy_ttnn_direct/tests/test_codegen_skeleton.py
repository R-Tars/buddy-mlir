from __future__ import annotations

import json
import py_compile
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    dump_execution_plan,
)


def _fake_plan(num_layers: int = 2) -> dict[str, object]:
    graph = import_hf_llama(
        "/tmp/fake-llama-codegen",
        config={
            "_name_or_path": "fake-llama-codegen",
            "model_type": "llama",
            "num_hidden_layers": num_layers,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        },
        state_dict_metadata=[],
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )
    return build_execution_plan(
        graph,
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
        },
    )


class PythonTTNNSkeletonCodegenTest(unittest.TestCase):
    def test_codegen_python_writes_skeleton_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            plan = _fake_plan(num_layers=2)
            dump_execution_plan(plan, plan_json)

            exit_code = main(
                [
                    "codegen-python",
                    "--plan-json",
                    str(plan_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "model.py").is_file())
            self.assertTrue((out_dir / "config.json").is_file())
            self.assertTrue((out_dir / "plan.json").is_file())
            self.assertTrue((out_dir / "README.md").is_file())

            source = (out_dir / "model.py").read_text()
            self.assertIn("import ttnn", source)
            self.assertIn("class BuddyLlama31TTNN", source)
            self.assertIn("def decode_step", source)
            self.assertIn("def decode_layer", source)
            self.assertIn("ttnn.add(residual, hidden)", source)
            self.assertIn(
                "TODO: ttnn.experimental.nlp_create_qkv_heads_decode",
                source,
            )
            self.assertIn(
                "TODO: paged_scaled_dot_product_attention_decode",
                source,
            )
            self.assertIn("raise NotImplementedError", source)

            py_compile.compile(
                str(out_dir / "model.py"),
                doraise=True,
            )

            config = json.loads((out_dir / "config.json").read_text())
            self.assertEqual(config["num_layers"], 2)
            self.assertEqual(config["template_config"]["device"], "p150a")
            copied_plan = json.loads((out_dir / "plan.json").read_text())
            self.assertEqual(copied_plan["layers"], plan["layers"])

    def test_codegen_python_dry_run_does_not_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "dry-run-output"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)

            exit_code = main(
                [
                    "codegen-python",
                    "--plan-json",
                    str(plan_json),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
