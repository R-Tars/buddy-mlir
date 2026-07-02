from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import (
    dump_graph_json,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
)


def _fake_graph(num_layers: int = 3):
    config = {
        "_name_or_path": "fake-llama-plan",
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
    }
    return import_hf_llama(
        "/tmp/fake-llama-plan",
        config=config,
        state_dict_metadata=[],
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )


def _seed_config() -> dict[str, object]:
    return {
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


class TemplateRegistryTest(unittest.TestCase):
    def test_build_execution_plan_is_deterministic(self) -> None:
        graph = _fake_graph(num_layers=3)
        plan = build_execution_plan(graph, _seed_config())

        expected_layer_templates = [
            "rmsnorm",
            "official_paged_attention_decode",
            "residual_add",
            "rmsnorm",
            "official_gated_mlp_decode",
            "residual_add",
        ]
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["mode"], "decode")
        self.assertEqual(plan["vocab_size"], 128)
        self.assertEqual(plan["hidden_size"], 16)
        self.assertEqual(plan["intermediate_size"], 32)
        self.assertEqual(len(plan["layers"]), 3)
        self.assertEqual(
            plan["layers"][0]["templates"], expected_layer_templates
        )
        self.assertEqual(
            plan["layers"][2]["templates"], expected_layer_templates
        )
        self.assertEqual(
            plan["final"],
            [
                "rmsnorm",
                "official_split_lm_head",
                "device_argmax_greedy",
            ],
        )
        self.assertEqual(
            plan["template_config"]["lm_head_split_count"], 8
        )

    def test_cli_plan_dumps_execution_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_json = root / "semantic.json"
            config_json = root / "config.json"
            plan_json = root / "plan.json"
            dump_graph_json(_fake_graph(num_layers=2), semantic_json)
            config_json.write_text(json.dumps(_seed_config()))

            exit_code = main(
                [
                    "plan",
                    "--semantic-json",
                    str(semantic_json),
                    "--config",
                    str(config_json),
                    "--out",
                    str(plan_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            dumped = json.loads(plan_json.read_text())
            self.assertEqual(len(dumped["layers"]), 2)
            self.assertEqual(
                dumped["layers"][1]["templates"][1],
                "official_paged_attention_decode",
            )
            self.assertEqual(
                dumped["final"],
                [
                    "rmsnorm",
                    "official_split_lm_head",
                    "device_argmax_greedy",
                ],
            )

    def test_plan_rejects_config_shape_mismatch(self) -> None:
        bad_config = _seed_config()
        bad_config["batch_size"] = 16

        with self.assertRaisesRegex(ValueError, "batch_size mismatch"):
            build_execution_plan(_fake_graph(), bad_config)


if __name__ == "__main__":
    unittest.main()
