from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import (
    dump_graph_json,
    load_graph_json,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_template_config,
)


DECODE_TEMPLATES = """
rmsnorm official_paged_attention_decode residual_add rmsnorm
official_gated_mlp_decode residual_add
""".split()
PREFILL_TEMPLATES = """
rmsnorm official_prefill_attention residual_add rmsnorm
official_gated_mlp_prefill residual_add
""".split()
FINAL_TEMPLATES = ["rmsnorm", "official_split_lm_head", "device_argmax_greedy"]


def _fake_graph(mode: str = "decode", num_layers: int = 3):
    seq_len = 1 if mode == "decode" else 128
    return import_hf_llama(
        f"/tmp/fake-llama-{mode}-plan",
        config={
            "_name_or_path": f"fake-llama-{mode}-plan",
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
        mode=mode,
        batch_size=32,
        seq_len=seq_len,
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
        plan = build_execution_plan(_fake_graph(num_layers=3), _seed_config())
        fields = ("schema_version", "mode", "vocab_size", "hidden_size", "intermediate_size")
        expected = (1, "decode", 128, 16, 32)
        self.assertEqual(tuple(plan[key] for key in fields), expected)
        self.assertEqual(len(plan["layers"]), 3)
        self.assertEqual(plan["layers"][0]["templates"], DECODE_TEMPLATES)
        self.assertEqual(plan["layers"][2]["templates"], DECODE_TEMPLATES)
        self.assertEqual(plan["final"], FINAL_TEMPLATES)
        self.assertEqual(plan["template_config"]["lm_head_split_count"], 8)

    def test_build_execution_plan_supports_prefill_graph(self) -> None:
        plan = build_execution_plan(_fake_graph("prefill", 2), _seed_config())
        self.assertEqual((plan["mode"], plan["seq_len"]), ("prefill", 128))
        self.assertEqual(plan["layers"][0]["templates"], PREFILL_TEMPLATES)
        self.assertEqual(
            plan["template_config"]["prefill_attention_template"],
            "official_prefill_attention",
        )

    def test_plan_dumps_execution_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic, config, output = (
                root / "semantic.json",
                root / "config.json",
                root / "plan.json",
            )
            dump_graph_json(_fake_graph(num_layers=2), semantic)
            config.write_text(json.dumps(_seed_config()))
            plan = build_execution_plan(
                load_graph_json(semantic), load_template_config(config)
            )
            dump_execution_plan(plan, output)
            dumped = json.loads(output.read_text())
        self.assertEqual(len(dumped["layers"]), 2)
        self.assertEqual(dumped["layers"][1]["templates"][1], DECODE_TEMPLATES[1])
        self.assertEqual(dumped["final"], FINAL_TEMPLATES)

    def test_plan_rejects_config_shape_mismatch(self) -> None:
        config = _seed_config()
        config["batch_size"] = 16
        with self.assertRaisesRegex(ValueError, "batch_size mismatch"):
            build_execution_plan(_fake_graph(), config)

    def test_plan_rejects_unimplemented_custom_fused_templates(self) -> None:
        cases = {
            "mlp_template": "custom_buddy_fused_mlp_decode",
            "lm_head_template": "custom_buddy_lmhead_argmax_decode",
        }
        for key, template in cases.items():
            with self.subTest(key=key):
                config = {**_seed_config(), key: template}
                with self.assertRaisesRegex(ValueError, f"{key} must be one of"):
                    build_execution_plan(_fake_graph(num_layers=2), config)

    def test_future_custom_ops_document_is_not_a_registry_hook(self) -> None:
        path = Path(__file__).parents[4] / "models/llama_ttnn_direct/docs/future/custom_ops.md"
        readme = path.read_text()
        for marker in (
            "buddy_fused_mlp_decode",
            "buddy_lmhead_argmax_decode",
            "planner deliberately rejects",
        ):
            self.assertIn(marker, readme)


if __name__ == "__main__":
    unittest.main()
