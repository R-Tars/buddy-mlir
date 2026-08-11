from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.diff import (
    diff_plan_against_official,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    dump_execution_plan,
)


LAYER_OPS = """
rmsnorm.attn linear.qkv_packed nlp_create_qkv_heads_decode
rotary_embedding_decode paged_update_cache
paged_scaled_dot_product_attention_decode nlp_concat_heads_decode linear.o_proj
residual_add rmsnorm.mlp linear.mlp_gate linear.mlp_up mul.silu linear.mlp_down
residual_add
""".split()
FINAL_OPS = ["rmsnorm.final", "split_lm_head", "argmax_or_sampling"]
REFERENCE = {"layer_ops": LAYER_OPS, "final_ops": FINAL_OPS}
FULL_LOGITS_REFERENCE = {"layer_ops": LAYER_OPS, "final_ops": FINAL_OPS[:-1]}


def _fake_plan(generation_template: str = "device_argmax_greedy") -> dict[str, object]:
    graph = import_hf_llama(
        "/tmp/fake-plan-diff",
        config={
            "_name_or_path": "fake-plan-diff",
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
            "generation_template": generation_template,
            "lm_head_split_count": 8,
            "dtype_recipe": "official_like_performance_seed",
        },
    )


class PlanDiffTest(unittest.TestCase):
    def assertCleanDiff(self, diff: dict[str, object]) -> None:
        self.assertEqual(
            {key: diff[key] for key in ("missing_ops", "extra_ops", "order_mismatch")},
            {"missing_ops": [], "extra_ops": [], "order_mismatch": []},
        )

    def test_generated_plan_matches_reference_after_expansion(self) -> None:
        diff = diff_plan_against_official(_fake_plan(), REFERENCE)
        self.assertCleanDiff(diff)
        self.assertEqual(diff["expanded"], REFERENCE)

    def test_full_logits_plan_matches_reference_without_argmax(self) -> None:
        diff = diff_plan_against_official(_fake_plan("full_logits"), FULL_LOGITS_REFERENCE)
        self.assertCleanDiff(diff)
        self.assertEqual(diff["expanded"]["final_ops"], FINAL_OPS[:-1])

    def test_reports_missing_and_extra_ops(self) -> None:
        plan = _fake_plan()
        plan["layers"][0]["templates"] = [
            "rmsnorm",
            "official_gated_mlp_decode",
            "residual_add",
        ]
        diff = diff_plan_against_official(plan, REFERENCE)
        self.assertTrue({"linear.qkv_packed", "rmsnorm.mlp"} <= set(diff["missing_ops"]))
        self.assertEqual((diff["extra_ops"], diff["order_mismatch"]), ([], []))

    def test_reports_order_mismatch_when_op_multiset_matches(self) -> None:
        reordered = list(LAYER_OPS)
        reordered[10:12] = reversed(reordered[10:12])
        diff = diff_plan_against_official(
            _fake_plan(), {"layer_ops": reordered, "final_ops": FINAL_OPS}
        )
        self.assertEqual((diff["missing_ops"], diff["extra_ops"]), ([], []))
        self.assertEqual(
            diff["order_mismatch"],
            [
                {"index": 10, "ours": "linear.mlp_gate", "official": "linear.mlp_up"},
                {"index": 11, "ours": "linear.mlp_up", "official": "linear.mlp_gate"},
            ],
        )

    def test_cli_diff_plan_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = root / "program"
            program.mkdir()
            dump_execution_plan(_fake_plan(), program / "execution_plan.json")
            for artifact in (
                "config.json",
                "README.md",
                "model.py",
                "run_decode.py",
                "semantic_graph.json",
                "weights_manifest.json",
            ):
                (program / artifact).write_text("{}")
            reference = root / "official.json"
            output = root / "inspect.json"
            reference.write_text(json.dumps(REFERENCE))
            exit_code = main(
                [
                    "inspect",
                    "--program-dir",
                    str(program),
                    "--official-template",
                    str(reference),
                    "--out",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertCleanDiff(json.loads(output.read_text())["plan_diff"])


if __name__ == "__main__":
    unittest.main()
