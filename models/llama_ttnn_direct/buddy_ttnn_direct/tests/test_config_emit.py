from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_emit import (
    emit_parameter_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import (
    dump_graph_json,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)


def _fake_config(num_layers: int = 2) -> dict[str, object]:
    return {
        "_name_or_path": "fake-llama-config-emit",
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


def _fake_weight_keys(num_layers: int = 2) -> list[str]:
    keys = ["model.embed_tokens.weight"]
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        keys.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    keys.extend(["model.norm.weight", "lm_head.weight"])
    return keys


def _fake_graph(num_layers: int = 2):
    return import_hf_llama(
        "/tmp/fake-llama-config-emit",
        config=_fake_config(num_layers=num_layers),
        state_dict_metadata=_fake_weight_keys(num_layers=num_layers),
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )


class ParameterConfigEmitTest(unittest.TestCase):
    def test_emit_parameter_config_classifies_weights(self) -> None:
        config = emit_parameter_config(_fake_graph(num_layers=2))
        weights = config["weights"]

        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["recipe"], "official_like_performance_seed")
        self.assertEqual(len(weights), 21)

        self.assertEqual(
            weights["model.embed_tokens.weight"]["role"], "embedding"
        )
        self.assertEqual(
            weights["model.embed_tokens.weight"]["target_dtype"], "bfloat16"
        )
        self.assertEqual(
            weights["model.embed_tokens.weight"]["layout"], "row_major"
        )

        q_proj = weights["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(q_proj["role"], "q_proj")
        self.assertEqual(q_proj["layer_id"], 0)
        self.assertEqual(q_proj["target_dtype"], "bfloat8_b")
        self.assertEqual(q_proj["packing"], "qkv_pack")
        self.assertEqual(q_proj["layout"], "tile")

        gate = weights["model.layers.1.mlp.gate_proj.weight"]
        self.assertEqual(gate["role"], "mlp_gate")
        self.assertEqual(gate["target_dtype"], "bfloat4_b")
        self.assertEqual(gate["packing"], "gate_up_group")

        down = weights["model.layers.1.mlp.down_proj.weight"]
        self.assertEqual(down["role"], "mlp_down")
        self.assertEqual(down["target_dtype"], "bfloat8_b")

        norm = weights["model.layers.0.input_layernorm.weight"]
        self.assertEqual(norm["role"], "input_norm")
        self.assertEqual(norm["target_dtype"], "bfloat16")
        self.assertEqual(norm["layout"], "row_major")

        lm_head = weights["lm_head.weight"]
        self.assertEqual(lm_head["role"], "lm_head")
        self.assertEqual(lm_head["target_dtype"], "bfloat8_b")
        self.assertEqual(lm_head["packing"], "vocab_split")
        self.assertFalse(lm_head["tied_to_embedding"])

        self.assertEqual(config["activations"]["target_dtype"], "bfloat16")
        self.assertEqual(config["lm_head"]["split_count"], 8)
        self.assertEqual(config["lm_head"]["split_axis"], "vocab")
        self.assertEqual(config["kv_cache"]["policy"], "paged")
        self.assertEqual(config["kv_cache"]["page_block_size"], 32)
        self.assertEqual(config["kv_cache"]["dtype"], "bfloat8_b")

    def test_emit_parameter_config_marks_tied_lm_head(self) -> None:
        tied_config = _fake_config(num_layers=1)
        tied_config["tie_word_embeddings"] = True
        graph = import_hf_llama(
            "/tmp/fake-llama-tied",
            config=tied_config,
            state_dict_metadata=["model.embed_tokens.weight"],
            mode="decode",
            batch_size=32,
            seq_len=1,
            max_cache_len=1024,
        )
        config = emit_parameter_config(graph, lm_head_split_count=4)

        shared = config["weights"]["model.embed_tokens.weight"]
        self.assertEqual(shared["role"], "shared")
        self.assertEqual(shared["shared_roles"], ["embedding", "lm_head"])
        self.assertTrue(config["lm_head"]["tied_to_embedding"])
        self.assertEqual(config["lm_head"]["split_count"], 4)

    def test_cli_emit_config_writes_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_json = root / "semantic.json"
            out = root / "param_config.json"
            dump_graph_json(_fake_graph(num_layers=1), semantic_json)

            exit_code = main(
                [
                    "emit-config",
                    "--semantic-json",
                    str(semantic_json),
                    "--lm-head-split-count",
                    "2",
                    "--kv-page-block-size",
                    "16",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            dumped = json.loads(out.read_text())
            self.assertEqual(dumped["lm_head"]["split_count"], 2)
            self.assertEqual(dumped["kv_cache"]["page_block_size"], 16)
            self.assertIn(
                "model.layers.0.self_attn.v_proj.weight",
                dumped["weights"],
            )

    def test_cli_emit_config_dry_run_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_json = root / "semantic.json"
            out = root / "param_config.json"
            dump_graph_json(_fake_graph(num_layers=1), semantic_json)

            exit_code = main(
                [
                    "emit-config",
                    "--semantic-json",
                    str(semantic_json),
                    "--out",
                    str(out),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
