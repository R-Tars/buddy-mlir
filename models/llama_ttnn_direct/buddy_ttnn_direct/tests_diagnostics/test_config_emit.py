from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_emit import CORRECTNESS_RECIPE, dump_parameter_config, emit_parameter_config, parameter_config_dry_run_report
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import dump_graph_json, load_graph_json
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import import_hf_llama


def _config(layers=2, *, tied=False):
    return {"_name_or_path": "fake-config-emit", "model_type": "llama", "num_hidden_layers": layers, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": tied}


def _graph(layers=2, *, tied=False):
    keys = ["model.embed_tokens.weight"]
    roles = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
    keys += [f"model.layers.{layer}.{role}" for layer in range(layers) for role in roles]
    keys += ["model.norm.weight"] + ([] if tied else ["lm_head.weight"])
    return import_hf_llama("/tmp/fake-config-emit", config=_config(layers, tied=tied), state_dict_metadata=keys, mode="decode", batch_size=32, seq_len=1, max_cache_len=1024)


class ParameterConfigEmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_correctness_recipe_uses_bfloat16_for_every_weight(self) -> None:
        config = emit_parameter_config(_graph(), recipe=CORRECTNESS_RECIPE)
        self.assertEqual(config["recipe"], CORRECTNESS_RECIPE)
        self.assertEqual({item["target_dtype"] for item in config["weights"].values()}, {"bfloat16"})
        self.assertEqual((config["lm_head"]["target_dtype"], config["kv_cache"]["dtype"]), ("bfloat16", "bfloat16"))

    def test_emit_parameter_config_classifies_weights(self) -> None:
        config = emit_parameter_config(_graph())
        weights = config["weights"]
        self.assertEqual((config["schema_version"], config["recipe"], len(weights)), (1, "official_like_performance_seed", 21))
        expected = {
            "model.embed_tokens.weight": ("embedding", "bfloat16", "none", "row_major", "dram"),
            "model.layers.0.self_attn.q_proj.weight": ("q_proj", "bfloat8_b", "qkv_pack", "tile", "dram"),
            "model.layers.1.mlp.gate_proj.weight": ("mlp_gate", "bfloat4_b", "gate_up_group", "tile", "dram"),
            "model.layers.1.mlp.down_proj.weight": ("mlp_down", "bfloat8_b", "none", "tile", "dram"),
            "model.layers.0.input_layernorm.weight": ("input_norm", "bfloat16", "none", "row_major", "dram"),
            "lm_head.weight": ("lm_head", "bfloat8_b", "vocab_split", "tile", "dram"),
        }
        for key, values in expected.items():
            item = weights[key]
            self.assertEqual(tuple(item.get(name) for name in ("role", "target_dtype", "packing", "layout", "memory_config")), values)
        self.assertEqual(weights["model.layers.0.self_attn.q_proj.weight"]["layer_id"], 0)
        self.assertFalse(weights["lm_head.weight"]["tied_to_embedding"])
        self.assertEqual(config["activations"]["target_dtype"], "bfloat16")
        lm_head = config["lm_head"]
        self.assertEqual((lm_head["split_count"], lm_head["split_axis"], len(lm_head["splits"]), lm_head["splits"][0], lm_head["splits"][-1]), (8, "vocab", 8, {"shard_id": 0, "vocab_start": 0, "vocab_end": 16}, {"shard_id": 7, "vocab_start": 112, "vocab_end": 128}))
        self.assertEqual(config["kv_cache"], {"policy": "paged", "page_block_size": 32, "dtype": "bfloat8_b"})

    def test_emit_parameter_config_applies_layer_dtype_override(self) -> None:
        weights = emit_parameter_config(_graph(), layer_dtype_overrides={1: {"mlp_intermediate": "bfloat8_b"}})["weights"]
        self.assertEqual([weights[f"model.layers.{layer}.mlp.{role}_proj.weight"]["target_dtype"] for layer, role in ((0, "gate"), (1, "gate"), (1, "up"), (1, "down"))], ["bfloat4_b", "bfloat8_b", "bfloat8_b", "bfloat8_b"])

    def test_emit_parameter_config_applies_weight_memory_override(self) -> None:
        descriptor = {"kind": "ttnn_dram_sharded_memory_config", "k": 4096, "n": 14336, "dram_grid_width": 8}
        weights = emit_parameter_config(_graph(1), weight_memory_overrides={"mlp_gate": descriptor})["weights"]
        self.assertEqual((weights["model.layers.0.mlp.gate_proj.weight"]["memory_config"], weights["model.layers.0.mlp.up_proj.weight"]["memory_config"]), (descriptor, "dram"))

    def test_emit_parameter_config_marks_tied_lm_head(self) -> None:
        config = emit_parameter_config(_graph(1, tied=True), lm_head_split_count=4)
        shared = config["weights"]["model.embed_tokens.weight"]
        self.assertEqual((shared["role"], shared["shared_roles"], config["lm_head"]["tied_to_embedding"], config["lm_head"]["split_count"]), ("shared", ["embedding", "lm_head"], True, 4))
        self.assertEqual(config["lm_head"]["splits"], [{"shard_id": index, "vocab_start": index * 32, "vocab_end": (index + 1) * 32} for index in range(4)])

    def test_emit_config_writes_metadata_json(self) -> None:
        semantic, out = self.root / "semantic.json", self.root / "config.json"
        dump_graph_json(_graph(1), semantic)
        dump_parameter_config(emit_parameter_config(load_graph_json(semantic), lm_head_split_count=2, kv_page_block_size=16), out)
        dumped = json.loads(out.read_text())
        self.assertEqual((dumped["lm_head"]["split_count"], dumped["lm_head"]["splits"], dumped["kv_cache"]["page_block_size"]), (2, [{"shard_id": 0, "vocab_start": 0, "vocab_end": 64}, {"shard_id": 1, "vocab_start": 64, "vocab_end": 128}], 16))
        self.assertIn("model.layers.0.self_attn.v_proj.weight", dumped["weights"])

    def test_emit_config_dry_run_does_not_write(self) -> None:
        semantic, out = self.root / "semantic.json", self.root / "config.json"
        dump_graph_json(_graph(1), semantic)
        self.assertTrue(parameter_config_dry_run_report(load_graph_json(semantic))["dry_run"])
        self.assertFalse(out.exists())
