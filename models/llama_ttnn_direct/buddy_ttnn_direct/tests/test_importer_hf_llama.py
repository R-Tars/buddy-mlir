from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import (
    load_graph_json,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    TensorMetadata,
    import_hf_llama,
)


def _fake_config(num_layers: int = 2) -> dict[str, object]:
    return {
        "_name_or_path": "fake-llama",
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


class LlamaSemanticImporterTest(unittest.TestCase):
    def test_importer_builds_decode_graph_from_fake_metadata(self) -> None:
        state_metadata = {
            key: TensorMetadata(key=key, shape=(1,))
            for key in _fake_weight_keys()
        }

        graph = import_hf_llama(
            "/tmp/fake-llama",
            config=_fake_config(),
            state_dict_metadata=state_metadata,
            mode="decode",
            batch_size=32,
            seq_len=1,
            max_cache_len=1024,
        )

        self.assertEqual(graph.model_name, "fake-llama")
        self.assertEqual(graph.num_layers, 2)
        self.assertEqual(graph.hidden_size, 16)
        self.assertEqual(graph.intermediate_size, 32)
        self.assertEqual(graph.num_attention_heads, 4)
        self.assertEqual(graph.num_key_value_heads, 2)
        self.assertEqual(graph.head_dim, 4)
        self.assertEqual(graph.vocab_size, 128)
        self.assertEqual(graph.mode, "decode")
        self.assertEqual(graph.batch_size, 32)
        self.assertEqual(graph.seq_len, 1)
        self.assertEqual(len(graph.layers), 2)

        layer0 = graph.layers[0]
        self.assertEqual(
            layer0.attention.q_proj,
            "model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            layer0.attention.k_proj,
            "model.layers.0.self_attn.k_proj.weight",
        )
        self.assertEqual(
            layer0.attention.v_proj,
            "model.layers.0.self_attn.v_proj.weight",
        )
        self.assertEqual(
            layer0.attention.o_proj,
            "model.layers.0.self_attn.o_proj.weight",
        )
        self.assertEqual(
            layer0.mlp.gate_proj,
            "model.layers.0.mlp.gate_proj.weight",
        )
        self.assertEqual(layer0.mlp.activation, "silu")
        self.assertTrue(layer0.attention.uses_paged_kv_cache)

    def test_cli_import_llama_dumps_json_from_fake_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "tiny-llama"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(_fake_config(num_layers=2))
            )
            weight_map = {
                key: "model-00001-of-00001.safetensors"
                for key in _fake_weight_keys(num_layers=2)
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map})
            )
            out = Path(tmpdir) / "llama_semantic.json"

            exit_code = main(
                [
                    "import-llama",
                    "--model-path",
                    str(model_dir),
                    "--mode",
                    "decode",
                    "--batch-size",
                    "32",
                    "--seq-len",
                    "1",
                    "--max-cache-len",
                    "1024",
                    "--dry-run",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            dumped = json.loads(out.read_text())
            self.assertEqual(dumped["num_layers"], 2)
            self.assertEqual(dumped["hidden_size"], 16)
            self.assertEqual(dumped["num_attention_heads"], 4)
            self.assertEqual(dumped["num_key_value_heads"], 2)
            self.assertEqual(dumped["head_dim"], 4)
            self.assertEqual(dumped["mode"], "decode")
            self.assertEqual(dumped["batch_size"], 32)
            self.assertEqual(dumped["seq_len"], 1)
            self.assertEqual(len(dumped["layers"]), 2)
            self.assertEqual(
                dumped["layers"][1]["mlp"]["down_proj"],
                "model.layers.1.mlp.down_proj.weight",
            )

            graph = load_graph_json(out)
            self.assertEqual(graph.lm_head.weight, "lm_head.weight")


if __name__ == "__main__":
    unittest.main()
