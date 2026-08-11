from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.artifacts import OFFLINE_ARTIFACT_MANIFESTS, prepare_offline_artifacts
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_emit import dump_parameter_config, emit_parameter_config
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import dump_graph_json
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import import_hf_llama
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
    _fake_weight_specs as _all_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import _write_fake_model_config


def _specs(layers: int = 2, *, lm_head: bool = True) -> dict[str, dict[str, object]]:
    result = {}
    for key, value in _all_weight_specs().items():
        if not lm_head and key == "lm_head.weight":
            continue
        if "model.layers." in key and int(key.split(".")[2]) >= layers:
            continue
        result[key] = value
    return result


class PrepareArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model, self.out = self.root / "model", self.root / "artifacts"

    def graph(self, layers: int, *, tied: bool = False, lm_head: bool = True):
        _write_fake_model_config(self.model)
        config = json.loads((self.model / "config.json").read_text())
        config.update(_name_or_path="fake-llama-artifacts", num_hidden_layers=layers, tie_word_embeddings=tied)
        (self.model / "config.json").write_text(json.dumps(config))
        specs = _specs(layers, lm_head=lm_head)
        _write_fake_model_weights(self.model, specs)
        return import_hf_llama(self.model, config=config, state_dict_metadata=list(specs), mode="decode", batch_size=32, seq_len=1, max_cache_len=1024)

    def test_prepare_offline_artifacts_writes_all_manifests(self) -> None:
        graph = self.graph(2)
        paths = prepare_offline_artifacts(self.model, graph, emit_parameter_config(graph, lm_head_split_count=2, kv_page_block_size=16), self.out)
        self.assertEqual(set(paths), set(OFFLINE_ARTIFACT_MANIFESTS))
        self.assertTrue(all(path.is_file() for path in paths.values()))
        weights = _load(self.out / "weights_manifest.json")
        self.assertFalse(weights["metadata_policy"]["loads_tensor_payloads"])
        self.assertEqual(weights["weight_count"], 21)
        q_proj = weights["weights"]["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual({key: q_proj[key] for key in ("role", "shape", "dtype", "filename")}, {"role": "q_proj", "shape": [16, 16], "dtype": "F32", "filename": "model-00001-of-00001.safetensors"})
        qkv = _load(self.out / "packed_qkv_manifest.json")
        self.assertEqual((len(qkv["layers"]), [item["complete"] for item in qkv["layers"]], [item["role"] for item in qkv["layers"][0]["source_weights"]]), (2, [True, True], ["q_proj", "k_proj", "v_proj"]))
        mlp = _load(self.out / "mlp_manifest.json")["layers"][1]
        self.assertEqual((mlp["gate_up_packing"], mlp["down_proj"]["key"]), ("gate_up_group", "model.layers.1.mlp.down_proj.weight"))
        lm_head = _load(self.out / "lm_head_splits_manifest.json")
        self.assertEqual((lm_head["split_count"], lm_head["tied_to_embedding"]), (2, False))
        self.assertEqual(lm_head["splits"], [
            {"shard_id": 0, "vocab_start": 0, "vocab_end": 64, "artifact_name": "lm_head_vocab_00.safetensors", "source_weight": "lm_head.weight"},
            {"shard_id": 1, "vocab_start": 64, "vocab_end": 128, "artifact_name": "lm_head_vocab_01.safetensors", "source_weight": "lm_head.weight"},
        ])
        kv = _load(self.out / "kv_cache_manifest.json")
        self.assertEqual((kv["policy"], kv["page_block_size"], kv["num_layers"], len(kv["layers"])), ("paged", 16, 2, 2))

    def test_prepare_artifacts_writes_manifest_directory(self) -> None:
        graph = self.graph(1)
        semantic, config_path = self.root / "semantic.json", self.root / "config.json"
        dump_graph_json(graph, semantic)
        dump_parameter_config(emit_parameter_config(graph, lm_head_split_count=1), config_path)
        paths = prepare_offline_artifacts(self.model, graph, json.loads(config_path.read_text()), self.out)
        self.assertEqual(set(paths), set(OFFLINE_ARTIFACT_MANIFESTS))
        self.assertTrue(all((self.out / name).is_file() for name in OFFLINE_ARTIFACT_MANIFESTS))
        layer = _load(self.out / "packed_qkv_manifest.json")["layers"]
        self.assertEqual((len(layer), layer[0]["complete"]), (1, True))

    def test_prepare_artifacts_records_tied_embedding_lm_head(self) -> None:
        graph = self.graph(1, tied=True, lm_head=False)
        prepare_offline_artifacts(self.model, graph, emit_parameter_config(graph, lm_head_split_count=4), self.out)
        shared = _load(self.out / "weights_manifest.json")["weights"]["model.embed_tokens.weight"]
        self.assertEqual((shared["role"], shared["shared_roles"], shared["tied_to_embedding"]), ("shared", ["embedding", "lm_head"], True))
        lm_head = _load(self.out / "lm_head_splits_manifest.json")
        self.assertEqual((lm_head["tied_to_embedding"], lm_head["weight"], lm_head["split_count"]), (True, "model.embed_tokens.weight", 4))


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())
