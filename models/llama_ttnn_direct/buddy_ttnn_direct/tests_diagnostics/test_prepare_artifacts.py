from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.artifacts import (
    OFFLINE_ARTIFACT_MANIFESTS,
    prepare_offline_artifacts,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_emit import (
    dump_parameter_config,
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
        "_name_or_path": "fake-llama-artifacts",
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


def _fake_weight_specs(
    num_layers: int = 2, include_lm_head: bool = True
) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {"dtype": "F32", "shape": [128, 16]},
        "model.norm.weight": {"dtype": "F32", "shape": [16]},
    }
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        specs.update(
            {
                f"{prefix}.input_layernorm.weight": {
                    "dtype": "F32",
                    "shape": [16],
                },
                f"{prefix}.self_attn.q_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 16],
                },
                f"{prefix}.self_attn.k_proj.weight": {
                    "dtype": "F32",
                    "shape": [8, 16],
                },
                f"{prefix}.self_attn.v_proj.weight": {
                    "dtype": "F32",
                    "shape": [8, 16],
                },
                f"{prefix}.self_attn.o_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 16],
                },
                f"{prefix}.post_attention_layernorm.weight": {
                    "dtype": "F32",
                    "shape": [16],
                },
                f"{prefix}.mlp.gate_proj.weight": {
                    "dtype": "F32",
                    "shape": [32, 16],
                },
                f"{prefix}.mlp.up_proj.weight": {
                    "dtype": "F32",
                    "shape": [32, 16],
                },
                f"{prefix}.mlp.down_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 32],
                },
            }
        )
    if include_lm_head:
        specs["lm_head.weight"] = {"dtype": "F32", "shape": [128, 16]}
    return specs


class PrepareArtifactsTest(unittest.TestCase):
    def test_prepare_offline_artifacts_writes_all_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            out_dir = root / "artifacts"
            specs = _fake_weight_specs(num_layers=2)
            _write_fake_model(model_dir, specs)
            graph = import_hf_llama(
                model_dir,
                config=_fake_config(num_layers=2),
                state_dict_metadata=list(specs),
                mode="decode",
                batch_size=32,
                seq_len=1,
                max_cache_len=1024,
            )
            config = emit_parameter_config(
                graph,
                lm_head_split_count=2,
                kv_page_block_size=16,
            )

            paths = prepare_offline_artifacts(model_dir, graph, config, out_dir)

            self.assertEqual(set(paths), set(OFFLINE_ARTIFACT_MANIFESTS))
            for path in paths.values():
                self.assertTrue(path.is_file())

            weights = _load(out_dir / "weights_manifest.json")
            self.assertFalse(
                weights["metadata_policy"]["loads_tensor_payloads"]
            )
            self.assertEqual(weights["weight_count"], 21)
            q_proj = weights["weights"][
                "model.layers.0.self_attn.q_proj.weight"
            ]
            self.assertEqual(q_proj["role"], "q_proj")
            self.assertEqual(q_proj["shape"], [16, 16])
            self.assertEqual(q_proj["dtype"], "F32")
            self.assertEqual(
                q_proj["filename"], "model-00001-of-00001.safetensors"
            )

            qkv = _load(out_dir / "packed_qkv_manifest.json")
            self.assertEqual(len(qkv["layers"]), 2)
            self.assertTrue(all(layer["complete"] for layer in qkv["layers"]))
            self.assertEqual(
                [item["role"] for item in qkv["layers"][0]["source_weights"]],
                ["q_proj", "k_proj", "v_proj"],
            )

            mlp = _load(out_dir / "mlp_manifest.json")
            self.assertEqual(mlp["layers"][1]["gate_up_packing"], "gate_up_group")
            self.assertEqual(
                mlp["layers"][1]["down_proj"]["key"],
                "model.layers.1.mlp.down_proj.weight",
            )

            lm_head = _load(out_dir / "lm_head_splits_manifest.json")
            self.assertEqual(lm_head["split_count"], 2)
            self.assertFalse(lm_head["tied_to_embedding"])
            self.assertEqual(
                lm_head["splits"],
                [
                    {
                        "shard_id": 0,
                        "vocab_start": 0,
                        "vocab_end": 64,
                        "artifact_name": "lm_head_vocab_00.safetensors",
                        "source_weight": "lm_head.weight",
                    },
                    {
                        "shard_id": 1,
                        "vocab_start": 64,
                        "vocab_end": 128,
                        "artifact_name": "lm_head_vocab_01.safetensors",
                        "source_weight": "lm_head.weight",
                    },
                ],
            )

            kv_cache = _load(out_dir / "kv_cache_manifest.json")
            self.assertEqual(kv_cache["policy"], "paged")
            self.assertEqual(kv_cache["page_block_size"], 16)
            self.assertEqual(kv_cache["num_layers"], 2)
            self.assertEqual(len(kv_cache["layers"]), 2)

    def test_cli_prepare_artifacts_writes_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            semantic_json = root / "semantic.json"
            config_json = root / "parameter_config.json"
            out_dir = root / "artifacts"
            specs = _fake_weight_specs(num_layers=1)
            _write_fake_model(model_dir, specs)
            graph = import_hf_llama(
                model_dir,
                config=_fake_config(num_layers=1),
                state_dict_metadata=list(specs),
                mode="decode",
                batch_size=32,
                seq_len=1,
                max_cache_len=1024,
            )
            dump_graph_json(graph, semantic_json)
            dump_parameter_config(
                emit_parameter_config(graph, lm_head_split_count=1),
                config_json,
            )

            exit_code = main(
                [
                    "prepare-artifacts",
                    "--model-path",
                    str(model_dir),
                    "--semantic-json",
                    str(semantic_json),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in OFFLINE_ARTIFACT_MANIFESTS:
                self.assertTrue((out_dir / name).is_file())
            qkv = _load(out_dir / "packed_qkv_manifest.json")
            self.assertEqual(len(qkv["layers"]), 1)
            self.assertTrue(qkv["layers"][0]["complete"])

    def test_prepare_artifacts_records_tied_embedding_lm_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            specs = _fake_weight_specs(num_layers=1, include_lm_head=False)
            _write_fake_model(model_dir, specs)
            config_obj = _fake_config(num_layers=1)
            config_obj["tie_word_embeddings"] = True
            graph = import_hf_llama(
                model_dir,
                config=config_obj,
                state_dict_metadata=list(specs),
                mode="decode",
                batch_size=32,
                seq_len=1,
                max_cache_len=1024,
            )
            config = emit_parameter_config(graph, lm_head_split_count=4)

            prepare_offline_artifacts(model_dir, graph, config, root / "out")

            weights = _load(root / "out" / "weights_manifest.json")
            shared = weights["weights"]["model.embed_tokens.weight"]
            self.assertEqual(shared["role"], "shared")
            self.assertEqual(shared["shared_roles"], ["embedding", "lm_head"])
            self.assertTrue(shared["tied_to_embedding"])

            lm_head = _load(root / "out" / "lm_head_splits_manifest.json")
            self.assertTrue(lm_head["tied_to_embedding"])
            self.assertEqual(lm_head["weight"], "model.embed_tokens.weight")
            self.assertEqual(lm_head["split_count"], 4)


def _write_fake_model(
    model_dir: Path,
    specs: dict[str, dict[str, object]],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001-of-00001.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: shard_name for key in specs},
            }
        )
    )
    _write_fake_safetensors(model_dir / shard_name, specs)


def _write_fake_safetensors(
    path: Path,
    specs: dict[str, dict[str, object]],
) -> None:
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for key, spec in specs.items():
        shape = [int(dim) for dim in spec["shape"]]
        dtype = str(spec["dtype"])
        byte_size = _numel(shape) * _dtype_size(dtype)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * offset)


def _dtype_size(dtype: str) -> int:
    return {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "U64": 8,
        "I32": 4,
        "U32": 4,
        "I16": 2,
        "U16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }[dtype]


def _numel(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
