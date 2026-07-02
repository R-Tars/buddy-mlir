from __future__ import annotations

import copy
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..semantic.graph import LlamaModelGraph
from ..semantic.validate import validate_llama_graph
from ..templates.lm_head import build_lm_head_split_ranges


GENERATED_ARTIFACTS = ("model.py", "config.json", "plan.json", "README.md")
OFFLINE_ARTIFACT_MANIFESTS = (
    "weights_manifest.json",
    "packed_qkv_manifest.json",
    "mlp_manifest.json",
    "lm_head_splits_manifest.json",
    "kv_cache_manifest.json",
)


def planned_artifact_paths(out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    return {name: root / name for name in GENERATED_ARTIFACTS}


def planned_offline_artifact_paths(out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    return {name: root / name for name in OFFLINE_ARTIFACT_MANIFESTS}


def ensure_output_dir(out_dir: str | Path) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text)


def prepare_offline_artifacts(
    model_path: str | Path,
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    validate_llama_graph(graph)
    root = ensure_output_dir(out_dir)
    paths = planned_offline_artifact_paths(root)
    manifests = build_offline_artifact_manifests(model_path, graph, config)
    for name, payload in manifests.items():
        write_json(paths[name], payload)
    return paths


def build_offline_artifact_manifests(
    model_path: str | Path,
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    validate_llama_graph(graph)
    metadata = collect_safetensors_header_metadata(model_path)
    embedding_weight = _embedding_weight_key(graph, config, metadata)
    weights = _build_weight_manifest(graph, config, metadata, embedding_weight)
    return {
        "weights_manifest.json": weights,
        "packed_qkv_manifest.json": _build_packed_qkv_manifest(
            graph, metadata
        ),
        "mlp_manifest.json": _build_mlp_manifest(graph, metadata),
        "lm_head_splits_manifest.json": _build_lm_head_manifest(
            graph,
            config,
            metadata,
            embedding_weight,
        ),
        "kv_cache_manifest.json": _build_kv_cache_manifest(graph, config),
    }


def collect_safetensors_header_metadata(
    model_path: str | Path,
) -> dict[str, dict[str, Any]]:
    path = Path(model_path)
    root = path if path.is_dir() else path.parent
    metadata: dict[str, dict[str, Any]] = {}
    safetensor_files: set[Path] = set()

    if path.is_file() and path.suffix == ".safetensors":
        safetensor_files.add(path)
    if path.is_dir():
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            _merge_index_metadata(root / index_name, metadata, safetensor_files)
        safetensor_files.update(path.glob("*.safetensors"))

    for file in sorted(safetensor_files):
        try:
            header = _read_safetensors_header(file)
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            continue
        for key, entry in header.items():
            metadata[key] = {
                "key": key,
                "shape": entry.get("shape"),
                "dtype": entry.get("dtype"),
                "filename": file.name,
                "data_offsets": entry.get("data_offsets"),
                "byte_size": entry.get("byte_size"),
            }

    return metadata


def _build_weight_manifest(
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    embedding_weight: str,
) -> dict[str, Any]:
    weights: dict[str, dict[str, Any]] = {}
    _add_manifest_weight(weights, embedding_weight, "embedding", None, metadata)

    for layer in graph.layers:
        layer_id = layer.layer_id
        _add_manifest_weight(
            weights, layer.input_norm.weight, "input_norm", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.attention.q_proj, "q_proj", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.attention.k_proj, "k_proj", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.attention.v_proj, "v_proj", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.attention.o_proj, "o_proj", layer_id, metadata
        )
        _add_manifest_weight(
            weights,
            layer.post_attention_norm.weight,
            "post_attention_norm",
            layer_id,
            metadata,
        )
        _add_manifest_weight(
            weights, layer.mlp.gate_proj, "mlp_gate", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.mlp.up_proj, "mlp_up", layer_id, metadata
        )
        _add_manifest_weight(
            weights, layer.mlp.down_proj, "mlp_down", layer_id, metadata
        )

    _add_manifest_weight(
        weights, graph.final_norm.weight, "final_norm", None, metadata
    )
    _add_manifest_weight(
        weights,
        graph.lm_head.weight,
        "lm_head",
        None,
        metadata,
        extra={"tied_to_embedding": graph.lm_head.tied_to_embedding},
    )

    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "mode": graph.mode,
        "metadata_policy": {
            "format": "safetensors_header",
            "loads_tensor_payloads": False,
            "conversion": "manifest_only",
        },
        "config_summary": _config_summary(config),
        "weight_count": len(weights),
        "weights": weights,
    }


def _build_packed_qkv_manifest(
    graph: LlamaModelGraph,
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "packing": "packed_qkv",
        "materialized": False,
        "layers": [
            {
                "layer_id": layer.layer_id,
                "packed_weight": (
                    f"model.layers.{layer.layer_id}.self_attn.wqkv_packed.weight"
                ),
                "source_axis": "output_features",
                "source_weights": [
                    _source_weight(layer.attention.q_proj, "q_proj", metadata),
                    _source_weight(layer.attention.k_proj, "k_proj", metadata),
                    _source_weight(layer.attention.v_proj, "v_proj", metadata),
                ],
                "complete": all(
                    key in metadata
                    for key in (
                        layer.attention.q_proj,
                        layer.attention.k_proj,
                        layer.attention.v_proj,
                    )
                ),
            }
            for layer in graph.layers
        ],
    }


def _build_mlp_manifest(
    graph: LlamaModelGraph,
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "activation": "silu",
        "materialized": False,
        "layers": [
            {
                "layer_id": layer.layer_id,
                "gate_up_packing": "gate_up_group",
                "gate_proj": _source_weight(
                    layer.mlp.gate_proj, "mlp_gate", metadata
                ),
                "up_proj": _source_weight(layer.mlp.up_proj, "mlp_up", metadata),
                "down_proj": _source_weight(
                    layer.mlp.down_proj, "mlp_down", metadata
                ),
            }
            for layer in graph.layers
        ],
    }


def _build_lm_head_manifest(
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    embedding_weight: str,
) -> dict[str, Any]:
    split_count, splits = _lm_head_split_config(graph, config)
    source_key = embedding_weight if graph.lm_head.tied_to_embedding else graph.lm_head.weight
    source = _source_weight(source_key, "lm_head", metadata)
    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "weight": source_key,
        "embedding_weight": embedding_weight,
        "tied_to_embedding": graph.lm_head.tied_to_embedding,
        "split_axis": "vocab",
        "split_count": split_count,
        "materialized": False,
        "source_weight": source,
        "splits": [
            {
                **copy.deepcopy(split),
                "artifact_name": (
                    f"lm_head_vocab_{split['shard_id']:02d}.safetensors"
                ),
                "source_weight": source_key,
            }
            for split in splits
        ],
    }


def _build_kv_cache_manifest(
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    kv_cache = config.get("kv_cache", {})
    if not isinstance(kv_cache, Mapping):
        kv_cache = {}
    page_block_size = int(kv_cache.get("page_block_size", 32))
    dtype = str(kv_cache.get("dtype", "bfloat8_b"))
    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "policy": str(kv_cache.get("policy", "paged")),
        "materialized": False,
        "page_block_size": page_block_size,
        "dtype": dtype,
        "num_layers": graph.num_layers,
        "batch_size": graph.batch_size,
        "max_cache_len": graph.max_cache_len,
        "num_kv_heads": graph.num_key_value_heads,
        "head_dim": graph.head_dim,
        "layers": [
            {
                "layer_id": layer.layer_id,
                "k_cache": f"layers.{layer.layer_id}.kv_cache.k",
                "v_cache": f"layers.{layer.layer_id}.kv_cache.v",
                "uses_paged_kv_cache": layer.attention.uses_paged_kv_cache,
            }
            for layer in graph.layers
        ],
    }


def _merge_index_metadata(
    index_path: Path,
    metadata: dict[str, dict[str, Any]],
    safetensor_files: set[Path],
) -> None:
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        return
    for key, filename in weight_map.items():
        if not isinstance(key, str):
            continue
        filename_str = str(filename)
        metadata[key] = {"key": key, "filename": filename_str}
        if filename_str.endswith(".safetensors"):
            safetensor_files.add(index_path.parent / filename_str)


def _read_safetensors_header(file: Path) -> dict[str, dict[str, Any]]:
    with file.open("rb") as handle:
        raw_header_size = handle.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"invalid safetensors header prefix: {file}")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"truncated safetensors header: {file}")
    raw_header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(raw_header, Mapping):
        raise ValueError(f"invalid safetensors header JSON: {file}")

    header: dict[str, dict[str, Any]] = {}
    for key, value in raw_header.items():
        if key == "__metadata__" or not isinstance(value, Mapping):
            continue
        data_offsets = _int_list(value.get("data_offsets"))
        byte_size = None
        if data_offsets is not None and len(data_offsets) == 2:
            byte_size = data_offsets[1] - data_offsets[0]
        header[str(key)] = {
            "dtype": str(value["dtype"]) if "dtype" in value else None,
            "shape": _int_list(value.get("shape")),
            "data_offsets": data_offsets,
            "byte_size": byte_size,
        }
    return header


def _embedding_weight_key(
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    weights = config.get("weights", {})
    if isinstance(weights, Mapping):
        for key, entry in weights.items():
            if not isinstance(entry, Mapping):
                continue
            shared_roles = entry.get("shared_roles", [])
            if entry.get("role") == "embedding" or "embedding" in shared_roles:
                return str(key)
    if graph.lm_head.tied_to_embedding and graph.lm_head.weight != "lm_head.weight":
        return graph.lm_head.weight
    if "model.embed_tokens.weight" in metadata:
        return "model.embed_tokens.weight"
    if "embed_tokens.weight" in metadata:
        return "embed_tokens.weight"
    return "model.embed_tokens.weight"


def _lm_head_split_config(
    graph: LlamaModelGraph,
    config: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    lm_head = config.get("lm_head", {})
    if not isinstance(lm_head, Mapping):
        lm_head = {}
    splits = lm_head.get("splits")
    if isinstance(splits, list) and splits:
        return int(lm_head.get("split_count", len(splits))), copy.deepcopy(splits)
    split_count = int(
        lm_head.get(
            "split_count",
            config.get("lm_head_split_count", graph.lm_head.split_count or 1),
        )
    )
    return split_count, build_lm_head_split_ranges(graph.vocab_size, split_count)


def _add_manifest_weight(
    weights: dict[str, dict[str, Any]],
    key: str,
    role: str,
    layer_id: int | None,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    entry = _source_weight(key, role, metadata)
    entry["layer_id"] = layer_id
    if extra:
        entry.update(extra)
    if key not in weights:
        weights[key] = entry
        return

    existing = weights[key]
    shared_roles = existing.setdefault("shared_roles", [existing["role"]])
    if role not in shared_roles:
        shared_roles.append(role)
    existing["role"] = "shared"
    if extra:
        existing.update(extra)


def _source_weight(
    key: str,
    role: str,
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    meta = metadata.get(key, {})
    return {
        "key": key,
        "role": role,
        "shape": copy.deepcopy(meta.get("shape")),
        "dtype": meta.get("dtype"),
        "filename": meta.get("filename"),
        "data_offsets": copy.deepcopy(meta.get("data_offsets")),
        "byte_size": meta.get("byte_size"),
        "metadata_available": key in metadata,
    }


def _config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("schema_version", "model_name", "mode", "recipe"):
        if key in config:
            summary[key] = config[key]
    if "lm_head" in config:
        lm_head = config["lm_head"]
        if isinstance(lm_head, Mapping):
            summary["lm_head_split_count"] = lm_head.get("split_count")
    if "kv_cache" in config:
        kv_cache = config["kv_cache"]
        if isinstance(kv_cache, Mapping):
            summary["kv_page_block_size"] = kv_cache.get("page_block_size")
    return summary


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return [int(item) for item in value]
