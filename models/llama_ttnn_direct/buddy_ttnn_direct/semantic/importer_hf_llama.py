from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import (
    AttentionDecodeNode,
    GatedMLPNode,
    LMHeadNode,
    LlamaLayerNode,
    LlamaModelGraph,
    RMSNormNode,
)
from .validate import validate_llama_graph


@dataclass(frozen=True)
class TensorMetadata:
    key: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    filename: str | None = None


def import_hf_llama(
    model_path: str | Path,
    *,
    mode: str,
    batch_size: int,
    seq_len: int,
    max_cache_len: int,
    generation_mode: str = "greedy",
    config: Mapping[str, Any] | object | None = None,
    state_dict_metadata: (
        Mapping[str, Any] | Iterable[str] | None
    ) = None,
    model_name: str | None = None,
) -> LlamaModelGraph:
    """Import HF Llama config and weight names into the semantic graph IR.

    The importer never calls ``torch.load``. Real tensor data is intentionally
    out of scope for Phase 1; when metadata is unavailable, canonical HF Llama
    weight names are inferred from the config.
    """

    model_path = Path(model_path)
    config_obj = config if config is not None else load_llama_config(model_path)
    metadata = (
        normalize_state_dict_metadata(state_dict_metadata)
        if state_dict_metadata is not None
        else collect_state_dict_metadata(model_path)
    )
    known_keys = set(metadata)

    num_layers = _config_int(config_obj, "num_hidden_layers", "n_layers")
    hidden_size = _config_int(config_obj, "hidden_size", "dim")
    intermediate_size = _config_int(
        config_obj, "intermediate_size", "hidden_dim"
    )
    num_attention_heads = _config_int(
        config_obj, "num_attention_heads", "n_heads"
    )
    num_key_value_heads = _config_int(
        config_obj,
        "num_key_value_heads",
        "n_kv_heads",
        default=num_attention_heads,
    )
    head_dim = _config_int(
        config_obj,
        "head_dim",
        default=hidden_size // num_attention_heads,
    )
    vocab_size = _config_int(config_obj, "vocab_size")
    rms_norm_eps = _config_float(
        config_obj, "rms_norm_eps", "norm_eps", default=1e-5
    )
    rope_theta = _config_float(config_obj, "rope_theta", default=10000.0)

    layers = [
        _import_layer(
            layer_id=i,
            known_keys=known_keys,
            eps=rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            uses_paged_kv_cache=(mode == "decode"),
        )
        for i in range(num_layers)
    ]

    embed_weight = _choose_key(
        ["model.embed_tokens.weight", "embed_tokens.weight"], known_keys
    )
    lm_head_weight = _choose_key(
        ["lm_head.weight", "model.lm_head.weight", embed_weight], known_keys
    )
    tie_word_embeddings = bool(
        _config_value(config_obj, "tie_word_embeddings", default=False)
    )
    tied_to_embedding = tie_word_embeddings or (
        bool(known_keys)
        and lm_head_weight == embed_weight
        and "lm_head.weight" not in known_keys
    )

    graph = LlamaModelGraph(
        model_name=model_name or _infer_model_name(model_path, config_obj),
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        layers=layers,
        final_norm=RMSNormNode(
            weight=_choose_key(
                ["model.norm.weight", "norm.weight"], known_keys
            ),
            eps=rms_norm_eps,
        ),
        lm_head=LMHeadNode(
            weight=lm_head_weight,
            tied_to_embedding=tied_to_embedding,
            split_count=None,
        ),
        mode=mode,
        batch_size=batch_size,
        seq_len=seq_len,
        max_cache_len=max_cache_len,
        generation_mode=generation_mode,
    )
    validate_llama_graph(graph)
    return graph


def load_llama_config(model_path: str | Path) -> Mapping[str, Any] | object:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.is_file() and config_path.name == "config.json":
        return json.loads(config_path.read_text())

    try:
        from transformers import AutoConfig
    except ImportError as err:
        raise FileNotFoundError(
            "Could not find a local config.json and transformers is not "
            f"installed, so model config cannot be loaded: {model_path}"
        ) from err

    return AutoConfig.from_pretrained(str(model_path))


def collect_state_dict_metadata(
    model_path: str | Path,
) -> dict[str, TensorMetadata]:
    path = Path(model_path)
    if not path.is_dir():
        return {}

    metadata: dict[str, TensorMetadata] = {}
    safetensor_files: set[Path] = set()

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, Mapping):
            continue
        for key, filename in weight_map.items():
            if not isinstance(key, str):
                continue
            filename_str = str(filename)
            metadata[key] = TensorMetadata(key=key, filename=filename_str)
            if filename_str.endswith(".safetensors"):
                safetensor_files.add(path / filename_str)

    safetensor_files.update(path.glob("*.safetensors"))
    _merge_safetensors_metadata(metadata, sorted(safetensor_files))
    return metadata


def normalize_state_dict_metadata(
    state_dict_metadata: Mapping[str, Any] | Iterable[str],
) -> dict[str, TensorMetadata]:
    if isinstance(state_dict_metadata, Mapping):
        normalized: dict[str, TensorMetadata] = {}
        for key, value in state_dict_metadata.items():
            normalized[str(key)] = _metadata_from_value(str(key), value)
        return normalized

    return {
        str(key): TensorMetadata(key=str(key))
        for key in state_dict_metadata
    }


def _metadata_from_value(key: str, value: Any) -> TensorMetadata:
    if isinstance(value, TensorMetadata):
        return value
    if isinstance(value, Mapping):
        shape = _shape_tuple(value.get("shape"))
        dtype = value.get("dtype")
        filename = value.get("filename")
        return TensorMetadata(
            key=key,
            shape=shape,
            dtype=str(dtype) if dtype is not None else None,
            filename=str(filename) if filename is not None else None,
        )

    shape = _shape_tuple(getattr(value, "shape", value))
    dtype = getattr(value, "dtype", None)
    return TensorMetadata(
        key=key,
        shape=shape,
        dtype=str(dtype) if dtype is not None else None,
    )


def _merge_safetensors_metadata(
    metadata: dict[str, TensorMetadata], files: list[Path]
) -> None:
    if not files:
        return
    try:
        from safetensors import safe_open
    except ImportError:
        return

    for file in files:
        if not file.is_file():
            continue
        try:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shape: tuple[int, ...] | None = None
                    dtype: str | None = None
                    try:
                        tensor_slice = handle.get_slice(key)
                        shape = _shape_tuple(tensor_slice.get_shape())
                        get_dtype = getattr(tensor_slice, "get_dtype", None)
                        if get_dtype is not None:
                            dtype = str(get_dtype())
                    except Exception:
                        pass
                    metadata[key] = TensorMetadata(
                        key=key,
                        shape=shape,
                        dtype=dtype,
                        filename=file.name,
                    )
        except Exception:
            continue


def _import_layer(
    *,
    layer_id: int,
    known_keys: set[str],
    eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    uses_paged_kv_cache: bool,
) -> LlamaLayerNode:
    prefix = f"model.layers.{layer_id}"
    return LlamaLayerNode(
        layer_id=layer_id,
        input_norm=RMSNormNode(
            weight=_choose_key(
                [
                    f"{prefix}.input_layernorm.weight",
                    f"layers.{layer_id}.attention_norm.weight",
                ],
                known_keys,
            ),
            eps=eps,
        ),
        attention=AttentionDecodeNode(
            layer_id=layer_id,
            q_proj=_choose_key(
                [f"{prefix}.self_attn.q_proj.weight"], known_keys
            ),
            k_proj=_choose_key(
                [f"{prefix}.self_attn.k_proj.weight"], known_keys
            ),
            v_proj=_choose_key(
                [f"{prefix}.self_attn.v_proj.weight"], known_keys
            ),
            o_proj=_choose_key(
                [f"{prefix}.self_attn.o_proj.weight"], known_keys
            ),
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            uses_paged_kv_cache=uses_paged_kv_cache,
        ),
        post_attention_norm=RMSNormNode(
            weight=_choose_key(
                [
                    f"{prefix}.post_attention_layernorm.weight",
                    f"layers.{layer_id}.ffn_norm.weight",
                ],
                known_keys,
            ),
            eps=eps,
        ),
        mlp=GatedMLPNode(
            layer_id=layer_id,
            gate_proj=_choose_key(
                [f"{prefix}.mlp.gate_proj.weight"], known_keys
            ),
            up_proj=_choose_key(
                [f"{prefix}.mlp.up_proj.weight"], known_keys
            ),
            down_proj=_choose_key(
                [f"{prefix}.mlp.down_proj.weight"], known_keys
            ),
            activation="silu",
        ),
    )


def _choose_key(candidates: list[str], known_keys: set[str]) -> str:
    for key in candidates:
        if key in known_keys:
            return key
    return candidates[0]


def _infer_model_name(
    model_path: Path, config: Mapping[str, Any] | object
) -> str:
    for attr in ("_name_or_path", "name_or_path", "model_name"):
        value = _config_value(config, attr, default=None)
        if value:
            return str(value)
    return model_path.name or str(model_path)


def _config_int(
    config: Mapping[str, Any] | object,
    *names: str,
    default: int | None = None,
) -> int:
    value = _config_value(config, *names, default=default)
    if value is None:
        joined = ", ".join(names)
        raise KeyError(f"missing required Llama config field: {joined}")
    return int(value)


def _config_float(
    config: Mapping[str, Any] | object,
    *names: str,
    default: float | None = None,
) -> float:
    value = _config_value(config, *names, default=default)
    if value is None:
        joined = ", ".join(names)
        raise KeyError(f"missing required Llama config field: {joined}")
    return float(value)


def _config_value(
    config: Mapping[str, Any] | object,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, Iterable):
        try:
            return tuple(int(dim) for dim in value)
        except (TypeError, ValueError):
            return None
    return None
