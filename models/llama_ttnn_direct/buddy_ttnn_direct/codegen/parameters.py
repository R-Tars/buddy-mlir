from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .artifacts import write_json


class ParameterMaterializationError(RuntimeError):
    """Raised when a requested parameter tensor cannot be materialized."""


@dataclass(frozen=True)
class TensorParameter:
    weight: Any
    source_key: str
    role: str
    shape: list[int]
    dtype: str | None = None


@dataclass(frozen=True)
class PackedTensorParameter:
    weight: Any
    source_keys: list[str]
    role: str
    shape: list[int]
    dtype: str | None = None
    packed_axis: str = "output_features"


def load_llama_parameters_from_manifests(
    *,
    model_path: str | Path,
    weights_manifest: str | Path | Mapping[str, Any],
    config: str | Path | Mapping[str, Any],
    device: Any | None = None,
    tensor_backend: str = "torch",
    layers: Iterable[int] | None = None,
    tensor_loader: Any | None = None,
) -> SimpleNamespace:
    """Materialize a nested parameters object from TTNN Direct manifests.

    The production backend returns torch tensors loaded from safetensors. Tests
    can inject a loader with ``load_tensor(key, entry)`` and ``cat`` methods.
    """
    if tensor_backend != "torch":
        raise ParameterMaterializationError(
            f"unsupported parameter materialization backend: {tensor_backend}"
        )
    if device is not None:
        raise ParameterMaterializationError(
            "PR-B only materializes host-side torch parameters; device must be None"
        )

    manifest = _load_mapping(weights_manifest)
    config_dict = _load_mapping(config)
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ParameterMaterializationError(
            "weights_manifest must contain a weights object"
        )

    loader = tensor_loader or TorchSafetensorsTensorLoader(model_path)
    num_layers = int(config_dict["num_layers"])
    selected_layers = _normalize_layers(layers, num_layers)
    tensor_records: dict[str, dict[str, Any]] = {}

    def load_role(
        role: str,
        *,
        layer_id: int | None,
        path: str,
    ) -> TensorParameter:
        key, entry = _find_weight(weights, role, layer_id)
        tensor = loader.load_tensor(key, entry)
        parameter = TensorParameter(
            weight=tensor,
            source_key=key,
            role=role,
            shape=_tensor_shape(tensor, entry),
            dtype=_tensor_dtype(tensor, entry),
        )
        tensor_records[path] = _tensor_record(parameter)
        return parameter

    embedding = load_role(
        "embedding",
        layer_id=None,
        path="embedding.weight",
    )
    final_norm = load_role(
        "final_norm",
        layer_id=None,
        path="final_norm.weight",
    )

    materialized_layers: list[Any | None] = [None] * num_layers
    for layer_id in selected_layers:
        attention = SimpleNamespace(
            q_proj=load_role(
                "q_proj",
                layer_id=layer_id,
                path=f"layers.{layer_id}.attention.q_proj.weight",
            ),
            k_proj=load_role(
                "k_proj",
                layer_id=layer_id,
                path=f"layers.{layer_id}.attention.k_proj.weight",
            ),
            v_proj=load_role(
                "v_proj",
                layer_id=layer_id,
                path=f"layers.{layer_id}.attention.v_proj.weight",
            ),
            o_proj=load_role(
                "o_proj",
                layer_id=layer_id,
                path=f"layers.{layer_id}.attention.o_proj.weight",
            ),
        )
        attention.wqkv_packed = _pack_qkv(
            loader,
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            path=f"layers.{layer_id}.attention.wqkv_packed.weight",
            tensor_records=tensor_records,
        )
        mlp = SimpleNamespace(
            gate_proj=load_role(
                "mlp_gate",
                layer_id=layer_id,
                path=f"layers.{layer_id}.mlp.gate_proj.weight",
            ),
            up_proj=load_role(
                "mlp_up",
                layer_id=layer_id,
                path=f"layers.{layer_id}.mlp.up_proj.weight",
            ),
            down_proj=load_role(
                "mlp_down",
                layer_id=layer_id,
                path=f"layers.{layer_id}.mlp.down_proj.weight",
            ),
        )
        materialized_layers[layer_id] = SimpleNamespace(
            layer_id=layer_id,
            attention=attention,
            mlp=mlp,
            input_norm=load_role(
                "input_norm",
                layer_id=layer_id,
                path=f"layers.{layer_id}.input_norm.weight",
            ),
            post_attention_norm=load_role(
                "post_attention_norm",
                layer_id=layer_id,
                path=f"layers.{layer_id}.post_attention_norm.weight",
            ),
        )

    lm_head = _materialize_lm_head(
        loader,
        weights,
        config_dict,
        tensor_records,
    )

    metadata = {
        "schema_version": 1,
        "backend": tensor_backend,
        "model_name": manifest.get("model_name") or config_dict.get("model_name"),
        "num_layers": num_layers,
        "materialized_layer_ids": selected_layers,
        "tensor_count": len(tensor_records),
        "tensors": tensor_records,
    }
    return SimpleNamespace(
        embedding=embedding,
        layers=materialized_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        metadata=metadata,
    )


def materialize_parameters_from_program(
    *,
    model_path: str | Path,
    program_dir: str | Path,
    backend: str = "torch",
    layers: Iterable[int] | None = None,
    out: str | Path | None = None,
    tensor_loader: Any | None = None,
) -> dict[str, Any]:
    root = Path(program_dir)
    params = load_llama_parameters_from_manifests(
        model_path=model_path,
        weights_manifest=root / "weights_manifest.json",
        config=root / "config.json",
        tensor_backend=backend,
        layers=layers,
        tensor_loader=tensor_loader,
    )
    report = build_parameter_materialization_report(
        params,
        model_path=model_path,
        program_dir=program_dir,
    )
    if out is not None:
        write_json(out, report)
    return report


def build_parameter_materialization_report(
    params: SimpleNamespace,
    *,
    model_path: str | Path,
    program_dir: str | Path,
) -> dict[str, Any]:
    metadata = dict(params.metadata)
    return {
        "schema_version": 1,
        "status": "pass",
        "backend": metadata["backend"],
        "model_path": str(model_path),
        "program_dir": str(program_dir),
        "model_name": metadata.get("model_name"),
        "num_layers": metadata["num_layers"],
        "materialized_layer_ids": list(metadata["materialized_layer_ids"]),
        "tensor_count": metadata["tensor_count"],
        "tensors": metadata["tensors"],
        "lm_head": {
            "split_count": len(params.lm_head.splits),
            "splits": [
                {
                    "shard_id": split.shard_id,
                    "vocab_start": split.vocab_start,
                    "vocab_end": split.vocab_end,
                    "shape": list(split.weight_param.shape),
                    "source_key": split.weight_param.source_key,
                }
                for split in params.lm_head.splits
            ],
        },
    }


def parse_layer_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    layer_ids: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer_ids.append(int(chunk))
    return layer_ids


class TorchSafetensorsTensorLoader:
    def __init__(self, model_path: str | Path):
        self.root = Path(model_path)
        if self.root.is_file():
            self.root = self.root.parent
        try:
            self.torch = importlib.import_module("torch")
            self.safetensors = importlib.import_module("safetensors")
        except ModuleNotFoundError as exc:
            raise ParameterMaterializationError(
                "materialize-parameters --backend torch requires torch and "
                "safetensors to be installed"
            ) from exc
        self._load_file = None
        try:
            self._load_file = importlib.import_module(
                "safetensors.torch"
            ).load_file
        except ModuleNotFoundError:
            self._load_file = None

    def load_tensor(self, key: str, entry: Mapping[str, Any]) -> Any:
        filename = entry.get("filename")
        if not filename:
            raise ParameterMaterializationError(
                f"weight {key!r} has no safetensors filename in manifest"
            )
        path = self.root / str(filename)
        safe_open = getattr(self.safetensors, "safe_open", None)
        if safe_open is not None:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        if self._load_file is None:
            raise ParameterMaterializationError(
                "safetensors.safe_open or safetensors.torch.load_file is required"
            )
        tensors = self._load_file(str(path), device="cpu")
        return tensors[key]

    def cat(self, tensors: Sequence[Any], *, dim: int) -> Any:
        return self.torch.cat(list(tensors), dim=dim)

    def slice_vocab(self, tensor: Any, start: int, end: int) -> Any:
        return tensor[start:end, :]


def _materialize_lm_head(
    loader: Any,
    weights: Mapping[str, Any],
    config: Mapping[str, Any],
    tensor_records: dict[str, dict[str, Any]],
) -> SimpleNamespace:
    source_key, source_entry = _find_weight(weights, "lm_head", None)
    source_tensor = loader.load_tensor(source_key, source_entry)
    source_param = TensorParameter(
        weight=source_tensor,
        source_key=source_key,
        role="lm_head",
        shape=_tensor_shape(source_tensor, source_entry),
        dtype=_tensor_dtype(source_tensor, source_entry),
    )
    tensor_records["lm_head.weight"] = _tensor_record(source_param)

    lm_head_config = config.get("lm_head", {})
    if not isinstance(lm_head_config, Mapping):
        raise ParameterMaterializationError("config.lm_head must be an object")
    splits = lm_head_config.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ParameterMaterializationError(
            "config.lm_head.splits must be a non-empty list"
        )

    split_params = []
    for split in splits:
        shard_id = int(split["shard_id"])
        vocab_start = int(split["vocab_start"])
        vocab_end = int(split["vocab_end"])
        tensor = loader.slice_vocab(source_tensor, vocab_start, vocab_end)
        parameter = TensorParameter(
            weight=tensor,
            source_key=source_key,
            role="lm_head_split",
            shape=_tensor_shape(tensor, source_entry),
            dtype=_tensor_dtype(tensor, source_entry),
        )
        path = f"lm_head.splits.{shard_id}.weight"
        tensor_records[path] = {
            **_tensor_record(parameter),
            "vocab_start": vocab_start,
            "vocab_end": vocab_end,
        }
        split_params.append(
            SimpleNamespace(
                shard_id=shard_id,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
                weight=parameter.weight,
                weight_param=parameter,
            )
        )

    return SimpleNamespace(weight=source_param.weight, weight_param=source_param, splits=split_params)


def _pack_qkv(
    loader: Any,
    q_proj: TensorParameter,
    k_proj: TensorParameter,
    v_proj: TensorParameter,
    *,
    path: str,
    tensor_records: dict[str, dict[str, Any]],
) -> PackedTensorParameter:
    _validate_qkv_shapes(q_proj, k_proj, v_proj)
    packed = loader.cat(
        [q_proj.weight, k_proj.weight, v_proj.weight],
        dim=0,
    )
    parameter = PackedTensorParameter(
        weight=packed,
        source_keys=[
            q_proj.source_key,
            k_proj.source_key,
            v_proj.source_key,
        ],
        role="wqkv_packed",
        shape=_shape_from_attr(packed)
        or [
            q_proj.shape[0] + k_proj.shape[0] + v_proj.shape[0],
            q_proj.shape[1],
        ],
        dtype=q_proj.dtype,
    )
    tensor_records[path] = {
        "role": parameter.role,
        "shape": list(parameter.shape),
        "dtype": parameter.dtype,
        "source_keys": list(parameter.source_keys),
        "packed_axis": parameter.packed_axis,
    }
    return parameter


def _validate_qkv_shapes(
    q_proj: TensorParameter,
    k_proj: TensorParameter,
    v_proj: TensorParameter,
) -> None:
    for parameter in (q_proj, k_proj, v_proj):
        if len(parameter.shape) != 2:
            raise ParameterMaterializationError(
                f"{parameter.source_key} must be rank-2 for QKV packing; "
                f"got shape {parameter.shape}"
            )
    input_dims = {q_proj.shape[1], k_proj.shape[1], v_proj.shape[1]}
    if len(input_dims) != 1:
        raise ParameterMaterializationError(
            "QKV source weights must have matching input feature dimension; "
            f"got {q_proj.shape}, {k_proj.shape}, {v_proj.shape}"
        )


def _find_weight(
    weights: Mapping[str, Any],
    role: str,
    layer_id: int | None,
) -> tuple[str, Mapping[str, Any]]:
    matches = []
    for key, entry in weights.items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("layer_id") != layer_id:
            continue
        if _entry_has_role(entry, role):
            matches.append((str(key), entry))
    if len(matches) != 1:
        raise ParameterMaterializationError(
            f"expected exactly one weight for role={role!r}, "
            f"layer_id={layer_id}; found {len(matches)}"
        )
    return matches[0]


def _entry_has_role(entry: Mapping[str, Any], role: str) -> bool:
    if entry.get("role") == role:
        return True
    shared_roles = entry.get("shared_roles", [])
    return isinstance(shared_roles, list) and role in shared_roles


def _normalize_layers(
    layers: Iterable[int] | None,
    num_layers: int,
) -> list[int]:
    if layers is None:
        return list(range(num_layers))
    normalized = sorted(set(int(layer_id) for layer_id in layers))
    for layer_id in normalized:
        if layer_id < 0 or layer_id >= num_layers:
            raise ParameterMaterializationError(
                f"layer id {layer_id} is out of range for {num_layers} layers"
            )
    return normalized


def _load_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text())


def _tensor_record(parameter: TensorParameter) -> dict[str, Any]:
    return {
        "role": parameter.role,
        "shape": list(parameter.shape),
        "dtype": parameter.dtype,
        "source_key": parameter.source_key,
    }


def _tensor_shape(
    tensor: Any,
    entry: Mapping[str, Any],
) -> list[int]:
    shape = _shape_from_attr(tensor)
    if shape is not None:
        return shape
    entry_shape = entry.get("shape")
    if isinstance(entry_shape, list):
        return [int(dim) for dim in entry_shape]
    raise ParameterMaterializationError(
        f"tensor for {entry.get('key')} does not expose shape metadata"
    )


def _shape_from_attr(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _tensor_dtype(
    tensor: Any,
    entry: Mapping[str, Any],
) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None:
        return str(dtype)
    entry_dtype = entry.get("dtype")
    return str(entry_dtype) if entry_dtype is not None else None
