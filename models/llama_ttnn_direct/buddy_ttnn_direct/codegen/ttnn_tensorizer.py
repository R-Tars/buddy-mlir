from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .artifacts import write_json
from .config_emit import emit_parameter_config
from ..semantic.dump import load_graph_json


DEFAULT_ROLE_GROUPS = ("mlp", "lm_head")
SUPPORTED_ROLE_GROUPS = {
    "embedding",
    "norm",
    "attention",
    "mlp",
    "lm_head",
}
NORM_ROLE_PATHS = {
    "input_norm": "input_norm",
    "post_attention_norm": "post_attention_norm",
}
MLP_ROLE_PATHS = {
    "mlp_gate": "gate_proj",
    "mlp_up": "up_proj",
    "mlp_down": "down_proj",
}
ATTENTION_ROLE_PATHS = {
    "wqkv_packed": "wqkv_packed",
    "o_proj": "o_proj",
}


class TTNNTensorizationError(RuntimeError):
    """Raised when host parameters cannot be converted to TTNN tensors."""


@dataclass(frozen=True)
class TensorizationResult:
    parameters: SimpleNamespace | None
    report: dict[str, Any]


def parse_role_groups(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return list(DEFAULT_ROLE_GROUPS)
    roles = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(roles).difference(SUPPORTED_ROLE_GROUPS))
    if unsupported:
        raise TTNNTensorizationError(
            "unsupported tensorization role group(s): "
            + ", ".join(unsupported)
        )
    return roles


def load_parameter_config_from_program(program_dir: str | Path) -> dict[str, Any]:
    root = Path(program_dir)
    graph = load_graph_json(root / "semantic_graph.json")
    config = json.loads((root / "config.json").read_text())
    template_config = config.get("template_config", {})
    if not isinstance(template_config, Mapping):
        template_config = {}
    lm_head = config.get("lm_head", {})
    if not isinstance(lm_head, Mapping):
        lm_head = {}
    return emit_parameter_config(
        graph,
        recipe=str(
            template_config.get(
                "dtype_recipe",
                "official_like_performance_seed",
            )
        ),
        lm_head_split_count=int(lm_head.get("split_count", 1)),
    )


def build_tensorization_plan(
    parameter_config: Mapping[str, Any],
    *,
    roles: Sequence[str],
    layers: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    role_groups = parse_role_groups(",".join(roles))
    selected_layers = _normalize_layers(layers)
    records: list[dict[str, Any]] = []
    weights = parameter_config.get("weights", {})
    if not isinstance(weights, Mapping):
        raise TTNNTensorizationError("parameter_config.weights must be an object")

    if "embedding" in role_groups:
        source_key, entry = _find_parameter_config_item(
            weights,
            "embedding",
            None,
        )
        records.append(
            {
                "path": "embedding.weight",
                "role_group": "embedding",
                "role": "embedding",
                "layer_id": None,
                "source_key": source_key,
                "target_dtype": str(entry.get("target_dtype")),
                "layout": str(entry.get("layout")),
                "memory_config": None,
            }
        )

    if "norm" in role_groups:
        for layer_id in _layer_ids_for_roles(
            weights,
            NORM_ROLE_PATHS,
            selected_layers,
        ):
            for role, attr in NORM_ROLE_PATHS.items():
                source_key, entry = _find_parameter_config_item(
                    weights,
                    role,
                    layer_id,
                )
                records.append(
                    {
                        "path": f"layers.{layer_id}.{attr}.weight",
                        "role_group": "norm",
                        "role": role,
                        "layer_id": layer_id,
                        "source_key": source_key,
                        "target_dtype": str(entry.get("target_dtype")),
                        "layout": str(entry.get("layout")),
                        "memory_config": None,
                    }
                )
        source_key, entry = _find_parameter_config_item(
            weights,
            "final_norm",
            None,
        )
        records.append(
            {
                "path": "final_norm.weight",
                "role_group": "norm",
                "role": "final_norm",
                "layer_id": None,
                "source_key": source_key,
                "target_dtype": str(entry.get("target_dtype")),
                "layout": str(entry.get("layout")),
                "memory_config": None,
            }
        )

    if "attention" in role_groups:
        for layer_id in _layer_ids_for_roles(
            weights,
            {"q_proj": "q_proj"},
            selected_layers,
        ):
            q_key, q_entry = _find_parameter_config_item(
                weights,
                "q_proj",
                layer_id,
            )
            k_key, k_entry = _find_parameter_config_item(
                weights,
                "k_proj",
                layer_id,
            )
            v_key, v_entry = _find_parameter_config_item(
                weights,
                "v_proj",
                layer_id,
            )
            _check_packed_qkv_config(layer_id, q_entry, k_entry, v_entry)
            records.append(
                {
                    "path": f"layers.{layer_id}.attention.wqkv_packed.weight",
                    "role_group": "attention",
                    "role": "wqkv_packed",
                    "layer_id": layer_id,
                    "source_key": ",".join([q_key, k_key, v_key]),
                    "source_keys": [q_key, k_key, v_key],
                    "target_dtype": str(q_entry.get("target_dtype")),
                    "layout": str(q_entry.get("layout")),
                    "memory_config": None,
                    "packing": "qkv_pack",
                }
            )
            o_key, o_entry = _find_parameter_config_item(
                weights,
                "o_proj",
                layer_id,
            )
            records.append(
                {
                    "path": f"layers.{layer_id}.attention.o_proj.weight",
                    "role_group": "attention",
                    "role": "o_proj",
                    "layer_id": layer_id,
                    "source_key": o_key,
                    "target_dtype": str(o_entry.get("target_dtype")),
                    "layout": str(o_entry.get("layout")),
                    "memory_config": None,
                }
            )

    if "mlp" in role_groups:
        for source_key, entry in sorted(weights.items()):
            if not isinstance(entry, Mapping):
                continue
            role = str(entry.get("role"))
            if role not in MLP_ROLE_PATHS:
                continue
            layer_id = int(entry["layer_id"])
            if selected_layers is not None and layer_id not in selected_layers:
                continue
            records.append(
                {
                    "path": (
                        f"layers.{layer_id}.mlp."
                        f"{MLP_ROLE_PATHS[role]}.weight"
                    ),
                    "role_group": "mlp",
                    "role": role,
                    "layer_id": layer_id,
                    "source_key": str(source_key),
                    "target_dtype": str(entry.get("target_dtype")),
                    "layout": str(entry.get("layout")),
                    "memory_config": None,
                }
            )

    if "lm_head" in role_groups:
        lm_source_key, lm_entry = _find_parameter_config_item(
            weights,
            "lm_head",
            None,
        )
        lm_head = parameter_config.get("lm_head", {})
        if not isinstance(lm_head, Mapping):
            raise TTNNTensorizationError("parameter_config.lm_head must be an object")
        splits = lm_head.get("splits", [])
        if not isinstance(splits, list):
            raise TTNNTensorizationError(
                "parameter_config.lm_head.splits must be a list"
            )
        for split in splits:
            shard_id = int(split["shard_id"])
            records.append(
                {
                    "path": f"lm_head.splits.{shard_id}.weight",
                    "role_group": "lm_head",
                    "role": "lm_head_split",
                    "layer_id": None,
                    "shard_id": shard_id,
                    "vocab_start": int(split["vocab_start"]),
                    "vocab_end": int(split["vocab_end"]),
                    "source_key": lm_source_key,
                    "target_dtype": str(lm_entry.get("target_dtype")),
                    "layout": str(lm_entry.get("layout")),
                    "memory_config": None,
                }
            )

    return records


def tensorization_dry_run_report(
    *,
    parameter_config: Mapping[str, Any],
    roles: Sequence[str],
    layers: Iterable[int] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    plan = build_tensorization_plan(
        parameter_config,
        roles=roles,
        layers=layers,
    )
    return {
        "schema_version": 1,
        "status": "dry_run",
        "dry_run": True,
        "device": device,
        "roles": list(roles),
        "tensor_count": len(plan),
        "tensors": [
            {**record, "status": "planned", "shape": None}
            for record in plan
        ],
    }


def to_ttnn_parameters(
    torch_params: SimpleNamespace | None,
    device: Any,
    parameter_config: Mapping[str, Any],
    *,
    roles: Sequence[str],
    layers: Iterable[int] | None = None,
    ttnn_module: Any | None = None,
    dry_run: bool = False,
) -> TensorizationResult:
    if dry_run:
        return TensorizationResult(
            parameters=None,
            report=tensorization_dry_run_report(
                parameter_config=parameter_config,
                roles=roles,
                layers=layers,
                device=str(device) if device is not None else None,
            ),
        )
    if torch_params is None:
        raise TTNNTensorizationError(
            "torch_params are required unless dry_run=True"
        )
    ttnn = ttnn_module or importlib.import_module("ttnn")
    if not hasattr(ttnn, "from_torch"):
        raise TTNNTensorizationError("ttnn module must provide from_torch")

    plan = build_tensorization_plan(
        parameter_config,
        roles=roles,
        layers=layers,
    )
    output = _empty_ttnn_parameters(torch_params)
    records = []
    for record in plan:
        torch_tensor = _get_torch_tensor(torch_params, record)
        converted = _from_torch(
            ttnn,
            torch_tensor,
            device=device,
            target_dtype=record["target_dtype"],
            layout=record["layout"],
        )
        _store_converted_tensor(output, record, converted)
        records.append(
            {
                **copy.deepcopy(record),
                "status": "tensorized",
                "shape": _tensor_shape(torch_tensor),
                "ttnn_dtype": str(_resolve_ttnn_dtype(ttnn, record["target_dtype"])),
                "ttnn_layout": str(_resolve_ttnn_layout(ttnn, record["layout"])),
            }
        )

    output.metadata = {
        "schema_version": 1,
        "backend": "ttnn",
        "roles": list(roles),
        "tensor_count": len(records),
        "tensors": records,
    }
    return TensorizationResult(
        parameters=output,
        report={
            "schema_version": 1,
            "status": "pass",
            "dry_run": False,
            "device": str(device),
            "roles": list(roles),
            "tensor_count": len(records),
            "tensors": records,
        },
    )


def tensorize_parameters_from_program_dry_run(
    *,
    program_dir: str | Path,
    roles: Sequence[str],
    layers: Iterable[int] | None = None,
    device: str | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    parameter_config = load_parameter_config_from_program(program_dir)
    report = tensorization_dry_run_report(
        parameter_config=parameter_config,
        roles=roles,
        layers=layers,
        device=device,
    )
    report["program_dir"] = str(program_dir)
    if out is not None:
        write_json(out, report)
    return report


def _from_torch(
    ttnn: Any,
    tensor: Any,
    *,
    device: Any,
    target_dtype: str,
    layout: str,
) -> Any:
    kwargs = {
        "device": device,
        "dtype": _resolve_ttnn_dtype(ttnn, target_dtype),
        "layout": _resolve_ttnn_layout(ttnn, layout),
    }
    return ttnn.from_torch(tensor, **kwargs)


def _empty_ttnn_parameters(torch_params: SimpleNamespace) -> SimpleNamespace:
    layer_count = len(getattr(torch_params, "layers", []))
    lm_splits = getattr(getattr(torch_params, "lm_head", None), "splits", [])
    return SimpleNamespace(
        embedding=None,
        layers=[None] * layer_count,
        final_norm=None,
        lm_head=SimpleNamespace(splits=[None] * len(lm_splits)),
        metadata={},
    )


def _get_torch_tensor(torch_params: SimpleNamespace, record: Mapping[str, Any]) -> Any:
    if record["role_group"] == "embedding":
        return torch_params.embedding.weight
    if record["role_group"] == "norm":
        if record["role"] == "final_norm":
            return torch_params.final_norm.weight
        layer = _materialized_layer(torch_params, int(record["layer_id"]))
        return getattr(layer, str(record["role"])).weight
    if record["role_group"] == "attention":
        layer = _materialized_layer(torch_params, int(record["layer_id"]))
        attr = ATTENTION_ROLE_PATHS[str(record["role"])]
        return getattr(layer.attention, attr).weight
    if record["role_group"] == "mlp":
        layer = _materialized_layer(torch_params, int(record["layer_id"]))
        attr = MLP_ROLE_PATHS[str(record["role"])]
        return getattr(layer.mlp, attr).weight
    if record["role_group"] == "lm_head":
        return torch_params.lm_head.splits[int(record["shard_id"])].weight
    raise TTNNTensorizationError(
        f"unsupported tensorization role group: {record['role_group']}"
    )


def _store_converted_tensor(
    output: SimpleNamespace,
    record: Mapping[str, Any],
    tensor: Any,
) -> None:
    if record["role_group"] == "embedding":
        output.embedding = SimpleNamespace(weight=tensor)
        return
    if record["role_group"] == "norm":
        if record["role"] == "final_norm":
            output.final_norm = SimpleNamespace(weight=tensor)
            return
        layer = _ensure_layer(output, int(record["layer_id"]))
        setattr(
            layer,
            str(record["role"]),
            SimpleNamespace(weight=tensor),
        )
        return
    if record["role_group"] == "attention":
        layer = _ensure_layer(output, int(record["layer_id"]))
        attention = _ensure_child_namespace(layer, "attention")
        setattr(
            attention,
            ATTENTION_ROLE_PATHS[str(record["role"])],
            SimpleNamespace(weight=tensor),
        )
        return
    if record["role_group"] == "mlp":
        layer_id = int(record["layer_id"])
        layer = _ensure_layer(output, layer_id)
        mlp = _ensure_child_namespace(layer, "mlp")
        setattr(
            mlp,
            MLP_ROLE_PATHS[str(record["role"])],
            SimpleNamespace(weight=tensor),
        )
        return
    if record["role_group"] == "lm_head":
        shard_id = int(record["shard_id"])
        output.lm_head.splits[shard_id] = SimpleNamespace(
            weight=tensor,
            shard_id=shard_id,
            vocab_start=record["vocab_start"],
            vocab_end=record["vocab_end"],
        )
        return
    raise TTNNTensorizationError(
        f"unsupported tensorization role group: {record['role_group']}"
    )


def _materialized_layer(torch_params: SimpleNamespace, layer_id: int) -> Any:
    layer = torch_params.layers[layer_id]
    if layer is None:
        raise TTNNTensorizationError(f"layer {layer_id} was not materialized")
    return layer


def _ensure_layer(output: SimpleNamespace, layer_id: int) -> SimpleNamespace:
    layer = output.layers[layer_id]
    if layer is None:
        layer = SimpleNamespace()
        output.layers[layer_id] = layer
    return layer


def _ensure_child_namespace(parent: SimpleNamespace, name: str) -> SimpleNamespace:
    child = getattr(parent, name, None)
    if child is None:
        child = SimpleNamespace()
        setattr(parent, name, child)
    return child


def _layer_ids_for_roles(
    weights: Mapping[str, Any],
    roles: Mapping[str, str],
    selected_layers: set[int] | None,
) -> list[int]:
    layer_ids = {
        int(entry["layer_id"])
        for entry in weights.values()
        if isinstance(entry, Mapping)
        and entry.get("role") in roles
        and entry.get("layer_id") is not None
    }
    if selected_layers is not None:
        layer_ids = layer_ids.intersection(selected_layers)
    return sorted(layer_ids)


def _check_packed_qkv_config(
    layer_id: int,
    q_entry: Mapping[str, Any],
    k_entry: Mapping[str, Any],
    v_entry: Mapping[str, Any],
) -> None:
    fields = ("target_dtype", "layout")
    for field in fields:
        values = {
            q_entry.get(field),
            k_entry.get(field),
            v_entry.get(field),
        }
        if len(values) != 1:
            raise TTNNTensorizationError(
                f"layer {layer_id} packed QKV {field} entries must match; "
                f"got {sorted(str(value) for value in values)}"
            )


def _find_parameter_config_item(
    weights: Mapping[str, Any],
    role: str,
    layer_id: int | None,
) -> tuple[str, Mapping[str, Any]]:
    matches = [
        (str(source_key), entry)
        for source_key, entry in weights.items()
        if isinstance(entry, Mapping)
        and entry.get("role") == role
        and entry.get("layer_id") == layer_id
    ]
    if len(matches) != 1:
        raise TTNNTensorizationError(
            f"expected exactly one parameter config entry for role={role!r}, "
            f"layer_id={layer_id}; found {len(matches)}"
        )
    return matches[0]


def _resolve_ttnn_dtype(ttnn: Any, target_dtype: str) -> Any:
    return getattr(ttnn, target_dtype, target_dtype)


def _resolve_ttnn_layout(ttnn: Any, layout: str) -> Any:
    if layout == "tile":
        return getattr(ttnn, "TILE_LAYOUT", layout)
    if layout == "row_major":
        return getattr(ttnn, "ROW_MAJOR_LAYOUT", layout)
    return getattr(ttnn, layout, layout)


def _tensor_shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _normalize_layers(layers: Iterable[int] | None) -> set[int] | None:
    if layers is None:
        return None
    return {int(layer_id) for layer_id in layers}
