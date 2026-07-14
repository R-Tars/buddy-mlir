from __future__ import annotations

from typing import Any


KEY_TENSOR_PATHS = {
    "embedding.weight",
    "layers.0.input_norm.weight",
    "layers.0.post_attention_norm.weight",
    "layers.0.attention.wqkv_packed.weight",
    "layers.0.attention.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "final_norm.weight",
    "lm_head.splits.0.weight",
}


def materialization_summary(params: Any) -> dict[str, Any]:
    metadata = dict(getattr(params, "metadata", {}))
    tensors = metadata.get("tensors", {})
    return {
        "backend": metadata.get("backend"),
        "model_name": metadata.get("model_name"),
        "num_layers": metadata.get("num_layers"),
        "materialized_layer_ids": list(metadata.get("materialized_layer_ids", [])),
        "tensor_count": metadata.get("tensor_count"),
        "key_paths": [path for path in KEY_TENSOR_PATHS if path in tensors],
    }


def tensorization_summary(report: dict[str, Any]) -> dict[str, Any]:
    tensors = report.get("tensors", [])
    tensor_paths = sorted(
        str(record["path"])
        for record in tensors
        if isinstance(record, dict) and record.get("path") is not None
    )
    key_paths = [
        record["path"]
        for record in tensors
        if isinstance(record, dict) and record.get("path") in KEY_TENSOR_PATHS
    ]
    key_tensor_records = {}
    for record in tensors:
        path = record.get("path") if isinstance(record, dict) else None
        if path not in key_paths:
            continue
        key_tensor_records[path] = {
            field: record.get(field)
            for field in (
                "role",
                "role_group",
                "target_dtype",
                "layout",
                "memory_config",
                "ttnn_dtype",
                "ttnn_layout",
                "ttnn_memory_config",
                "transform",
                "source_shape",
                "shape",
            )
        }
    return {
        "status": report.get("status"),
        "backend": "ttnn",
        "roles": list(report.get("roles", [])),
        "tensor_count": report.get("tensor_count"),
        "target_dtype_counts": _field_counts(tensors, "target_dtype"),
        "layout_counts": _field_counts(tensors, "layout"),
        "memory_config_counts": _field_counts(tensors, "memory_config"),
        "transform_counts": _field_counts(tensors, "transform"),
        "transform_paths_by_kind": _paths_by_field_value(tensors, "transform"),
        "ttnn_dtype_counts": _field_counts(tensors, "ttnn_dtype"),
        "ttnn_layout_counts": _field_counts(tensors, "ttnn_layout"),
        "ttnn_memory_config_counts": _field_counts(tensors, "ttnn_memory_config"),
        "tensor_paths": tensor_paths,
        "key_paths": key_paths,
        "key_tensors": key_tensor_records,
    }


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _paths_by_field_value(
    records: list[dict[str, Any]], field: str
) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for record in records:
        value = record.get(field)
        path = record.get("path")
        if value is None or path is None:
            continue
        paths.setdefault(str(value), []).append(str(path))
    return {key: sorted(value) for key, value in sorted(paths.items())}
