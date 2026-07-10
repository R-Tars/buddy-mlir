from __future__ import annotations

from typing import Any

from ..codegen.ttnn_tensorizer import LINEAR_WEIGHT_TRANSFORM
from .schema import (
    expected_layer_ids as _expected_layer_ids,
    int_list as _int_list,
    safe_int as _safe_int,
)


_TENSORIZED_PHYSICAL_SHAPE_TRANSFORMS = {
    LINEAR_WEIGHT_TRANSFORM,
    "reshape_embedding_weight_4d",
    "reshape_norm_weight_4d",
}


def lm_head_source_reference_complete(materialize: Any) -> bool:
    observed = lm_head_source_reference_observed(materialize)
    lm_head = observed.get("lm_head.weight")
    split0 = observed.get("lm_head.splits.0.weight")
    if not isinstance(lm_head, dict) or not isinstance(split0, dict):
        return False
    return (
        lm_head.get("materialization") == "metadata_reference"
        and lm_head.get("materialized") is False
        and split0.get("source_read") == "sliced_tensor"
    )


def lm_head_source_reference_observed(materialize: Any) -> dict[str, Any]:
    if not isinstance(materialize, dict):
        return {}
    key_tensors = materialize.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return {}
    observed: dict[str, Any] = {}
    for path in ("lm_head.weight", "lm_head.splits.0.weight"):
        tensor = key_tensors.get(path)
        if not isinstance(tensor, dict):
            continue
        observed[path] = {
            "shape": tensor.get("shape"),
            "materialization": tensor.get("materialization"),
            "materialized": tensor.get("materialized"),
            "source_read": tensor.get("source_read"),
        }
    return observed


def lm_head_transform_complete(
    tensorization: Any,
    split_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_count = _safe_int(split_count)
    if expected_count is None or expected_count <= 0:
        return False
    counts = tensorization.get("transform_counts")
    if not isinstance(counts, dict):
        return False
    observed_count = _safe_int(counts.get(LINEAR_WEIGHT_TRANSFORM))
    if observed_count is None or observed_count < expected_count:
        return False
    expected_paths = {
        f"lm_head.splits.{shard_id}.weight"
        for shard_id in range(expected_count)
    }
    transformed_paths = transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if transformed_paths and not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    split0 = key_tensors.get("lm_head.splits.0.weight")
    if not isinstance(split0, dict):
        return False
    shape = split0.get("shape")
    return (
        split0.get("transform") == LINEAR_WEIGHT_TRANSFORM
        and isinstance(shape, list)
        and len(shape) == 4
    )


def embedding_norm_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    paths = embedding_norm_weight_transform_paths(layer_count)
    embedding_paths = set(paths["embedding"])
    norm_paths = set(paths["norm"])
    if not embedding_paths or not norm_paths:
        return False
    embedding_transformed = transformed_tensor_paths(
        tensorization,
        "reshape_embedding_weight_4d",
    )
    norm_transformed = transformed_tensor_paths(
        tensorization,
        "reshape_norm_weight_4d",
    )
    if not embedding_paths.issubset(embedding_transformed):
        return False
    if not norm_paths.issubset(norm_transformed):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "embedding.weight",
        "layers.0.input_norm.weight",
        "layers.0.post_attention_norm.weight",
        "final_norm.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def embedding_norm_weight_transform_paths(
    layer_count: Any,
) -> dict[str, list[str]]:
    norm_paths = ["final_norm.weight"]
    for layer_id in _expected_layer_ids(layer_count):
        norm_paths.extend(
            [
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    return {
        "embedding": ["embedding.weight"],
        "norm": norm_paths,
    }


def embedding_norm_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = embedding_norm_weight_transform_paths(layer_count)
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths["embedding"] + expected_paths["norm"]:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "embedding_transformed_paths": sorted(
            transformed_tensor_paths(
                tensorization,
                "reshape_embedding_weight_4d",
            )
        ),
        "norm_transformed_paths": sorted(
            transformed_tensor_paths(
                tensorization,
                "reshape_norm_weight_4d",
            )
        ),
        "key_tensors": key_observed,
    }


def linear_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_paths = set(linear_weight_transform_paths(layer_count))
    if not expected_paths:
        return False
    transformed_paths = transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "layers.0.attention.wqkv_packed.weight",
        "layers.0.attention.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            if tensor.get("transform") != LINEAR_WEIGHT_TRANSFORM:
                return False
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def linear_weight_transform_paths(layer_count: Any) -> list[str]:
    paths: list[str] = []
    for layer_id in _expected_layer_ids(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
            ]
        )
    return paths


def decode_shell_linear_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
    split_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_paths = set(
        decode_shell_linear_weight_transform_paths(
            layer_count,
            split_count,
        )
    )
    if not expected_paths:
        return False
    transformed_paths = transformed_tensor_paths(
        tensorization,
        LINEAR_WEIGHT_TRANSFORM,
    )
    if not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.splits.0.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            if tensor.get("transform") != LINEAR_WEIGHT_TRANSFORM:
                return False
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def decode_shell_linear_weight_transform_paths(
    layer_count: Any,
    split_count: Any,
) -> list[str]:
    paths: list[str] = []
    for layer_id in _expected_layer_ids(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
            ]
        )
    split_count_int = _safe_int(split_count)
    if split_count_int is not None and split_count_int > 0:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(split_count_int)
        )
    return paths


def transformed_tensor_paths(
    tensorization: dict[str, Any],
    transform: str,
) -> set[str]:
    paths_by_kind = tensorization.get("transform_paths_by_kind")
    if isinstance(paths_by_kind, dict):
        paths = paths_by_kind.get(transform, [])
        if isinstance(paths, list):
            return {str(path) for path in paths}
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return set()
    return {
        str(path)
        for path, tensor in key_tensors.items()
        if isinstance(tensor, dict) and tensor.get("transform") == transform
    }


def decode_shell_linear_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
    split_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = decode_shell_linear_weight_transform_paths(
        layer_count,
        split_count,
    )
    transformed_paths = sorted(
        transformed_tensor_paths(tensorization, LINEAR_WEIGHT_TRANSFORM)
    )
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "transformed_paths": transformed_paths,
        "key_tensors": key_observed,
    }


def linear_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = linear_weight_transform_paths(layer_count)
    transformed_paths = sorted(
        transformed_tensor_paths(tensorization, LINEAR_WEIGHT_TRANSFORM)
    )
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "transformed_paths": transformed_paths,
        "key_tensors": key_observed,
    }


def lm_head_transform_observed(tensorization: Any) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    key_tensors = tensorization.get("key_tensors")
    split0 = {}
    if isinstance(key_tensors, dict):
        maybe_split0 = key_tensors.get("lm_head.splits.0.weight")
        if isinstance(maybe_split0, dict):
            split0 = {
                "transform": maybe_split0.get("transform"),
                "source_shape": maybe_split0.get("source_shape"),
                "shape": maybe_split0.get("shape"),
            }
    return {
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
        "lm_head.splits.0.weight": split0,
    }


def tensorized_physical_shape_mismatches(
    tensorization: Any,
) -> list[dict[str, Any]]:
    if not isinstance(tensorization, dict):
        return []
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return []
    mismatches = []
    for path, tensor in sorted(key_tensors.items()):
        if not isinstance(tensor, dict):
            continue
        transform = tensor.get("transform")
        if transform not in _TENSORIZED_PHYSICAL_SHAPE_TRANSFORMS:
            continue
        expected = expected_tensorized_physical_shape(tensor)
        observed = _int_list(tensor.get("shape"))
        source_shape = _int_list(tensor.get("source_shape"))
        if expected is None:
            mismatches.append(
                {
                    "path": path,
                    "transform": transform,
                    "source_shape": source_shape or None,
                    "observed": observed or None,
                    "expected": "known transform/source shape",
                }
            )
        elif observed != expected:
            mismatches.append(
                {
                    "path": path,
                    "transform": transform,
                    "source_shape": source_shape or None,
                    "observed": observed or None,
                    "expected": expected,
                }
            )
    return mismatches


def expected_tensorized_physical_shape(
    tensor: dict[str, Any],
) -> list[int] | None:
    transform = tensor.get("transform")
    source_shape = _int_list(tensor.get("source_shape"))
    if transform == LINEAR_WEIGHT_TRANSFORM:
        if len(source_shape) != 2:
            return None
        return [1, 1, source_shape[1], source_shape[0]]
    if transform == "reshape_embedding_weight_4d":
        if len(source_shape) != 2:
            return None
        return [1, 1, source_shape[0], source_shape[1]]
    if transform == "reshape_norm_weight_4d":
        if len(source_shape) != 1:
            return None
        hidden = source_shape[0]
        if hidden <= 0:
            return None
        if hidden % 32 == 0:
            return [1, 1, hidden // 32, 32]
        return [1, 1, 1, hidden]
    return None


def step_tensorization_summary(step: dict[str, Any]) -> dict[str, Any]:
    setup = step.get("parameter_setup") or {}
    tensorization = setup.get("tensorization") or {}
    return tensorization if isinstance(tensorization, dict) else {}
