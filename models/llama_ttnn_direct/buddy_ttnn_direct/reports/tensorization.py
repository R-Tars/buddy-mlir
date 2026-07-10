from __future__ import annotations

from typing import Any

from ..codegen.ttnn_tensorizer import LINEAR_WEIGHT_TRANSFORM
from .schema import int_list as _int_list


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
