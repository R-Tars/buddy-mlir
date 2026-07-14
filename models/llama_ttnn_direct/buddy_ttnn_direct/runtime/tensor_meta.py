from __future__ import annotations

from typing import Any


def tensor_shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def tensor_dtype(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def runtime_int_tensor(torch: Any, values: Any, *, name: str) -> Any:
    dtype = getattr(torch, "int32", None)
    tensor_fn = getattr(torch, "tensor", None)
    if callable(tensor_fn):
        try:
            tensor = tensor_fn(values, dtype=dtype)
        except TypeError:
            tensor = tensor_fn(values)
    else:
        zeros = getattr(torch, "zeros", None)
        if not callable(zeros):
            raise ValueError("torch module must provide tensor or zeros")
        shape = _nested_int_shape(values)
        tensor = zeros(shape, dtype=dtype) if dtype is not None else zeros(shape)
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor


def _nested_int_shape(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    if not values:
        return [0]
    return [len(values), *_nested_int_shape(values[0])]
