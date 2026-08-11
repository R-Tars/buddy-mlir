"""Small shared adapters for diagnostic stages.

The product runtime owns model loading, tensor metadata, structural checks,
report serialization, and device lifetime.  Diagnostic stages import those
owners through this module so private stage-to-stage APIs do not reappear.
"""

from __future__ import annotations

from typing import Any

from ..runtime_environment import collect_ttnn_environment
from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..runtime.model_loader import load_generated_model, to_namespace
from ..runtime.reports import (
    NUMERIC_REFERENCE_NOT_RUN_REASON,
    dry_run_reference,
    write_report,
)
from ..runtime.structural import (
    dtype_check,
    generated_observed_op_sequence,
    op_sequence_coverage_check,
    shape_check,
    value_check,
)
from ..runtime.tensor_meta import (
    runtime_int_tensor,
    tensor_dtype,
    tensor_shape,
)


def observed_op_sequence(ttnn: Any) -> list[str] | None:
    return generated_observed_op_sequence(None, ttnn)


def shape(tensor: Any) -> list[int] | None:
    return tensor_shape(tensor)


def output_shapes(
    value: Any,
    *,
    expected_names: Any | None = None,
) -> dict[str, list[int] | None]:
    """Map a diagnostic result to its named shape contract."""
    names = list(expected_names or [])
    if isinstance(value, tuple):
        if len(names) != len(value):
            names = ["query", "key", "value"] if len(value) == 3 else ["query", "key"]
        return {name: shape(item) for name, item in zip(names, value)}
    if len(names) == 1:
        return {names[0]: shape(value)}
    return {"output": shape(value)}


def dtype_name(dtype_seed: str) -> str:
    return "bfloat16" if dtype_seed == "bf16" else "float32"


def dtype(tensor: Any) -> str | None:
    return tensor_dtype(tensor)


def generated_model(path: Any, ttnn: Any) -> Any:
    return load_generated_model(path, ttnn)


def randn(torch: Any, shape: list[int], dtype_seed: str, *, name: str | None = None) -> Any:
    dtype_value = getattr(torch, "bfloat16", None) if dtype_seed == "bf16" else getattr(torch, "float32", None)
    try:
        tensor = torch.randn(tuple(shape), dtype=dtype_value)
    except TypeError:
        tensor = torch.randn(tuple(shape))
    if name is not None:
        try:
            tensor.name = name
        except AttributeError:
            pass
    return tensor


def zeros(torch: Any, shape: list[int], *, dtype: Any | None = None, name: str | None = None) -> Any:
    try:
        tensor = torch.zeros(tuple(shape), dtype=dtype)
    except TypeError:
        tensor = torch.zeros(tuple(shape))
    if name is not None:
        try:
            tensor.name = name
        except AttributeError:
            pass
    return tensor


def tensor_from_values(
    torch: Any,
    values: Any,
    *,
    dtype: Any,
    name: str,
    fallback_shape: list[int],
) -> Any:
    tensor_fn = getattr(torch, "tensor", None)
    if callable(tensor_fn):
        try:
            tensor = tensor_fn(values, dtype=dtype)
        except TypeError:
            tensor = tensor_fn(values)
    else:
        tensor = zeros(torch, fallback_shape, dtype=dtype, name=name)
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor


def runtime_index_tensor(
    torch: Any,
    *,
    name: str,
    shape: list[int],
    page_state: Any | None,
) -> Any:
    dtype_value = getattr(torch, "int32", None)
    if page_state is not None and name == "page_table":
        return tensor_from_values(torch, page_state.page_table, dtype=dtype_value, name=name, fallback_shape=shape)
    if page_state is not None and name == "cache_position":
        return tensor_from_values(torch, page_state.cache_position, dtype=dtype_value, name=name, fallback_shape=shape)
    return zeros(torch, shape, dtype=dtype_value, name=name)


def ttnn_dtype(ttnn: Any, dtype_seed: str) -> Any:
    return getattr(ttnn, "bfloat16", None) if dtype_seed == "bf16" else getattr(ttnn, "float32", None)


def synthetic_tensor_factory(
    *, ttnn: Any, torch: Any, device: Any, dtype_seed: str
) -> tuple[Any, Any]:
    dtype_value = ttnn_dtype(ttnn, dtype_seed)
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    count = 0

    def tensor(shape_value: list[int], *, name: str, zeros: bool = False, memory_config: Any | None = None) -> Any:
        nonlocal count
        host = globals()["zeros"](torch, shape_value, name=name) if zeros else randn(torch, shape_value, dtype_seed, name=name)
        count += 1
        kwargs = {"dtype": dtype_value, "layout": layout, "device": device}
        if memory_config is not None:
            kwargs["memory_config"] = memory_config
        return ttnn.from_torch(host, **kwargs)

    return tensor, lambda: count


def failed_diagnostic_report(
    base: dict[str, Any],
    *,
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
    ttnn_module: Any | None = None,
    include_runtime_metadata: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the common unavailable/error contract to a stage report."""
    report = dict(base)
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "error": message,
            "detail": detail,
        }
    )
    if include_runtime_metadata:
        report.update(
            {
                "ttnn_version": ttnn_version,
                "ttnn_environment": collect_ttnn_environment(ttnn_module),
            }
        )
    if extra:
        report.update(extra)
    return report
