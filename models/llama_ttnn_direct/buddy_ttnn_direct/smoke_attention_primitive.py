from __future__ import annotations

import importlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .smoke_mlp import (
    NO_TTNN_DEVICE_MESSAGE,
    NoTTNNDeviceError,
    _managed_ttnn_device,
)
from .smoke_decode_shell import (
    NUMERIC_REFERENCE_NOT_RUN_REASON,
    _dry_run_reference,
    _observed_op_sequence,
    _shape_check,
)
from .templates import ttnn_ops
from .templates.ttnn_ops import UnsupportedTTNNOp


ATTENTION_PRIMITIVES = (
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
)


def run_smoke_attention_primitive(
    *,
    out: str | Path,
    primitive: str,
    device: str,
    device_id: int = 0,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int = 1024,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    _validate_args(
        primitive=primitive,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
        dtype_seed=dtype_seed,
    )
    plan = _primitive_plan(
        primitive=primitive,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
    )

    if dry_run:
        report = _base_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "error": None,
                "ttnn_version": None,
                "reference": _dry_run_reference("attention_primitive"),
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_report(out, report)
        return report

    try:
        ttnn = (
            ttnn_module
            if ttnn_module is not None
            else importlib.import_module("ttnn")
        )
    except ImportError as err:
        report = _no_device_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        torch = (
            torch_module
            if torch_module is not None
            else importlib.import_module("torch")
        )
    except ImportError as err:
        report = _failed_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="missing_torch",
            message="torch is required for attention primitive smoke tensors.",
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            start = time.perf_counter()
            outputs = _run_primitive(
                primitive=primitive,
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                plan=plan,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            synchronize = getattr(ttnn, "synchronize_device", None)
            if callable(synchronize):
                synchronize(ttnn_device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        report = _base_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            dry_run=False,
            plan=plan,
        )
        output_shapes = _output_shapes(
            outputs,
            expected_names=plan["expected_output_shapes"].keys(),
        )
        reference = _attention_primitive_reference(
            primitive=primitive,
            plan=plan,
            output_shapes=output_shapes,
            observed_ops=_observed_op_sequence(ttnn),
        )
        passed = bool(reference["passed"])
        report.update(
            {
                "passed": passed,
                "status": "passed" if passed else "reference_mismatch",
                "latency_ms": latency_ms,
                "output_shapes": output_shapes,
                "error": (
                    None
                    if passed
                    else "attention primitive structural reference mismatch"
                ),
                "ttnn_version": getattr(ttnn, "__version__", None),
                "reference": reference,
            }
        )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            detail=str(err),
        )
    except UnsupportedTTNNOp as err:
        report = _failed_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except Exception as err:
        report = _failed_report(
            primitive=primitive,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )

    _write_report(out, report)
    return report


def _run_primitive(
    *,
    primitive: str,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    dtype = _ttnn_dtype(ttnn, dtype_seed)
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    memory_config = _memory_config(ttnn)

    def tensor(name: str) -> Any:
        shape = plan["input_shapes"][name]
        if "position" in name or name == "page_table":
            torch_tensor = _zeros(torch, shape)
        else:
            torch_tensor = _randn(torch, shape, dtype_seed)
        return ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
        )

    if primitive == "qkv_linear":
        return ttnn.linear(
            tensor("hidden"),
            tensor("qkv_weight"),
            memory_config=memory_config,
        )
    if primitive == "o_proj_linear":
        return ttnn.linear(
            tensor("attention"),
            tensor("o_proj_weight"),
            memory_config=memory_config,
        )
    if primitive == "nlp_create_qkv_heads_decode":
        return ttnn_ops.nlp_create_qkv_heads_decode(
            ttnn,
            tensor("fused_qkv"),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            memory_config=memory_config,
        )
    if primitive == "rotary_embedding_decode":
        return ttnn_ops.rotary_embedding_decode(
            ttnn,
            tensor("query"),
            tensor("key"),
            cos_matrix=tensor("cos_matrix"),
            sin_matrix=tensor("sin_matrix"),
            transformation_matrix=tensor("transformation_matrix"),
        )
    if primitive == "paged_update_cache":
        return ttnn_ops.paged_update_cache(
            ttnn,
            tensor("cache"),
            tensor("update"),
            update_idxs_tensor=tensor("cache_position"),
            page_table=tensor("page_table"),
        )
    if primitive == "paged_scaled_dot_product_attention_decode":
        return ttnn_ops.paged_sdpa_decode(
            ttnn,
            tensor("query"),
            tensor("key_cache"),
            tensor("value_cache"),
            tensor("page_table"),
            tensor("cache_position"),
            scale=float(head_dim) ** -0.5,
            memory_config=memory_config,
        )
    if primitive == "nlp_concat_heads_decode":
        return ttnn_ops.nlp_concat_heads_decode(
            ttnn,
            tensor("attention"),
            num_heads=num_heads,
            memory_config=memory_config,
        )
    raise AssertionError(f"unhandled primitive: {primitive}")


def _primitive_plan(
    *,
    primitive: str,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
) -> dict[str, Any]:
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    kv_size = num_kv_heads * head_dim
    page_count = max(1, (max_cache_len + 31) // 32)
    plans = {
        "qkv_linear": {
            "input_shapes": {
                "hidden": [batch_size, 1, hidden_size],
                "qkv_weight": [hidden_size, qkv_size],
            },
            "expected_output_shapes": {"qkv": [batch_size, 1, qkv_size]},
        },
        "nlp_create_qkv_heads_decode": {
            "input_shapes": {
                "fused_qkv": [batch_size, 1, qkv_size],
            },
            "expected_output_shapes": {
                "query": [batch_size, num_heads, 1, head_dim],
                "key": [batch_size, num_kv_heads, 1, head_dim],
                "value": [batch_size, num_kv_heads, 1, head_dim],
            },
        },
        "rotary_embedding_decode": {
            "input_shapes": {
                "query": [batch_size, num_heads, 1, head_dim],
                "key": [batch_size, num_kv_heads, 1, head_dim],
                "cos_matrix": [1, 1, head_dim, head_dim],
                "sin_matrix": [1, 1, head_dim, head_dim],
                "transformation_matrix": [1, 1, head_dim, head_dim],
            },
            "expected_output_shapes": {
                "query": [batch_size, num_heads, 1, head_dim],
                "key": [batch_size, num_kv_heads, 1, head_dim],
            },
        },
        "paged_update_cache": {
            "input_shapes": {
                "cache": [batch_size, max_cache_len, num_kv_heads, head_dim],
                "update": [batch_size, 1, num_kv_heads, head_dim],
                "cache_position": [batch_size],
                "page_table": [batch_size, page_count],
            },
            "expected_output_shapes": {
                "cache": [batch_size, max_cache_len, num_kv_heads, head_dim],
            },
        },
        "paged_scaled_dot_product_attention_decode": {
            "input_shapes": {
                "query": [batch_size, num_heads, 1, head_dim],
                "key_cache": [batch_size, max_cache_len, num_kv_heads, head_dim],
                "value_cache": [batch_size, max_cache_len, num_kv_heads, head_dim],
                "page_table": [batch_size, page_count],
                "cache_position": [batch_size],
            },
            "expected_output_shapes": {
                "attention": [batch_size, num_heads, 1, head_dim],
            },
        },
        "nlp_concat_heads_decode": {
            "input_shapes": {
                "attention": [batch_size, num_heads, 1, head_dim],
            },
            "expected_output_shapes": {
                "hidden": [batch_size, 1, num_heads * head_dim],
            },
        },
        "o_proj_linear": {
            "input_shapes": {
                "attention": [batch_size, 1, num_heads * head_dim],
                "o_proj_weight": [num_heads * head_dim, hidden_size],
            },
            "expected_output_shapes": {
                "hidden": [batch_size, 1, hidden_size],
            },
        },
    }
    plan = dict(plans[primitive])
    plan["tensor_conversion_count"] = len(plan["input_shapes"])
    return plan


def _base_report(
    *,
    primitive: str,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
    dtype_seed: str,
    dry_run: bool,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "attention_primitive",
        "primitive": primitive,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_cache_len": max_cache_len,
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "memory_config": "default_or_l1",
        "dry_run": dry_run,
        "input_shapes": plan["input_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "output_shapes": None,
        "tensor_conversion_count": plan["tensor_conversion_count"],
    }


def _no_device_report(
    *,
    primitive: str,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
    dtype_seed: str,
    plan: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    report = _base_report(
        primitive=primitive,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": "no_device",
            "latency_ms": None,
            "error": NO_TTNN_DEVICE_MESSAGE,
            "detail": detail,
            "ttnn_version": None,
        }
    )
    return report


def _failed_report(
    *,
    primitive: str,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
    dtype_seed: str,
    plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
) -> dict[str, Any]:
    report = _base_report(
        primitive=primitive,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "error": message,
            "detail": detail,
            "ttnn_version": ttnn_version,
        }
    )
    return report


def _attention_primitive_reference(
    *,
    primitive: str,
    plan: dict[str, Any],
    output_shapes: dict[str, list[int] | None],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    checks = [
        _shape_check(
            f"output.{name}",
            output_shapes.get(name),
            expected=shape,
        )
        for name, shape in plan["expected_output_shapes"].items()
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "primitive": primitive,
        "observed_ops": observed_ops,
        "checks": checks,
    }


def _output_shapes(
    outputs: Any,
    *,
    expected_names: Any | None = None,
) -> dict[str, list[int] | None]:
    names = list(expected_names or [])
    if isinstance(outputs, tuple):
        if len(names) != len(outputs):
            names = ["query", "key", "value"] if len(outputs) == 3 else ["query", "key"]
        return {
            names[index]: _shape(tensor)
            for index, tensor in enumerate(outputs)
        }
    if len(names) == 1:
        return {names[0]: _shape(outputs)}
    return {"output": _shape(outputs)}


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _randn(torch: Any, shape: list[int], dtype_seed: str) -> Any:
    dtype = (
        getattr(torch, "bfloat16", None)
        if dtype_seed == "bf16"
        else getattr(torch, "float32", None)
    )
    try:
        return torch.randn(tuple(shape), dtype=dtype)
    except TypeError:
        return torch.randn(tuple(shape))


def _zeros(torch: Any, shape: list[int]) -> Any:
    dtype = getattr(torch, "int32", None)
    try:
        return torch.zeros(tuple(shape), dtype=dtype)
    except TypeError:
        return torch.zeros(tuple(shape))


def _ttnn_dtype(ttnn: Any, dtype_seed: str) -> Any:
    if dtype_seed == "bf16":
        return getattr(ttnn, "bfloat16", None)
    return getattr(ttnn, "float32", None)


def _memory_config(ttnn: Any) -> Any | None:
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


@contextmanager
def _maybe_managed_device(ttnn: Any, device_id: int, injected_ttnn: Any | None):
    if injected_ttnn is not None:
        yield f"fake_device:{device_id}"
        return
    with _managed_ttnn_device(ttnn, device_id) as device:
        yield device


def _validate_args(
    *,
    primitive: str,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
    dtype_seed: str,
) -> None:
    if primitive not in ATTENTION_PRIMITIVES:
        raise ValueError(
            f"primitive must be one of {list(ATTENTION_PRIMITIVES)}"
        )
    for name, value in (
        ("batch_size", batch_size),
        ("hidden_size", hidden_size),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("max_cache_len", max_cache_len),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if num_heads * head_dim != hidden_size:
        raise ValueError("hidden_size must equal num_heads * head_dim")
    if dtype_seed not in {"bf16", "fp32"}:
        raise ValueError("dtype_seed must be one of: bf16, fp32")


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
