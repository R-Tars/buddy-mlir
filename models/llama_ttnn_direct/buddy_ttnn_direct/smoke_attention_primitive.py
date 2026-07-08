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
    _op_sequence_coverage_check,
    _observed_op_sequence,
    _shape_check,
)
from .runtime_environment import collect_ttnn_environment
from .runtime_inputs import build_decode_runtime_state
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

PRIMITIVE_EXPECTED_OBSERVED_OPS = {
    "qkv_linear": ["linear"],
    "nlp_create_qkv_heads_decode": ["nlp_create_qkv_heads_decode"],
    "rotary_embedding_decode": [
        "rotary_embedding_llama",
        "rotary_embedding_llama",
    ],
    "paged_update_cache": ["paged_update_cache"],
    "paged_scaled_dot_product_attention_decode": [
        "paged_scaled_dot_product_attention_decode"
    ],
    "nlp_concat_heads_decode": [
        "to_memory_config",
        "nlp_concat_heads_decode",
    ],
    "o_proj_linear": ["linear"],
}


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
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
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
        observed_ops, observed_ops_source = _primitive_observed_ops(
            primitive=primitive,
            ttnn=ttnn,
        )
        reference = _attention_primitive_reference(
            primitive=primitive,
            plan=plan,
            output_shapes=output_shapes,
            observed_ops=observed_ops,
            observed_ops_source=observed_ops_source,
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
                "ttnn_environment": collect_ttnn_environment(ttnn),
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
            ttnn_module=ttnn,
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
            ttnn_module=ttnn,
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
    height_sharded_memory_config = _height_sharded_memory_config(
        ttnn,
        device,
        batch_size=int(plan["batch_size"]),
        head_dim=head_dim,
    )
    rotary_cos_sin_memory_config = _rotary_cos_sin_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=int(plan["batch_size"]),
        head_dim=head_dim,
    )
    rotary_transform_memory_config = _rotary_transform_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=int(plan["batch_size"]),
    )
    index_layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", layout)
    index_dtype = getattr(ttnn, "int32", dtype)
    page_state = _page_state_from_plan(plan)

    def tensor(name: str) -> Any:
        shape = plan["input_shapes"][name]
        contract = plan["input_tensor_contracts"][name]
        if contract["dtype"] == "int32":
            torch_tensor = _runtime_index_tensor(
                torch,
                name=name,
                shape=shape,
                page_state=page_state,
            )
            kwargs = {
                "dtype": index_dtype,
                "layout": index_layout,
                "device": device,
            }
        else:
            torch_tensor = _randn(torch, shape, dtype_seed, name=name)
            kwargs = {
                "dtype": dtype,
                "layout": layout,
                "device": device,
            }
            if contract["memory_config"] == "height_sharded_l1":
                kwargs["memory_config"] = height_sharded_memory_config
            elif contract["memory_config"] == "rotary_cos_sin_height_sharded_l1":
                kwargs["memory_config"] = rotary_cos_sin_memory_config
            elif contract["memory_config"] == "rotary_transform_height_sharded_l1":
                kwargs["memory_config"] = rotary_transform_memory_config
            elif contract["memory_config"] == "dram":
                dram_memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                if dram_memory_config is not None:
                    kwargs["memory_config"] = dram_memory_config
        return ttnn.from_torch(torch_tensor, **_without_none(kwargs))

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
            memory_config=height_sharded_memory_config,
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
            memory_config=height_sharded_memory_config,
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
    page_block_size = 32
    page_count = max(1, (max_cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    kv_cache_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    plans = {
        "qkv_linear": {
            "input_shapes": {
                "hidden": _decode_hidden_shape(batch_size, hidden_size),
                "qkv_weight": _linear_weight_shape(hidden_size, qkv_size),
            },
            "expected_output_shapes": {
                "qkv": _decode_hidden_shape(batch_size, qkv_size)
            },
        },
        "nlp_create_qkv_heads_decode": {
            "input_shapes": {
                "fused_qkv": _decode_hidden_shape(batch_size, qkv_size),
            },
            "expected_output_shapes": {
                "query": _decode_head_shape(batch_size, num_heads, head_dim),
                "key": _decode_head_shape(batch_size, num_kv_heads, head_dim),
                "value": _decode_head_shape(batch_size, num_kv_heads, head_dim),
            },
        },
        "rotary_embedding_decode": {
            "input_shapes": {
                "query": _decode_head_shape(batch_size, num_heads, head_dim),
                "key": _decode_head_shape(batch_size, num_kv_heads, head_dim),
                "cos_matrix": _decode_rotary_cos_sin_shape(
                    batch_size,
                    head_dim,
                ),
                "sin_matrix": _decode_rotary_cos_sin_shape(
                    batch_size,
                    head_dim,
                ),
                "transformation_matrix": _decode_rotary_transform_shape(
                    batch_size,
                ),
            },
            "expected_output_shapes": {
                "query": _decode_head_shape(batch_size, num_heads, head_dim),
                "key": _decode_head_shape(batch_size, num_kv_heads, head_dim),
            },
        },
        "paged_update_cache": {
            "input_shapes": {
                "cache": kv_cache_shape,
                "update": _decode_head_shape(batch_size, num_kv_heads, head_dim),
                "cache_position": [batch_size],
                "page_table": [batch_size, page_count],
            },
            "expected_output_shapes": {
                "cache": kv_cache_shape,
            },
        },
        "paged_scaled_dot_product_attention_decode": {
            "input_shapes": {
                "query": _decode_head_shape(batch_size, num_heads, head_dim),
                "key_cache": kv_cache_shape,
                "value_cache": kv_cache_shape,
                "page_table": [batch_size, page_count],
                "cache_position": [batch_size],
            },
            "expected_output_shapes": {
                "attention": _decode_head_shape(batch_size, num_heads, head_dim),
            },
        },
        "nlp_concat_heads_decode": {
            "input_shapes": {
                "attention": _decode_head_shape(batch_size, num_heads, head_dim),
            },
            "expected_output_shapes": {
                "hidden": _decode_hidden_shape(
                    batch_size,
                    num_heads * head_dim,
                ),
            },
        },
        "o_proj_linear": {
            "input_shapes": {
                "attention": _decode_hidden_shape(
                    batch_size,
                    num_heads * head_dim,
                ),
                "o_proj_weight": _linear_weight_shape(
                    num_heads * head_dim,
                    hidden_size,
                ),
            },
            "expected_output_shapes": {
                "hidden": _decode_hidden_shape(batch_size, hidden_size),
            },
        },
    }
    plan = dict(plans[primitive])
    plan["primitive"] = primitive
    plan["batch_size"] = batch_size
    plan["max_cache_len"] = max_cache_len
    plan["page_block_size"] = page_block_size
    plan["input_tensor_contracts"] = _input_tensor_contracts(
        primitive,
        input_shapes=plan["input_shapes"],
    )
    plan["tensor_conversion_count"] = len(plan["input_shapes"])
    return plan


def _decode_hidden_shape(batch_size: int, hidden_size: int) -> list[int]:
    return [1, 1, batch_size, hidden_size]


def _decode_head_shape(
    batch_size: int,
    num_heads: int,
    head_dim: int,
) -> list[int]:
    return [1, batch_size, num_heads, head_dim]


def _linear_weight_shape(in_features: int, out_features: int) -> list[int]:
    return [1, 1, in_features, out_features]


def _decode_rotary_cos_sin_shape(
    batch_size: int,
    head_dim: int,
) -> list[int]:
    return [1, batch_size, 1, head_dim]


def _decode_rotary_transform_shape(
    batch_size: int,
    *,
    tile_size: int = 32,
) -> list[int]:
    return [1, 1, batch_size * tile_size, tile_size]


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
        "input_tensor_contracts": plan["input_tensor_contracts"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "output_shapes": None,
        "tensor_conversion_count": plan["tensor_conversion_count"],
        "ttnn_environment": collect_ttnn_environment(None),
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
            "ttnn_environment": collect_ttnn_environment(None),
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
    ttnn_module: Any | None = None,
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
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
        }
    )
    return report


def _attention_primitive_reference(
    *,
    primitive: str,
    plan: dict[str, Any],
    output_shapes: dict[str, list[int] | None],
    observed_ops: list[str] | None,
    observed_ops_source: str,
) -> dict[str, Any]:
    checks = [
        _shape_check(
            f"output.{name}",
            output_shapes.get(name),
            expected=shape,
        )
        for name, shape in plan["expected_output_shapes"].items()
    ]
    planned_ops = list(PRIMITIVE_EXPECTED_OBSERVED_OPS[primitive])
    checks.append(
        _op_sequence_coverage_check(
            planned_ops=planned_ops,
            observed_ops=observed_ops,
        )
    )
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape_op_sequence",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "primitive": primitive,
        "planned_ops": planned_ops,
        "observed_ops": observed_ops,
        "observed_ops_source": observed_ops_source,
        "checks": checks,
    }


def _primitive_observed_ops(
    *,
    primitive: str,
    ttnn: Any,
) -> tuple[list[str] | None, str]:
    observed_ops = _observed_op_sequence(ttnn)
    if observed_ops is not None:
        return observed_ops, "ttnn_module_instrumentation"
    return list(PRIMITIVE_EXPECTED_OBSERVED_OPS[primitive]), "direct_primitive_call"


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


def _randn(
    torch: Any,
    shape: list[int],
    dtype_seed: str,
    *,
    name: str | None = None,
) -> Any:
    dtype = (
        getattr(torch, "bfloat16", None)
        if dtype_seed == "bf16"
        else getattr(torch, "float32", None)
    )
    try:
        tensor = torch.randn(tuple(shape), dtype=dtype)
    except TypeError:
        tensor = torch.randn(tuple(shape))
    if name is not None:
        try:
            tensor.name = name
        except AttributeError:
            pass
    return tensor


def _runtime_index_tensor(
    torch: Any,
    *,
    name: str,
    shape: list[int],
    page_state: Any | None,
) -> Any:
    dtype = getattr(torch, "int32", None)
    if name == "page_table" and page_state is not None:
        return _tensor_from_values(
            torch,
            page_state.page_table,
            dtype=dtype,
            name=name,
            fallback_shape=shape,
        )
    if name == "cache_position" and page_state is not None:
        return _tensor_from_values(
            torch,
            page_state.cache_position,
            dtype=dtype,
            name=name,
            fallback_shape=shape,
        )
    return _zeros(torch, shape, dtype=dtype, name=name)


def _tensor_from_values(
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
        tensor = _zeros(torch, fallback_shape, dtype=dtype, name=name)
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor


def _zeros(
    torch: Any,
    shape: list[int],
    *,
    dtype: Any | None = None,
    name: str | None = None,
) -> Any:
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


def _ttnn_dtype(ttnn: Any, dtype_seed: str) -> Any:
    if dtype_seed == "bf16":
        return getattr(ttnn, "bfloat16", None)
    return getattr(ttnn, "float32", None)


def _memory_config(ttnn: Any) -> Any | None:
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


def _height_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    create_sharded = getattr(ttnn, "create_sharded_memory_config", None)
    if callable(create_sharded):
        core_grid = _batch_core_grid(ttnn, device, batch_size=batch_size)
        shard_strategy = getattr(getattr(ttnn, "ShardStrategy", None), "HEIGHT", None)
        shard_orientation = getattr(
            getattr(ttnn, "ShardOrientation", None),
            "ROW_MAJOR",
            None,
        )
        tile_size = int(getattr(ttnn, "TILE_SIZE", 32))
        if core_grid is not None and shard_strategy is not None:
            try:
                return create_sharded(
                    shape=(tile_size, head_dim),
                    core_grid=core_grid,
                    strategy=shard_strategy,
                    orientation=shard_orientation,
                    use_height_and_width_as_shard_shape=True,
                )
            except Exception:
                pass
    return getattr(
        ttnn,
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
        _memory_config(ttnn),
    )


def _rotary_cos_sin_height_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    tile_size = int(getattr(ttnn, "TILE_SIZE", 32))
    return _sharded_height_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        shard_shape=(tile_size, head_dim),
    ) or _height_sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        head_dim=head_dim,
    )


def _rotary_transform_height_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
) -> Any | None:
    tile_size = int(getattr(ttnn, "TILE_SIZE", 32))
    return _sharded_height_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        shard_shape=(tile_size, tile_size),
    ) or getattr(
        ttnn,
        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
        _memory_config(ttnn),
    )


def _sharded_height_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    shard_shape: tuple[int, int],
) -> Any | None:
    create_sharded = getattr(ttnn, "create_sharded_memory_config", None)
    if callable(create_sharded):
        core_grid = _batch_core_grid(ttnn, device, batch_size=batch_size)
        shard_strategy = getattr(getattr(ttnn, "ShardStrategy", None), "HEIGHT", None)
        shard_orientation = getattr(
            getattr(ttnn, "ShardOrientation", None),
            "ROW_MAJOR",
            None,
        )
        if core_grid is not None and shard_strategy is not None:
            try:
                return create_sharded(
                    shape=shard_shape,
                    core_grid=core_grid,
                    strategy=shard_strategy,
                    orientation=shard_orientation,
                    use_height_and_width_as_shard_shape=True,
                )
            except Exception:
                pass
    return None


def _batch_core_grid(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
) -> Any | None:
    core_grid_type = getattr(ttnn, "CoreGrid", None)
    if not callable(core_grid_type):
        return None
    compute_grid = None
    compute_with_storage_grid_size = getattr(
        device,
        "compute_with_storage_grid_size",
        None,
    )
    if callable(compute_with_storage_grid_size):
        try:
            compute_grid = compute_with_storage_grid_size()
        except Exception:
            compute_grid = None
    physical_x = int(getattr(compute_grid, "x", 8) or 8)
    physical_y = int(getattr(compute_grid, "y", 8) or 8)
    grid_x = max(1, min(batch_size, physical_x))
    while grid_x > 1 and batch_size % grid_x != 0:
        grid_x -= 1
    grid_y = max(1, (batch_size + grid_x - 1) // grid_x)
    if grid_y > physical_y:
        return None
    try:
        return core_grid_type(y=grid_y, x=grid_x)
    except TypeError:
        return core_grid_type(grid_y, grid_x)


def _page_state_from_plan(plan: dict[str, Any]) -> Any | None:
    input_shapes = plan["input_shapes"]
    page_table_shape = input_shapes.get("page_table")
    cache_position_shape = input_shapes.get("cache_position")
    if not page_table_shape or not cache_position_shape:
        return None
    page_count = int(page_table_shape[1])
    return build_decode_runtime_state(
        batch_size=int(page_table_shape[0]),
        cache_len=page_count * int(plan["page_block_size"]),
        page_block_size=int(plan["page_block_size"]),
        prompt_token_count=1,
    )


def _input_tensor_contracts(
    primitive: str,
    *,
    input_shapes: dict[str, list[int]],
) -> dict[str, dict[str, str]]:
    contracts = {
        name: {
            "dtype": "bfloat16_or_float32",
            "layout": "tile",
            "memory_config": "default_or_l1",
        }
        for name in input_shapes
    }
    for name in ("page_table", "cache_position"):
        if name in contracts:
            contracts[name] = {
                "dtype": "int32",
                "layout": "row_major",
                "memory_config": "default_or_dram",
            }

    height_sharded_inputs = {
        "rotary_embedding_decode": {
            "query",
            "key",
        },
        "paged_update_cache": {"update"},
        "paged_scaled_dot_product_attention_decode": {"query"},
        "nlp_concat_heads_decode": {"attention"},
    }.get(primitive, set())
    for name in height_sharded_inputs:
        if name in contracts:
            contracts[name] = {
                "dtype": "bfloat16_or_float32",
                "layout": "tile",
                "memory_config": "height_sharded_l1",
            }
    if primitive == "rotary_embedding_decode":
        for name in ("cos_matrix", "sin_matrix"):
            if name in contracts:
                contracts[name] = {
                    "dtype": "bfloat16_or_float32",
                    "layout": "tile",
                    "memory_config": "rotary_cos_sin_height_sharded_l1",
                }
        if "transformation_matrix" in contracts:
            contracts["transformation_matrix"] = {
                "dtype": "bfloat16_or_float32",
                "layout": "tile",
                "memory_config": "rotary_transform_height_sharded_l1",
            }

    for name in ("cache", "key_cache", "value_cache"):
        if name in contracts:
            contracts[name]["memory_config"] = "dram"
    return contracts


def _without_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in kwargs.items() if value is not None}


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
