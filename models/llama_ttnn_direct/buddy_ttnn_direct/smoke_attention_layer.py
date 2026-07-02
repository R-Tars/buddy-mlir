from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .smoke_attention_primitive import (
    _memory_config,
    _maybe_managed_device,
    _randn,
    _ttnn_dtype,
    _validate_args,
    _zeros,
)
from .smoke_decode_shell import (
    NUMERIC_REFERENCE_NOT_RUN_REASON,
    _dry_run_reference,
    _shape_check,
    _value_check,
)
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .templates import ttnn_ops
from .templates.ttnn_ops import UnsupportedTTNNOp


ATTENTION_LAYER_OPS = [
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache.k",
    "paged_update_cache.v",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
]


def run_smoke_attention_layer(
    *,
    out: str | Path,
    program_dir: str | Path,
    layer: int,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    num_layers = int(config["num_layers"])
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"layer must be in [0, {num_layers})")

    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    _validate_args(
        primitive="qkv_linear",
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_len=cache_len,
        dtype_seed=dtype_seed,
    )
    plan = _attention_layer_plan(
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_len=cache_len,
    )

    if dry_run:
        report = _base_report(
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "primitive_reports": [
                    _planned_op_report(
                        op_name,
                        plan,
                        dtype=_dtype_name(dtype_seed),
                    )
                    for op_name in ATTENTION_LAYER_OPS
                ],
                "output_shapes": None,
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "memory_config_conversion_count": 1,
                "error": None,
                "ttnn_version": None,
                "reference": _dry_run_reference("attention_layer"),
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
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
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
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="missing_torch",
            message="torch is required for attention layer smoke tensors.",
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            start = time.perf_counter()
            layer_result = _run_attention_layer(
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
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=False,
            plan=plan,
        )
        output_shapes = {
            "attention_output": _shape(layer_result["output"]),
            "key_cache": _shape(layer_result["key_cache"]),
            "value_cache": _shape(layer_result["value_cache"]),
        }
        reference = _attention_layer_reference(
            plan=plan,
            output_shapes=output_shapes,
            primitive_reports=layer_result["primitive_reports"],
        )
        passed = bool(reference["passed"])
        report.update(
            {
                "passed": passed,
                "status": "passed" if passed else "reference_mismatch",
                "latency_ms": latency_ms,
                "primitive_reports": layer_result["primitive_reports"],
                "output_shapes": output_shapes,
                "tensor_conversion_count": layer_result["tensor_conversion_count"],
                "memory_config_conversion_count": (
                    layer_result["memory_config_conversion_count"]
                ),
                "error": (
                    None
                    if passed
                    else "attention layer structural reference mismatch"
                ),
                "ttnn_version": getattr(ttnn, "__version__", None),
                "reference": reference,
            }
        )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            detail=str(err),
        )
    except UnsupportedTTNNOp as err:
        report = _failed_report(
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except Exception as err:
        report = _failed_report(
            program_dir=program_root,
            layer=layer,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )

    _write_report(out, report)
    return report


def _run_attention_layer(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    dtype = _ttnn_dtype(ttnn, dtype_seed)
    dtype_name = _dtype_name(dtype_seed)
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    memory_config = _memory_config(ttnn)
    tensor_conversion_count = 0
    memory_config_conversion_count = 0

    def tensor(name: str) -> Any:
        nonlocal tensor_conversion_count
        shape = plan["input_shapes"][name]
        torch_tensor = (
            _zeros(torch, shape)
            if name in {"page_table", "cache_position"}
            else _randn(torch, shape, dtype_seed)
        )
        tensor_conversion_count += 1
        return ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
        )

    primitive_reports: list[dict[str, Any]] = []

    hidden = tensor("hidden")
    qkv_weight = tensor("qkv_weight")
    qkv = _time_op(
        primitive_reports,
        "qkv_linear",
        lambda: ttnn.linear(hidden, qkv_weight, memory_config=memory_config),
        input_shapes=_op_input_shapes("qkv_linear", plan),
        expected_output_shapes=_op_expected_output_shapes("qkv_linear", plan),
        dtype=dtype_name,
        memory_config=memory_config,
    )
    q, k, v = _time_op(
        primitive_reports,
        "nlp_create_qkv_heads_decode",
        lambda: ttnn_ops.nlp_create_qkv_heads_decode(
            ttnn,
            qkv,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            memory_config=memory_config,
        ),
        input_shapes=_op_input_shapes("nlp_create_qkv_heads_decode", plan),
        expected_output_shapes=_op_expected_output_shapes(
            "nlp_create_qkv_heads_decode",
            plan,
        ),
        dtype=dtype_name,
        memory_config=memory_config,
    )

    cos = tensor("cos_matrix")
    sin = tensor("sin_matrix")
    transform = tensor("transformation_matrix")
    q, k = _time_op(
        primitive_reports,
        "rotary_embedding_decode",
        lambda: ttnn_ops.rotary_embedding_decode(
            ttnn,
            q,
            k,
            cos_matrix=cos,
            sin_matrix=sin,
            transformation_matrix=transform,
        ),
        input_shapes=_op_input_shapes("rotary_embedding_decode", plan),
        expected_output_shapes=_op_expected_output_shapes(
            "rotary_embedding_decode",
            plan,
        ),
        dtype=dtype_name,
        memory_config=None,
    )

    key_cache = tensor("key_cache")
    value_cache = tensor("value_cache")
    page_table = tensor("page_table")
    cache_position = tensor("cache_position")
    key_cache = _time_op(
        primitive_reports,
        "paged_update_cache.k",
        lambda: ttnn_ops.paged_update_cache(
            ttnn,
            key_cache,
            k,
            update_idxs_tensor=cache_position,
            page_table=page_table,
        ),
        input_shapes=_op_input_shapes("paged_update_cache.k", plan),
        expected_output_shapes=_op_expected_output_shapes(
            "paged_update_cache.k",
            plan,
        ),
        dtype=dtype_name,
        memory_config=None,
    )
    value_cache = _time_op(
        primitive_reports,
        "paged_update_cache.v",
        lambda: ttnn_ops.paged_update_cache(
            ttnn,
            value_cache,
            v,
            update_idxs_tensor=cache_position,
            page_table=page_table,
        ),
        input_shapes=_op_input_shapes("paged_update_cache.v", plan),
        expected_output_shapes=_op_expected_output_shapes(
            "paged_update_cache.v",
            plan,
        ),
        dtype=dtype_name,
        memory_config=None,
    )
    attention = _time_op(
        primitive_reports,
        "paged_scaled_dot_product_attention_decode",
        lambda: ttnn_ops.paged_sdpa_decode(
            ttnn,
            q,
            key_cache,
            value_cache,
            page_table,
            cache_position,
            scale=float(head_dim) ** -0.5,
            memory_config=memory_config,
        ),
        input_shapes=_op_input_shapes(
            "paged_scaled_dot_product_attention_decode",
            plan,
        ),
        expected_output_shapes=_op_expected_output_shapes(
            "paged_scaled_dot_product_attention_decode",
            plan,
        ),
        dtype=dtype_name,
        memory_config=memory_config,
    )
    concat = _time_op(
        primitive_reports,
        "nlp_concat_heads_decode",
        lambda: ttnn_ops.nlp_concat_heads_decode(
            ttnn,
            attention,
            num_heads=num_heads,
            memory_config=memory_config,
        ),
        input_shapes=_op_input_shapes("nlp_concat_heads_decode", plan),
        expected_output_shapes=_op_expected_output_shapes(
            "nlp_concat_heads_decode",
            plan,
        ),
        dtype=dtype_name,
        memory_config=memory_config,
    )
    if memory_config is not None:
        memory_config_conversion_count += 1

    o_proj_weight = tensor("o_proj_weight")
    output = _time_op(
        primitive_reports,
        "o_proj_linear",
        lambda: ttnn.linear(concat, o_proj_weight, memory_config=memory_config),
        input_shapes=_op_input_shapes("o_proj_linear", plan),
        expected_output_shapes=_op_expected_output_shapes("o_proj_linear", plan),
        dtype=dtype_name,
        memory_config=memory_config,
    )

    return {
        "output": output,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "primitive_reports": primitive_reports,
        "tensor_conversion_count": tensor_conversion_count,
        "memory_config_conversion_count": memory_config_conversion_count,
    }


def _time_op(
    reports: list[dict[str, Any]],
    name: str,
    fn: Callable[[], Any],
    *,
    input_shapes: dict[str, list[int]],
    expected_output_shapes: dict[str, list[int]],
    dtype: str,
    memory_config: Any | None,
) -> Any:
    start = time.perf_counter()
    output = fn()
    output_shapes = _output_shapes(
        output,
        expected_names=expected_output_shapes.keys(),
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    reports.append(
        {
            "primitive": name,
            "status": "passed",
            "latency_ms": latency_ms,
            "input_shapes": input_shapes,
            "expected_output_shapes": expected_output_shapes,
            "output_shapes": output_shapes,
            "dtype": dtype,
            "layout": "tile",
            "memory_config": _string_or_none(memory_config),
        }
    )
    return output


def _attention_layer_plan(
    *,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cache_len: int,
) -> dict[str, Any]:
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    page_count = max(1, (cache_len + 31) // 32)
    input_shapes = {
        "hidden": [batch_size, 1, hidden_size],
        "qkv_weight": [hidden_size, qkv_size],
        "cos_matrix": [1, 1, head_dim, head_dim],
        "sin_matrix": [1, 1, head_dim, head_dim],
        "transformation_matrix": [1, 1, head_dim, head_dim],
        "key_cache": [batch_size, cache_len, num_kv_heads, head_dim],
        "value_cache": [batch_size, cache_len, num_kv_heads, head_dim],
        "page_table": [batch_size, page_count],
        "cache_position": [batch_size],
        "o_proj_weight": [num_heads * head_dim, hidden_size],
    }
    return {
        "input_shapes": input_shapes,
        "expected_intermediate_shapes": {
            "qkv": [batch_size, 1, qkv_size],
            "query": [batch_size, num_heads, 1, head_dim],
            "key": [batch_size, num_kv_heads, 1, head_dim],
            "value": [batch_size, num_kv_heads, 1, head_dim],
            "attention": [batch_size, num_heads, 1, head_dim],
            "concat_heads": [batch_size, 1, num_heads * head_dim],
        },
        "expected_output_shapes": {
            "attention_output": [batch_size, 1, hidden_size],
            "key_cache": [batch_size, cache_len, num_kv_heads, head_dim],
            "value_cache": [batch_size, cache_len, num_kv_heads, head_dim],
        },
        "tensor_conversion_count": len(input_shapes),
    }


def _planned_op_report(
    name: str,
    plan: dict[str, Any],
    *,
    dtype: str,
) -> dict[str, Any]:
    return {
        "primitive": name,
        "status": "planned",
        "latency_ms": 0.0,
        "input_shapes": _op_input_shapes(name, plan),
        "expected_output_shapes": _op_expected_output_shapes(name, plan),
        "output_shapes": None,
        "dtype": dtype,
        "layout": "tile",
        "memory_config": "default_or_l1",
    }


def _op_input_shapes(name: str, plan: dict[str, Any]) -> dict[str, list[int]]:
    shapes = {
        **plan["input_shapes"],
        **plan["expected_intermediate_shapes"],
    }
    inputs_by_op = {
        "qkv_linear": ["hidden", "qkv_weight"],
        "nlp_create_qkv_heads_decode": ["qkv"],
        "rotary_embedding_decode": [
            "query",
            "key",
            "cos_matrix",
            "sin_matrix",
            "transformation_matrix",
        ],
        "paged_update_cache.k": [
            "key_cache",
            "key",
            "cache_position",
            "page_table",
        ],
        "paged_update_cache.v": [
            "value_cache",
            "value",
            "cache_position",
            "page_table",
        ],
        "paged_scaled_dot_product_attention_decode": [
            "query",
            "key_cache",
            "value_cache",
            "page_table",
            "cache_position",
        ],
        "nlp_concat_heads_decode": ["attention"],
        "o_proj_linear": ["concat_heads", "o_proj_weight"],
    }
    return {
        input_name: list(shapes[input_name])
        for input_name in inputs_by_op[name]
    }


def _op_expected_output_shapes(
    name: str,
    plan: dict[str, Any],
) -> dict[str, list[int]]:
    shapes = {
        **plan["expected_intermediate_shapes"],
        **plan["expected_output_shapes"],
        "key_cache": plan["input_shapes"]["key_cache"],
        "value_cache": plan["input_shapes"]["value_cache"],
    }
    outputs_by_op = {
        "qkv_linear": ["qkv"],
        "nlp_create_qkv_heads_decode": ["query", "key", "value"],
        "rotary_embedding_decode": ["query", "key"],
        "paged_update_cache.k": ["key_cache"],
        "paged_update_cache.v": ["value_cache"],
        "paged_scaled_dot_product_attention_decode": ["attention"],
        "nlp_concat_heads_decode": ["concat_heads"],
        "o_proj_linear": ["attention_output"],
    }
    return {
        output_name: list(shapes[output_name])
        for output_name in outputs_by_op[name]
    }


def _base_report(
    *,
    program_dir: Path,
    layer: int,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cache_len: int,
    dtype_seed: str,
    dry_run: bool,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "attention_layer",
        "program_dir": str(program_dir),
        "layer": layer,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "cache_len": cache_len,
        "dtype_seed": dtype_seed,
        "dtype": _dtype_name(dtype_seed),
        "layout": "tile",
        "memory_config": "default_or_l1",
        "dry_run": dry_run,
        "op_sequence": list(ATTENTION_LAYER_OPS),
        "input_shapes": plan["input_shapes"],
        "expected_intermediate_shapes": plan["expected_intermediate_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
    }


def _no_device_report(
    *,
    program_dir: Path,
    layer: int,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cache_len: int,
    dtype_seed: str,
    plan: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layer=layer,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": "no_device",
            "latency_ms": None,
            "primitive_reports": [],
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "memory_config_conversion_count": 0,
            "error": NO_TTNN_DEVICE_MESSAGE,
            "detail": detail,
            "ttnn_version": None,
        }
    )
    return report


def _failed_report(
    *,
    program_dir: Path,
    layer: int,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cache_len: int,
    dtype_seed: str,
    plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layer=layer,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "primitive_reports": [],
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "memory_config_conversion_count": 0,
            "error": message,
            "detail": detail,
            "ttnn_version": ttnn_version,
        }
    )
    return report


def _attention_layer_reference(
    *,
    plan: dict[str, Any],
    output_shapes: dict[str, list[int] | None],
    primitive_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = [
        _value_check(
            "primitive_count",
            len(primitive_reports),
            len(ATTENTION_LAYER_OPS),
        ),
        _value_check(
            "primitive_sequence",
            [report.get("primitive") for report in primitive_reports],
            list(ATTENTION_LAYER_OPS),
        ),
    ]
    for report in primitive_reports:
        primitive = str(report.get("primitive"))
        expected_shapes = report.get("expected_output_shapes") or {}
        output_shapes_by_name = report.get("output_shapes") or {}
        for name, shape in expected_shapes.items():
            checks.append(
                _shape_check(
                    f"primitive.{primitive}.{name}",
                    output_shapes_by_name.get(name),
                    expected=shape,
                )
            )
    for name, shape in plan["expected_output_shapes"].items():
        checks.append(
            _shape_check(
                f"output.{name}",
                output_shapes.get(name),
                expected=shape,
            )
        )
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "planned_ops": list(ATTENTION_LAYER_OPS),
        "checks": checks,
    }


def _output_shapes(
    output: Any,
    *,
    expected_names: Any | None = None,
) -> dict[str, list[int] | None]:
    names = list(expected_names or [])
    if isinstance(output, tuple):
        if len(names) != len(output):
            names = (
                ["query", "key", "value"]
                if len(output) == 3
                else ["query", "key"]
            )
        return {
            names[index]: _shape(tensor)
            for index, tensor in enumerate(output)
        }
    if len(names) == 1:
        return {names[0]: _shape(output)}
    return {"output": _shape(output)}


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _string_or_none(value: Any | None) -> str | None:
    return None if value is None else str(value)


def _dtype_name(dtype_seed: str) -> str:
    return "bfloat16" if dtype_seed == "bf16" else "float32"


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
