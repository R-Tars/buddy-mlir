from __future__ import annotations

import copy
import importlib
import time
from pathlib import Path
from typing import Any, Mapping

from ..runtime_environment import collect_ttnn_environment
from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..runtime.device import managed_ttnn_device as _maybe_managed_device
from ..reports.contracts import ATTENTION_PRIMITIVES
from ..ttnn_compat import UnsupportedTTNNOp, ops as ttnn_ops
from .attention_support import (
    decode_head_shape as _decode_head_shape,
    decode_hidden_shape as _decode_hidden_shape,
    decode_rotary_cos_sin_shape as _decode_rotary_cos_sin_shape,
    decode_rotary_transform_shape as _decode_rotary_transform_shape,
    height_sharded_memory_config as _height_sharded_memory_config,
    input_tensor_contracts as _input_tensor_contracts,
    linear_weight_shape as _linear_weight_shape,
    memory_config as _memory_config,
    page_state_from_plan as _page_state_from_plan,
    realize_sdpa_runtime_config as _realize_sdpa_runtime_config,
    rotary_cos_sin_height_sharded_memory_config as _rotary_cos_sin_height_sharded_memory_config,
    rotary_transform_height_sharded_memory_config as _rotary_transform_height_sharded_memory_config,
    runtime_index_tensor as _runtime_index_tensor,
    sharded_height_memory_config as _sharded_height_memory_config,
    ttnn_dtype as _ttnn_dtype,
    validate_args as _validate_args,
    without_none as _without_none,
    zeros as _zeros,
)
from .support import (
    NUMERIC_REFERENCE_NOT_RUN_REASON,
    dry_run_reference as _dry_run_reference,
    observed_op_sequence as _observed_op_sequence,
    op_sequence_coverage_check as _op_sequence_coverage_check,
    output_shapes as _output_shapes,
    failed_diagnostic_report as _failed_diagnostic_report,
    shape_check as _shape_check,
    shape as _shape,
    write_report as _write_report,
    randn as _randn,
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
    sdpa_runtime_config: Mapping[str, Any] | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if (
        sdpa_runtime_config is not None
        and primitive != "paged_scaled_dot_product_attention_decode"
    ):
        raise ValueError(
            "sdpa_runtime_config is only valid for the paged SDPA primitive"
        )
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
        if sdpa_runtime_config is not None:
            report["sdpa_runtime_config"] = copy.deepcopy(
                dict(sdpa_runtime_config)
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
                sdpa_runtime_config=sdpa_runtime_config,
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

    if sdpa_runtime_config is not None:
        report["sdpa_runtime_config"] = copy.deepcopy(dict(sdpa_runtime_config))
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
    sdpa_runtime_config: Mapping[str, Any] | None = None,
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
    sdpa_config = _realize_sdpa_runtime_config(sdpa_runtime_config, ttnn)

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
            memory_config=sdpa_config.get(
                "kernel_output_memory_config",
                memory_config,
            ),
            program_config=sdpa_config.get("program_config"),
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


def _base_report(**kwargs: Any) -> dict[str, Any]:
    plan = kwargs["plan"]
    dtype_seed = kwargs["dtype_seed"]
    return {
        "schema_version": 1,
        "template": "attention_primitive",
        "primitive": kwargs["primitive"],
        "device": kwargs["device"],
        "device_id": kwargs["device_id"],
        "batch_size": kwargs["batch_size"],
        "hidden_size": kwargs["hidden_size"],
        "num_heads": kwargs["num_heads"],
        "num_kv_heads": kwargs["num_kv_heads"],
        "head_dim": kwargs["head_dim"],
        "max_cache_len": kwargs["max_cache_len"],
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "memory_config": "default_or_l1",
        "dry_run": kwargs["dry_run"],
        "input_shapes": plan["input_shapes"],
        "input_tensor_contracts": plan["input_tensor_contracts"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "output_shapes": None,
        "tensor_conversion_count": plan["tensor_conversion_count"],
        "ttnn_environment": collect_ttnn_environment(None),
    }


def _error_report(**kwargs: Any) -> dict[str, Any]:
    error_keys = {"status", "message", "detail", "ttnn_version", "ttnn_module"}
    base = _base_report(
        **{key: value for key, value in kwargs.items() if key not in error_keys},
        dry_run=False,
    )
    return _failed_diagnostic_report(
        base,
        status=kwargs["status"],
        message=kwargs["message"],
        detail=kwargs["detail"],
        ttnn_version=kwargs.get("ttnn_version"),
        ttnn_module=kwargs.get("ttnn_module"),
    )


def _no_device_report(**kwargs: Any) -> dict[str, Any]:
    kwargs.update(
        status="no_device",
        message=NO_TTNN_DEVICE_MESSAGE,
        ttnn_version=None,
        ttnn_module=None,
    )
    return _error_report(**kwargs)


def _failed_report(**kwargs: Any) -> dict[str, Any]:
    return _error_report(**kwargs)


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
