from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .runtime_environment import collect_ttnn_environment
from .smoke_attention_primitive import _maybe_managed_device
from .smoke_decode_shell import (
    _dry_run_reference,
    _dtype,
    _dtype_check,
    _generated_observed_op_sequence,
    _load_generated_model,
    _op_sequence_coverage_check,
    _runtime_int_tensor,
    _shape,
    _shape_check,
    _to_namespace,
    _value_check,
)
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .smoke_single_layer_decode import (
    _embedding_weight_shape,
    _lm_head_split_shapes,
    _linear_weight_shape,
    _norm_weight_shape,
    _synthetic_tensor_factory,
    _write_report,
)
from .ttnn_compat import UnsupportedTTNNOp


PREFILL_LAYER_OPS = [
    "rms_norm.attn",
    "qkv_linear",
    "split_query_key_value_heads_prefill",
    "rotary_embedding_prefill",
    "scaled_dot_product_attention",
    "fill_cache.k",
    "fill_cache.v",
    "concat_heads_prefill",
    "o_proj_linear",
    "residual_add.attn",
    "rms_norm.mlp",
    "mlp_gate",
    "mlp_up",
    "mul_silu",
    "mlp_down",
    "residual_add.mlp",
]

PREFILL_FINAL_OPS = [
    "rms_norm.final",
    "split_lm_head",
    "argmax_or_sampling",
]


def run_smoke_prefill(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int = 1,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    parameters: Any | None = None,
    token_ids: Any | None = None,
    kv_cache: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers)
    num_layers = int(config["num_layers"])
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if layer_count > num_layers:
        raise ValueError(
            f"layers must be <= generated config num_layers ({num_layers})"
        )
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config["seq_len"]
    )
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")
    plan = _prefill_plan(
        layers=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        config=config,
    )

    if dry_run:
        report = _base_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "prefill_status": "dry_run",
                "kv_cache_source": "prefill",
                "latency_ms": 0.0,
                "output_shapes": None,
                "cache_population": _planned_cache_population(plan),
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "error": None,
                "ttnn_version": None,
                "reference": _dry_run_reference("prefill_smoke"),
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
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="no_device",
            message=NO_TTNN_DEVICE_MESSAGE,
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
        if parameters is None or token_ids is None or kv_cache is None:
            report = _unavailable_report(
                program_dir=program_root,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                prefill_len=prefill_len,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                plan=plan,
                status="missing_torch",
                message="torch is required to synthesize prefill tensors.",
                detail=str(err),
                ttnn_version=getattr(ttnn, "__version__", None),
                ttnn_module=ttnn,
            )
            _write_report(out, report)
            return report
        torch = None

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            if parameters is None:
                assert torch is not None
                state = _build_synthetic_prefill_state(
                    ttnn=ttnn,
                    torch=torch,
                    device=ttnn_device,
                    dtype_seed=dtype_seed,
                    plan=plan,
                )
                parameters = state.parameters
                token_ids = state.token_ids
                kv_cache = state.kv_cache
                tensor_conversion_count = state.tensor_conversion_count
                parameter_source = "synthetic"
                input_source = "synthetic"
            else:
                if token_ids is None or kv_cache is None:
                    raise ValueError(
                        "token_ids and kv_cache are required with injected "
                        "prefill parameters"
                    )
                tensor_conversion_count = 0
                parameter_source = "injected"
                input_source = "injected"

            report = _run_generated_prefill(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=parameters,
                config=config,
                layer_count=layer_count,
                device=ttnn_device,
                token_ids=token_ids,
                kv_cache=kv_cache,
                tensor_conversion_count=tensor_conversion_count,
                plan=plan,
            )
            report.update(
                {
                    **_base_report(
                        program_dir=program_root,
                        layers=layer_count,
                        device=device,
                        device_id=device_id,
                        batch_size=batch_size,
                        prefill_len=prefill_len,
                        cache_len=cache_len,
                        dtype_seed=dtype_seed,
                        dry_run=False,
                        plan=plan,
                    ),
                    **report,
                    "parameter_source": parameter_source,
                    "input_source": input_source,
                }
            )
    except NoTTNNDeviceError as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="no_device",
            message=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
    except UnsupportedTTNNOp as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
    except Exception as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
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


def _run_generated_prefill(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    layer_count: int,
    device: Any,
    token_ids: Any,
    kv_cache: Any,
    tensor_conversion_count: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    prefill_config = dict(config)
    prefill_config["num_layers"] = layer_count
    prefill_config["batch_size"] = int(plan["batch_size"])
    prefill_config["max_cache_len"] = int(plan["cache_len"])
    prefill_config["prefill"] = dict(prefill_config.get("prefill") or {})
    prefill_config["prefill"]["seq_len"] = int(plan["prefill_len"])
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(prefill_config),
    )

    start = time.perf_counter()
    token, kv_cache, cache_reports = model.prefill_prompt(
        token_ids,
        kv_cache,
        valid_seq_len=int(plan["prefill_len"]),
    )
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    output_shapes = {
        "token": _shape(token),
        "key_cache": _shape(kv_cache[0].k),
        "value_cache": _shape(kv_cache[0].v),
        "kv_cache_layers": [
            {
                "layer_id": layer_id,
                "key_cache": _shape(layer_cache.k),
                "value_cache": _shape(layer_cache.v),
            }
            for layer_id, layer_cache in enumerate(kv_cache[:layer_count])
        ],
    }
    reference = _prefill_reference(
        plan=plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output={"kind": "token", "shape": _shape(token), "dtype": _dtype(token)},
        observed_ops=_generated_observed_op_sequence(model, ttnn),
    )
    passed = bool(reference["passed"])
    return {
        "passed": passed,
        "status": "passed" if passed else "reference_mismatch",
        "prefill_status": "passed" if passed else "reference_mismatch",
        "kv_cache_source": "prefill",
        "latency_ms": latency_ms,
        "output_shapes": output_shapes,
        "cache_population": _observed_cache_population(
            plan=plan,
            cache_reports=cache_reports,
            output_shapes=output_shapes,
        ),
        "output": {
            "kind": "token",
            "shape": _shape(token),
            "dtype": _dtype(token),
            "repr": repr(token),
        },
        "tensor_conversion_count": tensor_conversion_count,
        "error": None if passed else "prefill structural reference mismatch",
        "ttnn_version": getattr(ttnn, "__version__", None),
        "ttnn_environment": collect_ttnn_environment(ttnn),
        "reference": reference,
    }


def _build_synthetic_prefill_state(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
) -> SimpleNamespace:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    inputs = plan["input_shapes"]
    params = plan["parameter_shapes"]
    layer_params = plan["layer_parameter_shapes"]
    lm_head_splits = []
    for shard_id, shape in enumerate(params["lm_head_splits"]):
        lm_head_splits.append(
            SimpleNamespace(
                shard_id=shard_id,
                weight=tensor(shape, name=f"lm_head_{shard_id}"),
            )
        )

    parameters = SimpleNamespace(
        embedding=SimpleNamespace(
            weight=tensor(params["embedding"], name="embedding")
        ),
        layers=[],
        final_norm=SimpleNamespace(
            weight=tensor(params["final_norm"], name="final_norm")
        ),
        lm_head=SimpleNamespace(splits=lm_head_splits),
    )
    for layer_id in range(int(plan["layers"])):
        parameters.layers.append(
            SimpleNamespace(
                attention=SimpleNamespace(
                    wqkv_packed=SimpleNamespace(
                        weight=tensor(
                            layer_params["attention_wqkv"],
                            name=f"layers.{layer_id}.attention_wqkv",
                        )
                    ),
                    o_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["attention_o_proj"],
                            name=f"layers.{layer_id}.o_proj",
                        )
                    ),
                    rotary=SimpleNamespace(
                        cos_matrix=tensor(
                            layer_params["rotary_cos_matrix"],
                            name=f"layers.{layer_id}.rotary_cos",
                        ),
                        sin_matrix=tensor(
                            layer_params["rotary_sin_matrix"],
                            name=f"layers.{layer_id}.rotary_sin",
                        ),
                        transformation_matrix=tensor(
                            layer_params["rotary_transformation_matrix"],
                            name=f"layers.{layer_id}.rotary_transform",
                        ),
                    ),
                ),
                input_norm=SimpleNamespace(
                    weight=tensor(
                        layer_params["input_norm"],
                        name=f"layers.{layer_id}.input_norm",
                    )
                ),
                post_attention_norm=SimpleNamespace(
                    weight=tensor(
                        layer_params["post_attention_norm"],
                        name=f"layers.{layer_id}.post_attention_norm",
                    )
                ),
                mlp=SimpleNamespace(
                    gate_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_gate"],
                            name=f"layers.{layer_id}.mlp_gate",
                        )
                    ),
                    up_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_up"],
                            name=f"layers.{layer_id}.mlp_up",
                        )
                    ),
                    down_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_down"],
                            name=f"layers.{layer_id}.mlp_down",
                        )
                    ),
                ),
            )
        )
    host_token_ids = _runtime_int_tensor(
        torch,
        [[0 for _ in range(int(plan["prefill_len"]))] for _ in range(int(plan["batch_size"]))],
        name="prefill_token_ids",
    )
    kwargs = {"device": device}
    dtype = getattr(
        ttnn,
        "uint32",
        getattr(ttnn, "int32", getattr(ttnn, "bfloat16", None)),
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    token_ids = ttnn.from_torch(host_token_ids, **kwargs)
    kv_cache = []
    for layer_id in range(int(plan["layers"])):
        kv_cache.append(
            SimpleNamespace(
                k=tensor(
                    inputs["key_cache"],
                    name=f"layers.{layer_id}.key_cache",
                    zeros=True,
                ),
                v=tensor(
                    inputs["value_cache"],
                    name=f"layers.{layer_id}.value_cache",
                    zeros=True,
                ),
            )
        )
    return SimpleNamespace(
        parameters=parameters,
        token_ids=token_ids,
        kv_cache=kv_cache,
        tensor_conversion_count=tensor_count() + 1,
    )


def _prefill_plan(
    *,
    layers: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab_size = int(config["vocab_size"])
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    kv_cache_config = config.get("kv_cache") or {}
    if not isinstance(kv_cache_config, dict):
        kv_cache_config = {}
    page_block_size = int(kv_cache_config.get("page_block_size", 32))
    page_count = max(1, (cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    kv_cache_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    lm_head_splits = _lm_head_split_shapes(config, hidden_size, vocab_size)
    input_shapes = {
        "token_ids": [batch_size, prefill_len],
        "key_cache": kv_cache_shape,
        "value_cache": kv_cache_shape,
    }
    layer_parameter_shapes = {
        "input_norm": _norm_weight_shape(hidden_size),
        "post_attention_norm": _norm_weight_shape(hidden_size),
        "attention_wqkv": _linear_weight_shape(hidden_size, qkv_size),
        "attention_o_proj": _linear_weight_shape(
            num_heads * head_dim,
            hidden_size,
        ),
        "rotary_cos_matrix": [1, 1, prefill_len, head_dim],
        "rotary_sin_matrix": [1, 1, prefill_len, head_dim],
        "rotary_transformation_matrix": _prefill_rotary_transform_shape(),
        "mlp_gate": _linear_weight_shape(hidden_size, intermediate_size),
        "mlp_up": _linear_weight_shape(hidden_size, intermediate_size),
        "mlp_down": _linear_weight_shape(intermediate_size, hidden_size),
    }
    parameter_shapes = {
        "embedding": _embedding_weight_shape(vocab_size, hidden_size),
        **layer_parameter_shapes,
        "final_norm": _norm_weight_shape(hidden_size),
        "lm_head_splits": lm_head_splits,
    }
    return {
        "layers": layers,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "vocab_size": vocab_size,
        "input_shapes": input_shapes,
        "parameter_shapes": parameter_shapes,
        "layer_parameter_shapes": layer_parameter_shapes,
        "rotary": dict(config.get("rotary") or {}),
        "expected_intermediate_shapes": {
            "embedding": [batch_size, prefill_len, hidden_size],
            "qkv": [batch_size, prefill_len, qkv_size],
            "query": [batch_size, num_heads, prefill_len, head_dim],
            "key": [batch_size, num_kv_heads, prefill_len, head_dim],
            "value": [batch_size, num_kv_heads, prefill_len, head_dim],
            "attention": [batch_size, num_heads, prefill_len, head_dim],
            "concat_heads": [batch_size, prefill_len, num_heads * head_dim],
            "attention_output": [batch_size, prefill_len, hidden_size],
            "mlp_intermediate": [batch_size, prefill_len, intermediate_size],
        },
        "expected_output_shapes": {
            "token": [batch_size, 1],
            "key_cache": kv_cache_shape,
            "value_cache": kv_cache_shape,
        },
        "kv_cache": {
            "policy": kv_cache_config.get("policy", "paged"),
            "template": kv_cache_config.get("template", "paged_kv_cache"),
            "page_block_size": page_block_size,
            "page_count": page_count,
            "max_num_blocks": max_num_blocks,
            "physical_shape": kv_cache_shape,
            "logical_shape": [batch_size, cache_len, num_kv_heads, head_dim],
            "source": "prefill",
            "write_policy": "fill_cache_per_user",
            "planned_user_count": batch_size,
        },
        "tensor_conversion_count": 4 + len(lm_head_splits) + 12 * layers,
        "op_sequence": _prefill_op_sequence(layers),
    }


def _prefill_op_sequence(layers: int) -> list[str]:
    ops = ["embedding"]
    for _ in range(layers):
        ops.extend(PREFILL_LAYER_OPS)
    ops.extend(PREFILL_FINAL_OPS)
    return ops


def _prefill_rotary_transform_shape() -> list[int]:
    return [1, 1, 32, 32]


def _base_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    dtype_seed: str,
    dry_run: bool,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "prefill_smoke",
        "program_dir": str(program_dir),
        "layers": layers,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": dry_run,
        "op_sequence": plan["op_sequence"],
        "input_shapes": plan["input_shapes"],
        "parameter_shapes": plan["parameter_shapes"],
        "layer_parameter_shapes": plan["layer_parameter_shapes"],
        "expected_intermediate_shapes": plan["expected_intermediate_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "kv_cache": plan["kv_cache"],
        "model_semantics": "prefill_populates_kv_cache_for_decode",
        "ttnn_environment": collect_ttnn_environment(None),
    }


def _unavailable_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    dtype_seed: str,
    plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
    ttnn_module: Any | None = None,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layers=layers,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "prefill_status": status,
            "kv_cache_source": "prefill",
            "latency_ms": None,
            "output_shapes": None,
            "cache_population": _planned_cache_population(plan),
            "tensor_conversion_count": 0,
            "error": message,
            "detail": detail,
            "ttnn_version": ttnn_version,
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
            "reference": {
                "kind": "prefill_smoke",
                "status": "not_run",
                "passed": False,
                "checks": [],
            },
        }
    )
    return report


def _prefill_reference(
    *,
    plan: dict[str, Any],
    layer_count: int,
    output_shapes: dict[str, Any],
    output: dict[str, Any],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    expected_outputs = plan["expected_output_shapes"]
    checks: list[dict[str, Any]] = [
        _value_check("layer_count", layer_count, plan["layers"]),
        _shape_check(
            "output.token",
            output_shapes.get("token"),
            accepted=[
                expected_outputs["token"],
                [expected_outputs["token"][0], 1],
                [expected_outputs["token"][0]],
            ],
        ),
        _dtype_check("output.token", output.get("dtype")),
    ]
    for layer in output_shapes.get("kv_cache_layers", []):
        layer_id = int(layer["layer_id"])
        checks.extend(
            [
                _shape_check(
                    f"kv_cache_layers.{layer_id}.key_cache",
                    layer.get("key_cache"),
                    expected=expected_outputs["key_cache"],
                ),
                _shape_check(
                    f"kv_cache_layers.{layer_id}.value_cache",
                    layer.get("value_cache"),
                    expected=expected_outputs["value_cache"],
                ),
            ]
        )
    checks.append(
        _op_sequence_coverage_check(
            planned_ops=plan["op_sequence"],
            observed_ops=observed_ops,
        )
    )
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "prefill_structural_shape_dtype_op_sequence",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "planned_ops": plan["op_sequence"],
        "observed_ops": observed_ops,
        "checks": checks,
    }


def _planned_cache_population(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "layer_id": layer_id,
            "status": "planned",
            "key_cache_shape": plan["expected_output_shapes"]["key_cache"],
            "value_cache_shape": plan["expected_output_shapes"]["value_cache"],
            "write_policy": "fill_cache_per_user",
            "update_shape_layout": "batch_heads_seq_head_dim",
            "planned_user_count": plan["batch_size"],
        }
        for layer_id in range(int(plan["layers"]))
    ]


def _observed_cache_population(
    *,
    plan: dict[str, Any],
    cache_reports: list[dict[str, Any]],
    output_shapes: dict[str, Any],
) -> list[dict[str, Any]]:
    by_layer = {
        int(report.get("layer_id", index)): dict(report)
        for index, report in enumerate(cache_reports)
        if isinstance(report, dict)
    }
    population = []
    for layer in output_shapes.get("kv_cache_layers", []):
        layer_id = int(layer["layer_id"])
        generated_report = by_layer.get(layer_id, {})
        population.append(
            {
                "layer_id": layer_id,
                "status": "filled",
                "key_cache_shape": layer.get("key_cache"),
                "value_cache_shape": layer.get("value_cache"),
                "expected_key_cache_shape": plan["expected_output_shapes"][
                    "key_cache"
                ],
                "expected_value_cache_shape": plan["expected_output_shapes"][
                    "value_cache"
                ],
                "write_policy": generated_report.get(
                    "write_policy",
                    "fill_cache_per_user",
                ),
                "update_shape_layout": generated_report.get(
                    "update_shape_layout",
                    "batch_heads_seq_head_dim",
                ),
                "planned_user_count": plan["batch_size"],
                "filled_user_count": generated_report.get(
                    "filled_user_count"
                ),
                "users": generated_report.get("users", []),
                "generated_report": generated_report,
            }
        )
    return population
