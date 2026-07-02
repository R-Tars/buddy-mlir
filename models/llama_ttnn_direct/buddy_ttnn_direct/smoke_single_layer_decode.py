from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
)
from .codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
    to_ttnn_parameters,
)
from .smoke_attention_primitive import (
    _maybe_managed_device,
    _randn,
    _ttnn_dtype,
    _zeros,
)
from .smoke_decode_shell import (
    NUMERIC_REFERENCE_NOT_RUN_REASON,
    _dry_run_reference,
    _dtype,
    _dtype_check,
    _load_generated_model,
    _observed_op_sequence,
    _shape,
    _shape_check,
    _to_namespace,
    _value_check,
)
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .templates.ttnn_ops import UnsupportedTTNNOp


DECODE_LAYER_OPS = [
    "rms_norm.attn",
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache.k",
    "paged_update_cache.v",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
    "residual_add.attn",
    "rms_norm.mlp",
    "mlp_gate",
    "mlp_up",
    "mul_silu",
    "mlp_down",
    "residual_add.mlp",
]

DECODE_FINAL_OPS = [
    "rms_norm.final",
    "split_lm_head",
    "argmax_or_sampling",
]
DECODE_PARAMETER_ROLES = ["embedding", "norm", "attention", "mlp", "lm_head"]


def _decode_op_sequence(layers: int) -> list[str]:
    ops = ["embedding"]
    for _ in range(layers):
        ops.extend(DECODE_LAYER_OPS)
    ops.extend(DECODE_FINAL_OPS)
    return ops


SINGLE_LAYER_DECODE_OPS = _decode_op_sequence(1)


def run_smoke_single_layer_decode(**kwargs: Any) -> dict[str, Any]:
    return run_smoke_decode_step(layers=1, **kwargs)


def run_smoke_decode_step(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int = 1,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    model_path: str | Path | None = None,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    parameters: Any | None = None,
    token_ids: Any | None = None,
    page_table: Any | None = None,
    cache_position: Any | None = None,
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
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    plan = _decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
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
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "output_shapes": None,
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "error": None,
                "ttnn_version": None,
                "trace": _trace_report(
                    requested=trace,
                    status="dry_run" if trace else "disabled",
                    iterations=trace_iterations if trace else 0,
                ),
                "reference": _dry_run_reference("generated_decode_step"),
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
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
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
        if parameters is None or any(
            item is None for item in (token_ids, page_table, cache_position, kv_cache)
        ):
            report = _failed_report(
                program_dir=program_root,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                plan=plan,
                status="missing_torch",
                message=(
                    "torch is required to synthesize generated decode "
                    "parameters and inputs."
                ),
                detail=str(err),
                ttnn_version=getattr(ttnn, "__version__", None),
            )
            _write_report(out, report)
            return report
        torch = None

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            if parameters is None:
                assert torch is not None
                if model_path is None:
                    state = _build_synthetic_decode_state(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        plan=plan,
                    )
                else:
                    state = _build_model_decode_state(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        plan=plan,
                        program_dir=program_root,
                        model_path=Path(model_path),
                    )
                parameters = state.parameters
                token_ids = state.token_ids
                page_table = state.page_table
                cache_position = state.cache_position
                kv_cache = state.kv_cache
                tensor_conversion_count = state.tensor_conversion_count
                parameter_source = state.parameter_source
                parameter_setup = getattr(state, "parameter_setup", None)
            else:
                tensor_conversion_count = 0
                parameter_source = "injected"
                parameter_setup = None

            if any(
                item is None for item in (token_ids, page_table, cache_position, kv_cache)
            ):
                raise ValueError(
                    "token_ids, page_table, cache_position, and kv_cache are "
                    "required when parameters are injected"
                )

            report = _run_generated_decode_step(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=parameters,
                config=config,
                layer_count=layer_count,
                device=ttnn_device,
                token_ids=token_ids,
                page_table=page_table,
                cache_position=cache_position,
                kv_cache=kv_cache,
                tensor_conversion_count=tensor_conversion_count,
                trace=trace,
                trace_iterations=trace_iterations,
                plan=plan,
            )
            report["parameter_source"] = parameter_source
            if parameter_setup is not None:
                report["parameter_setup"] = parameter_setup
            report["input_source"] = (
                "injected"
                if any(
                    item is not None
                    for item in (token_ids, page_table, cache_position, kv_cache)
                )
                and parameter_source == "injected"
                else "synthetic"
            )
            report.update(
                {
                    **_base_report(
                        program_dir=program_root,
                        layers=layer_count,
                        device=device,
                        device_id=device_id,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        dtype_seed=dtype_seed,
                        trace=trace,
                        trace_iterations=trace_iterations,
                        dry_run=False,
                        plan=plan,
                    ),
                    **report,
                }
            )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            detail=str(err),
        )
    except (ParameterMaterializationError, TTNNTensorizationError) as err:
        report = _failed_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="parameter_setup_error",
            message=str(err),
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except UnsupportedTTNNOp as err:
        report = _failed_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except Exception as err:
        report = _failed_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )

    _write_report(out, report)
    return report


def profile_decode_step(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int = 1,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    model_path: str | Path | None = None,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    parameters: Any | None = None,
    token_ids: Any | None = None,
    page_table: Any | None = None,
    cache_position: Any | None = None,
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
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    plan = _decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
    )

    if dry_run:
        report = _base_profile_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "section_latency_ms": _empty_section_latency(),
                "layer_profiles": [
                    _planned_layer_profile(layer_id)
                    for layer_id in range(layer_count)
                ],
                "bottleneck_summary": _bottleneck_summary(
                    _empty_section_latency(),
                    [],
                    tensor_conversion_ms=0.0,
                    trace_execute_ms=0.0,
                ),
                "output_shapes": None,
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "tensor_conversion_ms": 0.0,
                "host_copy_ms": 0.0,
                "trace": _trace_report(
                    requested=trace,
                    status="dry_run" if trace else "disabled",
                    iterations=trace_iterations if trace else 0,
                ),
                "reference": _dry_run_reference("generated_decode_step_profile"),
                "error": None,
                "ttnn_version": None,
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
        report = _profile_unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
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
        if parameters is None or any(
            item is None for item in (token_ids, page_table, cache_position, kv_cache)
        ):
            report = _profile_unavailable_report(
                program_dir=program_root,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                plan=plan,
                status="missing_torch",
                message=(
                    "torch is required to synthesize generated decode "
                    "parameters and inputs."
                ),
                detail=str(err),
                ttnn_version=getattr(ttnn, "__version__", None),
            )
            _write_report(out, report)
            return report
        torch = None

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            tensor_conversion_ms = 0.0
            if parameters is None:
                assert torch is not None
                conversion_start = time.perf_counter()
                if model_path is None:
                    synthetic = _build_synthetic_decode_state(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        plan=plan,
                    )
                else:
                    synthetic = _build_model_decode_state(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        plan=plan,
                        program_dir=program_root,
                        model_path=Path(model_path),
                    )
                tensor_conversion_ms = (time.perf_counter() - conversion_start) * 1000.0
                parameters = synthetic.parameters
                token_ids = synthetic.token_ids
                page_table = synthetic.page_table
                cache_position = synthetic.cache_position
                kv_cache = synthetic.kv_cache
                tensor_conversion_count = synthetic.tensor_conversion_count
                parameter_source = synthetic.parameter_source
                parameter_setup = getattr(synthetic, "parameter_setup", None)
            else:
                tensor_conversion_count = 0
                parameter_source = "injected"
                parameter_setup = None

            if any(
                item is None for item in (token_ids, page_table, cache_position, kv_cache)
            ):
                raise ValueError(
                    "token_ids, page_table, cache_position, and kv_cache are "
                    "required when parameters are injected"
                )

            profile = _run_generated_decode_profile(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=parameters,
                config=config,
                layer_count=layer_count,
                device=ttnn_device,
                token_ids=token_ids,
                page_table=page_table,
                cache_position=cache_position,
                kv_cache=kv_cache,
                trace=trace,
                trace_iterations=trace_iterations,
                plan=plan,
            )
            report = _base_profile_report(
                program_dir=program_root,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                trace=trace,
                trace_iterations=trace_iterations,
                dry_run=False,
                plan=plan,
            )
            profile["tensor_conversion_count"] = tensor_conversion_count
            profile["tensor_conversion_ms"] = tensor_conversion_ms
            profile["parameter_source"] = parameter_source
            if parameter_setup is not None:
                profile["parameter_setup"] = parameter_setup
            profile["input_source"] = (
                "injected"
                if any(
                    item is not None
                    for item in (token_ids, page_table, cache_position, kv_cache)
                )
                and parameter_source == "injected"
                else "synthetic"
            )
            profile["bottleneck_summary"] = _bottleneck_summary(
                profile["section_latency_ms"],
                profile["layer_profiles"],
                tensor_conversion_ms=tensor_conversion_ms,
                trace_execute_ms=float(
                    profile["trace"].get("execute_latency_ms") or 0.0
                ),
            )
            report.update(profile)
    except NoTTNNDeviceError as err:
        report = _profile_unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="no_device",
            message=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except (ParameterMaterializationError, TTNNTensorizationError) as err:
        report = _profile_unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="parameter_setup_error",
            message=str(err),
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except UnsupportedTTNNOp as err:
        report = _profile_unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
        )
    except Exception as err:
        report = _profile_unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            plan=plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
        )

    _write_report(out, report)
    return report


def _run_generated_decode_step(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    layer_count: int,
    device: Any,
    token_ids: Any,
    page_table: Any,
    cache_position: Any,
    kv_cache: Any,
    tensor_conversion_count: int,
    trace: bool,
    trace_iterations: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    decode_config = dict(config)
    decode_config["num_layers"] = layer_count
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(decode_config),
    )

    trace_report = _trace_report(requested=trace, status="disabled")
    if trace:
        if _trace_apis_available(ttnn):
            try:
                token, kv_cache, latency_ms, trace_report = (
                    _run_decode_step_with_trace(
                        ttnn=ttnn,
                        model=model,
                        device=device,
                        token_ids=token_ids,
                        page_table=page_table,
                        cache_position=cache_position,
                        kv_cache=kv_cache,
                        iterations=trace_iterations,
                    )
                )
            except Exception as err:
                trace_report = _trace_report(
                    requested=True,
                    status="trace_failed_fell_back_to_eager",
                    iterations=trace_iterations,
                    error=f"{type(err).__name__}: {err}",
                )
                token, kv_cache, latency_ms = _time_decode_step(
                    ttnn=ttnn,
                    model=model,
                    device=device,
                    token_ids=token_ids,
                    page_table=page_table,
                    cache_position=cache_position,
                    kv_cache=kv_cache,
                )
        else:
            trace_report = _trace_report(
                requested=True,
                status="trace_api_unavailable_fell_back_to_eager",
                iterations=trace_iterations,
            )
            token, kv_cache, latency_ms = _time_decode_step(
                ttnn=ttnn,
                model=model,
                device=device,
                token_ids=token_ids,
                page_table=page_table,
                cache_position=cache_position,
                kv_cache=kv_cache,
            )
    else:
        token, kv_cache, latency_ms = _time_decode_step(
            ttnn=ttnn,
            model=model,
            device=device,
            token_ids=token_ids,
            page_table=page_table,
            cache_position=cache_position,
            kv_cache=kv_cache,
        )

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
    output = {
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = _decode_step_reference(
        plan=plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output=output,
        observed_ops=_observed_op_sequence(ttnn),
    )
    passed = bool(reference["passed"])

    return {
        "passed": passed,
        "status": "passed" if passed else "reference_mismatch",
        "latency_ms": latency_ms,
        "output_shapes": output_shapes,
        "output": output,
        "tensor_conversion_count": tensor_conversion_count,
        "error": None if passed else "decode-step structural reference mismatch",
        "ttnn_version": getattr(ttnn, "__version__", None),
        "trace": trace_report,
        "reference": reference,
    }


def _decode_step_reference(
    *,
    plan: dict[str, Any],
    layer_count: int,
    output_shapes: dict[str, Any],
    output: dict[str, Any],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    expected_outputs = plan["expected_output_shapes"]
    expected_token = expected_outputs["token"]
    accepted_token_shapes = [
        expected_token,
        [expected_token[0]],
    ]
    checks: list[dict[str, Any]] = [
        _value_check("layer_count", layer_count, plan["layers"]),
        _shape_check(
            "output.token",
            output_shapes.get("token"),
            accepted=accepted_token_shapes,
        ),
        _dtype_check("output.token", output.get("dtype")),
        _shape_check(
            "output.key_cache",
            output_shapes.get("key_cache"),
            expected=expected_outputs["key_cache"],
        ),
        _shape_check(
            "output.value_cache",
            output_shapes.get("value_cache"),
            expected=expected_outputs["value_cache"],
        ),
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

    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape_dtype",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "planned_ops": list(plan["op_sequence"]),
        "observed_ops": observed_ops,
        "checks": checks,
    }


def _run_generated_decode_profile(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    layer_count: int,
    device: Any,
    token_ids: Any,
    page_table: Any,
    cache_position: Any,
    kv_cache: Any,
    trace: bool,
    trace_iterations: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    decode_config = dict(config)
    decode_config["num_layers"] = layer_count
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(decode_config),
    )

    section_latency = _empty_section_latency()
    layer_profiles: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    hidden, section_latency["embedding_ms"] = _time_section(
        lambda: model.embed(token_ids),
        ttnn=ttnn,
        device=device,
    )
    for layer_id in range(layer_count):
        layer_start = time.perf_counter()
        residual = hidden
        hidden, attn_norm_ms = _time_section(
            lambda layer_id=layer_id, hidden=hidden: model.rmsnorm(
                hidden,
                layer_id,
                kind="attn",
            ),
            ttnn=ttnn,
            device=device,
        )
        hidden, attention_ms = _time_section(
            lambda layer_id=layer_id, hidden=hidden: model.attention_decode(
                layer_id,
                hidden,
                page_table,
                cache_position,
                kv_cache,
            ),
            ttnn=ttnn,
            device=device,
        )
        hidden, attn_residual_ms = _time_section(
            lambda residual=residual, hidden=hidden: model.ops.add(
                residual,
                hidden,
            ),
            ttnn=ttnn,
            device=device,
        )

        residual = hidden
        hidden, mlp_norm_ms = _time_section(
            lambda layer_id=layer_id, hidden=hidden: model.rmsnorm(
                hidden,
                layer_id,
                kind="mlp",
            ),
            ttnn=ttnn,
            device=device,
        )
        hidden, mlp_ms = _time_section(
            lambda layer_id=layer_id, hidden=hidden: model.mlp_decode(
                layer_id,
                hidden,
            ),
            ttnn=ttnn,
            device=device,
        )
        hidden, mlp_residual_ms = _time_section(
            lambda residual=residual, hidden=hidden: model.ops.add(
                residual,
                hidden,
            ),
            ttnn=ttnn,
            device=device,
        )
        layer_total_ms = (time.perf_counter() - layer_start) * 1000.0
        layer_profiles.append(
            {
                "layer_id": layer_id,
                "rms_norm_attn_ms": attn_norm_ms,
                "attention_ms": attention_ms,
                "residual_add_attn_ms": attn_residual_ms,
                "rms_norm_mlp_ms": mlp_norm_ms,
                "mlp_ms": mlp_ms,
                "residual_add_mlp_ms": mlp_residual_ms,
                "total_ms": layer_total_ms,
            }
        )

    hidden, section_latency["final_norm_ms"] = _time_section(
        lambda: model.final_norm(hidden),
        ttnn=ttnn,
        device=device,
    )
    token, lm_head_profile = _profile_lm_head_and_argmax(
        model=model,
        hidden=hidden,
        ttnn=ttnn,
        device=device,
    )
    section_latency["lm_head_ms"] = lm_head_profile["lm_head_ms"]
    section_latency["argmax_ms"] = lm_head_profile["argmax_ms"]
    section_latency["host_copy_ms"] = 0.0

    trace_report = _trace_report(requested=trace, status="disabled")
    if trace:
        if _trace_apis_available(ttnn):
            try:
                _, _, _, trace_report = _run_decode_step_with_trace(
                    ttnn=ttnn,
                    model=model,
                    device=device,
                    token_ids=token_ids,
                    page_table=page_table,
                    cache_position=cache_position,
                    kv_cache=kv_cache,
                    iterations=trace_iterations,
                )
            except Exception as err:
                trace_report = _trace_report(
                    requested=True,
                    status="trace_failed_after_profile",
                    iterations=trace_iterations,
                    error=f"{type(err).__name__}: {err}",
                )
        else:
            trace_report = _trace_report(
                requested=True,
                status="trace_api_unavailable",
                iterations=trace_iterations,
            )

    latency_ms = (time.perf_counter() - total_start) * 1000.0
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
    output = {
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = _decode_step_reference(
        plan=plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output=output,
        observed_ops=_observed_op_sequence(ttnn),
    )
    passed = bool(reference["passed"])

    return {
        "passed": passed,
        "status": "profiled" if passed else "reference_mismatch",
        "latency_ms": latency_ms,
        "section_latency_ms": section_latency,
        "layer_profiles": layer_profiles,
        "lm_head_profile": lm_head_profile,
        "output_shapes": output_shapes,
        "output": output,
        "host_copy_ms": 0.0,
        "host_copy_status": "not_performed",
        "trace": trace_report,
        "error": None if passed else "decode-step profile structural mismatch",
        "ttnn_version": getattr(ttnn, "__version__", None),
        "reference": reference,
    }


def _profile_lm_head_and_argmax(
    *,
    model: Any,
    hidden: Any,
    ttnn: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    lm_head_config = model.config.lm_head
    split_count = int(
        _optional_attr(
            lm_head_config,
            "split_count",
            len(model.parameters.lm_head.splits),
        )
    )
    program_configs = _optional_attr(lm_head_config, "program_configs", None)
    split_configs = _optional_attr(lm_head_config, "splits", None)
    shard_logits = []

    split_start = time.perf_counter()
    for shard_id in range(split_count):
        split_params = model.parameters.lm_head.splits[shard_id]
        split_config = None
        if split_configs is not None and shard_id < len(split_configs):
            split_config = split_configs[shard_id]
        program_config = _optional_attr(split_config, "program_config", None)
        if (
            program_config is None
            and program_configs is not None
            and shard_id < len(program_configs)
        ):
            program_config = program_configs[shard_id]
        logits_i = model.ops.linear(
            hidden,
            split_params.weight,
            memory_config=_optional_attr(
                lm_head_config,
                "output_memory_config",
            ),
            program_config=program_config,
            compute_kernel_config=_optional_attr(
                lm_head_config,
                "compute_kernel_config",
            ),
            dtype=_optional_attr(lm_head_config, "output_dtype"),
        )
        shard_logits.append(logits_i)

    logits = model.ops.concat(
        shard_logits,
        dim=-1,
        memory_config=_optional_attr(
            lm_head_config,
            "concat_memory_config",
        ),
    )
    _synchronize(ttnn, device)
    lm_head_ms = (time.perf_counter() - split_start) * 1000.0

    generation_config = _optional_attr(model.config, "generation", None)
    generation_mode = _optional_attr(generation_config, "mode", "greedy")
    retain_logits = bool(_optional_attr(lm_head_config, "retain_logits", False))
    if generation_mode == "greedy" and not retain_logits:
        token, argmax_ms = _time_section(
            lambda: model.ops.argmax(logits, dim=-1),
            ttnn=ttnn,
            device=device,
        )
        argmax_status = "profiled"
    else:
        token = logits
        argmax_ms = 0.0
        argmax_status = "skipped"

    return token, {
        "split_count": split_count,
        "lm_head_ms": lm_head_ms,
        "argmax_ms": argmax_ms,
        "argmax_status": argmax_status,
    }


def _time_decode_step(
    *,
    ttnn: Any,
    model: Any,
    device: Any,
    token_ids: Any,
    page_table: Any,
    cache_position: Any,
    kv_cache: Any,
) -> tuple[Any, Any, float]:
    start = time.perf_counter()
    token, kv_cache = model.decode_step(
        token_ids,
        page_table,
        cache_position,
        kv_cache,
    )
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return token, kv_cache, latency_ms


def _run_decode_step_with_trace(
    *,
    ttnn: Any,
    model: Any,
    device: Any,
    token_ids: Any,
    page_table: Any,
    cache_position: Any,
    kv_cache: Any,
    iterations: int,
) -> tuple[Any, Any, float, dict[str, Any]]:
    trace_id = None
    release_trace = getattr(ttnn, "release_trace", None)
    try:
        capture_start = time.perf_counter()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        token, kv_cache = model.decode_step(
            token_ids,
            page_table,
            cache_position,
            kv_cache,
        )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0

        execute_samples = []
        for _ in range(iterations):
            execute_start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            synchronize = getattr(ttnn, "synchronize_device", None)
            if callable(synchronize):
                synchronize(device)
            execute_samples.append((time.perf_counter() - execute_start) * 1000.0)
        execute_latency_ms = sum(execute_samples)
        trace_report = _trace_report(
            requested=True,
            status="captured_and_executed",
            iterations=iterations,
            trace_id=trace_id,
            capture_latency_ms=capture_latency_ms,
            execute_latency_ms=execute_latency_ms,
            execute_samples_ms=execute_samples,
        )
        return token, kv_cache, capture_latency_ms + execute_latency_ms, trace_report
    finally:
        if trace_id is not None and callable(release_trace):
            release_trace(device, trace_id)


def _time_section(fn: Any, *, ttnn: Any, device: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    output = fn()
    _synchronize(ttnn, device)
    return output, (time.perf_counter() - start) * 1000.0


def _synchronize(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)


def _optional_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _build_synthetic_decode_state(
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
    token_ids = tensor(inputs["token_ids"], name="token_ids", zeros=True)
    page_table = tensor(inputs["page_table"], name="page_table", zeros=True)
    cache_position = tensor(
        inputs["cache_position"],
        name="cache_position",
        zeros=True,
    )
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
        page_table=page_table,
        cache_position=cache_position,
        kv_cache=kv_cache,
        tensor_conversion_count=tensor_count(),
        parameter_source="synthetic",
        input_source="synthetic",
    )


def _build_model_decode_state(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    program_dir: Path,
    model_path: Path,
) -> SimpleNamespace:
    host_params = load_llama_parameters_from_manifests(
        model_path=model_path,
        weights_manifest=program_dir / "weights_manifest.json",
        config=program_dir / "config.json",
        tensor_backend="torch",
        layers=range(int(plan["layers"])),
    )
    materialization_summary = _materialization_summary(host_params)
    result = to_ttnn_parameters(
        host_params,
        device,
        load_parameter_config_from_program(program_dir),
        roles=DECODE_PARAMETER_ROLES,
        layers=range(int(plan["layers"])),
        ttnn_module=ttnn,
    )
    assert result.parameters is not None
    tensor_conversion_count = int(result.report["tensor_count"])
    synthetic_rotary_count = _attach_synthetic_rotary_parameters(
        parameters=result.parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=plan,
    )
    tensor_conversion_count += synthetic_rotary_count
    synthetic_inputs = _build_synthetic_decode_inputs(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=plan,
    )
    return SimpleNamespace(
        parameters=result.parameters,
        token_ids=synthetic_inputs.token_ids,
        page_table=synthetic_inputs.page_table,
        cache_position=synthetic_inputs.cache_position,
        kv_cache=synthetic_inputs.kv_cache,
        tensor_conversion_count=(
            tensor_conversion_count + synthetic_inputs.tensor_conversion_count
        ),
        parameter_source="hf_model",
        input_source="synthetic",
        parameter_setup={
            "materialization": materialization_summary,
            "tensorization": _tensorization_summary(result.report),
            "synthetic_rotary_tensor_count": synthetic_rotary_count,
            "synthetic_runtime_input_tensor_count": (
                synthetic_inputs.tensor_conversion_count
            ),
        },
    )


def _materialization_summary(params: Any) -> dict[str, Any]:
    metadata = dict(getattr(params, "metadata", {}))
    tensors = metadata.get("tensors", {})
    key_paths = [
        path
        for path in (
            "embedding.weight",
            "layers.0.attention.wqkv_packed.weight",
            "layers.0.attention.o_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "final_norm.weight",
            "lm_head.splits.0.weight",
        )
        if path in tensors
    ]
    return {
        "backend": metadata.get("backend"),
        "model_name": metadata.get("model_name"),
        "num_layers": metadata.get("num_layers"),
        "materialized_layer_ids": list(
            metadata.get("materialized_layer_ids", [])
        ),
        "tensor_count": metadata.get("tensor_count"),
        "key_paths": key_paths,
    }


def _tensorization_summary(report: dict[str, Any]) -> dict[str, Any]:
    tensors = report.get("tensors", [])
    key_paths = [
        record["path"]
        for record in tensors
        if record.get("path")
        in {
            "embedding.weight",
            "layers.0.attention.wqkv_packed.weight",
            "layers.0.attention.o_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "final_norm.weight",
            "lm_head.splits.0.weight",
        }
    ]
    return {
        "status": report.get("status"),
        "backend": "ttnn",
        "roles": list(report.get("roles", [])),
        "tensor_count": report.get("tensor_count"),
        "key_paths": key_paths,
    }


def _build_synthetic_decode_inputs(
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
    token_ids = tensor(inputs["token_ids"], name="token_ids", zeros=True)
    page_table = tensor(inputs["page_table"], name="page_table", zeros=True)
    cache_position = tensor(
        inputs["cache_position"],
        name="cache_position",
        zeros=True,
    )
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
        token_ids=token_ids,
        page_table=page_table,
        cache_position=cache_position,
        kv_cache=kv_cache,
        tensor_conversion_count=tensor_count(),
    )


def _attach_synthetic_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
) -> int:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    layer_params = plan["layer_parameter_shapes"]
    for layer_id in range(int(plan["layers"])):
        layer = parameters.layers[layer_id]
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = SimpleNamespace()
            layer.attention = attention
        attention.rotary = SimpleNamespace(
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
        )
    return tensor_count()


def _synthetic_tensor_factory(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
) -> tuple[Any, Any]:
    dtype = _ttnn_dtype(ttnn, dtype_seed)
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    tensor_conversion_count = 0

    def tensor(shape: list[int], *, name: str, zeros: bool = False) -> Any:
        nonlocal tensor_conversion_count
        host_tensor = _zeros(torch, shape) if zeros else _randn(torch, shape, dtype_seed)
        try:
            host_tensor.name = name
        except AttributeError:
            pass
        tensor_conversion_count += 1
        return ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
        )

    return tensor, lambda: tensor_conversion_count


def _decode_step_plan(
    *,
    layers: int,
    batch_size: int,
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
    page_count = max(1, (cache_len + 31) // 32)
    lm_head_splits = _lm_head_split_shapes(config, hidden_size, vocab_size)
    input_shapes = {
        "token_ids": [batch_size, 1],
        "page_table": [batch_size, page_count],
        "cache_position": [batch_size],
        "key_cache": [batch_size, cache_len, num_kv_heads, head_dim],
        "value_cache": [batch_size, cache_len, num_kv_heads, head_dim],
    }
    layer_parameter_shapes = {
        "input_norm": [hidden_size],
        "post_attention_norm": [hidden_size],
        "attention_wqkv": [hidden_size, qkv_size],
        "attention_o_proj": [num_heads * head_dim, hidden_size],
        "rotary_cos_matrix": [1, 1, head_dim, head_dim],
        "rotary_sin_matrix": [1, 1, head_dim, head_dim],
        "rotary_transformation_matrix": [1, 1, head_dim, head_dim],
        "mlp_gate": [hidden_size, intermediate_size],
        "mlp_up": [hidden_size, intermediate_size],
        "mlp_down": [intermediate_size, hidden_size],
    }
    parameter_shapes = {
        "embedding": [vocab_size, hidden_size],
        **layer_parameter_shapes,
        "final_norm": [hidden_size],
        "lm_head_splits": lm_head_splits,
    }
    return {
        "layers": layers,
        "input_shapes": input_shapes,
        "parameter_shapes": parameter_shapes,
        "layer_parameter_shapes": layer_parameter_shapes,
        "expected_intermediate_shapes": {
            "embedding": [batch_size, 1, hidden_size],
            "qkv": [batch_size, 1, qkv_size],
            "query": [batch_size, num_heads, 1, head_dim],
            "key": [batch_size, num_kv_heads, 1, head_dim],
            "value": [batch_size, num_kv_heads, 1, head_dim],
            "attention": [batch_size, num_heads, 1, head_dim],
            "concat_heads": [batch_size, 1, num_heads * head_dim],
            "attention_output": [batch_size, 1, hidden_size],
            "mlp_intermediate": [batch_size, 1, intermediate_size],
        },
        "expected_output_shapes": {
            "token": [batch_size, 1],
            "key_cache": input_shapes["key_cache"],
            "value_cache": input_shapes["value_cache"],
        },
        "tensor_conversion_count": 5 + len(lm_head_splits) + 12 * layers,
        "op_sequence": _decode_op_sequence(layers),
    }


def _lm_head_split_shapes(
    config: dict[str, Any],
    hidden_size: int,
    vocab_size: int,
) -> list[list[int]]:
    lm_head_config = config.get("lm_head", {})
    split_count = int(lm_head_config.get("split_count", 1))
    split_configs = lm_head_config.get("splits")
    if split_configs:
        shapes = []
        for split in split_configs:
            vocab_start = int(split["vocab_start"])
            vocab_end = int(split["vocab_end"])
            shapes.append([hidden_size, vocab_end - vocab_start])
        return shapes

    base = vocab_size // split_count
    remainder = vocab_size % split_count
    shapes = []
    for shard_id in range(split_count):
        width = base + (1 if shard_id < remainder else 0)
        shapes.append([hidden_size, width])
    return shapes


def _base_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    dry_run: bool,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "generated_decode_step",
        "program_dir": str(program_dir),
        "layers": layers,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": dry_run,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations if trace else 0,
        "op_sequence": plan["op_sequence"],
        "input_shapes": plan["input_shapes"],
        "parameter_shapes": plan["parameter_shapes"],
        "layer_parameter_shapes": plan["layer_parameter_shapes"],
        "expected_intermediate_shapes": plan["expected_intermediate_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
    }


def _no_device_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    plan: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layers=layers,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": "no_device",
            "latency_ms": None,
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "error": NO_TTNN_DEVICE_MESSAGE,
            "detail": detail,
            "ttnn_version": None,
            "trace": _trace_report(
                requested=trace,
                status="unavailable" if trace else "disabled",
                iterations=trace_iterations if trace else 0,
            ),
        }
    )
    return report


def _failed_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layers=layers,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "error": message,
            "detail": detail,
            "ttnn_version": ttnn_version,
            "trace": _trace_report(
                requested=trace,
                status="unavailable" if trace else "disabled",
                iterations=trace_iterations if trace else 0,
            ),
        }
    )
    return report


def _base_profile_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    dry_run: bool,
    plan: dict[str, Any],
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        layers=layers,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        dry_run=dry_run,
        plan=plan,
    )
    report.update(
        {
            "template": "generated_decode_step_profile",
            "profile_sections": [
                "tensor_conversion_ms",
                "embedding_ms",
                "per_layer_attention_ms",
                "per_layer_mlp_ms",
                "final_norm_ms",
                "lm_head_ms",
                "argmax_ms",
                "host_copy_ms",
                "trace_execute_ms",
            ],
        }
    )
    return report


def _profile_unavailable_report(
    *,
    program_dir: Path,
    layers: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_version: str | None = None,
) -> dict[str, Any]:
    report = _base_profile_report(
        program_dir=program_dir,
        layers=layers,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        dry_run=False,
        plan=plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "section_latency_ms": _empty_section_latency(),
            "layer_profiles": [],
            "bottleneck_summary": _bottleneck_summary(
                _empty_section_latency(),
                [],
                tensor_conversion_ms=0.0,
                trace_execute_ms=0.0,
            ),
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "tensor_conversion_ms": 0.0,
            "host_copy_ms": 0.0,
            "trace": _trace_report(
                requested=trace,
                status="unavailable" if trace else "disabled",
                iterations=trace_iterations if trace else 0,
            ),
            "error": message,
            "detail": detail,
            "ttnn_version": ttnn_version,
        }
    )
    return report


def _empty_section_latency() -> dict[str, float]:
    return {
        "embedding_ms": 0.0,
        "final_norm_ms": 0.0,
        "lm_head_ms": 0.0,
        "argmax_ms": 0.0,
        "host_copy_ms": 0.0,
    }


def _planned_layer_profile(layer_id: int) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        "rms_norm_attn_ms": 0.0,
        "attention_ms": 0.0,
        "residual_add_attn_ms": 0.0,
        "rms_norm_mlp_ms": 0.0,
        "mlp_ms": 0.0,
        "residual_add_mlp_ms": 0.0,
        "total_ms": 0.0,
    }


def _bottleneck_summary(
    section_latency: dict[str, float],
    layer_profiles: list[dict[str, Any]],
    *,
    tensor_conversion_ms: float,
    trace_execute_ms: float,
) -> dict[str, Any]:
    attention_ms = sum(float(layer["attention_ms"]) for layer in layer_profiles)
    mlp_ms = sum(float(layer["mlp_ms"]) for layer in layer_profiles)
    layer_total_ms = sum(float(layer["total_ms"]) for layer in layer_profiles)
    sections = {
        "tensor_conversion_ms": tensor_conversion_ms,
        "embedding_ms": float(section_latency["embedding_ms"]),
        "per_layer_attention_ms": attention_ms,
        "per_layer_mlp_ms": mlp_ms,
        "layer_stack_ms": layer_total_ms,
        "final_norm_ms": float(section_latency["final_norm_ms"]),
        "lm_head_ms": float(section_latency["lm_head_ms"]),
        "argmax_ms": float(section_latency["argmax_ms"]),
        "host_copy_ms": float(section_latency["host_copy_ms"]),
        "trace_execute_ms": trace_execute_ms,
    }
    bottleneck = max(sections.items(), key=lambda item: item[1])
    return {
        "sections_ms": sections,
        "max_section": bottleneck[0],
        "max_section_ms": bottleneck[1],
    }


def _trace_apis_available(ttnn: Any) -> bool:
    return all(
        callable(getattr(ttnn, name, None))
        for name in (
            "begin_trace_capture",
            "end_trace_capture",
            "execute_trace",
        )
    )


def _trace_report(
    *,
    requested: bool,
    status: str,
    iterations: int = 0,
    trace_id: Any | None = None,
    capture_latency_ms: float | None = None,
    execute_latency_ms: float | None = None,
    execute_samples_ms: list[float] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": requested,
        "status": status,
        "iterations": iterations,
    }
    if trace_id is not None:
        report["trace_id"] = trace_id
    if capture_latency_ms is not None:
        report["capture_latency_ms"] = capture_latency_ms
    if execute_latency_ms is not None:
        report["execute_latency_ms"] = execute_latency_ms
    if execute_samples_ms is not None:
        report["execute_samples_ms"] = execute_samples_ms
    if error is not None:
        report["error"] = error
    return report


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
