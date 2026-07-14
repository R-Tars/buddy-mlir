from __future__ import annotations

import gc
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
from .runtime_environment import collect_ttnn_environment
from .runtime_inputs import (
    PromptTokenizationError,
    build_decode_rotary_runtime_state,
    build_decode_runtime_state,
    tokenize_prompt_for_decode,
)
from .runtime.kv_cache import (
    build_prompt_decode_kv_cache_tensors as _build_prompt_decode_kv_cache_tensors,
)
from .runtime.inputs import (
    build_prompt_decode_runtime_state_tensors as _runtime_decode_state_tensors,
)
from .runtime.plans import (
    decode_op_sequence as _runtime_decode_op_sequence,
    decode_output_kind as _runtime_decode_output_kind,
    decode_step_plan as _runtime_decode_step_plan,
    embedding_weight_shape as _runtime_embedding_weight_shape,
    lm_head_split_shapes as _runtime_lm_head_split_shapes,
    norm_weight_shape as _runtime_norm_weight_shape,
)
from .smoke_attention_primitive import (
    _decode_head_shape,
    _decode_hidden_shape,
    _decode_rotary_cos_sin_shape,
    _decode_rotary_transform_shape,
    _linear_weight_shape,
    _maybe_managed_device,
    _randn,
    _rotary_cos_sin_height_sharded_memory_config,
    _rotary_transform_height_sharded_memory_config,
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
    _runtime_int_tensor,
    _shape,
    _shape_check,
    _token_ids_tensor,
    _to_namespace,
    _value_check,
)
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .ttnn_compat import UnsupportedTTNNOp


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
DECODE_FINAL_LOGITS_OPS = [
    "rms_norm.final",
    "split_lm_head",
]
DECODE_PARAMETER_ROLES = ["embedding", "norm", "attention", "mlp", "lm_head"]


def _decode_op_sequence(layers: int, *, output_kind: str = "token") -> list[str]:
    ops = ["embedding"]
    for _ in range(layers):
        ops.extend(DECODE_LAYER_OPS)
    if output_kind == "logits":
        ops.extend(DECODE_FINAL_LOGITS_OPS)
    else:
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
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
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
                ttnn_module=ttnn,
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
                        prompt=prompt,
                        tokenizer_path=tokenizer_path,
                        tokenizer_module=tokenizer_module,
                    )
                parameters = state.parameters
                token_ids = state.token_ids
                page_table = state.page_table
                cache_position = state.cache_position
                kv_cache = state.kv_cache
                tensor_conversion_count = state.tensor_conversion_count
                parameter_source = state.parameter_source
                parameter_setup = getattr(state, "parameter_setup", None)
                input_source = getattr(state, "input_source", "synthetic")
                prompt_tokenization = getattr(
                    state, "prompt_tokenization", None
                )
                decode_runtime_state = getattr(
                    state, "decode_runtime_state", None
                )
                rotary_runtime_state = getattr(
                    state, "rotary_runtime_state", None
                )
                kv_cache_runtime_state = getattr(
                    state, "kv_cache_runtime_state", None
                )
            else:
                tensor_conversion_count = 0
                parameter_source = "injected"
                parameter_setup = None
                input_source = "injected"
                prompt_tokenization = None
                decode_runtime_state = None
                rotary_runtime_state = None
                kv_cache_runtime_state = None

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
                batch_size=batch_size,
                cache_len=cache_len,
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
            report["input_source"] = input_source
            if prompt_tokenization is not None:
                report["prompt_tokenization"] = prompt_tokenization
            if decode_runtime_state is not None:
                report["decode_runtime_state"] = decode_runtime_state
            if rotary_runtime_state is not None:
                report["rotary_runtime_state"] = rotary_runtime_state
            if kv_cache_runtime_state is not None:
                report["kv_cache_runtime_state"] = kv_cache_runtime_state
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
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
    ) as err:
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
            ttnn_module=ttnn,
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
            ttnn_module=ttnn,
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
            ttnn_module=ttnn,
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
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
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
                "lm_head_profile": _planned_lm_head_profile(config),
                "bottleneck_summary": _bottleneck_summary(
                    _empty_section_latency(),
                    [],
                    tensor_conversion_ms=0.0,
                    trace_execute_ms=0.0,
                ),
                "throughput_summary": _throughput_summary(
                    latency_ms=0.0,
                    batch_size=batch_size,
                    trace_report=None,
                    dry_run=True,
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
                ttnn_module=ttnn,
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
                        prompt=prompt,
                        tokenizer_path=tokenizer_path,
                        tokenizer_module=tokenizer_module,
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
                input_source = getattr(synthetic, "input_source", "synthetic")
                prompt_tokenization = getattr(
                    synthetic, "prompt_tokenization", None
                )
                decode_runtime_state = getattr(
                    synthetic, "decode_runtime_state", None
                )
                rotary_runtime_state = getattr(
                    synthetic, "rotary_runtime_state", None
                )
                kv_cache_runtime_state = getattr(
                    synthetic, "kv_cache_runtime_state", None
                )
            else:
                tensor_conversion_count = 0
                parameter_source = "injected"
                parameter_setup = None
                input_source = "injected"
                prompt_tokenization = None
                decode_runtime_state = None
                rotary_runtime_state = None
                kv_cache_runtime_state = None

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
                batch_size=batch_size,
                cache_len=cache_len,
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
            profile["input_source"] = input_source
            if prompt_tokenization is not None:
                profile["prompt_tokenization"] = prompt_tokenization
            if decode_runtime_state is not None:
                profile["decode_runtime_state"] = decode_runtime_state
            if rotary_runtime_state is not None:
                profile["rotary_runtime_state"] = rotary_runtime_state
            if kv_cache_runtime_state is not None:
                profile["kv_cache_runtime_state"] = kv_cache_runtime_state
            profile["bottleneck_summary"] = _bottleneck_summary(
                profile["section_latency_ms"],
                profile["layer_profiles"],
                tensor_conversion_ms=tensor_conversion_ms,
                trace_execute_ms=float(
                    profile["trace"].get("execute_latency_ms") or 0.0
                ),
            )
            profile["throughput_summary"] = _throughput_summary(
                latency_ms=float(profile["latency_ms"]),
                batch_size=batch_size,
                trace_report=profile.get("trace"),
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
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
    ) as err:
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
            ttnn_module=ttnn,
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
            ttnn_module=ttnn,
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
            ttnn_module=ttnn,
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
    batch_size: int,
    cache_len: int,
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
    decode_config["batch_size"] = batch_size
    decode_config["max_cache_len"] = cache_len
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(decode_config),
    )
    model.ops.enable_recording()

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

    output_kind = str(plan.get("output_kind", "token"))
    output_shapes = {
        output_kind: _shape(token),
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
        "kind": output_kind,
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = _decode_step_reference(
        plan=plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output=output,
        observed_ops=_generated_observed_op_sequence(model, ttnn),
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
        "ttnn_environment": collect_ttnn_environment(ttnn),
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
    output_kind = str(plan.get("output_kind", "token"))
    expected_output = expected_outputs[output_kind]
    accepted_output_shapes = [expected_output]
    if output_kind == "token":
        accepted_output_shapes.append([expected_output[0]])
    checks: list[dict[str, Any]] = [
        _value_check("layer_count", layer_count, plan["layers"]),
        _value_check("output_kind", output.get("kind"), output_kind),
        _shape_check(
            f"output.{output_kind}",
            output_shapes.get(output_kind),
            accepted=accepted_output_shapes,
        ),
        _dtype_check(f"output.{output_kind}", output.get("dtype")),
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

    checks.append(
        _op_sequence_coverage_check(
            planned_ops=plan["op_sequence"],
            observed_ops=observed_ops,
        )
    )
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape_dtype_op_sequence",
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


def _generated_observed_op_sequence(model: Any, ttnn: Any) -> list[str] | None:
    ops = getattr(model, "ops", None)
    op_log = getattr(ops, "op_log", None)
    if isinstance(op_log, list):
        return [str(item) for item in op_log]
    return _observed_op_sequence(ttnn)


def _op_sequence_coverage_check(
    *,
    planned_ops: list[str],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    if not isinstance(observed_ops, list):
        return {
            "name": "observed_op_sequence",
            "type": "sequence_coverage",
            "actual": None,
            "expected": planned_ops,
            "passed": False,
            "reason": "generated op instrumentation was not available",
        }

    planned_index = 0
    for observed in observed_ops:
        if (
            planned_index < len(planned_ops)
            and observed == planned_ops[planned_index]
        ):
            planned_index += 1
    missing = planned_ops[planned_index:]
    return {
        "name": "observed_op_sequence",
        "type": "sequence_coverage",
        "actual": observed_ops,
        "expected": planned_ops,
        "missing_from_ordered_coverage": missing,
        "passed": planned_index == len(planned_ops),
    }


def _run_generated_decode_profile(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    layer_count: int,
    batch_size: int,
    cache_len: int,
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
    decode_config["batch_size"] = batch_size
    decode_config["max_cache_len"] = cache_len
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(decode_config),
    )
    model.ops.enable_recording()

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
        hidden, reshape_hidden_ms = _time_section(
            lambda hidden=hidden: model.ops.reshape_decode_hidden_for_layer(
                hidden,
                op_name="reshape_hidden_decode",
            ),
            ttnn=ttnn,
            device=device,
        )
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
                op_name="residual_add.attn",
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
                op_name="residual_add.mlp",
            ),
            ttnn=ttnn,
            device=device,
        )
        layer_total_ms = (time.perf_counter() - layer_start) * 1000.0
        layer_profiles.append(
            {
                "layer_id": layer_id,
                "reshape_hidden_ms": reshape_hidden_ms,
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
    output_kind = str(plan.get("output_kind", "token"))
    output_shapes = {
        output_kind: _shape(token),
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
        "kind": output_kind,
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = _decode_step_reference(
        plan=plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output=output,
        observed_ops=_generated_observed_op_sequence(model, ttnn),
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
        "ttnn_environment": collect_ttnn_environment(ttnn),
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
            op_name="split_lm_head",
        )
        shard_logits.append(logits_i)

    logits = model.ops.concat(
        shard_logits,
        dim=-1,
        memory_config=_optional_attr(
            lm_head_config,
            "concat_memory_config",
        ),
        op_name="split_lm_head.concat",
    )
    _synchronize(ttnn, device)
    lm_head_ms = (time.perf_counter() - split_start) * 1000.0

    generation_config = _optional_attr(model.config, "generation", None)
    generation_mode = _optional_attr(generation_config, "mode", "greedy")
    retain_logits = bool(_optional_attr(lm_head_config, "retain_logits", False))
    if generation_mode == "greedy" and not retain_logits:
        def argmax_and_normalize() -> Any:
            token = model.ops.argmax(
                logits,
                dim=-1,
                op_name="argmax_or_sampling",
            )
            normalize_decode_token = getattr(
                model.ops,
                "normalize_decode_token",
                None,
            )
            if callable(normalize_decode_token):
                token = normalize_decode_token(
                    token,
                    batch_size=int(
                        _optional_attr(model.config, "batch_size", 1) or 1
                    ),
                )
            return token

        token, argmax_ms = _time_section(
            argmax_and_normalize,
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
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
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
    del host_params
    gc.collect()
    assert result.parameters is not None
    synthetic_inputs = _build_synthetic_decode_inputs(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=plan,
        prompt=prompt,
        tokenizer_path=tokenizer_path or model_path,
        tokenizer_module=tokenizer_module,
    )
    tensor_conversion_count = int(result.report["tensor_count"])
    rotary_runtime_state = None
    rotary_runtime_input_tensor_count = 0
    if synthetic_inputs.decode_runtime_state is not None:
        rotary_runtime = _attach_runtime_rotary_parameters(
            parameters=result.parameters,
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            plan=plan,
            cache_position_value=int(
                synthetic_inputs.decode_runtime_state[
                    "cache_position_value"
                ]
            ),
        )
        synthetic_rotary_count = 0
        rotary_runtime_input_tensor_count = (
            rotary_runtime.tensor_conversion_count
        )
        rotary_runtime_state = rotary_runtime.rotary_runtime_state
        tensor_conversion_count += rotary_runtime_input_tensor_count
    else:
        synthetic_rotary_count = _attach_synthetic_rotary_parameters(
            parameters=result.parameters,
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            plan=plan,
        )
        tensor_conversion_count += synthetic_rotary_count
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
        input_source=synthetic_inputs.input_source,
        prompt_tokenization=synthetic_inputs.prompt_tokenization,
        decode_runtime_state=synthetic_inputs.decode_runtime_state,
        rotary_runtime_state=rotary_runtime_state,
        kv_cache_runtime_state=synthetic_inputs.kv_cache_runtime_state,
        parameter_setup={
            "materialization": materialization_summary,
            "tensorization": _tensorization_summary(result.report),
            "synthetic_rotary_tensor_count": synthetic_rotary_count,
            "rotary_runtime_input_tensor_count": (
                rotary_runtime_input_tensor_count
            ),
            "rotary_runtime_state": rotary_runtime_state,
            "synthetic_runtime_input_tensor_count": (
                synthetic_inputs.synthetic_runtime_input_tensor_count
            ),
            "prompt_runtime_input_tensor_count": (
                synthetic_inputs.prompt_runtime_input_tensor_count
            ),
            "decode_runtime_state_input_tensor_count": (
                synthetic_inputs.decode_runtime_state_input_tensor_count
            ),
            "kv_cache_runtime_input_tensor_count": (
                synthetic_inputs.kv_cache_runtime_input_tensor_count
            ),
            "kv_cache_runtime_state": synthetic_inputs.kv_cache_runtime_state,
        },
    )


def _materialization_summary(params: Any) -> dict[str, Any]:
    metadata = dict(getattr(params, "metadata", {}))
    tensors = metadata.get("tensors", {})
    key_paths = [
        path
        for path in (
            "embedding.weight",
            "layers.0.input_norm.weight",
            "layers.0.post_attention_norm.weight",
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
    tensor_paths = sorted(
        str(record["path"])
        for record in tensors
        if isinstance(record, dict) and record.get("path") is not None
    )
    key_tensor_records = {}
    key_paths = [
        record["path"]
        for record in tensors
        if record.get("path")
        in {
            "embedding.weight",
            "layers.0.input_norm.weight",
            "layers.0.post_attention_norm.weight",
            "layers.0.attention.wqkv_packed.weight",
            "layers.0.attention.o_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "final_norm.weight",
            "lm_head.splits.0.weight",
        }
    ]
    for record in tensors:
        path = record.get("path")
        if path not in key_paths:
            continue
        key_tensor_records[path] = {
            "role": record.get("role"),
            "role_group": record.get("role_group"),
            "target_dtype": record.get("target_dtype"),
            "layout": record.get("layout"),
            "memory_config": record.get("memory_config"),
            "ttnn_dtype": record.get("ttnn_dtype"),
            "ttnn_layout": record.get("ttnn_layout"),
            "ttnn_memory_config": record.get("ttnn_memory_config"),
            "transform": record.get("transform"),
            "source_shape": record.get("source_shape"),
            "shape": record.get("shape"),
        }
    return {
        "status": report.get("status"),
        "backend": "ttnn",
        "roles": list(report.get("roles", [])),
        "tensor_count": report.get("tensor_count"),
        "target_dtype_counts": _field_counts(tensors, "target_dtype"),
        "layout_counts": _field_counts(tensors, "layout"),
        "memory_config_counts": _field_counts(tensors, "memory_config"),
        "transform_counts": _field_counts(tensors, "transform"),
        "transform_paths_by_kind": _paths_by_field_value(
            tensors,
            "transform",
        ),
        "ttnn_dtype_counts": _field_counts(tensors, "ttnn_dtype"),
        "ttnn_layout_counts": _field_counts(tensors, "ttnn_layout"),
        "ttnn_memory_config_counts": _field_counts(
            tensors,
            "ttnn_memory_config",
        ),
        "tensor_paths": tensor_paths,
        "key_paths": key_paths,
        "key_tensors": key_tensor_records,
    }


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _paths_by_field_value(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for record in records:
        value = record.get(field)
        path = record.get("path")
        if value is None or path is None:
            continue
        value = str(value)
        paths.setdefault(value, []).append(str(path))
    return {value: sorted(items) for value, items in sorted(paths.items())}


def _build_synthetic_decode_inputs(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
) -> SimpleNamespace:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    inputs = plan["input_shapes"]
    prompt_runtime_input_tensor_count = 0
    decode_runtime_state_input_tensor_count = 0
    kv_cache_runtime_input_tensor_count = 0
    prompt_tokenization = None
    decode_runtime_state = None
    kv_cache_runtime_state = None
    input_source = "synthetic"
    if prompt is None:
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
    else:
        prompt_runtime = _build_prompt_decode_token_ids(
            ttnn=ttnn,
            torch=torch,
            device=device,
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            tokenizer_module=tokenizer_module,
            batch_size=int(inputs["token_ids"][0]),
            vocab_size=plan.get("vocab_size"),
        )
        token_ids = prompt_runtime.token_ids
        prompt_runtime_input_tensor_count = (
            prompt_runtime.tensor_conversion_count
        )
        prompt_tokenization = prompt_runtime.prompt_tokenization
        runtime_state = _build_prompt_decode_runtime_state_tensors(
            ttnn=ttnn,
            torch=torch,
            device=device,
            batch_size=int(inputs["token_ids"][0]),
            cache_len=int((plan["kv_cache"]["logical_shape"])[1]),
            page_block_size=int(plan["kv_cache"]["page_block_size"]),
            prompt_token_count=int(prompt_tokenization["token_count"]),
        )
        page_table = runtime_state.page_table
        cache_position = runtime_state.cache_position
        decode_runtime_state_input_tensor_count = (
            runtime_state.tensor_conversion_count
        )
        decode_runtime_state = runtime_state.decode_runtime_state
        kv_runtime = _build_prompt_decode_kv_cache_tensors(
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            layer_count=int(plan["layers"]),
            batch_size=int(inputs["token_ids"][0]),
            cache_len=int((plan["kv_cache"]["logical_shape"])[1]),
            page_block_size=int(plan["kv_cache"]["page_block_size"]),
            num_kv_heads=int((plan["kv_cache"]["logical_shape"])[2]),
            head_dim=int((plan["kv_cache"]["logical_shape"])[3]),
        )
        kv_cache = kv_runtime.kv_cache
        kv_cache_runtime_input_tensor_count = (
            kv_runtime.tensor_conversion_count
        )
        kv_cache_runtime_state = kv_runtime.kv_cache_runtime_state
        input_source = "prompt_runtime"
    return SimpleNamespace(
        token_ids=token_ids,
        page_table=page_table,
        cache_position=cache_position,
        kv_cache=kv_cache,
        tensor_conversion_count=(
            tensor_count() + prompt_runtime_input_tensor_count
            + decode_runtime_state_input_tensor_count
            + kv_cache_runtime_input_tensor_count
        ),
        synthetic_runtime_input_tensor_count=tensor_count(),
        prompt_runtime_input_tensor_count=prompt_runtime_input_tensor_count,
        decode_runtime_state_input_tensor_count=(
            decode_runtime_state_input_tensor_count
        ),
        kv_cache_runtime_input_tensor_count=(
            kv_cache_runtime_input_tensor_count
        ),
        input_source=input_source,
        prompt_tokenization=prompt_tokenization,
        decode_runtime_state=decode_runtime_state,
        kv_cache_runtime_state=kv_cache_runtime_state,
    )


def _build_prompt_decode_runtime_state_tensors(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    prompt_token_count: int | list[int],
) -> SimpleNamespace:
    runtime_state = build_decode_runtime_state(
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        prompt_token_count=prompt_token_count,
    )
    kwargs = {"device": device}
    memory_config = _runtime_dram_memory_config(ttnn)
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    dtype = getattr(
        ttnn,
        "int32",
        getattr(ttnn, "uint32", getattr(ttnn, "bfloat16", None)),
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    page_table = ttnn.from_torch(
        _runtime_int_tensor(
            torch,
            runtime_state.page_table,
            name="runtime_page_table",
        ),
        **kwargs,
    )
    cache_position = ttnn.from_torch(
        _runtime_int_tensor(
            torch,
            runtime_state.cache_position,
            name="runtime_cache_position",
        ),
        **kwargs,
    )
    decode_runtime_state = runtime_state.to_report()
    decode_runtime_state["memory_config"] = "dram"
    decode_runtime_state["ttnn_memory_config"] = _config_repr(memory_config)
    return SimpleNamespace(
        page_table=page_table,
        cache_position=cache_position,
        tensor_conversion_count=2,
        decode_runtime_state=decode_runtime_state,
    )


_build_prompt_decode_runtime_state_tensors = _runtime_decode_state_tensors


def _build_prompt_decode_token_ids(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    prompt: str,
    tokenizer_path: str | Path | None,
    tokenizer_module: Any | None,
    batch_size: int,
    vocab_size: Any,
) -> SimpleNamespace:
    if tokenizer_path is None:
        raise PromptTokenizationError(
            "tokenizer_path or model_path is required when prompt is used"
        )
    tokenization = tokenize_prompt_for_decode(
        prompt=prompt,
        batch_size=batch_size,
        tokenizer_path=tokenizer_path,
        vocab_size=_safe_int_or_none(vocab_size),
        tokenizer_module=tokenizer_module,
    )
    host_tensor = _token_ids_tensor(
        torch,
        tokenization.token_ids,
        name="prompt_token_ids",
    )
    kwargs = {"device": device}
    memory_config = _runtime_dram_memory_config(ttnn)
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
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
    token_ids = ttnn.from_torch(host_tensor, **kwargs)
    return SimpleNamespace(
        token_ids=token_ids,
        tensor_conversion_count=1,
        prompt_tokenization=tokenization.to_report(),
    )


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _attach_runtime_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    cache_position_value: int,
) -> SimpleNamespace:
    layer_count = int(plan["layers"])
    layer_params = plan["layer_parameter_shapes"]
    cos_sin_shape = list(layer_params["rotary_cos_matrix"])
    batch_size = int(cos_sin_shape[1])
    head_dim = int(cos_sin_shape[-1])
    runtime_state = build_decode_rotary_runtime_state(
        layer_count=layer_count,
        batch_size=batch_size,
        head_dim=head_dim,
        cache_position_value=cache_position_value,
    )
    kwargs = {
        "device": device,
        "dtype": _ttnn_dtype(ttnn, dtype_seed),
    }
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    rotary_cos_sin_memory_config = _rotary_cos_sin_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    rotary_transform_memory_config = _rotary_transform_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
    )
    tensor_count = 0

    def runtime_tensor(
        name: str,
        shape: list[int],
        *,
        memory_config: Any | None,
    ) -> Any:
        nonlocal tensor_count
        tensor_count += 1
        tensor_kwargs = dict(kwargs)
        if memory_config is not None:
            tensor_kwargs["memory_config"] = memory_config
        return ttnn.from_torch(
            _runtime_float_tensor(
                torch,
                shape,
                dtype_seed=dtype_seed,
                name=name,
            ),
            **tensor_kwargs,
        )

    shared_rotary = SimpleNamespace(
        cos_matrix=runtime_tensor(
            "runtime.shared.rotary_cos",
            list(runtime_state.cos_sin_shape),
            memory_config=rotary_cos_sin_memory_config,
        ),
        sin_matrix=runtime_tensor(
            "runtime.shared.rotary_sin",
            list(runtime_state.cos_sin_shape),
            memory_config=rotary_cos_sin_memory_config,
        ),
        transformation_matrix=runtime_tensor(
            "runtime.shared.rotary_transform",
            list(runtime_state.transformation_shape),
            memory_config=rotary_transform_memory_config,
        ),
    )
    for layer_id in range(layer_count):
        layer = parameters.layers[layer_id]
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = SimpleNamespace()
            layer.attention = attention
        attention.rotary = shared_rotary

    rotary_runtime_state = runtime_state.to_report()
    rotary_runtime_state["tensor_count"] = tensor_count
    rotary_runtime_state["shared_across_layers"] = True
    rotary_runtime_state["memory_config"] = "height_sharded"
    rotary_runtime_state["ttnn_memory_config"] = _config_repr(
        rotary_cos_sin_memory_config
    )
    rotary_runtime_state["transform_memory_config"] = "height_sharded"
    rotary_runtime_state["transform_ttnn_memory_config"] = _config_repr(
        rotary_transform_memory_config
    )
    return SimpleNamespace(
        tensor_conversion_count=tensor_count,
        rotary_runtime_state=rotary_runtime_state,
    )


def _runtime_dram_memory_config(ttnn: Any) -> Any | None:
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def _config_repr(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _runtime_float_tensor(
    torch: Any,
    shape: list[int],
    *,
    dtype_seed: str,
    name: str,
) -> Any:
    zeros = getattr(torch, "zeros", None)
    if not callable(zeros):
        raise ValueError("torch module must provide zeros")
    dtype = (
        getattr(torch, "bfloat16", None)
        if dtype_seed == "bf16"
        else getattr(torch, "float32", None)
    )
    try:
        tensor = zeros(tuple(shape), dtype=dtype)
    except TypeError:
        tensor = zeros(tuple(shape))
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor


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
    output_kind = _decode_output_kind(config)
    expected_decode_output = (
        [batch_size, 1, vocab_size]
        if output_kind == "logits"
        else [batch_size, 1]
    )
    input_shapes = {
        "token_ids": [batch_size, 1],
        "page_table": [batch_size, page_count],
        "cache_position": [batch_size],
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
        "rotary_cos_matrix": _decode_rotary_cos_sin_shape(
            batch_size,
            head_dim,
        ),
        "rotary_sin_matrix": _decode_rotary_cos_sin_shape(
            batch_size,
            head_dim,
        ),
        "rotary_transformation_matrix": _decode_rotary_transform_shape(
            batch_size,
        ),
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
        "vocab_size": vocab_size,
        "input_shapes": input_shapes,
        "parameter_shapes": parameter_shapes,
        "layer_parameter_shapes": layer_parameter_shapes,
        "rotary": dict(config.get("rotary") or {}),
        "expected_intermediate_shapes": {
            "embedding": _decode_hidden_shape(batch_size, hidden_size),
            "qkv": _decode_hidden_shape(batch_size, qkv_size),
            "query": _decode_head_shape(batch_size, num_heads, head_dim),
            "key": _decode_head_shape(batch_size, num_kv_heads, head_dim),
            "value": _decode_head_shape(batch_size, num_kv_heads, head_dim),
            "attention": _decode_head_shape(batch_size, num_heads, head_dim),
            "concat_heads": _decode_hidden_shape(
                batch_size,
                num_heads * head_dim,
            ),
            "attention_output": _decode_hidden_shape(batch_size, hidden_size),
            "mlp_intermediate": _decode_hidden_shape(
                batch_size,
                intermediate_size,
            ),
        },
        "expected_output_shapes": {
            output_kind: expected_decode_output,
            "key_cache": input_shapes["key_cache"],
            "value_cache": input_shapes["value_cache"],
        },
        "output_kind": output_kind,
        "kv_cache": {
            "policy": kv_cache_config.get("policy", "paged"),
            "template": kv_cache_config.get("template", "paged_kv_cache"),
            "page_block_size": page_block_size,
            "page_count": page_count,
            "max_num_blocks": max_num_blocks,
            "physical_shape": kv_cache_shape,
            "logical_shape": [batch_size, cache_len, num_kv_heads, head_dim],
        },
        "tensor_conversion_count": 5 + len(lm_head_splits) + 12 * layers,
        "op_sequence": _decode_op_sequence(layers, output_kind=output_kind),
    }


def _decode_output_kind(config: dict[str, Any]) -> str:
    lm_head_config = config.get("lm_head")
    if not isinstance(lm_head_config, dict):
        lm_head_config = {}
    generation_config = config.get("generation")
    if not isinstance(generation_config, dict):
        generation_config = {}
    if bool(lm_head_config.get("retain_logits")):
        return "logits"
    if bool(generation_config.get("retain_logits")):
        return "logits"
    if generation_config.get("mode") == "full_logits":
        return "logits"
    if generation_config.get("template") == "full_logits":
        return "logits"
    return "token"


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
            shapes.append(_linear_weight_shape(hidden_size, vocab_end - vocab_start))
        return shapes

    base = vocab_size // split_count
    remainder = vocab_size % split_count
    shapes = []
    for shard_id in range(split_count):
        width = base + (1 if shard_id < remainder else 0)
        shapes.append(_linear_weight_shape(hidden_size, width))
    return shapes


def _embedding_weight_shape(vocab_size: int, hidden_size: int) -> list[int]:
    return [1, 1, vocab_size, hidden_size]


def _norm_weight_shape(hidden_size: int) -> list[int]:
    if hidden_size % 32 == 0:
        return [1, 1, hidden_size // 32, 32]
    return [1, 1, 1, hidden_size]


_decode_op_sequence = _runtime_decode_op_sequence
_decode_output_kind = _runtime_decode_output_kind
_decode_step_plan = _runtime_decode_step_plan
_embedding_weight_shape = _runtime_embedding_weight_shape
_lm_head_split_shapes = _runtime_lm_head_split_shapes
_norm_weight_shape = _runtime_norm_weight_shape


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
        "output_kind": plan["output_kind"],
        "op_sequence": plan["op_sequence"],
        "input_shapes": plan["input_shapes"],
        "parameter_shapes": plan["parameter_shapes"],
        "layer_parameter_shapes": plan["layer_parameter_shapes"],
        "expected_intermediate_shapes": plan["expected_intermediate_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "kv_cache": plan["kv_cache"],
        "ttnn_environment": collect_ttnn_environment(None),
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
            "ttnn_environment": collect_ttnn_environment(None),
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
    ttnn_module: Any | None = None,
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
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
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
    ttnn_module: Any | None = None,
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
            "throughput_summary": _throughput_summary(
                latency_ms=None,
                batch_size=batch_size,
                trace_report=None,
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
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
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
        "reshape_hidden_ms": 0.0,
        "rms_norm_attn_ms": 0.0,
        "attention_ms": 0.0,
        "residual_add_attn_ms": 0.0,
        "rms_norm_mlp_ms": 0.0,
        "mlp_ms": 0.0,
        "residual_add_mlp_ms": 0.0,
        "total_ms": 0.0,
    }


def _planned_lm_head_profile(config: dict[str, Any]) -> dict[str, Any]:
    lm_head = config.get("lm_head")
    if not isinstance(lm_head, dict):
        lm_head = {}
    generation = config.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    retain_logits = bool(lm_head.get("retain_logits")) or bool(
        generation.get("retain_logits")
    )
    try:
        split_count = int(lm_head.get("split_count", 0))
    except (TypeError, ValueError):
        split_count = 0
    return {
        "split_count": split_count,
        "lm_head_ms": 0.0,
        "argmax_ms": 0.0,
        "argmax_status": "skipped" if retain_logits else "profiled",
    }


def _bottleneck_summary(
    section_latency: dict[str, float],
    layer_profiles: list[dict[str, Any]],
    *,
    tensor_conversion_ms: float,
    trace_execute_ms: float,
) -> dict[str, Any]:
    reshape_hidden_ms = sum(
        float(layer.get("reshape_hidden_ms", 0.0)) for layer in layer_profiles
    )
    attention_ms = sum(float(layer["attention_ms"]) for layer in layer_profiles)
    mlp_ms = sum(float(layer["mlp_ms"]) for layer in layer_profiles)
    layer_total_ms = sum(float(layer["total_ms"]) for layer in layer_profiles)
    sections = {
        "tensor_conversion_ms": tensor_conversion_ms,
        "embedding_ms": float(section_latency["embedding_ms"]),
        "per_layer_reshape_hidden_ms": reshape_hidden_ms,
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


def _throughput_summary(
    *,
    latency_ms: float | None,
    batch_size: int,
    trace_report: dict[str, Any] | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "batch_size": batch_size,
        "generated_tokens_per_user": 1,
        "total_generated_tokens": batch_size,
        "basis": "decode_step_latency_ms",
    }
    if dry_run:
        summary.update(
            {
                "status": "dry_run",
                "latency_ms": 0.0,
                "tokens_per_second_per_user": 0.0,
                "aggregate_tokens_per_second": 0.0,
            }
        )
        return summary

    if latency_ms is None or latency_ms <= 0.0:
        summary.update(
            {
                "status": "unavailable",
                "latency_ms": latency_ms,
                "tokens_per_second_per_user": None,
                "aggregate_tokens_per_second": None,
            }
        )
        return summary

    tokens_per_second_per_user = 1000.0 / latency_ms
    summary.update(
        {
            "status": "measured",
            "latency_ms": latency_ms,
            "tokens_per_second_per_user": tokens_per_second_per_user,
            "aggregate_tokens_per_second": (
                tokens_per_second_per_user * batch_size
            ),
        }
    )

    trace_samples = (
        (trace_report or {}).get("execute_samples_ms")
        if trace_report is not None
        else None
    )
    if trace_samples:
        trace_mean_ms = sum(float(sample) for sample in trace_samples) / len(
            trace_samples
        )
        if trace_mean_ms > 0.0:
            trace_tps_per_user = 1000.0 / trace_mean_ms
            summary.update(
                {
                    "trace_execute_mean_ms": trace_mean_ms,
                    "trace_execute_tokens_per_second_per_user": (
                        trace_tps_per_user
                    ),
                    "trace_execute_aggregate_tokens_per_second": (
                        trace_tps_per_user * batch_size
                    ),
                    "trace_iterations": len(trace_samples),
                }
            )
    return summary


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
    safe_report = _json_safe_report_value(report)
    report.clear()
    report.update(safe_report)
    out_path.write_text(json.dumps(report, indent=2) + "\n")


def _json_safe_report_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_report_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_report_value(item) for item in value]
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }
