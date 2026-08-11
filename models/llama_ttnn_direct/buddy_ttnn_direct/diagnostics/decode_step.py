from __future__ import annotations

import gc
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
)
from ..codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
    to_ttnn_parameters,
)
from ..runtime_environment import collect_ttnn_environment
from ..runtime.config_runtime import realize_ttnn_config
from ..runtime.decode_inputs import DecodeInputBuffers
from ..runtime.inputs import (
    build_prompt_decode_runtime_state_tensors,
    build_token_ids_tensor,
)
from ..runtime.tokenizer import PromptTokenizationError, tokenize_prompt_for_decode
from ..runtime.kv_cache import build_prompt_decode_kv_cache_tensors
from ..runtime.model_loader import load_generated_model, to_namespace
from ..runtime.model_setup import materialization_summary, tensorization_summary
from ..runtime.plans import (
    DECODE_PARAMETER_ROLES,
    decode_op_sequence,
    decode_step_plan,
)
from ..runtime.decode import _time_decode_step
from ..runtime.device import managed_ttnn_device as _maybe_managed_device
from ..runtime.profile import GenerateSectionProfiler
from ..runtime.reports import trace_report as _trace_report, write_report
from ..runtime.rotary import attach_decode_rotary_parameters
from ..runtime.structural import (
    decode_step_reference,
    generated_observed_op_sequence,
)
from ..runtime.tensor_meta import tensor_dtype as _dtype, tensor_shape
from ..runtime.trace import (
    DecodeTraceSession,
    _synchronize,
    build_decode_trace_key,
    decode_trace_region_size,
)
from .attention_support import (
    rotary_cos_sin_height_sharded_memory_config as _rotary_cos_sin_height_sharded_memory_config,
    rotary_transform_height_sharded_memory_config as _rotary_transform_height_sharded_memory_config,
)
from .support import (
    dry_run_reference as _dry_run_reference,
    shape as _shape,
    synthetic_tensor_factory as _synthetic_tensor_factory,
)
from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..ttnn_compat import UnsupportedTTNNOp

SINGLE_LAYER_DECODE_OPS = decode_op_sequence(1)


def run_smoke_single_layer_decode(**kwargs: Any) -> dict[str, Any]:
    return run_smoke_decode_step(layers=1, **kwargs)


def run_smoke_decode_step(**kwargs: Any) -> dict[str, Any]:
    return _run_decode_diagnostic(profile=False, **kwargs)


def profile_decode_step(**kwargs: Any) -> dict[str, Any]:
    return _run_decode_diagnostic(profile=True, **kwargs)


def _run_decode_diagnostic(
    *,
    profile: bool,
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
    request = SimpleNamespace(
        program_dir=program_root,
        config=config,
        layers=layer_count,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        model_path=None if model_path is None else Path(model_path),
        dtype_seed=dtype_seed,
        trace=trace,
        trace_iterations=trace_iterations,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        tokenizer_module=tokenizer_module,
        profile=profile,
        plan=decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
        ),
    )

    if dry_run:
        report = _dry_run_report(request)
        write_report(out, report)
        return report

    dependency = "ttnn"
    ttnn = None
    try:
        ttnn = (
            ttnn_module if ttnn_module is not None else importlib.import_module("ttnn")
        )
        dependency = "torch"
        try:
            torch = torch_module or importlib.import_module("torch")
        except ImportError:
            if parameters is None or any(
                value is None
                for value in (token_ids, page_table, cache_position, kv_cache)
            ):
                raise
            torch = None
        dependency = None
        with _maybe_managed_device(
            ttnn,
            device_id,
            ttnn_module,
            trace_region_size=(
                decode_trace_region_size(config) if trace else None
            ),
        ) as ttnn_device:
            state = _prepare_decode_state(
                request=request,
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                injected=SimpleNamespace(
                    parameters=parameters,
                    token_ids=token_ids,
                    page_table=page_table,
                    cache_position=cache_position,
                    kv_cache=kv_cache,
                ),
            )
            runner = (
                _run_generated_decode_profile
                if profile
                else _run_generated_decode_step
            )
            result = runner(
                ttnn=ttnn,
                device=ttnn_device,
                request=request,
                state=state,
                torch=torch,
                realize_config=ttnn_module is None,
            )
            report = _base_report(request, dry_run=False)
            report.update(result)
            _attach_state_report(report, state)
            if profile:
                _attach_profile_summaries(report, request, state)
    except Exception as err:
        report = _failure_report(
            request,
            err,
            ttnn=ttnn,
            dependency=dependency,
        )

    write_report(out, report)
    return report


def _prepare_decode_state(
    *, request: Any, ttnn: Any, torch: Any, device: Any, injected: Any
) -> Any:
    start = time.perf_counter()
    if injected.parameters is None:
        assert torch is not None
        state = (
            _build_synthetic_decode_state(
                ttnn=ttnn,
                torch=torch,
                device=device,
                dtype_seed=request.dtype_seed,
                plan=request.plan,
            )
            if request.model_path is None
            else _build_model_decode_state(
                ttnn=ttnn,
                torch=torch,
                device=device,
                dtype_seed=request.dtype_seed,
                plan=request.plan,
                program_dir=request.program_dir,
                model_path=request.model_path,
                prompt=request.prompt,
                tokenizer_path=request.tokenizer_path,
                tokenizer_module=request.tokenizer_module,
            )
        )
    else:
        state = injected
        state.tensor_conversion_count = 0
        state.parameter_source = "injected"
        state.input_source = "injected"
    if any(
        getattr(state, name, None) is None
        for name in ("token_ids", "page_table", "cache_position", "kv_cache")
    ):
        raise ValueError(
            "token_ids, page_table, cache_position, and kv_cache are required "
            "when parameters are injected"
        )
    state.tensor_conversion_ms = (time.perf_counter() - start) * 1000.0
    return state


def _attach_state_report(report: dict[str, Any], state: Any) -> None:
    report.update(
        {
            "tensor_conversion_count": state.tensor_conversion_count,
            "parameter_source": state.parameter_source,
            "input_source": state.input_source,
        }
    )
    for name in (
        "parameter_setup",
        "prompt_tokenization",
        "decode_runtime_state",
        "rotary_runtime_state",
        "kv_cache_runtime_state",
    ):
        value = getattr(state, name, None)
        if value is not None:
            report[name] = value


def _attach_profile_summaries(
    report: dict[str, Any], request: Any, state: Any
) -> None:
    conversion_ms = state.tensor_conversion_ms
    report["tensor_conversion_ms"] = conversion_ms
    report["bottleneck_summary"] = _bottleneck_summary(
        report["section_latency_ms"],
        report["layer_profiles"],
        tensor_conversion_ms=conversion_ms,
        trace_execute_ms=float(report["trace"].get("execute_latency_ms") or 0.0),
    )
    report["throughput_summary"] = _throughput_summary(
        latency_ms=float(report["latency_ms"]),
        batch_size=request.batch_size,
        trace_report=report["trace"],
    )


def _run_generated_decode_step(
    *,
    ttnn: Any,
    device: Any,
    request: Any,
    state: Any,
    torch: Any,
    realize_config: bool = False,
) -> dict[str, Any]:
    model = _build_decode_model(
        ttnn=ttnn,
        device=device,
        request=request,
        state=state,
        realize_config=realize_config,
    )
    decode_args = {
        "ttnn": ttnn,
        "model": model,
        "device": device,
        "token_ids": state.token_ids,
        "page_table": state.page_table,
        "cache_position": state.cache_position,
        "kv_cache": state.kv_cache,
    }
    trace_report = _trace_report(requested=request.trace, status="disabled")
    if request.trace:
        if _trace_apis_available(ttnn):
            try:
                token, kv_cache, latency_ms, trace_report = (
                    _run_decode_step_with_trace(
                        ttnn=ttnn,
                        model=model,
                        device=device,
                        state=state,
                        request=request,
                        torch=torch,
                    )
                )
            except Exception as err:
                trace_report = _trace_report(
                    requested=True,
                    status="trace_failed_fell_back_to_eager",
                    iterations=request.trace_iterations,
                    error=f"{type(err).__name__}: {err}",
                )
                token, kv_cache, latency_ms = _time_decode_step(**decode_args)
        else:
            trace_report = _trace_report(
                requested=True,
                status="trace_api_unavailable_fell_back_to_eager",
                iterations=request.trace_iterations,
            )
            token, kv_cache, latency_ms = _time_decode_step(**decode_args)
    else:
        token, kv_cache, latency_ms = _time_decode_step(**decode_args)

    result = _decode_result(ttnn, model, token, kv_cache, request)
    result.update(
        {
        "status": "passed" if result["passed"] else "reference_mismatch",
        "latency_ms": latency_ms,
        "error": None if result["passed"] else "decode-step structural reference mismatch",
        "trace": trace_report,
        }
    )
    return result


def _run_generated_decode_profile(
    *,
    ttnn: Any,
    device: Any,
    request: Any,
    state: Any,
    torch: Any,
    realize_config: bool = False,
) -> dict[str, Any]:
    model = _build_decode_model(
        ttnn=ttnn,
        device=device,
        request=request,
        state=state,
        realize_config=realize_config,
    )
    profiler = GenerateSectionProfiler(ttnn=ttnn, device=device)
    profiler.install(model)
    total_start = time.perf_counter()
    token, kv_cache = model.decode_step(
        state.token_ids,
        state.page_table,
        state.cache_position,
        state.kv_cache,
    )
    _synchronize(ttnn, device)
    latency_ms = (time.perf_counter() - total_start) * 1000.0
    measured = profiler.to_report()
    sections = measured["sections_ms"]
    section_latency = {
        name: float(sections.get(name) or 0.0)
        for name in _empty_section_latency()
    }
    layer_profiles = []
    for observed in measured["decode_layer_profiles"]:
        profile = _planned_layer_profile(int(observed["layer_id"]))
        profile["attention_ms"] = float(observed["attention_ms"])
        profile["mlp_ms"] = float(observed["mlp_ms"])
        profile["total_ms"] = profile["attention_ms"] + profile["mlp_ms"]
        layer_profiles.append(profile)
    lm_head_profile = _planned_lm_head_profile(request.config)
    lm_head_profile.update(
        lm_head_ms=section_latency["lm_head_ms"],
        argmax_ms=section_latency["argmax_ms"],
        argmax_strategy=str(
            (request.config.get("lm_head") or {}).get(
                "argmax_strategy",
                "full_logits_untilize_multicore_argmax",
            )
        ),
    )

    trace_report = _trace_report(requested=request.trace, status="disabled")
    if request.trace:
        if _trace_apis_available(ttnn):
            try:
                trace_model = _build_decode_model(
                    ttnn=ttnn,
                    device=device,
                    request=request,
                    state=state,
                    realize_config=realize_config,
                )
                _, _, _, trace_report = _run_decode_step_with_trace(
                    ttnn=ttnn,
                    model=trace_model,
                    device=device,
                    state=state,
                    request=request,
                    torch=torch,
                )
            except Exception as err:
                trace_report = _trace_report(
                    requested=True,
                    status="trace_failed_after_profile",
                    iterations=request.trace_iterations,
                    error=f"{type(err).__name__}: {err}",
                )
        else:
            trace_report = _trace_report(
                requested=True,
                status="trace_api_unavailable",
                iterations=request.trace_iterations,
            )

    result = _decode_result(ttnn, model, token, kv_cache, request)
    result.update(
        {
            "status": "profiled" if result["passed"] else "reference_mismatch",
            "latency_ms": latency_ms,
            "section_latency_ms": section_latency,
            "layer_profiles": layer_profiles,
            "lm_head_profile": lm_head_profile,
            "profile_basis": measured["basis"],
            "host_copy_ms": 0.0,
            "host_copy_status": "not_performed",
            "trace": trace_report,
            "error": (
                None
                if result["passed"]
                else "decode-step profile structural mismatch"
            ),
        }
    )
    return result


def _build_decode_model(
    *, ttnn: Any, device: Any, request: Any, state: Any, realize_config: bool
) -> Any:
    generated = load_generated_model(request.program_dir / "model.py", ttnn)
    config = {
        **request.config,
        "num_layers": request.layers,
        "batch_size": request.batch_size,
        "max_cache_len": request.cache_len,
    }
    if realize_config:
        config = realize_ttnn_config(config, ttnn)
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=state.parameters,
        config=to_namespace(config),
    )
    model.ops.enable_recording()
    return model


def _decode_result(
    ttnn: Any, model: Any, token: Any, kv_cache: Any, request: Any
) -> dict[str, Any]:
    output_kind = str(request.plan.get("output_kind", "token"))
    output_shapes = {
        output_kind: _shape(token),
        "key_cache": _shape(kv_cache[0].k),
        "value_cache": _shape(kv_cache[0].v),
        "kv_cache_layers": [
            {
                "layer_id": layer_id,
                "key_cache": _shape(cache.k),
                "value_cache": _shape(cache.v),
            }
            for layer_id, cache in enumerate(kv_cache[: request.layers])
        ],
    }
    output = {
        "kind": output_kind,
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = decode_step_reference(
        plan=request.plan,
        layer_count=request.layers,
        output_shapes=output_shapes,
        output=output,
        observed_ops=generated_observed_op_sequence(model, ttnn),
    )
    return {
        "passed": bool(reference["passed"]),
        "output_shapes": output_shapes,
        "output": output,
        "ttnn_version": getattr(ttnn, "__version__", None),
        "ttnn_environment": collect_ttnn_environment(ttnn),
        "reference": reference,
    }


def _run_decode_step_with_trace(
    *,
    ttnn: Any,
    model: Any,
    device: Any,
    state: Any,
    request: Any,
    torch: Any,
) -> tuple[Any, Any, float, dict[str, Any]]:
    runtime_state = getattr(state, "decode_runtime_state", None) or {}
    positions = runtime_state.get("cache_position_values")
    if positions is None:
        position = int(runtime_state.get("cache_position_value") or 0)
        positions = [position] * int(request.batch_size)
    persistent_inputs = DecodeInputBuffers(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=request.dtype_seed,
        parameters=state.parameters,
        decode_plan=request.plan,
        batch_size=request.batch_size,
        cache_len=request.cache_len,
        prefill_effective_token_count=positions,
        token_input=state.token_ids,
    )
    session = DecodeTraceSession(
        ttnn=ttnn,
        device=device,
        model=model,
        persistent_inputs=persistent_inputs,
        kv_cache=state.kv_cache,
        key=build_decode_trace_key(
            device_id=request.device_id,
            config=request.config,
            decode_plan=request.plan,
            layer_count=request.layers,
            batch_size=request.batch_size,
            cache_len=request.cache_len,
            dtype_seed=request.dtype_seed,
        ),
    )
    try:
        capture_start = time.perf_counter()
        session.capture()
        capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0
        execute_samples = []
        execution = None
        for _ in range(request.trace_iterations):
            execution = session.execute()
            execute_samples.append(float(execution.latency_ms))
        assert execution is not None
        execute_latency_ms = sum(execute_samples)
    finally:
        session.close()
    runtime_report = session.to_report()
    trace_report = _trace_report(
        requested=True,
        status="captured_and_executed",
        iterations=request.trace_iterations,
        trace_id=session.trace_id,
        capture_latency_ms=capture_latency_ms,
        execute_latency_ms=execute_latency_ms,
        execute_samples_ms=execute_samples,
    )
    trace_report["runtime"] = runtime_report
    return (
        execution.token,
        execution.kv_cache,
        capture_latency_ms + execute_latency_ms,
        trace_report,
    )


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
    parameters = SimpleNamespace(
        embedding=_weight(tensor, params["embedding"], "embedding"),
        layers=[
            _synthetic_layer(tensor, layer_params, layer_id)
            for layer_id in range(int(plan["layers"]))
        ],
        final_norm=_weight(tensor, params["final_norm"], "final_norm"),
        lm_head=SimpleNamespace(
            splits=[
                SimpleNamespace(
                    shard_id=shard_id,
                    weight=tensor(shape, name=f"lm_head_{shard_id}"),
                )
                for shard_id, shape in enumerate(params["lm_head_splits"])
            ]
        ),
    )
    token_ids, page_table, cache_position = (
        tensor(inputs[name], name=name, zeros=True)
        for name in ("token_ids", "page_table", "cache_position")
    )
    return SimpleNamespace(
        parameters=parameters,
        token_ids=token_ids,
        page_table=page_table,
        cache_position=cache_position,
        kv_cache=_zero_kv_cache(
            tensor,
            inputs["key_cache"],
            inputs["value_cache"],
            int(plan["layers"]),
        ),
        tensor_conversion_count=tensor_count(),
        parameter_source="synthetic",
        input_source="synthetic",
    )


def _weight(tensor: Any, shape: list[int], name: str) -> Any:
    return SimpleNamespace(weight=tensor(shape, name=name))


def _synthetic_layer(tensor: Any, shapes: dict[str, Any], layer_id: int) -> Any:
    prefix = f"layers.{layer_id}"
    if "mlp_gate_up" in shapes:
        mlp = SimpleNamespace(
            gate_up_proj=_weight(tensor, shapes["mlp_gate_up"], f"{prefix}.mlp_gate_up"),
            down_proj=_weight(tensor, shapes["mlp_down"], f"{prefix}.mlp_down"),
        )
    else:
        mlp = SimpleNamespace(
            gate_proj=_weight(tensor, shapes["mlp_gate"], f"{prefix}.mlp_gate"),
            up_proj=_weight(tensor, shapes["mlp_up"], f"{prefix}.mlp_up"),
            down_proj=_weight(tensor, shapes["mlp_down"], f"{prefix}.mlp_down"),
        )
    attention = SimpleNamespace(
        wqkv_packed=_weight(tensor, shapes["attention_wqkv"], f"{prefix}.attention_wqkv"),
        o_proj=_weight(tensor, shapes["attention_o_proj"], f"{prefix}.o_proj"),
        rotary=SimpleNamespace(
            cos_matrix=tensor(shapes["rotary_cos_matrix"], name=f"{prefix}.rotary_cos"),
            sin_matrix=tensor(shapes["rotary_sin_matrix"], name=f"{prefix}.rotary_sin"),
            transformation_matrix=tensor(
                shapes["rotary_transformation_matrix"], name=f"{prefix}.rotary_transform"
            ),
        ),
    )
    return SimpleNamespace(
        attention=attention,
        input_norm=_weight(tensor, shapes["input_norm"], f"{prefix}.input_norm"),
        post_attention_norm=_weight(
            tensor, shapes["post_attention_norm"], f"{prefix}.post_attention_norm"
        ),
        mlp=mlp,
    )


def _zero_kv_cache(
    tensor: Any, key_shape: list[int], value_shape: list[int], layers: int
) -> list[Any]:
    return [
        SimpleNamespace(
            k=tensor(key_shape, name=f"layers.{i}.key_cache", zeros=True),
            v=tensor(value_shape, name=f"layers.{i}.value_cache", zeros=True),
        )
        for i in range(layers)
    ]


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
    materialization_report = materialization_summary(host_params)
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
        rotary_runtime = attach_decode_rotary_parameters(
            parameters=result.parameters,
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            plan=plan,
            cache_position_value=synthetic_inputs.decode_runtime_state.get(
                "cache_position_value"
            ),
            cache_position_values=synthetic_inputs.decode_runtime_state.get(
                "cache_position_values"
            ),
        )
        synthetic_rotary_count = 0
        rotary_runtime_input_tensor_count = rotary_runtime.tensor_conversion_count
        rotary_runtime_state = dict(rotary_runtime.rotary_runtime_state)
        rotary_runtime_state["matrix_shape"] = list(
            rotary_runtime_state["cos_sin_shape"]
        )
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
            "materialization": materialization_report,
            "tensorization": tensorization_summary(result.report),
            "synthetic_rotary_tensor_count": synthetic_rotary_count,
            "rotary_runtime_input_tensor_count": (rotary_runtime_input_tensor_count),
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
    kv_cache_runtime_input_tensor_count = 0
    synthetic_int_input_tensor_count = 0
    prompt_tokenization = decode_runtime_state = kv_cache_runtime_state = None
    input_source = "synthetic"
    batch_size = int(inputs["token_ids"][0])
    cache_len = int(plan["kv_cache"]["logical_shape"][1])
    page_block_size = int(plan["kv_cache"]["page_block_size"])
    if prompt is None:
        token_ids = build_token_ids_tensor(
            ttnn=ttnn,
            torch=torch,
            device=device,
            token_ids=[[0] for _ in range(batch_size)],
            name="synthetic_token_ids",
        )
        prompt_token_count = 1
        synthetic_int_input_tensor_count = 1
        kv_cache = _zero_kv_cache(
            tensor,
            inputs["key_cache"],
            inputs["value_cache"],
            int(plan["layers"]),
        )
    else:
        if tokenizer_path is None:
            raise PromptTokenizationError(
                "tokenizer_path or model_path is required when prompt is used"
            )
        vocab_size = plan.get("vocab_size")
        try:
            vocab_size = int(vocab_size)
        except (TypeError, ValueError):
            vocab_size = None
        tokenization = tokenize_prompt_for_decode(
            prompt=prompt,
            batch_size=batch_size,
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            tokenizer_module=tokenizer_module,
        )
        token_ids = build_token_ids_tensor(
            ttnn=ttnn,
            torch=torch,
            device=device,
            token_ids=tokenization.token_ids,
            name="prompt_token_ids",
        )
        prompt_runtime_input_tensor_count = 1
        prompt_tokenization = tokenization.to_report()
        prompt_token_count = int(prompt_tokenization["token_count"])
        kv_runtime = build_prompt_decode_kv_cache_tensors(
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            layer_count=int(plan["layers"]),
            batch_size=batch_size,
            cache_len=cache_len,
            page_block_size=page_block_size,
            num_kv_heads=int(plan["kv_cache"]["logical_shape"][2]),
            head_dim=int(plan["kv_cache"]["logical_shape"][3]),
        )
        kv_cache = kv_runtime.kv_cache
        kv_cache_runtime_input_tensor_count = kv_runtime.tensor_conversion_count
        kv_cache_runtime_state = kv_runtime.kv_cache_runtime_state
        input_source = "prompt_runtime"
    runtime_state = build_prompt_decode_runtime_state_tensors(
        ttnn=ttnn,
        torch=torch,
        device=device,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        prompt_token_count=prompt_token_count,
    )
    page_table = runtime_state.page_table
    cache_position = runtime_state.cache_position
    decode_runtime_state_input_tensor_count = runtime_state.tensor_conversion_count
    decode_runtime_state = runtime_state.decode_runtime_state
    synthetic_runtime_input_tensor_count = (
        tensor_count()
        + synthetic_int_input_tensor_count
        + (decode_runtime_state_input_tensor_count if prompt is None else 0)
    )
    return SimpleNamespace(
        token_ids=token_ids,
        page_table=page_table,
        cache_position=cache_position,
        kv_cache=kv_cache,
        tensor_conversion_count=(
            tensor_count()
            + synthetic_int_input_tensor_count
            + prompt_runtime_input_tensor_count
            + decode_runtime_state_input_tensor_count
            + kv_cache_runtime_input_tensor_count
        ),
        synthetic_runtime_input_tensor_count=(synthetic_runtime_input_tensor_count),
        prompt_runtime_input_tensor_count=prompt_runtime_input_tensor_count,
        decode_runtime_state_input_tensor_count=(
            decode_runtime_state_input_tensor_count
        ),
        kv_cache_runtime_input_tensor_count=(kv_cache_runtime_input_tensor_count),
        input_source=input_source,
        prompt_tokenization=prompt_tokenization,
        decode_runtime_state=decode_runtime_state,
        kv_cache_runtime_state=kv_cache_runtime_state,
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
    cos_sin_shape = list(layer_params["rotary_cos_matrix"])
    batch_size = int(cos_sin_shape[1])
    head_dim = int(cos_sin_shape[-1])
    cos_sin_memory_config = _rotary_cos_sin_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    transform_memory_config = _rotary_transform_height_sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
    )
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
                memory_config=cos_sin_memory_config,
            ),
            sin_matrix=tensor(
                layer_params["rotary_sin_matrix"],
                name=f"layers.{layer_id}.rotary_sin",
                memory_config=cos_sin_memory_config,
            ),
            transformation_matrix=tensor(
                layer_params["rotary_transformation_matrix"],
                name=f"layers.{layer_id}.rotary_transform",
                memory_config=transform_memory_config,
            ),
        )
    return tensor_count()


def _base_report(request: Any, *, dry_run: bool) -> dict[str, Any]:
    plan = request.plan
    report = {
        "schema_version": 1,
        "template": (
            "generated_decode_step_profile"
            if request.profile
            else "generated_decode_step"
        ),
        "program_dir": str(request.program_dir),
        "layers": request.layers,
        "device": request.device,
        "device_id": request.device_id,
        "batch_size": request.batch_size,
        "cache_len": request.cache_len,
        "dtype_seed": request.dtype_seed,
        "dtype": "bfloat16" if request.dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": dry_run,
        "trace_enabled": request.trace,
        "trace_iterations": request.trace_iterations if request.trace else 0,
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
    if request.profile:
        report["profile_sections"] = [
            "tensor_conversion_ms",
            "embedding_ms",
            "per_layer_attention_ms",
            "per_layer_mlp_ms",
            "final_norm_ms",
            "lm_head_ms",
            "argmax_ms",
            "host_copy_ms",
            "trace_execute_ms",
        ]
    return report


def _dry_run_report(request: Any) -> dict[str, Any]:
    report = _base_report(request, dry_run=True)
    report.update(
        {
            "passed": True,
            "status": "dry_run",
            "latency_ms": 0.0,
            "output_shapes": None,
            "tensor_conversion_count": request.plan["tensor_conversion_count"],
            "error": None,
            "ttnn_version": None,
            "trace": _trace_report(
                requested=request.trace,
                status="dry_run" if request.trace else "disabled",
                iterations=request.trace_iterations if request.trace else 0,
            ),
            "reference": _dry_run_reference(
                "generated_decode_step_profile"
                if request.profile
                else "generated_decode_step"
            ),
            "message": "Dry run only; TTNN device is not required.",
        }
    )
    if request.profile:
        empty = _empty_section_latency()
        report.update(
            {
                "section_latency_ms": empty,
                "layer_profiles": [
                    _planned_layer_profile(i) for i in range(request.layers)
                ],
                "lm_head_profile": _planned_lm_head_profile(request.config),
                "bottleneck_summary": _bottleneck_summary(
                    empty, [], tensor_conversion_ms=0.0, trace_execute_ms=0.0
                ),
                "throughput_summary": _throughput_summary(
                    latency_ms=0.0,
                    batch_size=request.batch_size,
                    trace_report=None,
                    dry_run=True,
                ),
                "tensor_conversion_ms": 0.0,
                "host_copy_ms": 0.0,
            }
        )
    return report


def _failure_report(
    request: Any, error: Exception, *, ttnn: Any, dependency: str | None
) -> dict[str, Any]:
    if dependency == "ttnn":
        status, message, detail = "no_device", NO_TTNN_DEVICE_MESSAGE, str(error)
    elif dependency == "torch":
        status = "missing_torch"
        message = "torch is required to synthesize generated decode parameters and inputs."
        detail = str(error)
    elif isinstance(error, NoTTNNDeviceError):
        status, message, detail = "no_device", NO_TTNN_DEVICE_MESSAGE, str(error)
    elif isinstance(error, (ParameterMaterializationError, TTNNTensorizationError, PromptTokenizationError)):
        status, message, detail = "parameter_setup_error", str(error), str(error)
    elif isinstance(error, UnsupportedTTNNOp):
        status, message, detail = "api_mismatch", str(error), error.op_name
    else:
        status = "runtime_error"
        message = f"{type(error).__name__}: {error}"
        detail = str(error)
    report = _base_report(request, dry_run=False)
    report.update(
        {
            "passed": False,
            "status": status,
            "latency_ms": None,
            "output_shapes": None,
            "tensor_conversion_count": 0,
            "error": message,
            "detail": detail,
            "ttnn_version": getattr(ttnn, "__version__", None),
            "ttnn_environment": collect_ttnn_environment(ttnn),
            "trace": _trace_report(
                requested=request.trace,
                status="unavailable" if request.trace else "disabled",
                iterations=request.trace_iterations if request.trace else 0,
            ),
        }
    )
    if request.profile:
        empty = _empty_section_latency()
        report.update(
            {
                "section_latency_ms": empty,
                "layer_profiles": [],
                "bottleneck_summary": _bottleneck_summary(
                    empty, [], tensor_conversion_ms=0.0, trace_execute_ms=0.0
                ),
                "throughput_summary": _throughput_summary(
                    latency_ms=None,
                    batch_size=request.batch_size,
                    trace_report=None,
                ),
                "tensor_conversion_ms": 0.0,
                "host_copy_ms": 0.0,
            }
        )
    return report


def _empty_section_latency() -> dict[str, float]:
    return dict.fromkeys(
        ("embedding_ms", "final_norm_ms", "lm_head_ms", "argmax_ms", "host_copy_ms"),
        0.0,
    )


def _planned_layer_profile(layer_id: int) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        **dict.fromkeys(
            (
                "reshape_hidden_ms", "rms_norm_attn_ms", "attention_ms",
                "residual_add_attn_ms", "rms_norm_mlp_ms", "mlp_ms",
                "residual_add_mlp_ms", "total_ms",
            ),
            0.0,
        ),
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
    if dry_run or latency_ms is None or latency_ms <= 0.0:
        tps = 0.0 if dry_run else None
        summary.update(
            status="dry_run" if dry_run else "unavailable",
            latency_ms=0.0 if dry_run else latency_ms,
            tokens_per_second_per_user=tps,
            aggregate_tokens_per_second=tps,
        )
        return summary

    tokens_per_second_per_user = 1000.0 / latency_ms
    summary.update(
        status="measured",
        latency_ms=latency_ms,
        tokens_per_second_per_user=tokens_per_second_per_user,
        aggregate_tokens_per_second=tokens_per_second_per_user * batch_size,
    )
    trace_samples = (trace_report or {}).get("execute_samples_ms")
    if trace_samples:
        trace_mean_ms = sum(map(float, trace_samples)) / len(trace_samples)
        if trace_mean_ms > 0.0:
            trace_tps_per_user = 1000.0 / trace_mean_ms
            summary.update(
                trace_execute_mean_ms=trace_mean_ms,
                trace_execute_tokens_per_second_per_user=trace_tps_per_user,
                trace_execute_aggregate_tokens_per_second=(
                    trace_tps_per_user * batch_size
                ),
                trace_iterations=len(trace_samples),
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
