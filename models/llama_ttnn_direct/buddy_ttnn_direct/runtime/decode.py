from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from .decode_inputs import (
    DecodeInputBuffers,
    PersistentDecodeInputsUnsupported,
)
from .inputs import (
    build_prompt_decode_runtime_state_tensors,
    build_token_ids_tensor,
)
from .reports import append_json_line
from .rotary import attach_decode_rotary_parameters
from .sdpa_context import SDPAContextRuntime
from .structural import (
    decode_step_reference,
    generated_observed_op_sequence,
    loop_generated_token_ids,
    loop_input_shapes,
    loop_output_shapes,
)
from .tensor_meta import tensor_dtype, tensor_shape
from .trace import (
    BucketedDecodeTraceSession,
    DecodeTraceKey,
    DecodeTraceSession,
)

# Compatibility names remain patchable for existing diagnostic tests.
_build_prompt_decode_runtime_state_tensors = (
    build_prompt_decode_runtime_state_tensors
)
_decode_step_reference = decode_step_reference
_generated_observed_op_sequence = generated_observed_op_sequence
_loop_generated_token_ids = loop_generated_token_ids
_loop_input_shapes = loop_input_shapes
_loop_output_shapes = loop_output_shapes


def prefill_token_direct_handoff(*, prefill_token: Any) -> SimpleNamespace:
    return SimpleNamespace(
        status="device_tensor_direct",
        source="prefill_output_tensor",
        token_ids=prefill_token,
        tensor_conversion_count=0,
        runtime_handoff="device_tensor_direct",
        runtime_host_roundtrip=False,
    )


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


def build_decode_runtime_for_position(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    parameters: Any,
    decode_plan: dict[str, Any],
    batch_size: int,
    cache_len: int,
    prefill_effective_token_count: int | Sequence[int],
    generated_token_index: int,
) -> SimpleNamespace:
    if isinstance(prefill_effective_token_count, int):
        prompt_token_counts: int | list[int] = (
            int(prefill_effective_token_count) + int(generated_token_index) + 1
        )
    else:
        prompt_token_counts = [
            int(value) + int(generated_token_index) + 1
            for value in prefill_effective_token_count
        ]
    runtime_state = _build_prompt_decode_runtime_state_tensors(
        ttnn=ttnn,
        torch=torch,
        device=device,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=int(decode_plan["kv_cache"]["page_block_size"]),
        prompt_token_count=prompt_token_counts,
    )
    cache_position_value = runtime_state.decode_runtime_state[
        "cache_position_value"
    ]
    cache_position_values = runtime_state.decode_runtime_state.get(
        "cache_position_values"
    )
    if cache_position_values is None:
        cache_position_values = [int(cache_position_value)] * batch_size
    rotary_runtime = attach_decode_rotary_parameters(
        parameters=parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=decode_plan,
        cache_position_value=cache_position_value,
        cache_position_values=cache_position_values,
    )
    return SimpleNamespace(
        page_table=runtime_state.page_table,
        cache_position=runtime_state.cache_position,
        decode_runtime_state=runtime_state.decode_runtime_state,
        rotary_runtime_state=rotary_runtime.rotary_runtime_state,
        tensor_conversion_count=(
            int(runtime_state.tensor_conversion_count)
            + int(rotary_runtime.tensor_conversion_count)
        ),
        decode_runtime_state_input_tensor_count=(
            runtime_state.tensor_conversion_count
        ),
        rotary_runtime_input_tensor_count=rotary_runtime.tensor_conversion_count,
    )


def _initial_decode_runtime(
    *,
    runtime_input_mode: str,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    parameters: Any,
    decode_plan: dict[str, Any],
    batch_size: int,
    cache_len: int,
    prefill_effective_token_count: int | Sequence[int],
    token_input: Any,
) -> tuple[SimpleNamespace, DecodeInputBuffers | None, str | None]:
    if runtime_input_mode == "persistent":
        try:
            buffers = DecodeInputBuffers(
                ttnn=ttnn,
                torch=torch,
                device=device,
                dtype_seed=dtype_seed,
                parameters=parameters,
                decode_plan=decode_plan,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_effective_token_count=prefill_effective_token_count,
                token_input=token_input,
            )
            return buffers.initial_runtime_state(), buffers, None
        except PersistentDecodeInputsUnsupported as err:
            fallback_reason = str(err)
    else:
        fallback_reason = None
    runtime = build_decode_runtime_for_position(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        parameters=parameters,
        decode_plan=decode_plan,
        batch_size=batch_size,
        cache_len=cache_len,
        prefill_effective_token_count=prefill_effective_token_count,
        generated_token_index=0,
    )
    return runtime, None, fallback_reason


def _runtime_input_report(
    *,
    requested_mode: str,
    decode_step_count: int,
    persistent_inputs: DecodeInputBuffers | None,
    fallback_reason: str | None,
) -> dict[str, Any]:
    if persistent_inputs is not None:
        report = persistent_inputs.to_report()
        report["runtime_input_mode_requested"] = requested_mode
        return report
    report = {
        "execution_mode": "eager",
        "runtime_input_mode": "recreate",
        "runtime_input_mode_requested": requested_mode,
        "new_device_tensors_per_decode_step": 5,
        "host_to_device_updates_per_decode_step": 5,
        "page_table_update_count": int(decode_step_count),
        "cache_position_update_count": int(decode_step_count),
        "rotary_buffer_update_count": int(decode_step_count),
        "token_device_copy_count": 0,
        "persistent_input_count": 0,
        "initial_device_tensor_creation_count": 0,
        "host_update_count": 5 * int(decode_step_count),
        "decode_step_count": int(decode_step_count),
        "page_table_reused": False,
        "cache_position_update": "recreate_from_host",
        "rotary_update": "recreate_from_host",
        "token_update": "device_tensor_direct_handoff",
    }
    if fallback_reason is not None:
        report["fallback_reason"] = fallback_reason
    return report


def _install_context_decode_runtime(
    context: Any,
    runtime: SimpleNamespace,
    *,
    page_table_updated: bool,
    cache_position_updated: bool,
    rotary_state_updated: bool,
) -> None:
    install = context.install_decode_runtime
    try:
        install(
            runtime,
            page_table_updated=page_table_updated,
            cache_position_updated=cache_position_updated,
            rotary_state_updated=rotary_state_updated,
        )
    except TypeError as err:
        if "unexpected keyword argument" not in str(err):
            raise
        install(runtime)


def _set_context_runtime_input_report(
    context: Any,
    report: dict[str, Any],
) -> None:
    setter = getattr(context, "set_runtime_input_report", None)
    if callable(setter):
        setter(report)


def run_decode_loop(
    *,
    context: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    batch_size: int,
    cache_len: int,
    layer_count: int,
    decode_step_count: int,
    generated_token_events: list[dict[str, Any]],
    initial_tensor_conversion_count: int,
    report_level: str = "full",
    diagnostics_path: str | None = None,
    diagnostics_reference_path: str | None = None,
    runtime_input_mode: str = "recreate",
    execution_mode: str = "eager",
    trace_key: DecodeTraceKey | None = None,
    trace_keys: Sequence[DecodeTraceKey] | None = None,
    teacher_forcing_token_ids_by_step: Sequence[Sequence[int]] | None = None,
) -> SimpleNamespace:
    teacher_forcing = _normalize_teacher_forcing(
        teacher_forcing_token_ids_by_step,
        decode_step_count=decode_step_count,
        batch_size=batch_size,
    )
    if teacher_forcing and execution_mode == "trace":
        raise ValueError("teacher forcing requires eager decode execution")
    teacher_forcing_conversion_count = 0
    if teacher_forcing:
        context.update_decode_token(
            _teacher_forcing_token_tensor(
                ttnn=ttnn,
                torch=torch,
                device=device,
                token_ids=teacher_forcing[0],
                step_index=0,
            )
        )
        teacher_forcing_conversion_count += 1
    effective_token_counts = context.prefill_tokenization.get(
        "effective_token_count_by_user",
        context.prefill_tokenization["effective_token_count"],
    )
    decode_runtime, persistent_inputs, fallback_reason = (
        _initial_decode_runtime(
            runtime_input_mode=runtime_input_mode,
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            parameters=context.parameters,
            decode_plan=decode_plan,
            batch_size=batch_size,
            cache_len=cache_len,
            prefill_effective_token_count=effective_token_counts,
            token_input=context.token_ids,
        )
    )
    _install_context_decode_runtime(
        context,
        decode_runtime,
        page_table_updated=persistent_inputs is None,
        cache_position_updated=persistent_inputs is None,
        rotary_state_updated=persistent_inputs is None,
    )
    decode_runtime_state = context.decode_runtime_state
    rotary_runtime_state = context.rotary_state
    tensor_conversion_count = (
        int(initial_tensor_conversion_count)
        + decode_runtime.tensor_conversion_count
        + teacher_forcing_conversion_count
    )
    decode_runtime_state_count = (
        decode_runtime.decode_runtime_state_input_tensor_count
    )
    decode_rotary_runtime_count = (
        decode_runtime.rotary_runtime_input_tensor_count
    )
    step_reports = []
    diagnostic_reference_ids: set[str] = set()
    observed_op_cursor = _observed_op_cursor(context.generated_model, ttnn)
    trace_session = None
    sdpa_context_runtime = SDPAContextRuntime(context.generated_model)
    if execution_mode == "trace" and decode_step_count > 0:
        if persistent_inputs is None:
            raise ValueError(
                "trace execution requires persistent decode inputs"
            )
        if trace_key is None and not trace_keys:
            raise ValueError("trace execution requires a DecodeTraceKey")
        trace_session = _create_decode_trace_session(
            ttnn=ttnn,
            device=device,
            model=context.generated_model,
            persistent_inputs=persistent_inputs,
            kv_cache=context.kv_cache,
            trace_key=trace_key,
            trace_keys=trace_keys,
        )
        trace_session.capture()
        observed_op_cursor = _observed_op_cursor(
            context.generated_model,
            ttnn,
        )
    try:
        for step_index in range(decode_step_count):
            step_decode_runtime_state = decode_runtime_state
            step_rotary_runtime_state = rotary_runtime_state
            input_shapes = _loop_input_shapes(
                token_ids=context.token_ids,
                page_table=context.page_table,
                cache_position=context.cache_position,
                kv_cache=context.kv_cache,
            )
            if trace_session is not None:
                execution = trace_session.execute()
                token = execution.token
                kv_cache = execution.kv_cache
                latency_ms = execution.latency_ms
            else:
                if sdpa_context_runtime.enabled:
                    sdpa_context_runtime.activate_for_context(
                        _runtime_active_context_len(step_decode_runtime_state)
                    )
                token, kv_cache, latency_ms = _time_decode_step(
                    ttnn=ttnn,
                    model=context.generated_model,
                    device=device,
                    token_ids=context.token_ids,
                    page_table=context.page_table,
                    cache_position=context.cache_position,
                    kv_cache=context.kv_cache,
                )
            context.update_kv_cache(kv_cache)
            output_shapes = _loop_output_shapes(
                token=token,
                kv_cache=context.kv_cache,
                layer_count=layer_count,
            )
            output = {
                "kind": "token",
                "shape": tensor_shape(token),
                "dtype": tensor_dtype(token),
                "repr": repr(token),
            }
            token_event = {
                "step_index": step_index,
                "token": token,
                "runtime_handoff": "device_tensor_direct",
                "runtime_host_roundtrip": False,
                "cache_position_value": step_decode_runtime_state.get(
                    "cache_position_value"
                ),
                "page_table_shape": input_shapes.get("page_table"),
                "token_shape": tensor_shape(token),
            }
            if teacher_forcing:
                token_event["teacher_forced_input_token_ids"] = list(
                    teacher_forcing[step_index]
                )
            position_values = step_decode_runtime_state.get(
                "cache_position_values"
            )
            if position_values is not None and len(set(position_values)) > 1:
                token_event["cache_position_values"] = position_values
            if trace_session is not None:
                materialization_start = time.perf_counter()
                materialization = _loop_generated_token_ids(
                    token=token,
                    ttnn=ttnn,
                    batch_size=batch_size,
                )
                token_event["materialized_token_ids_by_user"] = materialization[
                    "token_ids_by_user"
                ]
                token_event["materialization"] = materialization
                token_event["materialization_ms"] = (
                    time.perf_counter() - materialization_start
                ) * 1000.0
            generated_token_events.append(token_event)
            if trace_session is not None:
                observed_ops = trace_session.captured_model_ops
                if not observed_ops:
                    observed_ops = list(decode_plan["op_sequence"])
                    observed_ops_source = "generated_execution_plan"
                else:
                    observed_ops_source = "runtime_instrumentation"
            else:
                observed_ops, observed_op_cursor = _observed_ops_since(
                    context.generated_model,
                    ttnn,
                    observed_op_cursor,
                )
                observed_ops_source = "runtime_instrumentation"
                if observed_ops is None:
                    observed_ops = list(decode_plan["op_sequence"])
                    observed_ops_source = "generated_execution_plan"
            reference = _decode_step_reference(
                plan=decode_plan,
                layer_count=layer_count,
                output_shapes=output_shapes,
                output=output,
                observed_ops=observed_ops,
            )
            reference["observed_ops_source"] = observed_ops_source
            full_step_report = {
                "step_index": step_index,
                "status": (
                    "passed" if reference["passed"] else "reference_mismatch"
                ),
                "passed": bool(reference["passed"]),
                "latency_ms": latency_ms,
                "cache_position_value": step_decode_runtime_state.get(
                    "cache_position_value"
                ),
                "input_shapes": input_shapes,
                "decode_runtime_state": step_decode_runtime_state,
                "rotary_runtime_state": step_rotary_runtime_state,
                "output_shapes": output_shapes,
                "output": output,
                "generated_token_ids": token_event.get(
                    "materialized_token_ids_by_user", []
                ),
                "token_materialization": token_event.get(
                    "materialization",
                    {
                        "status": "deferred",
                        "source": "reporting_after_decode_loop",
                    },
                ),
                "token_materialization_ms": token_event.get(
                    "materialization_ms"
                ),
                "token_runtime_handoff": "device_tensor_direct",
                "runtime_host_roundtrip": False,
                "reference": reference,
            }
            if report_level == "full" and diagnostics_path is not None:
                diagnostic_step = dict(full_step_report)
                if diagnostics_reference_path is not None:
                    reference_id = _reference_id(reference)
                    if reference_id not in diagnostic_reference_ids:
                        append_json_line(
                            diagnostics_reference_path,
                            {
                                "reference_id": reference_id,
                                "reference": reference,
                            },
                        )
                        diagnostic_reference_ids.add(reference_id)
                    diagnostic_step["reference"] = {
                        "reference_id": reference_id,
                        "kind": reference.get("kind"),
                        "status": reference.get("status"),
                        "passed": reference.get("passed"),
                    }
                append_json_line(diagnostics_path, diagnostic_step)
            step_reports.append(_compact_step_report(full_step_report))
            next_token = token
            if teacher_forcing and step_index + 1 < decode_step_count:
                next_token = _teacher_forcing_token_tensor(
                    ttnn=ttnn,
                    torch=torch,
                    device=device,
                    token_ids=teacher_forcing[step_index + 1],
                    step_index=step_index + 1,
                )
                teacher_forcing_conversion_count += 1
                tensor_conversion_count += 1
            context.update_decode_token(next_token)
            if trace_session is not None:
                decode_runtime = execution.runtime_state
            elif persistent_inputs is not None:
                persistent_inputs.record_token(next_token)
            if trace_session is not None:
                _install_context_decode_runtime(
                    context,
                    decode_runtime,
                    page_table_updated=False,
                    cache_position_updated=True,
                    rotary_state_updated=True,
                )
                decode_runtime_state = context.decode_runtime_state
                rotary_runtime_state = context.rotary_state
            elif step_index + 1 < decode_step_count:
                if persistent_inputs is not None:
                    decode_runtime = persistent_inputs.advance()
                else:
                    decode_runtime = build_decode_runtime_for_position(
                        ttnn=ttnn,
                        torch=torch,
                        device=device,
                        dtype_seed=dtype_seed,
                        parameters=context.parameters,
                        decode_plan=decode_plan,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        prefill_effective_token_count=effective_token_counts,
                        generated_token_index=step_index + 1,
                    )
                _install_context_decode_runtime(
                    context,
                    decode_runtime,
                    page_table_updated=persistent_inputs is None,
                    cache_position_updated=True,
                    rotary_state_updated=True,
                )
                decode_runtime_state = context.decode_runtime_state
                rotary_runtime_state = context.rotary_state
                decode_runtime_state_count += (
                    decode_runtime.decode_runtime_state_input_tensor_count
                )
                decode_rotary_runtime_count += (
                    decode_runtime.rotary_runtime_input_tensor_count
                )
                tensor_conversion_count += (
                    decode_runtime.tensor_conversion_count
                )
    finally:
        if trace_session is not None:
            trace_session.close()

    runtime_input_report = _runtime_input_report(
        requested_mode=runtime_input_mode,
        decode_step_count=decode_step_count,
        persistent_inputs=persistent_inputs,
        fallback_reason=fallback_reason,
    )
    runtime_input_report["teacher_forcing"] = {
        "enabled": bool(teacher_forcing),
        "step_count": len(teacher_forcing),
        "device_tensor_creation_count": teacher_forcing_conversion_count,
        "execution_mode": "eager" if teacher_forcing else None,
    }
    if trace_session is not None:
        runtime_input_report.update(trace_session.to_report())
        runtime_input_report["runtime_input_mode_requested"] = (
            runtime_input_mode
        )
    elif sdpa_context_runtime.enabled:
        runtime_input_report["sdpa_context_buckets"] = (
            sdpa_context_runtime.to_report()
        )
    _set_context_runtime_input_report(context, runtime_input_report)
    return SimpleNamespace(
        generated_token_events=generated_token_events,
        step_reports=step_reports,
        decode_runtime_state=decode_runtime_state,
        rotary_runtime_state=rotary_runtime_state,
        tensor_conversion_count=tensor_conversion_count,
        decode_runtime_state_input_tensor_count=decode_runtime_state_count,
        decode_rotary_runtime_input_tensor_count=decode_rotary_runtime_count,
        diagnostics_step_count=(
            decode_step_count
            if report_level == "full" and diagnostics_path is not None
            else 0
        ),
        diagnostics_reference_count=len(diagnostic_reference_ids),
        runtime_input_report=runtime_input_report,
    )


def _normalize_teacher_forcing(
    token_ids_by_step: Sequence[Sequence[int]] | None,
    *,
    decode_step_count: int,
    batch_size: int,
) -> list[list[int]]:
    if token_ids_by_step is None:
        return []
    steps = [[int(token_id) for token_id in row] for row in token_ids_by_step]
    if len(steps) != int(decode_step_count):
        raise ValueError(
            "teacher forcing step count must match decode_step_count: "
            f"{len(steps)} != {decode_step_count}"
        )
    for step_index, row in enumerate(steps):
        if len(row) != int(batch_size):
            raise ValueError(
                "teacher forcing batch width must match batch_size at step "
                f"{step_index}: {len(row)} != {batch_size}"
            )
        if any(token_id < 0 for token_id in row):
            raise ValueError("teacher forcing token IDs must be non-negative")
    return steps


def _teacher_forcing_token_tensor(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    token_ids: Sequence[int],
    step_index: int,
) -> Any:
    return build_token_ids_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        token_ids=[[int(token_id)] for token_id in token_ids],
        name=f"teacher_forcing_token_ids_{step_index}",
    )


def _compact_step_report(step: dict[str, Any]) -> dict[str, Any]:
    reference = step["reference"]
    compact_reference: dict[str, Any]
    if bool(reference.get("passed")):
        compact_reference = {
            "kind": reference.get("kind"),
            "status": reference.get("status"),
            "passed": True,
            "failed_checks": [],
        }
    else:
        compact_reference = reference
    return {
        "step_index": step["step_index"],
        "status": step["status"],
        "passed": step["passed"],
        "latency_ms": step["latency_ms"],
        "cache_position_value": step["cache_position_value"],
        "input_shapes": step["input_shapes"],
        "output_shapes": step["output_shapes"],
        "output": step["output"],
        "generated_token_ids": step["generated_token_ids"],
        "token_materialization": step["token_materialization"],
        "token_materialization_ms": step["token_materialization_ms"],
        "token_runtime_handoff": step["token_runtime_handoff"],
        "runtime_host_roundtrip": step["runtime_host_roundtrip"],
        "reference": compact_reference,
    }


def _observed_op_cursor(model: Any, ttnn: Any) -> int:
    op_log = getattr(getattr(model, "ops", None), "op_log", None)
    if isinstance(op_log, list):
        op_log.clear()
        return 0
    observed = _generated_observed_op_sequence(model, ttnn)
    return len(observed) if isinstance(observed, list) else 0


def _observed_ops_since(
    model: Any,
    ttnn: Any,
    cursor: int,
) -> tuple[list[str] | None, int]:
    op_log = getattr(getattr(model, "ops", None), "op_log", None)
    if isinstance(op_log, list):
        start = cursor if 0 <= cursor <= len(op_log) else 0
        observed = [str(item) for item in op_log[start:]]
        op_log.clear()
        return observed, 0
    observed = _generated_observed_op_sequence(model, ttnn)
    if not isinstance(observed, list):
        return None, cursor
    start = cursor if 0 <= cursor <= len(observed) else 0
    return observed[start:], len(observed)


def _reference_id(reference: dict[str, Any]) -> str:
    encoded = json.dumps(
        reference,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"decode-{hashlib.sha256(encoded).hexdigest()[:16]}"


def run_decode_steady_iterations(
    *,
    context: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    batch_size: int,
    cache_len: int,
    warmup: int,
    iterations: int,
    runtime_input_mode: str = "recreate",
    execution_mode: str = "eager",
    trace_key: DecodeTraceKey | None = None,
    trace_keys: Sequence[DecodeTraceKey] | None = None,
) -> SimpleNamespace:
    """Run post-prefill decode without per-op profiling or host token copies."""

    total_steps = int(warmup) + int(iterations)
    effective_token_counts = [
        int(value)
        for value in context.prefill_tokenization.get(
            "effective_token_count_by_user",
            [context.prefill_tokenization["effective_token_count"]]
            * batch_size,
        )
    ]
    effective_token_count = max(effective_token_counts)
    if effective_token_count + total_steps > int(cache_len):
        raise ValueError(
            "decode-steady steps exceed cache capacity: "
            f"effective_prompt_tokens={effective_token_count}, "
            f"warmup={warmup}, iterations={iterations}, cache_len={cache_len}"
        )

    decode_runtime, persistent_inputs, fallback_reason = (
        _initial_decode_runtime(
            runtime_input_mode=runtime_input_mode,
            ttnn=ttnn,
            torch=torch,
            device=device,
            dtype_seed=dtype_seed,
            parameters=context.parameters,
            decode_plan=decode_plan,
            batch_size=batch_size,
            cache_len=cache_len,
            prefill_effective_token_count=effective_token_counts,
            token_input=context.token_ids,
        )
    )
    _install_context_decode_runtime(
        context,
        decode_runtime,
        page_table_updated=persistent_inputs is None,
        cache_position_updated=persistent_inputs is None,
        rotary_state_updated=persistent_inputs is None,
    )
    warmup_samples: list[float] = []
    measured_samples: list[float] = []
    cache_positions: list[int] = []
    tensor_conversion_count = int(decode_runtime.tensor_conversion_count)
    decode_runtime_state_count = int(
        decode_runtime.decode_runtime_state_input_tensor_count
    )
    decode_rotary_runtime_count = int(
        decode_runtime.rotary_runtime_input_tensor_count
    )

    trace_session = None
    sdpa_context_runtime = SDPAContextRuntime(context.generated_model)
    if execution_mode == "trace":
        if persistent_inputs is None:
            raise ValueError(
                "trace execution requires persistent decode inputs"
            )
        if trace_key is None and not trace_keys:
            raise ValueError("trace execution requires a DecodeTraceKey")
        trace_session = _create_decode_trace_session(
            ttnn=ttnn,
            device=device,
            model=context.generated_model,
            persistent_inputs=persistent_inputs,
            kv_cache=context.kv_cache,
            trace_key=trace_key,
            trace_keys=trace_keys,
        )
        trace_session.capture()

    try:
        for step_index in range(total_steps):
            if trace_session is not None:
                execution = trace_session.execute()
                token = execution.token
                kv_cache = execution.kv_cache
                latency_ms = execution.latency_ms
                decode_runtime = execution.runtime_state
                _install_context_decode_runtime(
                    context,
                    decode_runtime,
                    page_table_updated=False,
                    cache_position_updated=True,
                    rotary_state_updated=True,
                )
                position_values = execution.cache_positions
            else:
                step_start = time.perf_counter()
                if step_index > 0:
                    if persistent_inputs is not None:
                        decode_runtime = persistent_inputs.advance()
                    else:
                        decode_runtime = build_decode_runtime_for_position(
                            ttnn=ttnn,
                            torch=torch,
                            device=device,
                            dtype_seed=dtype_seed,
                            parameters=context.parameters,
                            decode_plan=decode_plan,
                            batch_size=batch_size,
                            cache_len=cache_len,
                            prefill_effective_token_count=(
                                effective_token_counts
                            ),
                            generated_token_index=step_index,
                        )
                    _install_context_decode_runtime(
                        context,
                        decode_runtime,
                        page_table_updated=persistent_inputs is None,
                        cache_position_updated=True,
                        rotary_state_updated=True,
                    )
                    tensor_conversion_count += int(
                        decode_runtime.tensor_conversion_count
                    )
                    decode_runtime_state_count += int(
                        decode_runtime.decode_runtime_state_input_tensor_count
                    )
                    decode_rotary_runtime_count += int(
                        decode_runtime.rotary_runtime_input_tensor_count
                    )
                if sdpa_context_runtime.enabled:
                    sdpa_context_runtime.activate_for_context(
                        _runtime_active_context_len(
                            decode_runtime.decode_runtime_state
                        )
                    )
                token, kv_cache = context.generated_model.decode_step(
                    context.token_ids,
                    context.page_table,
                    context.cache_position,
                    context.kv_cache,
                )
                synchronize = getattr(ttnn, "synchronize_device", None)
                if callable(synchronize):
                    synchronize(device)
                latency_ms = (time.perf_counter() - step_start) * 1000.0
                position_values = decode_runtime.decode_runtime_state[
                    "cache_position_values"
                ]

            context.update_kv_cache(kv_cache)
            context.update_decode_token(token)
            if persistent_inputs is not None and trace_session is None:
                persistent_inputs.record_token(token)
            cache_positions.append(max(int(value) for value in position_values))
            if step_index < warmup:
                warmup_samples.append(latency_ms)
            else:
                measured_samples.append(latency_ms)
    finally:
        if trace_session is not None:
            trace_session.close()

    runtime_input_report = _runtime_input_report(
        requested_mode=runtime_input_mode,
        decode_step_count=total_steps,
        persistent_inputs=persistent_inputs,
        fallback_reason=fallback_reason,
    )
    if trace_session is not None:
        runtime_input_report.update(trace_session.to_report())
        runtime_input_report["runtime_input_mode_requested"] = (
            runtime_input_mode
        )
    elif sdpa_context_runtime.enabled:
        runtime_input_report["sdpa_context_buckets"] = (
            sdpa_context_runtime.to_report()
        )
    _set_context_runtime_input_report(context, runtime_input_report)
    return SimpleNamespace(
        warmup_step_ms_samples=warmup_samples,
        measured_step_ms_samples=measured_samples,
        cache_positions=cache_positions,
        tensor_conversion_count=tensor_conversion_count,
        decode_runtime_state_input_tensor_count=decode_runtime_state_count,
        decode_rotary_runtime_input_tensor_count=decode_rotary_runtime_count,
        final_token=context.token_ids,
        final_kv_cache=context.kv_cache,
        runtime_input_report=runtime_input_report,
    )


def _create_decode_trace_session(
    *,
    ttnn: Any,
    device: Any,
    model: Any,
    persistent_inputs: Any,
    kv_cache: Any,
    trace_key: DecodeTraceKey | None,
    trace_keys: Sequence[DecodeTraceKey] | None,
) -> DecodeTraceSession | BucketedDecodeTraceSession:
    if trace_keys:
        return BucketedDecodeTraceSession(
            ttnn=ttnn,
            device=device,
            model=model,
            persistent_inputs=persistent_inputs,
            kv_cache=kv_cache,
            keys=trace_keys,
        )
    if trace_key is None:
        raise ValueError("trace execution requires a DecodeTraceKey")
    return DecodeTraceSession(
        ttnn=ttnn,
        device=device,
        model=model,
        persistent_inputs=persistent_inputs,
        kv_cache=kv_cache,
        key=trace_key,
    )


def _runtime_active_context_len(runtime_state: dict[str, Any]) -> int:
    values = runtime_state.get("cache_position_values")
    if values is None:
        value = runtime_state.get("cache_position_value")
        if value is None:
            raise ValueError("decode runtime state has no cache position")
        values = [value]
    return max(int(value) for value in values) + 1


def materialize_generate_token_events(
    token_events: list[dict[str, Any]],
    *,
    step_reports: list[dict[str, Any]],
    ttnn: Any,
    batch_size: int,
) -> SimpleNamespace:
    generated_token_ids_by_user = [[] for _ in range(batch_size)]
    per_step_token_metadata: list[dict[str, Any]] = []
    first_token_materialization_ms = 0.0
    first_token_status = "not_run"
    first_token_source = "none"
    first_token_ids_by_user: list[list[int]] = []
    step_reports_by_index = {
        int(report["step_index"]): report
        for report in step_reports
        if isinstance(report.get("step_index"), int)
    }
    for event in token_events:
        pre_materialized = event.get("materialized_token_ids_by_user")
        if pre_materialized is not None:
            token_ids = pre_materialized
            materialization = event["materialization"]
            materialization_ms = float(event["materialization_ms"])
            materialization_phase = "trace_step_reporting"
        else:
            materialization_start = time.perf_counter()
            materialization = _loop_generated_token_ids(
                token=event["token"],
                ttnn=ttnn,
                batch_size=batch_size,
            )
            materialization_ms = (
                time.perf_counter() - materialization_start
            ) * 1000.0
            token_ids = materialization["token_ids_by_user"]
            materialization_phase = "reporting_after_decode_loop"
        for user_index, row in enumerate(token_ids):
            generated_token_ids_by_user[user_index].extend(row)

        metadata = {
            "step_index": event["step_index"],
            "token_ids_by_user": token_ids,
            "token_materialization_status": materialization["status"],
            "token_materialization_source": materialization["source"],
            "token_materialization_ms": materialization_ms,
            "token_materialization_phase": materialization_phase,
            "runtime_handoff": event.get("runtime_handoff"),
            "runtime_host_roundtrip": bool(event.get("runtime_host_roundtrip")),
            "cache_position_value": event.get("cache_position_value"),
            "page_table_shape": event.get("page_table_shape"),
            "token_shape": event.get("token_shape"),
        }
        per_step_token_metadata.append(metadata)
        if event["step_index"] == "prefill":
            first_token_materialization_ms = materialization_ms
            first_token_status = materialization["status"]
            first_token_source = materialization["source"]
            first_token_ids_by_user = token_ids
            continue
        report = step_reports_by_index.get(int(event["step_index"]))
        if report is None:
            continue
        report["generated_token_ids"] = token_ids
        report["token_materialization"] = materialization
        report["token_materialization_ms"] = materialization_ms
        report["token_materialization_phase"] = materialization_phase

    return SimpleNamespace(
        generated_token_ids_by_user=generated_token_ids_by_user,
        per_step_token_metadata=per_step_token_metadata,
        first_token_materialization_ms=first_token_materialization_ms,
        first_token_status=first_token_status,
        first_token_source=first_token_source,
        first_token_ids_by_user=first_token_ids_by_user,
    )
