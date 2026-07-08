from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from .codegen.parameters import ParameterMaterializationError
from .codegen.ttnn_tensorizer import TTNNTensorizationError
from .runtime_environment import collect_ttnn_environment
from .runtime_inputs import PromptTokenizationError
from .runtime_inputs import detokenize_generated_token_ids
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .smoke_single_layer_decode import (
    _attach_runtime_rotary_parameters,
    _base_report,
    _build_model_decode_state,
    _build_prompt_decode_runtime_state_tensors,
    _decode_step_plan,
    _decode_step_reference,
    _dtype,
    _failed_report,
    _generated_observed_op_sequence,
    _load_generated_model,
    _maybe_managed_device,
    _no_device_report,
    _shape,
    _time_decode_step,
    _to_namespace,
    _trace_report,
    _write_report,
)
from .smoke_decode_shell import _dry_run_reference
from .templates.ttnn_ops import UnsupportedTTNNOp


def run_prompt_decode_loop(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    decode_steps: int = 2,
    layers: int = 1,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers)
    step_count = int(decode_steps)
    num_layers = int(config["num_layers"])
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if layer_count > num_layers:
        raise ValueError(
            f"layers must be <= generated config num_layers ({num_layers})"
        )
    if step_count <= 0:
        raise ValueError("decode_steps must be positive")
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    plan = _decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
    )

    if dry_run:
        report = _loop_base_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "runtime_status": "dry_run",
                "decode_loop_runtime_owned": False,
                "planned_decode_loop_runtime_owned": True,
                "input_source": "planned_prompt_decode_loop",
                "runtime_owner": "prompt_decode_loop",
                "latency_ms": 0.0,
                "step_reports": [],
                "per_step_token_metadata": [],
                "generated_token_ids": [],
                "generated_token_id_source": "dry_run",
                "token_materialization_status": "not_run",
                "generated_text": "",
                "generated_text_by_user": [],
                "generated_text_status": "not_run",
                "generated_text_source": "dry_run",
                "generated_text_report": {
                    "status": "not_run",
                    "source": "dry_run",
                    "generated_text_by_user": [],
                    "generated_text": "",
                },
                "output_shapes": None,
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "synthetic_runtime_input_tensor_count": 0,
                "synthetic_rotary_tensor_count": 0,
                "prompt_runtime_input_tensor_count": 1,
                "decode_runtime_state_input_tensor_count": 2 * step_count,
                "rotary_runtime_input_tensor_count": 3 * layer_count * step_count,
                "kv_cache_runtime_input_tensor_count": 2 * layer_count,
                "throughput_summary": _loop_throughput_summary(
                    latency_ms=0.0,
                    batch_size=batch_size,
                    decode_steps=step_count,
                    dry_run=True,
                ),
                "trace": _trace_report(requested=False, status="disabled"),
                "reference": _dry_run_reference("prompt_decode_loop"),
                "error": None,
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_report(out, report)
        return report

    if model_path is None:
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="missing_model_path",
            message="model_path is required for prompt decode loop execution",
            detail="model_path was not provided",
        )
        _write_report(out, report)
        return report
    if prompt is None:
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="missing_prompt",
            message="prompt is required for prompt decode loop execution",
            detail="prompt was not provided",
        )
        _write_report(out, report)
        return report
    if plan["output_kind"] != "token":
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="unsupported_output_kind",
            message="prompt decode loop requires token output",
            detail=f"output_kind={plan['output_kind']}",
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
        report = _loop_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
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
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="missing_torch",
            message="torch is required to build prompt decode loop tensors",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            state = _build_model_decode_state(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                plan=plan,
                program_dir=program_root,
                model_path=Path(model_path),
                prompt=prompt,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            generated = _load_generated_model(program_root / "model.py", ttnn)
            decode_config = dict(config)
            decode_config["num_layers"] = layer_count
            model = generated.BuddyLlama31TTNN(
                device=ttnn_device,
                parameters=state.parameters,
                config=_to_namespace(decode_config),
            )

            prompt_token_count = int(
                state.prompt_tokenization["token_count"]
            )
            token_ids = state.token_ids
            page_table = state.page_table
            cache_position = state.cache_position
            kv_cache = state.kv_cache
            decode_runtime_state = state.decode_runtime_state
            rotary_runtime_state = state.rotary_runtime_state
            tensor_conversion_count = int(state.tensor_conversion_count)
            decode_runtime_state_count = int(
                state.parameter_setup[
                    "decode_runtime_state_input_tensor_count"
                ]
            )
            rotary_runtime_count = int(
                state.parameter_setup["rotary_runtime_input_tensor_count"]
            )
            kv_cache_runtime_count = int(
                state.parameter_setup["kv_cache_runtime_input_tensor_count"]
            )
            step_reports = []
            generated_token_ids_by_user = [[] for _ in range(batch_size)]
            per_step_token_metadata = []
            total_start = time.perf_counter()

            for step_index in range(step_count):
                input_shapes = _loop_input_shapes(
                    token_ids=token_ids,
                    page_table=page_table,
                    cache_position=cache_position,
                    kv_cache=kv_cache,
                )
                token, kv_cache, latency_ms = _time_decode_step(
                    ttnn=ttnn,
                    model=model,
                    device=ttnn_device,
                    token_ids=token_ids,
                    page_table=page_table,
                    cache_position=cache_position,
                    kv_cache=kv_cache,
                )
                output_shapes = _loop_output_shapes(
                    token=token,
                    kv_cache=kv_cache,
                    layer_count=layer_count,
                )
                output = {
                    "kind": "token",
                    "shape": _shape(token),
                    "dtype": _dtype(token),
                    "repr": repr(token),
                }
                token_materialization = _loop_generated_token_ids(
                    token=token,
                    ttnn=ttnn,
                    batch_size=batch_size,
                )
                step_token_ids = token_materialization["token_ids_by_user"]
                for user_index, row in enumerate(step_token_ids):
                    if user_index < len(generated_token_ids_by_user):
                        generated_token_ids_by_user[user_index].extend(row)
                token_metadata = {
                    "step_index": step_index,
                    "token_ids_by_user": step_token_ids,
                    "token_materialization_status": (
                        token_materialization["status"]
                    ),
                    "token_materialization_source": (
                        token_materialization["source"]
                    ),
                    "cache_position_value": (
                        None
                        if decode_runtime_state is None
                        else decode_runtime_state.get(
                            "cache_position_value"
                        )
                    ),
                    "page_table_shape": input_shapes.get("page_table"),
                    "token_shape": _shape(token),
                }
                per_step_token_metadata.append(token_metadata)
                reference = _decode_step_reference(
                    plan=plan,
                    layer_count=layer_count,
                    output_shapes=output_shapes,
                    output=output,
                    observed_ops=_generated_observed_op_sequence(model, ttnn),
                )
                step_reports.append(
                    {
                        "step_index": step_index,
                        "status": (
                            "passed"
                            if reference["passed"]
                            else "reference_mismatch"
                        ),
                        "passed": bool(reference["passed"]),
                        "latency_ms": latency_ms,
                        "cache_position_value": (
                            None
                            if decode_runtime_state is None
                            else decode_runtime_state.get(
                                "cache_position_value"
                            )
                        ),
                        "input_shapes": input_shapes,
                        "decode_runtime_state": decode_runtime_state,
                        "rotary_runtime_state": rotary_runtime_state,
                        "output_shapes": output_shapes,
                        "output": output,
                        "generated_token_ids": step_token_ids,
                        "token_materialization": token_materialization,
                        "reference": reference,
                    }
                )

                token_ids = token
                if step_index + 1 < step_count:
                    runtime_state = _build_prompt_decode_runtime_state_tensors(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        page_block_size=int(plan["kv_cache"]["page_block_size"]),
                        prompt_token_count=(
                            prompt_token_count + step_index + 1
                        ),
                    )
                    page_table = runtime_state.page_table
                    cache_position = runtime_state.cache_position
                    decode_runtime_state = runtime_state.decode_runtime_state
                    decode_runtime_state_count += int(
                        runtime_state.tensor_conversion_count
                    )
                    tensor_conversion_count += int(
                        runtime_state.tensor_conversion_count
                    )
                    rotary_runtime = _attach_runtime_rotary_parameters(
                        parameters=state.parameters,
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        plan=plan,
                        cache_position_value=int(
                            decode_runtime_state["cache_position_value"]
                        ),
                    )
                    rotary_runtime_state = (
                        rotary_runtime.rotary_runtime_state
                    )
                    rotary_runtime_count += int(
                        rotary_runtime.tensor_conversion_count
                    )
                    tensor_conversion_count += int(
                        rotary_runtime.tensor_conversion_count
                    )

            latency_ms = (time.perf_counter() - total_start) * 1000.0
            passed = all(step["passed"] for step in step_reports)
            text_report = detokenize_generated_token_ids(
                token_ids_by_user=generated_token_ids_by_user,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            parameter_setup = dict(state.parameter_setup)
            parameter_setup.update(
                {
                    "decode_loop_runtime_owned": passed,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "decode_runtime_state_input_tensor_count": (
                        decode_runtime_state_count
                    ),
                    "rotary_runtime_input_tensor_count": rotary_runtime_count,
                    "kv_cache_runtime_input_tensor_count": (
                        kv_cache_runtime_count
                    ),
                    "decode_loop_step_count": step_count,
                }
            )
            report = _loop_base_report(
                program_dir=program_root,
                layers=layer_count,
                decode_steps=step_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                dry_run=False,
                plan=plan,
            )
            report.update(
                {
                    "passed": passed,
                    "status": "passed" if passed else "reference_mismatch",
                    "runtime_status": (
                        "passed" if passed else "reference_mismatch"
                    ),
                    "decode_loop_runtime_owned": passed,
                    "input_source": "prompt_decode_loop",
                    "runtime_owner": "prompt_decode_loop",
                    "parameter_source": state.parameter_source,
                    "parameter_setup": parameter_setup,
                    "prompt_tokenization": state.prompt_tokenization,
                    "decode_runtime_state": decode_runtime_state,
                    "rotary_runtime_state": rotary_runtime_state,
                    "kv_cache_runtime_state": state.kv_cache_runtime_state,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "prompt_runtime_input_tensor_count": (
                        state.parameter_setup[
                            "prompt_runtime_input_tensor_count"
                        ]
                    ),
                    "decode_runtime_state_input_tensor_count": (
                        decode_runtime_state_count
                    ),
                    "rotary_runtime_input_tensor_count": rotary_runtime_count,
                    "kv_cache_runtime_input_tensor_count": (
                        kv_cache_runtime_count
                    ),
                    "tensor_conversion_count": tensor_conversion_count,
                    "latency_ms": latency_ms,
                    "step_reports": step_reports,
                    "per_step_token_metadata": per_step_token_metadata,
                    "generated_token_ids": generated_token_ids_by_user,
                    "generated_token_id_source": _generated_token_id_source(
                        per_step_token_metadata
                    ),
                    "token_materialization_status": (
                        _generated_token_materialization_status(
                            per_step_token_metadata
                        )
                    ),
                    "generated_text": text_report["generated_text"],
                    "generated_text_by_user": (
                        text_report["generated_text_by_user"]
                    ),
                    "generated_text_status": text_report["status"],
                    "generated_text_source": text_report["source"],
                    "generated_text_report": text_report,
                    "output_shapes": (
                        step_reports[-1]["output_shapes"]
                        if step_reports
                        else None
                    ),
                    "output": step_reports[-1]["output"] if step_reports else None,
                    "throughput_summary": _loop_throughput_summary(
                        latency_ms=latency_ms,
                        batch_size=batch_size,
                        decode_steps=step_count,
                    ),
                    "trace": _trace_report(requested=False, status="disabled"),
                    "reference": _loop_reference_summary(step_reports),
                    "error": None
                    if passed
                    else "prompt decode loop structural reference mismatch",
                    "ttnn_version": getattr(ttnn, "__version__", None),
                    "ttnn_environment": collect_ttnn_environment(ttnn),
                }
            )
    except NoTTNNDeviceError as err:
        report = _loop_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            detail=str(err),
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
    ) as err:
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="parameter_setup_error",
            message=str(err),
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
    except UnsupportedTTNNOp as err:
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
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
        report = _loop_failed_report(
            program_dir=program_root,
            layers=layer_count,
            decode_steps=step_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
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


def _loop_base_report(
    *,
    program_dir: Path,
    layers: int,
    decode_steps: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
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
        trace=False,
        trace_iterations=1,
        dry_run=dry_run,
        plan=plan,
    )
    report.update(
        {
            "command": "prompt-decode-loop",
            "template": "prompt_decode_loop",
            "decode_steps": decode_steps,
            "max_new_tokens": decode_steps,
            "prefill_status": "not_run",
            "kv_cache_source": "empty_initialized",
            "model_semantics": "decode_only_empty_or_uninitialized_kv",
            "semantic_disclaimer": (
                "This decode-only loop does not run prefill and therefore "
                "does not prove prompt-conditioned full LLM inference."
            ),
            "trace_enabled": False,
            "trace_iterations": 0,
        }
    )
    return report


def _loop_no_device_report(**kwargs: Any) -> dict[str, Any]:
    decode_steps = kwargs.pop("decode_steps")
    report = _no_device_report(
        trace=False,
        trace_iterations=1,
        **kwargs,
    )
    report.update(
        {
            "command": "prompt-decode-loop",
            "template": "prompt_decode_loop",
            "decode_steps": decode_steps,
            "max_new_tokens": decode_steps,
            "prefill_status": "not_run",
            "kv_cache_source": "empty_initialized",
            "model_semantics": "decode_only_empty_or_uninitialized_kv",
            "semantic_disclaimer": (
                "This decode-only loop does not run prefill and therefore "
                "does not prove prompt-conditioned full LLM inference."
            ),
            "decode_loop_runtime_owned": False,
            "runtime_owner": "prompt_decode_loop",
            "per_step_token_metadata": [],
            "generated_token_ids": [],
            "generated_token_id_source": "no_device",
            "token_materialization_status": "not_run",
            "generated_text": "",
            "generated_text_by_user": [],
            "generated_text_status": "not_run",
            "generated_text_source": "no_device",
            "generated_text_report": {
                "status": "not_run",
                "source": "no_device",
                "generated_text_by_user": [],
                "generated_text": "",
            },
            "trace_enabled": False,
            "trace_iterations": 0,
            "trace": _trace_report(requested=False, status="disabled"),
        }
    )
    return report


def _loop_failed_report(
    *,
    decode_steps: int,
    **kwargs: Any,
) -> dict[str, Any]:
    report = _failed_report(
        trace=False,
        trace_iterations=1,
        **kwargs,
    )
    report.update(
        {
            "command": "prompt-decode-loop",
            "template": "prompt_decode_loop",
            "decode_steps": decode_steps,
            "max_new_tokens": decode_steps,
            "prefill_status": "not_run",
            "kv_cache_source": "empty_initialized",
            "model_semantics": "decode_only_empty_or_uninitialized_kv",
            "semantic_disclaimer": (
                "This decode-only loop does not run prefill and therefore "
                "does not prove prompt-conditioned full LLM inference."
            ),
            "decode_loop_runtime_owned": False,
            "runtime_owner": "prompt_decode_loop",
            "per_step_token_metadata": [],
            "generated_token_ids": [],
            "generated_token_id_source": "error",
            "token_materialization_status": "not_run",
            "generated_text": "",
            "generated_text_by_user": [],
            "generated_text_status": "not_run",
            "generated_text_source": "error",
            "generated_text_report": {
                "status": "not_run",
                "source": "error",
                "generated_text_by_user": [],
                "generated_text": "",
            },
            "trace_enabled": False,
            "trace_iterations": 0,
            "trace": _trace_report(requested=False, status="disabled"),
        }
    )
    return report


def _loop_input_shapes(
    *,
    token_ids: Any,
    page_table: Any,
    cache_position: Any,
    kv_cache: Any,
) -> dict[str, Any]:
    return {
        "token_ids": _shape(token_ids),
        "page_table": _shape(page_table),
        "cache_position": _shape(cache_position),
        "key_cache": _shape(kv_cache[0].k),
        "value_cache": _shape(kv_cache[0].v),
    }


def _loop_output_shapes(
    *,
    token: Any,
    kv_cache: Any,
    layer_count: int,
) -> dict[str, Any]:
    return {
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


def _loop_generated_token_ids(
    *,
    token: Any,
    ttnn: Any,
    batch_size: int,
) -> dict[str, Any]:
    materialized = _materialize_token_ids(token, ttnn=ttnn, batch_size=batch_size)
    if materialized is not None:
        return {
            "status": "materialized",
            "source": materialized["source"],
            "token_ids_by_user": materialized["token_ids_by_user"],
        }
    return {
        "status": "unavailable",
        "source": "placeholder_unmaterialized",
        "token_ids_by_user": [[-1] for _ in range(batch_size)],
        "message": "Token ids could not be materialized from the TTNN output.",
    }


def _materialize_token_ids(
    token: Any,
    *,
    ttnn: Any,
    batch_size: int,
) -> dict[str, Any] | None:
    direct = _normalize_token_ids(token, batch_size=batch_size)
    if direct is not None:
        return {"source": "tensor_value", "token_ids_by_user": direct}

    to_torch = getattr(ttnn, "to_torch", None)
    if callable(to_torch):
        try:
            host = to_torch(token)
        except Exception:
            host = None
        normalized = _normalize_token_ids(host, batch_size=batch_size)
        if normalized is not None:
            return {
                "source": "ttnn.to_torch",
                "token_ids_by_user": normalized,
            }
    return None


def _normalize_token_ids(
    value: Any,
    *,
    batch_size: int,
) -> list[list[int]] | None:
    if value is None:
        return None
    for attr in ("tolist",):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                return None
            break
    for attr in ("data", "values", "value"):
        if not isinstance(value, (list, tuple)) and hasattr(value, attr):
            value = getattr(value, attr)
            if hasattr(value, "tolist"):
                value = value.tolist()
            break
    if not isinstance(value, (list, tuple)):
        if batch_size == 1:
            try:
                return [[_as_int(value)]]
            except Exception:
                return None
        return None
    if not value:
        return None

    rows: list[list[int]] = []
    if isinstance(value[0], (list, tuple)) and len(value) == batch_size:
        for row in value:
            if not isinstance(row, (list, tuple)) or not row:
                return None
            rows.append([_last_token_value(row)])
    elif len(value) == batch_size:
        rows = [[_as_int(item)] for item in value]
    elif batch_size == 1:
        rows = [[_last_token_value(value)]]
    else:
        flat = _flatten_token_values(value)
        if len(flat) != batch_size:
            return None
        rows = [[item] for item in flat]

    if len(rows) != batch_size:
        return None
    return rows


def _last_token_value(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty token value")
        return _last_token_value(value[-1])
    return _as_int(value)


def _flatten_token_values(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        values: list[int] = []
        for item in value:
            values.extend(_flatten_token_values(item))
        return values
    return [_as_int(value)]


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _generated_token_id_source(
    per_step_token_metadata: list[dict[str, Any]],
) -> str:
    sources = {
        str(step.get("token_materialization_source"))
        for step in per_step_token_metadata
    }
    if not sources:
        return "none"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _generated_token_materialization_status(
    per_step_token_metadata: list[dict[str, Any]],
) -> str:
    statuses = {
        str(step.get("token_materialization_status"))
        for step in per_step_token_metadata
    }
    if not statuses:
        return "not_run"
    if statuses == {"materialized"}:
        return "materialized"
    if statuses == {"unavailable"}:
        return "unavailable"
    return "partial"


def _loop_reference_summary(
    step_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failed = [
        int(step["step_index"])
        for step in step_reports
        if not step.get("passed")
    ]
    return {
        "kind": "multi_step_structural_shape_dtype_op_sequence",
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "step_count": len(step_reports),
        "failed_steps": failed,
    }


def _loop_throughput_summary(
    *,
    latency_ms: float | None,
    batch_size: int,
    decode_steps: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    total_tokens = batch_size * decode_steps
    summary: dict[str, Any] = {
        "batch_size": batch_size,
        "generated_tokens_per_user": decode_steps,
        "total_generated_tokens": total_tokens,
        "basis": "prompt_decode_loop_latency_ms",
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
    tokens_per_second_per_user = 1000.0 * decode_steps / latency_ms
    summary.update(
        {
            "status": "measured",
            "latency_ms": latency_ms,
            "tokens_per_second_per_user": tokens_per_second_per_user,
            "aggregate_tokens_per_second": (
                1000.0 * total_tokens / latency_ms
            ),
        }
    )
    return summary
