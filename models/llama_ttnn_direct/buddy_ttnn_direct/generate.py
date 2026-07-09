from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from .codegen.parameters import ParameterMaterializationError
from .codegen.ttnn_tensorizer import TTNNTensorizationError
from .runtime_environment import collect_ttnn_environment
from .runtime import GenerateSectionProfiler, TTNNDirectRuntimeContext
from .runtime.decode import (
    build_decode_runtime_for_position as _build_decode_runtime_for_position,
    materialize_generate_token_events as _materialize_generate_token_events,
    prefill_token_direct_handoff as _prefill_token_direct_handoff,
)
from .runtime.generate import build_generate_state as _build_generate_state
from .runtime.profile import (
    profile_generate_from_generate_report as _profile_generate_from_generate_report,
)
from .runtime.reports import (
    cache_population_summary as _cache_population_summary,
    default_generate_report_path as _default_generate_report_path,
    generate_base_report as _generate_base_report,
    generate_end_to_end_contract as _generate_end_to_end_contract,
    generate_failed_report as _generate_failed_report,
    generate_no_device_report as _generate_no_device_report,
    generate_reference_summary as _generate_reference_summary,
    generate_throughput_summary as _generate_throughput_summary,
    host_copy_not_run_profile as _host_copy_not_run_profile,
    host_copy_profile as _host_copy_profile,
    section_profile_not_run as _section_profile_not_run,
)
from .runtime.tokenizer import (
    PromptTokenizationError,
    detokenize_generated_token_ids,
)
from .smoke_decode_shell import (
    _dry_run_reference,
    _dtype,
    _shape,
    _to_namespace,
)
from .smoke_mlp import NoTTNNDeviceError
from .smoke_prefill import (
    _observed_cache_population,
    _planned_cache_population,
    _prefill_plan,
    _prefill_reference,
)
from .smoke_single_layer_decode import (
    _decode_step_plan,
    _decode_step_reference,
    _generated_observed_op_sequence,
    _load_generated_model,
    _time_decode_step,
    _trace_report,
    _write_report,
)
from .decode_loop import (
    _generated_token_id_source,
    _generated_token_materialization_status,
    _loop_input_shapes,
    _loop_output_shapes,
    _materialize_token_ids,
)
from .templates.ttnn_ops import UnsupportedTTNNOp


GENERATE_RUNTIME_OWNER = "TTNNDirectRuntimeContext"


def run_generate(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int = 2,
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
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers)
    token_count = int(max_new_tokens)
    num_layers = int(config["num_layers"])
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if layer_count > num_layers:
        raise ValueError(
            f"layers must be <= generated config num_layers ({num_layers})"
        )
    if token_count <= 0:
        raise ValueError("max_new_tokens must be positive")

    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config.get("seq_len", 1)
    )
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")

    decode_plan = _decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
    )
    prefill_plan = _prefill_plan(
        layers=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        config=config,
    )
    decode_step_count = max(0, token_count - 1)

    if dry_run:
        prefill_cache_population = _planned_cache_population(prefill_plan)
        report = _generate_base_report(
            program_dir=program_root,
            program_num_layers=num_layers,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "runtime_status": "dry_run",
                "prefill_status": "dry_run",
                "decode_loop_runtime_owned": False,
                "planned_decode_loop_runtime_owned": True,
                "kv_cache_source": "prefill",
                "runtime_owner": GENERATE_RUNTIME_OWNER,
                "generated_token_ids": [],
                "generated_text": "",
                "generated_text_by_user": [],
                "generated_text_status": "not_run",
                "generated_text_source": "dry_run",
                "prefill": {
                    "status": "dry_run",
                    "cache_population": prefill_cache_population,
                },
                "prefill_cache_population": prefill_cache_population,
                "prefill_cache_population_summary": (
                    _cache_population_summary(prefill_cache_population)
                ),
                "step_reports": [],
                "per_step_token_metadata": [],
                "tensor_conversion_count": (
                    prefill_plan["tensor_conversion_count"]
                    + decode_plan["tensor_conversion_count"]
                ),
                "runtime_context": {
                    "class": "TTNNDirectRuntimeContext",
                    "status": "planned",
                    "owns": [
                        "parameters",
                        "kv_cache",
                        "page_table",
                        "rotary_state",
                        "tokenizer",
                        "generated_model",
                    ],
                    "parameter_tensorization_count_per_generate": 1,
                    "parameter_tensorization_count_per_decode_step": 0,
                    "kv_cache_initialization_count_per_generate": 1,
                    "kv_cache_reinitialized_per_step": False,
                    "decode_token_runtime_handoff": "device_tensor_direct",
                    "decode_token_host_roundtrip_per_step": False,
                    "host_token_materialization_for_reporting_only": True,
                    "decode_step_count": decode_step_count,
                },
                "parameter_tensorization_count_per_generate": 1,
                "parameter_tensorization_count_per_decode_step": 0,
                "kv_cache_initialization_count_per_generate": 1,
                "kv_cache_reinitialized_per_step": False,
                "decode_token_runtime_handoff": "device_tensor_direct",
                "decode_token_host_roundtrip_per_step": False,
                "host_token_materialization_for_reporting_only": True,
                "synthetic_runtime_input_tensor_count": 0,
                "synthetic_rotary_tensor_count": 0,
                "synthetic_kv_cache_tensor_count": 0,
                "host_copy_profile": _host_copy_not_run_profile("dry_run"),
                "section_profile": _section_profile_not_run("dry_run"),
                "trace": _trace_report(requested=False, status="disabled"),
                "reference": _dry_run_reference("generate"),
                "error": None,
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        report["end_to_end_contract"] = _generate_end_to_end_contract(report)
        _write_report(out, report)
        return report

    if model_path is None:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_model_path",
            message="model_path is required for generate execution",
            detail="model_path was not provided",
        )
        _write_report(out, report)
        return report
    if prompt is None:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_prompt",
            message="prompt is required for generate execution",
            detail="prompt was not provided",
        )
        _write_report(out, report)
        return report
    if decode_plan["output_kind"] != "token":
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="unsupported_output_kind",
            message="generate requires token output",
            detail=f"output_kind={decode_plan['output_kind']}",
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
        report = _generate_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
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
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_torch",
            message="torch is required to build generate runtime tensors",
            detail=str(err),
            ttnn_module=ttnn,
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
            context = _build_generate_state(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
                program_dir=program_root,
                model_path=Path(model_path),
                prompt=prompt,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            generated = _load_generated_model(program_root / "model.py", ttnn)
            generate_config = dict(config)
            generate_config["num_layers"] = layer_count
            generate_config["batch_size"] = batch_size
            generate_config["max_cache_len"] = cache_len
            generate_config["seq_len"] = 1
            generate_config["prefill"] = dict(
                generate_config.get("prefill") or {}
            )
            generate_config["prefill"]["seq_len"] = prefill_len
            model = generated.BuddyLlama31TTNN(
                device=ttnn_device,
                parameters=context.parameters,
                config=_to_namespace(generate_config),
            )
            section_profiler = GenerateSectionProfiler(
                ttnn=ttnn,
                device=ttnn_device,
            )
            section_profiler.install(model)
            context.install_generated_model(
                generated_module=generated,
                generated_model=model,
            )

            total_start = time.perf_counter()
            prefill_start = time.perf_counter()
            prefill_token, kv_cache, cache_reports = (
                context.generated_model.prefill_prompt(
                    context.prefill_token_ids,
                    context.kv_cache,
                    context.prefill_page_table,
                )
            )
            context.update_kv_cache(kv_cache)
            synchronize = getattr(ttnn, "synchronize_device", None)
            if callable(synchronize):
                synchronize(ttnn_device)
            prefill_latency_ms = (time.perf_counter() - prefill_start) * 1000.0
            prefill_output_shapes = {
                "token": _shape(prefill_token),
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
            prefill_cache_population = _observed_cache_population(
                plan=prefill_plan,
                cache_reports=cache_reports,
                output_shapes=prefill_output_shapes,
            )
            prefill_reference = _prefill_reference(
                plan=prefill_plan,
                layer_count=layer_count,
                output_shapes=prefill_output_shapes,
                output={
                    "kind": "token",
                    "shape": _shape(prefill_token),
                    "dtype": _dtype(prefill_token),
                },
                observed_ops=_generated_observed_op_sequence(
                    context.generated_model,
                    ttnn,
                ),
            )
            first_token = _prefill_token_direct_handoff(
                prefill_token=prefill_token
            )
            context.update_decode_token(first_token.token_ids)
            generated_token_events = [
                {
                    "step_index": "prefill",
                    "token": first_token.token_ids,
                    "runtime_handoff": first_token.runtime_handoff,
                    "runtime_host_roundtrip": (
                        first_token.runtime_host_roundtrip
                    ),
                    "cache_position_value": (
                        context.prefill_tokenization["effective_token_count"] - 1
                    ),
                    "token_shape": _shape(first_token.token_ids),
                }
            ]

            decode_runtime = _build_decode_runtime_for_position(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                parameters=context.parameters,
                decode_plan=decode_plan,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_effective_token_count=(
                    context.prefill_tokenization["effective_token_count"]
                ),
                generated_token_index=0,
            )
            context.install_decode_runtime(decode_runtime)
            decode_runtime_state = context.decode_runtime_state
            rotary_runtime_state = context.rotary_state
            tensor_conversion_count = (
                context.tensor_conversion_count
                + first_token.tensor_conversion_count
                + decode_runtime.tensor_conversion_count
            )
            decode_runtime_state_count = (
                decode_runtime.decode_runtime_state_input_tensor_count
            )
            decode_rotary_runtime_count = (
                decode_runtime.rotary_runtime_input_tensor_count
            )
            step_reports = []
            for step_index in range(decode_step_count):
                input_shapes = _loop_input_shapes(
                    token_ids=context.token_ids,
                    page_table=context.page_table,
                    cache_position=context.cache_position,
                    kv_cache=context.kv_cache,
                )
                token, kv_cache, latency_ms = _time_decode_step(
                    ttnn=ttnn,
                    model=context.generated_model,
                    device=ttnn_device,
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
                    "shape": _shape(token),
                    "dtype": _dtype(token),
                    "repr": repr(token),
                }
                generated_token_events.append(
                    {
                        "step_index": step_index,
                        "token": token,
                        "runtime_handoff": "device_tensor_direct",
                        "runtime_host_roundtrip": False,
                        "cache_position_value": decode_runtime_state.get(
                            "cache_position_value"
                        ),
                        "page_table_shape": input_shapes.get("page_table"),
                        "token_shape": _shape(token),
                    }
                )
                reference = _decode_step_reference(
                    plan=decode_plan,
                    layer_count=layer_count,
                    output_shapes=output_shapes,
                    output=output,
                    observed_ops=_generated_observed_op_sequence(
                        context.generated_model,
                        ttnn,
                    ),
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
                        "cache_position_value": decode_runtime_state.get(
                            "cache_position_value"
                        ),
                        "input_shapes": input_shapes,
                        "decode_runtime_state": decode_runtime_state,
                        "rotary_runtime_state": rotary_runtime_state,
                        "output_shapes": output_shapes,
                        "output": output,
                        "generated_token_ids": [],
                        "token_materialization": {
                            "status": "deferred",
                            "source": "reporting_after_decode_loop",
                        },
                        "token_materialization_ms": None,
                        "token_runtime_handoff": "device_tensor_direct",
                        "runtime_host_roundtrip": False,
                        "reference": reference,
                    }
                )
                context.update_decode_token(token)
                if step_index + 1 < decode_step_count:
                    decode_runtime = _build_decode_runtime_for_position(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        parameters=context.parameters,
                        decode_plan=decode_plan,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        prefill_effective_token_count=(
                            context.prefill_tokenization["effective_token_count"]
                        ),
                        generated_token_index=step_index + 1,
                    )
                    context.install_decode_runtime(decode_runtime)
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

            latency_ms = (time.perf_counter() - total_start) * 1000.0
            token_materialization = _materialize_generate_token_events(
                generated_token_events,
                step_reports=step_reports,
                ttnn=ttnn,
                batch_size=batch_size,
            )
            generated_token_ids_by_user = (
                token_materialization.generated_token_ids_by_user
            )
            per_step_token_metadata = (
                token_materialization.per_step_token_metadata
            )
            text_report = detokenize_generated_token_ids(
                token_ids_by_user=generated_token_ids_by_user,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            host_copy_profile = _host_copy_profile(
                first_token_materialization_ms=(
                    token_materialization.first_token_materialization_ms
                ),
                step_reports=step_reports,
            )
            decode_passed = all(step["passed"] for step in step_reports)
            passed = bool(prefill_reference["passed"] and decode_passed)
            parameter_setup = dict(context.parameter_setup)
            parameter_setup.update(
                {
                    "generate_runtime_owned": passed,
                    "decode_loop_runtime_owned": (
                        decode_step_count == 0 or decode_passed
                    ),
                    "prefill_prompt_runtime_input_tensor_count": (
                        context.prefill_prompt_runtime_input_tensor_count
                    ),
                    "prefill_rotary_runtime_input_tensor_count": (
                        context.prefill_rotary_runtime_input_tensor_count
                    ),
                    "prefill_first_token_tensor_conversion_count": (
                        first_token.tensor_conversion_count
                    ),
                    "decode_runtime_state_input_tensor_count": (
                        decode_runtime_state_count
                    ),
                    "decode_rotary_runtime_input_tensor_count": (
                        decode_rotary_runtime_count
                    ),
                    "decode_loop_step_count": decode_step_count,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "parameter_tensorization_count_per_generate": (
                        context.parameter_tensorization_count_per_generate
                    ),
                    "parameter_tensorization_count_per_decode_step": (
                        context.parameter_tensorization_count_per_decode_step
                    ),
                    "kv_cache_initialization_count_per_generate": (
                        context.kv_cache_initialization_count_per_generate
                    ),
                    "kv_cache_reinitialized_per_step": (
                        context.kv_cache_reinitialized_per_step
                    ),
                    "decode_token_runtime_handoff": (
                        context.decode_token_runtime_handoff
                    ),
                    "decode_token_host_roundtrip_per_step": (
                        context.decode_token_host_roundtrip_per_step
                    ),
                    "host_token_materialization_for_reporting_only": (
                        context.host_token_materialization_for_reporting_only
                    ),
                }
            )
            report = _generate_base_report(
                program_dir=program_root,
                program_num_layers=num_layers,
                layers=layer_count,
                max_new_tokens=token_count,
                decode_steps=decode_step_count,
                prefill_len=prefill_len,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                dry_run=False,
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
            )
            report.update(
                {
                    "passed": passed,
                    "status": "passed" if passed else "reference_mismatch",
                    "runtime_status": (
                        "passed" if passed else "reference_mismatch"
                    ),
                    "prefill_status": (
                        "passed"
                        if prefill_reference["passed"]
                        else "reference_mismatch"
                    ),
                    "decode_loop_runtime_owned": (
                        decode_step_count == 0 or decode_passed
                    ),
                    "generate_runtime_owned": passed,
                    "kv_cache_source": "prefill",
                    "input_source": "prompt_prefill",
                    "runtime_owner": GENERATE_RUNTIME_OWNER,
                    "parameter_source": context.parameter_source,
                    "parameter_setup": parameter_setup,
                    "prompt_tokenization": context.prefill_tokenization,
                    "prefill_tokenization": context.prefill_tokenization,
                    "decode_runtime_state": decode_runtime_state,
                    "rotary_runtime_state": rotary_runtime_state,
                    "kv_cache_runtime_state": context.kv_cache_runtime_state,
                    "runtime_context": context.to_report(
                        decode_step_count=decode_step_count
                    ),
                    "parameter_tensorization_count_per_generate": (
                        context.parameter_tensorization_count_per_generate
                    ),
                    "parameter_tensorization_count_per_decode_step": (
                        context.parameter_tensorization_count_per_decode_step
                    ),
                    "kv_cache_initialization_count_per_generate": (
                        context.kv_cache_initialization_count_per_generate
                    ),
                    "kv_cache_reinitialized_per_step": (
                        context.kv_cache_reinitialized_per_step
                    ),
                    "decode_token_runtime_handoff": (
                        context.decode_token_runtime_handoff
                    ),
                    "decode_token_host_roundtrip_per_step": (
                        context.decode_token_host_roundtrip_per_step
                    ),
                    "host_token_materialization_for_reporting_only": (
                        context.host_token_materialization_for_reporting_only
                    ),
                    "prefill": {
                        "status": (
                            "passed"
                            if prefill_reference["passed"]
                            else "reference_mismatch"
                        ),
                        "latency_ms": prefill_latency_ms,
                        "output_shapes": prefill_output_shapes,
                        "output": {
                            "kind": "token",
                            "shape": _shape(prefill_token),
                            "dtype": _dtype(prefill_token),
                            "repr": repr(prefill_token),
                        },
                        "first_token": {
                            "status": (
                                token_materialization.first_token_status
                            ),
                            "source": (
                                token_materialization.first_token_source
                            ),
                            "token_ids_by_user": (
                                token_materialization.first_token_ids_by_user
                            ),
                            "token_shape": _shape(first_token.token_ids),
                            "runtime_handoff": first_token.runtime_handoff,
                            "runtime_host_roundtrip": (
                                first_token.runtime_host_roundtrip
                            ),
                            "host_roundtrip": False,
                            "host_materialization_for_reporting": True,
                            "host_materialization_ms": (
                                token_materialization
                                .first_token_materialization_ms
                            ),
                        },
                        "cache_population": prefill_cache_population,
                        "reference": prefill_reference,
                    },
                    "prefill_cache_population": prefill_cache_population,
                    "prefill_cache_population_summary": (
                        _cache_population_summary(prefill_cache_population)
                    ),
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
                        else prefill_output_shapes
                    ),
                    "output": step_reports[-1]["output"] if step_reports else None,
                    "tensor_conversion_count": tensor_conversion_count,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "host_copy_profile": host_copy_profile,
                    "section_profile": section_profiler.to_report(
                        host_copy_profile=host_copy_profile,
                    ),
                    "latency_ms": latency_ms,
                    "throughput_summary": _generate_throughput_summary(
                        latency_ms=latency_ms,
                        batch_size=batch_size,
                        max_new_tokens=token_count,
                    ),
                    "trace": _trace_report(requested=False, status="disabled"),
                    "reference": _generate_reference_summary(
                        prefill_reference=prefill_reference,
                        step_reports=step_reports,
                    ),
                    "error": None
                    if passed
                    else "generate structural reference mismatch",
                    "ttnn_version": getattr(ttnn, "__version__", None),
                    "ttnn_environment": collect_ttnn_environment(ttnn),
                }
            )
            report["end_to_end_contract"] = _generate_end_to_end_contract(
                report
            )
    except NoTTNNDeviceError as err:
        report = _generate_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            detail=str(err),
            ttnn_module=ttnn,
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
    ) as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="parameter_setup_error",
            message=str(err),
            detail=str(err),
            ttnn_module=ttnn,
        )
    except UnsupportedTTNNOp as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_module=ttnn,
        )
    except Exception as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_module=ttnn,
        )

    _write_report(out, report)
    return report


def run_profile_generate(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int = 2,
    layers: int = 1,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    generate_report: str | Path | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    profile_path = Path(out)
    generate_report_path = (
        Path(generate_report)
        if generate_report is not None
        else _default_generate_report_path(profile_path)
    )
    generate_payload = run_generate(
        out=generate_report_path,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_new_tokens,
        layers=layers,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=dry_run,
        ttnn_module=ttnn_module,
        torch_module=torch_module,
        tokenizer_module=tokenizer_module,
    )
    report = _profile_generate_from_generate_report(
        generate_payload,
        profile_path=profile_path,
        generate_report_path=generate_report_path,
    )
    _write_report(profile_path, report)
    return report


class _maybe_generate_device:
    def __init__(self, ttnn: Any, device_id: int, injected: Any | None) -> None:
        self.ttnn = ttnn
        self.device_id = device_id
        self.injected = injected
        self.device = None
        self.opened = False

    def __enter__(self) -> Any:
        if self.injected is not None:
            self.device = f"fake-device:{self.device_id}"
            return self.device
        open_device = getattr(self.ttnn, "open_device", None)
        if not callable(open_device):
            raise NoTTNNDeviceError("ttnn.open_device is not available")
        self.device = open_device(device_id=self.device_id)
        self.opened = True
        return self.device

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.opened:
            return
        close_device = getattr(self.ttnn, "close_device", None)
        if callable(close_device):
            close_device(self.device)
