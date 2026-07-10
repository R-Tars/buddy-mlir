from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from ..codegen.parameters import ParameterMaterializationError
from ..codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
)
from ..smoke_decode_shell import (
    _to_namespace,
)
from ..smoke_mlp import NoTTNNDeviceError
from ..smoke_prefill import (
    _prefill_plan,
)
from ..smoke_single_layer_decode import (
    _decode_step_plan,
    _load_generated_model,
)
from ..ttnn_compat import UnsupportedTTNNOp
from .decode import (
    materialize_generate_token_events as _materialize_generate_token_events,
    run_decode_loop,
)
from .device import maybe_generate_device
from .prefill import (
    run_prefill_prompt,
)
from .profile import (
    GenerateSectionProfiler,
)
from .reports import (
    generate_dry_run_report as _generate_dry_run_report,
    generate_failed_report as _generate_failed_report,
    generate_no_device_report as _generate_no_device_report,
    generate_success_report as _generate_success_report,
    write_report as _write_report,
)
from .tokenizer import (
    PromptTokenizationError,
    detokenize_generated_token_ids,
)
from .state import build_generate_state


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
    observer: Any | None = None,
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
        report = _generate_dry_run_report(
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
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
        )
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
        with maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
            context = build_generate_state(
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
                observer=observer,
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
            prefill_result = run_prefill_prompt(
                context=context,
                ttnn=ttnn,
                device=ttnn_device,
                prefill_plan=prefill_plan,
                layer_count=layer_count,
            )
            prefill_token = prefill_result.prefill_token
            kv_cache = prefill_result.kv_cache
            prefill_latency_ms = prefill_result.latency_ms
            prefill_output_shapes = prefill_result.output_shapes
            prefill_cache_population = prefill_result.cache_population
            prefill_reference = prefill_result.reference
            first_token = prefill_result.first_token
            generated_token_events = list(prefill_result.generated_token_events)
            observe_kv_cache = getattr(
                observer,
                "observe_prefill_kv_cache",
                None,
            )
            if callable(observe_kv_cache):
                observe_kv_cache(
                    kv_cache,
                    effective_token_count=context.prefill_tokenization[
                        "effective_token_count"
                    ],
                )

            decode_loop = run_decode_loop(
                context=context,
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                decode_plan=decode_plan,
                batch_size=batch_size,
                cache_len=cache_len,
                layer_count=layer_count,
                decode_step_count=decode_step_count,
                generated_token_events=generated_token_events,
                initial_tensor_conversion_count=(
                    context.tensor_conversion_count
                    + first_token.tensor_conversion_count
                ),
            )
            generated_token_events = decode_loop.generated_token_events
            step_reports = decode_loop.step_reports
            decode_runtime_state = decode_loop.decode_runtime_state
            rotary_runtime_state = decode_loop.rotary_runtime_state
            tensor_conversion_count = decode_loop.tensor_conversion_count
            decode_runtime_state_count = (
                decode_loop.decode_runtime_state_input_tensor_count
            )
            decode_rotary_runtime_count = (
                decode_loop.decode_rotary_runtime_input_tensor_count
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
            report = _generate_success_report(
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
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
                context=context,
                prefill_token=prefill_token,
                first_token=first_token,
                prefill_latency_ms=prefill_latency_ms,
                prefill_output_shapes=prefill_output_shapes,
                prefill_cache_population=prefill_cache_population,
                prefill_reference=prefill_reference,
                step_reports=step_reports,
                per_step_token_metadata=per_step_token_metadata,
                generated_token_ids_by_user=generated_token_ids_by_user,
                token_materialization=token_materialization,
                text_report=text_report,
                decode_runtime_state=decode_runtime_state,
                rotary_runtime_state=rotary_runtime_state,
                tensor_conversion_count=tensor_conversion_count,
                decode_runtime_state_input_tensor_count=(
                    decode_runtime_state_count
                ),
                decode_rotary_runtime_input_tensor_count=(
                    decode_rotary_runtime_count
                ),
                latency_ms=latency_ms,
                section_profiler=section_profiler,
                ttnn_module=ttnn,
            )
            observation_summary = getattr(observer, "summary", None)
            if callable(observation_summary):
                report["correctness_observations"] = observation_summary()
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
