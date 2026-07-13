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
from ..smoke_mlp import NoTTNNDeviceError
from ..smoke_prefill import (
    _prefill_plan,
)
from ..smoke_single_layer_decode import (
    _decode_step_plan,
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
    compact_generate_report as _compact_generate_report,
    generate_dry_run_report as _generate_dry_run_report,
    generate_failed_report as _generate_failed_report,
    generate_no_device_report as _generate_no_device_report,
    generate_success_report as _generate_success_report,
    reset_json_lines as _reset_json_lines,
    write_report as _write_report,
)
from .tokenizer import (
    PromptTokenizationError,
    detokenize_generated_token_ids,
    tokenize_prompt_for_prefill,
)
from .session import build_runtime_session
from .state import build_generate_state


def run_generate(
    *,
    out: str | Path | None = None,
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
    report_level: str | None = None,
) -> dict[str, Any]:
    resolved_report_level = _resolve_report_level(
        out=out,
        report_level=report_level,
    )
    diagnostics_path = _diagnostics_path(out, resolved_report_level)
    if diagnostics_path is not None:
        diagnostics_path.unlink(missing_ok=True)
        _diagnostics_reference_path(diagnostics_path).unlink(missing_ok=True)
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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )
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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )
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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

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
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

    try:
        prefill_tokenization = tokenize_prompt_for_prefill(
            prompt=prompt,
            batch_size=batch_size,
            prefill_len=prefill_len,
            tokenizer_path=tokenizer_path or model_path,
            vocab_size=prefill_plan.get("vocab_size"),
            tokenizer_module=tokenizer_module,
        )
    except (PromptTokenizationError, ValueError) as err:
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
            status="prompt_tokenization_error",
            message=str(err),
            detail=str(err),
            ttnn_module=ttnn,
        )
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

    required_cache_len = (
        int(prefill_tokenization.effective_token_count) + decode_step_count
    )
    cache_capacity = {
        "effective_prompt_tokens": int(
            prefill_tokenization.effective_token_count
        ),
        "decode_steps": decode_step_count,
        "required_cache_len": required_cache_len,
        "configured_cache_len": cache_len,
        "passed": required_cache_len <= cache_len,
    }
    if required_cache_len > cache_len:
        message = (
            "generate exceeds KV cache capacity: "
            "effective prompt tokens "
            f"({prefill_tokenization.effective_token_count}) "
            f"+ decode steps ({decode_step_count}) requires cache_len >= "
            f"{required_cache_len}, got {cache_len}"
        )
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
            status="cache_capacity_exceeded",
            message=message,
            detail=message,
            ttnn_module=ttnn,
        )
        report["prompt_tokenization"] = prefill_tokenization.to_report()
        report["cache_capacity"] = cache_capacity
        return _finalize_report(
            report,
            out=out,
            report_level=resolved_report_level,
            diagnostics_path=diagnostics_path,
        )

    try:
        with maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
            session = build_runtime_session(
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
                config=config,
                layer_count=layer_count,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_len=prefill_len,
                observer=observer,
                prefill_tokenization=prefill_tokenization,
            )
            context = session.context
            model = session.generated_model
            section_profiler = GenerateSectionProfiler(
                ttnn=ttnn,
                device=ttnn_device,
            )
            section_profiler.install(model)

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

            if diagnostics_path is not None:
                _reset_json_lines(diagnostics_path)
                _reset_json_lines(
                    _diagnostics_reference_path(diagnostics_path)
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
                report_level=resolved_report_level,
                diagnostics_path=(
                    str(diagnostics_path)
                    if diagnostics_path is not None
                    else None
                ),
                diagnostics_reference_path=(
                    str(_diagnostics_reference_path(diagnostics_path))
                    if diagnostics_path is not None
                    else None
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
            report["cache_capacity"] = cache_capacity
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

    return _finalize_report(
        report,
        out=out,
        report_level=resolved_report_level,
        diagnostics_path=diagnostics_path,
    )


def _resolve_report_level(
    *,
    out: str | Path | None,
    report_level: str | None,
) -> str:
    if report_level is None:
        return "full" if out is not None else "none"
    normalized = str(report_level).lower()
    if normalized not in {"none", "summary", "full"}:
        raise ValueError(
            "report_level must be one of: none, summary, full"
        )
    if normalized == "none" and out is not None:
        raise ValueError("report_level=none cannot be combined with out")
    if normalized in {"summary", "full"} and out is None:
        raise ValueError(f"report_level={normalized} requires out")
    return normalized


def _diagnostics_path(
    out: str | Path | None,
    report_level: str,
) -> Path | None:
    if out is None or report_level != "full":
        return None
    out_path = Path(out)
    return out_path.with_name(f"{out_path.stem}.steps.jsonl")


def _diagnostics_reference_path(diagnostics_path: Path) -> Path:
    name = diagnostics_path.name
    if name.endswith(".steps.jsonl"):
        name = f"{name[:-len('.steps.jsonl')]}.references.jsonl"
    else:
        name = f"{diagnostics_path.stem}.references.jsonl"
    return diagnostics_path.with_name(name)


def _finalize_report(
    report: dict[str, Any],
    *,
    out: str | Path | None,
    report_level: str,
    diagnostics_path: Path | None,
) -> dict[str, Any]:
    if diagnostics_path is not None and diagnostics_path.is_file():
        reference_path = _diagnostics_reference_path(diagnostics_path)
        report["diagnostics"] = {
            "decode_steps": str(diagnostics_path),
            "references": (
                str(reference_path) if reference_path.is_file() else None
            ),
            "format": "jsonl",
            "status": "written",
        }
    report["report_level"] = report_level
    finalized = (
        report
        if report_level == "full"
        else _compact_generate_report(report, report_level=report_level)
    )
    if out is not None:
        _write_report(out, finalized)
    return finalized
