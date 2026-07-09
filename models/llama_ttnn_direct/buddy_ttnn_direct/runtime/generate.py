from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
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
from ..decode_loop import (
    _generated_token_id_source,
    _generated_token_materialization_status,
    _materialize_token_ids,
)
from ..runtime_environment import collect_ttnn_environment
from ..smoke_decode_shell import (
    _dtype,
    _shape,
    _to_namespace,
)
from ..smoke_mlp import NoTTNNDeviceError
from ..smoke_prefill import (
    _prefill_plan,
)
from ..smoke_single_layer_decode import (
    DECODE_PARAMETER_ROLES,
    _decode_step_plan,
    _load_generated_model,
    _materialization_summary,
    _tensorization_summary,
    _trace_report,
    _write_report,
)
from ..templates.ttnn_ops import UnsupportedTTNNOp
from .context import TTNNDirectRuntimeContext
from .decode import (
    materialize_generate_token_events as _materialize_generate_token_events,
    run_decode_loop,
)
from .kv_cache import build_prompt_decode_kv_cache_tensors
from .prefill import (
    attach_prefill_rotary_parameters,
    build_prefill_page_table_tensor,
    prefill_token_ids_tensor,
    run_prefill_prompt,
)
from .profile import (
    GenerateSectionProfiler,
    profile_generate_from_generate_report as _profile_generate_from_generate_report,
)
from .reports import (
    cache_population_summary as _cache_population_summary,
    default_generate_report_path as _default_generate_report_path,
    generate_base_report as _generate_base_report,
    generate_dry_run_report as _generate_dry_run_report,
    generate_end_to_end_contract as _generate_end_to_end_contract,
    generate_failed_report as _generate_failed_report,
    generate_no_device_report as _generate_no_device_report,
    generate_reference_summary as _generate_reference_summary,
    generate_throughput_summary as _generate_throughput_summary,
    host_copy_profile as _host_copy_profile,
)
from .tokenizer import (
    PromptTokenizationError,
    detokenize_generated_token_ids,
    tokenize_prompt_for_prefill,
)


GENERATE_RUNTIME_OWNER = "TTNNDirectRuntimeContext"


def build_generate_state(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    program_dir: Path,
    model_path: Path,
    prompt: str,
    tokenizer_path: str | Path,
    tokenizer_module: Any | None,
) -> TTNNDirectRuntimeContext:
    host_params = load_llama_parameters_from_manifests(
        model_path=model_path,
        weights_manifest=program_dir / "weights_manifest.json",
        config=program_dir / "config.json",
        tensor_backend="torch",
        layers=range(int(decode_plan["layers"])),
    )
    materialization_summary = _materialization_summary(host_params)
    result = to_ttnn_parameters(
        host_params,
        device,
        load_parameter_config_from_program(program_dir),
        roles=DECODE_PARAMETER_ROLES,
        layers=range(int(decode_plan["layers"])),
        ttnn_module=ttnn,
    )
    assert result.parameters is not None
    prefill_tokenization = tokenize_prompt_for_prefill(
        prompt=prompt,
        batch_size=int(prefill_plan["batch_size"]),
        prefill_len=int(prefill_plan["prefill_len"]),
        tokenizer_path=tokenizer_path,
        vocab_size=prefill_plan.get("vocab_size"),
        tokenizer_module=tokenizer_module,
    )
    prefill_token_ids = prefill_token_ids_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        token_ids=prefill_tokenization.token_ids,
    )
    prefill_page_table = build_prefill_page_table_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        batch_size=int(prefill_plan["batch_size"]),
        cache_len=int(prefill_plan["cache_len"]),
        page_block_size=int(prefill_plan["kv_cache"]["page_block_size"]),
        prompt_token_count=int(prefill_tokenization.effective_token_count),
    )
    kv_runtime = build_prompt_decode_kv_cache_tensors(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        layer_count=int(prefill_plan["layers"]),
        batch_size=int(prefill_plan["batch_size"]),
        cache_len=int(prefill_plan["cache_len"]),
        page_block_size=int(prefill_plan["kv_cache"]["page_block_size"]),
        num_kv_heads=int(prefill_plan["kv_cache"]["logical_shape"][2]),
        head_dim=int(prefill_plan["kv_cache"]["logical_shape"][3]),
    )
    prefill_rotary = attach_prefill_rotary_parameters(
        parameters=result.parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        prefill_plan=prefill_plan,
    )
    tensorization_count = int(result.report["tensor_count"])
    tensor_conversion_count = (
        tensorization_count
        + 1
        + int(prefill_page_table.tensor_conversion_count)
        + int(kv_runtime.tensor_conversion_count)
        + int(prefill_rotary.tensor_conversion_count)
    )
    return TTNNDirectRuntimeContext(
        parameters=result.parameters,
        prefill_token_ids=prefill_token_ids,
        prefill_page_table=prefill_page_table.page_table,
        kv_cache=kv_runtime.kv_cache,
        tensor_conversion_count=tensor_conversion_count,
        parameter_source="hf_model",
        input_source="prompt_prefill",
        prefill_tokenization=prefill_tokenization.to_report(),
        prefill_page_table_runtime_state=(
            prefill_page_table.prefill_page_table_runtime_state
        ),
        kv_cache_runtime_state=kv_runtime.kv_cache_runtime_state,
        prefill_prompt_runtime_input_tensor_count=1,
        prefill_page_table_runtime_input_tensor_count=(
            prefill_page_table.tensor_conversion_count
        ),
        prefill_rotary_runtime_input_tensor_count=(
            prefill_rotary.tensor_conversion_count
        ),
        parameter_setup={
            "materialization": materialization_summary,
            "tensorization": _tensorization_summary(result.report),
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "prefill_prompt_runtime_input_tensor_count": 1,
            "prefill_page_table_runtime_input_tensor_count": (
                prefill_page_table.tensor_conversion_count
            ),
            "prefill_page_table_runtime_state": (
                prefill_page_table.prefill_page_table_runtime_state
            ),
            "prefill_rotary_runtime_input_tensor_count": (
                prefill_rotary.tensor_conversion_count
            ),
            "kv_cache_runtime_input_tensor_count": (
                kv_runtime.tensor_conversion_count
            ),
            "kv_cache_runtime_state": kv_runtime.kv_cache_runtime_state,
        },
        tokenizer_path=tokenizer_path,
        tokenizer_module=tokenizer_module,
    )


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
        with _maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
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
