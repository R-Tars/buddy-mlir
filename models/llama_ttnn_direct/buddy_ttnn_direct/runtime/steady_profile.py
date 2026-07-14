from __future__ import annotations

import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..runtime_environment import collect_ttnn_environment
from .profile_baselines import (
    PROFILE_GENERATE_OFFICIAL_BASELINE_ID,
    PROFILE_GENERATE_OFFICIAL_BATCH_SIZE,
    PROFILE_GENERATE_OFFICIAL_TPS_PER_USER,
)
from .reports import (
    host_copy_not_run_profile,
    section_profile_not_run,
    write_report as _write_report,
)


def run_profile_decode_steady(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    input_prompts: str | Path | None = None,
    instruct: bool = False,
    tokenizer_path: str | Path | None = None,
    layers: int | None = None,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    warmup: int = 5,
    iterations: int = 50,
    after_prefill: bool = True,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    """Measure repeated full decode iterations after one prompt prefill."""

    from ..codegen.parameters import ParameterMaterializationError
    from ..codegen.ttnn_tensorizer import TTNNTensorizationError
    from ..ttnn_compat import UnsupportedTTNNOp
    from .decode import run_decode_steady_iterations
    from .device import maybe_generate_device
    from .errors import NoTTNNDeviceError
    from .plans import decode_step_plan, prefill_plan as build_prefill_plan
    from .prefill import run_prefill_prompt
    from .session import build_runtime_session
    from .tokenizer import (
        PromptTokenizationError,
        load_prompt_batch,
        tokenize_prompts_for_prefill,
    )

    profile_path = Path(out)
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    num_layers = int(config["num_layers"])
    layer_count = num_layers if layers is None else int(layers)
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config.get("seq_len", 1)
    )
    warmup = int(warmup)
    iterations = int(iterations)
    if layer_count <= 0 or layer_count > num_layers:
        raise ValueError(f"layers must be in [1, {num_layers}]")
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not after_prefill:
        raise ValueError("decode-steady requires after_prefill=True")

    decode_plan = decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
    )
    prefill_plan = build_prefill_plan(
        layers=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        config=config,
    )
    base = _decode_steady_report_base(
        profile_path=profile_path,
        program_root=program_root,
        num_layers=num_layers,
        layer_count=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        prefill_len=prefill_len,
        warmup=warmup,
        iterations=iterations,
        device=device,
        device_id=device_id,
        dtype_seed=dtype_seed,
    )
    if dry_run:
        report = {
            **base,
            "status": "dry_run",
            "passed": True,
            "dry_run": True,
            "prefill_status": "planned",
            "acceptance": {
                "status": "dry_run",
                "passed": True,
                "failed_checks": [],
            },
            "message": "Steady decode profile planned; no device was opened.",
        }
        _write_report(profile_path, report)
        return report

    if model_path is None or (prompt is None and input_prompts is None):
        missing = "model_path" if model_path is None else "prompt"
        report = _decode_steady_failed_report(
            base,
            status=f"missing_{missing}",
            message=(
                f"{missing} is required for decode-steady execution"
                if missing == "model_path"
                else "prompt or input_prompts is required for "
                "decode-steady execution"
            ),
        )
        _write_report(profile_path, report)
        return report

    try:
        ttnn = ttnn_module or importlib.import_module("ttnn")
    except ImportError as err:
        report = _decode_steady_failed_report(
            base,
            status="no_device",
            message=str(err),
        )
        _write_report(profile_path, report)
        return report
    try:
        torch = torch_module or importlib.import_module("torch")
    except ImportError as err:
        report = _decode_steady_failed_report(
            base,
            status="missing_torch",
            message=str(err),
            ttnn_module=ttnn,
        )
        _write_report(profile_path, report)
        return report

    try:
        prompt_batch = load_prompt_batch(
            prompt=prompt,
            input_prompts=input_prompts,
            batch_size=batch_size,
        )
        prefill_tokenization = tokenize_prompts_for_prefill(
            prompt_batch=prompt_batch,
            prefill_len=prefill_len,
            tokenizer_path=tokenizer_path or model_path,
            vocab_size=prefill_plan.get("vocab_size"),
            tokenizer_module=tokenizer_module,
            instruct=instruct,
            reject_truncation=input_prompts is not None,
            padding_token_id=0 if input_prompts is not None else None,
        )
    except (PromptTokenizationError, ValueError) as err:
        report = _decode_steady_failed_report(
            base,
            status="prompt_tokenization_error",
            message=str(err),
            ttnn_module=ttnn,
        )
        _write_report(profile_path, report)
        return report

    try:
        with maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
            setup_start = time.perf_counter()
            session = build_runtime_session(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
                program_dir=program_root,
                model_path=Path(model_path),
                prompt=prompt_batch.prompts[0],
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
                config=config,
                layer_count=layer_count,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_len=prefill_len,
                prefill_tokenization=prefill_tokenization,
            )
            setup_ms = (time.perf_counter() - setup_start) * 1000.0
            context = session.context
            prefill_result = run_prefill_prompt(
                context=context,
                ttnn=ttnn,
                device=ttnn_device,
                prefill_plan=prefill_plan,
                layer_count=layer_count,
            )
            decode_result = run_decode_steady_iterations(
                context=context,
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                decode_plan=decode_plan,
                batch_size=batch_size,
                cache_len=cache_len,
                warmup=warmup,
                iterations=iterations,
            )

            samples = list(decode_result.measured_step_ms_samples)
            warmup_samples = list(decode_result.warmup_step_ms_samples)
            decode_step_ms_mean = sum(samples) / len(samples)
            decode_step_ms_p50 = statistics.median(samples)
            tokens_per_second_per_user = 1000.0 / decode_step_ms_mean
            aggregate_tokens_per_second = (
                tokens_per_second_per_user * batch_size
            )
            prefill_passed = bool(prefill_result.reference.get("passed"))
            checks = [
                {
                    "name": "decode_steady.prefill",
                    "passed": prefill_passed,
                },
                {
                    "name": "decode_steady.sample_count",
                    "passed": len(samples) == iterations,
                    "observed": len(samples),
                    "expected": iterations,
                },
                {
                    "name": "decode_steady.positive_latency",
                    "passed": all(value > 0.0 for value in samples),
                },
                {
                    "name": "decode_steady.device_token_handoff",
                    "passed": context.decode_token_host_roundtrip_per_step is False,
                },
            ]
            failed_checks = [
                check["name"] for check in checks if not check["passed"]
            ]
            passed = not failed_checks
            measured_positions = decode_result.cache_positions[warmup:]
            report = {
                **base,
                "status": "profiled" if passed else "profile_incomplete",
                "passed": passed,
                "dry_run": False,
                "setup_ms": setup_ms,
                "prefill_status": "passed" if prefill_passed else "failed",
                "prefill_ms": float(prefill_result.latency_ms),
                "prefill_cache_population": prefill_result.cache_population,
                "warmup_step_ms_samples": warmup_samples,
                "warmup_total_ms": sum(warmup_samples),
                "decode_step_ms_samples": samples,
                "decode_step_ms_p50": decode_step_ms_p50,
                "decode_step_ms_mean": decode_step_ms_mean,
                "decode_step_ms_min": min(samples),
                "decode_step_ms_max": max(samples),
                "decode_total_ms": sum(samples),
                "tokens_per_second_per_user": tokens_per_second_per_user,
                "aggregate_tokens_per_second": aggregate_tokens_per_second,
                "throughput_summary": {
                    "status": "measured",
                    "measured_tokens_per_user": iterations,
                    "measured_aggregate_tokens": iterations * batch_size,
                    "tokens_per_second_per_user": tokens_per_second_per_user,
                    "aggregate_tokens_per_second": aggregate_tokens_per_second,
                    "official_reference_tokens_per_second_per_user": (
                        PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
                    ),
                    "ratio_of_official_reference": (
                        tokens_per_second_per_user
                        / PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
                    ),
                },
                "cache_position_start": (
                    measured_positions[0] if measured_positions else None
                ),
                "cache_position_end": (
                    measured_positions[-1] if measured_positions else None
                ),
                "runtime_context": context.to_report(
                    decode_step_count=warmup + iterations
                ),
                "parameter_setup": context.parameter_setup,
                "decode_runtime_state_input_tensor_count": (
                    decode_result.decode_runtime_state_input_tensor_count
                ),
                "decode_rotary_runtime_input_tensor_count": (
                    decode_result.decode_rotary_runtime_input_tensor_count
                ),
                "decode_runtime_tensor_conversion_count": (
                    decode_result.tensor_conversion_count
                ),
                "acceptance": {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "checks": checks,
                    "failed_checks": failed_checks,
                },
                "ttnn_environment": collect_ttnn_environment(ttnn),
                "message": (
                    "Post-prefill steady decode measured with warmup excluded."
                ),
                "error": None,
            }
    except NoTTNNDeviceError as err:
        report = _decode_steady_failed_report(
            base,
            status="no_device",
            message=str(err),
            ttnn_module=ttnn,
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
        UnsupportedTTNNOp,
    ) as err:
        report = _decode_steady_failed_report(
            base,
            status="runtime_setup_error",
            message=f"{type(err).__name__}: {err}",
            ttnn_module=ttnn,
        )
    except Exception as err:
        report = _decode_steady_failed_report(
            base,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            ttnn_module=ttnn,
        )

    _write_report(profile_path, report)
    return report


def _decode_steady_report_base(
    *,
    profile_path: Path,
    program_root: Path,
    num_layers: int,
    layer_count: int,
    batch_size: int,
    cache_len: int,
    prefill_len: int,
    warmup: int,
    iterations: int,
    device: str,
    device_id: int,
    dtype_seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": "profile",
        "mode": "decode-steady",
        "template": "post_prefill_decode_steady_profile",
        "profile_report": str(profile_path),
        "program_dir": str(program_root),
        "program_num_layers": num_layers,
        "layers": layer_count,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "prefill_len": prefill_len,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "after_prefill": True,
        "warmup": warmup,
        "iterations": iterations,
        "compile_excluded_by_warmup": warmup > 0,
        "timing_scope": {
            "basis": "official_tt_transformers_decode_iteration",
            "includes": [
                "decode runtime metadata preparation",
                "generated model decode_step",
                "device synchronization",
            ],
            "excludes": [
                "parameter materialization and tensorization",
                "prompt prefill",
                "warmup iterations",
                "host token materialization and text decoding",
                "per-op diagnostic synchronization",
            ],
            "runtime_metadata_in_timed_region": True,
            "device_token_handoff": True,
            "per_op_section_profiler_installed": False,
        },
        "official_reference": {
            "id": PROFILE_GENERATE_OFFICIAL_BASELINE_ID,
            "batch_size": PROFILE_GENERATE_OFFICIAL_BATCH_SIZE,
            "tokens_per_second_per_user": (
                PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
            ),
        },
        "official_performance_parity_claimed": False,
        "setup_ms": None,
        "prefill_status": "not_run",
        "prefill_ms": None,
        "warmup_step_ms_samples": [],
        "warmup_total_ms": None,
        "decode_step_ms_samples": [],
        "decode_step_ms_p50": None,
        "decode_step_ms_mean": None,
        "decode_step_ms_min": None,
        "decode_step_ms_max": None,
        "decode_total_ms": None,
        "tokens_per_second_per_user": None,
        "aggregate_tokens_per_second": None,
        "throughput_summary": {
            "status": "not_run",
            "tokens_per_second_per_user": None,
            "aggregate_tokens_per_second": None,
        },
        "host_copy_profile": host_copy_not_run_profile("decode_steady"),
        "section_profile": section_profile_not_run("decode_steady"),
        "ttnn_environment": collect_ttnn_environment(None),
        "error": None,
    }
def _decode_steady_failed_report(
    base: dict[str, Any],
    *,
    status: str,
    message: str,
    ttnn_module: Any | None = None,
) -> dict[str, Any]:
    return {
        **base,
        "status": status,
        "passed": False,
        "dry_run": False,
        "acceptance": {
            "status": "failed",
            "passed": False,
            "failed_checks": ["decode_steady.execution"],
        },
        "message": message,
        "error": {
            "type": status,
            "message": message,
        },
        "ttnn_environment": collect_ttnn_environment(ttnn_module),
    }
