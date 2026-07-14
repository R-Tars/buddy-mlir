from __future__ import annotations

import importlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..runtime_environment import collect_ttnn_environment
from .reports import write_report


OFFICIAL_EXTERNAL_TTFT_MS = 57.0
OFFICIAL_RELEASE_LOCAL_TTFT_MS = 52.33
OFFICIAL_MATCHED_LOCAL_TTFT_MS = 126.30
OFFICIAL_RELEASE_TAG = "v0.64.0-dev20251030"
OFFICIAL_RELEASE_COMMIT = "b76035fbdac81d8f9974976471dc60fc005e1bfb"
OFFICIAL_MATCHED_COMMIT = "61e690c25202111b52cbc1fbc9148b6524070c6f"
OFFICIAL_MODELS_URL = "https://github.com/tenstorrent/tt-metal/tree/main/models"


def run_profile_prefill_steady(
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
    device: str = "p150a",
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    warmup: int = 1,
    iterations: int = 3,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
    prefill_execution_mode: str | None = None,
) -> dict[str, Any]:
    from .device import maybe_generate_device
    from .plans import decode_step_plan, prefill_plan as build_prefill_plan
    from .prefill import run_prefill_prompt
    from .prefill_trace import resolve_prefill_execution_mode
    from .session import build_runtime_session
    from .structural import loop_generated_token_ids
    from .tokenizer import (
        load_prompt_batch,
        tokenize_prompts_for_prefill,
    )

    report_path = Path(out)
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers or config["num_layers"])
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config.get("seq_len", 1)
    )
    warmup = int(warmup)
    iterations = int(iterations)
    execution_mode = resolve_prefill_execution_mode(prefill_execution_mode)
    if layer_count <= 0 or layer_count > int(config["num_layers"]):
        raise ValueError(
            f"layers must be in [1, {int(config['num_layers'])}]"
        )
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    base: dict[str, Any] = {
        "schema_version": 1,
        "command": "profile",
        "mode": "prefill-steady",
        "status": "running",
        "passed": False,
        "program_dir": str(program_root.resolve()),
        "buddy_commit": _git_head(Path(__file__).resolve().parents[4]),
        "layers": layer_count,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "device": device,
        "device_id": int(device_id),
        "dtype_seed": dtype_seed,
        "warmup": warmup,
        "iterations": iterations,
        "prefill_execution_mode": execution_mode,
        "metric_contract": {
            "batch_prefill_latency_ms": "one full Buddy batch32 prefill",
            "average_ttft_ms_per_user": (
                "batch_prefill_latency_ms / batch_size, matching the official "
                "simple_text_demo formula"
            ),
            "compile_excluded": True,
            "host_token_materialization_excluded": True,
        },
        "official_references": {
            "release_local_ttft_ms": OFFICIAL_RELEASE_LOCAL_TTFT_MS,
            "release_tag": OFFICIAL_RELEASE_TAG,
            "release_commit": OFFICIAL_RELEASE_COMMIT,
            "release_local_source": (
                "Goal 0 corresponding-release three-run median, parsed from "
                "official demo logs"
            ),
            "release_comparison_scope": "corresponding-release-reference",
            "matched_local_ttft_ms": OFFICIAL_MATCHED_LOCAL_TTFT_MS,
            "matched_tt_metal_commit": OFFICIAL_MATCHED_COMMIT,
            "matched_local_source": (
                "Goal 0 current-commit official-greedy three-run median"
            ),
            "matched_comparison_scope": "same-tt-metal-commit-local",
            "external_ttft_ms": OFFICIAL_EXTERNAL_TTFT_MS,
            "external_release_tag": OFFICIAL_RELEASE_TAG,
            "external_source": OFFICIAL_MODELS_URL,
            "provenance_note": (
                "The published 57 ms belongs to the corresponding release; "
                "it is not a main-branch or same-commit measurement."
            ),
        },
        "warmup_batch_latency_ms_samples": [],
        "batch_latency_ms_samples": [],
        "average_ttft_ms_per_user_samples": [],
        "token_ids_by_iteration": [],
        "error": None,
    }
    write_report(report_path, base)
    if dry_run:
        base.update(
            {
                "status": "dry_run",
                "passed": True,
                "dry_run": True,
                "acceptance": {
                    "status": "dry_run",
                    "passed": True,
                    "failed_checks": [],
                },
            }
        )
        write_report(report_path, base)
        return base
    if execution_mode != "eager":
        base.update(
            {
                "status": "unsupported_prefill_trace",
                "error": (
                    "steady prefill profiling requires eager mode because the "
                    "current TTNN prefill residual add synchronizes during trace "
                    "capture"
                ),
            }
        )
        write_report(report_path, base)
        return base
    if model_path is None or (prompt is None and input_prompts is None):
        base.update(
            {
                "status": "missing_runtime_input",
                "error": "model_path and prompt or input_prompts are required",
            }
        )
        write_report(report_path, base)
        return base

    try:
        ttnn = ttnn_module or importlib.import_module("ttnn")
        torch = torch_module or importlib.import_module("torch")
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
        prompt_batch = load_prompt_batch(
            prompt=prompt,
            input_prompts=input_prompts,
            batch_size=batch_size,
        )
        tokenization = tokenize_prompts_for_prefill(
            prompt_batch=prompt_batch,
            prefill_len=prefill_len,
            tokenizer_path=tokenizer_path or model_path,
            vocab_size=prefill_plan.get("vocab_size"),
            tokenizer_module=tokenizer_module,
            instruct=instruct,
            reject_truncation=input_prompts is not None,
            padding_token_id=0 if input_prompts is not None else None,
        )
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
                prompt=prompt_batch.prompts[0],
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
                config=config,
                layer_count=layer_count,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_len=prefill_len,
                prefill_tokenization=tokenization,
            )
            ops = getattr(session.generated_model, "ops", None)
            enable_recording = getattr(ops, "enable_recording", None)
            if callable(enable_recording):
                enable_recording()
            results = []
            captured_ops: list[str] = []
            for index in range(warmup + iterations):
                op_log = getattr(ops, "op_log", None)
                op_cursor = len(op_log) if isinstance(op_log, list) else None
                result = run_prefill_prompt(
                    context=session.context,
                    ttnn=ttnn,
                    device=ttnn_device,
                    prefill_plan=prefill_plan,
                    layer_count=layer_count,
                    execution_mode="eager",
                )
                if index < warmup:
                    base["warmup_batch_latency_ms_samples"].append(
                        float(result.latency_ms)
                    )
                else:
                    base["batch_latency_ms_samples"].append(
                        float(result.latency_ms)
                    )
                    results.append(result)
                    if not captured_ops and op_cursor is not None:
                        captured_ops = [
                            str(value) for value in op_log[op_cursor:]
                        ]
                write_report(report_path, base)

            token_ids = []
            structural_passed = True
            for result in results:
                materialized = loop_generated_token_ids(
                    token=result.prefill_token,
                    ttnn=ttnn,
                    batch_size=batch_size,
                )
                token_ids.append(materialized["token_ids_by_user"])
                structural_passed = structural_passed and bool(
                    result.reference.get("passed")
                )
            base["ttnn_environment"] = collect_ttnn_environment(ttnn)

        samples = [float(value) for value in base["batch_latency_ms_samples"]]
        per_user = [value / batch_size for value in samples]
        mean = statistics.fmean(samples)
        per_user_mean = mean / batch_size
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        cv = stdev / mean if mean else None
        observed_tt_metal_commit = (base.get("ttnn_environment") or {}).get(
            "tt_metal_git_commit"
        )
        token_agreement = bool(token_ids) and all(
            value == token_ids[0] for value in token_ids[1:]
        )
        operation_counts = Counter(captured_ops)
        cache_fill_calls = sum(
            count
            for name, count in operation_counts.items()
            if name in ("fill_cache.k", "fill_cache.v")
        )
        base.update(
            {
                "status": "profiled",
                "batch_latency_ms_mean": mean,
                "batch_latency_ms_p50": statistics.median(samples),
                "batch_latency_ms_p90": _percentile(samples, 0.90),
                "batch_latency_ms_min": min(samples),
                "batch_latency_ms_max": max(samples),
                "batch_latency_ms_stdev": stdev,
                "coefficient_of_variation": cv,
                "average_ttft_ms_per_user_samples": per_user,
                "average_ttft_ms_per_user_mean": per_user_mean,
                "average_ttft_ms_per_user_p50": statistics.median(per_user),
                "ratio_of_release_local_official": (
                    OFFICIAL_RELEASE_LOCAL_TTFT_MS / per_user_mean
                ),
                "ratio_of_matched_local_official": (
                    OFFICIAL_MATCHED_LOCAL_TTFT_MS / per_user_mean
                ),
                "ratio_of_external_reference": (
                    OFFICIAL_EXTERNAL_TTFT_MS / per_user_mean
                ),
                "token_ids_by_iteration": token_ids,
                "token_agreement": token_agreement,
                "structural_reference_passed": structural_passed,
                "execution_graph_summary": {
                    "source": "TTNNCompatOps runtime instrumentation",
                    "operation_count": len(captured_ops),
                    "operation_counts": dict(sorted(operation_counts.items())),
                    "cache_fill_call_count": cache_fill_calls,
                    "expected_cache_fill_call_count": (
                        2 * batch_size * layer_count
                    ),
                    "cache_fill_contract": (
                        "paged_fill_cache scalar batch_idx per user"
                    ),
                    "official_release_contract": (
                        "sequential user prefill with two cache fills per layer"
                    ),
                    "batch_aware_paged_fill_cache_available": False,
                    "batch_aware_api_evidence": (
                        "TTNN paged_fill_cache batch_idx_tensor selects one "
                        "scalar user; current and release official models loop"
                    ),
                    "trace_candidate_status": "blocked_by_ttnn_event_sync",
                },
                "milestones": _milestones(per_user_mean),
                "reference_eligibility": {
                    "same_commit_reference": (
                        observed_tt_metal_commit == OFFICIAL_MATCHED_COMMIT
                    ),
                    "corresponding_release_reference": True,
                    "published_release_reference": True,
                },
            }
        )
        checks = [
            {
                "name": "prefill_steady.sample_count",
                "passed": len(samples) == iterations,
            },
            {
                "name": "prefill_steady.token_agreement",
                "passed": token_agreement,
            },
            {
                "name": "prefill_steady.structural_reference",
                "passed": structural_passed,
            },
            {
                "name": "prefill_steady.cv_le_1_5_percent",
                "passed": cv is not None and cv * 100.0 <= 1.5,
            },
            {
                "name": "prefill_steady.release_p5",
                "passed": bool(
                    base["milestones"]["P5_ge_95_percent_release_local"]
                ),
            },
        ]
        failed = [check["name"] for check in checks if not check["passed"]]
        base["passed"] = not failed
        base["acceptance"] = {
            "status": "passed" if not failed else "failed",
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
        }
    except Exception as error:
        base.update(
            {
                "status": "failed",
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    write_report(report_path, base)
    return base


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _milestones(ttft_ms: float) -> dict[str, bool]:
    return {
        "P0_functional": True,
        "P1_le_500_ms": ttft_ms <= 500.0,
        "P2_le_200_ms": ttft_ms <= 200.0,
        "P3_le_100_ms": ttft_ms <= 100.0,
        "P4_ge_80_percent_matched_local": (
            OFFICIAL_MATCHED_LOCAL_TTFT_MS / ttft_ms >= 0.80
        ),
        "P5_ge_95_percent_matched_local": (
            OFFICIAL_MATCHED_LOCAL_TTFT_MS / ttft_ms >= 0.95
        ),
        "P4_ge_80_percent_release_local": (
            OFFICIAL_RELEASE_LOCAL_TTFT_MS / ttft_ms >= 0.80
        ),
        "P5_ge_95_percent_release_local": (
            OFFICIAL_RELEASE_LOCAL_TTFT_MS / ttft_ms >= 0.95
        ),
    }


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
