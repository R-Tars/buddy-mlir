from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..runtime.generate import run_generate
from .depth_sweep_support import (
    depth_summary,
    ensure_report_written,
    exception_details,
    field_counts,
    load_program_num_layers,
    project_fields,
    resolve_depths,
    resolve_output_root,
    result_summary,
    run_depth_sequence,
    run_isolated_command,
    should_isolate_depth_steps,
    sweep_status,
    write_json,
)


DEFAULT_GENERATE_DEPTH_TARGETS = (1, 2, 4, 8, 16, "full")
GENERATE_RECORD_FIELDS = """
layers batch_size cache_len max_new_tokens generated_token_budget prefill_len
parameter_source input_source runtime_owner model_semantics prefill_status
kv_cache_source decode_loop_runtime_owned generate_runtime_owned generated_text_status
generated_text_source generated_text runtime_context
parameter_tensorization_count_per_generate parameter_tensorization_count_per_decode_step
kv_cache_reinitialized_per_step error isolated_subprocess
""".split()
CACHE_DIAGNOSTIC_FIELDS = """
layer_id status write_policy update_shape_layout key_update_shape value_update_shape
key_cache_shape value_cache_shape page_table_shape planned_user_count filled_user_count
""".split()
FAILED_STEP_FIELDS = """
step_index status cache_position_value input_shapes output_shapes decode_runtime_state
rotary_runtime_state error
""".split()


def run_generate_depth_sweep(
    *,
    program_dir: str | Path,
    out: str | Path,
    depths: str | list[int | str] | tuple[int | str, ...] | None = None,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
    max_new_tokens: int = 4,
    prefill_len: int | None = None,
    batch_size: int | None = None,
    cache_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    require_full_depth: bool = False,
    isolate_depth_steps: bool = False,
    tokenizer_module: Any | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    program_root, out_path = Path(program_dir), Path(out)
    program_num_layers = load_program_num_layers(program_root)
    resolved_depths = resolve_depths(
        depths, program_num_layers=program_num_layers,
        defaults=DEFAULT_GENERATE_DEPTH_TARGETS,
    )
    report_root = resolve_output_root(out_path, reports_dir, suffix="reports")
    model_root = Path(model_path) if model_path is not None else None
    isolate = should_isolate_depth_steps(
        isolate_depth_steps,
        dry_run=dry_run,
        injected_modules=(tokenizer_module, ttnn_module, torch_module),
    )

    def execute(depth: int, report_path: Path) -> dict[str, Any]:
        if isolate:
            return _run_isolated_generate_depth(
                report_path=report_path, program_root=program_root,
                model_path=model_root, prompt=prompt, tokenizer_path=tokenizer_path,
                max_new_tokens=max_new_tokens, depth=depth, prefill_len=prefill_len,
                device=device, device_id=device_id, batch_size=batch_size,
                cache_len=cache_len, dtype_seed=dtype_seed,
            )
        generated = run_generate(
            out=report_path, program_dir=program_root, model_path=model_root,
            prompt=prompt, tokenizer_path=tokenizer_path,
            max_new_tokens=max_new_tokens, layers=depth, prefill_len=prefill_len,
            device=device, device_id=device_id, batch_size=batch_size,
            cache_len=cache_len, dtype_seed=dtype_seed, dry_run=dry_run,
            tokenizer_module=tokenizer_module, ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        ensure_report_written(report_path, generated)
        return generated

    def base_report(depth: int, report_path: Path, status: str) -> dict[str, Any]:
        return _base_depth_generate_report(
            depth=depth, report_path=report_path, program_root=program_root,
            status=status, max_new_tokens=max_new_tokens, prefill_len=prefill_len,
            batch_size=batch_size, cache_len=cache_len, device=device,
            device_id=device_id, dtype_seed=dtype_seed,
        )

    def skipped(depth: int, report_path: Path, reason: str) -> dict[str, Any]:
        generated = base_report(depth, report_path, "skipped")
        generated.update(
            passed=False, reason=reason, error=reason,
            failure_diagnostics={"depth": depth, "status": "skipped", "reason": reason},
        )
        write_json(report_path, generated)
        return _generate_record(depth=depth, generate=generated, report_path=report_path)

    def failed(depth: int, report_path: Path, exception: Exception) -> dict[str, Any]:
        error = exception_details(exception)
        generated = base_report(depth, report_path, "fail")
        generated.update(
            passed=False, error=error,
            failure_diagnostics={
                "depth": depth, "status": "exception",
                "error": project_fields(error, ("type", "message")),
            },
        )
        write_json(report_path, generated)
        return _generate_record(depth=depth, generate=generated, report_path=report_path)

    records = run_depth_sequence(
        resolved_depths,
        report_path_for=lambda depth: report_root / f"generate_depth_{depth}.json",
        execute=execute,
        make_record=lambda depth, payload, path: _generate_record(
            depth=depth, generate=payload, report_path=path
        ),
        make_skipped=skipped,
        make_exception=failed,
        dry_run=dry_run,
    )
    acceptance = _generate_depth_acceptance(
        records, depths=resolved_depths, program_num_layers=program_num_layers,
        dry_run=dry_run, require_full_depth=require_full_depth,
    )
    report = {
        "schema_version": 1,
        "command": "generate-depth-sweep",
        "status": sweep_status(records, dry_run=dry_run, accepted=acceptance["passed"]),
        "passed": bool(acceptance["passed"]),
        "dry_run": bool(dry_run),
        "require_full_depth": bool(require_full_depth),
        "isolate_depth_steps": isolate,
        "program_dir": str(program_root),
        "model_path": str(model_root) if model_root else None,
        "reports_dir": str(report_root),
        **depth_summary(resolved_depths, program_num_layers),
        "max_new_tokens": int(max_new_tokens), "prefill_len": prefill_len,
        "batch_size": batch_size, "cache_len": cache_len,
        "device": device, "device_id": device_id, "dtype_seed": dtype_seed,
        "prompt_runtime_requested": prompt is not None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "status_counts": field_counts(records, "status"),
        "prefill_status_counts": field_counts(records, "prefill_status"),
        "generated_text_status_counts": field_counts(records, "generated_text_status"),
        "model_semantics_counts": field_counts(records, "model_semantics"),
        **result_summary(records),
        "failed_depth_diagnostics": [
            record["failure_diagnostics"] for record in records
            if record.get("passed") is False and record.get("status") != "skipped"
            and record.get("failure_diagnostics") is not None
        ],
        "records": records,
        "acceptance": acceptance,
    }
    write_json(out_path, report)
    return report


def _generate_record(
    *, depth: int, generate: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    prefill, throughput = generate.get("prefill") or {}, generate.get("throughput_summary") or {}
    token_ids = generate.get("generated_token_ids") or []
    contract = generate.get("end_to_end_contract")
    contract = contract if isinstance(contract, dict) else {}
    return {
        "depth": depth, "status": generate.get("status"),
        "passed": bool(generate.get("passed")), "generate_report": str(report_path),
        "generate_report_exists": report_path.is_file(), "reason": generate.get("reason"),
        **project_fields(generate, GENERATE_RECORD_FIELDS),
        "end_to_end_contract_status": contract.get("status"),
        "end_to_end_failed_checks": contract.get("failed_checks", []),
        "generated_token_count_by_user": [len(row) for row in token_ids if isinstance(row, list)],
        "prefill_cache_population": prefill.get("cache_population", []),
        "prefill_first_token": prefill.get("first_token"),
        "throughput_status": throughput.get("status"),
        "tokens_per_second_per_user": throughput.get("tokens_per_second_per_user"),
        "aggregate_tokens_per_second": throughput.get("aggregate_tokens_per_second"),
        "failure_diagnostics": _generate_failure_diagnostics(
            depth=depth, generate=generate, report_path=report_path
        ),
    }


def _run_isolated_generate_depth(
    *, report_path: Path, program_root: Path, model_path: Path | None,
    prompt: str | None, tokenizer_path: str | Path | None, max_new_tokens: int,
    depth: int, prefill_len: int | None, device: str, device_id: int,
    batch_size: int | None, cache_len: int | None, dtype_seed: str,
) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "generate", "--program-dir", str(program_root), "--max-new-tokens",
        str(max_new_tokens), "--layers", str(depth), "--device", device,
        "--device-id", str(device_id), "--dtype-seed", dtype_seed,
        "--out", str(report_path),
    ]
    for flag, value in (("--model-path", model_path), ("--prompt", prompt),
                        ("--tokenizer-path", tokenizer_path), ("--prefill-len", prefill_len),
                        ("--batch-size", batch_size), ("--cache-len", cache_len)):
        if value is not None:
            command.extend([flag, str(value)])

    def missing() -> dict[str, Any]:
        generated = _base_depth_generate_report(
            depth=depth, report_path=report_path, program_root=program_root,
            status="fail", max_new_tokens=max_new_tokens, prefill_len=prefill_len,
            batch_size=batch_size, cache_len=cache_len, device=device,
            device_id=device_id, dtype_seed=dtype_seed,
        )
        generated["error"] = "isolated generate depth step did not write a report"
        return generated

    return run_isolated_command(command, report_path=report_path, missing_report=missing)


def _generate_failure_diagnostics(
    *, depth: int, generate: dict[str, Any], report_path: Path
) -> dict[str, Any] | None:
    if bool(generate.get("passed")):
        return None
    prefill = generate.get("prefill") if isinstance(generate.get("prefill"), dict) else {}
    contract = generate.get("end_to_end_contract")
    contract = contract if isinstance(contract, dict) else {}
    steps = generate.get("step_reports")
    steps = steps if isinstance(steps, list) else []
    failed_step = next((step for step in steps if isinstance(step, dict) and not bool(step.get("passed"))), None)
    return {
        "depth": depth, "status": generate.get("status"),
        "generate_report": str(report_path), "error": generate.get("error"),
        "model_semantics": generate.get("model_semantics"), "layers": generate.get("layers"),
        "layout": generate.get("layout"),
        "end_to_end_contract": {
            "status": contract.get("status"), "failed_checks": contract.get("failed_checks", [])
        },
        "prefill": {
            "status": generate.get("prefill_status"),
            "cache_population": _cache_population_diagnostics(prefill.get("cache_population", [])),
        },
        "decode": {
            "decode_loop_runtime_owned": generate.get("decode_loop_runtime_owned"),
            "step_count": len(steps), "failed_step": _failed_step_diagnostics(failed_step),
        },
    }


def _cache_population_diagnostics(cache_population: Any) -> list[dict[str, Any]]:
    if not isinstance(cache_population, list):
        return []
    return [project_fields(entry, CACHE_DIAGNOSTIC_FIELDS) for entry in cache_population if isinstance(entry, dict)]


def _failed_step_diagnostics(step: Any) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    reference = step.get("reference") if isinstance(step.get("reference"), dict) else {}
    return {
        **project_fields(step, FAILED_STEP_FIELDS),
        "reference_status": reference.get("status"),
        "reference_failed_checks": reference.get("failed_checks", []),
        "observed_ops": reference.get("observed_ops"),
        "expected_ops": reference.get("expected_ops"),
    }


def _generate_depth_acceptance(
    records: list[dict[str, Any]], *, depths: list[int], program_num_layers: int,
    dry_run: bool, require_full_depth: bool,
) -> dict[str, Any]:
    checks, by_depth = [], {int(record["depth"]): record for record in records}
    required_depths = [depth for depth in (1, 2, 4) if depth in depths and depth <= program_num_layers]
    for depth in required_depths:
        record = by_depth.get(depth, {})
        checks.append({"name": f"generate_depth_sweep.depth_{depth}_passed",
                       "passed": bool(record.get("passed")), "status": record.get("status")})
    if require_full_depth:
        record = by_depth.get(program_num_layers, {})
        checks.append({"name": "generate_depth_sweep.full_depth_passed",
                       "passed": bool(record.get("passed")), "status": record.get("status"),
                       "depth": program_num_layers})
    evidence = [record for record in records if record.get("status") != "skipped"]
    checks.extend([
        {"name": "generate_depth_sweep.prefill_evidence",
         "passed": all(record.get("prefill_status") in {"passed", "dry_run"} for record in evidence)},
        {"name": "generate_depth_sweep.text_evidence",
         "passed": all(record.get("generated_text_status") in {"fallback", "decoded", "placeholder", "not_run"} for record in evidence)},
        {"name": "generate_depth_sweep.report_files",
         "passed": all(record.get("generate_report") is not None and bool(record.get("generate_report_exists")) for record in records),
         "observed": [{"depth": record.get("depth"), "path": record.get("generate_report"),
                       "exists": bool(record.get("generate_report_exists"))} for record in records]},
        {"name": "generate_depth_sweep.persistent_runtime_context",
         "passed": all(record.get("parameter_tensorization_count_per_generate") == 1
                       and record.get("parameter_tensorization_count_per_decode_step") == 0
                       and record.get("kv_cache_reinitialized_per_step") is False for record in evidence)},
    ])
    failed_records = [record for record in evidence if record.get("passed") is False]
    checks.append({
        "name": "generate_depth_sweep.failure_diagnostics",
        "passed": all(isinstance(record.get("failure_diagnostics"), dict) for record in failed_records),
        "observed": [{"depth": record.get("depth"), "status": record.get("status"),
                      "has_failure_diagnostics": isinstance(record.get("failure_diagnostics"), dict)}
                     for record in failed_records],
    })
    if dry_run:
        checks.append({"name": "generate_depth_sweep.dry_run",
                       "passed": all(record.get("status") == "dry_run" for record in records)})
    failed = [item["name"] for item in checks if not item["passed"]]
    return {"status": "passed" if not failed else "failed", "passed": not failed,
            "required_depths": required_depths, "require_full_depth": bool(require_full_depth),
            "failed_checks": failed, "checks": checks}


def _base_depth_generate_report(
    *, depth: int, report_path: Path, program_root: Path, status: str,
    max_new_tokens: int, prefill_len: int | None, batch_size: int | None,
    cache_len: int | None, device: str, device_id: int, dtype_seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1, "command": "generate", "mode": "generate",
        "status": status, "passed": False, "program_dir": str(program_root),
        "report": str(report_path), "layers": depth,
        "max_new_tokens": int(max_new_tokens), "prefill_len": prefill_len,
        "batch_size": batch_size, "cache_len": cache_len, "device": device,
        "device_id": device_id, "dtype_seed": dtype_seed,
        "generated_token_ids": [], "generated_text": "", "generated_text_by_user": [],
        "generated_text_status": "not_run", "prefill_status": status,
        "kv_cache_source": "prefill", "model_semantics": "prompt_conditioned_prefill_decode",
        "decode_loop_runtime_owned": False, "generate_runtime_owned": False,
        "step_reports": [], "prefill": {"status": status, "cache_population": []},
        "throughput_summary": {},
    }
