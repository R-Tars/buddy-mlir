from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from ..generate import run_generate
from .decode_depth_sweep import resolve_decode_depths


DEFAULT_GENERATE_DEPTH_TARGETS = (1, 2, 4, 8, 16, "full")


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
    tokenizer_module: Any | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    out_path = Path(out)
    config = json.loads((program_root / "config.json").read_text())
    program_num_layers = int(config["num_layers"])
    resolved_depths = resolve_decode_depths(
        DEFAULT_GENERATE_DEPTH_TARGETS if depths is None else depths,
        program_num_layers=program_num_layers,
    )
    report_root = (
        Path(reports_dir)
        if reports_dir is not None
        else out_path.parent / f"{out_path.stem}_reports"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    model_path_for_generate = Path(model_path) if model_path is not None else None

    records = []
    stop_after_failure = False
    for depth in resolved_depths:
        report_path = report_root / f"generate_depth_{depth}.json"
        if stop_after_failure:
            reason = "blocked by an earlier depth failure"
            records.append(
                {
                    "depth": depth,
                    "status": "skipped",
                    "passed": False,
                    "generate_report": str(report_path),
                    "reason": reason,
                    "failure_diagnostics": {
                        "depth": depth,
                        "status": "skipped",
                        "reason": reason,
                    },
                }
            )
            continue
        try:
            generate = run_generate(
                out=report_path,
                program_dir=program_root,
                model_path=model_path_for_generate,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                max_new_tokens=max_new_tokens,
                layers=depth,
                prefill_len=prefill_len,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                tokenizer_module=tokenizer_module,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
            record = _generate_record(
                depth=depth,
                generate=generate,
                report_path=report_path,
            )
            if record["status"] == "no_device":
                stop_after_failure = True
            elif not record["passed"] and not dry_run:
                stop_after_failure = True
        except Exception as exc:  # pragma: no cover - defensive CLI path.
            record = {
                "depth": depth,
                "status": "fail",
                "passed": False,
                "generate_report": str(report_path),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "failure_diagnostics": {
                    "depth": depth,
                    "status": "exception",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            }
            stop_after_failure = True
        records.append(record)

    status_counts = _field_counts(records, "status")
    prefill_status_counts = _field_counts(records, "prefill_status")
    text_status_counts = _field_counts(records, "generated_text_status")
    model_semantics_counts = _field_counts(records, "model_semantics")
    acceptance = _generate_depth_acceptance(
        records,
        depths=resolved_depths,
        program_num_layers=program_num_layers,
        dry_run=dry_run,
        require_full_depth=require_full_depth,
    )
    if dry_run:
        status = "dry_run"
    elif any(record.get("status") == "no_device" for record in records):
        status = "no_device"
    elif acceptance["passed"]:
        status = "pass"
    else:
        status = "fail"

    report = {
        "schema_version": 1,
        "command": "generate-depth-sweep",
        "status": status,
        "passed": bool(acceptance["passed"]),
        "dry_run": bool(dry_run),
        "require_full_depth": bool(require_full_depth),
        "program_dir": str(program_root),
        "model_path": (
            str(model_path_for_generate) if model_path_for_generate else None
        ),
        "reports_dir": str(report_root),
        "program_num_layers": program_num_layers,
        "depths": resolved_depths,
        "depth_count": len(resolved_depths),
        "max_depth": max(resolved_depths) if resolved_depths else None,
        "covered_full_depth": program_num_layers in resolved_depths,
        "max_new_tokens": int(max_new_tokens),
        "prefill_len": prefill_len,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "prompt_runtime_requested": prompt is not None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "status_counts": status_counts,
        "prefill_status_counts": prefill_status_counts,
        "generated_text_status_counts": text_status_counts,
        "model_semantics_counts": model_semantics_counts,
        "passed_depth_count": sum(
            1 for record in records if record.get("passed") is True
        ),
        "failed_depths": [
            record["depth"]
            for record in records
            if record.get("passed") is False
            and record.get("status") != "skipped"
        ],
        "failed_depth_diagnostics": [
            record["failure_diagnostics"]
            for record in records
            if record.get("passed") is False
            and record.get("status") != "skipped"
            and record.get("failure_diagnostics") is not None
        ],
        "records": records,
        "acceptance": acceptance,
    }
    _write_json(out_path, report)
    return report


def _generate_record(
    *,
    depth: int,
    generate: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    prefill = generate.get("prefill") or {}
    throughput = generate.get("throughput_summary") or {}
    generated_token_ids = generate.get("generated_token_ids") or []
    end_to_end_contract = generate.get("end_to_end_contract")
    if not isinstance(end_to_end_contract, dict):
        end_to_end_contract = {}
    token_counts = [
        len(row)
        for row in generated_token_ids
        if isinstance(row, list)
    ]
    return {
        "depth": depth,
        "status": generate.get("status"),
        "passed": bool(generate.get("passed")),
        "generate_report": str(report_path),
        "layers": generate.get("layers"),
        "batch_size": generate.get("batch_size"),
        "cache_len": generate.get("cache_len"),
        "max_new_tokens": generate.get("max_new_tokens"),
        "generated_token_budget": generate.get("generated_token_budget"),
        "prefill_len": generate.get("prefill_len"),
        "parameter_source": generate.get("parameter_source"),
        "input_source": generate.get("input_source"),
        "runtime_owner": generate.get("runtime_owner"),
        "model_semantics": generate.get("model_semantics"),
        "prefill_status": generate.get("prefill_status"),
        "kv_cache_source": generate.get("kv_cache_source"),
        "decode_loop_runtime_owned": generate.get("decode_loop_runtime_owned"),
        "generate_runtime_owned": generate.get("generate_runtime_owned"),
        "end_to_end_contract_status": end_to_end_contract.get("status"),
        "end_to_end_failed_checks": end_to_end_contract.get(
            "failed_checks",
            [],
        ),
        "generated_text_status": generate.get("generated_text_status"),
        "generated_text_source": generate.get("generated_text_source"),
        "generated_text": generate.get("generated_text"),
        "generated_token_count_by_user": token_counts,
        "prefill_cache_population": prefill.get("cache_population", []),
        "prefill_first_token": prefill.get("first_token"),
        "runtime_context": generate.get("runtime_context"),
        "parameter_tensorization_count_per_generate": generate.get(
            "parameter_tensorization_count_per_generate"
        ),
        "parameter_tensorization_count_per_decode_step": generate.get(
            "parameter_tensorization_count_per_decode_step"
        ),
        "kv_cache_reinitialized_per_step": generate.get(
            "kv_cache_reinitialized_per_step"
        ),
        "throughput_status": throughput.get("status"),
        "tokens_per_second_per_user": throughput.get(
            "tokens_per_second_per_user"
        ),
        "aggregate_tokens_per_second": throughput.get(
            "aggregate_tokens_per_second"
        ),
        "error": generate.get("error"),
        "failure_diagnostics": _generate_failure_diagnostics(
            depth=depth,
            generate=generate,
        ),
    }


def _generate_failure_diagnostics(
    *,
    depth: int,
    generate: dict[str, Any],
) -> dict[str, Any] | None:
    if bool(generate.get("passed")):
        return None
    prefill = generate.get("prefill")
    if not isinstance(prefill, dict):
        prefill = {}
    contract = generate.get("end_to_end_contract")
    if not isinstance(contract, dict):
        contract = {}
    step_reports = generate.get("step_reports")
    if not isinstance(step_reports, list):
        step_reports = []
    failed_step = next(
        (
            step
            for step in step_reports
            if isinstance(step, dict) and not bool(step.get("passed"))
        ),
        None,
    )
    return {
        "depth": depth,
        "status": generate.get("status"),
        "error": generate.get("error"),
        "model_semantics": generate.get("model_semantics"),
        "layers": generate.get("layers"),
        "layout": generate.get("layout"),
        "end_to_end_contract": {
            "status": contract.get("status"),
            "failed_checks": contract.get("failed_checks", []),
        },
        "prefill": {
            "status": generate.get("prefill_status"),
            "cache_population": _cache_population_diagnostics(
                prefill.get("cache_population", [])
            ),
        },
        "decode": {
            "decode_loop_runtime_owned": generate.get(
                "decode_loop_runtime_owned"
            ),
            "step_count": len(step_reports),
            "failed_step": _failed_step_diagnostics(failed_step),
        },
    }


def _cache_population_diagnostics(cache_population: Any) -> list[dict[str, Any]]:
    if not isinstance(cache_population, list):
        return []
    diagnostics = []
    for entry in cache_population:
        if not isinstance(entry, dict):
            continue
        diagnostics.append(
            {
                "layer_id": entry.get("layer_id"),
                "status": entry.get("status"),
                "write_policy": entry.get("write_policy"),
                "update_shape_layout": entry.get("update_shape_layout"),
                "key_update_shape": entry.get("key_update_shape"),
                "value_update_shape": entry.get("value_update_shape"),
                "key_cache_shape": entry.get("key_cache_shape"),
                "value_cache_shape": entry.get("value_cache_shape"),
                "page_table_shape": entry.get("page_table_shape"),
                "planned_user_count": entry.get("planned_user_count"),
                "filled_user_count": entry.get("filled_user_count"),
            }
        )
    return diagnostics


def _failed_step_diagnostics(step: Any) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    reference = step.get("reference")
    if not isinstance(reference, dict):
        reference = {}
    return {
        "step_index": step.get("step_index"),
        "status": step.get("status"),
        "cache_position_value": step.get("cache_position_value"),
        "input_shapes": step.get("input_shapes"),
        "output_shapes": step.get("output_shapes"),
        "decode_runtime_state": step.get("decode_runtime_state"),
        "rotary_runtime_state": step.get("rotary_runtime_state"),
        "reference_status": reference.get("status"),
        "reference_failed_checks": reference.get("failed_checks", []),
        "observed_ops": reference.get("observed_ops"),
        "expected_ops": reference.get("expected_ops"),
        "error": step.get("error"),
    }


def _generate_depth_acceptance(
    records: list[dict[str, Any]],
    *,
    depths: list[int],
    program_num_layers: int,
    dry_run: bool,
    require_full_depth: bool,
) -> dict[str, Any]:
    checks = []
    record_by_depth = {int(record["depth"]): record for record in records}
    required_depths = [
        depth
        for depth in (1, 2, 4)
        if depth in depths and depth <= program_num_layers
    ]
    for depth in required_depths:
        record = record_by_depth.get(depth, {})
        checks.append(
            {
                "name": f"generate_depth_sweep.depth_{depth}_passed",
                "passed": bool(record.get("passed")),
                "status": record.get("status"),
            }
        )
    if require_full_depth:
        full_record = record_by_depth.get(program_num_layers, {})
        checks.append(
            {
                "name": "generate_depth_sweep.full_depth_passed",
                "passed": bool(full_record.get("passed")),
                "status": full_record.get("status"),
                "depth": program_num_layers,
            }
        )
    evidence_records = [
        record
        for record in records
        if record.get("status") != "skipped"
    ]
    checks.append(
        {
            "name": "generate_depth_sweep.prefill_evidence",
            "passed": all(
                record.get("prefill_status") in {"passed", "dry_run"}
                for record in evidence_records
            ),
        }
    )
    checks.append(
        {
            "name": "generate_depth_sweep.text_evidence",
            "passed": all(
                record.get("generated_text_status")
                in {"fallback", "decoded", "placeholder", "not_run"}
                for record in evidence_records
            ),
        }
    )
    checks.append(
        {
            "name": "generate_depth_sweep.report_files",
            "passed": all(
                record.get("generate_report") is not None
                for record in records
            ),
        }
    )
    checks.append(
        {
            "name": "generate_depth_sweep.persistent_runtime_context",
            "passed": all(
                record.get("parameter_tensorization_count_per_generate") == 1
                and record.get("parameter_tensorization_count_per_decode_step") == 0
                and record.get("kv_cache_reinitialized_per_step") is False
                for record in evidence_records
            ),
        }
    )
    failed_evidence_records = [
        record
        for record in records
        if record.get("status") != "skipped"
        and record.get("passed") is False
    ]
    checks.append(
        {
            "name": "generate_depth_sweep.failure_diagnostics",
            "passed": all(
                isinstance(record.get("failure_diagnostics"), dict)
                for record in failed_evidence_records
            ),
            "observed": [
                {
                    "depth": record.get("depth"),
                    "status": record.get("status"),
                    "has_failure_diagnostics": isinstance(
                        record.get("failure_diagnostics"),
                        dict,
                    ),
                }
                for record in failed_evidence_records
            ],
        }
    )
    if dry_run:
        checks.append(
            {
                "name": "generate_depth_sweep.dry_run",
                "passed": all(record.get("status") == "dry_run" for record in records),
            }
        )
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "required_depths": required_depths,
        "require_full_depth": bool(require_full_depth),
        "failed_checks": failed,
        "checks": checks,
    }


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
