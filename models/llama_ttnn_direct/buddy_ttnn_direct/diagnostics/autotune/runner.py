from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ...autotune.measurement import (
    build_candidate_config,
    candidate_fingerprint,
    prompt_corpus_sha256,
    resolve_runtime_commit,
)
from ...autotune.schema import (
    ContractViolation,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
)
from ...compiler.tuning import (
    OFFICIAL_LINEAR_OUTPUTS,
    OFFICIAL_PROGRAM_CONFIG,
)
from ...semantic.importer_hf_llama import import_hf_llama
from ...templates.registry import load_template_config
from .candidate import (
    ProfileRunner,
    measure_state as _measure_state,
    structured_candidate_state as _structured_candidate_state,
    write_json as _write_report,
)
from .selection import (
    AUTOTUNE_LEVELS,
    AUTOTUNE_METRIC,
    confirmation_promotion_decision as _confirmation_promotion_decision,
    level_values as _level_values,
    select_winner as _select_winner,
    selection_summary as _selection_summary,
    winner_summary as _winner_summary,
)


def run_layered_autotune(
    *,
    model_path: str | Path,
    config_path: str | Path,
    out: str | Path,
    prompt: str | None,
    tokenizer_path: str | Path | None = None,
    candidates_dir: str | Path | None = None,
    layers: int | None = None,
    prefill_len: int | None = None,
    batch_size: int | None = None,
    cache_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    warmup: int = 5,
    iterations: int = 10,
    confirm_warmup: int = 5,
    confirm_iterations: int = 50,
    min_relative_improvement: float = 0.01,
    dry_run: bool = False,
    resume: bool = True,
    profile_runner: ProfileRunner | None = None,
) -> dict[str, Any]:
    """Tune one hardware axis at a time with post-prefill steady decode."""

    model_root = Path(model_path)
    seed_path = Path(config_path)
    report_path = Path(out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_root = (
        Path(candidates_dir)
        if candidates_dir is not None
        else report_path.parent / f"{report_path.stem}_candidates"
    )
    state_root.mkdir(parents=True, exist_ok=True)

    seed = load_template_config(seed_path)
    if not seed.get("official_config_profile"):
        raise ContractViolation(
            "layered autotune requires an imported official_config_profile"
        )
    if not dry_run and not prompt:
        raise ValueError("prompt is required for hardware autotune")
    _validate_measurement_counts(
        warmup=warmup,
        iterations=iterations,
        confirm_warmup=confirm_warmup,
        confirm_iterations=confirm_iterations,
    )
    if float(min_relative_improvement) < 0.0:
        raise ValueError("min_relative_improvement must be non-negative")
    if dtype_seed != "bf16":
        raise ContractViolation(
            "autotune runtime dtype seed is frozen to 'bf16'; "
            f"observed {dtype_seed!r}"
        )

    graph = import_hf_llama(
        model_root,
        mode="decode",
        batch_size=int(seed["batch_size"]),
        seq_len=int(seed["decode_seq_len"]),
        max_cache_len=int(seed["max_cache_len"]),
        generation_mode=(
            "greedy"
            if seed["generation_template"] == "device_argmax_greedy"
            else "sampling"
        ),
    )
    layer_count = graph.num_layers if layers is None else int(layers)
    if layer_count <= 0 or layer_count > graph.num_layers:
        raise ValueError(f"layers must be in [1, {graph.num_layers}]")

    selected_state = {
        "lm_head_split_count": int(seed["lm_head_split_count"]),
        "memory_layout": OFFICIAL_LINEAR_OUTPUTS,
        "program_config": OFFICIAL_PROGRAM_CONFIG,
    }
    root_incumbent_state = copy.deepcopy(selected_state)
    precision_contract = PrecisionContract.from_template_config(seed)
    precision_hash = precision_contract.hash
    execution_contract = ExecutionContract.from_template_config(
        seed,
        prompt_corpus_sha256=prompt_corpus_sha256(prompt),
    )
    runtime_commit = resolve_runtime_commit()
    effective_prefill_len = int(
        seed["prefill_seq_len"] if prefill_len is None else prefill_len
    )
    effective_batch_size = int(seed["batch_size"] if batch_size is None else batch_size)
    effective_cache_len = int(seed["max_cache_len"] if cache_len is None else cache_len)
    target = {
        "device": device,
        "batch_size": effective_batch_size,
        "decode_seq_len": int(seed["decode_seq_len"]),
        "prefill_len": effective_prefill_len,
        "cache_len": effective_cache_len,
        "page_block_size": 32,
        "runtime_dtype_seed": dtype_seed,
    }
    invocation = {
        "model_path": str(model_root.resolve()),
        "config_path": str(seed_path.resolve()),
        "prompt_sha256": execution_contract.prompt_corpus_sha256,
        "tokenizer_path": (
            str(Path(tokenizer_path).resolve()) if tokenizer_path is not None else None
        ),
        "layers": layer_count,
        "prefill_len": effective_prefill_len,
        "batch_size": effective_batch_size,
        "cache_len": effective_cache_len,
        "device": device,
        "device_id": int(device_id),
        "dtype_seed": dtype_seed,
        "warmup": int(warmup),
        "iterations": int(iterations),
        "runtime_commit": runtime_commit,
        "precision_contract": precision_contract.to_dict(),
        "execution_contract": execution_contract.to_dict(),
    }
    report = _base_report(
        report_path=report_path,
        state_root=state_root,
        seed_path=seed_path,
        model_root=model_root,
        invocation=invocation,
        dry_run=dry_run,
        min_relative_improvement=float(min_relative_improvement),
    )
    measurements: dict[str, dict[str, Any]] = {}
    search_measurement = MeasurementContract(
        warmup=int(warmup),
        iterations=int(iterations),
        kind="candidate_search",
    )

    for level_index, (level_name, state_key) in enumerate(
        AUTOTUNE_LEVELS,
        start=1,
    ):
        candidates = []
        for value in _level_values(state_key, selected_state[state_key]):
            state = copy.deepcopy(selected_state)
            state[state_key] = value
            structured_state = _structured_candidate_state(
                graph=graph,
                seed=seed,
                state=state,
            )
            candidate_config = build_candidate_config(
                graph=graph,
                model_root=model_root,
                precision_contract=precision_contract,
                expected_precision_hash=precision_hash,
                execution_contract=execution_contract,
                measurement_contract=search_measurement,
                runtime_commit=runtime_commit,
                device=device,
                device_id=device_id,
                target=target,
                tunable_state=structured_state,
            )
            fingerprint = candidate_fingerprint(candidate_config)
            report["active_candidate"] = {
                "level": level_index,
                "level_name": level_name,
                "state": copy.deepcopy(state),
                "config": structured_state,
                "fingerprint": fingerprint,
            }
            _write_report(report_path, report)
            reused = fingerprint in measurements
            if reused:
                record = copy.deepcopy(measurements[fingerprint])
                record["measurement_reused"] = True
                record["measurement_reused_from"] = record["candidate_id"]
            else:
                record = _measure_state(
                    state=state,
                    candidate_config=candidate_config,
                    state_root=state_root,
                    graph=graph,
                    seed=seed,
                    model_root=model_root,
                    prompt=prompt,
                    tokenizer_path=tokenizer_path,
                    layer_count=layer_count,
                    prefill_len=effective_prefill_len,
                    batch_size=effective_batch_size,
                    cache_len=effective_cache_len,
                    device=device,
                    device_id=device_id,
                    dtype_seed=dtype_seed,
                    warmup=warmup,
                    iterations=iterations,
                    dry_run=dry_run,
                    resume=resume,
                    profile_runner=profile_runner,
                )
                measurements[fingerprint] = copy.deepcopy(record)
            record["level"] = level_index
            record["level_name"] = level_name
            record["varied_key"] = state_key
            record["varied_value"] = value
            candidates.append(record)
            report["unique_measurement_count"] = len(measurements)
            report["last_completed_candidate"] = {
                "candidate_id": record["candidate_id"],
                "status": record.get("status"),
                "passed": record.get("passed"),
                "metric_value": record.get("metric_value"),
            }
            report["active_candidate"] = None
            _write_report(report_path, report)

        winner = _select_winner(
            candidates,
            dry_run=dry_run,
            min_relative_improvement=float(min_relative_improvement),
        )
        level = {
            "level": level_index,
            "name": level_name,
            "state_key": state_key,
            "metric": AUTOTUNE_METRIC,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "winner": _winner_summary(winner),
            "selection": _selection_summary(candidates, winner),
            "status": "dry_run" if dry_run else ("passed" if winner else "failed"),
            "passed": bool(dry_run or winner is not None),
        }
        report["levels"].append(level)
        if winner is None:
            report.update(
                {
                    "status": "failed",
                    "passed": False,
                    "failed_level": level_name,
                    "selected_state": copy.deepcopy(selected_state),
                }
            )
            _write_report(report_path, report)
            return report
        selected_state = copy.deepcopy(winner["state"])
        report["selected_state"] = copy.deepcopy(selected_state)
        report["selected_config"] = _structured_candidate_state(
            graph=graph,
            seed=seed,
            state=selected_state,
        )
        _write_report(report_path, report)

    if dry_run:
        report.update(
            {
                "status": "dry_run",
                "passed": True,
                "confirmation": None,
                "acceptance": {
                    "status": "dry_run",
                    "passed": True,
                    "failed_checks": [],
                },
            }
        )
        _write_report(report_path, report)
        return report

    def confirm_state(state: dict[str, Any], label: str) -> dict[str, Any]:
        measurement_contract = MeasurementContract(
            warmup=int(confirm_warmup),
            iterations=int(confirm_iterations),
            kind="winner_confirmation",
        )
        structured_state = _structured_candidate_state(
            graph=graph,
            seed=seed,
            state=state,
        )
        candidate_config = build_candidate_config(
            graph=graph,
            model_root=model_root,
            precision_contract=precision_contract,
            expected_precision_hash=precision_hash,
            execution_contract=execution_contract,
            measurement_contract=measurement_contract,
            runtime_commit=runtime_commit,
            device=device,
            device_id=device_id,
            target=target,
            tunable_state=structured_state,
        )
        fingerprint = candidate_fingerprint(candidate_config)
        report["active_candidate"] = {
            "level": "confirmation",
            "level_name": label,
            "state": copy.deepcopy(state),
            "config": structured_state,
            "fingerprint": fingerprint,
        }
        _write_report(report_path, report)
        result = _measure_state(
            state=state,
            candidate_config=candidate_config,
            state_root=state_root,
            graph=graph,
            seed=seed,
            model_root=model_root,
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            layer_count=layer_count,
            prefill_len=effective_prefill_len,
            batch_size=effective_batch_size,
            cache_len=effective_cache_len,
            device=device,
            device_id=device_id,
            dtype_seed=dtype_seed,
            warmup=confirm_warmup,
            iterations=confirm_iterations,
            dry_run=False,
            resume=resume,
            profile_runner=profile_runner,
        )
        result["measurement_kind"] = label
        report["active_candidate"] = None
        _write_report(report_path, report)
        return result

    provisional_selected_state = copy.deepcopy(selected_state)
    challenger_confirmation = confirm_state(
        provisional_selected_state,
        "provisional_winner_confirmation",
    )
    if provisional_selected_state != root_incumbent_state:
        incumbent_confirmation = confirm_state(
            root_incumbent_state,
            "root_incumbent_confirmation",
        )
        promotion = _confirmation_promotion_decision(
            challenger=challenger_confirmation,
            incumbent=incumbent_confirmation,
            min_relative_improvement=float(min_relative_improvement),
        )
        if promotion["promoted"]:
            confirmation = challenger_confirmation
            selected_state = provisional_selected_state
        else:
            confirmation = incumbent_confirmation
            selected_state = copy.deepcopy(root_incumbent_state)
    else:
        incumbent_confirmation = challenger_confirmation
        confirmation = challenger_confirmation
        promotion = {
            "status": "not_required",
            "promoted": False,
            "reason": "provisional winner is the root incumbent",
            "challenger_metric": confirmation.get("metric_value"),
            "incumbent_metric": confirmation.get("metric_value"),
            "relative_improvement": 0.0,
            "minimum_relative_improvement": float(min_relative_improvement),
        }

    passed = bool(confirmation.get("passed"))
    report.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "provisional_selected_state": provisional_selected_state,
            "selected_state": copy.deepcopy(selected_state),
            "selected_config": _structured_candidate_state(
                graph=graph,
                seed=seed,
                state=selected_state,
            ),
            "confirmation": confirmation,
            "provisional_winner_confirmation": challenger_confirmation,
            "root_incumbent_confirmation": incumbent_confirmation,
            "promotion_decision": promotion,
            "acceptance": {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "checks": [
                    {
                        "name": "autotune.precision_frozen_levels_selected",
                        "passed": len(report["levels"]) == 3,
                    },
                    {
                        "name": "autotune.winner_confirmed_with_steady_decode",
                        "passed": passed,
                    },
                ],
                "failed_checks": (
                    [] if passed else ["autotune.winner_confirmed_with_steady_decode"]
                ),
            },
        }
    )
    _write_report(report_path, report)
    return report


def _base_report(
    *,
    report_path: Path,
    state_root: Path,
    seed_path: Path,
    model_root: Path,
    invocation: dict[str, Any],
    dry_run: bool,
    min_relative_improvement: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": "diagnose",
        "stage": "autotune",
        "strategy": "progressive_precision_frozen_decode_steady",
        "metric": AUTOTUNE_METRIC,
        "metric_direction": "maximize",
        "selection_policy": {
            "minimum_relative_improvement": min_relative_improvement,
            "below_threshold": "keep_incumbent",
        },
        "status": "running",
        "passed": False,
        "dry_run": bool(dry_run),
        "model_path": str(model_root),
        "config_path": str(seed_path),
        "report_path": str(report_path),
        "candidates_dir": str(state_root),
        "invocation": invocation,
        "levels": [],
        "unique_measurement_count": 0,
        "selected_state": None,
        "selected_config": None,
        "provisional_selected_state": None,
        "confirmation": None,
        "provisional_winner_confirmation": None,
        "root_incumbent_confirmation": None,
        "promotion_decision": None,
        "active_candidate": None,
        "last_completed_candidate": None,
        "acceptance": None,
    }


def _validate_measurement_counts(
    *,
    warmup: int,
    iterations: int,
    confirm_warmup: int,
    confirm_iterations: int,
) -> None:
    if int(warmup) < 0 or int(confirm_warmup) < 0:
        raise ValueError("autotune warmup counts must be non-negative")
    if int(iterations) <= 0 or int(confirm_iterations) <= 0:
        raise ValueError("autotune iteration counts must be positive")
