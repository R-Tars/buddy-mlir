from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ..runtime.generate import run_generate
from ..runtime.observations import TTNNObservationCollector
from .hf_reference import (
    capture_hf_reference,
    load_hf_reference,
    write_hf_reference,
)
from .metrics import compare_snapshots, compare_top_token


DEFAULT_CHECKS = (
    "top_token",
    "logits_pcc",
    "hidden_pcc",
    "kv_cache_pcc",
)


def run_correctness(
    *,
    out_dir: str | Path,
    program_dir: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path | None,
    prompt: str,
    layers: int,
    prefill_len: int,
    batch_size: int,
    cache_len: int,
    device: str,
    device_id: int = 0,
    dtype_seed: str = "bf16",
    reference_dtype: str = "bfloat16",
    hf_reference: str | Path | None = None,
    checks: tuple[str, ...] = DEFAULT_CHECKS,
    pcc_threshold: float = 0.99,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
    tokenizer_module: Any | None = None,
    runtime_input_mode: str | None = None,
) -> dict[str, Any]:
    requested_checks = _validate_checks(checks)
    if not 0.0 <= float(pcc_threshold) <= 1.0:
        raise ValueError("pcc_threshold must be between 0 and 1")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    hf_reference_path = root / "hf_reference.json"
    observations_path = root / "ttnn_observations.json"
    generate_report_path = root / "generate.json"
    report_path = root / "validation_report.json"

    if hf_reference is None:
        reference = capture_hf_reference(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            prompt=prompt,
            layers=layers,
            prefill_len=prefill_len,
            dtype=reference_dtype,
            torch_module=torch_module,
            transformers_module=transformers_module,
        )
        reference_source = "captured"
    else:
        reference = load_hf_reference(
            hf_reference,
            model_path=model_path,
            prompt=prompt,
            layers=layers,
            prefill_len=prefill_len,
            dtype=reference_dtype,
        )
        reference_source = "provided"
    write_hf_reference(hf_reference_path, reference)

    ttnn = ttnn_module or importlib.import_module("ttnn")
    torch = torch_module or importlib.import_module("torch")
    collector = TTNNObservationCollector(
        ttnn=ttnn,
        torch=torch,
        capture_hidden="hidden_pcc" in requested_checks,
        capture_logits=(
            "logits_pcc" in requested_checks
            or "top_token" in requested_checks
        ),
        capture_kv_cache="kv_cache_pcc" in requested_checks,
    )
    generate = run_generate(
        out=generate_report_path,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path or model_path,
        max_new_tokens=1,
        layers=layers,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        ttnn_module=ttnn_module,
        torch_module=torch_module,
        tokenizer_module=tokenizer_module,
        observer=collector,
        runtime_input_mode=runtime_input_mode,
    )
    observations = collector.to_report()
    _write_json(observations_path, observations)

    comparisons = _compare_requested(
        requested_checks=requested_checks,
        generate=generate,
        observed=observations["checkpoints"],
        reference=reference["checkpoints"],
        reference_top_token=int(reference["top_token"]),
        pcc_threshold=float(pcc_threshold),
    )
    runtime_passed = bool(generate.get("passed"))
    passed = runtime_passed and bool(comparisons) and all(
        comparison["passed"] for comparison in comparisons
    )
    status = "pass" if passed else str(generate.get("status") or "fail")
    if not passed and status == "passed":
        status = "fail"
    report = {
        "schema_version": 1,
        "command": "validate",
        "suite": "correctness",
        "status": status,
        "passed": passed,
        "program_dir": str(program_dir),
        "model_path": str(model_path),
        "layers": int(layers),
        "batch_size": int(batch_size),
        "prefill_len": int(prefill_len),
        "cache_len": int(cache_len),
        "device": device,
        "device_id": int(device_id),
        "dtype_seed": dtype_seed,
        "reference_dtype": reference_dtype,
        "reference_source": reference_source,
        "provided_hf_reference": (
            str(hf_reference) if hf_reference is not None else None
        ),
        "pcc_threshold": float(pcc_threshold),
        "requested_checks": list(requested_checks),
        "top_token": {
            "observed": _observed_top_token(generate),
            "reference": int(reference["top_token"]),
        },
        "comparisons": comparisons,
        "failed_checks": [
            comparison["name"]
            for comparison in comparisons
            if not comparison["passed"]
        ],
        "runtime_status": generate.get("status"),
        "runtime_input_mode": generate.get("runtime_input_mode"),
        "runtime_inputs": generate.get("runtime_inputs"),
        "reports": {
            "generate": str(generate_report_path),
            "hf_reference": str(hf_reference_path),
            "ttnn_observations": str(observations_path),
        },
    }
    _write_json(report_path, report)
    return report


def _compare_requested(
    *,
    requested_checks: tuple[str, ...],
    generate: dict[str, Any],
    observed: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    reference_top_token: int,
    pcc_threshold: float,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if "top_token" in requested_checks:
        observed_token = _observed_top_token(generate)
        if observed_token is None:
            comparisons.append(_missing_check("top_token", "observed token"))
        else:
            comparisons.append(
                compare_top_token(observed_token, reference_top_token)
            )
    prefixes: list[tuple[str, str]] = []
    if "logits_pcc" in requested_checks:
        prefixes.append(("logits_pcc", "prefill.logits"))
    if "hidden_pcc" in requested_checks:
        prefixes.extend(
            ("hidden_pcc", name)
            for name in reference
            if name.endswith("hidden")
        )
    if "kv_cache_pcc" in requested_checks:
        prefixes.extend(
            ("kv_cache_pcc", name)
            for name in reference
            if name.endswith((".key_cache", ".value_cache"))
        )
    for check_kind, name in prefixes:
        if name not in observed:
            comparisons.append(_missing_check(name, "TTNN observation"))
            continue
        if name not in reference:
            comparisons.append(_missing_check(name, "HF reference"))
            continue
        comparison = compare_snapshots(
            observed[name],
            reference[name],
            pcc_threshold=pcc_threshold,
        )
        comparison["check_kind"] = check_kind
        comparisons.append(comparison)
    return comparisons


def _observed_top_token(generate: dict[str, Any]) -> int | None:
    rows = generate.get("generated_token_ids")
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    if not isinstance(first, list) or not first:
        return None
    return int(first[0])


def _missing_check(name: str, missing: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "missing",
        "passed": False,
        "missing": missing,
    }


def _validate_checks(checks: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(check) for check in checks))
    unknown = sorted(set(requested).difference(DEFAULT_CHECKS))
    if unknown:
        raise ValueError(f"unknown correctness checks: {', '.join(unknown)}")
    if not requested:
        raise ValueError("at least one correctness check is required")
    return requested


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
