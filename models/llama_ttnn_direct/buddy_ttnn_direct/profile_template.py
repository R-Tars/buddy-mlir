from __future__ import annotations

import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .smoke_mlp import (
    MLP_SMOKE_OPS,
    NO_TTNN_DEVICE_MESSAGE,
    NoTTNNDeviceError,
    _managed_ttnn_device,
    _run_ttnn_mlp_smoke,
)


MLP_PROFILE_OPS = [
    {"name": "linear.gate", "count": 1},
    {"name": "linear.up", "count": 1},
    {"name": "mul.silu", "count": 1},
    {"name": "linear.down", "count": 1},
]


def profile_template(
    *,
    template: str,
    config_path: str | Path,
    out: str | Path,
    warmup: int,
    iterations: int,
    trace: bool = False,
    dry_run: bool = False,
    device_id: int = 0,
    dtype_seed: str = "bf16",
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if template != "mlp_decode":
        raise ValueError("Phase 10 profiling only supports template=mlp_decode")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    config = json.loads(Path(config_path).read_text())
    shape = _profile_shape_from_config(config)
    if dry_run:
        report = _base_profile_report(
            template=template,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
            trace=trace,
            dry_run=True,
            device_id=device_id,
            dtype_seed=dtype_seed,
        )
        report.update(
            {
                "status": "dry_run",
                "latency_ms": _latency_stats([0.0] * iterations),
                "trace": _trace_report(
                    requested=trace,
                    status="dry_run" if trace else "disabled",
                ),
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_json(out, report)
        return report

    try:
        ttnn = (
            ttnn_module
            if ttnn_module is not None
            else importlib.import_module("ttnn")
        )
    except ImportError as err:
        report = _profile_unavailable_report(
            template=template,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
            trace=trace,
            device_id=device_id,
            dtype_seed=dtype_seed,
            status="no_device",
            error=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
        )
        _write_json(out, report)
        return report

    try:
        torch = (
            torch_module
            if torch_module is not None
            else importlib.import_module("torch")
        )
    except ImportError as err:
        report = _profile_unavailable_report(
            template=template,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
            trace=trace,
            device_id=device_id,
            dtype_seed=dtype_seed,
            status="missing_torch",
            error="torch is required for TTNN template profiling.",
            detail=str(err),
        )
        _write_json(out, report)
        return report

    try:
        with _managed_ttnn_device(ttnn, device_id) as ttnn_device:
            report = _profile_mlp_with_device(
                ttnn=ttnn,
                torch=torch,
                ttnn_device=ttnn_device,
                template=template,
                shape=shape,
                warmup=warmup,
                iterations=iterations,
                trace=trace,
                device_id=device_id,
                dtype_seed=dtype_seed,
            )
    except NoTTNNDeviceError as err:
        report = _profile_unavailable_report(
            template=template,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
            trace=trace,
            device_id=device_id,
            dtype_seed=dtype_seed,
            status="no_device",
            error=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
        )
    except Exception as err:
        report = _profile_unavailable_report(
            template=template,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
            trace=trace,
            device_id=device_id,
            dtype_seed=dtype_seed,
            status="runtime_error",
            error="TTNN template profiling failed.",
            detail=f"{type(err).__name__}: {err}",
        )

    _write_json(out, report)
    return report


def _profile_mlp_with_device(
    *,
    ttnn: Any,
    torch: Any,
    ttnn_device: Any,
    template: str,
    shape: dict[str, Any],
    warmup: int,
    iterations: int,
    trace: bool,
    device_id: int,
    dtype_seed: str,
) -> dict[str, Any]:
    for index in range(warmup):
        _run_ttnn_mlp_smoke(
            ttnn=ttnn,
            torch=torch,
            ttnn_device=ttnn_device,
            device=shape["device"],
            device_id=device_id,
            batch_size=shape["batch_size"],
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            dtype_seed=dtype_seed,
            pcc_threshold=0.0,
            seed=index,
        )

    trace_info = _trace_report(
        requested=trace,
        status="disabled",
    )
    samples: list[float] = []
    pcc_values: list[float] = []
    if trace and _trace_apis_available(ttnn):
        try:
            trace_id = ttnn.begin_trace_capture(ttnn_device, cq_id=0)
            capture_report = _run_ttnn_mlp_smoke(
                ttnn=ttnn,
                torch=torch,
                ttnn_device=ttnn_device,
                device=shape["device"],
                device_id=device_id,
                batch_size=shape["batch_size"],
                hidden_size=shape["hidden_size"],
                intermediate_size=shape["intermediate_size"],
                dtype_seed=dtype_seed,
                pcc_threshold=0.0,
                seed=warmup,
            )
            ttnn.end_trace_capture(ttnn_device, trace_id, cq_id=0)
            pcc_values.append(float(capture_report.get("pcc") or 0.0))
            for _ in range(iterations):
                start = time.perf_counter()
                ttnn.execute_trace(ttnn_device, trace_id, cq_id=0, blocking=True)
                synchronize = getattr(ttnn, "synchronize_device", None)
                if callable(synchronize):
                    synchronize(ttnn_device)
                samples.append((time.perf_counter() - start) * 1000.0)
            release_trace = getattr(ttnn, "release_trace", None)
            if callable(release_trace):
                release_trace(ttnn_device, trace_id)
            trace_info = _trace_report(
                requested=True,
                status="captured",
                trace_id=trace_id,
            )
        except Exception as err:
            samples = []
            pcc_values = []
            trace_info = _trace_report(
                requested=True,
                status="trace_failed_fell_back_to_eager",
                detail=f"{type(err).__name__}: {err}",
            )
    elif trace:
        trace_info = _trace_report(
            requested=True,
            status="trace_api_unavailable_fell_back_to_eager",
        )

    if not samples:
        for index in range(iterations):
            iteration_report = _run_ttnn_mlp_smoke(
                ttnn=ttnn,
                torch=torch,
                ttnn_device=ttnn_device,
                device=shape["device"],
                device_id=device_id,
                batch_size=shape["batch_size"],
                hidden_size=shape["hidden_size"],
                intermediate_size=shape["intermediate_size"],
                dtype_seed=dtype_seed,
                pcc_threshold=0.0,
                seed=warmup + index + 1,
            )
            samples.append(float(iteration_report["latency_ms"]))
            pcc_values.append(float(iteration_report.get("pcc") or 0.0))

    report = _base_profile_report(
        template=template,
        shape=shape,
        warmup=warmup,
        iterations=iterations,
        trace=trace,
        dry_run=False,
        device_id=device_id,
        dtype_seed=dtype_seed,
    )
    report.update(
        {
            "status": "profiled",
            "latency_ms": _latency_stats(samples),
            "trace": trace_info,
            "pcc": {
                "min": min(pcc_values) if pcc_values else None,
                "last": pcc_values[-1] if pcc_values else None,
            },
        }
    )
    return report


def _profile_shape_from_config(config: dict[str, Any]) -> dict[str, Any]:
    template_config = config.get("template_config", {})
    if not isinstance(template_config, dict):
        template_config = {}
    model_name = (
        config.get("model_name")
        or template_config.get("model")
        or config.get("model")
        or "unknown"
    )
    device = config.get("device") or template_config.get("device") or "unknown"
    batch_size = int(config.get("batch_size") or template_config.get("batch_size") or 1)
    hidden_size = int(config.get("hidden_size") or 1024)
    intermediate_size = int(config.get("intermediate_size") or hidden_size * 4)
    return {
        "model_name": str(model_name),
        "device": str(device),
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def _base_profile_report(
    *,
    template: str,
    shape: dict[str, Any],
    warmup: int,
    iterations: int,
    trace: bool,
    dry_run: bool,
    device_id: int,
    dtype_seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": template,
        "model_name": shape["model_name"],
        "device": shape["device"],
        "device_id": device_id,
        "batch_size": shape["batch_size"],
        "hidden_size": shape["hidden_size"],
        "intermediate_size": shape["intermediate_size"],
        "warmup": warmup,
        "iterations": iterations,
        "trace_enabled": trace,
        "dry_run": dry_run,
        "dtype_seed": dtype_seed,
        "ops": list(MLP_PROFILE_OPS),
        "ttnn_ops": list(MLP_SMOKE_OPS),
    }


def _profile_unavailable_report(
    *,
    template: str,
    shape: dict[str, Any],
    warmup: int,
    iterations: int,
    trace: bool,
    device_id: int,
    dtype_seed: str,
    status: str,
    error: str,
    detail: str,
) -> dict[str, Any]:
    report = _base_profile_report(
        template=template,
        shape=shape,
        warmup=warmup,
        iterations=iterations,
        trace=trace,
        dry_run=False,
        device_id=device_id,
        dtype_seed=dtype_seed,
    )
    report.update(
        {
            "status": status,
            "latency_ms": _latency_stats([0.0] * iterations),
            "trace": _trace_report(
                requested=trace,
                status="unavailable" if trace else "disabled",
            ),
            "error": error,
            "detail": detail,
        }
    )
    return report


def _trace_apis_available(ttnn: Any) -> bool:
    return all(
        callable(getattr(ttnn, name, None))
        for name in (
            "begin_trace_capture",
            "end_trace_capture",
            "execute_trace",
        )
    )


def _trace_report(
    *,
    requested: bool,
    status: str,
    trace_id: Any | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    report = {
        "requested": requested,
        "status": status,
    }
    if trace_id is not None:
        report["trace_id"] = trace_id
    if detail is not None:
        report["detail"] = detail
    return report


def _latency_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(samples)
    return {
        "mean": float(statistics.fmean(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _write_json(out: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
