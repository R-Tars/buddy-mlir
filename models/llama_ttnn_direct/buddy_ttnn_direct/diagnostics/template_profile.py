from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from ..autotune.microbench import summarize_samples
from ..autotune.schema import MeasurementContract
from ..runtime.reports import write_report
from ..smoke_mlp import (
    MLP_SMOKE_OPS, NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError, managed_mlp_device,
    prepare_mlp_smoke_on_device,
)

MLP_PROFILE_OPS = [{"name": name, "count": 1} for name in ("linear.gate", "linear.up", "mul.silu", "linear.down")]
PCC_THRESHOLD = 0.99
TRACE_APIS = ("begin_trace_capture", "end_trace_capture", "execute_trace", "release_trace")


def profile_template(
    *, template: str, config_path: str | Path, out: str | Path,
    warmup: int, iterations: int, trace: bool = False, dry_run: bool = False,
    device_id: int = 0, dtype_seed: str = "bf16",
    ttnn_module: Any | None = None, torch_module: Any | None = None,
) -> dict[str, Any]:
    if template != "mlp_decode":
        raise ValueError("template-profile only supports template=mlp_decode")
    contract = MeasurementContract(warmup=warmup, iterations=iterations)
    shape = _shape(json.loads(Path(config_path).read_text()))
    report = {
        "schema_version": 3, "template": template, **shape,
        "device_id": device_id, "warmup": contract.warmup,
        "iterations": contract.iterations, "trace_enabled": trace,
        "dry_run": dry_run, "dtype_seed": dtype_seed, "ops": MLP_PROFILE_OPS,
        "ttnn_ops": list(MLP_SMOKE_OPS),
        "autotune_measurement_contract": contract.to_dict(),
        "worker": "smoke_mlp.prepare_mlp_smoke_on_device",
        "tensor_lifecycle": {"persistent_tensor_reuse": True, "tensor_recreation_per_iteration": False},
    }
    if dry_run:
        report.update(
            status="dry_run", passed=True, latency_ms=_latency([0.0]),
            trace={"requested": trace, "status": "dry_run" if trace else "disabled"},
            device_lifecycle={"open_count": 0, "close_count": 0},
            message="Dry run only; TTNN device is not required.",
        )
        return _emit(out, report)
    try:
        ttnn = ttnn_module or importlib.import_module("ttnn")
    except ImportError as error:
        return _emit(out, _failed(report, "no_device", NO_TTNN_DEVICE_MESSAGE, error))
    try:
        torch = torch_module or importlib.import_module("torch")
    except ImportError as error:
        return _emit(out, _failed(report, "missing_torch", "torch is required.", error))
    try:
        with managed_mlp_device(ttnn, device_id) as device:
            report.update(_measure(ttnn, torch, device, shape, contract, trace, dtype_seed))
        report["device_lifecycle"] = {"open_count": 1, "close_count": 1,
            "shared_across_warmup_and_measurement": True}
    except NoTTNNDeviceError as error:
        report = _failed(report, "no_device", NO_TTNN_DEVICE_MESSAGE, error)
    except Exception as error:
        report = _failed(report, "runtime_error", "TTNN profiling failed.", error)
    return _emit(out, report)

def _measure(ttnn: Any, torch: Any, device: Any, shape: dict[str, Any],
             contract: MeasurementContract, trace: bool, dtype_seed: str
             ) -> dict[str, Any]:
    execute, measure_pcc = prepare_mlp_smoke_on_device(
        ttnn=ttnn, torch=torch, ttnn_device=device,
        batch_size=shape["batch_size"], hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        dtype_seed=dtype_seed, seed=0,
    )
    samples = None
    trace_report = {"requested": trace, "status": "disabled"}
    if trace and all(callable(getattr(ttnn, name, None)) for name in TRACE_APIS):
        execute()
        _synchronize(ttnn, device)
        samples, pcc, trace_report = _trace_measure(
            ttnn, device, execute, measure_pcc, contract
        )
    elif trace:
        trace_report = {"requested": True, "status": "trace_api_unavailable_fell_back_to_eager"}
    if samples is None:
        samples, output = _eager_measure(ttnn, device, execute, contract)
        pcc = measure_pcc(output)
    passed = pcc >= PCC_THRESHOLD
    return {
        "status": "profiled" if passed else "pcc_below_threshold",
        "passed": passed, "latency_ms": _latency(samples),
        "pcc": {"min": pcc, "last": pcc, "threshold": PCC_THRESHOLD, "passed": passed},
        "trace": trace_report,
    }

def _trace_measure(ttnn: Any, device: Any, execute: Callable[[], Any],
                   measure_pcc: Callable[[Any], float], contract: MeasurementContract
                   ) -> tuple[list[float] | None, float, dict[str, Any]]:
    trace_id = None
    error = None
    samples: list[float] = []
    try:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        output = execute()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        for _ in range(contract.warmup):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        for _ in range(contract.iterations):
            start = time.perf_counter_ns()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        pcc = measure_pcc(output)
    except Exception as caught:
        error = caught
    if trace_id is not None:
        try:
            ttnn.release_trace(device, trace_id)
        except Exception as caught:
            error = error or caught
    if error is not None:
        return None, 0.0, {"requested": True,
            "status": "trace_failed_fell_back_to_eager",
            "detail": f"{type(error).__name__}: {error}"}
    return samples, pcc, {
        "requested": True, "status": "captured", "capture_count": 1,
        "warmup_execute_count": contract.warmup,
        "measured_execute_count": contract.iterations,
        "release_count": 1,
    }

def _eager_measure(ttnn: Any, device: Any, execute: Callable[[], Any],
                   contract: MeasurementContract):
    for _ in range(contract.warmup):
        execute()
        _synchronize(ttnn, device)
    samples, output = [], None
    for _ in range(contract.iterations):
        start = time.perf_counter_ns()
        output = execute()
        _synchronize(ttnn, device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples, output

def _shape(config: dict[str, Any]) -> dict[str, Any]:
    template = config.get("template_config")
    template = template if isinstance(template, dict) else {}
    hidden = int(config.get("hidden_size") or 1024)
    return {
        "model_name": str(config.get("model_name") or config.get("model") or "unknown"),
        "device": str(config.get("device") or template.get("device") or "unknown"),
        "batch_size": int(config.get("batch_size") or template.get("batch_size") or 1),
        "hidden_size": hidden,
        "intermediate_size": int(config.get("intermediate_size") or hidden * 4),
    }

def _synchronize(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)

def _latency(samples: list[float]) -> dict[str, float]:
    stats = summarize_samples(samples)
    return {key: float(stats[key]) for key in ("mean", "p50", "p90")}

def _failed(report: dict[str, Any], status: str, message: str, error: Exception):
    report.update(
        status=status, passed=False, error=message,
        detail=f"{type(error).__name__}: {error}",
        latency_ms=_latency([0.0]),
        trace={"requested": report["trace_enabled"],
               "status": "unavailable" if report["trace_enabled"] else "disabled"},
    )
    return report

def _emit(out: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    write_report(out, report)
    return report
