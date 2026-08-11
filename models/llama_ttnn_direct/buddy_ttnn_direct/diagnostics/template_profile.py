from __future__ import annotations

import importlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..autotune.microbench import MLP_SMOKE_OPS, prepare_mlp_smoke_on_device
from ..runtime.device import managed_ttnn_device
from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..runtime.reports import write_report

MLP_PROFILE_OPS = [{"name": name, "count": 1} for name in ("linear.gate", "linear.up", "mul.silu", "linear.down")]
PCC_THRESHOLD = 0.99
TRACE_APIS = ("begin_trace_capture", "end_trace_capture", "execute_trace", "release_trace")


@dataclass(frozen=True)
class ProfileMeasurementContract:
    warmup: int
    iterations: int

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("measurement warmup must be non-negative")
        if self.iterations <= 0:
            raise ValueError("measurement iterations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "candidate_search",
            "scope": "post_prefill_steady_decode",
            "metric": "tokens_per_second_per_user",
            "warmup": self.warmup,
            "iterations": self.iterations,
            "repetitions": 1,
            "synchronize_device": True,
        }


def profile_template(
    *, template: str, config_path: str | Path, out: str | Path,
    warmup: int, iterations: int, trace: bool = False, dry_run: bool = False,
    device_id: int = 0, dtype_seed: str = "bf16",
    ttnn_module: Any | None = None, torch_module: Any | None = None,
) -> dict[str, Any]:
    if template != "mlp_decode":
        raise ValueError("template-profile only supports template=mlp_decode")
    contract = ProfileMeasurementContract(warmup=warmup, iterations=iterations)
    shape = _shape(json.loads(Path(config_path).read_text()))
    report = {
        "schema_version": 3, "template": template, **shape,
        "device_id": device_id, "warmup": contract.warmup,
        "iterations": contract.iterations, "trace_enabled": trace,
        "dry_run": dry_run, "dtype_seed": dtype_seed, "ops": MLP_PROFILE_OPS,
        "ttnn_ops": list(MLP_SMOKE_OPS),
        "autotune_measurement_contract": contract.to_dict(),
        "worker": "autotune.microbench.prepare_mlp_smoke_on_device",
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
        with managed_ttnn_device(ttnn, device_id) as device:
            report.update(_measure(ttnn, torch, device, shape, contract, trace, dtype_seed))
        report["device_lifecycle"] = {"open_count": 1, "close_count": 1,
            "shared_across_warmup_and_measurement": True}
    except NoTTNNDeviceError as error:
        report = _failed(report, "no_device", NO_TTNN_DEVICE_MESSAGE, error)
    except Exception as error:
        report = _failed(report, "runtime_error", "TTNN profiling failed.", error)
    return _emit(out, report)

def _measure(ttnn: Any, torch: Any, device: Any, shape: dict[str, Any],
             contract: ProfileMeasurementContract, trace: bool, dtype_seed: str
             ) -> dict[str, Any]:
    execute, measure_pcc = prepare_mlp_smoke_on_device(
        ttnn=ttnn, torch=torch, ttnn_device=device,
        batch_size=shape["batch_size"], hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        dtype_seed=dtype_seed, seed=0,
    )
    samples = None
    fallback_allowed = not trace
    trace_report = {"requested": trace, "status": "disabled"}
    if trace and all(callable(getattr(ttnn, name, None)) for name in TRACE_APIS):
        execute()
        _synchronize(ttnn, device)
        samples, pcc, trace_report, fallback_allowed = _trace_measure(
            ttnn, device, execute, measure_pcc, contract
        )
    elif trace:
        trace_report = {"requested": True, "status": "trace_api_unavailable_fell_back_to_eager"}
        fallback_allowed = True
    if samples is None:
        if trace and not fallback_allowed:
            return {
                "status": trace_report["status"], "passed": False,
                "latency_ms": _latency([0.0]), "trace": trace_report,
            }
        if trace_report.get("failure_stage"):
            trace_report["fallback_executed"] = True
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
                   measure_pcc: Callable[[Any], float], contract: ProfileMeasurementContract
                   ) -> tuple[list[float] | None, float | None, dict[str, Any], bool]:
    trace_id = None
    error: Exception | None = None
    release_error: Exception | None = None
    failure_stage: str | None = None
    samples: list[float] = []
    cleanup: dict[str, Any] = {
        "end_attempted": False, "end_succeeded": False,
        "release_attempted": False, "release_succeeded": False, "errors": [],
    }
    try:
        failure_stage = "begin_capture"
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        failure_stage = "capture_body"
        output = execute()
        failure_stage = "end_capture"
        cleanup["end_attempted"] = True
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        cleanup["end_succeeded"] = True
        failure_stage = "warmup_replay"
        for _ in range(contract.warmup):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        failure_stage = "measured_replay"
        for _ in range(contract.iterations):
            start = time.perf_counter_ns()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        failure_stage = "pcc"
        pcc = measure_pcc(output)
        failure_stage = None
    except Exception as caught:
        error = caught
    if trace_id is not None and failure_stage == "capture_body":
        cleanup["end_attempted"] = True
        try:
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            cleanup["end_succeeded"] = True
        except Exception as caught:
            cleanup["errors"].append(f"end: {type(caught).__name__}: {caught}")
    if trace_id is not None:
        cleanup["release_attempted"] = True
        try:
            ttnn.release_trace(device, trace_id)
            cleanup["release_succeeded"] = True
        except Exception as caught:
            release_error = caught
            cleanup["errors"].append(f"release: {type(caught).__name__}: {caught}")
    if error is None and release_error is None:
        return samples, pcc, {
            "requested": True, "status": "captured", "capture_count": 1,
            "warmup_execute_count": contract.warmup,
            "measured_execute_count": contract.iterations,
            "release_count": 1,
        }, False
    fallback_allowed = failure_stage == "begin_capture" or (
        failure_stage in {"capture_body", "warmup_replay", "measured_replay"}
        and cleanup["end_succeeded"] and cleanup["release_succeeded"]
    )
    cleanup["safe_for_eager_fallback"] = fallback_allowed
    detail = []
    if error is not None:
        detail.append(f"{type(error).__name__}: {error}")
    detail.extend(cleanup["errors"])
    status = "trace_release_failed" if release_error is not None else (
        "trace_cleanup_failed" if failure_stage in {"capture_body", "end_capture"}
        and not fallback_allowed
        else "runtime_error" if failure_stage == "pcc"
        else "trace_failed_fell_back_to_eager"
    )
    return None, None, {
        "requested": True, "failure_stage": failure_stage or "release",
        "detail": "; ".join(detail), "capture_cleanup": cleanup,
        "fallback_executed": False, "status": status,
    }, fallback_allowed

def _eager_measure(ttnn: Any, device: Any, execute: Callable[[], Any],
                   contract: ProfileMeasurementContract):
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
    values = sorted(float(sample) for sample in samples)
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight

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
