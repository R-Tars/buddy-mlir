from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

from ..autotune.microbench import (
    MLP_SMOKE_OPS,
    prepare_mlp_smoke_on_device,
)
from ..runtime.device import managed_ttnn_device
from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..runtime.reports import write_report
from .support import failed_diagnostic_report as _failed_diagnostic_report



def run_smoke_mlp(
    *,
    out: str | Path,
    device: str,
    device_id: int = 0,
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    pcc_threshold: float = 0.99,
    seed: int = 0,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    _validate_shape_args(batch_size, hidden_size, intermediate_size)
    if dtype_seed not in {"bf16", "fp32"}:
        raise ValueError("dtype_seed must be one of: bf16, fp32")

    if dry_run:
        report = _base_report(
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_seed=dtype_seed,
            dry_run=True,
            pcc_threshold=pcc_threshold,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "pcc": None,
                "latency_ms": 0.0,
                "message": "Dry run only; TTNN device is not required.",
            }
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
        report = _no_device_report(
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_seed=dtype_seed,
            pcc_threshold=pcc_threshold,
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
        report = _failed_report(
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_seed=dtype_seed,
            pcc_threshold=pcc_threshold,
            status="missing_torch",
            message="torch is required for TTNN MLP smoke PCC reference.",
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        injected_device = (
            ttnn_module
            if callable(getattr(ttnn_module, "from_torch", None))
            else None
        )
        with managed_ttnn_device(ttnn, device_id, injected_device) as ttnn_device:
            report = _run_ttnn_mlp_smoke(
                ttnn=ttnn,
                torch=torch,
                ttnn_device=ttnn_device,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype_seed=dtype_seed,
                pcc_threshold=pcc_threshold,
                seed=seed,
            )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_seed=dtype_seed,
            pcc_threshold=pcc_threshold,
            detail=str(err),
        )
    except Exception as err:
        report = _failed_report(
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_seed=dtype_seed,
            pcc_threshold=pcc_threshold,
            status="runtime_error",
            message="TTNN MLP smoke execution failed.",
            detail=f"{type(err).__name__}: {err}",
        )

    _write_report(out, report)
    return report


def _run_ttnn_mlp_smoke(
    *, ttnn: Any, torch: Any, ttnn_device: Any, device: str, device_id: int,
    batch_size: int, hidden_size: int, intermediate_size: int,
    dtype_seed: str, pcc_threshold: float, seed: int,
) -> dict[str, Any]:
    execute, measure_pcc = prepare_mlp_smoke_on_device(
        ttnn=ttnn, torch=torch, ttnn_device=ttnn_device, batch_size=batch_size,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        dtype_seed=dtype_seed, seed=seed,
    )
    start = time.perf_counter()
    out = execute()
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(ttnn_device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    pcc = measure_pcc(out)
    passed = pcc >= pcc_threshold
    report = _base_report(
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype_seed=dtype_seed,
        dry_run=False,
        pcc_threshold=pcc_threshold,
    )
    report.update(
        {
            "passed": passed,
            "status": "passed" if passed else "pcc_below_threshold",
            "pcc": pcc,
            "latency_ms": latency_ms,
        }
    )
    return report


def _base_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "mlp_decode",
        "device": kwargs["device"],
        "device_id": kwargs["device_id"],
        "batch_size": kwargs["batch_size"],
        "hidden_size": kwargs["hidden_size"],
        "intermediate_size": kwargs["intermediate_size"],
        "dtype_seed": kwargs["dtype_seed"],
        "dry_run": kwargs["dry_run"],
        "pcc_threshold": kwargs["pcc_threshold"],
        "ttnn_ops": list(MLP_SMOKE_OPS),
    }


def _error_report(**kwargs: Any) -> dict[str, Any]:
    error_keys = {"status", "message", "detail"}
    base = _base_report(
        **{key: value for key, value in kwargs.items() if key not in error_keys},
        dry_run=False,
    )
    return _failed_diagnostic_report(
        base,
        status=kwargs["status"],
        message=kwargs["message"],
        detail=kwargs["detail"],
        include_runtime_metadata=False,
        extra={"pcc": None},
    )


def _no_device_report(**kwargs: Any) -> dict[str, Any]:
    kwargs.update(status="no_device", message=NO_TTNN_DEVICE_MESSAGE)
    return _error_report(**kwargs)


def _failed_report(**kwargs: Any) -> dict[str, Any]:
    return _error_report(**kwargs)


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    write_report(out, report)


def _validate_shape_args(
    batch_size: int, hidden_size: int, intermediate_size: int
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
