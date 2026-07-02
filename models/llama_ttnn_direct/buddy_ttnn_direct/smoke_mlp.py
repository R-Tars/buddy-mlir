from __future__ import annotations

import importlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


NO_TTNN_DEVICE_MESSAGE = "No TTNN device detected. Use --dry-run or run on P150A."
MLP_SMOKE_OPS = ["linear", "linear", "mul_silu", "linear"]


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
        with _managed_ttnn_device(ttnn, device_id) as ttnn_device:
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


class NoTTNNDeviceError(RuntimeError):
    pass


def _run_ttnn_mlp_smoke(
    *,
    ttnn: Any,
    torch: Any,
    ttnn_device: Any,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_seed: str,
    pcc_threshold: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch_dtype = torch.bfloat16 if dtype_seed == "bf16" else torch.float32
    hidden = torch.randn(
        (batch_size, 1, hidden_size),
        dtype=torch_dtype,
    ).to(torch.float32)
    gate_weight = torch.randn(
        (hidden_size, intermediate_size),
        dtype=torch_dtype,
    ).to(torch.float32)
    up_weight = torch.randn(
        (hidden_size, intermediate_size),
        dtype=torch_dtype,
    ).to(torch.float32)
    down_weight = torch.randn(
        (intermediate_size, hidden_size),
        dtype=torch_dtype,
    ).to(torch.float32)

    reference = torch.nn.functional.silu(hidden @ gate_weight)
    reference = reference * (hidden @ up_weight)
    reference = reference @ down_weight

    ttnn_dtype = _ttnn_dtype(ttnn, dtype_seed)
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn_dtype,
        layout=layout,
        device=ttnn_device,
    )
    gate_weight_tt = ttnn.from_torch(
        gate_weight,
        dtype=ttnn_dtype,
        layout=layout,
        device=ttnn_device,
    )
    up_weight_tt = ttnn.from_torch(
        up_weight,
        dtype=ttnn_dtype,
        layout=layout,
        device=ttnn_device,
    )
    down_weight_tt = ttnn.from_torch(
        down_weight,
        dtype=ttnn_dtype,
        layout=layout,
        device=ttnn_device,
    )

    start = time.perf_counter()
    gate = ttnn.linear(hidden_tt, gate_weight_tt)
    up = ttnn.linear(hidden_tt, up_weight_tt)
    mid = _ttnn_mul_silu(ttnn, gate, up)
    out = ttnn.linear(mid, down_weight_tt)
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(ttnn_device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    output = ttnn.to_torch(out).to(torch.float32)
    pcc = _pearson_corr(torch, reference, output)
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


@contextmanager
def _managed_ttnn_device(ttnn: Any, device_id: int):
    manage_device = getattr(ttnn, "manage_device", None)
    if callable(manage_device):
        try:
            manager = manage_device(device_id=device_id)
        except TypeError:
            manager = manage_device(device_id)
        enter = getattr(manager, "__enter__", None)
        exit_ = getattr(manager, "__exit__", None)
        if enter is None or exit_ is None:
            raise NoTTNNDeviceError("ttnn manage_device is not a context manager")
        try:
            device = enter()
        except Exception as err:
            raise NoTTNNDeviceError(err) from err
        exc_info = (None, None, None)
        try:
            yield device
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            exit_(*exc_info)
        return

    open_device = getattr(ttnn, "open_device", None)
    close_device = getattr(ttnn, "close_device", None)
    if callable(open_device):
        try:
            try:
                device = open_device(device_id=device_id)
            except TypeError:
                device = open_device(device_id)
        except Exception as err:
            raise NoTTNNDeviceError(err) from err
        try:
            yield device
        finally:
            if callable(close_device):
                close_device(device)
        return

    create_device = getattr(ttnn, "CreateDevice", None) or getattr(
        ttnn, "create_device", None
    )
    if callable(create_device):
        try:
            device = create_device(device_id)
        except Exception as err:
            raise NoTTNNDeviceError(err) from err
        try:
            yield device
        finally:
            if callable(close_device):
                close_device(device)
        return

    raise NoTTNNDeviceError("ttnn does not expose a device opener")


def _ttnn_mul_silu(ttnn: Any, gate: Any, up: Any) -> Any:
    kwargs: dict[str, Any] = {}
    unary_with_param = getattr(ttnn, "UnaryWithParam", None)
    unary_op_type = getattr(ttnn, "UnaryOpType", None)
    if unary_with_param is not None and unary_op_type is not None:
        silu = getattr(unary_op_type, "SILU", None)
        if silu is not None:
            kwargs["input_tensor_a_activations"] = [unary_with_param(silu)]
    mul = getattr(ttnn, "mul", None) or getattr(ttnn, "multiply", None)
    if mul is None:
        raise AttributeError("ttnn must provide mul or multiply")
    return mul(gate, up, **kwargs)


def _ttnn_dtype(ttnn: Any, dtype_seed: str) -> Any:
    if dtype_seed == "bf16":
        return getattr(ttnn, "bfloat16", None)
    return getattr(ttnn, "float32", None)


def _pearson_corr(torch: Any, expected: Any, actual: Any) -> float:
    lhs = expected.reshape(-1).to(torch.float32)
    rhs = actual.reshape(-1).to(torch.float32)
    lhs = lhs - torch.mean(lhs)
    rhs = rhs - torch.mean(rhs)
    denom = torch.sqrt(torch.sum(lhs * lhs) * torch.sum(rhs * rhs))
    if float(denom) == 0.0:
        return 0.0
    return float(torch.sum(lhs * rhs) / denom)


def _base_report(
    *,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_seed: str,
    dry_run: bool,
    pcc_threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "mlp_decode",
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype_seed": dtype_seed,
        "dry_run": dry_run,
        "pcc_threshold": pcc_threshold,
        "ttnn_ops": list(MLP_SMOKE_OPS),
    }


def _no_device_report(
    *,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_seed: str,
    pcc_threshold: float,
    detail: str,
) -> dict[str, Any]:
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
            "passed": False,
            "status": "no_device",
            "pcc": None,
            "latency_ms": None,
            "error": NO_TTNN_DEVICE_MESSAGE,
            "detail": detail,
        }
    )
    return report


def _failed_report(
    *,
    device: str,
    device_id: int,
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_seed: str,
    pcc_threshold: float,
    status: str,
    message: str,
    detail: str,
) -> dict[str, Any]:
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
            "passed": False,
            "status": status,
            "pcc": None,
            "latency_ms": None,
            "error": message,
            "detail": detail,
        }
    )
    return report


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")


def _validate_shape_args(
    batch_size: int, hidden_size: int, intermediate_size: int
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
