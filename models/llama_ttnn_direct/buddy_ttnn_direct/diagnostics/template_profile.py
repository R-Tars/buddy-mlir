from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..autotune.microbench import summarize_samples
from ..autotune.schema import MeasurementContract
from ..runtime.reports import write_report
from ..smoke_mlp import MLP_SMOKE_OPS, NO_TTNN_DEVICE_MESSAGE, run_smoke_mlp


MLP_PROFILE_OPS = [
    {"name": name, "count": 1}
    for name in ("linear.gate", "linear.up", "mul.silu", "linear.down")
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
        raise ValueError("template-profile only supports template=mlp_decode")
    contract = MeasurementContract(warmup=warmup, iterations=iterations)
    shape = _shape(json.loads(Path(config_path).read_text()))
    report = {
        "schema_version": 2,
        "template": template,
        **shape,
        "device_id": device_id,
        "warmup": contract.warmup,
        "iterations": contract.iterations,
        "trace_enabled": trace,
        "dry_run": dry_run,
        "dtype_seed": dtype_seed,
        "ops": MLP_PROFILE_OPS,
        "ttnn_ops": list(MLP_SMOKE_OPS),
        "autotune_measurement_contract": contract.to_dict(),
        "worker": "smoke_mlp.run_smoke_mlp",
    }
    if dry_run:
        report.update(
            status="dry_run",
            latency_ms=_latency([0.0]),
            trace={"requested": trace, "status": "dry_run" if trace else "disabled"},
            message="Dry run only; TTNN device is not required.",
        )
        write_report(out, report)
        return report

    with tempfile.TemporaryDirectory(prefix="ttnn-template-profile-") as root:
        samples, workers = [], []
        for seed in range(warmup + iterations):
            worker = run_smoke_mlp(
                out=Path(root) / f"sample-{seed}.json",
                device=shape["device"],
                device_id=device_id,
                batch_size=shape["batch_size"],
                hidden_size=shape["hidden_size"],
                intermediate_size=shape["intermediate_size"],
                dtype_seed=dtype_seed,
                pcc_threshold=0.0,
                seed=seed,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
            workers.append(worker)
            if worker.get("status") != "passed":
                break
            if seed >= warmup:
                samples.append(float(worker["latency_ms"]))
    failed = next((item for item in workers if item.get("status") != "passed"), None)
    if failed is not None:
        report.update(
            status=failed.get("status"),
            error=failed.get("error", NO_TTNN_DEVICE_MESSAGE),
            detail=failed.get("detail"),
            latency_ms=_latency([0.0]),
            trace={"requested": trace, "status": "unavailable" if trace else "disabled"},
        )
    else:
        report.update(
            status="profiled",
            latency_ms=_latency(samples),
            pcc={"min": min(item["pcc"] for item in workers), "last": workers[-1]["pcc"]},
            trace={"requested": trace, "status": "eager_worker" if trace else "disabled"},
        )
    write_report(out, report)
    return report


def _shape(config: dict[str, Any]) -> dict[str, Any]:
    raw_template = config.get("template_config")
    template = raw_template if isinstance(raw_template, dict) else {}
    hidden = int(config.get("hidden_size") or 1024)
    return {
        "model_name": str(config.get("model_name") or config.get("model") or "unknown"),
        "device": str(config.get("device") or template.get("device") or "unknown"),
        "batch_size": int(config.get("batch_size") or template.get("batch_size") or 1),
        "hidden_size": hidden,
        "intermediate_size": int(config.get("intermediate_size") or hidden * 4),
    }


def _latency(samples: list[float]) -> dict[str, float]:
    stats = summarize_samples(samples)
    return {key: float(stats[key]) for key in ("mean", "p50", "p90")}
