from __future__ import annotations

import argparse
import copy
import datetime
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .measurement import (
    MeasurementCandidate,
    candidate_fingerprint,
    with_measurement_contract,
)
from .schema import (
    CandidateConfig,
    MeasurementContract,
    canonical_json,
    sha256_json,
)
from .transfer import LayerTransferPlan

MICROBENCH_SCHEMA_VERSION = 1
WORKER_PROTOCOL_VERSION = 1
_WORKER_ENTRYPOINT = (
    "from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.microbench "
    "import main; raise SystemExit(main())"
)
_GRANULARITIES = {"op", "region"}


class MicrobenchmarkError(ValueError):
    """Raised when a benchmark request or isolated worker result is invalid."""


@dataclass(frozen=True)
class BenchmarkTarget:
    granularity: str
    name: str
    layer_group: str
    representative_layer: int
    operations: tuple[str, ...] = ()
    sample_metric: str = "latency_ms"
    sample_unit: str = "ms"

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise MicrobenchmarkError(
                f"granularity must be one of {sorted(_GRANULARITIES)}"
            )
        if not self.name:
            raise MicrobenchmarkError("benchmark target name must be non-empty")
        if not self.layer_group:
            raise MicrobenchmarkError("benchmark layer_group must be non-empty")
        if self.representative_layer < 0:
            raise MicrobenchmarkError(
                "benchmark representative_layer must be non-negative"
            )
        if self.granularity == "region" and not self.operations:
            raise MicrobenchmarkError(
                "region benchmark must identify at least one operation"
            )
        if any(not operation for operation in self.operations):
            raise MicrobenchmarkError("benchmark operations must be non-empty")
        if not self.sample_metric or not self.sample_unit:
            raise MicrobenchmarkError(
                "benchmark sample metric and unit must be non-empty"
            )

    @classmethod
    def op(
        cls,
        name: str,
        *,
        layer_group: str,
        representative_layer: int,
    ) -> "BenchmarkTarget":
        return cls(
            granularity="op",
            name=name,
            layer_group=layer_group,
            representative_layer=representative_layer,
            operations=(name,),
        )

    @classmethod
    def region(
        cls,
        name: str,
        operations: Sequence[str],
        *,
        layer_group: str,
        representative_layer: int,
    ) -> "BenchmarkTarget":
        return cls(
            granularity="region",
            name=name,
            layer_group=layer_group,
            representative_layer=representative_layer,
            operations=tuple(str(operation) for operation in operations),
        )

    def validate_transfer_plan(self, plan: LayerTransferPlan) -> None:
        group = plan.group(self.layer_group)
        if group.representative_layer != self.representative_layer:
            raise MicrobenchmarkError(
                f"target {self.name!r} selects representative layer "
                f"{self.representative_layer}, but {group.name!r} uses "
                f"{group.representative_layer}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "name": self.name,
            "layer_group": self.layer_group,
            "representative_layer": self.representative_layer,
            "operations": list(self.operations),
            "sample_metric": self.sample_metric,
            "sample_unit": self.sample_unit,
        }


@dataclass(frozen=True)
class ActiveMicrobenchmarkRunner:
    """Bind scheduler candidates to the isolated microbenchmark runtime."""

    candidate_configs: Mapping[str, CandidateConfig]
    targets: Mapping[str, BenchmarkTarget]
    transfer_plan: LayerTransferPlan
    cache_dir: str | Path
    worker: str
    payloads: Mapping[str, Mapping[str, Any]] | None = None
    python_executable: str | Path | None = None
    cwd: str | Path | None = None
    environment: Mapping[str, str] | None = None
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.candidate_configs:
            raise MicrobenchmarkError(
                "active microbenchmark runner requires candidate configs"
            )
        if not self.targets:
            raise MicrobenchmarkError(
                "active microbenchmark runner requires benchmark targets"
            )

    def __call__(
        self,
        candidate: MeasurementCandidate,
        contract: MeasurementContract,
        round_name: str,
    ) -> dict[str, Any]:
        base = self.candidate_configs.get(candidate.candidate_id)
        if base is None:
            raise MicrobenchmarkError(
                f"no CandidateConfig for {candidate.candidate_id!r}"
            )
        target = self.targets.get(candidate.candidate_id) or self.targets.get(
            candidate.operator_name
        )
        if target is None:
            raise MicrobenchmarkError(
                f"no BenchmarkTarget for {candidate.operator_name!r}"
            )
        payloads = self.payloads or {}
        payload = payloads.get(candidate.candidate_id) or payloads.get(
            candidate.operator_name
        )
        measured = run_microbenchmark(
            candidate=with_measurement_contract(base, contract),
            target=target,
            transfer_plan=self.transfer_plan,
            cache_dir=self.cache_dir,
            worker=self.worker,
            payload=payload,
            python_executable=self.python_executable,
            cwd=self.cwd,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        measured["active_scheduler"] = {
            "round": round_name,
            "operator": candidate.operator_name,
            "candidate_id": candidate.candidate_id,
            "analytical_score": candidate.analytical_score,
            "l1_bytes": candidate.l1_bytes,
            "measurement_contract": contract.to_dict(),
        }
        return measured


class MeasurementCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"

    def load(
        self,
        *,
        cache_key: str,
        identity: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        path = self.path_for(cache_key)
        if not path.is_file():
            return None, "not_found"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None, "invalid_json"
        if payload.get("schema_version") != MICROBENCH_SCHEMA_VERSION:
            return None, "schema_mismatch"
        if payload.get("cache_key") != cache_key:
            return None, "key_mismatch"
        if canonical_json(payload.get("identity")) != canonical_json(identity):
            return None, "identity_mismatch"
        report = payload.get("report")
        if not isinstance(report, dict) or report.get("status") != "passed":
            return None, "incomplete_measurement"
        return copy.deepcopy(report), "hit"

    def store(
        self,
        *,
        cache_key: str,
        identity: Mapping[str, Any],
        report: Mapping[str, Any],
    ) -> Path:
        if report.get("status") != "passed":
            raise MicrobenchmarkError("only passed measurements may be cached")
        path = self.path_for(cache_key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        payload = {
            "schema_version": MICROBENCH_SCHEMA_VERSION,
            "cache_key": cache_key,
            "identity": copy.deepcopy(dict(identity)),
            "report": copy.deepcopy(dict(report)),
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
        return path


def run_microbenchmark(
    *,
    candidate: CandidateConfig,
    target: BenchmarkTarget,
    transfer_plan: LayerTransferPlan,
    cache_dir: str | Path,
    worker: str,
    payload: Mapping[str, Any] | None = None,
    python_executable: str | Path | None = None,
    cwd: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    target.validate_transfer_plan(transfer_plan)
    if ":" not in worker:
        raise MicrobenchmarkError(
            "isolated worker must use the import path form 'module:callable'"
        )
    if timeout_seconds <= 0:
        raise MicrobenchmarkError("timeout_seconds must be positive")
    serialized_payload = _json_object(payload or {}, "benchmark payload")
    fingerprint = candidate_fingerprint(candidate)
    group = transfer_plan.group(target.layer_group)
    identity = {
        "schema_version": MICROBENCH_SCHEMA_VERSION,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "candidate_fingerprint": fingerprint,
        "candidate": candidate.to_dict(),
        "target": target.to_dict(),
        "transfer_group": group.to_dict(),
        "worker": worker,
        "payload_sha256": sha256_json(serialized_payload),
    }
    cache_key = sha256_json(identity)
    cache = MeasurementCache(cache_dir)
    cached, cache_reason = cache.load(cache_key=cache_key, identity=identity)
    cache_path = cache.path_for(cache_key)
    if cached is not None:
        cached["cache"] = {
            "hit": True,
            "measurement_reused": True,
            "key": cache_key,
            "path": str(cache_path),
            "lookup": cache_reason,
        }
        return cached

    repetitions: list[dict[str, Any]] = []
    for repetition in range(candidate.measurement_contract.repetitions):
        result = _run_isolated_repetition(
            cache=cache,
            cache_key=cache_key,
            candidate=candidate,
            target=target,
            worker=worker,
            payload=serialized_payload,
            repetition=repetition,
            python_executable=python_executable,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        repetitions.append(result)
        if result["status"] != "passed":
            break

    passed = len(repetitions) == candidate.measurement_contract.repetitions and all(
        item["status"] == "passed" for item in repetitions
    )
    raw_samples = [
        list(item.get("samples") or [])
        for item in repetitions
        if item.get("status") == "passed"
    ]
    flattened = [sample for samples in raw_samples for sample in samples]
    statistics_report = _sample_statistics(flattened) if passed else None
    report = {
        "schema_version": MICROBENCH_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "candidate_fingerprint": fingerprint,
        "runtime_commit": candidate.runtime_commit,
        "target": target.to_dict(),
        "measurement_contract": candidate.measurement_contract.to_dict(),
        "execution_contract": candidate.execution_contract.to_dict(),
        "transfer": {
            "strategy": "representative_layer_transfer",
            "group": group.to_dict(),
            "plan_sha256": sha256_json(transfer_plan.to_dict()),
        },
        "worker": {
            "callable": worker,
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "isolated_subprocess": True,
            "one_process_per_repetition": True,
        },
        "payload_sha256": sha256_json(serialized_payload),
        "statistics": statistics_report,
        "raw_samples": raw_samples,
        "repetitions": repetitions,
        "instrumentation": _aggregate_instrumentation(repetitions),
        "cache": {
            "hit": False,
            "measurement_reused": False,
            "key": cache_key,
            "path": str(cache_path),
            "lookup": cache_reason,
        },
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error": next(
            (
                item.get("error")
                for item in repetitions
                if item.get("status") != "passed"
            ),
            None,
        ),
    }
    if passed:
        cache.store(cache_key=cache_key, identity=identity, report=report)
    return report


def make_worker_response(
    request: Mapping[str, Any],
    samples: Sequence[float],
    *,
    program_cache_count: int | None = None,
    trace_capture_count: int | None = None,
    new_tensor_allocations: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = request.get("candidate")
    target = request.get("target")
    if not isinstance(candidate, Mapping) or not isinstance(target, Mapping):
        raise MicrobenchmarkError("worker request is missing candidate or target")
    measurement = candidate.get("measurement_contract")
    execution = candidate.get("execution_contract")
    if not isinstance(measurement, Mapping) or not isinstance(execution, Mapping):
        raise MicrobenchmarkError("worker request is missing frozen contracts")
    return {
        "schema_version": MICROBENCH_SCHEMA_VERSION,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "status": "passed",
        "repetition": int(request["repetition"]),
        "worker_pid": os.getpid(),
        "warmup_completed": int(measurement["warmup"]),
        "execution_contract": copy.deepcopy(dict(execution)),
        "sample_metric": target["sample_metric"],
        "sample_unit": target["sample_unit"],
        "samples": [float(sample) for sample in samples],
        "instrumentation": {
            "program_cache_count": program_cache_count,
            "trace_capture_count": trace_capture_count,
            "new_tensor_allocations": new_tensor_allocations,
        },
        "metadata": _json_object(metadata or {}, "worker metadata"),
    }


def build_microbench_report(
    results: Sequence[Mapping[str, Any]],
    *,
    transfer_plan: LayerTransferPlan,
) -> dict[str, Any]:
    normalized = [copy.deepcopy(dict(result)) for result in results]
    passed = bool(normalized) and all(
        result.get("status") == "passed" for result in normalized
    )
    return {
        "schema_version": MICROBENCH_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "measurement_count": len(normalized),
        "cache_hit_count": sum(
            1 for result in normalized if (result.get("cache") or {}).get("hit")
        ),
        "granularity_counts": {
            granularity: sum(
                1
                for result in normalized
                if (result.get("target") or {}).get("granularity") == granularity
            )
            for granularity in sorted(_GRANULARITIES)
        },
        "transfer_report": transfer_plan.to_dict(),
        "measurements": normalized,
    }


def write_microbench_report(
    path: str | Path,
    results: Sequence[Mapping[str, Any]],
    *,
    transfer_plan: LayerTransferPlan,
) -> dict[str, Any]:
    report = build_microbench_report(results, transfer_plan=transfer_plan)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    return report


def _run_isolated_repetition(
    *,
    cache: MeasurementCache,
    cache_key: str,
    candidate: CandidateConfig,
    target: BenchmarkTarget,
    worker: str,
    payload: Mapping[str, Any],
    repetition: int,
    python_executable: str | Path | None,
    cwd: str | Path | None,
    environment: Mapping[str, str] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    work_root = cache.root / "work"
    log_root = cache.root / "process_logs"
    work_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{cache_key[:12]}-r{repetition}-", dir=work_root
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        request_path = temporary / "request.json"
        response_path = temporary / "response.json"
        request = {
            "schema_version": MICROBENCH_SCHEMA_VERSION,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "repetition": repetition,
            "candidate": candidate.to_dict(),
            "candidate_fingerprint": candidate_fingerprint(candidate),
            "target": target.to_dict(),
            "payload": copy.deepcopy(dict(payload)),
        }
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        command = [
            str(python_executable or sys.executable),
            "-c",
            _WORKER_ENTRYPOINT,
            "--isolated-worker",
            worker,
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        child_environment = os.environ.copy()
        if environment:
            child_environment.update(
                {str(key): str(value) for key, value in environment.items()}
            )
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=child_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()

        log_path = log_root / f"{cache_key}.rep{repetition}.log"
        log_path.write_text(
            f"pid={process.pid}\n"
            f"exit_code={process.returncode}\n"
            f"timed_out={str(timed_out).lower()}\n"
            f"command={json.dumps(command)}\n"
            f"--- stdout ---\n{stdout}"
            f"--- stderr ---\n{stderr}"
        )
        process_report = {
            "pid": process.pid,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "log": str(log_path),
            "stdout_tail": stdout[-4096:],
            "stderr_tail": stderr[-4096:],
        }
        response: dict[str, Any] | None = None
        response_error: str | None = None
        if response_path.is_file():
            try:
                raw_response = json.loads(response_path.read_text())
                if isinstance(raw_response, dict):
                    response = raw_response
                else:
                    response_error = "isolated worker response is not an object"
            except (OSError, json.JSONDecodeError) as exc:
                response_error = f"unable to read isolated worker response: {exc}"
        process_report["worker_response"] = copy.deepcopy(response)
        process_report["worker_response_error"] = response_error
        if timed_out:
            return _failed_repetition(
                repetition,
                process_report,
                f"isolated worker timed out after {timeout_seconds:.3f}s",
            )
        if process.returncode != 0:
            worker_error = response.get("error") if response is not None else None
            detail = f": {worker_error}" if worker_error else ""
            return _failed_repetition(
                repetition,
                process_report,
                f"isolated worker exited with status {process.returncode}{detail}",
            )
        if response is None:
            return _failed_repetition(
                repetition,
                process_report,
                response_error or "isolated worker did not write a response",
            )
        try:
            samples, instrumentation, metadata = _validate_worker_response(
                response,
                request=request,
            )
        except MicrobenchmarkError as exc:
            return _failed_repetition(
                repetition,
                process_report,
                f"invalid isolated worker response: {exc}",
            )
        return {
            "repetition": repetition,
            "status": "passed",
            "passed": True,
            "samples": samples,
            "statistics": _sample_statistics(samples),
            "instrumentation": instrumentation,
            "metadata": metadata,
            "process": process_report,
            "error": None,
        }


def _validate_worker_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> tuple[list[float], dict[str, int | None], dict[str, Any]]:
    if response.get("schema_version") != MICROBENCH_SCHEMA_VERSION:
        raise MicrobenchmarkError("response schema_version mismatch")
    if response.get("worker_protocol_version") != WORKER_PROTOCOL_VERSION:
        raise MicrobenchmarkError("worker protocol version mismatch")
    if response.get("status") != "passed":
        raise MicrobenchmarkError(str(response.get("error") or "worker failed"))
    if int(response.get("repetition", -1)) != int(request["repetition"]):
        raise MicrobenchmarkError("worker repetition acknowledgement mismatch")
    candidate = request["candidate"]
    measurement = candidate["measurement_contract"]
    execution = candidate["execution_contract"]
    if int(response.get("warmup_completed", -1)) != int(measurement["warmup"]):
        raise MicrobenchmarkError("worker did not acknowledge every warmup")
    if canonical_json(response.get("execution_contract")) != canonical_json(execution):
        raise MicrobenchmarkError("worker execution contract acknowledgement mismatch")
    target = request["target"]
    for key in ("sample_metric", "sample_unit"):
        if response.get(key) != target[key]:
            raise MicrobenchmarkError(f"worker {key} mismatch")
    raw_samples = response.get("samples")
    if not isinstance(raw_samples, list):
        raise MicrobenchmarkError("worker samples must be a list")
    expected_count = int(measurement["iterations"])
    if len(raw_samples) != expected_count:
        raise MicrobenchmarkError(
            f"worker returned {len(raw_samples)} samples; expected {expected_count}"
        )
    samples: list[float] = []
    for sample in raw_samples:
        value = float(sample)
        if not math.isfinite(value) or value <= 0:
            raise MicrobenchmarkError("worker samples must be finite positive numbers")
        samples.append(value)

    raw_instrumentation = response.get("instrumentation")
    if raw_instrumentation is None:
        raw_instrumentation = {}
    if not isinstance(raw_instrumentation, Mapping):
        raise MicrobenchmarkError("worker instrumentation must be an object")
    instrumentation: dict[str, int | None] = {}
    for key in (
        "program_cache_count",
        "trace_capture_count",
        "new_tensor_allocations",
    ):
        raw_value = raw_instrumentation.get(key)
        if raw_value is None:
            instrumentation[key] = None
            continue
        value = int(raw_value)
        if value < 0:
            raise MicrobenchmarkError(f"worker {key} must be non-negative")
        instrumentation[key] = value
    metadata = _json_object(response.get("metadata") or {}, "worker metadata")
    return samples, instrumentation, metadata


def _sample_statistics(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        raise MicrobenchmarkError("cannot summarize an empty sample set")
    values = [float(sample) for sample in samples]
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    return {
        "sample_count": len(values),
        "mean": mean,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else None,
        "coefficient_of_variation_percent": (stdev / mean * 100.0 if mean else None),
        "min": min(values),
        "max": max(values),
    }


def summarize_samples(samples: Sequence[float]) -> dict[str, Any]:
    """Public statistics helper shared by small diagnostic adapters."""

    return _sample_statistics(samples)


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(sample) for sample in samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate_instrumentation(
    repetitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keys = (
        "program_cache_count",
        "trace_capture_count",
        "new_tensor_allocations",
    )
    per_repetition = [
        copy.deepcopy(dict(item.get("instrumentation") or {}))
        for item in repetitions
        if item.get("status") == "passed"
    ]
    result: dict[str, Any] = {"per_repetition": per_repetition}
    for key in keys:
        values = [
            int(item[key]) for item in per_repetition if item.get(key) is not None
        ]
        result[key] = max(values) if key == "program_cache_count" and values else None
        if key != "program_cache_count":
            result[key] = sum(values) if values else None
    return result


def _failed_repetition(
    repetition: int,
    process: Mapping[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "repetition": repetition,
        "status": "failed",
        "passed": False,
        "samples": [],
        "statistics": None,
        "instrumentation": {
            "program_cache_count": None,
            "trace_capture_count": None,
            "new_tensor_allocations": None,
        },
        "metadata": {},
        "process": copy.deepcopy(dict(process)),
        "error": message,
    }


def _json_object(value: Any, label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise MicrobenchmarkError(f"{label} must be JSON serializable") from exc
    if not isinstance(normalized, dict):
        raise MicrobenchmarkError(f"{label} must be an object")
    return normalized


def _load_worker(path: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    module_name, separator, attribute = path.rpartition(":")
    if not separator or not module_name or not attribute:
        raise MicrobenchmarkError(
            "worker must use the import path form 'module:callable'"
        )
    module = importlib.import_module(module_name)
    worker = getattr(module, attribute, None)
    if not callable(worker):
        raise MicrobenchmarkError(f"worker is not callable: {path}")
    return worker


def _isolated_worker_main(args: argparse.Namespace) -> int:
    response_path = Path(args.response)
    try:
        request = json.loads(Path(args.request).read_text())
        worker = _load_worker(args.isolated_worker)
        response = worker(request)
        if not isinstance(response, Mapping):
            raise MicrobenchmarkError("worker must return a mapping")
        payload = dict(response)
        exit_code = 0 if payload.get("status") == "passed" else 1
    except Exception as exc:
        payload = {
            "schema_version": MICROBENCH_SCHEMA_VERSION,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    response_path.write_text(json.dumps(payload, indent=2) + "\n")
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autotune microbenchmark worker")
    parser.add_argument("--isolated-worker", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _isolated_worker_main(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
