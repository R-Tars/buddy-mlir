from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .legality import DeviceDescriptor
from .measurement import file_sha256, validate_decode_profile_contract
from .schema import (
    ContractViolation,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    sha256_json,
)

BASELINE_SCHEMA_VERSION = 1
BASELINE_REPETITIONS = 3
BASELINE_MINIMUM_TOKENS_PER_SECOND_PER_USER = 33.5
BASELINE_MAXIMUM_CV = 0.015

_REQUIRED_ARTIFACT_FILES = (
    "baseline_decode.json",
    "baseline_op_graph.json",
    "baseline_config.json",
    "baseline_runtime_identity.json",
    "baseline_source_manifest.json",
)
_REQUIRED_PROGRAM_FILES = (
    "config.json",
    "execution_plan.json",
    "model.py",
)
_EXPECTED_REPORT_FIELDS = {
    "mode": "decode-steady",
    "layers": 32,
    "batch_size": 32,
    "prefill_len": 256,
    "cache_len": 1024,
    "warmup": 5,
    "iterations": 100,
    "execution_mode": "trace",
    "runtime_input_mode": "persistent",
    "after_prefill": True,
    "prefill_execution_mode": "eager",
    "device": "p150a",
    "device_id": 0,
    "trace_capture_count": 1,
    "trace_execute_count": 105,
    "persistent_input_count": 7,
    "program_compile_count_after_capture": 0,
}


class BaselineArtifactError(ValueError):
    """Raised when Phase 0 evidence is incomplete or violates its contract."""


def build_baseline_artifact(
    report_paths: Sequence[str | Path],
    *,
    program_dir: str | Path,
    seed_config: str | Path,
    prompt_corpus: str | Path,
    ttnn_binary: str | Path,
    out_dir: str | Path,
    repo_root: str | Path | None = None,
    buddy_commit: str | None = None,
    tt_metal_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the immutable Phase 0 baseline bundle from three profile reports."""

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    failure_path = destination / "baseline_failure.json"
    try:
        reports, resolved_reports = _load_reports(report_paths)
        resolved_program_dir = _require_directory(program_dir, "program directory")
        program_paths = {
            name: _require_file(resolved_program_dir / name, f"program {name}")
            for name in _REQUIRED_PROGRAM_FILES
        }
        resolved_seed_config = _require_file(seed_config, "seed config")
        resolved_prompt_corpus = _require_file(prompt_corpus, "prompt corpus")
        resolved_ttnn_binary = _require_file(ttnn_binary, "TTNN binary")

        config = _read_json_object(program_paths["config.json"], "program config")
        execution_plan = _read_json_object(
            program_paths["execution_plan.json"], "execution plan"
        )
        seed = _read_json_object(resolved_seed_config, "seed config")
        prompt_sha256 = file_sha256(resolved_prompt_corpus)
        precision_contract = PrecisionContract.from_template_config(seed)
        execution_contract = ExecutionContract.from_template_config(
            seed, prompt_corpus_sha256=prompt_sha256
        )
        measurement_contract = MeasurementContract.final_confirmation()
        _validate_config_contract(config, seed)
        normalized_reports = _validate_reports(
            reports,
            resolved_reports=resolved_reports,
            program_dir=resolved_program_dir,
            program_config=config,
        )

        throughput = [
            float(report["tokens_per_second_per_user"]) for report in reports
        ]
        throughput_mean = statistics.fmean(throughput)
        throughput_stdev = statistics.pstdev(throughput)
        throughput_cv = throughput_stdev / throughput_mean
        throughput_median = statistics.median(throughput)
        checks = [
            _check(
                "baseline.median_tokens_per_second_per_user",
                throughput_median
                >= BASELINE_MINIMUM_TOKENS_PER_SECOND_PER_USER,
                throughput_median,
                BASELINE_MINIMUM_TOKENS_PER_SECOND_PER_USER,
                comparison=">=",
            ),
            _check(
                "baseline.coefficient_of_variation",
                throughput_cv <= BASELINE_MAXIMUM_CV,
                throughput_cv,
                BASELINE_MAXIMUM_CV,
                comparison="<=",
            ),
        ]
        failed_checks = [row["name"] for row in checks if not row["passed"]]
        if failed_checks:
            raise BaselineArtifactError(
                "baseline acceptance failed: " + ", ".join(failed_checks)
            )

        tt_metal_commit, resolved_tt_metal_root = _resolve_tt_metal_identity(
            reports, tt_metal_root=tt_metal_root
        )
        resolved_repo_root = (
            _require_directory(repo_root, "Buddy repository")
            if repo_root is not None
            else _discover_git_root(Path(__file__).resolve())
        )
        resolved_buddy_commit = buddy_commit or _git_commit(resolved_repo_root)
        _validate_commit("Buddy commit", resolved_buddy_commit)

        baseline_config_hash = sha256_json(config)
        trace_key = reports[0]["trace_key"]
        trace_identity = sha256_json(trace_key)
        baseline_decode = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "autotune-baseline",
            "status": "passed",
            "passed": True,
            "measurement_contract": measurement_contract.to_dict(),
            "workload": {
                key: value
                for key, value in _EXPECTED_REPORT_FIELDS.items()
                if key
                not in {
                    "trace_capture_count",
                    "trace_execute_count",
                    "persistent_input_count",
                    "program_compile_count_after_capture",
                }
            },
            "trace_contract": {
                "trace_capture_count": 1,
                "trace_execute_count": 105,
                "persistent_input_count": 7,
                "program_compile_count_after_capture": 0,
                "fixed_page_table": True,
                "device_token_handoff": True,
                "force_argmax": True,
                "trace_key": trace_key,
                "trace_identity_sha256": trace_identity,
            },
            "repetitions": normalized_reports,
            "summary": {
                "tokens_per_second_per_user": {
                    "values": throughput,
                    "mean": throughput_mean,
                    "median": throughput_median,
                    "p90": _percentile(throughput, 0.90),
                    "stdev": throughput_stdev,
                    "coefficient_of_variation": throughput_cv,
                    "coefficient_of_variation_percent": throughput_cv * 100.0,
                },
                "decode_step_ms": _statistics(
                    [
                        sample
                        for report in normalized_reports
                        for sample in report["decode_step_ms_samples"]
                    ]
                ),
            },
            "acceptance": {
                "status": "passed",
                "passed": True,
                "checks": checks,
                "failed_checks": [],
            },
        }
        autotune = config.get("autotune")
        if not isinstance(autotune, Mapping):
            raise BaselineArtifactError("program config is missing autotune graph")
        operators = autotune.get("operators")
        edges = autotune.get("edges")
        if not isinstance(operators, Mapping) or not isinstance(edges, Mapping):
            raise BaselineArtifactError(
                "program autotune graph must contain operators and edges"
            )
        baseline_op_graph = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "autotune-baseline-op-graph",
            "status": "passed",
            "passed": True,
            "program_config_sha256": baseline_config_hash,
            "execution_plan_sha256": sha256_json(execution_plan),
            "operator_count": len(operators),
            "edge_count": len(edges),
            "templates": autotune.get("templates") or {},
            "operators": dict(operators),
            "edges": dict(edges),
            "memory_configs": autotune.get("memory_configs") or {},
            "core_grids": autotune.get("core_grids") or {},
            "extra_program_configs": autotune.get("extra_program_configs") or {},
            "execution_plan": execution_plan,
        }
        runtime_identity = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "autotune-baseline-runtime-identity",
            "status": "passed",
            "passed": True,
            "buddy": {
                "repository": str(resolved_repo_root),
                "commit": resolved_buddy_commit,
            },
            "tt_metal": {
                "repository": str(resolved_tt_metal_root),
                "commit": tt_metal_commit,
            },
            "ttnn_binary": _file_identity(resolved_ttnn_binary),
            "device_descriptor": DeviceDescriptor.p150a().to_dict(),
            "prompt_corpus": _file_identity(resolved_prompt_corpus),
            "seed_config": _file_identity(resolved_seed_config),
            "program": {
                "directory": str(resolved_program_dir),
                "config_sha256": baseline_config_hash,
                "config_source_sha256": file_sha256(program_paths["config.json"]),
                "execution_plan_sha256": sha256_json(execution_plan),
                "execution_plan_source_sha256": file_sha256(
                    program_paths["execution_plan.json"]
                ),
                "generated_model_sha256": file_sha256(program_paths["model.py"]),
            },
            "precision_contract": precision_contract.to_dict(),
            "execution_contract": execution_contract.to_dict(),
            "trace_identity_sha256": trace_identity,
            "profile_report_sha256": [
                file_sha256(path) for path in resolved_reports
            ],
        }

        _atomic_write_json(destination / "baseline_decode.json", baseline_decode)
        _atomic_write_json(destination / "baseline_op_graph.json", baseline_op_graph)
        _atomic_write_json(destination / "baseline_config.json", config)
        _atomic_write_json(
            destination / "baseline_runtime_identity.json", runtime_identity
        )
        source_paths = {
            **{
                f"profile_report_{index}": path
                for index, path in enumerate(resolved_reports)
            },
            "program_config": program_paths["config.json"],
            "execution_plan": program_paths["execution_plan.json"],
            "generated_model": program_paths["model.py"],
            "seed_config": resolved_seed_config,
            "prompt_corpus": resolved_prompt_corpus,
            "ttnn_binary": resolved_ttnn_binary,
        }
        source_manifest = {
            label: _file_identity(path) for label, path in source_paths.items()
        }
        _atomic_write_json(
            destination / "baseline_source_manifest.json", source_manifest
        )
        artifact_manifest = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "autotune-baseline-manifest",
            "files": {
                filename: _file_identity(destination / filename, include_path=False)
                for filename in _REQUIRED_ARTIFACT_FILES
            },
        }
        _atomic_write_json(
            destination / "baseline_artifact_manifest.json", artifact_manifest
        )
        if failure_path.exists():
            failure_path.unlink()
        verification = verify_baseline_artifact(destination)
        if not verification["passed"]:
            raise BaselineArtifactError(
                "freshly built baseline did not verify: "
                + ", ".join(verification["errors"])
            )
        return baseline_decode
    except Exception as exc:
        failure = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "autotune-baseline",
            "status": "failed",
            "passed": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "failure_report_written": True,
        }
        _atomic_write_json(failure_path, failure)
        return failure


def verify_baseline_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    errors: list[str] = []
    if (root / "baseline_failure.json").is_file():
        errors.append("baseline_failure.json is present")

    artifacts = {
        filename: _try_read_json(root / filename)
        for filename in _REQUIRED_ARTIFACT_FILES
    }
    manifest = _try_read_json(root / "baseline_artifact_manifest.json")
    for filename, payload in artifacts.items():
        if not isinstance(payload, dict):
            errors.append(f"baseline artifact is missing or malformed: {filename}")
    if not isinstance(manifest, dict):
        errors.append("baseline_artifact_manifest.json is missing or malformed")

    decode = artifacts.get("baseline_decode.json")
    identity = artifacts.get("baseline_runtime_identity.json")
    config = artifacts.get("baseline_config.json")
    op_graph = artifacts.get("baseline_op_graph.json")
    if isinstance(decode, dict):
        if decode.get("schema_version") != BASELINE_SCHEMA_VERSION:
            errors.append("baseline decode schema version mismatch")
        if decode.get("status") != "passed" or not decode.get("passed"):
            errors.append("baseline decode is not passed")
        repetitions = decode.get("repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != BASELINE_REPETITIONS:
            errors.append("baseline decode must contain exactly three repetitions")
        else:
            for index, repetition in enumerate(repetitions):
                samples = repetition.get("decode_step_ms_samples")
                if not isinstance(samples, list) or len(samples) != 100:
                    errors.append(
                        f"baseline repetition {index} must contain 100 raw samples"
                    )
        acceptance = decode.get("acceptance")
        if not isinstance(acceptance, dict) or not acceptance.get("passed"):
            errors.append("baseline acceptance is not passed")
    if isinstance(identity, dict):
        if identity.get("schema_version") != BASELINE_SCHEMA_VERSION:
            errors.append("baseline runtime identity schema version mismatch")
        try:
            PrecisionContract.from_dict(identity.get("precision_contract") or {})
        except Exception as exc:
            errors.append(f"baseline precision contract is invalid: {exc}")
    if isinstance(config, dict) and isinstance(identity, dict):
        expected = ((identity.get("program") or {}).get("config_sha256"))
        if sha256_json(config) != expected:
            errors.append("baseline config hash mismatch")
    if isinstance(op_graph, dict) and isinstance(config, dict):
        if op_graph.get("program_config_sha256") != sha256_json(config):
            errors.append("baseline op graph config hash mismatch")

    if isinstance(manifest, dict):
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            errors.append("baseline artifact manifest has no files object")
        else:
            for filename in _REQUIRED_ARTIFACT_FILES:
                record = files.get(filename)
                path = root / filename
                if not isinstance(record, Mapping):
                    errors.append(f"artifact manifest entry is missing: {filename}")
                    continue
                _verify_file_record(path, record, errors, label="artifact")

    source_manifest = artifacts.get("baseline_source_manifest.json")
    if isinstance(source_manifest, dict):
        for label, record in source_manifest.items():
            if not isinstance(record, Mapping):
                errors.append(f"source manifest entry is malformed: {label}")
                continue
            path = Path(str(record.get("path", "")))
            _verify_file_record(path, record, errors, label=f"source {label}")

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "stage": "autotune-baseline-verification",
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "artifact_dir": str(root),
        "errors": errors,
        "checked_artifact_file_count": len(_REQUIRED_ARTIFACT_FILES),
        "checked_source_count": (
            len(source_manifest) if isinstance(source_manifest, dict) else 0
        ),
    }


def _load_reports(
    report_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[Path]]:
    if len(report_paths) != BASELINE_REPETITIONS:
        raise BaselineArtifactError(
            f"baseline requires exactly {BASELINE_REPETITIONS} profile reports"
        )
    resolved = [
        _require_file(path, f"profile report {index}")
        for index, path in enumerate(report_paths)
    ]
    if len(set(resolved)) != len(resolved):
        raise BaselineArtifactError("baseline profile reports must be distinct files")
    return [
        _read_json_object(path, f"profile report {index}")
        for index, path in enumerate(resolved)
    ], resolved


def _validate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    resolved_reports: Sequence[Path],
    program_dir: Path,
    program_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_trace_key: Any = None
    expected_tt_metal_commit: str | None = None
    normalized: list[dict[str, Any]] = []
    for index, (report, path) in enumerate(zip(reports, resolved_reports)):
        try:
            validate_decode_profile_contract(
                report,
                program_config=program_config,
                expected_layers=_EXPECTED_REPORT_FIELDS["layers"],
                measurement_contract=MeasurementContract.final_confirmation(),
                expected_trace_capture_count=1,
                expected_trace_execute_count=105,
                expected_workload=_EXPECTED_REPORT_FIELDS,
            )
        except ContractViolation as exc:
            raise BaselineArtifactError(
                f"profile report {index} violates the baseline contract: {exc}"
            ) from exc
        mismatches: dict[str, Any] = {}
        reported_program_dir = Path(str(report.get("program_dir", ""))).resolve()
        if reported_program_dir != program_dir:
            mismatches["program_dir"] = {
                "expected": str(program_dir),
                "observed": str(reported_program_dir),
            }
        acceptance = report.get("acceptance")
        if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
            mismatches["acceptance"] = {
                "expected": "passed",
                "observed": acceptance,
            }
        samples = report.get("decode_step_ms_samples")
        if (
            not isinstance(samples, list)
            or len(samples) != 100
            or any(not _positive_number(value) for value in samples)
        ):
            mismatches["decode_step_ms_samples"] = {
                "expected": "100 positive finite values",
                "observed": len(samples) if isinstance(samples, list) else None,
            }
        throughput = report.get("tokens_per_second_per_user")
        if not _positive_number(throughput):
            mismatches["tokens_per_second_per_user"] = {
                "expected": "positive finite value",
                "observed": throughput,
            }
        trace_key = report.get("trace_key")
        if not isinstance(trace_key, Mapping):
            mismatches["trace_key"] = {
                "expected": "object",
                "observed": trace_key,
            }
        elif expected_trace_key is None:
            expected_trace_key = trace_key
        elif trace_key != expected_trace_key:
            mismatches["trace_key"] = {
                "expected": expected_trace_key,
                "observed": trace_key,
            }
        environment = report.get("ttnn_environment")
        commit = (
            environment.get("tt_metal_git_commit")
            if isinstance(environment, Mapping)
            else None
        )
        if not isinstance(commit, str):
            mismatches["tt_metal_git_commit"] = {
                "expected": "40-character git commit",
                "observed": commit,
            }
        elif expected_tt_metal_commit is None:
            expected_tt_metal_commit = commit
        elif commit != expected_tt_metal_commit:
            mismatches["tt_metal_git_commit"] = {
                "expected": expected_tt_metal_commit,
                "observed": commit,
            }
        if mismatches:
            raise BaselineArtifactError(
                f"profile report {index} violates the baseline contract: "
                + json.dumps(mismatches, sort_keys=True)
            )

        values = [float(value) for value in samples]
        normalized.append(
            {
                "index": index,
                "source_path": str(path),
                "source_sha256": file_sha256(path),
                "tokens_per_second_per_user": float(throughput),
                "decode_step_ms": _statistics(values),
                "decode_step_ms_samples": values,
                "warmup_step_ms_samples": [
                    float(value) for value in report.get("warmup_step_ms_samples") or []
                ],
                "trace_capture_count": report["trace_capture_count"],
                "trace_execute_count": report["trace_execute_count"],
                "persistent_input_count": report["persistent_input_count"],
                "program_compile_count_after_capture": report[
                    "program_compile_count_after_capture"
                ],
            }
        )
    return normalized


def _validate_config_contract(
    config: Mapping[str, Any], seed: Mapping[str, Any]
) -> None:
    expected = {
        "official_config_profile": seed.get("official_config_profile"),
        "runtime_input_mode": "persistent",
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise BaselineArtifactError(
            "generated program violates the frozen config contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _resolve_tt_metal_identity(
    reports: Sequence[Mapping[str, Any]],
    *,
    tt_metal_root: str | Path | None,
) -> tuple[str, Path]:
    environment = reports[0]["ttnn_environment"]
    reported_commit = str(environment["tt_metal_git_commit"])
    _validate_commit("tt-metal commit", reported_commit)
    root_value = tt_metal_root or environment.get("tt_metal_home")
    if not root_value:
        raise BaselineArtifactError("tt-metal repository path is unavailable")
    root = _require_directory(root_value, "tt-metal repository")
    observed_commit = _git_commit(root)
    if observed_commit != reported_commit:
        raise BaselineArtifactError(
            "tt-metal report commit differs from repository HEAD: "
            f"reported {reported_commit}, observed {observed_commit}"
        )
    return reported_commit, root


def _statistics(samples: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in samples]
    if not values:
        raise BaselineArtifactError("latency statistics require at least one sample")
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    return {
        "count": len(values),
        "mean": mean,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean,
        "coefficient_of_variation_percent": stdev / mean * 100.0,
    }


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _check(
    name: str,
    passed: bool,
    observed: float,
    expected: float,
    *,
    comparison: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "comparison": comparison,
    }


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _discover_git_root(start: Path) -> Path:
    output = _run_git(start.parent, "rev-parse", "--show-toplevel")
    return Path(output).resolve()


def _git_commit(root: Path) -> str:
    commit = _run_git(root, "rev-parse", "HEAD")
    _validate_commit(f"git commit for {root}", commit)
    return commit


def _run_git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineArtifactError(
            f"failed to query git repository {root}: {exc}"
        ) from exc
    return result.stdout.strip()


def _validate_commit(label: str, value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise BaselineArtifactError(f"{label} is not a full lowercase git commit")


def _require_file(value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise BaselineArtifactError(f"{label} is missing: {path}")
    return path


def _require_directory(value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise BaselineArtifactError(f"{label} is missing: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineArtifactError(f"{label} is malformed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineArtifactError(f"{label} must be a JSON object: {path}")
    return payload


def _try_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _file_identity(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if include_path:
        record["path"] = str(path)
    return record


def _verify_file_record(
    path: Path,
    record: Mapping[str, Any],
    errors: list[str],
    *,
    label: str,
) -> None:
    if not path.is_file():
        errors.append(f"{label} file is missing: {path}")
        return
    if file_sha256(path) != record.get("sha256"):
        errors.append(f"{label} file hash mismatch: {path.name}")
    if path.stat().st_size != record.get("bytes"):
        errors.append(f"{label} file size mismatch: {path.name}")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
