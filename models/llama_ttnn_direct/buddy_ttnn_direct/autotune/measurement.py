from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..semantic.graph import LlamaModelGraph, graph_to_dict
from .schema import (
    CandidateConfig,
    ContractViolation,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    canonical_json,
    sha256_json,
)


@dataclass(frozen=True)
class MeasurementCandidate:
    """A legal candidate plus the analytical evidence used to schedule it."""

    candidate_id: str
    operator_name: str
    candidate_kind: str
    analytical_score: float
    l1_bytes: int
    source: str
    is_incumbent: bool = False
    _metadata_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.operator_name or not self.candidate_kind:
            raise ContractViolation(
                "measurement candidate id, operator, and kind must be non-empty"
            )
        if not self.source:
            raise ContractViolation("measurement candidate source must be non-empty")
        if not math.isfinite(float(self.analytical_score)):
            raise ContractViolation("analytical score must be finite")
        if float(self.analytical_score) < 0.0:
            raise ContractViolation("analytical score must be non-negative")
        if self.l1_bytes < 0:
            raise ContractViolation("candidate l1_bytes must be non-negative")
        metadata = json.loads(self._metadata_json)
        if not isinstance(metadata, dict):
            raise ContractViolation("measurement candidate metadata must be an object")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        operator_name: str,
        candidate_kind: str,
        analytical_score: float,
        l1_bytes: int,
        source: str,
        is_incumbent: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MeasurementCandidate":
        return cls(
            candidate_id=str(candidate_id),
            operator_name=str(operator_name),
            candidate_kind=str(candidate_kind),
            analytical_score=float(analytical_score),
            l1_bytes=int(l1_bytes),
            source=str(source),
            is_incumbent=bool(is_incumbent),
            _metadata_json=canonical_json(metadata or {}),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        value = json.loads(self._metadata_json)
        if not isinstance(value, dict):
            raise ContractViolation("measurement candidate metadata must be an object")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "operator": self.operator_name,
            "candidate_kind": self.candidate_kind,
            "analytical_score": self.analytical_score,
            "l1_bytes": self.l1_bytes,
            "source": self.source,
            "is_incumbent": self.is_incumbent,
            "metadata": self.metadata,
        }


def build_candidate_config(
    *,
    graph: LlamaModelGraph,
    model_root: str | Path,
    precision_contract: PrecisionContract,
    expected_precision_hash: str,
    execution_contract: ExecutionContract,
    measurement_contract: MeasurementContract,
    runtime_commit: str,
    device: str,
    device_id: int,
    target: Mapping[str, Any],
    tunable_state: Mapping[str, Any],
) -> CandidateConfig:
    root = Path(model_root)
    return CandidateConfig.create(
        precision_contract=precision_contract,
        expected_precision_hash=expected_precision_hash,
        execution_contract=execution_contract,
        measurement_contract=measurement_contract,
        semantic_graph_sha256=sha256_json(graph_to_dict(graph)),
        model_config_sha256=file_sha256(root / "config.json"),
        weights_recipe_sha256=weights_recipe_sha256(root, precision_contract),
        runtime_commit=runtime_commit,
        device_descriptor={
            "device": str(device),
            "device_id": int(device_id),
            "architecture": _device_architecture(device),
        },
        target=target,
        tunable_state=tunable_state,
    )


def with_measurement_contract(
    candidate: CandidateConfig,
    measurement_contract: MeasurementContract,
) -> CandidateConfig:
    """Clone a candidate identity for one successive-halving round."""

    return CandidateConfig.create(
        precision_contract=candidate.precision_contract,
        expected_precision_hash=candidate.expected_precision_hash,
        execution_contract=candidate.execution_contract,
        measurement_contract=measurement_contract,
        semantic_graph_sha256=candidate.semantic_graph_sha256,
        model_config_sha256=candidate.model_config_sha256,
        weights_recipe_sha256=candidate.weights_recipe_sha256,
        runtime_commit=candidate.runtime_commit,
        device_descriptor=candidate.device_descriptor,
        target=candidate.target,
        tunable_state=candidate.tunable_state,
    )


def candidate_fingerprint(candidate: CandidateConfig) -> str:
    return sha256_json(candidate.identity_payload())


def prompt_corpus_sha256(prompt: str | None) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ContractViolation(f"required fingerprint input is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def force_argmax_enabled(program_config: Mapping[str, Any]) -> bool:
    generation = program_config.get("generation")
    if not isinstance(generation, Mapping):
        return False
    return (
        "argmax" in str(generation.get("template", "")).lower()
        and str(generation.get("mode", "")).lower() == "greedy"
    )


def validate_decode_profile_contract(
    profile: Mapping[str, Any],
    *,
    program_config: Mapping[str, Any],
    expected_layers: int,
    measurement_contract: MeasurementContract,
    expected_trace_capture_count: int,
    expected_trace_execute_count: int,
    expected_workload: Mapping[str, Any] | None = None,
    require_runtime_input_stability: bool = True,
    handoff_evidence: str = "runtime_inputs",
) -> dict[str, Any]:
    """Validate the production trace/persistent steady-decode contract."""

    expected = {
        **dict(expected_workload or {}),
        "status": "profiled",
        "passed": True,
        "layers": expected_layers,
        "warmup": measurement_contract.warmup,
        "iterations": measurement_contract.iterations,
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "after_prefill": True,
        "trace_capture_count": expected_trace_capture_count,
        "trace_execute_count": expected_trace_execute_count,
        "program_compile_count_after_capture": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": profile.get(key)}
        for key, value in expected.items()
        if profile.get(key) != value
    }
    if program_config.get("num_layers") != expected_layers:
        mismatches["program_num_layers"] = {
            "expected": expected_layers,
            "observed": program_config.get("num_layers"),
        }

    persistent_count = profile.get("persistent_input_count")
    if (
        not isinstance(persistent_count, int)
        or isinstance(persistent_count, bool)
        or persistent_count <= 0
    ):
        mismatches["persistent_input_count"] = {
            "expected": "positive integer",
            "observed": persistent_count,
        }

    if handoff_evidence not in {"runtime_inputs", "runtime_context"}:
        raise ContractViolation(f"unsupported handoff evidence policy: {handoff_evidence}")

    runtime_inputs = profile.get("runtime_inputs")
    if require_runtime_input_stability or handoff_evidence == "runtime_inputs":
        if not isinstance(runtime_inputs, Mapping):
            mismatches["runtime_inputs"] = {
                "expected": "object",
                "observed": type(runtime_inputs).__name__,
            }
            runtime_inputs = {}
    if require_runtime_input_stability:
        for key, value in (
            ("new_device_tensors_per_decode_step", 0),
            ("host_to_device_updates_per_decode_step", 0),
            ("page_table_reused", True),
        ):
            if runtime_inputs.get(key) != value:
                mismatches[key] = {
                    "expected": value,
                    "observed": runtime_inputs.get(key),
                }

    if handoff_evidence == "runtime_inputs":
        token_update = runtime_inputs.get("token_update")
        if token_update != "captured_device_to_device_copy":
            mismatches["runtime_inputs.token_update"] = {
                "expected": "captured_device_to_device_copy",
                "observed": token_update,
            }
    else:
        token_update = None
        runtime_context = profile.get("runtime_context")
        if not isinstance(runtime_context, Mapping):
            mismatches["runtime_context"] = {
                "expected": "object",
                "observed": type(runtime_context).__name__,
            }
        elif (
            runtime_context.get("decode_token_runtime_handoff")
            != "device_tensor_direct"
            or runtime_context.get("decode_token_host_roundtrip_per_step") is not False
        ):
            mismatches["runtime_context.device_token_handoff"] = {
                "expected": "device_tensor_direct without host roundtrip",
                "observed": {
                    "handoff": runtime_context.get("decode_token_runtime_handoff"),
                    "host_roundtrip": runtime_context.get(
                        "decode_token_host_roundtrip_per_step"
                    ),
                },
            }
    if not force_argmax_enabled(program_config):
        mismatches["force_argmax"] = {"expected": True, "observed": False}
    if mismatches:
        raise ContractViolation(
            "profile violates the decode measurement contract: "
            + canonical_json(mismatches)
        )

    return {
        "layers": expected_layers,
        "batch_size": int(profile["batch_size"]),
        "prefill_len": int(profile["prefill_len"]),
        "cache_len": int(profile["cache_len"]),
        "after_prefill": True,
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "persistent_input_count": persistent_count,
        "trace_capture_count": expected_trace_capture_count,
        "trace_execute_count": expected_trace_execute_count,
        "force_argmax": True,
        "page_table_reused": True,
        "device_token_handoff": token_update or "device_tensor_direct",
    }


def weights_recipe_sha256(
    model_root: str | Path, precision_contract: PrecisionContract
) -> str:
    root = Path(model_root)
    index_files = sorted(root.glob("*.safetensors.index.json"))
    weight_files = sorted(root.glob("*.safetensors"))
    identity = {
        "precision_hash": precision_contract.hash,
        "index_files": [
            {"name": path.name, "sha256": file_sha256(path)} for path in index_files
        ],
        "weight_files": [
            {"name": path.name, "size": path.stat().st_size} for path in weight_files
        ],
    }
    return sha256_json(identity)


def resolve_runtime_commit(repo_root: str | Path | None = None) -> str:
    env_commit = os.environ.get("TT_METAL_GIT_COMMIT")
    if env_commit:
        return env_commit
    candidates: list[Path] = []
    for name in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    candidates.append(
        root
        / "thirdparty"
        / "tt-mlir"
        / "third_party"
        / "tt-metal"
        / "src"
        / "tt-metal"
    )
    for candidate in candidates:
        commit = _git_commit(candidate)
        if commit:
            return commit
    raise ContractViolation(
        "unable to resolve the TTNN/tt-metal runtime commit for fingerprinting"
    )


def _git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _device_architecture(device: str) -> str | None:
    return {"p150a": "blackhole"}.get(str(device).lower())
