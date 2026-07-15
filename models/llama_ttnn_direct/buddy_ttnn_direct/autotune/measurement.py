from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..semantic.graph import LlamaModelGraph, graph_to_dict
from .schema import (
    CandidateConfig,
    ContractViolation,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    sha256_json,
)


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
