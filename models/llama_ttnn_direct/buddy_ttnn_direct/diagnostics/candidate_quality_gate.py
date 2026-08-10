from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from ..runtime.reports import write_report
from .performance_correctness import DEFAULT_REFERENCE, run_performance_correctness

QUALITY_PREFILL_LEN = 512
CORRECTNESS_CHECKS = (
    "official_sample_count", "buddy_batch_complete", "buddy_top1",
    "buddy_top5", "buddy_min_user_top1", "buddy_min_user_top5",
)


def build_candidate_quality_gate(
    *, official_tt_metal_root: str | Path, official_python: str | Path | None,
    accuracy_tokens: int, gate_root: str | Path,
):
    official = Path(official_tt_metal_root).resolve()
    reference = official / DEFAULT_REFERENCE
    if not official.is_dir():
        raise ValueError(f"official TT-Metal root does not exist: {official}")
    if not reference.is_file():
        raise ValueError(f"official quality reference does not exist: {reference}")
    identity = {
        "quality_reference_sha256": _sha256(reference),
        "accuracy_token_count": int(accuracy_tokens),
        "official_tt_metal_commit": _git_commit(official),
        "official_python": str(Path(official_python or sys.executable).resolve()),
    }
    destination = Path(gate_root).resolve()

    def run(**values: Any) -> Mapping[str, Any]:
        candidate = values["candidate"]
        fingerprint = str(values["candidate_fingerprint"])
        cache_identity = {
            **identity, "candidate_fingerprint": fingerprint,
            "candidate_runtime_commit": str(getattr(candidate, "runtime_commit", "")),
            "workload": {
                "layers": int(values["layers"]), "batch_size": int(values["batch_size"]),
                "candidate_prefill_len": int(values["prefill_len"]),
                "cache_len": int(values["cache_len"]),
            },
        }
        evidence = destination / fingerprint[:24] / "quality.json"
        try:
            cached = json.loads(evidence.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if (isinstance(cached, dict) and cached.get("status") == "passed"
                and cached.get("passed") is True
                and cached.get("candidate_gate_cache_identity") == cache_identity):
            return _result(cached, evidence, cache_identity, resumed=True)
        report = run_performance_correctness(
            out=evidence, buddy_program=values["program_dir"],
            official_tt_metal_root=official, model_path=values["model_path"],
            tokenizer_path=values.get("tokenizer_path"), official_python=official_python,
            token_count=int(accuracy_tokens), layers=int(values["layers"]),
            batch_size=int(values["batch_size"]), prefill_len=QUALITY_PREFILL_LEN,
            cache_len=int(values["cache_len"]), device=str(values["device"]),
            device_id=int(values["device_id"]),
        )
        report["candidate_gate_cache_identity"] = cache_identity
        write_report(evidence, report)
        return _result(report, evidence, cache_identity, resumed=False)

    return run

def _result(
    report: Mapping[str, Any], evidence: Path, identity: Mapping[str, Any], *, resumed: bool
) -> dict[str, Any]:
    acceptance = report.get("acceptance")
    checks = acceptance.get("checks", {}) if isinstance(acceptance, Mapping) else {}
    execution = bool(
        report.get("status") == "passed" and report.get("passed") is True
        and isinstance(acceptance, Mapping) and acceptance.get("status") == "passed"
        and acceptance.get("passed") is True
    )
    correctness = execution and all(bool(checks.get(name)) for name in CORRECTNESS_CHECKS)
    quality = execution and bool(checks.get("official_buddy_min_user_greedy_agreement"))
    passed = bool(correctness and quality)
    return {
        "status": "passed" if passed else "failed", "passed": passed,
        "correctness_passed": correctness, "quality_passed": quality,
        "gate_status": "passed" if passed else "failed", "evidence_path": str(evidence),
        "evidence_sha256": _sha256(evidence), "evidence_identity": dict(identity),
        "cache_identity": dict(identity), "acceptance": acceptance, "resumed": resumed,
    }

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"official TT-Metal root is not a Git checkout: {root}") from exc
