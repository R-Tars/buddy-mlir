from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..codegen.program import write_decode_program_bundle
from ..templates.registry import build_execution_plan
from .measurement import build_candidate_config, candidate_fingerprint
from .schema import CandidateConfig, MeasurementContract
from .search import SearchCandidate
from .space import SearchSpaceConfig


@dataclass
class ModelCandidateEvaluator:
    context: Mapping[str, Any]
    output_root: Path
    candidate_spaces: Mapping[str, SearchSpaceConfig]
    resume: bool = True

    def candidate_config(
        self, candidate: SearchCandidate, contract: MeasurementContract
    ) -> CandidateConfig:
        return self._config(candidate.space, contract)

    def active_measurement(
        self, candidate: Any, contract: MeasurementContract, round_name: str
    ) -> dict[str, Any]:
        return self._measure(
            self.candidate_spaces[candidate.candidate_id],
            contract,
            f"active-{candidate.operator_name}-{round_name}",
        )

    def active_full_model(self, candidate: Any) -> dict[str, Any]:
        contract = MeasurementContract(
            warmup=5, iterations=50, repetitions=1, kind="active_full_model_trace"
        )
        return self._measure(
            self.candidate_spaces[candidate.candidate_id], contract, "active-full-model"
        )

    def layer_evaluator(self, candidate: SearchCandidate) -> dict[str, Any]:
        return self._measure(
            candidate.space,
            MeasurementContract(warmup=5, iterations=30, repetitions=1, kind="layer_confirmation"),
            "layer-confirmation",
        )

    def full_model_evaluator(self, candidate: SearchCandidate) -> dict[str, Any]:
        return self._measure(
            candidate.space,
            MeasurementContract(warmup=5, iterations=50, repetitions=1, kind="full_model_trace"),
            "full-model-trace",
        )

    def confirmation_runner(
        self, candidate: SearchCandidate, arm: str, repetition: int, contract: MeasurementContract
    ) -> dict[str, Any]:
        return self._measure(candidate.space, contract, f"long-confirmation-{arm}-{repetition}")

    def _config(self, space: SearchSpaceConfig, contract: MeasurementContract) -> CandidateConfig:
        c = self.context
        return build_candidate_config(
            graph=c["graph"],
            model_root=c["model_root"],
            precision_contract=c["precision"],
            expected_precision_hash=c["precision"].hash,
            execution_contract=c["execution"],
            measurement_contract=contract,
            runtime_commit=c["runtime_commit"],
            device=c["device"],
            device_id=c["device_id"],
            target={
                "device": c["device"],
                "batch_size": c["batch_size"],
                "cache_len": c["cache_len"],
                "prefill_len": c["prefill_len"],
                "page_block_size": 32,
            },
            tunable_state={"space": space.tunable_dict()},
        )

    def _measure(
        self, space: SearchSpaceConfig, contract: MeasurementContract, label: str
    ) -> dict[str, Any]:
        candidate = self._config(space, contract)
        fingerprint = candidate_fingerprint(candidate)
        root = self.output_root / fingerprint[:24] / _slug(label)
        profile_path, measurement_path = root / "profile.json", root / "measurement.json"
        root.mkdir(parents=True, exist_ok=True)
        cached = _read_json(measurement_path) if self.resume else {}
        if (
            cached.get("candidate_fingerprint") == fingerprint
            and cached.get("measurement_contract") == contract.to_dict()
            and cached.get("status") == "passed"
        ):
            cached["resumed"] = True
            return cached
        try:
            from .search import buildable_template_config

            config = buildable_template_config(dict(self.context["seed"]), space)
            program_dir = root / "program"
            write_decode_program_bundle(
                graph=self.context["graph"],
                plan=build_execution_plan(self.context["graph"], config),
                template_config=config,
                model_path=self.context["model_root"],
                out_dir=program_dir,
            )
            command = self._command(program_dir, profile_path, contract)
            started = time.monotonic()
            process = subprocess.run(
                command, cwd=str(_repo_root()), text=True, capture_output=True, check=False
            )
            elapsed = time.monotonic() - started
            (root / "profile_process.log").write_text(
                f"exit_code={process.returncode}\n--- stdout ---\n{process.stdout}--- stderr ---\n{process.stderr}",
                encoding="utf-8",
            )
            report = _normalize(
                _read_json(profile_path), candidate, contract, elapsed, profile_path, command
            )
            if process.returncode and report["passed"]:
                report.update(
                    status="failed",
                    passed=False,
                    error={"type": "profile_subprocess_exit", "message": str(process.returncode)},
                )
        except Exception as exc:
            report = {
                "status": "failed",
                "passed": False,
                "candidate_fingerprint": fingerprint,
                "measurement_contract": contract.to_dict(),
                "isolated_subprocess": True,
                "worker": {"isolated_subprocess": True},
                "device_seconds": 0.0,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        measurement_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def _command(self, program_dir: Path, output: Path, contract: MeasurementContract) -> list[str]:
        c = self.context
        command = [
            sys.executable,
            "-m",
            "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
            "profile",
            "--mode",
            "decode-steady",
            "--program-dir",
            str(program_dir),
            "--model-path",
            str(c["model_root"]),
            "--prompt",
            c["prompt"],
            "--layers",
            str(c["layers"]),
            "--prefill-len",
            str(c["prefill_len"]),
            "--batch-size",
            str(c["batch_size"]),
            "--cache-len",
            str(c["cache_len"]),
            "--device",
            c["device"],
            "--device-id",
            str(c["device_id"]),
            "--dtype-seed",
            "bf16",
            "--warmup",
            str(contract.warmup),
            "--iterations",
            str(contract.iterations),
            "--after-prefill",
            "--execution-mode",
            c["execution"].execution_mode,
            "--runtime-input-mode",
            c["execution"].runtime_input_mode,
            "--out",
            str(output),
        ]
        if c.get("tokenizer_path"):
            command.extend(("--tokenizer-path", str(c["tokenizer_path"])))
        return command


def _normalize(
    profile: Mapping[str, Any],
    candidate: CandidateConfig,
    contract: MeasurementContract,
    elapsed: float,
    path: Path,
    command: list[str],
) -> dict[str, Any]:
    latency, throughput = profile.get("decode_step_ms_p50"), profile.get(
        "tokens_per_second_per_user"
    )
    passed = bool(profile.get("passed") and latency is not None and throughput is not None)
    return {
        "status": "passed" if passed else str(profile.get("status", "failed")),
        "passed": passed,
        "candidate_fingerprint": candidate_fingerprint(candidate),
        "measurement_contract": contract.to_dict(),
        "tokens_per_second_per_user": throughput,
        "latency_ms": latency,
        "statistics": {
            "mean": profile.get("decode_step_ms_mean"),
            "p50": latency,
            "p90": profile.get("decode_step_ms_max"),
            "sample_count": len(profile.get("decode_step_ms_samples") or []),
        },
        "objective": {
            "name": "tokens_per_second_per_user",
            "value": throughput,
            "direction": "maximize",
        },
        "correctness_passed": passed,
        "quality_passed": passed,
        "isolated_subprocess": True,
        "worker": {"isolated_subprocess": True, "command": command},
        "device_seconds": max(0.0, elapsed),
        "profile_report": str(path),
        "error": profile.get("error"),
        "resumed": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
def _slug(value: str) -> str:
    result = "".join(x if x.isalnum() else "-" for x in value.lower())
    return result.strip("-") or "candidate"
