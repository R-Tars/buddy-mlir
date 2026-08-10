from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..codegen.program import write_decode_program_bundle
from ..templates.registry import build_execution_plan
from .measurement import build_candidate_config, candidate_fingerprint
from .schema import CandidateConfig, MeasurementContract
from .search import SearchCandidate
from .space import SearchSpaceConfig

CandidateGateRunner = Callable[..., Mapping[str, Any]]


@dataclass
class ModelCandidateEvaluator:
    context: Mapping[str, Any]
    output_root: Path
    candidate_spaces: Mapping[str, SearchSpaceConfig]
    resume: bool = True
    candidate_gate_runner: CandidateGateRunner | None = None

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
            run_gate=True,
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
        self,
        space: SearchSpaceConfig,
        contract: MeasurementContract,
        label: str,
        *,
        run_gate: bool = False,
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
            and cached.get("program_dir")
            and all(cached.get(key) == value for key, value in _contract_fields(candidate, contract, None).items() if key != "program_dir")
            and (not run_gate or _complete_gate(cached.get("candidate_gate"), fingerprint))
        ):
            cached["resumed"] = True
            return cached
        program_dir = root / "program"
        try:
            from .search import buildable_template_config

            config = buildable_template_config(dict(self.context["seed"]), space)
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
                _read_json(profile_path),
                candidate,
                contract,
                elapsed,
                profile_path,
                command,
                program_dir,
            )
            if process.returncode and report["passed"]:
                report.update(
                    status="failed",
                    passed=False,
                    error={"type": "profile_subprocess_exit", "message": str(process.returncode)},
                )
            if run_gate and report["passed"]:
                report = self._apply_gate(report, candidate, program_dir)
        except Exception as exc:
            report = {
                "status": "failed",
                "passed": False,
                "candidate_fingerprint": fingerprint,
                "measurement_contract": contract.to_dict(),
                **_contract_fields(candidate, contract, program_dir),
                "correctness_passed": False,
                "quality_passed": False,
                "gate_status": "not_evaluated",
                "candidate_gate": None,
                "isolated_subprocess": True,
                "worker": {"isolated_subprocess": True},
                "device_seconds": 0.0,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        measurement_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def _apply_gate(
        self,
        report: dict[str, Any],
        candidate: CandidateConfig,
        program_dir: Path,
    ) -> dict[str, Any]:
        fingerprint = candidate_fingerprint(candidate)
        if self.candidate_gate_runner is None:
            return {
                **report,
                "gate_status": "not_evaluated",
                "correctness_passed": False,
                "quality_passed": False,
                "candidate_gate": None,
            }
        try:
            gate = dict(self.candidate_gate_runner(
                program_dir=program_dir,
                candidate=candidate,
                candidate_fingerprint=fingerprint,
                model_path=self.context["model_root"],
                tokenizer_path=self.context.get("tokenizer_path"),
                layers=self.context["layers"],
                batch_size=self.context["batch_size"],
                prefill_len=self.context["prefill_len"],
                cache_len=self.context["cache_len"],
                device=self.context["device"],
                device_id=self.context["device_id"],
            ))
        except Exception as exc:
            gate = {
                "status": "failed",
                "passed": False,
                "correctness_passed": False,
                "quality_passed": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        complete = _complete_gate(gate, fingerprint)
        return {
            **report,
            "gate_status": "passed" if complete else "failed",
            "correctness_passed": bool(complete and gate.get("correctness_passed")),
            "quality_passed": bool(complete and gate.get("quality_passed")),
            "candidate_gate": gate,
        }

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
    program_dir: Path | None = None,
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
        **_contract_fields(candidate, contract, program_dir),
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
        "correctness_passed": False,
        "quality_passed": False,
        "gate_status": "not_evaluated",
        "candidate_gate": None,
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


def _contract_fields(
    candidate: CandidateConfig,
    contract: MeasurementContract,
    program_dir: Path | None,
) -> dict[str, Any]:
    execution = candidate.execution_contract
    return {
        "warmup": contract.warmup,
        "iterations": contract.iterations,
        "execution_mode": execution.execution_mode,
        "runtime_input_mode": execution.runtime_input_mode,
        "after_prefill": execution.after_prefill,
        "sampling": execution.sampling,
        "page_table": execution.page_table,
        "program_dir": str(program_dir) if program_dir is not None else None,
    }


def _complete_gate(gate: Any, fingerprint: str) -> bool:
    if not isinstance(gate, Mapping):
        return False
    identity = gate.get("cache_identity")
    return bool(
        gate.get("status") == "passed"
        and gate.get("passed") is True
        and gate.get("correctness_passed") is True
        and gate.get("quality_passed") is True
        and isinstance(gate.get("evidence_path"), str)
        and bool(gate.get("evidence_path"))
        and Path(gate["evidence_path"]).is_file()
        and isinstance(gate.get("evidence_sha256"), str)
        and len(gate["evidence_sha256"]) == 64
        and isinstance(identity, Mapping)
        and identity.get("candidate_fingerprint") == fingerprint
    )
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
def _slug(value: str) -> str:
    result = "".join(x if x.isalnum() else "-" for x in value.lower())
    return result.strip("-") or "candidate"
