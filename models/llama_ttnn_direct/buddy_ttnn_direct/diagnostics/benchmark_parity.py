from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

from ..runtime.reports import write_report
from .official_support import (
    DEFAULT_PROMPTS,
    failed_latency_report as _failed_parse,
    latency_report_from_samples as _parsed_samples,
    official_pytest_command,
    official_source_environment,
    parse_official_accuracy_samples,
    parse_official_latency_report as _parse_official_log,
    pytest_collection_modifyitems,
    pytest_configure,
)
from .process_support import (
    absolute_path as _absolute_path,
    git_diff_sha256 as _git_diff_sha256,
    git_value as _git_value,
    run_logged_command as _run_command,
    sha256_file as _sha256,
)

OFFICIAL_EXTERNAL_REFERENCE_TPSU = 33.1
OFFICIAL_EXTERNAL_REFERENCE_TTFT_MS = 57.0
OFFICIAL_EXTERNAL_RELEASE_TAG = "v0.64.0-dev20251030"
OFFICIAL_EXTERNAL_RELEASE_COMMIT = "b76035fbdac81d8f9974976471dc60fc005e1bfb"
RUN_GROUPS = ("official_release_runs", "official_local_runs", "buddy_local_runs")
RESUME_CONTRACT_KEYS = (
    "buddy_program", "official_tt_metal_root", "model_path", "tokenizer_path",
    "input_prompts", "official_python", "official_release_root",
    "official_release_python", "official_release_runtime_root", "device", "device_id",
    "batch_size", "requested_prefill_len", "cache_len", "page_block_size", "warmup",
    "iterations", "repetitions",
)
CommandRunner = Callable[
    [Sequence[str], Path, dict[str, str], Path, float | None, int | None],
    int,
]

def run_benchmark_parity(
    *,
    out: str | Path,
    buddy_program: str | Path,
    official_tt_metal_root: str | Path,
    model_path: str | Path,
    input_prompts: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    batch_size: int = 32,
    prefill_len: int = 128,
    cache_len: int = 1024,
    page_block_size: int = 32,
    warmup: int = 5,
    iterations: int = 50,
    repetitions: int = 3,
    device: str = "p150a",
    device_id: int = 0,
    official_python: str | Path | None = None,
    official_release_root: str | Path | None = None,
    official_release_python: str | Path | None = None,
    official_release_runtime_root: str | Path | None = None,
    timeout_seconds: float | None = 3600.0,
    address_space_limit_bytes: int | None = 95_000_000_000,
    dry_run: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run matched official-demo, official-greedy, and Buddy decode profiles."""

    report_path, program_root = Path(out).resolve(), Path(buddy_program).resolve()
    previous_report, official_root = _load_previous_report(report_path), Path(official_tt_metal_root).resolve()
    model_root = Path(model_path).resolve()
    tokenizer_root = Path(tokenizer_path or model_root).resolve()
    prompts_path = Path(input_prompts or official_root / DEFAULT_PROMPTS).resolve()
    official_python_path = _absolute_path(official_python or sys.executable)
    release_root = Path(official_release_root).resolve() if official_release_root else None
    release_python_path = _absolute_path(official_release_python) if official_release_python else None
    release_runtime_root = (
        Path(official_release_runtime_root).resolve() if official_release_runtime_root else release_root
    )
    if (release_root is None) != (release_python_path is None):
        raise ValueError("official_release_root and official_release_python must be provided together")
    if release_root is None and release_runtime_root is not None:
        raise ValueError("official_release_runtime_root requires official_release_root")
    runs_root = report_path.parent / f"{report_path.stem}_runs"
    tensor_cache_root = report_path.parent.parent / "runtime_artifacts/official_tensor_cache"
    staged_prompts_path = runs_root / "inputs" / "input_prompts.json"
    runner = command_runner or _run_command

    for name, value in (
        ("batch_size", batch_size), ("prefill_len", prefill_len),
        ("cache_len", cache_len), ("page_block_size", page_block_size),
        ("iterations", iterations), ("repetitions", repetitions),
    ):
        _validate_positive(name, value)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    context = {
        "report_path": report_path, "program_root": program_root,
        "official_root": official_root, "model_root": model_root,
        "tokenizer_root": tokenizer_root, "prompts_path": prompts_path,
        "official_python": official_python_path, "release_root": release_root,
        "release_python": release_python_path, "release_runtime_root": release_runtime_root,
        "batch_size": batch_size, "prefill_len": prefill_len, "cache_len": cache_len,
        "page_block_size": page_block_size, "warmup": warmup, "iterations": iterations,
        "repetitions": repetitions, "device": device, "device_id": device_id,
        "timeout_seconds": timeout_seconds,
        "address_space_limit_bytes": address_space_limit_bytes,
    }
    base = _base_report(context)
    write_report(report_path, base)

    try:
        metadata = _collect_metadata(context)
        base.update(metadata)
        commands = _planned_commands(
            runs_root=runs_root, tensor_cache_root=tensor_cache_root,
            program_root=program_root, official_root=official_root,
            model_root=model_root, tokenizer_root=tokenizer_root,
            prompts_path=staged_prompts_path, official_python=official_python_path,
            release_root=release_root, release_python=release_python_path,
            release_runtime_root=release_runtime_root, layer_count=int(metadata["layers"]),
            batch_size=batch_size, effective_prefill_len=int(metadata["effective_prefill_len"]),
            cache_len=cache_len, page_block_size=page_block_size, warmup=warmup,
            iterations=iterations, repetitions=repetitions, device=device, device_id=device_id,
        )
        base.update({"planned_runs": commands, "staged_input_prompts": str(staged_prompts_path)})
        if dry_run:
            base.update({
                "status": "dry_run", "passed": True, "dry_run": True,
                "acceptance": {"status": "dry_run", "passed": True, "failed_checks": []},
            })
            write_report(report_path, base)
            return base

        runs_root.mkdir(parents=True, exist_ok=True)
        staged_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prompts_path, staged_prompts_path)
        resumable = _resumable_runs(previous_report, current_report=base, plans=commands)
        for plan in commands:
            run_key = (plan["profile"], int(plan["repetition"]))
            run, source = resumable.get(run_key), "previous-report"
            if run is None:
                run, source = _load_resumable_artifact(
                    plan, warmup=warmup, iterations=iterations
                ), "validated-run-artifacts"
            if run is None:
                run, source = _execute_planned_run(
                    plan=plan, page_block_size=page_block_size, cache_len=cache_len,
                    warmup=warmup, iterations=iterations, runner=runner,
                    timeout_seconds=timeout_seconds,
                    address_space_limit_bytes=address_space_limit_bytes,
                ), None
            run = {**run, "resumed": source is not None, "resume_source": source}
            group = (
                "buddy_local_runs" if plan["implementation"] == "buddy" else
                "official_release_runs" if plan["comparison_scope"] == "release-reference" else
                "official_local_runs"
            )
            base[group].append(run)
            write_report(report_path, base)

        _finalize_report(base)
    except Exception as error:
        base.update({
            "status": "failed", "passed": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "acceptance": {
                "status": "failed", "passed": False,
                "failed_checks": ["benchmark_parity.execution"],
            },
        })

    write_report(report_path, base)
    return base

def _load_previous_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None

def _resumable_runs(
    previous_report: dict[str, Any] | None,
    *,
    current_report: dict[str, Any],
    plans: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    if previous_report is None:
        return {}
    if any(previous_report.get(key) != current_report.get(key) for key in RESUME_CONTRACT_KEYS):
        return {}
    plans_by_key = {(plan["profile"], int(plan["repetition"])): plan for plan in plans}
    resumed: dict[tuple[str, int], dict[str, Any]] = {}
    for group in RUN_GROUPS:
        for run in previous_report.get(group, []):
            if not isinstance(run, dict) or not run.get("passed"):
                continue
            key = (str(run.get("profile")), int(run.get("repetition", 0)))
            plan = plans_by_key.get(key)
            if plan is not None and run.get("command") == plan.get("command"):
                resumed[key] = run
    return resumed

def _load_resumable_artifact(
    plan: dict[str, Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any] | None:
    run_dir = Path(plan["run_dir"])
    contract = _run_contract(plan, warmup=warmup, iterations=iterations)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        try:
            recorded_contract = json.loads(contract_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if recorded_contract != contract:
            return None

    if plan["implementation"] == "official":
        if not _pytest_xml_passed(run_dir / "pytest.xml"):
            return None
        parsed = _parse_official_log(Path(plan["log_path"]), warmup=warmup, iterations=iterations)
    else:
        parsed = _parse_buddy_report(Path(plan["profile_report"]), warmup=warmup, iterations=iterations)
    if not parsed.get("passed"):
        return None
    if not contract_path.is_file():
        write_report(contract_path, contract)
    return _run_result(plan=plan, parsed=parsed, return_code=0,
                       elapsed_seconds=None, runner_error=None)

def _pytest_xml_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return False
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return bool(suites) and all(
        int(suite.attrib.get("errors", "0")) == 0
        and int(suite.attrib.get("failures", "0")) == 0
        and int(suite.attrib.get("tests", "0")) > 0
        for suite in suites
    )

def _run_contract(
    plan: dict[str, Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    required = ("implementation", "profile", "comparison_scope", "repetition", "command")
    optional = ("official_root", "runtime_root", "runtime_mode", "model_path", "tensor_cache_path")
    contract = {key: plan[key] for key in required}
    contract.update({key: plan.get(key) for key in optional})
    contract.update(schema_version=1, repetition=int(plan["repetition"]),
                    warmup=int(warmup), iterations=int(iterations))
    return contract

def _base_report(context: Mapping[str, Any]) -> dict[str, Any]:
    c = context
    return {
        "schema_version": 1, "command": "diagnose", "stage": "benchmark-parity",
        "status": "running", "passed": False, "dry_run": False,
        "report_path": str(c["report_path"]),
        "official_external_reference": {
            "decode_tpsu": OFFICIAL_EXTERNAL_REFERENCE_TPSU, "ttft_ms": OFFICIAL_EXTERNAL_REFERENCE_TTFT_MS,
            "release_tag": OFFICIAL_EXTERNAL_RELEASE_TAG, "release_commit": OFFICIAL_EXTERNAL_RELEASE_COMMIT,
            "comparison_scope": "cross-version-reference-only",
        },
        "official_external_reference_tpsu": OFFICIAL_EXTERNAL_REFERENCE_TPSU,
        "buddy_program": str(c["program_root"]), "official_tt_metal_root": str(c["official_root"]),
        "model_path": str(c["model_root"]), "tokenizer_path": str(c["tokenizer_root"]),
        "input_prompts": str(c["prompts_path"]), "official_python": str(c["official_python"]),
        "official_tensor_cache_root": str(c["report_path"].parent.parent / "runtime_artifacts/official_tensor_cache"),
        "official_release_root": str(c["release_root"]) if c["release_root"] else None,
        "official_release_python": str(c["release_python"]) if c["release_python"] else None,
        "official_release_runtime_root": str(c["release_runtime_root"]) if c["release_runtime_root"] else None,
        "device": c["device"], "device_id": c["device_id"],
        "batch_size": c["batch_size"], "requested_prefill_len": c["prefill_len"],
        "cache_len": c["cache_len"], "page_block_size": c["page_block_size"],
        "warmup": c["warmup"], "iterations": c["iterations"],
        "repetitions": c["repetitions"], "timeout_seconds": c["timeout_seconds"],
        "address_space_limit_bytes": c["address_space_limit_bytes"],
        "official_local_runs": [], "official_release_runs": [], "buddy_local_runs": [],
        "official_release_median_tpsu": None, "buddy_ratio_of_release_official": None,
        "official_local_median_tpsu": None, "buddy_local_median_tpsu": None,
        "buddy_ratio_of_local_official": None, "coefficient_of_variation": {},
        "semantic_match": {}, "error": None,
    }

def _collect_metadata(context: Mapping[str, Any]) -> dict[str, Any]:
    c = context
    program_root, official_root = c["program_root"], c["official_root"]
    model_root, tokenizer_root, prompts_path = c["model_root"], c["tokenizer_root"], c["prompts_path"]
    release_root, release_python = c["release_root"], c["release_python"]
    release_runtime_root = c["release_runtime_root"]
    required = {
        "buddy program": program_root, "official tt-metal root": official_root,
        "model": model_root, "tokenizer": tokenizer_root,
        "prompt file": prompts_path, "official Python": c["official_python"],
    }
    required.update({name: path for name, path in (
        ("official release root", release_root),
        ("official release Python", release_python),
        ("official release runtime root", release_runtime_root),
    ) if path is not None})
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError("missing benchmark inputs: " + ", ".join(missing))

    config_path = program_root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing Buddy program config: {config_path}")
    layers = int(json.loads(config_path.read_text())["num_layers"])
    prompt_values = _load_prompt_values(prompts_path, c["batch_size"])
    token_lengths = _token_lengths(prompt_values, tokenizer_path=tokenizer_root, instruct=True)
    max_prompt_tokens = max(token_lengths)
    effective_prefill_len = _effective_prefill_len(c["prefill_len"], max_prompt_tokens)
    if effective_prefill_len > c["cache_len"]:
        raise ValueError(f"effective prefill length {effective_prefill_len} exceeds cache length {c['cache_len']}")

    repo_root = Path(__file__).resolve().parents[4]
    buddy_tt_metal_root = Path(os.environ.get("TT_METAL_HOME", str(official_root))).resolve()
    official_commit = _git_value(official_root, "rev-parse", "HEAD")
    buddy_tt_metal_commit = _git_value(buddy_tt_metal_root, "rev-parse", "HEAD")
    buddy_commit = _git_value(repo_root, "rev-parse", "HEAD")
    same_tt_metal_commit = official_commit == buddy_tt_metal_commit
    if not same_tt_metal_commit:
        raise ValueError(f"official and Buddy tt-metal commits differ: {official_commit} != {buddy_tt_metal_commit}")

    return {
        "buddy_commit": buddy_commit, "official_tt_metal_commit": official_commit,
        "buddy_tt_metal_root": str(buddy_tt_metal_root),
        "buddy_tt_metal_commit": buddy_tt_metal_commit, "same_tt_metal_commit": same_tt_metal_commit,
        "official_release": (_release_identity(release_root, release_python, release_runtime_root)
                             if release_root is not None else None),
        "official_execution_features": _official_execution_features(official_root=official_root,
                                                                      release_root=release_root),
        "layers": layers,
        "model_config_sha256": _sha256(model_root / "config.json"),
        "input_prompts_sha256": _sha256(prompts_path), "prompt_count": len(prompt_values),
        "prompt_token_lengths": token_lengths, "max_prompt_tokens": max_prompt_tokens,
        "effective_prefill_len": effective_prefill_len,
        "prefill_adjustment": {
            "requested": c["prefill_len"], "effective": effective_prefill_len,
            "reason": (
                "official prompts require the next power-of-two prefill bucket"
                if effective_prefill_len != c["prefill_len"]
                else "requested bucket fits every prompt"
            ),
        },
        "sampling": {
            "official-demo": "official non-uniform device sampling",
            "official-greedy": "force argmax (temperature=0)", "buddy": "force argmax",
        },
        "trace_mode": {profile: "trace" for profile in ("official-demo", "official-greedy", "buddy")},
    }

def _release_identity(
    release_root: Path,
    release_python: Path | None,
    release_runtime_root: Path | None,
) -> dict[str, Any]:
    runtime_root = release_runtime_root or release_root
    commit = _git_value(release_root, "rev-parse", "HEAD")
    runtime_commit = _git_value(runtime_root, "rev-parse", "HEAD")
    if commit != OFFICIAL_EXTERNAL_RELEASE_COMMIT:
        raise ValueError(f"official release root is not the published reference commit: {commit} != {OFFICIAL_EXTERNAL_RELEASE_COMMIT}")
    if runtime_commit != commit:
        raise ValueError(f"official release model and runtime commits differ: {commit} != {runtime_commit}")
    source = _git_identity(release_root)
    runtime = _git_identity(runtime_root)
    binaries = {
        "ttnn_binary": runtime_root / "ttnn" / "ttnn" / "_ttnn.so",
        "metal_library": runtime_root / "build" / "lib" / "libtt_metal.so",
    }
    return {
        "root": str(release_root), "python": str(release_python), "commit": commit,
        "expected_release_tag": OFFICIAL_EXTERNAL_RELEASE_TAG,
        "expected_release_commit": OFFICIAL_EXTERNAL_RELEASE_COMMIT,
        "matches_external_release_commit": True,
        "source_dirty": source["dirty"], "tracked_source_dirty": source["tracked_dirty"],
        "tracked_source_changes": source["tracked_changes"],
        "source_status": source["status"], "source_diff_sha256": source["diff_sha256"],
        "runtime_root": str(runtime_root), "runtime_commit": runtime_commit,
        "runtime_matches_release_commit": True, "runtime_mode": _release_runtime_mode(runtime_root),
        "runtime_source_dirty": runtime["dirty"], "runtime_tracked_source_dirty": runtime["tracked_dirty"],
        "runtime_tracked_source_changes": runtime["tracked_changes"],
        "runtime_source_status": runtime["status"], "runtime_source_diff_sha256": runtime["diff_sha256"],
        **{f"runtime_{name}": str(path) if path.is_file() else None for name, path in binaries.items()},
        **{f"runtime_{name}_sha256": _sha256(path) if path.is_file() else None
           for name, path in binaries.items()},
        "sampling": "force argmax (temperature=0) in the release demo",
        "trace_mode": "trace", "comparison_scope": "cross-version-reference-only",
    }

def _git_identity(root: Path) -> dict[str, Any]:
    changes = _git_value(root, "diff", "--name-only").splitlines()
    status = _git_value(root, "status", "--porcelain=v1")
    return {
        "dirty": bool(status), "tracked_dirty": bool(changes),
        "tracked_changes": changes, "status": status.splitlines(),
        "diff_sha256": _git_diff_sha256(root),
    }

def _official_execution_features(
    *,
    official_root: Path,
    release_root: Path | None,
) -> dict[str, dict[str, Any]]:
    def profile(source_supports_prefetcher: bool, sampling_mode: str,
                comparison_scope: str) -> dict[str, Any]:
        return {
            "use_prefetcher": False, "use_prefetcher_requested_by_command": False,
            "prefetcher_supported_by_source": source_supports_prefetcher,
            "global_cb": None, "global_cb_active": False, "sub_device_id": None,
            "trace": True, "sampling_mode": sampling_mode, "comparison_scope": comparison_scope,
            "selection_reason": (
                "the parity command omits --use_prefetcher and the official fixture/CLI default is disabled"
                if source_supports_prefetcher
                else "the target source does not expose the Llama prefetcher path"
            ),
        }

    current = _source_mentions_prefetcher(official_root)
    features = {
        "official-demo": profile(current, "official non-uniform device sampling", "same-commit-local"),
        "official-greedy": profile(current, "force argmax (temperature=0)", "same-commit-local"),
    }
    if release_root is not None:
        features["official-release-demo"] = profile(_source_mentions_prefetcher(release_root),
            "force argmax (temperature=0) in the release demo", "release-reference")
    return features

def _source_mentions_prefetcher(root: Path) -> bool:
    paths = ("models/tt_transformers/demo/conftest.py",
             "models/tt_transformers/demo/simple_text_demo.py", "models/tt_transformers/tt/common.py")
    return any((root / path).is_file() and "use_prefetcher" in (root / path).read_text()
               for path in paths)

def _release_runtime_mode(runtime_root: Path) -> str:
    ttnn_binary = runtime_root / "ttnn" / "ttnn" / "_ttnn.so"
    metal_library = runtime_root / "build" / "lib" / "libtt_metal.so"
    if ttnn_binary.is_file() and metal_library.is_file():
        return "source-build"
    return "wheel"

def _planned_commands(
    *,
    runs_root: Path,
    tensor_cache_root: Path,
    program_root: Path,
    official_root: Path,
    model_root: Path,
    tokenizer_root: Path,
    prompts_path: Path,
    official_python: Path,
    release_root: Path | None,
    release_python: Path | None,
    release_runtime_root: Path | None,
    layer_count: int,
    batch_size: int,
    effective_prefill_len: int,
    cache_len: int,
    page_block_size: int,
    warmup: int,
    iterations: int,
    repetitions: int,
    device: str,
    device_id: int,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    generated_tokens = 1 + warmup + iterations
    profiles: list[dict[str, Any]] = []
    if release_root is not None and release_python is not None:
        profiles.append({
            "profile": "official-release-demo", "root": release_root,
            "python": release_python, "runtime": release_runtime_root or release_root,
            "scope": "release-reference", "cache": OFFICIAL_EXTERNAL_RELEASE_COMMIT,
        })
    local_commit = _git_value(official_root, "rev-parse", "HEAD")
    profiles.extend({
        "profile": profile, "root": official_root, "python": official_python,
        "runtime": official_root, "scope": "same-commit-local", "cache": local_commit,
    } for profile in ("official-demo", "official-greedy"))
    for item in profiles:
        for repetition in range(1, repetitions + 1):
            profile = item["profile"]
            run_dir = runs_root / profile / f"repetition_{repetition:02d}"
            command = official_pytest_command(item["python"],
                "performance-batch-32 and not log-probs", [
                    "--input_prompts", str(prompts_path), "--instruct", "1",
                    "--repeat_batches", "1", "--max_seq_len", str(cache_len),
                    "--batch_size", str(batch_size), "--max_generated_tokens",
                    str(generated_tokens), "--stop_at_eos", "0"])
            command.extend(["--enable_trace", "1"] if item["scope"] == "release-reference"
                           else ["--enable_trace", "--mode", "full"])
            command.extend(["--junitxml", str(run_dir / "pytest.xml")])
            plans.append({
                "implementation": "official", "profile": profile,
                "comparison_scope": item["scope"], "repetition": repetition,
                "run_dir": str(run_dir), "log_path": str(run_dir / "run.log"),
                "command": command, "model_path": str(model_root),
                "official_root": str(item["root"]), "runtime_root": str(item["runtime"]),
                "runtime_mode": (_release_runtime_mode(item["runtime"])
                                 if item["scope"] == "release-reference" else "source-build"),
                "tensor_cache_path": str(tensor_cache_root / item["cache"]),
            })

    buddy_python = _absolute_path(sys.executable)
    for repetition in range(1, repetitions + 1):
        run_dir = runs_root / "buddy-greedy" / f"repetition_{repetition:02d}"
        profile_report = run_dir / "profile.json"
        command = [
            str(buddy_python), "-m",
            "models.llama_ttnn_direct.buddy_ttnn_direct.cli", "profile",
            "--mode", "decode-steady", "--program-dir", str(program_root),
            "--model-path", str(model_root), "--tokenizer-path", str(tokenizer_root),
            "--input-prompts", str(prompts_path), "--instruct",
            "--layers", str(layer_count), "--prefill-len", str(effective_prefill_len),
            "--batch-size", str(batch_size), "--cache-len", str(cache_len),
            "--warmup", str(warmup), "--iterations", str(iterations), "--after-prefill",
            "--runtime-input-mode", "persistent", "--execution-mode", "trace",
            "--device", device, "--device-id", str(device_id),
            "--dtype-seed", "bf16", "--out", str(profile_report),
        ]
        plans.append({
            "implementation": "buddy", "profile": "buddy-greedy",
            "comparison_scope": "same-commit-local", "repetition": repetition,
            "run_dir": str(run_dir), "log_path": str(run_dir / "run.log"),
            "profile_report": str(profile_report), "command": command,
        })
    return plans

def _execute_planned_run(
    *,
    plan: dict[str, Any],
    page_block_size: int,
    cache_len: int,
    warmup: int,
    iterations: int,
    runner: CommandRunner,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> dict[str, Any]:
    run_dir, log_path = Path(plan["run_dir"]), Path(plan["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_report(run_dir / "run_contract.json",
                 _run_contract(plan, warmup=warmup, iterations=iterations))
    environment = os.environ.copy()
    cwd = Path.cwd()
    if plan["implementation"] == "official":
        official_root = Path(plan["official_root"])
        cwd = official_root
        release = plan["comparison_scope"] == "release-reference"
        release_source_build = release and plan.get("runtime_mode") == "source-build"
        runtime_root = Path(plan["runtime_root"])
        local = plan["comparison_scope"] == "same-commit-local"
        environment = official_source_environment(official_root,
            model_root=Path(plan["model_path"]),
            tensor_cache_path=Path(plan["tensor_cache_path"]), profile=plan["profile"],
            cache_len=cache_len, page_block_size=page_block_size,
            runtime_root=runtime_root if release_source_build else official_root if local else None,
            preserve_pythonpath=local)
        if release:
            environment.pop("CONDA_PREFIX", None)
            if release_source_build:
                release_environment_root = Path(plan["command"][0]).parent.parent
                environment.update({
                    "TT_METAL_HOME": str(runtime_root),
                    "TT_METAL_BUILD_HOME": str(runtime_root / "build"),
                    "TT_METAL_RUNTIME_ROOT": str(runtime_root),
                    "LD_LIBRARY_PATH": os.pathsep.join((str(runtime_root / "build/lib"),
                                                         str(release_environment_root / "lib"))),
                })
            else:
                for variable in ("LD_LIBRARY_PATH", "TT_METAL_BUILD_HOME",
                                 "TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT"):
                    environment.pop(variable, None)
        else:
            environment["TT_METAL_HOME"] = str(official_root)
            environment["TT_METAL_RUNTIME_ROOT"] = str(official_root)

    started_at = time.time()
    try:
        return_code = runner(plan["command"], cwd, environment, log_path,
                             timeout_seconds, address_space_limit_bytes)
        runner_error = None
    except Exception as error:
        return_code = -1
        runner_error = f"{type(error).__name__}: {error}"
    elapsed_seconds = time.time() - started_at

    parsed = (_parse_official_log(log_path, warmup=warmup, iterations=iterations)
              if plan["implementation"] == "official" else
              _parse_buddy_report(Path(plan["profile_report"]), warmup=warmup,
                                  iterations=iterations))
    return _run_result(plan=plan, parsed=parsed, return_code=return_code,
                       elapsed_seconds=elapsed_seconds, runner_error=runner_error)

def _run_result(
    *,
    plan: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
    elapsed_seconds: float | None,
    runner_error: str | None,
) -> dict[str, Any]:
    passed = return_code == 0 and bool(parsed["passed"])
    first_samples = [*parsed["warmup_step_ms_samples"], *parsed["decode_step_ms_samples"]]
    first_decode_step_ms = first_samples[0] if first_samples else None
    latency_keys = ("warmup_step_ms_samples", "decode_step_ms_samples",
                    "decode_step_ms_mean", "decode_step_ms_p50",
                    "tokens_per_second_per_user", "sample_source")
    return {
        **{key: plan[key] for key in ("implementation", "profile", "comparison_scope",
                                      "repetition", "command")},
        "status": "passed" if passed else "failed", "passed": passed,
        "return_code": return_code, "elapsed_seconds": elapsed_seconds,
        "cwd": str(Path(plan["official_root"]) if plan["implementation"] == "official"
                   else Path.cwd()), "log_path": str(plan["log_path"]),
        **{key: plan.get(key) for key in ("profile_report", "runtime_root", "runtime_mode")},
        **{key: parsed[key] for key in latency_keys},
        "first_decode_step_ms": first_decode_step_ms,
        "first_decode_tokens_per_second_per_user": (1000.0 / first_decode_step_ms
                                                     if first_decode_step_ms else None),
        "error": runner_error or parsed.get("error"),
    }

def _parse_buddy_report(
    report_path: Path,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if not report_path.is_file():
        return _failed_parse(f"Buddy profile was not written: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return _failed_parse(f"invalid Buddy profile: {error}")
    warmup_samples = [float(value) for value in report.get("warmup_step_ms_samples", [])]
    measured_samples = [float(value) for value in report.get("decode_step_ms_samples", [])]
    parsed = _parsed_samples(warmup_samples=warmup_samples,
        measured_samples=measured_samples, sample_source="buddy_profile_json")
    if (len(warmup_samples), len(measured_samples)) != (warmup, iterations):
        parsed.update(passed=False, error=("Buddy sample counts differ from requested warmup/iterations: "
            f"{len(warmup_samples)}/{len(measured_samples)} != {warmup}/{iterations}"))
    if not report.get("passed"):
        parsed["passed"] = False
        parsed["error"] = report.get("error") or report.get("message")
    return parsed

def _finalize_report(report: dict[str, Any]) -> None:
    official = report["official_local_runs"]
    runs = {
        profile: [run for run in official if run["profile"] == profile]
        for profile in ("official-demo", "official-greedy")
    }
    runs.update({
        "buddy-greedy": report["buddy_local_runs"],
        "official-release-demo": report.get("official_release_runs", []),
    })
    values = {profile: _successful_tpsu(items) for profile, items in runs.items()}
    medians = {p: statistics.median(v) if v else None for p, v in values.items()}
    demo_median, greedy_median = medians["official-demo"], medians["official-greedy"]
    buddy_median, release_median = medians["buddy-greedy"], medians["official-release-demo"]
    release_first_values = [
        float(run["first_decode_tokens_per_second_per_user"])
        for run in runs["official-release-demo"]
        if run.get("passed")
        and run.get("first_decode_tokens_per_second_per_user") is not None
    ]
    cv = {f"{p.removesuffix('-demo') if p == 'official-release-demo' else p}-percent":
          _coefficient_of_variation_percent(v) for p, v in values.items()}
    latency_statistics = {profile: _latency_statistics(items) for profile, items in runs.items()}
    local_ratio = buddy_median / greedy_median if buddy_median is not None and greedy_median else None
    release_ratio = buddy_median / release_median if buddy_median is not None and release_median else None
    report.update(
        {
            "official_demo_local_median_tpsu": demo_median,
            "official_greedy_local_median_tpsu": greedy_median,
            "official_local_median_tpsu": greedy_median,
            "official_local_primary_profile": "official-greedy",
            "buddy_local_median_tpsu": buddy_median,
            "official_release_median_tpsu": release_median,
            "official_release_first_decode_median_tpsu": statistics.median(release_first_values)
                if release_first_values else None,
            "buddy_ratio_of_local_official": local_ratio,
            "buddy_ratio_of_release_official": release_ratio,
            "release_ratio_of_external_reference": release_median / OFFICIAL_EXTERNAL_REFERENCE_TPSU
                if release_median is not None else None,
            "coefficient_of_variation": cv,
            "decode_latency_statistics_ms": latency_statistics,
            "performance_milestones": {
                "M6_local_official_90_percent": local_ratio is not None and local_ratio >= 0.90,
                "M7_local_official_95_percent": local_ratio is not None and local_ratio >= 0.95,
                "M8_local_official_98_percent_cv_1_5_percent": local_ratio is not None
                    and local_ratio >= 0.98
                    and cv["buddy-greedy-percent"] is not None
                    and cv["buddy-greedy-percent"] <= 1.5,
                "release_reference_98_percent": release_ratio is not None and release_ratio >= 0.98,
            },
            "semantic_match": {
                "same_tt_metal_commit": report.get("same_tt_metal_commit"),
                "same_device": True, "same_model": True, "same_prompt_file": True,
                "same_batch_size": True, "same_cache_len": True,
                "same_page_block_size": True, "same_requested_prefill_len": True,
                "same_physical_prefill_bucket": False,
                "same_prompt_tokenization_mode": True,
                "decode_positions_follow_actual_prompt_lengths": True,
                "prefill_timing_comparable": False,
                "official_greedy_matches_buddy_sampling": True,
                "official_demo_matches_buddy_sampling": False,
                "official_trace_vs_buddy_eager": False,
                "official_trace_matches_buddy_trace": True,
                "decode_workload_comparable": True,
                "primary_official_profile": "official-greedy",
                "release_reference_same_tt_metal_commit": False,
                "release_reference_directly_comparable_to_buddy": False,
            },
            "comparison_contract": {
                "primary_parity_ratio": "buddy_ratio_of_local_official",
                "primary_reason": "Buddy and official-greedy use the same tt-metal commit",
                "release_reference_ratio": "buddy_ratio_of_release_official",
                "release_reference_reason": "validates the published release speed but crosses tt-metal versions",
            },
        }
    )

    expected = int(report["repetitions"])
    local_profiles = ("official-demo", "official-greedy", "buddy-greedy")
    checks = [_check(f"{p}.successful-runs", len(values[p]) == expected,
                     len(values[p]), expected) for p in local_profiles]
    checks += [_check(f"{p}.cv-percent", cv[f"{p}-percent"] is not None
                      and cv[f"{p}-percent"] <= 1.5, cv[f"{p}-percent"], 1.5)
               for p in local_profiles]
    checks += [_check("same-tt-metal-commit", bool(report.get("same_tt_metal_commit")),
                      report.get("same_tt_metal_commit"), True),
               _check("raw-samples", all(len(run["decode_step_ms_samples"]) == report["iterations"]
                     for p in local_profiles for run in runs[p]), None, report["iterations"])]
    if report.get("official_release") is not None:
        release_values = values["official-release-demo"]
        release_runs = runs["official-release-demo"]
        checks += [_check("official-release.successful-runs", len(release_values) == expected,
                          len(release_values), expected),
                   _check("official-release.cv-percent", cv["official-release-percent"] is not None
                          and cv["official-release-percent"] <= 1.5,
                          cv["official-release-percent"], 1.5),
                   _check("official-release.commit",
                          bool(report["official_release"].get("matches_external_release_commit")),
                          report["official_release"].get("commit"), OFFICIAL_EXTERNAL_RELEASE_COMMIT),
                   _check("official-release.raw-samples",
                          all(len(run["decode_step_ms_samples"]) == report["iterations"]
                              for run in release_runs), None, report["iterations"])]
    failed_checks = [check["name"] for check in checks if not check["passed"]]
    passed = not failed_checks
    status = "passed" if passed else "failed"
    report.update({
        "status": status, "passed": passed,
        "acceptance": {"status": status, "passed": passed, "checks": checks, "failed_checks": failed_checks},
        "error": None,
    })

def _successful_tpsu(runs: Sequence[dict[str, Any]]) -> list[float]:
    return [float(run["tokens_per_second_per_user"]) for run in runs
            if run.get("passed") and run.get("tokens_per_second_per_user") is not None]

def _latency_statistics(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        float(sample)
        for run in runs
        if run.get("passed")
        for sample in run.get("decode_step_ms_samples", [])
    ]
    if not samples:
        return {key: value for key, value in (("sample_count", 0), ("mean", None),
            ("p50", None), ("p90", None), ("min", None), ("max", None),
            ("stdev", None), ("coefficient_of_variation_percent", None))}
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return {
        "sample_count": len(samples),
        "mean": mean,
        "p50": statistics.median(samples),
        "p90": _percentile(samples, 0.90),
        "min": min(samples),
        "max": max(samples),
        "stdev": stdev,
        "coefficient_of_variation_percent": (stdev / mean * 100.0 if mean else None),
    }

def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def _coefficient_of_variation_percent(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values) / mean * 100.0

def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}

def _load_prompt_values(path: Path, batch_size: int) -> list[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("input prompts JSON must be a list")
    prompts = []
    for entry in payload:
        if isinstance(entry, str):
            prompt = entry
        elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompt = entry["prompt"]
        else:
            raise ValueError("each input prompt must be a string or {'prompt': string}")
        prompts.append(prompt)
    if len(prompts) < batch_size:
        raise ValueError(
            f"input prompt file has {len(prompts)} prompts; batch size is {batch_size}"
        )
    return prompts[:batch_size]

def _token_lengths(
    prompts: Sequence[str],
    *,
    tokenizer_path: Path,
    instruct: bool,
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    lengths = []
    for prompt in prompts:
        tokenized = (tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            if instruct else tokenizer(prompt, add_special_tokens=True))
        token_ids = _input_ids(tokenized)
        lengths.append(len(token_ids))
    return lengths

def _input_ids(tokenized: Any) -> Sequence[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, Sequence) and tokenized and isinstance(tokenized[0], Sequence):
        tokenized = tokenized[0]
    if not isinstance(tokenized, Sequence) or isinstance(tokenized, (str, bytes)):
        raise ValueError("tokenizer did not return an input_ids sequence")
    return tokenized

def _effective_prefill_len(requested: int, max_prompt_tokens: int) -> int:
    required = max(requested, max_prompt_tokens)
    return 1 << (required - 1).bit_length()

def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
