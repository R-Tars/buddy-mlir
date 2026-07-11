from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ...codegen.program import write_decode_program_bundle
from ...runtime.profile import run_profile_decode_steady
from ...templates.registry import build_execution_plan
from .selection import AUTOTUNE_METRIC


ProfileRunner = Callable[..., dict[str, Any]]


def measure_state(
    *,
    state: dict[str, Any],
    fingerprint: str,
    state_root: Path,
    graph: Any,
    seed: dict[str, Any],
    model_root: Path,
    prompt: str | None,
    tokenizer_path: str | Path | None,
    layer_count: int,
    prefill_len: int | None,
    batch_size: int | None,
    cache_len: int | None,
    device: str,
    device_id: int,
    dtype_seed: str,
    warmup: int,
    iterations: int,
    dry_run: bool,
    resume: bool,
    profile_runner: ProfileRunner | None,
) -> dict[str, Any]:
    candidate_id = _candidate_id(state, fingerprint)
    candidate_dir = state_root / candidate_id
    program_dir = candidate_dir / "program"
    profile_path = candidate_dir / "steady_profile.json"
    measurement_path = candidate_dir / "measurement.json"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        resumed = _load_resumable_measurement(
            measurement_path,
            profile_path,
            fingerprint,
        )
        if resumed is not None:
            resumed["resumed"] = True
            return resumed

    if not dry_run:
        ownership = check_device_ownership(device_id)
        if not ownership["available"]:
            return {
                "candidate_id": candidate_id,
                "fingerprint": fingerprint,
                "state": copy.deepcopy(state),
                "status": "device_occupied",
                "passed": False,
                "metric_value": None,
                "device_ownership": ownership,
                "program_dir": str(program_dir),
                "profile_report": None,
                "error": ownership["message"],
                "resumed": False,
                "measurement_reused": False,
            }
    else:
        ownership = {"checked": False, "available": None}

    try:
        candidate_seed = copy.deepcopy(seed)
        candidate_seed["lm_head_split_count"] = int(
            state["lm_head_split_count"]
        )
        candidate_seed["dtype_recipe"] = str(state["dtype_recipe"])
        plan = build_execution_plan(graph, candidate_seed)
        plan["template_config"]["autotune"] = {
            "schema_version": 1,
            "memory_layout": state["memory_layout"],
            "program_config": state["program_config"],
        }
        write_decode_program_bundle(
            graph=graph,
            plan=plan,
            template_config=candidate_seed,
            model_path=model_root,
            out_dir=program_dir,
        )
        write_json(candidate_dir / "seed_config.json", candidate_seed)
        write_json(
            candidate_dir / "candidate.json",
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "fingerprint": fingerprint,
                "state": state,
            },
        )
        profile = _run_candidate_profile(
            profile_runner=profile_runner,
            profile_path=profile_path,
            program_dir=program_dir,
            model_root=model_root,
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            layer_count=layer_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            warmup=warmup,
            iterations=iterations,
            dry_run=dry_run,
        )
        record = {
            "candidate_id": candidate_id,
            "fingerprint": fingerprint,
            "state": copy.deepcopy(state),
            "status": profile.get("status"),
            "passed": bool(profile.get("passed")),
            "metric_value": profile.get(AUTOTUNE_METRIC),
            "decode_step_ms_p50": profile.get("decode_step_ms_p50"),
            "decode_step_ms_mean": profile.get("decode_step_ms_mean"),
            "aggregate_tokens_per_second": profile.get(
                "aggregate_tokens_per_second"
            ),
            "setup_ms": profile.get("setup_ms"),
            "prefill_ms": profile.get("prefill_ms"),
            "warmup": int(warmup),
            "iterations": int(iterations),
            "device_ownership": ownership,
            "program_dir": str(program_dir),
            "profile_report": str(profile_path),
            "error": profile.get("error"),
            "resumed": False,
            "measurement_reused": False,
        }
    except Exception as exc:
        record = {
            "candidate_id": candidate_id,
            "fingerprint": fingerprint,
            "state": copy.deepcopy(state),
            "status": "candidate_error",
            "passed": False,
            "metric_value": None,
            "device_ownership": ownership,
            "program_dir": str(program_dir),
            "profile_report": str(profile_path),
            "error": f"{type(exc).__name__}: {exc}",
            "resumed": False,
            "measurement_reused": False,
        }
    write_json(
        measurement_path,
        {"schema_version": 1, "fingerprint": fingerprint, "record": record},
    )
    return record


def check_device_ownership(device_id: int) -> dict[str, Any]:
    node = Path(f"/dev/tenstorrent/{int(device_id)}")
    if not node.exists():
        return {
            "checked": True,
            "available": False,
            "device_node": str(node),
            "holders": [],
            "message": f"Tenstorrent device node does not exist: {node}",
        }
    result = subprocess.run(
        ["fuser", str(node)],
        text=True,
        capture_output=True,
        check=False,
    )
    all_holders = [item for item in result.stdout.split() if item.isdigit()]
    current_pid = str(os.getpid())
    self_holders = [pid for pid in all_holders if pid == current_pid]
    holders = [pid for pid in all_holders if pid != current_pid]
    available = not holders
    return {
        "checked": True,
        "available": available,
        "device_node": str(node),
        "holders": holders,
        "self_holders": self_holders,
        "fuser_returncode": result.returncode,
        "message": (
            "device node has no external owner"
            if available
            else f"device node is held by pid(s): {', '.join(holders) or 'unknown'}"
        ),
    }


def state_fingerprint(state: dict[str, Any], invocation: dict[str, Any]) -> str:
    payload = json.dumps(
        {"state": state, "invocation": invocation},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _run_candidate_profile(
    *,
    profile_runner: ProfileRunner | None,
    profile_path: Path,
    program_dir: Path,
    model_root: Path,
    prompt: str | None,
    tokenizer_path: str | Path | None,
    layer_count: int,
    prefill_len: int | None,
    device: str,
    device_id: int,
    batch_size: int | None,
    cache_len: int | None,
    dtype_seed: str,
    warmup: int,
    iterations: int,
    dry_run: bool,
) -> dict[str, Any]:
    kwargs = {
        "out": profile_path,
        "program_dir": program_dir,
        "model_path": model_root,
        "prompt": prompt,
        "tokenizer_path": tokenizer_path,
        "layers": layer_count,
        "prefill_len": prefill_len,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "dtype_seed": dtype_seed,
        "warmup": warmup,
        "iterations": iterations,
        "after_prefill": True,
        "dry_run": dry_run,
    }
    if profile_runner is not None:
        return profile_runner(**kwargs)
    if dry_run:
        return run_profile_decode_steady(**kwargs)
    return _run_profile_subprocess(**kwargs)


def _run_profile_subprocess(**kwargs: Any) -> dict[str, Any]:
    profile_path = Path(kwargs["out"])
    command = [
        sys.executable,
        "-m",
        "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "profile",
        "--mode",
        "decode-steady",
        "--program-dir",
        str(kwargs["program_dir"]),
        "--model-path",
        str(kwargs["model_path"]),
        "--prompt",
        str(kwargs["prompt"]),
        "--layers",
        str(kwargs["layers"]),
        "--device",
        str(kwargs["device"]),
        "--device-id",
        str(kwargs["device_id"]),
        "--dtype-seed",
        str(kwargs["dtype_seed"]),
        "--warmup",
        str(kwargs["warmup"]),
        "--iterations",
        str(kwargs["iterations"]),
        "--after-prefill",
        "--out",
        str(profile_path),
    ]
    for option, key in (
        ("--tokenizer-path", "tokenizer_path"),
        ("--prefill-len", "prefill_len"),
        ("--batch-size", "batch_size"),
        ("--cache-len", "cache_len"),
    ):
        if kwargs.get(key) is not None:
            command.extend((option, str(kwargs[key])))

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    process_log = profile_path.with_name("profile_process.log")
    process_log.write_text(
        f"exit_code={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}"
        f"--- stderr ---\n{result.stderr}"
    )
    if profile_path.is_file():
        return json.loads(profile_path.read_text())
    return {
        "status": "profile_subprocess_error",
        "passed": False,
        "error": {
            "type": "subprocess_exit",
            "message": (
                f"profile subprocess exited {result.returncode}; "
                f"see {process_log}"
            ),
        },
    }


def _candidate_id(state: dict[str, Any], fingerprint: str) -> str:
    return (
        f"split{state['lm_head_split_count']}-"
        f"{_slug(state['dtype_recipe'])}-"
        f"{_slug(state['memory_layout'])}-"
        f"{_slug(state['program_config'])}-"
        f"{fingerprint[:12]}"
    )


def _slug(value: Any) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in str(value).lower()
    ).strip("-")


def _load_resumable_measurement(
    measurement_path: Path,
    profile_path: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not measurement_path.is_file():
        return None
    payload = json.loads(measurement_path.read_text())
    if payload.get("fingerprint") != fingerprint:
        return None
    record = payload.get("record")
    if not isinstance(record, dict):
        return None
    if record.get("profile_report") and not profile_path.is_file():
        return None
    if record.get("passed") is not True and not _terminal_candidate_failure(
        record
    ):
        return None
    return copy.deepcopy(record)


def _terminal_candidate_failure(record: dict[str, Any]) -> bool:
    error = record.get("error")
    if isinstance(error, dict):
        message = str(error.get("message", ""))
    else:
        message = str(error or "")
    terminal_markers = (
        "beyond max L1 size",
        "Output memory config must be sharded for DRAM sharded program config",
    )
    return record.get("status") == "runtime_error" and any(
        marker in message for marker in terminal_markers
    )
