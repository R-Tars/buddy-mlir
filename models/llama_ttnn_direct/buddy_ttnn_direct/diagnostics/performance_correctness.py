from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..correctness.performance_recipe import (
    DEFAULT_GREEDY_AGREEMENT_THRESHOLD,
    OFFICIAL_PERFORMANCE_TOP1_THRESHOLD,
    OFFICIAL_PERFORMANCE_TOP5_THRESHOLD,
    greedy_agreement,
    load_official_performance_reference,
    token_accuracy,
)
from ..runtime.generate import run_generate
from ..runtime.reports import write_report
from .benchmark_parity import (
    PYTEST_PAGE_PARAMS_ENV,
    PYTEST_PLUGIN_ENABLED_ENV,
    PYTEST_PROFILE_ENV,
    _git_value,
    _run_command,
    _sha256,
    parse_official_accuracy_samples,
)

DEFAULT_REFERENCE = (
    "models/tt_transformers/tests/reference_outputs/" "Llama-3.1-8B-Instruct.refpt"
)
BUDDY_STATIC_PREFILL_TOKEN_LIMIT = 256
OfficialRunner = Callable[
    [Sequence[str], Path, dict[str, str], Path, float | None, int | None],
    int,
]
BuddyRunner = Callable[..., dict[str, Any]]


def run_performance_correctness(
    *,
    out: str | Path,
    buddy_program: str | Path,
    official_tt_metal_root: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    official_python: str | Path | None = None,
    accuracy_reference: str | Path | None = None,
    token_count: int = 500,
    layers: int = 32,
    batch_size: int = 32,
    prefill_len: int = 512,
    cache_len: int = 1024,
    page_block_size: int = 32,
    device: str = "p150a",
    device_id: int = 0,
    timeout_seconds: float | None = 3600.0,
    address_space_limit_bytes: int | None = 95_000_000_000,
    dry_run: bool = False,
    official_runner: OfficialRunner | None = None,
    buddy_runner: BuddyRunner | None = None,
) -> dict[str, Any]:
    report_path = Path(out).resolve()
    program_root = Path(buddy_program).resolve()
    official_root = Path(official_tt_metal_root).resolve()
    model_root = Path(model_path).resolve()
    tokenizer_root = Path(tokenizer_path or model_root).resolve()
    official_python_path = Path(
        os.path.abspath(os.path.expanduser(str(official_python or sys.executable)))
    )
    reference_path = Path(
        accuracy_reference or official_root / DEFAULT_REFERENCE
    ).resolve()
    runs_root = report_path.parent / f"{report_path.stem}_runs"
    official_run_dir = runs_root / "official-performance-recipe"
    official_log = official_run_dir / "run.log"
    buddy_report_path = runs_root / "buddy-performance-recipe" / "generate.json"
    base = {
        "schema_version": 1,
        "command": "diagnose",
        "stage": "performance-correctness",
        "status": "running",
        "passed": False,
        "dry_run": bool(dry_run),
        "report_path": str(report_path),
        "buddy_program": str(program_root),
        "official_tt_metal_root": str(official_root),
        "model_path": str(model_root),
        "tokenizer_path": str(tokenizer_root),
        "official_python": str(official_python_path),
        "accuracy_reference": str(reference_path),
        "token_count": int(token_count),
        "layers": int(layers),
        "batch_size": int(batch_size),
        "prefill_len": int(prefill_len),
        "cache_len": int(cache_len),
        "page_block_size": int(page_block_size),
        "device": device,
        "device_id": int(device_id),
        "thresholds": {
            "published_official_top1": 0.90,
            "published_official_top5": 0.98,
            "top1_gate": OFFICIAL_PERFORMANCE_TOP1_THRESHOLD,
            "top5_gate": OFFICIAL_PERFORMANCE_TOP5_THRESHOLD,
            "greedy_agreement_gate": DEFAULT_GREEDY_AGREEMENT_THRESHOLD,
            "rounding_allowance": 0.005,
        },
        "error": None,
    }
    write_report(report_path, base)

    try:
        _validate_inputs(
            program_root=program_root,
            official_root=official_root,
            model_root=model_root,
            tokenizer_root=tokenizer_root,
            official_python=official_python_path,
            token_count=token_count,
            layers=layers,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
        )
        reference = load_official_performance_reference(
            reference_path,
            token_count=token_count,
        )
        reference_prompt_ids = reference["prompt_token_ids"]
        if len(reference_prompt_ids) != int(prefill_len):
            raise ValueError(
                "fixed corpus prompt length does not match prefill_len: "
                f"{len(reference_prompt_ids)} != {prefill_len}"
            )
        buddy_prefill_token_count = min(
            len(reference_prompt_ids),
            BUDDY_STATIC_PREFILL_TOKEN_LIMIT,
        )
        buddy_prefill_ids = reference_prompt_ids[:buddy_prefill_token_count]
        prompt_replay_ids = reference_prompt_ids[buddy_prefill_token_count:]
        prompt = _buddy_prompt_from_reference(
            buddy_prefill_ids,
            tokenizer_root=tokenizer_root,
        )
        prompt_ids = prompt["roundtrip_token_ids"]
        if prompt_ids != buddy_prefill_ids:
            raise ValueError(
                "Buddy prefill prompt tokenization does not match fixed corpus"
            )
        base.update(
            {
                "official_tt_metal_commit": _git_value(
                    official_root, "rev-parse", "HEAD"
                ),
                "fixed_corpus": {
                    "kind": reference["kind"],
                    "path": reference["path"],
                    "sha256": reference["sha256"],
                    "full_sequence_token_count": reference["full_sequence_token_count"],
                    "prompt_token_count": len(reference["prompt_token_ids"]),
                    "evaluated_token_count": int(token_count),
                    "prompt_token_ids_sha256": _token_ids_sha256(
                        reference["prompt_token_ids"]
                    ),
                    "target_token_ids_sha256": _token_ids_sha256(
                        reference["target_token_ids"]
                    ),
                    "top5_token_ids_sha256": _token_ids_sha256(
                        reference["top5_token_ids"]
                    ),
                    "buddy_prompt_roundtrip_exact": True,
                    "buddy_prompt_adaptation": prompt["adaptation"],
                    "buddy_static_prefill_token_limit": (
                        BUDDY_STATIC_PREFILL_TOKEN_LIMIT
                    ),
                    "buddy_prefill_token_count": buddy_prefill_token_count,
                    "buddy_prompt_replay_decode_token_count": len(prompt_replay_ids),
                    "buddy_prompt_replay_token_ids_sha256": (
                        _token_ids_sha256(prompt_replay_ids)
                    ),
                },
            }
        )
        command = _official_command(
            official_python=official_python_path,
            token_count=token_count,
            layers=layers,
            junit_path=official_run_dir / "pytest.xml",
        )
        base["official_run"] = {
            "profile": "official-token-accuracy",
            "optimization_recipe": "performance",
            "teacher_forcing": True,
            "trace": False,
            "sampling": "force argmax",
            "command": command,
            "cwd": str(official_root),
            "log_path": str(official_log),
            "status": "planned",
        }
        base["buddy_run"] = {
            "profile": "buddy-performance-recipe",
            "optimization_recipe": "official_static_performance_config",
            "teacher_forcing": True,
            "trace": False,
            "sampling": "force argmax",
            "runtime_input_mode": "persistent",
            "prefill_token_count": buddy_prefill_token_count,
            "prompt_replay_decode_token_count": len(prompt_replay_ids),
            "report_path": str(buddy_report_path),
            "status": "planned",
        }
        if dry_run:
            base.update(
                {
                    "status": "dry_run",
                    "passed": True,
                    "acceptance": {
                        "status": "dry_run",
                        "passed": True,
                        "failed_checks": [],
                    },
                }
            )
            write_report(report_path, base)
            return base

        official_run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        return_code = (official_runner or _run_command)(
            command,
            official_root,
            _official_environment(
                official_root=official_root,
                model_root=model_root,
                tensor_cache_path=runs_root / "official_tensor_cache",
                cache_len=cache_len,
                page_block_size=page_block_size,
            ),
            official_log,
            timeout_seconds,
            address_space_limit_bytes,
        )
        official_predictions = parse_official_accuracy_samples(official_log)
        base["official_run"].update(
            {
                "return_code": int(return_code),
                "elapsed_seconds": time.time() - started,
                "sample_count": len(official_predictions),
                "prediction_sha256": _token_ids_sha256(official_predictions),
                "status": (
                    "passed"
                    if return_code == 0 and len(official_predictions) == token_count
                    else "failed"
                ),
            }
        )
        write_report(report_path, base)
        if return_code != 0 or len(official_predictions) != token_count:
            raise RuntimeError(
                "official performance correctness run failed or returned an "
                f"unexpected sample count: rc={return_code}, "
                f"samples={len(official_predictions)}, expected={token_count}"
            )

        teacher_forcing_token_ids = [
            *prompt_replay_ids,
            *reference["target_token_ids"][:-1],
        ]
        teacher_forcing = [
            [int(token_id)] * batch_size for token_id in teacher_forcing_token_ids
        ]
        buddy_generated_token_count = len(prompt_replay_ids) + token_count
        buddy_report_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        buddy = (buddy_runner or run_generate)(
            out=buddy_report_path,
            program_dir=program_root,
            model_path=model_root,
            prompt=prompt["text"],
            tokenizer_path=tokenizer_root,
            max_new_tokens=buddy_generated_token_count,
            layers=layers,
            prefill_len=buddy_prefill_token_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed="bf16",
            dry_run=False,
            report_level="summary",
            runtime_input_mode="persistent",
            execution_mode="eager",
            prefill_execution_mode="eager",
            teacher_forcing_token_ids_by_step=teacher_forcing,
        )
        base["buddy_run"].update(
            {
                "elapsed_seconds": time.time() - started,
                "runtime_status": buddy.get("status"),
                "report_sha256": _sha256(buddy_report_path),
                "status": "passed" if buddy.get("passed") else "failed",
            }
        )
        if not buddy.get("passed"):
            raise RuntimeError(
                "Buddy performance correctness generation failed: "
                f"{buddy.get('status')}: {buddy.get('message') or buddy.get('error')}"
            )
        buddy_rows = _prediction_rows(
            buddy,
            batch_size=batch_size,
            token_count=token_count,
            prediction_offset=len(prompt_replay_ids),
        )
        base["buddy_run"]["prediction_sha256_by_user"] = [
            _token_ids_sha256(row) for row in buddy_rows
        ]

        official_accuracy = token_accuracy(
            official_predictions,
            reference["top5_token_ids"],
        )
        buddy_per_user = [
            token_accuracy(row, reference["top5_token_ids"]) for row in buddy_rows
        ]
        buddy_aggregate = token_accuracy(
            [token for row in buddy_rows for token in row],
            list(reference["top5_token_ids"]) * batch_size,
        )
        agreement_per_user = [
            greedy_agreement(row, official_predictions) for row in buddy_rows
        ]
        metrics = {
            "official": official_accuracy,
            "buddy_aggregate": buddy_aggregate,
            "buddy_user_zero": buddy_per_user[0],
            "buddy_min_user_top1_accuracy": min(
                value["top1_accuracy"] for value in buddy_per_user
            ),
            "buddy_min_user_top5_accuracy": min(
                value["top5_accuracy"] for value in buddy_per_user
            ),
            "buddy_per_user": buddy_per_user,
            "official_buddy_greedy_agreement_user_zero": agreement_per_user[0],
            "official_buddy_min_user_greedy_agreement": min(
                value["agreement"] for value in agreement_per_user
            ),
            "official_buddy_per_user_greedy_agreement": agreement_per_user,
        }
        base["metrics"] = metrics
        checks = {
            "official_sample_count": len(official_predictions) == token_count,
            "buddy_batch_complete": len(buddy_rows) == batch_size,
            "official_top1": official_accuracy["top1_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP1_THRESHOLD,
            "official_top5": official_accuracy["top5_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP5_THRESHOLD,
            "buddy_top1": buddy_aggregate["top1_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP1_THRESHOLD,
            "buddy_top5": buddy_aggregate["top5_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP5_THRESHOLD,
            "buddy_min_user_top1": metrics["buddy_min_user_top1_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP1_THRESHOLD,
            "buddy_min_user_top5": metrics["buddy_min_user_top5_accuracy"]
            >= OFFICIAL_PERFORMANCE_TOP5_THRESHOLD,
            "official_buddy_min_user_greedy_agreement": metrics[
                "official_buddy_min_user_greedy_agreement"
            ]
            >= DEFAULT_GREEDY_AGREEMENT_THRESHOLD,
        }
        failed = [name for name, passed in checks.items() if not passed]
        base.update(
            {
                "status": "passed" if not failed else "failed",
                "passed": not failed,
                "acceptance": {
                    "status": "passed" if not failed else "failed",
                    "passed": not failed,
                    "checks": checks,
                    "failed_checks": failed,
                },
            }
        )
    except Exception as error:
        base.update(
            {
                "status": "failed",
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "acceptance": {
                    "status": "failed",
                    "passed": False,
                    "failed_checks": ["performance_correctness.execution"],
                },
            }
        )

    write_report(report_path, base)
    return base


def _validate_inputs(**values: Any) -> None:
    for name in (
        "program_root",
        "official_root",
        "model_root",
        "tokenizer_root",
        "official_python",
    ):
        if not Path(values[name]).exists():
            raise ValueError(f"missing {name}: {values[name]}")
    for name in (
        "token_count",
        "layers",
        "batch_size",
        "prefill_len",
        "cache_len",
    ):
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(values["prefill_len"]) + int(values["token_count"]) - 1 > int(
        values["cache_len"]
    ):
        raise ValueError("fixed corpus plus generated tokens exceeds cache_len")


def _official_command(
    *,
    official_python: Path,
    token_count: int,
    layers: int,
    junit_path: Path,
) -> list[str]:
    return [
        str(official_python),
        "-m",
        "pytest",
        "-s",
        "-q",
        "models/tt_transformers/demo/simple_text_demo.py",
        "-k",
        "performance-ci-token-matching",
        "-p",
        ("models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics." "benchmark_parity"),
        "--max_generated_tokens",
        str(token_count),
        "--num_layers",
        str(layers),
        "--disable_trace",
        "--stop_at_eos",
        "0",
        "--mode",
        "full",
        "--junitxml",
        str(junit_path),
    ]


def _official_environment(
    *,
    official_root: Path,
    model_root: Path,
    tensor_cache_path: Path,
    cache_len: int,
    page_block_size: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[4]
    python_paths = [
        str(repo_root),
        str(official_root),
        str(official_root / "tools"),
        str(official_root / "ttnn"),
        str(official_root / "tt_eager"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "HF_MODEL": str(model_root),
            "MESH_DEVICE": "P150",
            "TT_CACHE_PATH": str(tensor_cache_path),
            "TT_METAL_HOME": str(official_root),
            PYTEST_PLUGIN_ENABLED_ENV: "1",
            PYTEST_PROFILE_ENV: "official-token-accuracy",
            PYTEST_PAGE_PARAMS_ENV: json.dumps(
                {
                    "page_block_size": page_block_size,
                    "page_max_num_blocks_per_dp": (cache_len * 32 // page_block_size),
                },
                separators=(",", ":"),
            ),
        }
    )
    environment.pop("LLAMA_DIR", None)
    return environment


def _buddy_prompt_from_reference(
    prompt_token_ids: Sequence[int],
    *,
    tokenizer_root: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root),
        local_files_only=True,
    )
    prompt_ids = [int(token_id) for token_id in prompt_token_ids]
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    decode_ids = prompt_ids
    adaptation = "none"
    if prompt_ids and bos_token_id is not None and prompt_ids[0] == int(bos_token_id):
        decode_ids = prompt_ids[1:]
        adaptation = "strip serialized BOS; Buddy tokenizer restores one BOS"
    text = tokenizer.decode(
        decode_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    tokenized = tokenizer(text, add_special_tokens=True)
    roundtrip = tokenized["input_ids"]
    if hasattr(roundtrip, "tolist"):
        roundtrip = roundtrip.tolist()
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "roundtrip_token_ids": [int(token_id) for token_id in roundtrip],
        "adaptation": adaptation,
    }


def _prediction_rows(
    report: dict[str, Any],
    *,
    batch_size: int,
    token_count: int,
    prediction_offset: int = 0,
) -> list[list[int]]:
    rows = report.get("generated_token_ids")
    if not isinstance(rows, list) or len(rows) != batch_size:
        raise ValueError("Buddy report does not contain the expected token batch")
    normalized = [[int(token_id) for token_id in row] for row in rows]
    expected_count = int(prediction_offset) + int(token_count)
    if any(len(row) != expected_count for row in normalized):
        raise ValueError(
            "Buddy prediction row length does not match prompt replay plus "
            "token_count"
        )
    return [row[prediction_offset:expected_count] for row in normalized]


def _token_ids_sha256(token_ids: Sequence[Any]) -> str:
    encoded = json.dumps(
        token_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
