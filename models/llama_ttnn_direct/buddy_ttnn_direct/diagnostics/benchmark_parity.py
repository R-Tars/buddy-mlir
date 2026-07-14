from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

from ..runtime.reports import write_report

OFFICIAL_EXTERNAL_REFERENCE_TPSU = 33.1
OFFICIAL_EXTERNAL_REFERENCE_TTFT_MS = 57.0
OFFICIAL_EXTERNAL_RELEASE_TAG = "v0.64.0-dev20251030"
OFFICIAL_EXTERNAL_RELEASE_COMMIT = "b76035fbdac81d8f9974976471dc60fc005e1bfb"
DEFAULT_PROMPTS = (
    "models/tt_transformers/demo/sample_prompts/"
    "input_data_questions_prefill_128.json"
)
PYTEST_PLUGIN_ENABLED_ENV = "BUDDY_PARITY_PYTEST_PLUGIN"
PYTEST_PROFILE_ENV = "BUDDY_PARITY_OFFICIAL_PROFILE"
PYTEST_PAGE_PARAMS_ENV = "BUDDY_PARITY_PAGE_PARAMS"
PYTEST_GRAPH_CAPTURE_DIR_ENV = "BUDDY_PARITY_GRAPH_CAPTURE_DIR"
EXACT_SAMPLE_PREFIX = "BUDDY_PARITY_SAMPLE"
_EXACT_SAMPLE_RE = re.compile(
    rf"{EXACT_SAMPLE_PREFIX} token_iteration=(?P<iteration>\d+) "
    r"duration_ms=(?P<duration_ms>\d+(?:\.\d+)?)"
)
_ROUNDED_SAMPLE_RE = re.compile(
    r"Iteration (?P<iteration>\d+): "
    r"(?P<duration_ms>\d+(?:\.\d+)?)ms @ "
    r"(?P<tpsu>\d+(?:\.\d+)?) tok/s/user"
)


CommandRunner = Callable[
    [Sequence[str], Path, dict[str, str], Path, float | None, int | None],
    int,
]


def pytest_configure(config: Any) -> None:
    """Inject exact official benchmark settings when loaded as a pytest plugin."""

    if os.environ.get(PYTEST_PLUGIN_ENABLED_ENV) != "1":
        return

    page_params = json.loads(os.environ[PYTEST_PAGE_PARAMS_ENV])
    config.option.page_params = page_params
    config.option.enable_trace = True
    config.option.stop_at_eos = 0
    if os.environ.get(PYTEST_PROFILE_ENV) == "official-greedy":
        config.option.sampling_params = {
            "temperature": 0,
            "top_p": 0.08,
            "top_k": 32,
        }

    graph_capture_dir = os.environ.get(PYTEST_GRAPH_CAPTURE_DIR_ENV)
    if graph_capture_dir:
        _install_official_trace_graph_capture(Path(graph_capture_dir))

    from models.perf.benchmarking_utils import BenchmarkProfiler

    if getattr(BenchmarkProfiler, "_buddy_parity_patched", False):
        return
    original_end = BenchmarkProfiler.end

    def end_with_exact_sample(
        self: Any,
        step_name: str,
        iteration: int = 0,
    ) -> None:
        original_end(self, step_name, iteration)
        if not step_name.startswith("inference_decode_time_"):
            return
        token_iteration = int(step_name.rsplit("_", 1)[1])
        duration_ms = self.get_duration(step_name, iteration) * 1000.0
        print(
            f"{EXACT_SAMPLE_PREFIX} token_iteration={token_iteration} "
            f"duration_ms={duration_ms:.6f}",
            flush=True,
        )

    BenchmarkProfiler.end = end_with_exact_sample
    BenchmarkProfiler._buddy_parity_patched = True


def _install_official_trace_graph_capture(output_dir: Path) -> None:
    """Wrap official trace capture without modifying the tt-metal checkout."""

    import ttnn

    if getattr(ttnn, "_buddy_graph_capture_patched", False):
        return
    original_begin = ttnn.begin_trace_capture
    original_end = ttnn.end_trace_capture
    state: dict[str, Any] = {
        "active": None,
        "captures": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    def begin_with_graph(device: Any, *args: Any, **kwargs: Any) -> Any:
        if state["active"] is not None:
            raise RuntimeError("nested official trace graph capture is unsupported")
        graph = ttnn.graph
        run_mode = getattr(getattr(graph, "RunMode", None), "NORMAL", None)
        if run_mode is None:
            graph.begin_graph_capture()
        else:
            graph.begin_graph_capture(run_mode)
        entry = {
            "index": len(state["captures"]),
            "program_cache_entries_before": _device_program_cache_entries(device),
        }
        state["active"] = entry
        try:
            trace_id = original_begin(device, *args, **kwargs)
        except Exception:
            graph.end_graph_capture()
            state["active"] = None
            raise
        entry["trace_id"] = str(trace_id)
        entry["device"] = device
        return trace_id

    def end_with_graph(
        device: Any,
        trace_id: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_end(device, trace_id, *args, **kwargs)
        entry = state["active"]
        if entry is None:
            raise RuntimeError("official trace ended without graph capture state")
        captured_graph = ttnn.graph.end_graph_capture()
        graph_path = output_dir / f"trace_{entry['index']:03d}.json"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "source": "official-tt-metal-trace-capture",
            "raw_graph": captured_graph,
        }
        try:
            from ttnn.graph_tracer_utils import GraphTracerUtils

            payload["serialized_graph"] = GraphTracerUtils.serialize_graph(
                captured_graph
            )
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            payload["serialization_error"] = f"{type(error).__name__}: {error}"
        graph_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        entry.pop("device", None)
        entry.update(
            {
                "graph_path": str(graph_path),
                "program_cache_entries_after": (_device_program_cache_entries(device)),
            }
        )
        state["captures"].append(entry)
        state["active"] = None
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "captures": state["captures"],
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        return result

    ttnn.begin_trace_capture = begin_with_graph
    ttnn.end_trace_capture = end_with_graph
    ttnn._buddy_graph_capture_patched = True


def _device_program_cache_entries(device: Any) -> int | None:
    count = getattr(device, "num_program_cache_entries", None)
    if not callable(count):
        return None
    try:
        return int(count())
    except (RuntimeError, TypeError, ValueError):
        return None


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
    timeout_seconds: float | None = 3600.0,
    address_space_limit_bytes: int | None = 95_000_000_000,
    dry_run: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run matched official-demo, official-greedy, and Buddy decode profiles."""

    report_path = Path(out).resolve()
    program_root = Path(buddy_program).resolve()
    official_root = Path(official_tt_metal_root).resolve()
    model_root = Path(model_path).resolve()
    tokenizer_root = Path(tokenizer_path or model_root).resolve()
    prompts_path = Path(input_prompts or official_root / DEFAULT_PROMPTS).resolve()
    official_python_path = _absolute_path(official_python or sys.executable)
    release_root = (
        Path(official_release_root).resolve() if official_release_root else None
    )
    release_python_path = (
        _absolute_path(official_release_python) if official_release_python else None
    )
    if (release_root is None) != (release_python_path is None):
        raise ValueError(
            "official_release_root and official_release_python must be provided together"
        )
    runs_root = report_path.parent / f"{report_path.stem}_runs"
    tensor_cache_root = (
        report_path.parent.parent / "runtime_artifacts" / "official_tensor_cache"
    )
    staged_prompts_path = runs_root / "inputs" / "input_prompts.json"
    runner = command_runner or _run_command

    _validate_positive("batch_size", batch_size)
    _validate_positive("prefill_len", prefill_len)
    _validate_positive("cache_len", cache_len)
    _validate_positive("page_block_size", page_block_size)
    _validate_positive("iterations", iterations)
    _validate_positive("repetitions", repetitions)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    base = _base_report(
        report_path=report_path,
        program_root=program_root,
        official_root=official_root,
        model_root=model_root,
        tokenizer_root=tokenizer_root,
        prompts_path=prompts_path,
        official_python=official_python_path,
        release_root=release_root,
        release_python=release_python_path,
        batch_size=batch_size,
        requested_prefill_len=prefill_len,
        cache_len=cache_len,
        page_block_size=page_block_size,
        warmup=warmup,
        iterations=iterations,
        repetitions=repetitions,
        device=device,
        device_id=device_id,
        timeout_seconds=timeout_seconds,
        address_space_limit_bytes=address_space_limit_bytes,
    )
    write_report(report_path, base)

    try:
        metadata = _collect_metadata(
            program_root=program_root,
            official_root=official_root,
            model_root=model_root,
            tokenizer_root=tokenizer_root,
            prompts_path=prompts_path,
            official_python=official_python_path,
            release_root=release_root,
            release_python=release_python_path,
            batch_size=batch_size,
            requested_prefill_len=prefill_len,
            cache_len=cache_len,
        )
        base.update(metadata)
        commands = _planned_commands(
            runs_root=runs_root,
            tensor_cache_root=tensor_cache_root,
            program_root=program_root,
            official_root=official_root,
            model_root=model_root,
            tokenizer_root=tokenizer_root,
            prompts_path=staged_prompts_path,
            official_python=official_python_path,
            release_root=release_root,
            release_python=release_python_path,
            layer_count=int(metadata["layers"]),
            batch_size=batch_size,
            effective_prefill_len=int(metadata["effective_prefill_len"]),
            cache_len=cache_len,
            page_block_size=page_block_size,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
            device=device,
            device_id=device_id,
        )
        base["planned_runs"] = commands
        base["staged_input_prompts"] = str(staged_prompts_path)
        if dry_run:
            base.update(
                {
                    "status": "dry_run",
                    "passed": True,
                    "dry_run": True,
                    "acceptance": {
                        "status": "dry_run",
                        "passed": True,
                        "failed_checks": [],
                    },
                }
            )
            write_report(report_path, base)
            return base

        runs_root.mkdir(parents=True, exist_ok=True)
        staged_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prompts_path, staged_prompts_path)
        for plan in commands:
            run = _execute_planned_run(
                plan=plan,
                page_block_size=page_block_size,
                cache_len=cache_len,
                warmup=warmup,
                iterations=iterations,
                runner=runner,
                timeout_seconds=timeout_seconds,
                address_space_limit_bytes=address_space_limit_bytes,
            )
            target = (
                base["buddy_local_runs"]
                if plan["implementation"] == "buddy"
                else (
                    base["official_release_runs"]
                    if plan["comparison_scope"] == "release-reference"
                    else base["official_local_runs"]
                )
            )
            target.append(run)
            write_report(report_path, base)

        _finalize_report(base)
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
                    "failed_checks": ["benchmark_parity.execution"],
                },
            }
        )

    write_report(report_path, base)
    return base


def _base_report(
    *,
    report_path: Path,
    program_root: Path,
    official_root: Path,
    model_root: Path,
    tokenizer_root: Path,
    prompts_path: Path,
    official_python: Path,
    release_root: Path | None,
    release_python: Path | None,
    batch_size: int,
    requested_prefill_len: int,
    cache_len: int,
    page_block_size: int,
    warmup: int,
    iterations: int,
    repetitions: int,
    device: str,
    device_id: int,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": "diagnose",
        "stage": "benchmark-parity",
        "status": "running",
        "passed": False,
        "dry_run": False,
        "report_path": str(report_path),
        "official_external_reference": {
            "decode_tpsu": OFFICIAL_EXTERNAL_REFERENCE_TPSU,
            "ttft_ms": OFFICIAL_EXTERNAL_REFERENCE_TTFT_MS,
            "release_tag": OFFICIAL_EXTERNAL_RELEASE_TAG,
            "release_commit": OFFICIAL_EXTERNAL_RELEASE_COMMIT,
            "comparison_scope": "cross-version-reference-only",
        },
        "official_external_reference_tpsu": OFFICIAL_EXTERNAL_REFERENCE_TPSU,
        "buddy_program": str(program_root),
        "official_tt_metal_root": str(official_root),
        "model_path": str(model_root),
        "tokenizer_path": str(tokenizer_root),
        "input_prompts": str(prompts_path),
        "official_python": str(official_python),
        "official_tensor_cache_root": str(
            report_path.parent.parent / "runtime_artifacts" / "official_tensor_cache"
        ),
        "official_release_root": str(release_root) if release_root else None,
        "official_release_python": (str(release_python) if release_python else None),
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "requested_prefill_len": requested_prefill_len,
        "cache_len": cache_len,
        "page_block_size": page_block_size,
        "warmup": warmup,
        "iterations": iterations,
        "repetitions": repetitions,
        "timeout_seconds": timeout_seconds,
        "address_space_limit_bytes": address_space_limit_bytes,
        "official_local_runs": [],
        "official_release_runs": [],
        "buddy_local_runs": [],
        "official_release_median_tpsu": None,
        "buddy_ratio_of_release_official": None,
        "official_local_median_tpsu": None,
        "buddy_local_median_tpsu": None,
        "buddy_ratio_of_local_official": None,
        "coefficient_of_variation": {},
        "semantic_match": {},
        "error": None,
    }


def _collect_metadata(
    *,
    program_root: Path,
    official_root: Path,
    model_root: Path,
    tokenizer_root: Path,
    prompts_path: Path,
    official_python: Path,
    release_root: Path | None,
    release_python: Path | None,
    batch_size: int,
    requested_prefill_len: int,
    cache_len: int,
) -> dict[str, Any]:
    required = {
        "buddy program": program_root,
        "official tt-metal root": official_root,
        "model": model_root,
        "tokenizer": tokenizer_root,
        "prompt file": prompts_path,
        "official Python": official_python,
    }
    if release_root is not None:
        required["official release root"] = release_root
    if release_python is not None:
        required["official release Python"] = release_python
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError("missing benchmark inputs: " + ", ".join(missing))

    config_path = program_root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing Buddy program config: {config_path}")
    config = json.loads(config_path.read_text())
    layers = int(config["num_layers"])
    prompt_values = _load_prompt_values(prompts_path, batch_size)
    token_lengths = _token_lengths(
        prompt_values,
        tokenizer_path=tokenizer_root,
        instruct=True,
    )
    max_prompt_tokens = max(token_lengths)
    effective_prefill_len = _effective_prefill_len(
        requested_prefill_len,
        max_prompt_tokens,
    )
    if effective_prefill_len > cache_len:
        raise ValueError(
            f"effective prefill length {effective_prefill_len} exceeds "
            f"cache length {cache_len}"
        )

    repo_root = Path(__file__).resolve().parents[4]
    buddy_tt_metal_root = Path(
        os.environ.get("TT_METAL_HOME", str(official_root))
    ).resolve()
    official_commit = _git_value(official_root, "rev-parse", "HEAD")
    buddy_tt_metal_commit = _git_value(
        buddy_tt_metal_root,
        "rev-parse",
        "HEAD",
    )
    buddy_commit = _git_value(repo_root, "rev-parse", "HEAD")
    same_tt_metal_commit = official_commit == buddy_tt_metal_commit
    if not same_tt_metal_commit:
        raise ValueError(
            "official and Buddy tt-metal commits differ: "
            f"{official_commit} != {buddy_tt_metal_commit}"
        )

    release_metadata = None
    if release_root is not None:
        release_commit = _git_value(release_root, "rev-parse", "HEAD")
        release_diff_names = _git_value(
            release_root, "diff", "--name-only"
        ).splitlines()
        release_status = _git_value(release_root, "status", "--porcelain=v1")
        release_metadata = {
            "root": str(release_root),
            "python": str(release_python),
            "commit": release_commit,
            "expected_release_tag": OFFICIAL_EXTERNAL_RELEASE_TAG,
            "expected_release_commit": OFFICIAL_EXTERNAL_RELEASE_COMMIT,
            "matches_external_release_commit": (
                release_commit == OFFICIAL_EXTERNAL_RELEASE_COMMIT
            ),
            "source_dirty": bool(release_status),
            "tracked_source_dirty": bool(release_diff_names),
            "tracked_source_changes": release_diff_names,
            "source_status": release_status.splitlines(),
            "source_diff_sha256": _git_diff_sha256(release_root),
            "sampling": "force argmax (temperature=0) in the release demo",
            "trace_mode": "trace",
            "comparison_scope": "cross-version-reference-only",
        }
        if release_commit != OFFICIAL_EXTERNAL_RELEASE_COMMIT:
            raise ValueError(
                "official release root is not the published reference commit: "
                f"{release_commit} != {OFFICIAL_EXTERNAL_RELEASE_COMMIT}"
            )

    official_execution_features = _official_execution_features(
        official_root=official_root,
        release_root=release_root,
    )

    return {
        "buddy_commit": buddy_commit,
        "official_tt_metal_commit": official_commit,
        "buddy_tt_metal_root": str(buddy_tt_metal_root),
        "buddy_tt_metal_commit": buddy_tt_metal_commit,
        "same_tt_metal_commit": same_tt_metal_commit,
        "official_release": release_metadata,
        "official_execution_features": official_execution_features,
        "layers": layers,
        "model_config_sha256": _sha256(model_root / "config.json"),
        "input_prompts_sha256": _sha256(prompts_path),
        "prompt_count": len(prompt_values),
        "prompt_token_lengths": token_lengths,
        "max_prompt_tokens": max_prompt_tokens,
        "effective_prefill_len": effective_prefill_len,
        "prefill_adjustment": {
            "requested": requested_prefill_len,
            "effective": effective_prefill_len,
            "reason": (
                "official prompts require the next power-of-two prefill bucket"
                if effective_prefill_len != requested_prefill_len
                else "requested bucket fits every prompt"
            ),
        },
        "sampling": {
            "official-demo": "official non-uniform device sampling",
            "official-greedy": "force argmax (temperature=0)",
            "buddy": "force argmax",
        },
        "trace_mode": {
            "official-demo": "trace",
            "official-greedy": "trace",
            "buddy": "trace",
        },
    }


def _official_execution_features(
    *,
    official_root: Path,
    release_root: Path | None,
) -> dict[str, dict[str, Any]]:
    """Describe execution controls selected by the parity harness."""

    current_supports_prefetcher = _source_mentions_prefetcher(official_root)

    def profile(
        *,
        source_supports_prefetcher: bool,
        sampling_mode: str,
        comparison_scope: str,
    ) -> dict[str, Any]:
        return {
            "use_prefetcher": False,
            "use_prefetcher_requested_by_command": False,
            "prefetcher_supported_by_source": source_supports_prefetcher,
            "global_cb": None,
            "global_cb_active": False,
            "sub_device_id": None,
            "trace": True,
            "sampling_mode": sampling_mode,
            "comparison_scope": comparison_scope,
            "selection_reason": (
                "the parity command omits --use_prefetcher and the official "
                "fixture/CLI default is disabled"
                if source_supports_prefetcher
                else "the target source does not expose the Llama prefetcher path"
            ),
        }

    features = {
        "official-demo": profile(
            source_supports_prefetcher=current_supports_prefetcher,
            sampling_mode="official non-uniform device sampling",
            comparison_scope="same-commit-local",
        ),
        "official-greedy": profile(
            source_supports_prefetcher=current_supports_prefetcher,
            sampling_mode="force argmax (temperature=0)",
            comparison_scope="same-commit-local",
        ),
    }
    if release_root is not None:
        features["official-release-demo"] = profile(
            source_supports_prefetcher=_source_mentions_prefetcher(release_root),
            sampling_mode="force argmax (temperature=0) in the release demo",
            comparison_scope="release-reference",
        )
    return features


def _source_mentions_prefetcher(root: Path) -> bool:
    relative_paths = (
        Path("models/tt_transformers/demo/conftest.py"),
        Path("models/tt_transformers/demo/simple_text_demo.py"),
        Path("models/tt_transformers/tt/common.py"),
    )
    for relative_path in relative_paths:
        source_path = root / relative_path
        if source_path.is_file() and "use_prefetcher" in source_path.read_text():
            return True
    return False


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
    plugin_name = (
        "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics." "benchmark_parity"
    )
    test_file = "models/tt_transformers/demo/simple_text_demo.py"
    official_profiles: list[tuple[str, Path, Path, str, str]] = []
    if release_root is not None and release_python is not None:
        official_profiles.append(
            (
                "official-release-demo",
                release_root,
                release_python,
                "release-reference",
                OFFICIAL_EXTERNAL_RELEASE_COMMIT,
            )
        )
    official_profiles.extend(
        (
            profile,
            official_root,
            official_python,
            "same-commit-local",
            _git_value(official_root, "rev-parse", "HEAD"),
        )
        for profile in ("official-demo", "official-greedy")
    )
    for (
        profile,
        profile_root,
        profile_python,
        comparison_scope,
        cache_name,
    ) in official_profiles:
        for repetition in range(1, repetitions + 1):
            run_dir = runs_root / profile / f"repetition_{repetition:02d}"
            command = [
                str(profile_python),
                "-m",
                "pytest",
                "-s",
                "-q",
                test_file,
                "-k",
                "performance-batch-32 and not log-probs",
                "-p",
                plugin_name,
                "--input_prompts",
                str(prompts_path),
                "--instruct",
                "1",
                "--repeat_batches",
                "1",
                "--max_seq_len",
                str(cache_len),
                "--batch_size",
                str(batch_size),
                "--max_generated_tokens",
                str(generated_tokens),
                "--stop_at_eos",
                "0",
            ]
            if comparison_scope == "release-reference":
                command.extend(["--enable_trace", "1"])
            else:
                command.extend(["--enable_trace", "--mode", "full"])
            command.extend(["--junitxml", str(run_dir / "pytest.xml")])
            plans.append(
                {
                    "implementation": "official",
                    "profile": profile,
                    "comparison_scope": comparison_scope,
                    "repetition": repetition,
                    "run_dir": str(run_dir),
                    "log_path": str(run_dir / "run.log"),
                    "command": command,
                    "model_path": str(model_root),
                    "official_root": str(profile_root),
                    "tensor_cache_path": str(tensor_cache_root / cache_name),
                }
            )

    buddy_python = _absolute_path(sys.executable)
    for repetition in range(1, repetitions + 1):
        profile = "buddy-greedy"
        run_dir = runs_root / profile / f"repetition_{repetition:02d}"
        profile_report = run_dir / "profile.json"
        command = [
            str(buddy_python),
            "-m",
            "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
            "profile",
            "--mode",
            "decode-steady",
            "--program-dir",
            str(program_root),
            "--model-path",
            str(model_root),
            "--tokenizer-path",
            str(tokenizer_root),
            "--input-prompts",
            str(prompts_path),
            "--instruct",
            "--layers",
            str(layer_count),
            "--prefill-len",
            str(effective_prefill_len),
            "--batch-size",
            str(batch_size),
            "--cache-len",
            str(cache_len),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--after-prefill",
            "--runtime-input-mode",
            "persistent",
            "--execution-mode",
            "trace",
            "--device",
            device,
            "--device-id",
            str(device_id),
            "--dtype-seed",
            "bf16",
            "--out",
            str(profile_report),
        ]
        plans.append(
            {
                "implementation": "buddy",
                "profile": profile,
                "comparison_scope": "same-commit-local",
                "repetition": repetition,
                "run_dir": str(run_dir),
                "log_path": str(run_dir / "run.log"),
                "profile_report": str(profile_report),
                "command": command,
            }
        )
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
    run_dir = Path(plan["run_dir"])
    log_path = Path(plan["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    cwd = Path.cwd()
    if plan["implementation"] == "official":
        official_root = Path(plan["official_root"])
        cwd = official_root
        repo_root = Path(__file__).resolve().parents[4]
        python_paths = [
            str(repo_root),
            str(official_root),
            str(official_root / "tools"),
        ]
        if plan["comparison_scope"] == "same-commit-local":
            python_paths.extend(
                [
                    str(official_root / "ttnn"),
                    str(official_root / "tt_eager"),
                ]
            )
            existing_pythonpath = environment.get("PYTHONPATH")
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(python_paths),
                "HF_MODEL": plan["model_path"],
                "MESH_DEVICE": "P150",
                "TT_CACHE_PATH": plan["tensor_cache_path"],
                PYTEST_PLUGIN_ENABLED_ENV: "1",
                PYTEST_PROFILE_ENV: plan["profile"],
                PYTEST_PAGE_PARAMS_ENV: json.dumps(
                    {
                        "page_block_size": page_block_size,
                        "page_max_num_blocks_per_dp": (
                            cache_len * 32 // page_block_size
                        ),
                    },
                    separators=(",", ":"),
                ),
            }
        )
        environment.pop("LLAMA_DIR", None)
        if plan["comparison_scope"] == "release-reference":
            for variable in (
                "CONDA_PREFIX",
                "LD_LIBRARY_PATH",
                "TT_METAL_BUILD_HOME",
                "TT_METAL_HOME",
            ):
                environment.pop(variable, None)
        else:
            environment["TT_METAL_HOME"] = str(official_root)

    started_at = time.time()
    try:
        return_code = runner(
            plan["command"],
            cwd,
            environment,
            log_path,
            timeout_seconds,
            address_space_limit_bytes,
        )
        runner_error = None
    except Exception as error:
        return_code = -1
        runner_error = f"{type(error).__name__}: {error}"
    elapsed_seconds = time.time() - started_at

    if plan["implementation"] == "official":
        parsed = _parse_official_log(
            log_path,
            warmup=warmup,
            iterations=iterations,
        )
    else:
        parsed = _parse_buddy_report(
            Path(plan["profile_report"]),
            warmup=warmup,
            iterations=iterations,
        )
    passed = return_code == 0 and bool(parsed["passed"])
    first_samples = [
        *parsed["warmup_step_ms_samples"],
        *parsed["decode_step_ms_samples"],
    ]
    first_decode_step_ms = first_samples[0] if first_samples else None
    return {
        "implementation": plan["implementation"],
        "profile": plan["profile"],
        "comparison_scope": plan["comparison_scope"],
        "repetition": plan["repetition"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "command": plan["command"],
        "cwd": str(cwd),
        "log_path": str(log_path),
        "profile_report": plan.get("profile_report"),
        "warmup_step_ms_samples": parsed["warmup_step_ms_samples"],
        "decode_step_ms_samples": parsed["decode_step_ms_samples"],
        "decode_step_ms_mean": parsed["decode_step_ms_mean"],
        "decode_step_ms_p50": parsed["decode_step_ms_p50"],
        "tokens_per_second_per_user": parsed["tokens_per_second_per_user"],
        "first_decode_step_ms": first_decode_step_ms,
        "first_decode_tokens_per_second_per_user": (
            1000.0 / first_decode_step_ms if first_decode_step_ms else None
        ),
        "sample_source": parsed["sample_source"],
        "error": runner_error or parsed.get("error"),
    }


def _parse_official_log(
    log_path: Path,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if not log_path.is_file():
        return _failed_parse(f"official log was not written: {log_path}")
    text = log_path.read_text(errors="replace")
    samples_by_iteration = {
        int(match.group("iteration")): float(match.group("duration_ms"))
        for match in _EXACT_SAMPLE_RE.finditer(text)
    }
    source = "benchmark_profiler_exact"
    if not samples_by_iteration:
        samples_by_iteration = {
            int(match.group("iteration")): float(match.group("duration_ms"))
            for match in _ROUNDED_SAMPLE_RE.finditer(text)
            if int(match.group("iteration")) > 0
        }
        source = "official_debug_log_rounded_ms"
    ordered = [samples_by_iteration[index] for index in sorted(samples_by_iteration)]
    expected = warmup + iterations
    if len(ordered) < expected:
        return _failed_parse(
            f"official profile produced {len(ordered)} non-compile samples; "
            f"expected at least {expected}",
            sample_source=source,
        )
    return _parsed_samples(
        warmup_samples=ordered[:warmup],
        measured_samples=ordered[warmup : warmup + iterations],
        sample_source=source,
    )


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
    warmup_samples = [
        float(value) for value in report.get("warmup_step_ms_samples", [])
    ]
    measured_samples = [
        float(value) for value in report.get("decode_step_ms_samples", [])
    ]
    parsed = _parsed_samples(
        warmup_samples=warmup_samples,
        measured_samples=measured_samples,
        sample_source="buddy_profile_json",
    )
    if len(warmup_samples) != warmup or len(measured_samples) != iterations:
        parsed.update(
            {
                "passed": False,
                "error": (
                    "Buddy sample counts differ from requested warmup/iterations: "
                    f"{len(warmup_samples)}/{len(measured_samples)} != "
                    f"{warmup}/{iterations}"
                ),
            }
        )
    if not report.get("passed"):
        parsed["passed"] = False
        parsed["error"] = report.get("error") or report.get("message")
    return parsed


def _parsed_samples(
    *,
    warmup_samples: Sequence[float],
    measured_samples: Sequence[float],
    sample_source: str,
) -> dict[str, Any]:
    samples = [float(value) for value in measured_samples]
    valid = bool(samples) and all(
        math.isfinite(value) and value > 0 for value in samples
    )
    mean = statistics.fmean(samples) if valid else None
    return {
        "passed": valid,
        "warmup_step_ms_samples": [float(value) for value in warmup_samples],
        "decode_step_ms_samples": samples,
        "decode_step_ms_mean": mean,
        "decode_step_ms_p50": statistics.median(samples) if valid else None,
        "tokens_per_second_per_user": 1000.0 / mean if mean else None,
        "sample_source": sample_source,
        "error": None if valid else "no valid measured latency samples",
    }


def _failed_parse(
    message: str,
    *,
    sample_source: str = "not_available",
) -> dict[str, Any]:
    return {
        "passed": False,
        "warmup_step_ms_samples": [],
        "decode_step_ms_samples": [],
        "decode_step_ms_mean": None,
        "decode_step_ms_p50": None,
        "tokens_per_second_per_user": None,
        "sample_source": sample_source,
        "error": message,
    }


def _finalize_report(report: dict[str, Any]) -> None:
    official_runs = report["official_local_runs"]
    release_runs = report.get("official_release_runs", [])
    buddy_runs = report["buddy_local_runs"]
    demo_runs = [run for run in official_runs if run["profile"] == "official-demo"]
    greedy_runs = [run for run in official_runs if run["profile"] == "official-greedy"]
    demo_values = _successful_tpsu(demo_runs)
    greedy_values = _successful_tpsu(greedy_runs)
    buddy_values = _successful_tpsu(buddy_runs)
    release_values = _successful_tpsu(release_runs)
    demo_median = statistics.median(demo_values) if demo_values else None
    greedy_median = statistics.median(greedy_values) if greedy_values else None
    buddy_median = statistics.median(buddy_values) if buddy_values else None
    release_median = statistics.median(release_values) if release_values else None
    release_first_values = [
        float(run["first_decode_tokens_per_second_per_user"])
        for run in release_runs
        if run.get("passed")
        and run.get("first_decode_tokens_per_second_per_user") is not None
    ]
    cv = {
        "official-demo-percent": _coefficient_of_variation_percent(demo_values),
        "official-greedy-percent": _coefficient_of_variation_percent(greedy_values),
        "buddy-greedy-percent": _coefficient_of_variation_percent(buddy_values),
        "official-release-percent": _coefficient_of_variation_percent(release_values),
    }
    report.update(
        {
            "official_demo_local_median_tpsu": demo_median,
            "official_greedy_local_median_tpsu": greedy_median,
            "official_local_median_tpsu": greedy_median,
            "official_local_primary_profile": "official-greedy",
            "buddy_local_median_tpsu": buddy_median,
            "official_release_median_tpsu": release_median,
            "official_release_first_decode_median_tpsu": (
                statistics.median(release_first_values)
                if release_first_values
                else None
            ),
            "buddy_ratio_of_local_official": (
                buddy_median / greedy_median
                if buddy_median is not None and greedy_median
                else None
            ),
            "buddy_ratio_of_release_official": (
                buddy_median / release_median
                if buddy_median is not None and release_median
                else None
            ),
            "release_ratio_of_external_reference": (
                release_median / OFFICIAL_EXTERNAL_REFERENCE_TPSU
                if release_median is not None
                else None
            ),
            "coefficient_of_variation": cv,
            "semantic_match": {
                "same_tt_metal_commit": report.get("same_tt_metal_commit"),
                "same_device": True,
                "same_model": True,
                "same_prompt_file": True,
                "same_batch_size": True,
                "same_cache_len": True,
                "same_page_block_size": True,
                "same_requested_prefill_len": True,
                "same_physical_prefill_bucket": False,
                "same_prompt_tokenization_mode": True,
                "decode_positions_follow_actual_prompt_lengths": True,
                "prefill_timing_comparable": False,
                "official_greedy_matches_buddy_sampling": True,
                "official_demo_matches_buddy_sampling": False,
                "official_trace_vs_buddy_eager": True,
                "decode_workload_comparable": True,
                "primary_official_profile": "official-greedy",
                "release_reference_same_tt_metal_commit": False,
                "release_reference_directly_comparable_to_buddy": False,
            },
            "comparison_contract": {
                "primary_parity_ratio": "buddy_ratio_of_local_official",
                "primary_reason": (
                    "Buddy and official-greedy use the same tt-metal commit"
                ),
                "release_reference_ratio": "buddy_ratio_of_release_official",
                "release_reference_reason": (
                    "validates the published release speed but crosses tt-metal versions"
                ),
            },
        }
    )

    expected = int(report["repetitions"])
    checks = [
        _check(
            "official-demo.successful-runs",
            len(demo_values) == expected,
            len(demo_values),
            expected,
        ),
        _check(
            "official-greedy.successful-runs",
            len(greedy_values) == expected,
            len(greedy_values),
            expected,
        ),
        _check(
            "buddy-greedy.successful-runs",
            len(buddy_values) == expected,
            len(buddy_values),
            expected,
        ),
        _check(
            "official-demo.cv-percent",
            cv["official-demo-percent"] is not None
            and cv["official-demo-percent"] <= 1.5,
            cv["official-demo-percent"],
            1.5,
        ),
        _check(
            "official-greedy.cv-percent",
            cv["official-greedy-percent"] is not None
            and cv["official-greedy-percent"] <= 1.5,
            cv["official-greedy-percent"],
            1.5,
        ),
        _check(
            "buddy-greedy.cv-percent",
            cv["buddy-greedy-percent"] is not None
            and cv["buddy-greedy-percent"] <= 1.5,
            cv["buddy-greedy-percent"],
            1.5,
        ),
        _check(
            "same-tt-metal-commit",
            bool(report.get("same_tt_metal_commit")),
            report.get("same_tt_metal_commit"),
            True,
        ),
        _check(
            "raw-samples",
            all(
                len(run["decode_step_ms_samples"]) == report["iterations"]
                for run in [*demo_runs, *greedy_runs, *buddy_runs]
            ),
            None,
            report["iterations"],
        ),
    ]
    if report.get("official_release") is not None:
        checks.extend(
            [
                _check(
                    "official-release.successful-runs",
                    len(release_values) == expected,
                    len(release_values),
                    expected,
                ),
                _check(
                    "official-release.cv-percent",
                    cv["official-release-percent"] is not None
                    and cv["official-release-percent"] <= 1.5,
                    cv["official-release-percent"],
                    1.5,
                ),
                _check(
                    "official-release.commit",
                    bool(
                        report["official_release"].get(
                            "matches_external_release_commit"
                        )
                    ),
                    report["official_release"].get("commit"),
                    OFFICIAL_EXTERNAL_RELEASE_COMMIT,
                ),
                _check(
                    "official-release.raw-samples",
                    all(
                        len(run["decode_step_ms_samples"]) == report["iterations"]
                        for run in release_runs
                    ),
                    None,
                    report["iterations"],
                ),
            ]
        )
    failed_checks = [check["name"] for check in checks if not check["passed"]]
    passed = not failed_checks
    report.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "acceptance": {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "checks": checks,
                "failed_checks": failed_checks,
            },
            "error": None,
        }
    )


def _successful_tpsu(runs: Sequence[dict[str, Any]]) -> list[float]:
    return [
        float(run["tokens_per_second_per_user"])
        for run in runs
        if run.get("passed") and run.get("tokens_per_second_per_user") is not None
    ]


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
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_prompt_values(path: Path, batch_size: int) -> list[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("input prompts JSON must be a list")
    prompts: list[str] = []
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
    lengths: list[int] = []
    for prompt in prompts:
        if instruct:
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            tokenized = tokenizer(prompt, add_special_tokens=True)
        token_ids = _input_ids(tokenized)
        lengths.append(len(token_ids))
    return lengths


def _input_ids(tokenized: Any) -> Sequence[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, Sequence)
        and tokenized
        and isinstance(tokenized[0], Sequence)
    ):
        tokenized = tokenized[0]
    if not isinstance(tokenized, Sequence) or isinstance(tokenized, (str, bytes)):
        raise ValueError("tokenizer did not return an input_ids sequence")
    return tokenized


def _effective_prefill_len(requested: int, max_prompt_tokens: int) -> int:
    required = max(requested, max_prompt_tokens)
    return 1 << (required - 1).bit_length()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cannot read git metadata for {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_diff_sha256(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cannot hash git diff for {root}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return hashlib.sha256(result.stdout).hexdigest()


def _run_command(
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def set_limits() -> None:
        if address_space_limit_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (address_space_limit_bytes, address_space_limit_bytes),
            )

    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            preexec_fn=set_limits,
        )
    return int(result.returncode)


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value))))
