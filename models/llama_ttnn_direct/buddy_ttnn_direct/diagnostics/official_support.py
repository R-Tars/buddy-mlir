from __future__ import annotations

import json
import linecache
import math
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
PYTEST_PLUGIN_ENABLED_ENV = "BUDDY_PARITY_PYTEST_PLUGIN"
PYTEST_PROFILE_ENV = "BUDDY_PARITY_OFFICIAL_PROFILE"
PYTEST_PAGE_PARAMS_ENV = "BUDDY_PARITY_PAGE_PARAMS"
PYTEST_GRAPH_CAPTURE_DIR_ENV = "BUDDY_PARITY_GRAPH_CAPTURE_DIR"
EXACT_SAMPLE_PREFIX = "BUDDY_PARITY_SAMPLE"
ACCURACY_SAMPLE_PREFIX = "BUDDY_ACCURACY_SAMPLE"
OFFICIAL_PYTEST_PLUGIN = "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity"
FORCE_ARGMAX_FIX_COMMIT = "0a0510e7fc197f599c9d5974a0f1f6776378db03"
_EXACT_SAMPLE_RE = re.compile(rf"{EXACT_SAMPLE_PREFIX} token_iteration=(?P<iteration>\d+) duration_ms=(?P<duration_ms>\d+(?:\.\d+)?)")
_ROUNDED_SAMPLE_RE = re.compile(r"Iteration (?P<iteration>\d+): (?P<duration_ms>\d+(?:\.\d+)?)ms @ (?P<tpsu>\d+(?:\.\d+)?) tok/s/user")
_ACCURACY_SAMPLE_RE = re.compile(rf"{ACCURACY_SAMPLE_PREFIX} token_iteration=(?P<iteration>\d+) predicted_token=(?P<predicted_token>\d+)")

def git_commit_ancestry(
    root: str | Path,
    ancestor: str = FORCE_ARGMAX_FIX_COMMIT,
    descendant: str = "HEAD",
) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False, capture_output=True, text=True,
    )
    return {
        "ancestor": ancestor,
        "descendant": descendant,
        "contains_ancestor": result.returncode == 0 if result.returncode in (0, 1) else None,
        "exit_status": int(result.returncode),
        "error": result.stderr.strip() or None,
    }

def measurement_provenance(*, repetitions: int, warmup: int, iterations: int) -> dict[str, Any]:
    formal = (int(repetitions), int(warmup), int(iterations)) == (3, 5, 100)
    return {
        "classification": "formal_product_performance" if formal else "harness_smoke",
        "measurement_contract": (
            f"{int(repetitions)} repetitions / {int(warmup)} warmup / "
            f"{int(iterations)} measured"
        ),
        "not_a_performance_baseline": not formal,
        "baseline_comparable": formal,
    }

def reference_provenance(
    *,
    buddy_sha: str,
    local_official_sha: str,
    current_main_sha: str | None,
    pinned_release_sha: str,
    pinned_release_tpsu: float,
    local_force_argmax_fix_present: bool | None,
    current_main_force_argmax_fix_present: bool | None,
) -> dict[str, Any]:
    same_runtime = bool(buddy_sha and buddy_sha == local_official_sha)
    pinned_release = bool(buddy_sha and buddy_sha == pinned_release_sha)
    current_main = bool(current_main_sha and local_official_sha == current_main_sha)
    return {
        "same_runtime_commit_comparable": same_runtime,
        "pinned_release_comparable": pinned_release,
        "current_main_comparable": current_main,
        "official_reference_provenance": {
            "same_runtime_commit": {
                "git_sha": local_official_sha,
                "contains_single_chip_force_argmax_fix": local_force_argmax_fix_present,
                "comparison_scope": "same-runtime-commit",
                "comparable_to_buddy": same_runtime,
                "reason": "same_sha" if same_runtime else "version_mismatch",
            },
            "pinned_release": {
                "git_sha": pinned_release_sha,
                "reference_tpsu": float(pinned_release_tpsu),
                "comparison_scope": "cross-version-release-reference",
                "comparable_to_buddy": pinned_release,
                "reason": "same_sha" if pinned_release else "version_mismatch",
            },
            "current_main": {
                "git_sha": current_main_sha,
                "contains_single_chip_force_argmax_fix": current_main_force_argmax_fix_present,
                "comparison_scope": "upstream-current-reference",
                "comparable_to_local_official": current_main,
                "reason": "same_sha" if current_main else (
                    "version_mismatch" if current_main_sha else "current_main_sha_unavailable"
                ),
            },
        },
    }

def pytest_configure(config: Any) -> None:
    if os.environ.get(PYTEST_PLUGIN_ENABLED_ENV) != "1":
        return
    config.option.page_params = json.loads(os.environ[PYTEST_PAGE_PARAMS_ENV])
    profile = os.environ.get(PYTEST_PROFILE_ENV)
    config.option.enable_trace = profile != "official-token-accuracy"
    config.option.stop_at_eos = 0
    if profile in {"official-greedy", "official-token-accuracy"}:
        config.option.sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}
    graph_capture_dir = os.environ.get(PYTEST_GRAPH_CAPTURE_DIR_ENV)
    if graph_capture_dir:
        _install_official_trace_graph_capture(Path(graph_capture_dir))

    from models.perf.benchmarking_utils import BenchmarkProfiler

    if getattr(BenchmarkProfiler, "_buddy_parity_patched", False):
        return
    original_end = BenchmarkProfiler.end

    def end_with_exact_sample(self: Any, step_name: str, iteration: int = 0) -> None:
        original_end(self, step_name, iteration)
        if step_name.startswith("inference_decode_time_"):
            token_iteration = int(step_name.rsplit("_", 1)[1])
            duration_ms = self.get_duration(step_name, iteration) * 1000.0
            print(f"{EXACT_SAMPLE_PREFIX} token_iteration={token_iteration} duration_ms={duration_ms:.6f}", flush=True)

    BenchmarkProfiler.end = end_with_exact_sample
    BenchmarkProfiler._buddy_parity_patched = True

def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if os.environ.get(PYTEST_PLUGIN_ENABLED_ENV) != "1" or os.environ.get(PYTEST_PROFILE_ENV) != "official-token-accuracy":
        return
    for item in items:
        token_accuracy = getattr(getattr(item, "module", None), "TokenAccuracy", None)
        if token_accuracy is None or getattr(token_accuracy, "_buddy_accuracy_patched", False):
            continue
        original_collect = token_accuracy.collect_predicted_tokens

        def collect_with_sample(self: Any, token: Any) -> Any:
            iteration = len(self.store_predicted_tokens)
            token_id = int(getattr(token, "item", lambda: token)())
            print(f"{ACCURACY_SAMPLE_PREFIX} token_iteration={iteration} predicted_token={token_id}", flush=True)
            return original_collect(self, token_id)

        token_accuracy.collect_predicted_tokens = collect_with_sample
        token_accuracy._buddy_accuracy_patched = True
        break

def parse_official_accuracy_samples(log_path: str | Path) -> list[int]:
    source = Path(log_path)
    if not source.is_file():
        return []
    indexed = {
        int(match.group("iteration")): int(match.group("predicted_token"))
        for match in _ACCURACY_SAMPLE_RE.finditer(source.read_text(errors="replace"))
    }
    return [indexed[index] for index in sorted(indexed)]

def parse_official_latency_report(log_path: str | Path, *, warmup: int, iterations: int) -> dict[str, Any]:
    source = Path(log_path)
    if not source.is_file():
        return failed_latency_report(f"official log was not written: {source}")
    text = source.read_text(errors="replace")
    indexed = {int(m.group("iteration")): float(m.group("duration_ms")) for m in _EXACT_SAMPLE_RE.finditer(text)}
    sample_source = "benchmark_profiler_exact"
    if not indexed:
        indexed = {int(m.group("iteration")): float(m.group("duration_ms"))
                   for m in _ROUNDED_SAMPLE_RE.finditer(text) if int(m.group("iteration")) > 0}
        sample_source = "official_debug_log_rounded_ms"
    ordered = [indexed[index] for index in sorted(indexed)]
    expected = warmup + iterations
    if len(ordered) < expected:
        return failed_latency_report(f"official profile produced {len(ordered)} non-compile samples; expected at least {expected}", sample_source=sample_source)
    return latency_report_from_samples(
        warmup_samples=ordered[:warmup],
        measured_samples=ordered[warmup:expected],
        sample_source=sample_source,
    )

def latency_report_from_samples(
    *,
    warmup_samples: Sequence[float],
    measured_samples: Sequence[float],
    sample_source: str,
) -> dict[str, Any]:
    warmups, samples = [float(v) for v in warmup_samples], [float(v) for v in measured_samples]
    valid = bool(samples) and all(math.isfinite(value) and value > 0 for value in samples)
    mean = statistics.fmean(samples) if valid else None
    return {
        "passed": valid,
        "warmup_step_ms_samples": warmups,
        "decode_step_ms_samples": samples,
        "decode_step_ms_mean": mean,
        "decode_step_ms_p50": statistics.median(samples) if valid else None,
        "tokens_per_second_per_user": 1000.0 / mean if mean else None,
        "sample_source": sample_source,
        "error": None if valid else "no valid measured latency samples",
    }

def failed_latency_report(message: str, *, sample_source: str = "not_available") -> dict[str, Any]:
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

def encode_page_params(cache_len: int, page_block_size: int) -> str:
    params = {"page_block_size": int(page_block_size),
              "page_max_num_blocks_per_dp": int(cache_len) * 32 // int(page_block_size)}
    return json.dumps(params, separators=(",", ":"))

def official_pytest_command(
    python: str | Path,
    selector: str,
    arguments: Sequence[str],
    *,
    junit_path: Path | None = None,
) -> list[str]:
    command = [str(python), "-m", "pytest", "-s", "-q",
               "models/tt_transformers/demo/simple_text_demo.py", "-k", selector,
               "-p", OFFICIAL_PYTEST_PLUGIN, *arguments]
    if junit_path is not None:
        command.extend(["--junitxml", str(junit_path)])
    return command

def official_source_environment(
    source_root: Path,
    model_root: Path,
    tensor_cache_path: Path,
    profile: str,
    cache_len: int,
    page_block_size: int,
    *,
    runtime_root: Path | None = None,
    preserve_pythonpath: bool = False,
    graph_capture_dir: Path | None = None,
    remove: Sequence[str] = (),
) -> dict[str, str]:
    environment = os.environ.copy()
    # Do not let a caller's pytest process settings inject another checkout's
    # plugins or options into the official comparison subprocess.
    for variable in ("PYTEST_PLUGINS", "PYTEST_ADDOPTS"):
        environment.pop(variable, None)
    paths = [str(path) for path in (
        Path(__file__).resolve().parents[4], source_root, source_root / "tools"
    )]
    if runtime_root is not None:
        paths.extend(str(runtime_root / name) for name in ("ttnn", "tt_eager"))
    if preserve_pythonpath and environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(paths), "HF_MODEL": str(model_root),
        "MESH_DEVICE": "P150", "TT_CACHE_PATH": str(tensor_cache_path),
        PYTEST_PLUGIN_ENABLED_ENV: "1", PYTEST_PROFILE_ENV: profile,
        PYTEST_PAGE_PARAMS_ENV: encode_page_params(cache_len, page_block_size),
    })
    if graph_capture_dir is not None:
        environment[PYTEST_GRAPH_CAPTURE_DIR_ENV] = str(graph_capture_dir)
    environment.pop("LLAMA_DIR", None)
    for variable in remove:
        environment.pop(variable, None)
    return environment

def _install_official_trace_graph_capture(output_dir: Path) -> None:
    import ttnn
    from ttnn.decorators import FastOperation

    if getattr(ttnn, "_buddy_graph_capture_patched", False):
        return
    original_begin, original_end = ttnn.begin_trace_capture, ttnn.end_trace_capture
    original_operation_call = FastOperation.__call__
    state: dict[str, Any] = {"active": None, "captures": []}
    output_dir.mkdir(parents=True, exist_ok=True)

    def call_with_trace_record(operation: Any, *args: Any, **kwargs: Any) -> Any:
        entry = state["active"]
        if entry is not None:
            entry["python_io_data"].append({
                "name": operation.python_fully_qualified_name,
                "arguments": {
                    "callsite": _trace_callsite(sys._getframe(1)),
                    "positional_types": [_trace_value(value) for value in args],
                    "keyword_types": {key: _trace_value(value)
                                      for key, value in sorted(kwargs.items())},
                },
                "input_tensor_ids": [], "output_tensor_ids": [],
            })
        return original_operation_call(operation, *args, **kwargs)

    FastOperation.__call__ = call_with_trace_record

    def begin_with_graph(device: Any, *args: Any, **kwargs: Any) -> Any:
        if state["active"] is not None:
            raise RuntimeError("nested official trace graph capture is unsupported")
        entry = {
            "index": len(state["captures"]),
            "program_cache_entries_before": _device_program_cache_entries(device),
            "python_io_data": [],
        }
        state["active"] = entry
        try:
            trace_id = original_begin(device, *args, **kwargs)
        except Exception:
            state["active"] = None
            raise
        entry["trace_id"] = str(trace_id)
        return trace_id

    def end_with_graph(device: Any, trace_id: Any, *args: Any, **kwargs: Any) -> Any:
        entry = state["active"]
        if entry is None:
            raise RuntimeError("official trace ended without graph capture state")
        try:
            result = original_end(device, trace_id, *args, **kwargs)
            python_io_data = entry.pop("python_io_data")
        finally:
            state["active"] = None

        graph_path = output_dir / f"trace_{entry['index']:03d}.json"
        sidecar_path = graph_path.with_suffix(".python_io.json")
        graph_path.write_text(json.dumps({
            "schema_version": 1,
            "source": "official-tt-metal-python-io-trace-capture",
            "capture_index": entry["index"],
            "trace_id": entry["trace_id"],
            "python_io_sidecar": sidecar_path.name,
        }, indent=2) + "\n")
        sidecar_path.write_text(json.dumps(python_io_data, default=str) + "\n")
        entry.update(graph_path=str(graph_path),
                     operation_count=len(python_io_data),
                     program_cache_entries_after=_device_program_cache_entries(device))
        state["captures"].append(entry)
        manifest = {"schema_version": 1, "captures": state["captures"]}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        return result

    ttnn.begin_trace_capture = begin_with_graph
    ttnn.end_trace_capture = end_with_graph
    ttnn._buddy_graph_capture_patched = True

def _trace_callsite(frame: Any) -> str:
    filename, lineno = frame.f_code.co_filename, frame.f_lineno
    return (f"{Path(filename).name}:{frame.f_code.co_name}:{lineno}:"
            f"{linecache.getline(filename, lineno).strip()}")

def _trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): _trace_value(item) for key, item in list(value.items())[:8]}
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"

def _device_program_cache_entries(device: Any) -> int | None:
    count = getattr(device, "num_program_cache_entries", None)
    try:
        return int(count()) if callable(count) else None
    except (RuntimeError, TypeError, ValueError):
        return None
