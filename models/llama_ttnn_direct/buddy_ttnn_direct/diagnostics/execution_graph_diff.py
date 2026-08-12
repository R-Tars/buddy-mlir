from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from ..runtime.reports import write_report
from ..runtime.trace import GRAPH_CAPTURE_PATH_ENV
from .official_support import (
    DEFAULT_PROMPTS,
    official_pytest_command,
    official_source_environment,
)
from .process_support import (
    absolute_path,
    git_head,
    require_path,
    run_logged_command,
)

FOCUS_OPERATIONS = (
    "to_layout", "to_memory_config", "reshape", "slice", "typecast", "concat",
    "untilize", "argmax", "qkv_linear", "paged_cache_update", "sdpa",
    "concat_heads", "o_projection", "rms_norm", "mlp",
)

CommandRunner = Callable[
    [Sequence[str], Path, dict[str, str], Path, float | None, int | None],
    int,
]

def run_execution_graph_diff(
    *,
    out: str | Path,
    buddy_program: str | Path,
    official_tt_metal_root: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    input_prompts: str | Path | None = None,
    batch_size: int = 32,
    prefill_len: int = 256,
    cache_len: int = 1024,
    page_block_size: int = 32,
    device: str = "p150a",
    device_id: int = 0,
    official_python: str | Path | None = None,
    timeout_seconds: float | None = 3600.0,
    address_space_limit_bytes: int | None = 95_000_000_000,
    dry_run: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    report_path, program_root = Path(out).resolve(), Path(buddy_program).resolve()
    official_root, model_root = Path(official_tt_metal_root).resolve(), Path(model_path).resolve()
    tokenizer_root = Path(tokenizer_path or model_root).resolve()
    prompts_path = Path(input_prompts or official_root / DEFAULT_PROMPTS).resolve()
    official_python_path = absolute_path(official_python or sys.executable)
    runs_root = report_path.parent / f"{report_path.stem}_runs"
    config_path = program_root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    layer_count = int(config.get("num_layers", 32)) if isinstance(config, dict) else 32
    runner = command_runner or run_logged_command
    if not dry_run:
        for path, description in (
            (program_root, "Buddy program directory"),
            (official_root, "official tt-metal root"),
            (model_root, "model path"),
            (prompts_path, "input prompts"),
            (official_python_path, "official Python"),
        ):
            require_path(path, description)
        staged_prompts_path = runs_root / "inputs" / "input_prompts.json"
        staged_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prompts_path, staged_prompts_path)
        execution_prompts_path = staged_prompts_path
    else:
        execution_prompts_path = prompts_path
    plans = _planned_runs(
        runs_root=runs_root, program_root=program_root, official_root=official_root,
        model_root=model_root, tokenizer_root=tokenizer_root,
        prompts_path=execution_prompts_path,
        official_python=official_python_path, layer_count=layer_count,
        batch_size=batch_size, prefill_len=prefill_len, cache_len=cache_len,
        page_block_size=page_block_size, device=device, device_id=device_id,
    )
    report: dict[str, Any] = {
        "schema_version": 1, "command": "diagnose", "stage": "execution-graph-diff",
        "status": "running", "passed": False,
        "comparison_scope": "full_decode_trace_compile_capture",
        "buddy_program": str(program_root), "official_tt_metal_root": str(official_root),
        "official_commit": git_head(official_root), "official_python": str(official_python_path),
        "model_path": str(model_root), "tokenizer_path": str(tokenizer_root),
        "input_prompts": str(prompts_path),
        "settings": {
            "layers": layer_count, "batch_size": int(batch_size),
            "prefill_len": int(prefill_len), "cache_len": int(cache_len),
            "page_block_size": int(page_block_size), "device": device,
            "device_id": int(device_id), "execution_mode": "trace",
            "runtime_input_mode": "persistent",
        },
        "focus_operations": list(FOCUS_OPERATIONS),
        "planned_runs": [{key: value for key, value in plan.items() if key != "environment"}
                         for plan in plans],
        "runs": [],
    }
    write_report(report_path, report)
    if dry_run:
        report.update({"status": "dry_run", "passed": True, "dry_run": True})
        write_report(report_path, report)
        return report

    try:
        for plan in plans:
            run = _execute_run(
                plan, runner=runner, timeout_seconds=timeout_seconds,
                address_space_limit_bytes=address_space_limit_bytes,
            )
            report["runs"].append(run)
            write_report(report_path, report)
            if not run["passed"]:
                raise RuntimeError(f"{run['implementation']} graph capture failed; see {run['log_path']}")

        by_implementation = {run["implementation"]: run for run in report["runs"]}
        captures = {
            name: _select_decode_capture([Path(path) for path in run["graph_paths"]])
            for name, run in by_implementation.items()
        }
        report.update({
            "status": "passed", "passed": True,
            "selected_graphs": {name: capture["path"] for name, capture in captures.items()},
            "compile_cache": {
                "buddy": _buddy_compile_cache(by_implementation["buddy"]),
                "official": _official_compile_cache(by_implementation["official"],
                                                     Path(captures["official"]["path"])),
            },
            "execution_graph_diff": compare_execution_graphs(captures["buddy"], captures["official"]),
        })
    except Exception as error:
        report.update(status="failed", passed=False,
                      error={"type": type(error).__name__, "message": str(error)})
    write_report(report_path, report)
    return report

def compare_execution_graphs(
    buddy: Mapping[str, Any],
    official: Mapping[str, Any],
) -> dict[str, Any]:
    buddy_ops, official_ops = list(buddy["operations"]), list(official["operations"])
    buddy_counts = Counter(op["canonical_name"] for op in buddy_ops)
    official_counts = Counter(op["canonical_name"] for op in official_ops)
    focus_diff = [{
        "operation": name, "buddy_count": int(buddy_counts.get(name, 0)),
        "official_count": int(official_counts.get(name, 0)),
        "count_delta": int(buddy_counts.get(name, 0)) - int(official_counts.get(name, 0)),
        "buddy_samples": [dict(op) for op in buddy_ops if op["canonical_name"] == name][:3],
        "official_samples": [dict(op) for op in official_ops if op["canonical_name"] == name][:3],
    } for name in FOCUS_OPERATIONS]
    return {
        "buddy_operation_count": len(buddy_ops),
        "official_operation_count": len(official_ops),
        "buddy_capture_format": buddy["format"],
        "official_capture_format": official["format"],
        "focus_diff": focus_diff,
        "buddy_operation_counts": dict(sorted(buddy_counts.items())),
        "official_operation_counts": dict(sorted(official_counts.items())),
        "buddy_operations": buddy_ops,
        "official_operations": official_ops,
    }

def load_execution_graph(path: str | Path) -> dict[str, Any]:
    graph_path = Path(path).resolve()
    payload = json.loads(graph_path.read_text())
    sidecar = graph_path.with_suffix(".python_io.json")
    if sidecar.is_file():
        operations = _normalize_python_io(json.loads(sidecar.read_text()))
        capture_format = "ttnn-python-io-sidecar"
    else:
        raw_graph = payload.get("raw_graph") if isinstance(payload, dict) and "raw_graph" in payload else payload
        operations = _normalize_raw_graph(raw_graph)
        capture_format = "ttnn-raw-call-graph"
        if not operations and isinstance(payload, dict):
            operations = _normalize_serialized_graph(payload.get("serialized_graph"))
            capture_format = "ttnn-serialized-graph"
    if not operations:
        raise ValueError(f"execution graph has no operations: {graph_path}")
    return {"path": str(graph_path), "format": capture_format,
            "operations": operations, "decode_score": _decode_score(operations)}

def _normalize_python_io(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    return [_operation_record(name=str(record["name"]),
                              arguments=record.get("arguments", {}),
                              input_tensor_ids=record.get("input_tensor_ids"),
                              output_tensor_ids=record.get("output_tensor_ids"))
            for record in records if isinstance(record, dict) and record.get("name")
            and not _is_trace_control(str(record["name"]))]

def _normalize_raw_graph(raw_graph: Any) -> list[dict[str, Any]]:
    nodes = raw_graph
    if isinstance(raw_graph, dict):
        nodes = raw_graph.get("content") or raw_graph.get("nodes") or []
    if not isinstance(nodes, list):
        return []
    operations = []
    depth = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("node_type")
        if node_type == "function_end":
            depth = max(0, depth - 1)
            continue
        if node_type != "function_start":
            continue
        params = node.get("params") or {}
        name = str(params.get("name") or node.get("operation") or "")
        if depth == 0 and name and not _is_trace_control(name):
            operations.append(_operation_record(
                name=name, arguments=node.get("arguments", params.get("arguments", []))))
        depth += 1
    return operations

def _normalize_serialized_graph(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    return [_operation_record(name=str(item["operation"]), arguments=item.get("arguments", []))
            for item in content if isinstance(item, dict) and item.get("operation")
            and not _is_trace_control(str(item["operation"]))]

def _operation_record(
    *,
    name: str,
    arguments: Any,
    input_tensor_ids: Any = None,
    output_tensor_ids: Any = None,
) -> dict[str, Any]:
    return {"name": name, "canonical_name": _canonical_operation(name, arguments),
            "input_tensor_ids": _int_list(input_tensor_ids),
            "output_tensor_ids": _int_list(output_tensor_ids),
            "metadata": _extract_metadata(arguments)}

def _canonical_operation(name: str, arguments: Any) -> str:
    lowered = f"{name} {_compact_json(arguments)}".lower()
    ordered_aliases = (
        ("to_memory_config", ("to_memory_config",)),
        ("to_layout", ("to_layout",)),
        ("paged_cache_update", ("paged_update_cache", "paged_cache_update")),
        ("concat_heads", ("concat_heads", "nlp_concat_heads")),
        ("qkv_linear", ("qkv", "wqkv", "query_key_value")),
        ("o_projection", ("o_proj", "wo_linear", "output_projection")),
        ("rms_norm", ("rms_norm", "rmsnorm")),
        ("sdpa", ("scaled_dot_product_attention", "sdpa")),
        ("argmax", ("argmax",)),
        ("untilize", ("untilize", "to_row_major")),
        ("typecast", ("typecast",)),
        ("reshape", ("reshape",)),
        ("slice", ("slice",)),
        ("concat", ("concat",)),
        ("mlp", ("mlp", "feed_forward", "ffn")),
    )
    for canonical, aliases in ordered_aliases:
        if any(alias in lowered for alias in aliases):
            return canonical
    return name.rsplit("::", 1)[-1].rsplit(".", 1)[-1].lower()

def _extract_metadata(arguments: Any) -> dict[str, Any]:
    text = _compact_json(arguments)
    shapes = set()
    for match in re.finditer(r"(?:logical_)?shape\s*[=:]\s*(?:Shape\()?\[([^\]]+)\]",
                             text, re.IGNORECASE):
        values = [value.strip() for value in match.group(1).split(",")]
        if values and all(re.fullmatch(r"-?\d+", value) for value in values):
            shapes.add(tuple(int(value) for value in values))
    dtypes = sorted({(match.group(1) or match.group(2)).upper() for match in re.finditer(
        r"DataType::([A-Z0-9_]+)|ttnn\.(bfloat16|float32|uint32|int32)", text, re.IGNORECASE)})
    layouts = sorted(set(re.findall(r"(?:Layout::|layout[=:]\s*)(TILE|ROW_MAJOR)",
                                    text, re.IGNORECASE)))
    grids = sorted(set(re.findall(
        r"(?:CoreGrid|grid_size|compute_with_storage_grid_size)[^,}\]]{0,120}",
        text, re.IGNORECASE)))
    return {
        "shapes": [list(shape) for shape in sorted(shapes)],
        "dtypes": dtypes,
        "layouts": [layout.upper() for layout in layouts],
        "memory_configs": _matching_snippets(text, "memory_config"),
        "program_configs": _matching_snippets(text, "program_config"),
        "grids": grids[:8],
    }

def _planned_runs(
    *,
    runs_root: Path,
    program_root: Path,
    official_root: Path,
    model_root: Path,
    tokenizer_root: Path,
    prompts_path: Path,
    official_python: Path,
    layer_count: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    page_block_size: int,
    device: str,
    device_id: int,
) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[4]
    official_run = runs_root / "official"
    official_graphs = official_run / "graphs"
    official_command = official_pytest_command(official_python,
        "performance-batch-32 and not log-probs", [
            "--input_prompts", str(prompts_path), "--instruct", "1",
            "--repeat_batches", "1", "--max_seq_len", str(cache_len),
            "--batch_size", str(batch_size), "--max_generated_tokens", "3",
            "--stop_at_eos", "0", "--enable_trace",
        ])
    official_env = official_source_environment(official_root, model_root=model_root,
        tensor_cache_path=runs_root / "official_tensor_cache", profile="official-release-demo",
        cache_len=cache_len, page_block_size=page_block_size,
        runtime_root=official_root,
        graph_capture_dir=official_graphs,
        remove=("CONDA_PREFIX", "LD_LIBRARY_PATH", "TT_METAL_BUILD_HOME", "TT_METAL_HOME"))

    buddy_run = runs_root / "buddy"
    buddy_graph = buddy_run / "graphs" / "decode_trace.json"
    buddy_profile = buddy_run / "profile.json"
    buddy_command = [
        sys.executable, "-m", "models.llama_ttnn_direct.buddy_ttnn_direct.cli", "profile",
        "--mode", "decode-steady", "--program-dir", str(program_root),
        "--model-path", str(model_root), "--tokenizer-path", str(tokenizer_root),
        "--input-prompts", str(prompts_path), "--instruct", "--layers", str(layer_count),
        "--prefill-len", str(prefill_len), "--batch-size", str(batch_size),
        "--cache-len", str(cache_len), "--warmup", "1", "--iterations", "1",
        "--after-prefill", "--runtime-input-mode", "persistent", "--execution-mode", "trace",
        "--device", device, "--device-id", str(device_id), "--dtype-seed", "bf16",
        "--out", str(buddy_profile),
    ]
    buddy_env = _buddy_source_environment(repo_root, official_root, buddy_graph)
    return [
        {
            "implementation": "official", "command": official_command,
            "cwd": str(official_root), "environment": official_env,
            "run_dir": str(official_run), "log_path": str(official_run / "run.log"),
            "graph_root": str(official_graphs),
        },
        {
            "implementation": "buddy", "command": buddy_command,
            "cwd": str(repo_root), "environment": buddy_env,
            "run_dir": str(buddy_run), "log_path": str(buddy_run / "run.log"),
            "graph_root": str(buddy_graph.parent), "profile_path": str(buddy_profile),
        },
    ]

def _buddy_source_environment(
    repo_root: Path,
    tt_metal_root: Path,
    graph_path: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [repo_root / "build-ttmlir/python_packages",
                    repo_root / "build-ttmlir/runtime/python", tt_metal_root,
                    tt_metal_root / "ttnn", tt_metal_root / "tt_eager", repo_root]
    library_paths = [Path(sys.base_prefix) / "lib", tt_metal_root / "build/lib",
                     repo_root / "build-ttmlir/lib"]
    if environment.get("PYTHONPATH"):
        python_paths.extend(Path(path) for path in environment["PYTHONPATH"].split(os.pathsep))
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.extend(Path(path) for path in environment["LD_LIBRARY_PATH"].split(os.pathsep))
    environment.update({
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(map(str, python_paths))),
        "LD_LIBRARY_PATH": os.pathsep.join(dict.fromkeys(map(str, library_paths))),
        "TT_METAL_HOME": str(tt_metal_root),
        "TT_METAL_RUNTIME_ROOT": str(tt_metal_root),
        "TT_METAL_BUILD_HOME": str(tt_metal_root / "build"),
        GRAPH_CAPTURE_PATH_ENV: str(graph_path),
    })
    return environment

def _execute_run(
    plan: Mapping[str, Any],
    *,
    runner: CommandRunner,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> dict[str, Any]:
    run_dir, log_path = Path(plan["run_dir"]), Path(plan["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_root = Path(plan["graph_root"])
    graph_root.mkdir(parents=True, exist_ok=True)
    for stale_graph in graph_root.glob("*.json"):
        stale_graph.unlink()
    profile_path = plan.get("profile_path")
    if profile_path and Path(profile_path).is_file():
        Path(profile_path).unlink()
    started = time.time()
    return_code = runner(plan["command"], Path(plan["cwd"]), dict(plan["environment"]),
                         log_path, timeout_seconds, address_space_limit_bytes)
    graph_paths = sorted(
        str(path.resolve())
        for path in graph_root.glob("*.json")
        if path.name != "manifest.json" and not path.name.endswith(".python_io.json")
    )
    profile = (json.loads(Path(profile_path).read_text())
               if profile_path and Path(profile_path).is_file() else None)
    passed = return_code == 0 and bool(graph_paths)
    return {
        "implementation": plan["implementation"], "status": "passed" if passed else "failed",
        "passed": passed, "return_code": int(return_code),
        "elapsed_seconds": time.time() - started, "command": list(plan["command"]),
        "cwd": plan["cwd"], "log_path": str(log_path),
        "graph_paths": graph_paths, "profile": profile,
    }

def _select_decode_capture(paths: Sequence[Path]) -> dict[str, Any]:
    captures = [load_execution_graph(path) for path in paths]
    if not captures:
        raise ValueError("no execution graph captures were produced")
    return max(captures, key=lambda capture: (int(capture["decode_score"]),
                                               len(capture["operations"])))

def _decode_score(operations: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter(op["canonical_name"] for op in operations)
    return sum(weight * counts.get(name, 0) for name, weight in (
        ("paged_cache_update", 100), ("sdpa", 100), ("concat_heads", 50),
        ("argmax", 10))) + len(operations)

def _buddy_compile_cache(run: Mapping[str, Any]) -> dict[str, Any]:
    profile = run.get("profile") or {}
    runtime_inputs = profile.get("runtime_inputs") or {}
    keys = (
        "program_compile_count_after_capture",
        "program_compile_count_during_capture",
        "program_cache_entries_before_compile",
        "program_cache_entries_after_compile",
        "program_cache_entries_after_capture",
        "program_cache_entries_after_execute",
    )
    return {key: profile.get(key) if profile.get(key) is not None else runtime_inputs.get(key)
            for key in keys}

def _official_compile_cache(run: Mapping[str, Any], selected_graph: Path) -> dict[str, Any]:
    manifest_path = Path(run["graph_paths"][0]).parent / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "unavailable"}
    manifest = json.loads(manifest_path.read_text())
    for capture in manifest.get("captures", []):
        if Path(capture.get("graph_path", "")).resolve() == selected_graph.resolve():
            before = capture.get("program_cache_entries_before")
            after = capture.get("program_cache_entries_after")
            return {
                "status": "observed",
                "program_cache_entries_before": before,
                "program_cache_entries_after": after,
                "compile_cache_misses_during_capture": (max(0, int(after) - int(before))
                    if before is not None and after is not None else None),
            }
    return {"status": "selected_capture_missing_from_manifest"}

def _matching_snippets(text: str, needle: str) -> list[str]:
    lowered, snippets, cursor = text.lower(), [], 0
    while len(snippets) < 8:
        index = lowered.find(needle, cursor)
        if index < 0:
            break
        snippets.append(text[index : index + 320])
        cursor = index + len(needle)
    return sorted(set(snippets))

def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)

def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result

def _is_trace_control(name: str) -> bool:
    controls = ("begin_trace_capture", "end_trace_capture", "execute_trace", "release_trace")
    return any(control in name.lower() for control in controls)
