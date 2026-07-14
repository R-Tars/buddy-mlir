from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from ..runtime.reports import write_report
from ..runtime.trace import GRAPH_CAPTURE_PATH_ENV
from .benchmark_parity import (
    DEFAULT_PROMPTS,
    PYTEST_GRAPH_CAPTURE_DIR_ENV,
    PYTEST_PAGE_PARAMS_ENV,
    PYTEST_PLUGIN_ENABLED_ENV,
    PYTEST_PROFILE_ENV,
)

FOCUS_OPERATIONS = (
    "to_layout",
    "to_memory_config",
    "reshape",
    "slice",
    "typecast",
    "concat",
    "untilize",
    "argmax",
    "qkv_linear",
    "paged_cache_update",
    "sdpa",
    "concat_heads",
    "o_projection",
    "rms_norm",
    "mlp",
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
    """Capture and compare one full Buddy and official decode trace graph."""

    report_path = Path(out).resolve()
    program_root = Path(buddy_program).resolve()
    official_root = Path(official_tt_metal_root).resolve()
    model_root = Path(model_path).resolve()
    tokenizer_root = Path(tokenizer_path or model_root).resolve()
    prompts_path = Path(input_prompts or official_root / DEFAULT_PROMPTS).resolve()
    official_python_path = Path(
        os.path.abspath(os.path.expanduser(str(official_python or sys.executable)))
    )
    runs_root = report_path.parent / f"{report_path.stem}_runs"
    config = _read_program_config(program_root)
    layer_count = int(config.get("num_layers", 32))
    runner = command_runner or _run_command
    plans = _planned_runs(
        runs_root=runs_root,
        program_root=program_root,
        official_root=official_root,
        model_root=model_root,
        tokenizer_root=tokenizer_root,
        prompts_path=prompts_path,
        official_python=official_python_path,
        layer_count=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        page_block_size=page_block_size,
        device=device,
        device_id=device_id,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "diagnose",
        "stage": "execution-graph-diff",
        "status": "running",
        "passed": False,
        "comparison_scope": "full_decode_trace_compile_capture",
        "buddy_program": str(program_root),
        "official_tt_metal_root": str(official_root),
        "official_commit": _git_head(official_root),
        "official_python": str(official_python_path),
        "model_path": str(model_root),
        "tokenizer_path": str(tokenizer_root),
        "input_prompts": str(prompts_path),
        "settings": {
            "layers": layer_count,
            "batch_size": int(batch_size),
            "prefill_len": int(prefill_len),
            "cache_len": int(cache_len),
            "page_block_size": int(page_block_size),
            "device": device,
            "device_id": int(device_id),
            "execution_mode": "trace",
            "runtime_input_mode": "persistent",
        },
        "focus_operations": list(FOCUS_OPERATIONS),
        "planned_runs": [_public_plan(plan) for plan in plans],
        "runs": [],
    }
    write_report(report_path, report)
    if dry_run:
        report.update(
            {
                "status": "dry_run",
                "passed": True,
                "dry_run": True,
            }
        )
        write_report(report_path, report)
        return report

    _require_path(program_root, "Buddy program directory")
    _require_path(official_root, "official tt-metal root")
    _require_path(model_root, "model path")
    _require_path(prompts_path, "input prompts")
    _require_path(official_python_path, "official Python")

    try:
        for plan in plans:
            run = _execute_run(
                plan,
                runner=runner,
                timeout_seconds=timeout_seconds,
                address_space_limit_bytes=address_space_limit_bytes,
            )
            report["runs"].append(run)
            write_report(report_path, report)
            if not run["passed"]:
                raise RuntimeError(
                    f"{run['implementation']} graph capture failed; "
                    f"see {run['log_path']}"
                )

        official_run = next(
            run for run in report["runs"] if run["implementation"] == "official"
        )
        buddy_run = next(
            run for run in report["runs"] if run["implementation"] == "buddy"
        )
        official_capture = _select_decode_capture(
            [Path(path) for path in official_run["graph_paths"]]
        )
        buddy_capture = _select_decode_capture(
            [Path(path) for path in buddy_run["graph_paths"]]
        )
        comparison = compare_execution_graphs(
            buddy_capture,
            official_capture,
        )
        report.update(
            {
                "status": "passed",
                "passed": True,
                "selected_graphs": {
                    "buddy": buddy_capture["path"],
                    "official": official_capture["path"],
                },
                "compile_cache": {
                    "buddy": _buddy_compile_cache(buddy_run),
                    "official": _official_compile_cache(
                        official_run,
                        Path(official_capture["path"]),
                    ),
                },
                "execution_graph_diff": comparison,
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
    write_report(report_path, report)
    return report


def compare_execution_graphs(
    buddy: Mapping[str, Any],
    official: Mapping[str, Any],
) -> dict[str, Any]:
    buddy_ops = list(buddy["operations"])
    official_ops = list(official["operations"])
    buddy_counts = Counter(op["canonical_name"] for op in buddy_ops)
    official_counts = Counter(op["canonical_name"] for op in official_ops)
    focus_diff = []
    for name in FOCUS_OPERATIONS:
        buddy_count = int(buddy_counts.get(name, 0))
        official_count = int(official_counts.get(name, 0))
        focus_diff.append(
            {
                "operation": name,
                "buddy_count": buddy_count,
                "official_count": official_count,
                "count_delta": buddy_count - official_count,
                "buddy_samples": _samples_for(buddy_ops, name),
                "official_samples": _samples_for(official_ops, name),
            }
        )
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
        raw_graph = (
            payload.get("raw_graph")
            if isinstance(payload, dict) and "raw_graph" in payload
            else payload
        )
        operations = _normalize_raw_graph(raw_graph)
        capture_format = "ttnn-raw-call-graph"
        if not operations and isinstance(payload, dict):
            operations = _normalize_serialized_graph(payload.get("serialized_graph"))
            capture_format = "ttnn-serialized-graph"
    if not operations:
        raise ValueError(f"execution graph has no operations: {graph_path}")
    return {
        "path": str(graph_path),
        "format": capture_format,
        "operations": operations,
        "decode_score": _decode_score(operations),
    }


def _normalize_python_io(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    operations = []
    for record in records:
        if not isinstance(record, dict) or not record.get("name"):
            continue
        name = str(record["name"])
        if _is_trace_control(name):
            continue
        arguments = record.get("arguments", {})
        operations.append(
            _operation_record(
                name=name,
                arguments=arguments,
                input_tensor_ids=record.get("input_tensor_ids"),
                output_tensor_ids=record.get("output_tensor_ids"),
            )
        )
    return operations


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
            operations.append(
                _operation_record(
                    name=name,
                    arguments=node.get("arguments", params.get("arguments", [])),
                )
            )
        depth += 1
    return operations


def _normalize_serialized_graph(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    operations = []
    for item in content:
        if not isinstance(item, dict) or not item.get("operation"):
            continue
        name = str(item["operation"])
        if _is_trace_control(name):
            continue
        operations.append(
            _operation_record(name=name, arguments=item.get("arguments", []))
        )
    return operations


def _operation_record(
    *,
    name: str,
    arguments: Any,
    input_tensor_ids: Any = None,
    output_tensor_ids: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "canonical_name": _canonical_operation(name, arguments),
        "input_tensor_ids": _int_list(input_tensor_ids),
        "output_tensor_ids": _int_list(output_tensor_ids),
        "metadata": _extract_metadata(arguments),
    }


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
    simple = name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return simple.lower()


def _extract_metadata(arguments: Any) -> dict[str, Any]:
    text = _compact_json(arguments)
    shapes = set()
    for match in re.finditer(
        r"(?:logical_)?shape\s*[=:]\s*(?:Shape\()?\[([^\]]+)\]",
        text,
        re.IGNORECASE,
    ):
        values = [value.strip() for value in match.group(1).split(",")]
        if values and all(re.fullmatch(r"-?\d+", value) for value in values):
            shapes.add(tuple(int(value) for value in values))
    dtypes = sorted(
        {
            (match.group(1) or match.group(2)).upper()
            for match in re.finditer(
                r"DataType::([A-Z0-9_]+)|ttnn\.(bfloat16|float32|uint32|int32)",
                text,
                re.IGNORECASE,
            )
        }
    )
    layouts = sorted(
        set(
            re.findall(
                r"(?:Layout::|layout[=:]\s*)(TILE|ROW_MAJOR)",
                text,
                re.IGNORECASE,
            )
        )
    )
    memory_configs = _matching_snippets(text, "memory_config")
    program_configs = _matching_snippets(text, "program_config")
    grids = sorted(
        set(
            re.findall(
                r"(?:CoreGrid|grid_size|compute_with_storage_grid_size)[^,}\]]{0,120}",
                text,
                re.IGNORECASE,
            )
        )
    )
    return {
        "shapes": [list(shape) for shape in sorted(shapes)],
        "dtypes": dtypes,
        "layouts": [layout.upper() for layout in layouts],
        "memory_configs": memory_configs,
        "program_configs": program_configs,
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
    plugin = (
        "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics." "benchmark_parity"
    )
    official_run = runs_root / "official"
    official_graphs = official_run / "graphs"
    official_command = [
        str(official_python),
        "-m",
        "pytest",
        "-s",
        "-q",
        "models/tt_transformers/demo/simple_text_demo.py",
        "-k",
        "performance-batch-32 and not log-probs",
        "-p",
        plugin,
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
        "3",
        "--stop_at_eos",
        "0",
        "--enable_trace",
        "1",
    ]
    official_env = os.environ.copy()
    official_env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root), str(official_root), str(official_root / "tools")]
            ),
            "HF_MODEL": str(model_root),
            "MESH_DEVICE": "P150",
            "TT_CACHE_PATH": str(runs_root / "official_tensor_cache"),
            PYTEST_PLUGIN_ENABLED_ENV: "1",
            PYTEST_PROFILE_ENV: "official-release-demo",
            PYTEST_PAGE_PARAMS_ENV: json.dumps(
                {
                    "page_block_size": int(page_block_size),
                    "page_max_num_blocks_per_dp": (
                        int(cache_len) * 32 // int(page_block_size)
                    ),
                },
                separators=(",", ":"),
            ),
            PYTEST_GRAPH_CAPTURE_DIR_ENV: str(official_graphs),
        }
    )
    official_env.pop("LLAMA_DIR", None)
    for variable in (
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "TT_METAL_BUILD_HOME",
        "TT_METAL_HOME",
    ):
        official_env.pop(variable, None)

    buddy_run = runs_root / "buddy"
    buddy_graph = buddy_run / "graphs" / "decode_trace.json"
    buddy_profile = buddy_run / "profile.json"
    buddy_command = [
        sys.executable,
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
        str(prefill_len),
        "--batch-size",
        str(batch_size),
        "--cache-len",
        str(cache_len),
        "--warmup",
        "1",
        "--iterations",
        "1",
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
        str(buddy_profile),
    ]
    buddy_env = os.environ.copy()
    buddy_env[GRAPH_CAPTURE_PATH_ENV] = str(buddy_graph)
    return [
        {
            "implementation": "official",
            "command": official_command,
            "cwd": str(official_root),
            "environment": official_env,
            "run_dir": str(official_run),
            "log_path": str(official_run / "run.log"),
            "graph_root": str(official_graphs),
        },
        {
            "implementation": "buddy",
            "command": buddy_command,
            "cwd": str(repo_root),
            "environment": buddy_env,
            "run_dir": str(buddy_run),
            "log_path": str(buddy_run / "run.log"),
            "graph_root": str(buddy_graph.parent),
            "profile_path": str(buddy_profile),
        },
    ]


def _execute_run(
    plan: Mapping[str, Any],
    *,
    runner: CommandRunner,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> dict[str, Any]:
    run_dir = Path(plan["run_dir"])
    log_path = Path(plan["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_root = Path(plan["graph_root"])
    graph_root.mkdir(parents=True, exist_ok=True)
    for stale_graph in graph_root.glob("*.json"):
        stale_graph.unlink()
    profile_path = plan.get("profile_path")
    if profile_path and Path(profile_path).is_file():
        Path(profile_path).unlink()
    started = time.time()
    return_code = runner(
        plan["command"],
        Path(plan["cwd"]),
        dict(plan["environment"]),
        log_path,
        timeout_seconds,
        address_space_limit_bytes,
    )
    graph_paths = sorted(
        str(path.resolve())
        for path in graph_root.glob("*.json")
        if path.name != "manifest.json" and not path.name.endswith(".python_io.json")
    )
    profile = None
    if profile_path and Path(profile_path).is_file():
        profile = json.loads(Path(profile_path).read_text())
    return {
        "implementation": plan["implementation"],
        "status": "passed" if return_code == 0 and graph_paths else "failed",
        "passed": return_code == 0 and bool(graph_paths),
        "return_code": int(return_code),
        "elapsed_seconds": time.time() - started,
        "command": list(plan["command"]),
        "cwd": plan["cwd"],
        "log_path": str(log_path),
        "graph_paths": graph_paths,
        "profile": profile,
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "environment"}


def _select_decode_capture(paths: Sequence[Path]) -> dict[str, Any]:
    captures = [load_execution_graph(path) for path in paths]
    if not captures:
        raise ValueError("no execution graph captures were produced")
    return max(
        captures,
        key=lambda capture: (
            int(capture["decode_score"]),
            len(capture["operations"]),
        ),
    )


def _decode_score(operations: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter(op["canonical_name"] for op in operations)
    return (
        100 * counts.get("paged_cache_update", 0)
        + 100 * counts.get("sdpa", 0)
        + 50 * counts.get("concat_heads", 0)
        + 10 * counts.get("argmax", 0)
        + len(operations)
    )


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
    return {
        key: (
            profile.get(key)
            if profile.get(key) is not None
            else runtime_inputs.get(key)
        )
        for key in keys
    }


def _official_compile_cache(
    run: Mapping[str, Any], selected_graph: Path
) -> dict[str, Any]:
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
                "compile_cache_misses_during_capture": (
                    max(0, int(after) - int(before))
                    if before is not None and after is not None
                    else None
                ),
            }
    return {"status": "selected_capture_missing_from_manifest"}


def _samples_for(
    operations: Sequence[Mapping[str, Any]], name: str
) -> list[dict[str, Any]]:
    samples = [op for op in operations if op["canonical_name"] == name]
    return [dict(sample) for sample in samples[:3]]


def _read_program_config(program_root: Path) -> dict[str, Any]:
    path = program_root / "config.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


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


def _matching_snippets(text: str, needle: str) -> list[str]:
    lowered = text.lower()
    snippets = []
    cursor = 0
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
    lowered = name.lower()
    return any(
        control in lowered
        for control in (
            "begin_trace_capture",
            "end_trace_capture",
            "execute_trace",
            "release_trace",
        )
    )


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
