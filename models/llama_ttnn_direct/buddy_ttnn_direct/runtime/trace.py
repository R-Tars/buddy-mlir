from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .sdpa_context import (
    SDPAContextRuntime,
    raw_sdpa_context_bucket_configs,
)

EXECUTION_MODES = ("eager", "trace")
P150_LLAMA31_8B_TRACE_REGION_SIZE = 52_000_000
GRAPH_CAPTURE_PATH_ENV = "BUDDY_TTNN_DECODE_GRAPH_PATH"


class DecodeTraceUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodeTraceKey:
    device_id: int
    program_config_hash: str
    layer_count: int
    batch_size: int
    cache_len: int
    page_block_size: int
    dtype_recipe: str
    argmax_strategy: str
    context_bucket_name: str | None = None
    min_context_len: int | None = None
    max_context_len: int | None = None
    sdpa_config_key: str | None = None

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeTraceExecution:
    token: Any
    kv_cache: Any
    latency_ms: float
    cache_positions: list[int]
    runtime_state: Any


class DecodeTraceSession:
    """Own one full-device decode trace and its stable tensor handles."""

    def __init__(
        self,
        *,
        ttnn: Any,
        device: Any,
        model: Any,
        persistent_inputs: Any,
        kv_cache: Any,
        key: DecodeTraceKey,
        cq_id: int = 0,
        graph_capture_path: str | Path | None = None,
        prepare_token_input: bool = True,
    ) -> None:
        _require_trace_apis(ttnn)
        self.ttnn = ttnn
        self.device = device
        self.model = model
        self.persistent_inputs = persistent_inputs
        self.kv_cache_handles = kv_cache
        self.key = key
        self.cq_id = int(cq_id)
        self.prepare_token_input = bool(prepare_token_input)
        configured_graph_path = graph_capture_path or os.environ.get(
            GRAPH_CAPTURE_PATH_ENV
        )
        self.graph_capture_path = (
            Path(configured_graph_path).resolve()
            if configured_graph_path
            else None
        )
        self.trace_id: Any | None = None
        self.output_token: Any | None = None
        self.capture_count = 0
        self.execute_count = 0
        self.compile_run_count = 0
        self.trace_input_update_count = 0
        self.captured_model_ops: list[str] | None = None
        self.program_cache_entries_before_compile = _program_cache_entries(
            device
        )
        self.program_cache_entries_after_compile: int | None = None
        self.program_cache_entries_after_capture: int | None = None
        self.program_cache_entries_after_execute: int | None = None
        self._released = False

    def capture(self) -> None:
        if self.trace_id is not None:
            raise RuntimeError("decode trace has already been captured")
        if self._released:
            raise RuntimeError("decode trace session has been released")

        # Compile every program on scratch update targets so live autoregressive
        # inputs retain their pre-capture values.
        if self.prepare_token_input:
            self.persistent_inputs.prepare_token_for_trace()
        self.persistent_inputs.materialize_rotary_for_trace()
        token, _ = self.model.decode_step(
            self.persistent_inputs.token_input,
            self.persistent_inputs.page_table,
            self.persistent_inputs.cache_position,
            self.kv_cache_handles,
        )
        token_scratch = self.ttnn.clone(self.persistent_inputs.token_input)
        cache_position_scratch = self.ttnn.clone(
            self.persistent_inputs.cache_position
        )
        rotary_index_scratch = self.ttnn.clone(
            self.persistent_inputs.rotary_index
        )
        self.ttnn.copy(token, token_scratch)
        self.ttnn.plus_one(
            cache_position_scratch,
            skip_negative_entries=True,
        )
        self.ttnn.plus_one(rotary_index_scratch)
        _synchronize(self.ttnn, self.device)
        del token_scratch, cache_position_scratch, rotary_index_scratch
        self.compile_run_count += 1
        self.program_cache_entries_after_compile = _program_cache_entries(
            self.device
        )
        _flush_device_profiler_for_audit(self.ttnn, self.device)

        graph_capture_state = _begin_graph_capture(
            self.ttnn,
            self.graph_capture_path,
        )
        trace_id = None
        op_log = getattr(getattr(self.model, "ops", None), "op_log", None)
        op_cursor = len(op_log) if isinstance(op_log, list) else None
        try:
            trace_id = self.ttnn.begin_trace_capture(
                self.device,
                cq_id=self.cq_id,
            )
            self.persistent_inputs.materialize_rotary_for_trace()
            token, kv_cache = self.model.decode_step(
                self.persistent_inputs.token_input,
                self.persistent_inputs.page_table,
                self.persistent_inputs.cache_position,
                self.kv_cache_handles,
            )
            self.ttnn.copy(token, self.persistent_inputs.token_input)
            self.ttnn.plus_one(
                self.persistent_inputs.cache_position,
                skip_negative_entries=True,
            )
            self.ttnn.plus_one(self.persistent_inputs.rotary_index)
        except Exception:
            if trace_id is not None:
                self.ttnn.end_trace_capture(
                    self.device,
                    trace_id,
                    cq_id=self.cq_id,
                )
                _release_trace(self.ttnn, self.device, trace_id)
            if graph_capture_state is not None:
                _end_graph_capture(
                    self.ttnn,
                    self.graph_capture_path,
                    graph_capture_state,
                )
            raise
        self.ttnn.end_trace_capture(
            self.device,
            trace_id,
            cq_id=self.cq_id,
        )
        if graph_capture_state is not None:
            _end_graph_capture(
                self.ttnn,
                self.graph_capture_path,
                graph_capture_state,
            )

        self.trace_id = trace_id
        self.output_token = token
        self.kv_cache_handles = kv_cache
        self.capture_count += 1
        if isinstance(op_log, list) and op_cursor is not None:
            self.captured_model_ops = [str(item) for item in op_log[op_cursor:]]
        self.program_cache_entries_after_capture = _program_cache_entries(
            self.device
        )

    def execute(self) -> DecodeTraceExecution:
        if self.trace_id is None or self.output_token is None:
            raise RuntimeError("decode trace must be captured before replay")
        self.persistent_inputs.validate_trace_execution()
        executed_positions = list(self.persistent_inputs.positions)
        start = time.perf_counter()
        self.ttnn.execute_trace(
            self.device,
            self.trace_id,
            cq_id=self.cq_id,
            blocking=False,
        )
        _synchronize(self.ttnn, self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        self.execute_count += 1
        self.trace_input_update_count += 3
        runtime_state = self.persistent_inputs.record_trace_execution(
            self.output_token
        )
        self.program_cache_entries_after_execute = _program_cache_entries(
            self.device
        )
        return DecodeTraceExecution(
            token=self.output_token,
            kv_cache=self.kv_cache_handles,
            latency_ms=latency_ms,
            cache_positions=executed_positions,
            runtime_state=runtime_state,
        )

    def to_report(self) -> dict[str, Any]:
        after_capture = self.program_cache_entries_after_capture
        after_execute = self.program_cache_entries_after_execute
        compile_count_after_capture = None
        if after_capture is not None and after_execute is not None:
            compile_count_after_capture = max(0, after_execute - after_capture)
        compile_count_during_capture = None
        if (
            self.program_cache_entries_after_compile is not None
            and after_capture is not None
        ):
            compile_count_during_capture = max(
                0,
                after_capture - self.program_cache_entries_after_compile,
            )
        return {
            "execution_mode": "trace",
            "trace_key": self.key.to_report(),
            "trace_capture_count": self.capture_count,
            "trace_execute_count": self.execute_count,
            "compile_run_count": self.compile_run_count,
            "persistent_input_count": int(
                self.persistent_inputs.device_tensor_creation_count
            ),
            "trace_input_update_count": self.trace_input_update_count,
            "trace_input_updates_per_decode_step": 3,
            "captured_model_op_count": (
                len(self.captured_model_ops)
                if self.captured_model_ops is not None
                else None
            ),
            "captured_model_ops": (
                list(self.captured_model_ops)
                if self.captured_model_ops is not None
                else None
            ),
            "program_compile_count_after_capture": (
                compile_count_after_capture
            ),
            "program_compile_count_during_capture": (
                compile_count_during_capture
            ),
            "program_cache_entries_before_compile": (
                self.program_cache_entries_before_compile
            ),
            "program_cache_entries_after_compile": (
                self.program_cache_entries_after_compile
            ),
            "program_cache_entries_after_capture": after_capture,
            "program_cache_entries_after_execute": after_execute,
            "token_update": "captured_device_to_device_copy",
            "cache_position_update": "captured_ttnn.plus_one_in_place",
            "rotary_update": "captured_device_embedding_from_cache",
            "page_table_reused": True,
            "trace_released": self._released,
            "execution_graph_path": (
                str(self.graph_capture_path)
                if self.graph_capture_path is not None
                else None
            ),
        }

    def close(self) -> None:
        if self.trace_id is not None and not self._released:
            _release_trace(self.ttnn, self.device, self.trace_id)
            self._released = True


class BucketedDecodeTraceSession:
    """Own one decode trace per active-context SDPA bucket."""

    def __init__(
        self,
        *,
        ttnn: Any,
        device: Any,
        model: Any,
        persistent_inputs: Any,
        kv_cache: Any,
        keys: Sequence[DecodeTraceKey],
        cq_id: int = 0,
    ) -> None:
        self.ttnn = ttnn
        self.device = device
        self.model = model
        self.persistent_inputs = persistent_inputs
        self.kv_cache_handles = kv_cache
        self.keys = tuple(keys)
        self.cq_id = int(cq_id)
        self.sdpa_runtime = SDPAContextRuntime(model)
        configured_names = {bucket.name for bucket in self.sdpa_runtime.buckets}
        key_names = {key.context_bucket_name for key in self.keys}
        if not self.keys or None in key_names or key_names != configured_names:
            raise ValueError(
                "bucketed decode trace keys must match generated SDPA context buckets"
            )
        self.sessions: dict[str, DecodeTraceSession] = {}
        self.captured_model_ops: list[str] | None = None
        self.bucket_latency_ms: dict[str, list[float]] = {
            str(key.context_bucket_name): [] for key in self.keys
        }
        self.active_bucket_name: str | None = None
        self.trace_switch_count = 0
        self.trace_switch_overhead_ms: list[float] = []
        self.program_cache_entries_before_capture = _program_cache_entries(
            device
        )
        self.program_cache_entries_after_capture: int | None = None
        self._closed = False

    def capture(self) -> None:
        if self.sessions:
            raise RuntimeError(
                "bucketed decode traces have already been captured"
            )
        captured_ops: list[str] = []
        for index, key in enumerate(self.keys):
            name = str(key.context_bucket_name)
            self.sdpa_runtime.activate_name(name)
            session = DecodeTraceSession(
                ttnn=self.ttnn,
                device=self.device,
                model=self.model,
                persistent_inputs=self.persistent_inputs,
                kv_cache=self.kv_cache_handles,
                key=key,
                cq_id=self.cq_id,
                prepare_token_input=index == 0,
            )
            session.capture()
            self.sessions[name] = session
            if session.captured_model_ops:
                for operation in session.captured_model_ops:
                    if operation not in captured_ops:
                        captured_ops.append(operation)
        self.captured_model_ops = captured_ops or None
        self.active_bucket_name = None
        self.program_cache_entries_after_capture = _program_cache_entries(
            self.device
        )
        self.sdpa_runtime.active_bucket_name = None
        self.sdpa_runtime.reset_metrics()

    def execute(self) -> DecodeTraceExecution:
        if not self.sessions:
            raise RuntimeError(
                "bucketed decode traces must be captured before replay"
            )
        context_len = (
            max(int(value) for value in self.persistent_inputs.positions) + 1
        )
        select_start = time.perf_counter_ns()
        bucket = self.sdpa_runtime.activate_for_context(context_len)
        session = self.sessions[bucket.name]
        switch_overhead_ms = (
            time.perf_counter_ns() - select_start
        ) / 1_000_000.0
        if (
            self.active_bucket_name is not None
            and self.active_bucket_name != bucket.name
        ):
            self.trace_switch_count += 1
            self.trace_switch_overhead_ms.append(switch_overhead_ms)
        self.active_bucket_name = bucket.name
        execution = session.execute()
        self.kv_cache_handles = execution.kv_cache
        latency_ms = execution.latency_ms + switch_overhead_ms
        self.bucket_latency_ms[bucket.name].append(latency_ms)
        return replace(execution, latency_ms=latency_ms)

    def to_report(self) -> dict[str, Any]:
        reports = {
            name: session.to_report() for name, session in self.sessions.items()
        }
        after_execute = _program_cache_entries(self.device)
        after_capture = self.program_cache_entries_after_capture
        compile_count_after_capture = None
        if after_capture is not None and after_execute is not None:
            compile_count_after_capture = max(0, after_execute - after_capture)
        compile_count_during_capture = None
        if (
            self.program_cache_entries_before_capture is not None
            and after_capture is not None
        ):
            compile_count_during_capture = max(
                0,
                after_capture - self.program_cache_entries_before_capture,
            )
        return {
            "execution_mode": "trace",
            "trace_key": None,
            "trace_keys": [key.to_report() for key in self.keys],
            "trace_capture_count": sum(
                int(report["trace_capture_count"])
                for report in reports.values()
            ),
            "trace_execute_count": sum(
                int(report["trace_execute_count"])
                for report in reports.values()
            ),
            "compile_run_count": sum(
                int(report["compile_run_count"]) for report in reports.values()
            ),
            "persistent_input_count": int(
                self.persistent_inputs.device_tensor_creation_count
            ),
            "trace_input_update_count": sum(
                int(report["trace_input_update_count"])
                for report in reports.values()
            ),
            "trace_input_updates_per_decode_step": 3,
            "captured_model_op_count": (
                len(self.captured_model_ops)
                if self.captured_model_ops is not None
                else None
            ),
            "captured_model_ops": (
                list(self.captured_model_ops)
                if self.captured_model_ops is not None
                else None
            ),
            "program_compile_count_after_capture": compile_count_after_capture,
            "program_compile_count_during_capture": compile_count_during_capture,
            "program_cache_entries_before_capture": (
                self.program_cache_entries_before_capture
            ),
            "program_cache_entries_after_capture": after_capture,
            "program_cache_entries_after_execute": after_execute,
            "token_update": "captured_device_to_device_copy",
            "cache_position_update": "captured_ttnn.plus_one_in_place",
            "rotary_update": "captured_device_embedding_from_cache",
            "page_table_reused": True,
            "trace_released": self._closed,
            "sdpa_context_buckets": self.sdpa_runtime.to_report(),
            "trace_switch_count": self.trace_switch_count,
            "trace_switch_overhead_ms": list(self.trace_switch_overhead_ms),
            "bucket_latency_ms_samples": {
                name: list(samples)
                for name, samples in self.bucket_latency_ms.items()
            },
            "bucket_traces": reports,
        }

    def close(self) -> None:
        if self._closed:
            return
        for session in self.sessions.values():
            session.close()
        self._closed = True


def build_decode_trace_key(
    *,
    device_id: int,
    config: dict[str, Any],
    decode_plan: dict[str, Any],
    layer_count: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    context_bucket_name: str | None = None,
    min_context_len: int | None = None,
    max_context_len: int | None = None,
    sdpa_config_key: str | None = None,
) -> DecodeTraceKey:
    lm_head = dict(config.get("lm_head") or {})
    template_config = dict(config.get("template_config") or {})
    dtype_recipe = template_config.get("dtype_recipe") or dtype_seed
    argmax_strategy = (
        lm_head.get("argmax_strategy")
        or template_config.get("lm_head_argmax_strategy")
        or "full_logits_untilize_multicore_argmax"
    )
    program_payload = {
        "config": config,
        "decode_plan": decode_plan,
        "layer_count": int(layer_count),
        "batch_size": int(batch_size),
        "cache_len": int(cache_len),
        "dtype_seed": dtype_seed,
        "context_bucket_name": context_bucket_name,
        "min_context_len": min_context_len,
        "max_context_len": max_context_len,
        "sdpa_config_key": sdpa_config_key,
    }
    encoded = json.dumps(
        program_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return DecodeTraceKey(
        device_id=int(device_id),
        program_config_hash=hashlib.sha256(encoded).hexdigest(),
        layer_count=int(layer_count),
        batch_size=int(batch_size),
        cache_len=int(cache_len),
        page_block_size=int(decode_plan["kv_cache"]["page_block_size"]),
        dtype_recipe=str(dtype_recipe),
        argmax_strategy=str(argmax_strategy),
        context_bucket_name=context_bucket_name,
        min_context_len=min_context_len,
        max_context_len=max_context_len,
        sdpa_config_key=sdpa_config_key,
    )


def build_decode_trace_bucket_keys(
    *,
    device_id: int,
    config: dict[str, Any],
    decode_plan: dict[str, Any],
    layer_count: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
) -> tuple[DecodeTraceKey, ...]:
    keys = []
    for bucket in raw_sdpa_context_bucket_configs(config):
        keys.append(
            build_decode_trace_key(
                device_id=device_id,
                config=config,
                decode_plan=decode_plan,
                layer_count=layer_count,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                context_bucket_name=str(bucket["name"]),
                min_context_len=int(bucket["min_context_len"]),
                max_context_len=int(bucket["max_context_len"]),
                sdpa_config_key=str(bucket["config_key"]),
            )
        )
    return tuple(keys)


def decode_trace_region_size(config: dict[str, Any]) -> int:
    bucket_count = len(raw_sdpa_context_bucket_configs(config))
    return P150_LLAMA31_8B_TRACE_REGION_SIZE * max(1, bucket_count)


def resolve_execution_mode(requested: str | None) -> str:
    mode = requested or "eager"
    if mode not in EXECUTION_MODES:
        raise ValueError(
            "execution_mode must be one of: " + ", ".join(EXECUTION_MODES)
        )
    return mode


def _program_cache_entries(device: Any) -> int | None:
    count = getattr(device, "num_program_cache_entries", None)
    if not callable(count):
        return None
    return int(count())


def _synchronize(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)


def _release_trace(ttnn: Any, device: Any, trace_id: Any) -> None:
    release = getattr(ttnn, "release_trace", None)
    if callable(release):
        release(device, trace_id)


def _flush_device_profiler_for_audit(ttnn: Any, device: Any) -> None:
    if os.environ.get("BUDDY_TTNN_PROFILER_AUDIT") != "1":
        return
    read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
    if not callable(read_profiler):
        raise RuntimeError(
            "profiler audit requires ttnn.ReadDeviceProfiler for segmented dumps"
        )
    read_profiler(device)


def _begin_graph_capture(ttnn: Any, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    graph = getattr(ttnn, "graph", None)
    begin = getattr(graph, "begin_graph_capture", None)
    if not callable(begin):
        raise DecodeTraceUnsupported(
            "execution graph diagnostics require ttnn.graph.begin_graph_capture"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    original_safe_arg_str = getattr(graph, "_safe_arg_str", None)
    if callable(original_safe_arg_str):
        graph._safe_arg_str = _recursive_safe_arg_str(original_safe_arg_str)
    run_mode = getattr(getattr(graph, "RunMode", None), "NORMAL", None)
    try:
        if run_mode is None:
            begin()
        else:
            begin(run_mode)
    except Exception:
        if callable(original_safe_arg_str):
            graph._safe_arg_str = original_safe_arg_str
        raise
    return {"original_safe_arg_str": original_safe_arg_str}


def _end_graph_capture(
    ttnn: Any,
    path: Path | None,
    state: dict[str, Any],
) -> None:
    if path is None:
        return
    graph = getattr(ttnn, "graph", None)
    end_to_file = getattr(graph, "end_graph_capture_to_file", None)
    try:
        if callable(end_to_file):
            end_to_file(str(path))
            return
        end = getattr(graph, "end_graph_capture", None)
        if not callable(end):
            raise DecodeTraceUnsupported(
                "execution graph diagnostics require ttnn.graph.end_graph_capture"
            )
        captured = end()
        path.write_text(json.dumps(captured, indent=2, default=str) + "\n")
    finally:
        original = state.get("original_safe_arg_str")
        if callable(original):
            graph._safe_arg_str = original


def _recursive_safe_arg_str(original: Any) -> Any:
    def stringify(value: Any) -> str:
        if isinstance(value, list):
            return "[" + ", ".join(stringify(item) for item in value) + "]"
        if isinstance(value, tuple):
            return "(" + ", ".join(stringify(item) for item in value) + ")"
        if isinstance(value, dict):
            return (
                "{"
                + ", ".join(
                    f"{key}: {stringify(item)}" for key, item in value.items()
                )
                + "}"
            )
        return original(value)

    return stringify


def _missing_trace_apis(ttnn: Any) -> list[str]:
    return [
        name
        for name in (
            "begin_trace_capture",
            "end_trace_capture",
            "execute_trace",
            "copy",
            "plus_one",
            "clone",
        )
        if not callable(getattr(ttnn, name, None))
    ]


def _require_trace_apis(ttnn: Any) -> None:
    missing = _missing_trace_apis(ttnn)
    if missing:
        raise DecodeTraceUnsupported(
            "full decode trace requires TTNN APIs: " + ", ".join(missing)
        )
