from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .trace import (
    _begin_graph_capture,
    _end_graph_capture,
    _program_cache_entries,
    _release_trace,
    _synchronize,
)


PREFILL_EXECUTION_MODES = ("eager", "trace")
PREFILL_GRAPH_CAPTURE_PATH_ENV = "BUDDY_TTNN_PREFILL_GRAPH_PATH"


class PrefillTraceUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefillTraceKey:
    prefill_len: int
    batch_size: int
    config_hash: str
    device_id: int
    layer_count: int
    cache_len: int

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrefillTraceExecution:
    token: Any
    kv_cache: Any
    cache_reports: list[dict[str, Any]]
    latency_ms: float


class PrefillTraceSession:
    """Capture and replay one prompt-shape-specific batched prefill."""

    def __init__(
        self,
        *,
        ttnn: Any,
        device: Any,
        model: Any,
        token_ids: Any,
        page_table: Any,
        kv_cache: Any,
        valid_seq_len: int | list[int] | None,
        key: PrefillTraceKey,
        cq_id: int = 0,
        graph_capture_path: str | Path | None = None,
    ) -> None:
        _require_prefill_trace_apis(ttnn)
        self.ttnn = ttnn
        self.device = device
        self.model = model
        self.token_ids = token_ids
        self.page_table = page_table
        self.kv_cache_handles = kv_cache
        self.valid_seq_len = valid_seq_len
        self.key = key
        self.cq_id = int(cq_id)
        configured_graph_path = graph_capture_path or os.environ.get(
            PREFILL_GRAPH_CAPTURE_PATH_ENV
        )
        self.graph_capture_path = (
            Path(configured_graph_path).resolve() if configured_graph_path else None
        )
        self.trace_id: Any | None = None
        self.output_token: Any | None = None
        self.hidden_input: Any | None = None
        self.cache_reports: list[dict[str, Any]] = []
        self.capture_count = 0
        self.execute_count = 0
        self.compile_run_count = 0
        self.trace_input_update_count = 0
        self.compile_latency_ms: float | None = None
        self.capture_latency_ms: float | None = None
        self.last_input_update_latency_ms = 0.0
        self.last_replay_latency_ms: float | None = None
        self.program_cache_entries_before_compile = _program_cache_entries(device)
        self.program_cache_entries_after_compile: int | None = None
        self.program_cache_entries_after_capture: int | None = None
        self.program_cache_entries_after_execute: int | None = None
        self._released = False

    def capture(self) -> None:
        if self.trace_id is not None:
            raise RuntimeError("prefill trace has already been captured")
        if self._released:
            raise RuntimeError("prefill trace session has been released")

        compile_start = time.perf_counter()
        self.hidden_input = self._prepare_hidden(self.token_ids)
        self._call_prefill()
        _synchronize(self.ttnn, self.device)
        self.compile_latency_ms = (time.perf_counter() - compile_start) * 1000.0
        self.compile_run_count += 1
        self.program_cache_entries_after_compile = _program_cache_entries(self.device)

        graph_capture_state = _begin_graph_capture(
            self.ttnn,
            self.graph_capture_path,
        )
        trace_id = None
        capture_start = time.perf_counter()
        try:
            trace_id = self.ttnn.begin_trace_capture(
                self.device,
                cq_id=self.cq_id,
            )
            token, kv_cache, cache_reports = self._call_prefill()
            self.ttnn.end_trace_capture(
                self.device,
                trace_id,
                cq_id=self.cq_id,
            )
        except Exception:
            if trace_id is not None:
                try:
                    self.ttnn.end_trace_capture(
                        self.device,
                        trace_id,
                        cq_id=self.cq_id,
                    )
                finally:
                    _release_trace(self.ttnn, self.device, trace_id)
            if graph_capture_state is not None:
                _end_graph_capture(
                    self.ttnn,
                    self.graph_capture_path,
                    graph_capture_state,
                )
            raise
        if graph_capture_state is not None:
            _end_graph_capture(
                self.ttnn,
                self.graph_capture_path,
                graph_capture_state,
            )
        _synchronize(self.ttnn, self.device)
        self.capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0
        self.trace_id = trace_id
        self.output_token = token
        self.kv_cache_handles = kv_cache
        self.cache_reports = cache_reports
        self.capture_count += 1
        self.program_cache_entries_after_capture = _program_cache_entries(self.device)

    def update_inputs(
        self,
        *,
        token_ids: Any | None = None,
        page_table: Any | None = None,
        force_token_update: bool = False,
    ) -> None:
        update_start = time.perf_counter()
        updated = False
        if token_ids is not None and (
            force_token_update or token_ids is not self.token_ids
        ):
            if self.hidden_input is not None and callable(
                getattr(self.model, "embed", None)
            ):
                new_hidden = self._prepare_hidden(token_ids)
                self.ttnn.copy(new_hidden, self.hidden_input)
                self.token_ids = token_ids
                self.trace_input_update_count += 1
                updated = True
            elif token_ids is not self.token_ids:
                self.ttnn.copy(token_ids, self.token_ids)
                self.token_ids = token_ids
                self.trace_input_update_count += 1
                updated = True
        if page_table is not None and page_table is not self.page_table:
            self.ttnn.copy(page_table, self.page_table)
            self.page_table = page_table
            self.trace_input_update_count += 1
            updated = True
        if updated:
            _synchronize(self.ttnn, self.device)
            self.last_input_update_latency_ms = (
                time.perf_counter() - update_start
            ) * 1000.0
        else:
            self.last_input_update_latency_ms = 0.0

    def execute(self) -> PrefillTraceExecution:
        if self.trace_id is None or self.output_token is None:
            raise RuntimeError("prefill trace must be captured before replay")
        start = time.perf_counter()
        self.ttnn.execute_trace(
            self.device,
            self.trace_id,
            cq_id=self.cq_id,
            blocking=False,
        )
        _synchronize(self.ttnn, self.device)
        replay_latency_ms = (time.perf_counter() - start) * 1000.0
        self.last_replay_latency_ms = replay_latency_ms
        latency_ms = self.last_input_update_latency_ms + replay_latency_ms
        self.execute_count += 1
        self.program_cache_entries_after_execute = _program_cache_entries(self.device)
        return PrefillTraceExecution(
            token=self.output_token,
            kv_cache=self.kv_cache_handles,
            cache_reports=self.cache_reports,
            latency_ms=latency_ms,
        )

    def to_report(self) -> dict[str, Any]:
        after_capture = self.program_cache_entries_after_capture
        after_execute = self.program_cache_entries_after_execute
        return {
            "execution_mode": "trace",
            "trace_key": self.key.to_report(),
            "trace_capture_count": self.capture_count,
            "trace_execute_count": self.execute_count,
            "compile_run_count": self.compile_run_count,
            "persistent_input_count": 3,
            "trace_input_update_count": self.trace_input_update_count,
            "compile_latency_ms": self.compile_latency_ms,
            "capture_latency_ms": self.capture_latency_ms,
            "input_update_latency_ms": self.last_input_update_latency_ms,
            "replay_latency_ms": self.last_replay_latency_ms,
            "program_compile_count_after_capture": (
                max(0, int(after_execute) - int(after_capture))
                if after_capture is not None and after_execute is not None
                else None
            ),
            "program_compile_count_during_capture": (
                max(
                    0,
                    int(after_capture)
                    - int(self.program_cache_entries_after_compile),
                )
                if after_capture is not None
                and self.program_cache_entries_after_compile is not None
                else None
            ),
            "program_cache_entries_before_compile": (
                self.program_cache_entries_before_compile
            ),
            "program_cache_entries_after_compile": (
                self.program_cache_entries_after_compile
            ),
            "program_cache_entries_after_capture": after_capture,
            "program_cache_entries_after_execute": after_execute,
            "cache_dispatch_contract": "captured_batch_prefill_replay",
            "host_cache_fill_dispatches_per_replay": 0,
            "embedding_contract": "outside_trace_persistent_tile_hidden",
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

    def _prepare_hidden(self, token_ids: Any) -> Any:
        embed = getattr(self.model, "embed", None)
        if not callable(embed):
            return token_ids
        hidden = embed(token_ids)
        ops = getattr(self.model, "ops", None)
        ensure_tile = getattr(ops, "ensure_tile_layout", None)
        if callable(ensure_tile):
            hidden = ensure_tile(
                hidden,
                op_name="to_layout.tile.prefill_trace_input",
            )
        normalize = getattr(ops, "reshape_prefill_hidden_for_layer", None)
        if callable(normalize):
            hidden = normalize(
                hidden,
                op_name="reshape_prefill_trace_input",
            )
        return hidden

    def _call_prefill(self) -> tuple[Any, Any, list[dict[str, Any]]]:
        original_embed = getattr(self.model, "embed", None)
        if self.hidden_input is None or not callable(original_embed):
            return self.model.prefill_prompt(
                self.token_ids,
                self.kv_cache_handles,
                self.page_table,
                valid_seq_len=self.valid_seq_len,
            )
        self.model.embed = lambda _token_ids: self.hidden_input
        try:
            return self.model.prefill_prompt(
                self.token_ids,
                self.kv_cache_handles,
                self.page_table,
                valid_seq_len=self.valid_seq_len,
            )
        finally:
            self.model.embed = original_embed


def build_prefill_trace_key(
    *,
    device_id: int,
    config: dict[str, Any],
    prefill_plan: dict[str, Any],
    layer_count: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    dtype_seed: str,
) -> PrefillTraceKey:
    payload = {
        "config": config,
        "prefill_plan": prefill_plan,
        "layer_count": int(layer_count),
        "batch_size": int(batch_size),
        "prefill_len": int(prefill_len),
        "cache_len": int(cache_len),
        "dtype_seed": dtype_seed,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return PrefillTraceKey(
        prefill_len=int(prefill_len),
        batch_size=int(batch_size),
        config_hash=hashlib.sha256(encoded).hexdigest(),
        device_id=int(device_id),
        layer_count=int(layer_count),
        cache_len=int(cache_len),
    )


def resolve_prefill_execution_mode(requested: str | None) -> str:
    mode = requested or "eager"
    if mode not in PREFILL_EXECUTION_MODES:
        raise ValueError(
            "prefill_execution_mode must be one of: "
            + ", ".join(PREFILL_EXECUTION_MODES)
        )
    return mode


def eager_prefill_execution_report(
    *, batch_size: int, layer_count: int
) -> dict[str, Any]:
    return {
        "execution_mode": "eager",
        "trace_key": None,
        "trace_capture_count": 0,
        "trace_execute_count": 0,
        "compile_run_count": 0,
        "persistent_input_count": 2,
        "trace_input_update_count": 0,
        "cache_dispatch_contract": "host_enqueued_per_user_cache_fill",
        "host_cache_fill_dispatches_per_replay": (
            2 * int(batch_size) * int(layer_count)
        ),
        "page_table_reused": True,
        "execution_graph_path": None,
    }


def _require_prefill_trace_apis(ttnn: Any) -> None:
    missing = [
        name
        for name in (
            "begin_trace_capture",
            "end_trace_capture",
            "execute_trace",
            "copy",
        )
        if not callable(getattr(ttnn, name, None))
    ]
    if missing:
        raise PrefillTraceUnsupported(
            "prefill trace requires TTNN APIs: " + ", ".join(missing)
        )
