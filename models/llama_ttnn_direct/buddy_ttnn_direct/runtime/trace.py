from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any


EXECUTION_MODES = ("eager", "trace")
P150_LLAMA31_8B_TRACE_REGION_SIZE = 52_000_000


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
    ) -> None:
        _require_trace_apis(ttnn)
        self.ttnn = ttnn
        self.device = device
        self.model = model
        self.persistent_inputs = persistent_inputs
        self.kv_cache_handles = kv_cache
        self.key = key
        self.cq_id = int(cq_id)
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

        trace_id = self.ttnn.begin_trace_capture(
            self.device,
            cq_id=self.cq_id,
        )
        op_log = getattr(getattr(self.model, "ops", None), "op_log", None)
        op_cursor = len(op_log) if isinstance(op_log, list) else None
        try:
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
            self.ttnn.end_trace_capture(
                self.device,
                trace_id,
                cq_id=self.cq_id,
            )
            _release_trace(self.ttnn, self.device, trace_id)
            raise
        self.ttnn.end_trace_capture(
            self.device,
            trace_id,
            cq_id=self.cq_id,
        )

        self.trace_id = trace_id
        self.output_token = token
        self.kv_cache_handles = kv_cache
        self.capture_count += 1
        if isinstance(op_log, list) and op_cursor is not None:
            self.captured_model_ops = [
                str(item) for item in op_log[op_cursor:]
            ]
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
        }

    def close(self) -> None:
        if self.trace_id is not None and not self._released:
            _release_trace(self.ttnn, self.device, self.trace_id)
            self._released = True


def build_decode_trace_key(
    *,
    device_id: int,
    config: dict[str, Any],
    decode_plan: dict[str, Any],
    layer_count: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
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
    )


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
