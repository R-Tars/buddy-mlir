from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


class SDPAContextRuntimeError(ValueError):
    """Raised when generated SDPA bucket configuration is invalid."""


@dataclass(frozen=True)
class RuntimeSDPAContextBucket:
    name: str
    min_context_len: int
    max_context_len: int
    representative_context_len: int
    config_key: str
    program_config: Any
    kernel_output_memory_config: Any
    post_sdpa_output_memory_config: Any

    def contains(self, context_len: int) -> bool:
        return self.min_context_len <= int(context_len) <= self.max_context_len

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_context_len": self.min_context_len,
            "max_context_len": self.max_context_len,
            "representative_context_len": self.representative_context_len,
            "config_key": self.config_key,
        }


class SDPAContextRuntime:
    """Select and install a generated SDPA config for the active context."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.buckets = load_runtime_sdpa_context_buckets(model)
        self._by_name = {bucket.name: bucket for bucket in self.buckets}
        self.active_bucket_name: str | None = None
        self.selection_count = 0
        self.config_switch_count = 0
        self.selection_overhead_ms: list[float] = []
        self.transitions: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.buckets)

    def bucket_for_context(self, context_len: int) -> RuntimeSDPAContextBucket:
        for bucket in self.buckets:
            if bucket.contains(context_len):
                return bucket
        raise SDPAContextRuntimeError(
            f"active context length {context_len} has no runtime SDPA bucket"
        )

    def activate_for_context(
        self, context_len: int
    ) -> RuntimeSDPAContextBucket:
        return self.activate(
            self.bucket_for_context(context_len), context_len=context_len
        )

    def activate_name(self, name: str) -> RuntimeSDPAContextBucket:
        try:
            bucket = self._by_name[name]
        except KeyError as exc:
            raise SDPAContextRuntimeError(
                f"unknown runtime SDPA bucket: {name}"
            ) from exc
        return self.activate(
            bucket, context_len=bucket.representative_context_len
        )

    def activate(
        self,
        bucket: RuntimeSDPAContextBucket,
        *,
        context_len: int,
    ) -> RuntimeSDPAContextBucket:
        start = time.perf_counter_ns()
        previous = self.active_bucket_name
        attention = self.model.config.attention
        attention.sdpa_program_config = bucket.program_config
        attention.sdpa_kernel_output_memory_config = (
            bucket.kernel_output_memory_config
        )
        attention.sdpa_output_memory_config = (
            bucket.post_sdpa_output_memory_config
        )
        attention.concat_heads_input_memory_config = (
            bucket.post_sdpa_output_memory_config
        )
        self.active_bucket_name = bucket.name
        self.selection_count += 1
        overhead_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        self.selection_overhead_ms.append(overhead_ms)
        if previous is not None and previous != bucket.name:
            self.config_switch_count += 1
            self.transitions.append(
                {
                    "from": previous,
                    "to": bucket.name,
                    "context_len": int(context_len),
                    "overhead_ms": overhead_ms,
                }
            )
        return bucket

    def to_report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bucket_count": len(self.buckets),
            "buckets": [bucket.to_report() for bucket in self.buckets],
            "active_bucket": self.active_bucket_name,
            "selection_count": self.selection_count,
            "config_switch_count": self.config_switch_count,
            "selection_overhead_ms": list(self.selection_overhead_ms),
            "trace_switch_overhead_ms": [
                float(item["overhead_ms"]) for item in self.transitions
            ],
            "transitions": list(self.transitions),
        }

    def reset_metrics(self) -> None:
        self.selection_count = 0
        self.config_switch_count = 0
        self.selection_overhead_ms.clear()
        self.transitions.clear()


def load_runtime_sdpa_context_buckets(
    model: Any,
) -> tuple[RuntimeSDPAContextBucket, ...]:
    attention = getattr(getattr(model, "config", None), "attention", None)
    values = getattr(attention, "sdpa_context_buckets", None)
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SDPAContextRuntimeError(
            "attention.sdpa_context_buckets must be a list"
        )
    buckets = tuple(_runtime_bucket(value) for value in values)
    expected_start = 1
    names = set()
    config_keys = set()
    for bucket in buckets:
        if bucket.name in names or bucket.config_key in config_keys:
            raise SDPAContextRuntimeError(
                "runtime SDPA bucket names and config keys must be unique"
            )
        if bucket.min_context_len != expected_start:
            raise SDPAContextRuntimeError(
                "runtime SDPA buckets must be ordered and contiguous"
            )
        names.add(bucket.name)
        config_keys.add(bucket.config_key)
        expected_start = bucket.max_context_len + 1
    return buckets


def raw_sdpa_context_bucket_configs(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attention = config.get("attention")
    if not isinstance(attention, Mapping):
        return []
    values = attention.get("sdpa_context_buckets")
    if values is None:
        return []
    if not isinstance(values, list) or not all(
        isinstance(item, Mapping) for item in values
    ):
        raise SDPAContextRuntimeError(
            "attention.sdpa_context_buckets must be a list"
        )
    return [dict(item) for item in values]


def _runtime_bucket(value: Any) -> RuntimeSDPAContextBucket:
    item = _as_mapping(value)
    required = (
        "name",
        "min_context_len",
        "max_context_len",
        "representative_context_len",
        "config_key",
        "program_config",
        "kernel_output_memory_config",
        "post_sdpa_output_memory_config",
    )
    missing = [name for name in required if name not in item]
    if missing:
        raise SDPAContextRuntimeError(
            "runtime SDPA bucket is missing fields: " + ", ".join(missing)
        )
    start = int(item["min_context_len"])
    end = int(item["max_context_len"])
    representative = int(item["representative_context_len"])
    if start <= 0 or end < start or not start <= representative <= end:
        raise SDPAContextRuntimeError("runtime SDPA bucket bounds are invalid")
    return RuntimeSDPAContextBucket(
        name=str(item["name"]),
        min_context_len=start,
        max_context_len=end,
        representative_context_len=representative,
        config_key=str(item["config_key"]),
        program_config=item["program_config"],
        kernel_output_memory_config=item["kernel_output_memory_config"],
        post_sdpa_output_memory_config=item["post_sdpa_output_memory_config"],
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return dict(fields)
    raise SDPAContextRuntimeError("runtime SDPA bucket must be an object")
