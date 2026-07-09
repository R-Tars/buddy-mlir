from __future__ import annotations

import time
from typing import Any


class GenerateSectionProfiler:
    """Records first-pass generate section timings from generated model calls."""

    def __init__(self, *, ttnn: Any, device: Any) -> None:
        self.ttnn = ttnn
        self.device = device
        self.phase_stack: list[str] = []
        self.section_latency_ms = {
            "embedding_ms": 0.0,
            "prefill_attention_ms": 0.0,
            "decode_attention_ms": 0.0,
            "mlp_ms": 0.0,
            "prefill_mlp_ms": 0.0,
            "decode_mlp_ms": 0.0,
            "final_norm_ms": 0.0,
            "lm_head_ms": 0.0,
            "argmax_ms": 0.0,
        }
        self.prefill_layer_profiles: dict[int, dict[str, Any]] = {}
        self.decode_layer_profiles: dict[int, dict[str, Any]] = {}
        self.lm_head_argmax_total_ms = 0.0
        self.argmax_total_ms = 0.0

    def install(self, model: Any) -> None:
        if getattr(model, "_buddy_generate_section_profiler", None) is self:
            return
        setattr(model, "_buddy_generate_section_profiler", self)
        self._wrap_phase(model, "prefill_prompt", "prefill")
        self._wrap_phase(model, "decode_step", "decode")
        self._wrap_method(model, "embed", self._record_embedding)
        self._wrap_method(model, "attention_prefill", self._record_prefill_attention)
        self._wrap_method(model, "attention_decode", self._record_decode_attention)
        self._wrap_method(model, "mlp_decode", self._record_mlp)
        self._wrap_method(model, "final_norm", self._record_final_norm)
        self._wrap_method(model, "lm_head_argmax", self._record_lm_head_argmax)
        self._wrap_ops_method(model, "argmax", self._record_argmax)

    def to_report(
        self,
        *,
        host_copy_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        host_copy_ms = _float_or_none((host_copy_profile or {}).get("total_ms"))
        sections = dict(self.section_latency_ms)
        sections["host_copy_ms"] = host_copy_ms
        return {
            "status": "measured",
            "basis": "generated model method wrappers",
            "sections_ms": sections,
            "prefill_layer_profiles": self._ordered_layer_profiles(
                self.prefill_layer_profiles
            ),
            "decode_layer_profiles": self._ordered_layer_profiles(
                self.decode_layer_profiles
            ),
            "lm_head_argmax_total_ms": self.lm_head_argmax_total_ms,
            "host_copy_ms": host_copy_ms,
        }

    def _wrap_phase(self, model: Any, name: str, phase: str) -> None:
        original = getattr(model, name, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.phase_stack.append(phase)
            try:
                return original(*args, **kwargs)
            finally:
                self.phase_stack.pop()

        setattr(model, name, wrapped)

    def _wrap_method(self, model: Any, name: str, recorder: Any) -> None:
        original = getattr(model, name, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            output = original(*args, **kwargs)
            _synchronize_ttnn(self.ttnn, self.device)
            recorder((time.perf_counter() - start) * 1000.0, args, kwargs)
            return output

        setattr(model, name, wrapped)

    def _wrap_ops_method(self, model: Any, name: str, recorder: Any) -> None:
        ops = getattr(model, "ops", None)
        if ops is None:
            return
        original = getattr(ops, name, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            output = original(*args, **kwargs)
            _synchronize_ttnn(self.ttnn, self.device)
            recorder((time.perf_counter() - start) * 1000.0, args, kwargs)
            return output

        setattr(ops, name, wrapped)

    def _record_embedding(
        self,
        latency_ms: float,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.section_latency_ms["embedding_ms"] += latency_ms

    def _record_prefill_attention(
        self,
        latency_ms: float,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.section_latency_ms["prefill_attention_ms"] += latency_ms
        profile = self._layer_profile(self.prefill_layer_profiles, args)
        profile["attention_ms"] += latency_ms

    def _record_decode_attention(
        self,
        latency_ms: float,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.section_latency_ms["decode_attention_ms"] += latency_ms
        profile = self._layer_profile(self.decode_layer_profiles, args)
        profile["attention_ms"] += latency_ms

    def _record_mlp(
        self,
        latency_ms: float,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        phase = self.phase_stack[-1] if self.phase_stack else "unknown"
        section = f"{phase}_mlp_ms"
        if section in self.section_latency_ms:
            self.section_latency_ms[section] += latency_ms
        self.section_latency_ms["mlp_ms"] += latency_ms
        target = (
            self.prefill_layer_profiles
            if phase == "prefill"
            else self.decode_layer_profiles
        )
        self._layer_profile(target, args)["mlp_ms"] += latency_ms

    def _record_final_norm(
        self,
        latency_ms: float,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.section_latency_ms["final_norm_ms"] += latency_ms

    def _record_lm_head_argmax(
        self,
        latency_ms: float,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.lm_head_argmax_total_ms += latency_ms
        self.section_latency_ms["lm_head_ms"] = max(
            0.0,
            self.lm_head_argmax_total_ms - self.argmax_total_ms,
        )

    def _record_argmax(
        self,
        latency_ms: float,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self.argmax_total_ms += latency_ms
        self.section_latency_ms["argmax_ms"] = self.argmax_total_ms

    def _layer_profile(
        self,
        profiles: dict[int, dict[str, Any]],
        args: tuple[Any, ...],
    ) -> dict[str, Any]:
        layer_id = int(args[0]) if args else -1
        profile = profiles.setdefault(
            layer_id,
            {
                "layer_id": layer_id,
                "attention_ms": 0.0,
                "mlp_ms": 0.0,
            },
        )
        return profile

    @staticmethod
    def _ordered_layer_profiles(
        profiles: dict[int, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [profiles[layer_id] for layer_id in sorted(profiles)]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _synchronize_ttnn(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
