from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..smoke_single_layer_decode import _write_report
from .reports import default_generate_report_path as _default_generate_report_path


PROFILE_GENERATE_OFFICIAL_BASELINE_ID = "tt_metal_official_llama31_8b_b32"
PROFILE_GENERATE_OFFICIAL_TPS_PER_USER = 33.1
PROFILE_GENERATE_OFFICIAL_BATCH_SIZE = 32
PROFILE_GENERATE_MILESTONE_IDS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


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


def profile_generate_from_generate_report(
    generate: dict[str, Any],
    *,
    profile_path: Path,
    generate_report_path: Path,
) -> dict[str, Any]:
    throughput = generate.get("throughput_summary") or {}
    step_latencies = [
        latency
        for latency in (
            _float_or_none(step.get("latency_ms"))
            for step in generate.get("step_reports", [])
        )
        if latency is not None
    ]
    prefill_ms = _float_or_none((generate.get("prefill") or {}).get("latency_ms"))
    decode_total_ms = sum(step_latencies) if step_latencies else None
    decode_step_ms_mean = (
        decode_total_ms / len(step_latencies)
        if decode_total_ms is not None and step_latencies
        else None
    )
    tokens_per_second_per_user = _float_or_none(
        throughput.get("tokens_per_second_per_user")
    )
    aggregate_tokens_per_second = _float_or_none(
        throughput.get("aggregate_tokens_per_second")
    )
    host_copy = generate.get("host_copy_profile") or {}
    section_profile = generate.get("section_profile") or {}
    host_copy_ms = _float_or_none(host_copy.get("total_ms"))
    decode_token_materialization_samples = [
        latency
        for latency in (
            _float_or_none(value)
            for value in host_copy.get(
                "decode_token_materialization_ms_samples", []
            )
        )
        if latency is not None
    ]
    dry_run = generate.get("status") == "dry_run"
    generate_ran = bool(generate.get("passed"))
    has_positive_throughput = (
        tokens_per_second_per_user is not None
        and tokens_per_second_per_user > 0.0
    )
    performance_milestones = _profile_generate_performance_milestones(
        generate,
        tokens_per_second_per_user=tokens_per_second_per_user,
    )
    acceptance = _profile_generate_acceptance(
        dry_run=dry_run,
        generate_ran=generate_ran,
        has_positive_throughput=has_positive_throughput,
        performance_milestones=performance_milestones,
    )
    if dry_run:
        status = "dry_run"
    elif acceptance["passed"]:
        status = "profiled"
    else:
        status = generate.get("status") or "profile_incomplete"

    return {
        "schema_version": 1,
        "command": "profile-generate",
        "mode": "profile-generate",
        "template": "prefill_then_decode_generate_profile",
        "status": status,
        "passed": bool(acceptance["passed"]),
        "dry_run": dry_run,
        "program_dir": generate.get("program_dir"),
        "generate_report": str(generate_report_path),
        "profile_report": str(profile_path),
        "generate_status": generate.get("status"),
        "generate_passed": bool(generate.get("passed")),
        "program_num_layers": generate.get("program_num_layers"),
        "layers": generate.get("layers"),
        "batch_size": generate.get("batch_size"),
        "prefill_len": generate.get("prefill_len"),
        "cache_len": generate.get("cache_len"),
        "max_new_tokens": generate.get("max_new_tokens"),
        "decode_steps": generate.get("decode_steps"),
        "prefill_status": generate.get("prefill_status"),
        "kv_cache_source": generate.get("kv_cache_source"),
        "parameter_source": generate.get("parameter_source"),
        "input_source": generate.get("input_source"),
        "runtime_owner": generate.get("runtime_owner"),
        "model_semantics": generate.get("model_semantics"),
        "generate_runtime_owned": generate.get("generate_runtime_owned"),
        "decode_loop_runtime_owned": generate.get("decode_loop_runtime_owned"),
        "runtime_context": generate.get("runtime_context"),
        "parameter_setup": generate.get("parameter_setup"),
        "decode_token_runtime_handoff": generate.get(
            "decode_token_runtime_handoff"
        ),
        "decode_token_host_roundtrip_per_step": generate.get(
            "decode_token_host_roundtrip_per_step"
        ),
        "host_token_materialization_for_reporting_only": generate.get(
            "host_token_materialization_for_reporting_only"
        ),
        "end_to_end_contract": generate.get("end_to_end_contract"),
        "synthetic_runtime_input_tensor_count": generate.get(
            "synthetic_runtime_input_tensor_count"
        ),
        "synthetic_rotary_tensor_count": generate.get(
            "synthetic_rotary_tensor_count"
        ),
        "synthetic_kv_cache_tensor_count": generate.get(
            "synthetic_kv_cache_tensor_count"
        ),
        "prefill_prompt_runtime_input_tensor_count": (
            _setup_count(generate, "prefill_prompt_runtime_input_tensor_count")
        ),
        "prefill_rotary_runtime_input_tensor_count": (
            _setup_count(generate, "prefill_rotary_runtime_input_tensor_count")
        ),
        "decode_runtime_state_input_tensor_count": (
            _setup_count(generate, "decode_runtime_state_input_tensor_count")
        ),
        "decode_rotary_runtime_input_tensor_count": (
            _setup_count(generate, "decode_rotary_runtime_input_tensor_count")
        ),
        "kv_cache_runtime_input_tensor_count": (
            _setup_count(generate, "kv_cache_runtime_input_tensor_count")
        ),
        "generated_text_status": generate.get("generated_text_status"),
        "generated_token_count_by_user": _generated_token_counts(generate),
        "latency_ms": _float_or_none(generate.get("latency_ms")),
        "prefill_ms": prefill_ms,
        "decode_step_ms_mean": decode_step_ms_mean,
        "decode_step_ms_min": min(step_latencies) if step_latencies else None,
        "decode_step_ms_max": max(step_latencies) if step_latencies else None,
        "decode_step_ms_samples": step_latencies,
        "host_copy_ms": host_copy_ms,
        "host_copy_profile": host_copy,
        "section_profile": section_profile,
        "prefill_first_token_materialization_ms": _float_or_none(
            host_copy.get("prefill_first_token_ms")
        ),
        "decode_token_materialization_ms_mean": _float_or_none(
            host_copy.get("decode_token_materialization_ms_mean")
        ),
        "decode_token_materialization_ms_samples": (
            decode_token_materialization_samples
        ),
        "tokens_per_second_per_user": tokens_per_second_per_user,
        "aggregate_tokens_per_second": aggregate_tokens_per_second,
        "throughput_summary": throughput,
        "sections": _profile_generate_sections(
            prefill_ms=prefill_ms,
            decode_total_ms=decode_total_ms,
            decode_step_ms_mean=decode_step_ms_mean,
            host_copy_profile=host_copy,
            section_profile=section_profile,
        ),
        "per_layer": _profile_generate_per_layer(generate),
        "performance_milestones": performance_milestones,
        "acceptance": acceptance,
        "official_performance_parity_claimed": False,
        "message": (
            "First generate profile only; no official performance parity is "
            "claimed."
        ),
        "error": None if acceptance["passed"] else generate.get("error"),
        "ttnn_environment": generate.get("ttnn_environment"),
    }


def run_profile_generate(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int = 2,
    layers: int = 1,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    generate_report: str | Path | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    from .generate import run_generate

    profile_path = Path(out)
    generate_report_path = (
        Path(generate_report)
        if generate_report is not None
        else _default_generate_report_path(profile_path)
    )
    generate_payload = run_generate(
        out=generate_report_path,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_new_tokens,
        layers=layers,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=dry_run,
        ttnn_module=ttnn_module,
        torch_module=torch_module,
        tokenizer_module=tokenizer_module,
    )
    report = profile_generate_from_generate_report(
        generate_payload,
        profile_path=profile_path,
        generate_report_path=generate_report_path,
    )
    _write_report(profile_path, report)
    return report


def _profile_generate_sections(
    *,
    prefill_ms: float | None,
    decode_total_ms: float | None,
    decode_step_ms_mean: float | None,
    host_copy_profile: dict[str, Any],
    section_profile: dict[str, Any],
) -> dict[str, Any]:
    unavailable = {
        "status": "unavailable",
        "value_ms": None,
        "reason": (
            "section-level timers were not measured for this generate run"
        ),
    }
    measured_sections = section_profile.get("sections_ms")
    if not isinstance(measured_sections, dict):
        measured_sections = {}
    host_copy_ms = _float_or_none(host_copy_profile.get("total_ms"))
    host_copy_status = host_copy_profile.get("status") or "unavailable"
    if host_copy_ms is None:
        host_copy_section = {
            "status": host_copy_status,
            "value_ms": None,
            "host_roundtrip_present": bool(
                host_copy_profile.get("host_roundtrip_present")
            ),
            "reason": host_copy_profile.get(
                "reason",
                "host token materialization did not run",
            ),
        }
    else:
        host_copy_section = {
            "status": "measured",
            "value_ms": host_copy_ms,
            "host_roundtrip_present": bool(
                host_copy_profile.get("host_roundtrip_present")
            ),
            "runtime_host_roundtrip_present": bool(
                host_copy_profile.get("runtime_host_roundtrip_present")
            ),
            "runtime_handoff": host_copy_profile.get("runtime_handoff"),
            "host_materialization_for_reporting": bool(
                host_copy_profile.get("host_materialization_for_reporting")
            ),
            "prefill_first_token_ms": _float_or_none(
                host_copy_profile.get("prefill_first_token_ms")
            ),
            "decode_token_materialization_ms_total": _float_or_none(
                host_copy_profile.get(
                    "decode_token_materialization_ms_total"
                )
            ),
            "decode_token_materialization_ms_mean": _float_or_none(
                host_copy_profile.get("decode_token_materialization_ms_mean")
            ),
            "basis": host_copy_profile.get("basis"),
        }
    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_step_ms_mean": decode_step_ms_mean,
        "embedding_ms": _profile_section(
            measured_sections,
            "embedding_ms",
            unavailable,
        ),
        "prefill_attention_ms": _profile_section(
            measured_sections,
            "prefill_attention_ms",
            unavailable,
        ),
        "decode_attention_ms": _profile_section(
            measured_sections,
            "decode_attention_ms",
            unavailable,
        ),
        "mlp_ms": _profile_section(measured_sections, "mlp_ms", unavailable),
        "prefill_mlp_ms": _profile_section(
            measured_sections,
            "prefill_mlp_ms",
            unavailable,
        ),
        "decode_mlp_ms": _profile_section(
            measured_sections,
            "decode_mlp_ms",
            unavailable,
        ),
        "final_norm_ms": _profile_section(
            measured_sections,
            "final_norm_ms",
            unavailable,
        ),
        "lm_head_ms": _profile_section(
            measured_sections,
            "lm_head_ms",
            unavailable,
        ),
        "argmax_ms": _profile_section(
            measured_sections,
            "argmax_ms",
            unavailable,
        ),
        "host_copy_ms": host_copy_section,
    }


def _profile_section(
    sections: dict[str, Any],
    name: str,
    unavailable: dict[str, Any],
) -> dict[str, Any]:
    value = _float_or_none(sections.get(name))
    if value is None:
        return dict(unavailable)
    return {
        "status": "measured",
        "value_ms": value,
        "basis": "generated model method wrappers",
    }


def _profile_generate_per_layer(generate: dict[str, Any]) -> dict[str, Any]:
    section_profile = generate.get("section_profile")
    if (
        isinstance(section_profile, dict)
        and section_profile.get("status") == "measured"
    ):
        return {
            "status": "measured",
            "layers": generate.get("layers"),
            "prefill": section_profile.get("prefill_layer_profiles", []),
            "decode": section_profile.get("decode_layer_profiles", []),
            "attention_ms": {
                "prefill": sum(
                    _float_or_none(layer.get("attention_ms")) or 0.0
                    for layer in section_profile.get(
                        "prefill_layer_profiles", []
                    )
                ),
                "decode": sum(
                    _float_or_none(layer.get("attention_ms")) or 0.0
                    for layer in section_profile.get(
                        "decode_layer_profiles", []
                    )
                ),
            },
            "mlp_ms": {
                "prefill": sum(
                    _float_or_none(layer.get("mlp_ms")) or 0.0
                    for layer in section_profile.get(
                        "prefill_layer_profiles", []
                    )
                ),
                "decode": sum(
                    _float_or_none(layer.get("mlp_ms")) or 0.0
                    for layer in section_profile.get(
                        "decode_layer_profiles", []
                    )
                ),
            },
        }
    return {
        "status": "unavailable",
        "layers": generate.get("layers"),
        "attention_ms": None,
        "mlp_ms": None,
        "reason": (
            "per-layer section timers were not measured for this generate run"
        ),
    }


def _profile_generate_performance_milestones(
    generate: dict[str, Any],
    *,
    tokens_per_second_per_user: float | None,
) -> dict[str, Any]:
    official_reference = _profile_generate_official_reference()
    official_tps = (
        _float_or_none(official_reference.get("tokens_per_second_per_user"))
        or PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
    )
    official_batch_size = (
        _int_or_none(official_reference.get("batch_size"))
        or PROFILE_GENERATE_OFFICIAL_BATCH_SIZE
    )
    dry_run = generate.get("status") == "dry_run"
    generate_passed = bool(generate.get("passed")) and not dry_run
    layers = _int_or_none(generate.get("layers"))
    program_num_layers = _int_or_none(generate.get("program_num_layers"))
    batch_size = _int_or_none(generate.get("batch_size"))

    def reason_for(
        *,
        batch32_required: bool = False,
        throughput_required: bool = False,
        full_depth_required: bool = False,
    ) -> str | None:
        if dry_run:
            return "dry_run"
        if not generate_passed:
            return "generate_not_passed"
        if full_depth_required:
            if program_num_layers is None:
                return "program_num_layers_unavailable"
            if layers != program_num_layers:
                return "not_full_depth_profile"
        if batch32_required and batch_size != official_batch_size:
            return "requires_batch32_profile"
        if throughput_required and tokens_per_second_per_user is None:
            return "tokens_per_second_per_user_unavailable"
        return None

    def entry(
        milestone_id: str,
        name: str,
        *,
        passed: bool,
        observed: dict[str, Any],
        expected: dict[str, Any],
        threshold_tokens_per_second_per_user: float | None = None,
        ratio_of_official: float | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": milestone_id,
            "name": name,
            "passed": bool(passed),
            "status": "passed" if passed else "not_met",
            "observed": observed,
            "expected": expected,
        }
        if threshold_tokens_per_second_per_user is not None:
            result["threshold_tokens_per_second_per_user"] = (
                threshold_tokens_per_second_per_user
            )
        if ratio_of_official is not None:
            result["ratio_of_official"] = ratio_of_official
        if dry_run and not passed:
            result["status"] = "dry_run"
        if reason is not None and not passed:
            result["reason"] = reason
        return result

    m0_reason = reason_for()
    m0_passed = (
        m0_reason is None and layers is not None and layers >= 1
    )
    m1_reason = reason_for(full_depth_required=True)
    m1_passed = (
        m1_reason is None
        and layers is not None
        and program_num_layers is not None
        and layers == program_num_layers
    )
    m2_reason = reason_for(batch32_required=True, throughput_required=True)
    m2_passed = (
        m2_reason is None
        and tokens_per_second_per_user is not None
        and tokens_per_second_per_user > 1.0
    )
    milestones = [
        entry(
            "M0",
            "1-layer generate works",
            passed=m0_passed,
            observed={
                "generate_passed": generate_passed,
                "layers": layers,
            },
            expected={"generate_passed": True, "layers_min": 1},
            reason=m0_reason,
        ),
        entry(
            "M1",
            "full-depth generate works, any speed",
            passed=m1_passed,
            observed={
                "generate_passed": generate_passed,
                "layers": layers,
                "program_num_layers": program_num_layers,
            },
            expected={
                "generate_passed": True,
                "layers": program_num_layers,
            },
            reason=m1_reason,
        ),
        entry(
            "M2",
            "batch32 decode t/s/u > 1",
            passed=m2_passed,
            observed={
                "generate_passed": generate_passed,
                "batch_size": batch_size,
                "tokens_per_second_per_user": tokens_per_second_per_user,
            },
            expected={
                "batch_size": official_batch_size,
                "tokens_per_second_per_user": "> 1.0",
            },
            threshold_tokens_per_second_per_user=1.0,
            reason=m2_reason,
        ),
    ]
    for milestone_id, ratio, name in (
        ("M3", 0.10, ">10% official 33.1 t/s/u"),
        ("M4", 0.30, ">30% official"),
        ("M5", 0.60, ">60% official"),
        ("M6", 0.90, ">90% official"),
    ):
        threshold = official_tps * ratio
        reason = reason_for(
            batch32_required=True,
            throughput_required=True,
        )
        passed = (
            reason is None
            and tokens_per_second_per_user is not None
            and tokens_per_second_per_user > threshold
        )
        milestones.append(
            entry(
                milestone_id,
                name,
                passed=passed,
                observed={
                    "generate_passed": generate_passed,
                    "batch_size": batch_size,
                    "tokens_per_second_per_user": (
                        tokens_per_second_per_user
                    ),
                },
                expected={
                    "batch_size": official_batch_size,
                    "tokens_per_second_per_user": f"> {threshold:.4f}",
                },
                threshold_tokens_per_second_per_user=threshold,
                ratio_of_official=ratio,
                reason=reason,
            )
        )

    highest_passed = None
    for milestone in milestones:
        if milestone["passed"]:
            highest_passed = milestone["id"]
    next_milestone = next(
        (
            {
                "id": milestone["id"],
                "name": milestone["name"],
                "reason": milestone.get("reason"),
            }
            for milestone in milestones
            if not milestone["passed"]
        ),
        None,
    )
    return {
        "schema_version": 1,
        "basis": "PR-7 generate performance milestone ladder",
        "dry_run": dry_run,
        "official_reference": official_reference,
        "observed": {
            "status": generate.get("status"),
            "passed": bool(generate.get("passed")),
            "program_num_layers": program_num_layers,
            "layers": layers,
            "batch_size": batch_size,
            "tokens_per_second_per_user": tokens_per_second_per_user,
            "aggregate_tokens_per_second": (
                _float_or_none(
                    (generate.get("throughput_summary") or {}).get(
                        "aggregate_tokens_per_second"
                    )
                )
            ),
            "model_semantics": generate.get("model_semantics"),
            "prefill_status": generate.get("prefill_status"),
            "kv_cache_source": generate.get("kv_cache_source"),
        },
        "milestones": milestones,
        "highest_passed": highest_passed,
        "next_milestone": next_milestone,
        "official_performance_parity_claimed": False,
    }


def _profile_generate_official_reference() -> dict[str, Any]:
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "performance_baselines.json"
    )
    fallback = {
        "id": PROFILE_GENERATE_OFFICIAL_BASELINE_ID,
        "role": "official_8b_target",
        "metric": "decode_tokens_per_second_per_user",
        "model": "Llama 3.1 8B",
        "batch_size": PROFILE_GENERATE_OFFICIAL_BATCH_SIZE,
        "tokens_per_second_per_user": (
            PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
        ),
        "aggregate_tokens_per_second": (
            PROFILE_GENERATE_OFFICIAL_TPS_PER_USER
            * PROFILE_GENERATE_OFFICIAL_BATCH_SIZE
        ),
        "source": "reference/performance_baselines.json",
        "baseline_file": "reference/performance_baselines.json",
        "resolved": False,
    }
    try:
        payload = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return fallback
    for entry in entries:
        if (
            isinstance(entry, dict)
            and entry.get("id") == PROFILE_GENERATE_OFFICIAL_BASELINE_ID
        ):
            return {
                "id": entry.get("id"),
                "role": entry.get("role"),
                "metric": payload.get("metric"),
                "implementation": entry.get("implementation"),
                "frontend": entry.get("frontend"),
                "model": entry.get("model"),
                "batch_size": entry.get("batch_size"),
                "tokens_per_second_per_user": entry.get(
                    "decode_tokens_per_second_per_user"
                ),
                "aggregate_tokens_per_second": entry.get(
                    "aggregate_tokens_per_second"
                ),
                "source": entry.get("source"),
                "notes": entry.get("notes"),
                "baseline_file": "reference/performance_baselines.json",
                "resolved": True,
            }
    return fallback


def _profile_generate_milestones_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    milestones = value.get("milestones")
    if not isinstance(milestones, list):
        return False
    ids = [
        milestone.get("id")
        for milestone in milestones
        if isinstance(milestone, dict)
    ]
    if ids != list(PROFILE_GENERATE_MILESTONE_IDS):
        return False
    return isinstance(value.get("official_reference"), dict) and isinstance(
        value.get("observed"),
        dict,
    )


def _profile_generate_acceptance(
    *,
    dry_run: bool,
    generate_ran: bool,
    has_positive_throughput: bool,
    performance_milestones: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        {
            "name": "profile_generate.full_generated_model_can_run",
            "passed": bool(dry_run or generate_ran),
            "dry_run": dry_run,
        },
        {
            "name": "profile_generate.tokens_per_second_per_user_positive",
            "passed": bool(dry_run or has_positive_throughput),
            "dry_run": dry_run,
        },
        {
            "name": "profile_generate.no_official_parity_claim",
            "passed": True,
        },
        {
            "name": "profile_generate.performance_milestones",
            "passed": _profile_generate_milestones_complete(
                performance_milestones
            ),
            "observed": [
                milestone.get("id")
                for milestone in performance_milestones.get(
                    "milestones",
                    [],
                )
                if isinstance(milestone, dict)
            ],
            "expected": list(PROFILE_GENERATE_MILESTONE_IDS),
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [
            check["name"] for check in checks if not check["passed"]
        ],
    }


def _generated_token_counts(generate: dict[str, Any]) -> list[int]:
    rows = generate.get("generated_token_ids") or []
    return [len(row) for row in rows if isinstance(row, list)]


def _setup_count(generate: dict[str, Any], key: str) -> Any:
    setup = generate.get("parameter_setup")
    if isinstance(setup, dict):
        return setup.get(key)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _synchronize_ttnn(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
