from __future__ import annotations

from types import SimpleNamespace

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.sdpa_context import (
    SDPAContextRuntime,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.trace import (
    BucketedDecodeTraceSession,
    DecodeTraceKey,
    build_decode_trace_bucket_keys,
    decode_trace_region_size,
)


class _Device:
    def __init__(self) -> None:
        self.program_cache_entries = 10

    def num_program_cache_entries(self) -> int:
        return self.program_cache_entries


class _TTNN:
    def __init__(self) -> None:
        self.calls = []
        self.trace_count = 0

    def begin_trace_capture(self, device, **kwargs):
        self.trace_count += 1
        trace_id = f"trace-{self.trace_count}"
        self.calls.append(("begin", trace_id))
        return trace_id

    def end_trace_capture(self, device, trace_id, **kwargs):
        self.calls.append(("end", trace_id))

    def execute_trace(self, device, trace_id, **kwargs):
        self.calls.append(("execute", trace_id))

    def copy(self, source, target):
        self.calls.append(("copy", source, target))

    def clone(self, source):
        return object()

    def plus_one(self, target, **kwargs):
        self.calls.append(("plus_one", target))

    def synchronize_device(self, device):
        self.calls.append(("synchronize", device))

    def release_trace(self, device, trace_id):
        self.calls.append(("release", trace_id))


class _Inputs:
    def __init__(self) -> None:
        self.token_input = object()
        self.page_table = object()
        self.cache_position = object()
        self.rotary_index = object()
        self.positions = [127, 127]
        self.device_tensor_creation_count = 6
        self.trace_token_input_creation_count = 0

    def prepare_token_for_trace(self) -> None:
        self.token_input = object()
        self.device_tensor_creation_count += 1
        self.trace_token_input_creation_count += 1

    def materialize_rotary_for_trace(self) -> None:
        pass

    def validate_trace_execution(self) -> None:
        pass

    def record_trace_execution(self, token):
        self.positions = [value + 1 for value in self.positions]
        return SimpleNamespace(positions=list(self.positions))


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            attention=SimpleNamespace(
                sdpa_program_config="base-program",
                sdpa_kernel_output_memory_config="base-kernel",
                sdpa_output_memory_config="base-post",
                concat_heads_input_memory_config="base-post",
                sdpa_context_buckets=[
                    _bucket("short", 1, 128),
                    _bucket("long", 129, 256),
                ],
            )
        )
        self.ops = SimpleNamespace(op_log=[])
        self.decode_programs = []

    def decode_step(self, *args):
        self.decode_programs.append(self.config.attention.sdpa_program_config)
        self.ops.op_log.append("paged_scaled_dot_product_attention_decode")
        return object(), args[-1]


def test_runtime_uses_active_context_boundaries() -> None:
    model = _Model()
    runtime = SDPAContextRuntime(model)

    first = runtime.activate_for_context(128)
    second = runtime.activate_for_context(129)

    assert first.name == "short"
    assert second.name == "long"
    assert model.config.attention.sdpa_program_config == "program-long"
    assert (
        model.config.attention.sdpa_kernel_output_memory_config == "kernel-long"
    )
    assert (
        model.config.attention.concat_heads_input_memory_config == "post-long"
    )
    assert runtime.to_report()["config_switch_count"] == 1


def test_bucketed_trace_captures_once_per_config_and_switches_at_129() -> None:
    ttnn = _TTNN()
    device = _Device()
    model = _Model()
    inputs = _Inputs()
    keys = (_key("short", 1, 128), _key("long", 129, 256))
    session = BucketedDecodeTraceSession(
        ttnn=ttnn,
        device=device,
        model=model,
        persistent_inputs=inputs,
        kv_cache=[object()],
        keys=keys,
    )

    session.capture()
    first = session.execute()
    second = session.execute()
    report = session.to_report()
    session.close()

    assert inputs.trace_token_input_creation_count == 1
    assert first.cache_positions == [127, 127]
    assert second.cache_positions == [128, 128]
    assert ("execute", "trace-1") in ttnn.calls
    assert ("execute", "trace-2") in ttnn.calls
    assert report["trace_capture_count"] == 2
    assert report["trace_execute_count"] == 2
    assert report["program_compile_count_after_capture"] == 0
    assert report["trace_switch_count"] == 1
    assert len(report["trace_switch_overhead_ms"]) == 1
    assert report["bucket_latency_ms_samples"]["short"]
    assert report["bucket_latency_ms_samples"]["long"]


def test_trace_keys_and_region_size_include_bucket_identity() -> None:
    config = {
        "attention": {
            "sdpa_context_buckets": [
                _bucket_dict("short", 1, 128),
                _bucket_dict("long", 129, 256),
            ]
        },
        "lm_head": {},
        "template_config": {},
    }
    plan = {"kv_cache": {"page_block_size": 32}}
    keys = build_decode_trace_bucket_keys(
        device_id=0,
        config=config,
        decode_plan=plan,
        layer_count=32,
        batch_size=32,
        cache_len=256,
        dtype_seed="bf16",
    )

    assert [key.context_bucket_name for key in keys] == ["short", "long"]
    assert keys[0].program_config_hash != keys[1].program_config_hash
    assert decode_trace_region_size(config) == 104_000_000


def _bucket(name: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(**_bucket_dict(name, start, end))


def _bucket_dict(name: str, start: int, end: int) -> dict:
    return {
        "name": name,
        "min_context_len": start,
        "max_context_len": end,
        "representative_context_len": end,
        "config_key": f"config-{name}",
        "program_config": f"program-{name}",
        "kernel_output_memory_config": f"kernel-{name}",
        "post_sdpa_output_memory_config": f"post-{name}",
    }


def _key(name: str, start: int, end: int) -> DecodeTraceKey:
    return DecodeTraceKey(
        device_id=0,
        program_config_hash=f"hash-{name}",
        layer_count=32,
        batch_size=32,
        cache_len=256,
        page_block_size=32,
        dtype_recipe="mixed",
        argmax_strategy="argmax",
        context_bucket_name=name,
        min_context_len=start,
        max_context_len=end,
        sdpa_config_key=f"config-{name}",
    )
