from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.trace import (
    DecodeTraceKey,
    DecodeTraceSession,
    build_decode_trace_key,
    resolve_execution_mode,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.prefill_trace import (
    PrefillTraceKey,
    PrefillTraceSession,
    build_prefill_trace_key,
    resolve_prefill_execution_mode,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.device import (
    GenerateDeviceSession,
)


class _Device:
    def __init__(self) -> None:
        self.program_cache_entries = 10

    def num_program_cache_entries(self) -> int:
        return self.program_cache_entries


class _TTNN:
    def __init__(self, device: _Device) -> None:
        self.device = device
        self.calls: list[tuple[object, ...]] = []

    def begin_trace_capture(self, device: object, **kwargs: object) -> str:
        self.calls.append(("begin", device, kwargs))
        return "trace-7"

    def end_trace_capture(
        self, device: object, trace_id: object, **kwargs: object
    ) -> None:
        self.calls.append(("end", device, trace_id, kwargs))

    def execute_trace(self, device: object, trace_id: object, **kwargs: object) -> None:
        self.calls.append(("execute", device, trace_id, kwargs))

    def copy(self, source: object, target: object) -> None:
        self.calls.append(("copy", source, target))

    def clone(self, source: object) -> object:
        target = object()
        self.calls.append(("clone", source, target))
        return target

    def plus_one(self, target: object, **kwargs: object) -> None:
        self.calls.append(("plus_one", target, kwargs))

    def synchronize_device(self, device: object) -> None:
        self.calls.append(("synchronize", device))

    def release_trace(self, device: object, trace_id: object) -> None:
        self.calls.append(("release", device, trace_id))


class _Graph:
    RunMode = SimpleNamespace(NORMAL="normal")

    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls

    def begin_graph_capture(self, mode: object) -> None:
        self.calls.append(("begin_graph", mode))

    def end_graph_capture_to_file(self, path: str) -> None:
        self.calls.append(("end_graph", path))
        Path(path).write_text("[]\n")


class _Inputs:
    def __init__(self) -> None:
        self.token_input = object()
        self.page_table = object()
        self.cache_position = object()
        self.rotary_index = object()
        self.positions = [8, 9]
        self.device_tensor_creation_count = 6
        self.trace_token_input_creation_count = 0
        self.materialize_count = 0
        self.recorded_tokens: list[object] = []
        self.validation_count = 0

    def materialize_rotary_for_trace(self) -> None:
        self.materialize_count += 1

    def prepare_token_for_trace(self) -> None:
        self.token_input = object()
        self.device_tensor_creation_count += 1
        self.trace_token_input_creation_count += 1

    def validate_trace_execution(self) -> None:
        self.validation_count += 1

    def record_trace_execution(self, token: object) -> SimpleNamespace:
        self.recorded_tokens.append(token)
        self.positions = [value + 1 for value in self.positions]
        return SimpleNamespace(positions=list(self.positions))


class _Model:
    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls
        self.decode_count = 0
        self.output_token = object()

    def decode_step(self, *args: object) -> tuple[object, object]:
        self.decode_count += 1
        self.calls.append(("decode_step", *args))
        return self.output_token, args[-1]


class _PrefillModel:
    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls
        self.prefill_count = 0
        self.output_token = object()

    def prefill_prompt(
        self,
        token_ids: object,
        kv_cache: object,
        page_table: object,
        *,
        valid_seq_len: object,
    ) -> tuple[object, object, list[dict[str, object]]]:
        self.prefill_count += 1
        self.calls.append(
            (
                "prefill_prompt",
                token_ids,
                kv_cache,
                page_table,
                valid_seq_len,
            )
        )
        return self.output_token, kv_cache, [{"layer_id": 0}]


def _key() -> DecodeTraceKey:
    return DecodeTraceKey(
        device_id=0,
        program_config_hash="abc",
        layer_count=32,
        batch_size=32,
        cache_len=1024,
        page_block_size=32,
        dtype_recipe="mixed_bfp8_bf16",
        argmax_strategy="full_logits_untilize_multicore_argmax",
    )


def _prefill_key() -> PrefillTraceKey:
    return PrefillTraceKey(
        prefill_len=128,
        batch_size=32,
        config_hash="prefill-abc",
        device_id=0,
        layer_count=32,
        cache_len=1024,
    )


def test_prefill_trace_captures_batch_and_replays_nonblocking() -> None:
    device = _Device()
    ttnn = _TTNN(device)
    model = _PrefillModel(ttnn.calls)
    token_ids = object()
    page_table = object()
    kv_cache = [object(), object()]
    session = PrefillTraceSession(
        ttnn=ttnn,
        device=device,
        model=model,
        token_ids=token_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        valid_seq_len=[17] * 32,
        key=_prefill_key(),
    )

    session.capture()
    session.update_inputs(token_ids=token_ids, page_table=page_table)
    result = session.execute()
    session.close()
    report = session.to_report()

    assert model.prefill_count == 2
    assert result.token is model.output_token
    assert result.kv_cache is kv_cache
    assert result.cache_reports == [{"layer_id": 0}]
    assert report["execution_mode"] == "trace"
    assert report["trace_capture_count"] == 1
    assert report["trace_execute_count"] == 1
    assert report["compile_run_count"] == 1
    assert report["persistent_input_count"] == 3
    assert report["trace_input_update_count"] == 0
    assert report["host_cache_fill_dispatches_per_replay"] == 0
    assert report["program_compile_count_after_capture"] == 0
    assert report["program_compile_count_during_capture"] == 0
    assert report["trace_released"] is True
    assert ("execute", device, "trace-7", {"cq_id": 0, "blocking": False}) in ttnn.calls
    assert ttnn.calls[-1] == ("release", device, "trace-7")


def test_prefill_trace_key_is_stable_and_shape_sensitive() -> None:
    config = {"prefill": {"cache_write_policy": "fill_cache_per_user"}}
    plan = {"batch_size": 32, "prefill_len": 128}
    first = build_prefill_trace_key(
        device_id=0,
        config=config,
        prefill_plan=plan,
        layer_count=32,
        batch_size=32,
        prefill_len=128,
        cache_len=1024,
        dtype_seed="bf16",
    )
    second = build_prefill_trace_key(
        device_id=0,
        config=config,
        prefill_plan=plan,
        layer_count=32,
        batch_size=32,
        prefill_len=128,
        cache_len=1024,
        dtype_seed="bf16",
    )
    changed = build_prefill_trace_key(
        device_id=0,
        config=config,
        prefill_plan=plan,
        layer_count=32,
        batch_size=32,
        prefill_len=256,
        cache_len=1024,
        dtype_seed="bf16",
    )

    assert first == second
    assert first.config_hash != changed.config_hash
    assert first.to_report()["prefill_len"] == 128


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "eager"), ("eager", "eager"), ("trace", "trace")],
)
def test_resolve_prefill_execution_mode(
    requested: str | None, expected: str
) -> None:
    assert resolve_prefill_execution_mode(requested) == expected


def test_decode_trace_captures_full_step_and_replays_nonblocking() -> None:
    device = _Device()
    ttnn = _TTNN(device)
    inputs = _Inputs()
    model = _Model(ttnn.calls)
    kv_cache = [object(), object()]
    session = DecodeTraceSession(
        ttnn=ttnn,
        device=device,
        model=model,
        persistent_inputs=inputs,
        kv_cache=kv_cache,
        key=_key(),
    )

    session.capture()
    result = session.execute()
    report = session.to_report()
    session.close()

    assert model.decode_count == 2
    assert inputs.materialize_count == 2
    assert result.token is model.output_token
    assert result.kv_cache is kv_cache
    assert result.cache_positions == [8, 9]
    assert result.runtime_state.positions == [9, 10]
    assert inputs.recorded_tokens == [model.output_token]
    assert inputs.positions == [9, 10]
    assert inputs.validation_count == 1
    assert report["execution_mode"] == "trace"
    assert report["trace_capture_count"] == 1
    assert report["trace_execute_count"] == 1
    assert report["compile_run_count"] == 1
    assert report["persistent_input_count"] == 7
    assert report["trace_input_update_count"] == 3
    assert report["program_compile_count_after_capture"] == 0
    assert report["program_compile_count_during_capture"] == 0
    assert ("execute", device, "trace-7", {"cq_id": 0, "blocking": False}) in ttnn.calls
    assert ttnn.calls[-1] == ("release", device, "trace-7")


def test_decode_trace_exports_graph_only_when_requested(tmp_path: Path) -> None:
    device = _Device()
    ttnn = _TTNN(device)
    ttnn.graph = _Graph(ttnn.calls)
    graph_path = tmp_path / "decode.json"
    session = DecodeTraceSession(
        ttnn=ttnn,
        device=device,
        model=_Model(ttnn.calls),
        persistent_inputs=_Inputs(),
        kv_cache=[object(), object()],
        key=_key(),
        graph_capture_path=graph_path,
    )

    session.capture()
    report = session.to_report()
    session.close()

    assert graph_path.is_file()
    assert report["execution_graph_path"] == str(graph_path)
    begin_graph = ttnn.calls.index(("begin_graph", "normal"))
    begin_trace = next(
        index for index, call in enumerate(ttnn.calls) if call[0] == "begin"
    )
    end_trace = next(index for index, call in enumerate(ttnn.calls) if call[0] == "end")
    end_graph = next(
        index for index, call in enumerate(ttnn.calls) if call[0] == "end_graph"
    )
    assert begin_graph < begin_trace < end_trace < end_graph


def test_decode_trace_key_is_stable_and_configuration_sensitive() -> None:
    config = {
        "template_config": {"dtype_recipe": "mixed"},
        "lm_head": {"argmax_strategy": "force"},
    }
    plan = {"kv_cache": {"page_block_size": 32}}
    first = build_decode_trace_key(
        device_id=0,
        config=config,
        decode_plan=plan,
        layer_count=32,
        batch_size=32,
        cache_len=1024,
        dtype_seed="bf16",
    )
    second = build_decode_trace_key(
        device_id=0,
        config=config,
        decode_plan=plan,
        layer_count=32,
        batch_size=32,
        cache_len=1024,
        dtype_seed="bf16",
    )
    changed = build_decode_trace_key(
        device_id=0,
        config=config,
        decode_plan=plan,
        layer_count=1,
        batch_size=32,
        cache_len=1024,
        dtype_seed="bf16",
    )

    assert first == second
    assert first.program_config_hash != changed.program_config_hash
    assert first.dtype_recipe == "mixed"
    assert first.argmax_strategy == "force"
    assert first.to_report()["page_block_size"] == 32


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "eager"), ("eager", "eager"), ("trace", "trace")],
)
def test_resolve_execution_mode(requested: str | None, expected: str) -> None:
    assert resolve_execution_mode(requested) == expected


def test_resolve_execution_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        resolve_execution_mode("unknown")


def test_trace_requires_complete_ttnn_api() -> None:
    with pytest.raises(RuntimeError, match="execute_trace"):
        DecodeTraceSession(
            ttnn=SimpleNamespace(
                begin_trace_capture=lambda *_args, **_kwargs: 1,
                end_trace_capture=lambda *_args, **_kwargs: None,
                copy=lambda *_args, **_kwargs: None,
                plus_one=lambda *_args, **_kwargs: None,
                clone=lambda value: value,
            ),
            device=object(),
            model=object(),
            persistent_inputs=object(),
            kv_cache=[],
            key=_key(),
        )


def test_device_session_applies_trace_region_only_when_requested() -> None:
    calls: list[dict[str, object]] = []
    ttnn = SimpleNamespace(
        open_device=lambda **kwargs: calls.append(kwargs) or object(),
        close_device=lambda _device: None,
    )

    with GenerateDeviceSession(
        ttnn,
        0,
        None,
        trace_region_size=52_000_000,
    ):
        pass
    with GenerateDeviceSession(ttnn, 1, None):
        pass

    assert calls == [
        {"device_id": 0, "trace_region_size": 52_000_000},
        {"device_id": 1},
    ]
