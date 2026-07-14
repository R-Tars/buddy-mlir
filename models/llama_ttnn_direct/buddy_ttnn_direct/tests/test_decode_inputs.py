from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.decode_inputs import (
    DecodeInputBuffers,
    resolve_runtime_input_mode,
)


class _HostTensor:
    def __init__(self, values: object, dtype: object | None = None) -> None:
        self.values = values
        self.dtype = dtype
        self.shape = self._shape(values)
        self.name: str | None = None

    @classmethod
    def _shape(cls, values: object) -> tuple[int, ...]:
        if not isinstance(values, (list, range)):
            return ()
        values = list(values)
        if not values:
            return (0,)
        return (len(values), *cls._shape(values[0]))


class _Torch:
    int32 = "torch.int32"
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"

    @staticmethod
    def tensor(values: object, dtype: object | None = None) -> _HostTensor:
        return _HostTensor(values, dtype)


class _TTNN:
    int32 = "ttnn.int32"
    uint32 = "ttnn.uint32"
    bfloat16 = "ttnn.bfloat16"
    float32 = "ttnn.float32"
    ROW_MAJOR_LAYOUT = "row_major"
    TILE_LAYOUT = "tile"
    DRAM_MEMORY_CONFIG = "dram"
    L1_HEIGHT_SHARDED_MEMORY_CONFIG = "l1_height_sharded"

    def __init__(self) -> None:
        self.from_torch_calls: list[SimpleNamespace] = []
        self.plus_one_calls: list[tuple[SimpleNamespace, dict[str, object]]] = (
            []
        )

    def from_torch(
        self, tensor: _HostTensor, **kwargs: object
    ) -> SimpleNamespace:
        result = SimpleNamespace(
            source=tensor.name,
            shape=list(tensor.shape),
            values=tensor.values,
            kwargs=kwargs,
        )
        self.from_torch_calls.append(result)
        return result

    def plus_one(self, tensor: SimpleNamespace, **kwargs: object) -> None:
        self.plus_one_calls.append((tensor, kwargs))

    @staticmethod
    def embedding(
        index: SimpleNamespace,
        cache: SimpleNamespace,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(shape=[*index.shape, cache.shape[-1]])

    @staticmethod
    def unsqueeze_to_4D(tensor: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(shape=[1, *tensor.shape])

    @staticmethod
    def transpose(
        tensor: SimpleNamespace,
        dim0: int,
        dim1: int,
    ) -> SimpleNamespace:
        shape = list(tensor.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return SimpleNamespace(shape=shape)

    @staticmethod
    def interleaved_to_sharded(
        tensor: SimpleNamespace,
        memory_config: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            shape=tensor.shape,
            memory_config=memory_config,
        )


def _decode_plan() -> dict[str, object]:
    return {
        "layers": 2,
        "kv_cache": {"page_block_size": 32},
        "layer_parameter_shapes": {
            "rotary_cos_matrix": [1, 4, 1, 128],
        },
        "rotary": {"theta": 500000.0, "scaling": None},
    }


def _parameters() -> SimpleNamespace:
    return SimpleNamespace(
        layers=[SimpleNamespace(), SimpleNamespace()],
    )


def test_persistent_decode_inputs_allocate_once_and_reuse_page_table() -> None:
    ttnn = _TTNN()
    parameters = _parameters()
    buffers = DecodeInputBuffers(
        ttnn=ttnn,
        torch=_Torch(),
        device=object(),
        dtype_seed="bf16",
        parameters=parameters,
        decode_plan=_decode_plan(),
        batch_size=4,
        cache_len=64,
        prefill_effective_token_count=[8, 9, 10, 11],
        token_input=SimpleNamespace(shape=[4, 1]),
    )

    initial = buffers.initial_runtime_state()
    page_table = initial.page_table
    assert len(ttnn.from_torch_calls) == 6
    assert initial.tensor_conversion_count == 6
    assert initial.decode_runtime_state["cache_position_values"] == [
        8,
        9,
        10,
        11,
    ]

    buffers.record_token(SimpleNamespace(shape=[4, 1]))
    advanced = buffers.advance()

    assert advanced.page_table is page_table
    assert advanced.tensor_conversion_count == 0
    assert len(ttnn.from_torch_calls) == 6
    assert advanced.decode_runtime_state["cache_position_values"] == [
        9,
        10,
        11,
        12,
    ]
    assert len(ttnn.plus_one_calls) == 2
    assert ttnn.plus_one_calls[0][1] == {"skip_negative_entries": True}
    assert (
        parameters.layers[0].attention.rotary.cos_matrix is buffers.rotary_cos
    )
    assert (
        parameters.layers[1].attention.rotary.sin_matrix is buffers.rotary_sin
    )

    report = buffers.to_report()
    assert report["runtime_input_mode"] == "persistent"
    assert report["new_device_tensors_per_decode_step"] == 0
    assert report["host_to_device_updates_per_decode_step"] == 0
    assert report["page_table_update_count"] == 0
    assert report["cache_position_update_count"] == 1
    assert report["rotary_buffer_update_count"] == 1
    assert report["token_device_copy_count"] == 0


def test_trace_uses_dedicated_token_input_buffer() -> None:
    ttnn = _TTNN()
    ttnn.clone = lambda tensor: SimpleNamespace(
        shape=list(tensor.shape),
        cloned_from=tensor,
    )
    token = SimpleNamespace(shape=[4, 1])
    buffers = DecodeInputBuffers(
        ttnn=ttnn,
        torch=_Torch(),
        device=object(),
        dtype_seed="bf16",
        parameters=_parameters(),
        decode_plan=_decode_plan(),
        batch_size=4,
        cache_len=64,
        prefill_effective_token_count=[8, 9, 10, 11],
        token_input=token,
    )

    buffers.prepare_token_for_trace()
    buffers.prepare_token_for_trace()

    assert buffers.token_input is not token
    assert buffers.token_input.cloned_from is token
    assert buffers.device_tensor_creation_count == 7
    assert buffers.trace_token_input_creation_count == 1


def test_trace_capacity_allows_last_slot_then_rejects_next_step() -> None:
    ttnn = _TTNN()
    buffers = DecodeInputBuffers(
        ttnn=ttnn,
        torch=_Torch(),
        device=object(),
        dtype_seed="bf16",
        parameters=_parameters(),
        decode_plan=_decode_plan(),
        batch_size=4,
        cache_len=64,
        prefill_effective_token_count=[63, 63, 63, 63],
        token_input=SimpleNamespace(shape=[4, 1]),
    )

    buffers.record_trace_execution(SimpleNamespace(shape=[4, 1]))

    assert buffers.positions == [64, 64, 64, 64]
    with pytest.raises(ValueError, match="cache capacity"):
        buffers.validate_trace_execution()


@pytest.mark.parametrize(
    ("requested", "config", "expected"),
    [
        ("recreate", {"runtime_input_mode": "persistent"}, "recreate"),
        (None, {"runtime_input_mode": "recreate"}, "recreate"),
        (
            None,
            {"template_config": {"runtime_input_mode": "recreate"}},
            "recreate",
        ),
        (None, {}, "persistent"),
    ],
)
def test_resolve_runtime_input_mode(
    requested: str | None,
    config: dict[str, object],
    expected: str,
) -> None:
    assert resolve_runtime_input_mode(requested, config=config) == expected


def test_resolve_runtime_input_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="runtime_input_mode"):
        resolve_runtime_input_mode("unknown")
