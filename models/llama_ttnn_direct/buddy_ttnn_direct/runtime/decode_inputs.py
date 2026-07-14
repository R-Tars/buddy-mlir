from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from .inputs import build_decode_runtime_state
from .rotary import (
    build_decode_rotary_cache_host_tensors,
    build_decode_rotary_host_tensors,
    decode_rotary_cos_sin_memory_config,
    decode_rotary_transform_memory_config,
    install_shared_rotary_parameters,
)
from .tensor_meta import runtime_int_tensor

RUNTIME_INPUT_MODES = ("recreate", "persistent")


class PersistentDecodeInputsUnsupported(RuntimeError):
    pass


class DecodeInputBuffers:
    """Persistent device inputs shared by all decode steps in one session."""

    def __init__(
        self,
        *,
        ttnn: Any,
        torch: Any,
        device: Any,
        dtype_seed: str,
        parameters: Any,
        decode_plan: dict[str, Any],
        batch_size: int,
        cache_len: int,
        prefill_effective_token_count: int | Sequence[int],
        token_input: Any,
    ) -> None:
        _require_persistent_apis(ttnn)
        self.ttnn = ttnn
        self.torch = torch
        self.device = device
        self.dtype_seed = dtype_seed
        self.parameters = parameters
        self.decode_plan = decode_plan
        self.batch_size = int(batch_size)
        self.cache_len = int(cache_len)
        self.layer_count = int(decode_plan["layers"])
        self.head_dim = int(
            decode_plan["layer_parameter_shapes"]["rotary_cos_matrix"][-1]
        )
        self.page_block_size = int(decode_plan["kv_cache"]["page_block_size"])
        self.positions = _initial_decode_positions(
            prefill_effective_token_count,
            batch_size=self.batch_size,
        )
        if any(position >= self.cache_len for position in self.positions):
            raise ValueError("initial decode position exceeds cache capacity")

        runtime_state = build_decode_runtime_state(
            batch_size=self.batch_size,
            cache_len=self.cache_len,
            page_block_size=self.page_block_size,
            prompt_token_count=[position + 1 for position in self.positions],
        )
        self._page_count = runtime_state.page_count
        self._max_num_blocks = runtime_state.max_num_blocks
        self.page_table = self._from_torch_int(
            runtime_int_tensor(
                torch,
                runtime_state.page_table,
                name="persistent_decode_page_table",
            ),
            dtype_name="int32",
        )
        self.cache_position = self._from_torch_int(
            runtime_int_tensor(
                torch,
                runtime_state.cache_position,
                name="persistent_decode_cache_position",
            ),
            dtype_name="int32",
        )
        self.rotary_index = self._from_torch_int(
            runtime_int_tensor(
                torch,
                [runtime_state.cache_position],
                name="persistent_decode_rotary_index",
            ),
            dtype_name="uint32",
        )

        rotary = dict(decode_plan.get("rotary") or {})
        theta = float(rotary.get("theta", 10000.0))
        scaling = rotary.get("scaling")
        cache_host = build_decode_rotary_cache_host_tensors(
            torch=torch,
            cache_len=self.cache_len,
            head_dim=self.head_dim,
            theta=theta,
            scaling=scaling,
            dtype_seed=dtype_seed,
        )
        tensor_kwargs = self._float_tensor_kwargs()
        self.rotary_cos_cache = ttnn.from_torch(cache_host.cos, **tensor_kwargs)
        self.rotary_sin_cache = ttnn.from_torch(cache_host.sin, **tensor_kwargs)
        first_host = build_decode_rotary_host_tensors(
            torch=torch,
            positions=self.positions,
            head_dim=self.head_dim,
            theta=theta,
            scaling=scaling,
            dtype_seed=dtype_seed,
        )
        transform_kwargs = self._float_tensor_kwargs()
        transform_memory = decode_rotary_transform_memory_config(
            ttnn,
            device,
            batch_size=self.batch_size,
        )
        if transform_memory is not None:
            transform_kwargs["memory_config"] = transform_memory
        self.rotary_transformation = ttnn.from_torch(
            first_host.transformation,
            **transform_kwargs,
        )
        self.rotary_cos = None
        self.rotary_sin = None
        self.token_input = token_input

        self.device_tensor_creation_count = 6
        self.host_update_count = 0
        self.page_table_update_count = 0
        self.cache_position_update_count = 0
        self.rotary_buffer_update_count = 0
        self.token_device_copy_count = 0
        self.step_count = 0
        self._materialize_rotary()

    @classmethod
    def supported(cls, ttnn: Any) -> bool:
        return not _missing_persistent_apis(ttnn)

    def initial_runtime_state(self) -> SimpleNamespace:
        return self._runtime_state(tensor_conversion_count=6)

    def record_token(self, token_input: Any) -> None:
        self.token_input = token_input
        self.step_count += 1

    def advance(self) -> SimpleNamespace:
        if any(position + 1 >= self.cache_len for position in self.positions):
            raise ValueError("decode input update exceeds cache capacity")
        self.ttnn.plus_one(
            self.cache_position,
            skip_negative_entries=True,
        )
        self.ttnn.plus_one(self.rotary_index)
        self.positions = [position + 1 for position in self.positions]
        self.cache_position_update_count += 1
        self._materialize_rotary()
        self.rotary_buffer_update_count += 1
        return self._runtime_state(tensor_conversion_count=0)

    def to_report(self) -> dict[str, Any]:
        return {
            "execution_mode": "eager",
            "runtime_input_mode": "persistent",
            "new_device_tensors_per_decode_step": 0,
            "host_to_device_updates_per_decode_step": 0,
            "page_table_update_count": self.page_table_update_count,
            "cache_position_update_count": self.cache_position_update_count,
            "rotary_buffer_update_count": self.rotary_buffer_update_count,
            "token_device_copy_count": self.token_device_copy_count,
            "persistent_input_count": self.device_tensor_creation_count,
            "initial_device_tensor_creation_count": (
                self.device_tensor_creation_count
            ),
            "host_update_count": self.host_update_count,
            "decode_step_count": self.step_count,
            "page_table_reused": True,
            "cache_position_update": "ttnn.plus_one_in_place",
            "rotary_update": "device_embedding_from_full_cos_sin_cache",
            "token_update": "device_tensor_direct_handoff",
        }

    def _runtime_state(
        self, *, tensor_conversion_count: int
    ) -> SimpleNamespace:
        return SimpleNamespace(
            page_table=self.page_table,
            cache_position=self.cache_position,
            decode_runtime_state=self._decode_runtime_report(),
            rotary_runtime_state=self._rotary_runtime_report(),
            tensor_conversion_count=tensor_conversion_count,
            decode_runtime_state_input_tensor_count=(
                3 if tensor_conversion_count else 0
            ),
            rotary_runtime_input_tensor_count=(
                3 if tensor_conversion_count else 0
            ),
        )

    def _decode_runtime_report(self) -> dict[str, Any]:
        uniform = self.positions[0] if len(set(self.positions)) == 1 else None
        return {
            "status": "built",
            "source": "persistent_decode_input_buffers",
            "runtime_input_mode": "persistent",
            "batch_size": self.batch_size,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "page_count": self._page_count,
            "max_num_blocks": self._max_num_blocks,
            "cache_position_value": uniform,
            "cache_position_values": list(self.positions),
            "page_table_shape": [self.batch_size, self._page_count],
            "cache_position_shape": [self.batch_size],
            "rotary_index_shape": [1, self.batch_size],
            "memory_config": "dram",
        }

    def _rotary_runtime_report(self) -> dict[str, Any]:
        rotary = dict(self.decode_plan.get("rotary") or {})
        uniform = self.positions[0] if len(set(self.positions)) == 1 else None
        return {
            "status": "built",
            "source": "persistent_device_rotary_cache",
            "mode": "decode",
            "runtime_input_mode": "persistent",
            "theta": float(rotary.get("theta", 10000.0)),
            "scaling": rotary.get("scaling"),
            "cache_position_value": uniform,
            "cache_position_values": list(self.positions),
            "positions": list(self.positions),
            "cos_sin_shape": [1, self.batch_size, 1, self.head_dim],
            "cos_sin_cache_shape": [self.cache_len, self.head_dim],
            "transformation_shape": [
                1,
                1,
                self.batch_size * 32,
                32,
            ],
            "tensor_count": 3,
            "shared_across_layers": True,
            "position_update": "ttnn.plus_one_in_place",
        }

    def _materialize_rotary(self) -> None:
        memory_config = decode_rotary_cos_sin_memory_config(
            self.ttnn,
            self.device,
            batch_size=self.batch_size,
            head_dim=self.head_dim,
        )
        embedding_kwargs = {
            "layout": getattr(self.ttnn, "TILE_LAYOUT", None),
            "memory_config": getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None),
        }
        embedding_kwargs = {
            name: value
            for name, value in embedding_kwargs.items()
            if value is not None
        }
        cos = self.ttnn.embedding(
            self.rotary_index,
            self.rotary_cos_cache,
            **embedding_kwargs,
        )
        sin = self.ttnn.embedding(
            self.rotary_index,
            self.rotary_sin_cache,
            **embedding_kwargs,
        )
        cos = self.ttnn.unsqueeze_to_4D(cos)
        sin = self.ttnn.unsqueeze_to_4D(sin)
        cos = self.ttnn.transpose(cos, 1, 2)
        sin = self.ttnn.transpose(sin, 1, 2)
        if memory_config is not None:
            cos = self.ttnn.interleaved_to_sharded(cos, memory_config)
            sin = self.ttnn.interleaved_to_sharded(sin, memory_config)
        self.rotary_cos = cos
        self.rotary_sin = sin
        install_shared_rotary_parameters(
            parameters=self.parameters,
            cos_matrix=cos,
            sin_matrix=sin,
            transformation_matrix=self.rotary_transformation,
            layer_count=self.layer_count,
        )

    def _from_torch_int(self, tensor: Any, *, dtype_name: str) -> Any:
        kwargs = {
            "device": self.device,
            "dtype": getattr(self.ttnn, dtype_name, None),
            "layout": getattr(self.ttnn, "ROW_MAJOR_LAYOUT", None),
            "memory_config": getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None),
        }
        return self.ttnn.from_torch(
            tensor,
            **{
                name: value
                for name, value in kwargs.items()
                if value is not None
            },
        )

    def _float_tensor_kwargs(self) -> dict[str, Any]:
        dtype_name = "bfloat16" if self.dtype_seed == "bf16" else "float32"
        kwargs = {
            "device": self.device,
            "dtype": getattr(self.ttnn, dtype_name, None),
            "layout": getattr(self.ttnn, "TILE_LAYOUT", None),
            "memory_config": getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None),
        }
        return {
            name: value for name, value in kwargs.items() if value is not None
        }


def resolve_runtime_input_mode(
    requested: str | None,
    *,
    config: dict[str, Any] | None = None,
) -> str:
    config = config or {}
    template_value = config.get("template_config")
    template_config = template_value if isinstance(template_value, dict) else {}
    mode = requested or str(
        config.get("runtime_input_mode")
        or template_config.get("runtime_input_mode")
        or "persistent"
    )
    if mode not in RUNTIME_INPUT_MODES:
        raise ValueError(
            "runtime_input_mode must be one of: "
            + ", ".join(RUNTIME_INPUT_MODES)
        )
    return mode


def _initial_decode_positions(
    value: int | Sequence[int], *, batch_size: int
) -> list[int]:
    if isinstance(value, int):
        positions = [int(value)] * batch_size
    else:
        positions = [int(position) for position in value]
    if len(positions) != batch_size:
        raise ValueError(
            "prefill effective token count length must match batch size"
        )
    if any(position < 0 for position in positions):
        raise ValueError("decode positions must be non-negative")
    return positions


def _missing_persistent_apis(ttnn: Any) -> list[str]:
    return [
        name
        for name in (
            "from_torch",
            "plus_one",
            "embedding",
            "unsqueeze_to_4D",
            "transpose",
            "interleaved_to_sharded",
        )
        if not callable(getattr(ttnn, name, None))
    ]


def _require_persistent_apis(ttnn: Any) -> None:
    missing = _missing_persistent_apis(ttnn)
    if missing:
        raise PersistentDecodeInputsUnsupported(
            "persistent decode inputs require TTNN APIs: " + ", ".join(missing)
        )
