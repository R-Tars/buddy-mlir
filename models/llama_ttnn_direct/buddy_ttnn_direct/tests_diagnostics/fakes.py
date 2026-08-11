"""Shared fakes for diagnostics and product integration tests.

These fixtures deliberately live outside individual test modules.  Keeping the
fake TTNN surface in one place prevents diagnostic tests from becoming import
dependencies of product tests or of one another.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np


class FakeTorchTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str | None = None,
        name: str = "torch_tensor",
        values: Any | None = None,
    ) -> None:
        self.shape = list(shape)
        self.dtype = dtype
        self.name = name
        self.values = values


class FakeTTNNTensor:
    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str | None = "ttnn.bfloat16",
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: list[int] | None = None,
        dtype: str = "ttnn.bfloat16",
        mem_config: str | None = None,
        torch_value: Any | None = None,
    ) -> None:
        self.name = name
        self.torch_value = torch_value
        self.shape = (
            list(torch_value.shape)
            if torch_value is not None
            else list(shape or [32, 1, 16])
        )
        self.dtype = str(torch_value.dtype) if torch_value is not None else dtype
        self._mem_config = mem_config

    def memory_config(self) -> str | None:
        return self._mem_config

    def __repr__(self) -> str:
        return f"FakeTensor({self.name})"


class MiniTensor:
    def __init__(self, values: Any, dtype: str = "float32") -> None:
        np_dtype = (
            np.int64
            if dtype.startswith("int") or dtype.startswith("uint")
            else np.float64
        )
        self.array = np.array(values, dtype=np_dtype)
        self.dtype = dtype
        self.shape = list(self.array.shape)

    def transpose(self, dim0: int, dim1: int) -> "MiniTensor":
        return MiniTensor(np.swapaxes(self.array, dim0, dim1), self.dtype)

    def mean(self, *, dim: int, keepdim: bool) -> "MiniTensor":
        return MiniTensor(
            np.mean(self.array, axis=dim, keepdims=keepdim),
            self.dtype,
        )

    def reshape(self, *shape: int) -> "MiniTensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return MiniTensor(np.reshape(self.array, shape), self.dtype)

    def tolist(self) -> list[Any]:
        return self.array.tolist()

    def detach(self) -> "MiniTensor":
        return self

    def cpu(self) -> "MiniTensor":
        return self

    def __matmul__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array @ _mini_array(other), self.dtype)

    def __add__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array + _mini_array(other), self.dtype)

    def __radd__(self, other: Any) -> "MiniTensor":
        return self.__add__(other)

    def __mul__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array * _mini_array(other), self.dtype)

    def __rmul__(self, other: Any) -> "MiniTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array / _mini_array(other), self.dtype)

    def __rtruediv__(self, other: Any) -> "MiniTensor":
        return MiniTensor(_mini_array(other) / self.array, self.dtype)

    def __neg__(self) -> "MiniTensor":
        return MiniTensor(-self.array, self.dtype)


def _nested_shape(values: Any) -> list[int]:
    shape: list[int] = []
    current = values
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def _mini_array(value: Any) -> np.ndarray:
    if isinstance(value, MiniTensor):
        return value.array
    if isinstance(value, FakeTensor) and value.torch_value is not None:
        return value.torch_value.array
    return np.array(value)


def _has_numeric(value: Any) -> bool:
    return isinstance(value, MiniTensor) or (
        isinstance(value, FakeTensor) and value.torch_value is not None
    )


def _to_mini(value: Any) -> MiniTensor:
    if isinstance(value, MiniTensor):
        return value
    if isinstance(value, FakeTensor) and value.torch_value is not None:
        return value.torch_value
    raise TypeError(f"expected numeric fake tensor, got {type(value).__name__}")


def _fake_torch() -> Any:
    module = types.SimpleNamespace()
    module.bfloat16 = "torch.bfloat16"
    module.float32 = "torch.float32"
    module.int32 = "torch.int32"
    module.int64 = "torch.int64"

    def randn(shape, dtype=None):
        return FakeTorchTensor(tuple(shape), dtype=dtype, name="randn")

    def zeros(shape, dtype=None):
        return FakeTorchTensor(tuple(shape), dtype=dtype, name="zeros")

    def tensor(values, dtype=None):
        return FakeTorchTensor(
            tuple(_nested_shape(values)),
            dtype=dtype,
            name="tensor",
            values=values,
        )

    module.randn = randn
    module.zeros = zeros
    module.tensor = tensor
    return module


def _fake_numeric_torch() -> Any:
    module = types.SimpleNamespace()
    module.float32 = "float32"
    module.int64 = "int64"

    def embedding(token_ids: MiniTensor, weight: MiniTensor) -> MiniTensor:
        return MiniTensor(weight.array[token_ids.array.astype(np.int64)])

    def silu(tensor: MiniTensor) -> MiniTensor:
        return MiniTensor(tensor.array / (1.0 + np.exp(-tensor.array)))

    module.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(embedding=embedding, silu=silu)
    )
    module.rsqrt = lambda tensor: MiniTensor(1.0 / np.sqrt(tensor.array))
    module.cat = lambda tensors, dim: MiniTensor(
        np.concatenate([tensor.array for tensor in tensors], axis=dim)
    )
    module.argmax = lambda tensor, dim: MiniTensor(
        np.argmax(tensor.array, axis=dim), dtype="int64"
    )
    return module


def _torch_embedding(torch: Any, token_ids: MiniTensor, weight: MiniTensor) -> MiniTensor:
    return torch.nn.functional.embedding(token_ids, weight)


def _torch_rms_norm(
    torch: Any,
    hidden: MiniTensor,
    weight: MiniTensor,
    epsilon: float,
) -> MiniTensor:
    values = hidden.array * (1.0 / np.sqrt(np.mean(hidden.array**2, axis=-1, keepdims=True) + epsilon))
    return MiniTensor(values * weight.array, hidden.dtype)


def _torch_linear(torch: Any, activation: MiniTensor, weight: MiniTensor) -> MiniTensor:
    values = activation.array @ np.swapaxes(weight.array, -1, -2)
    return MiniTensor(values, activation.dtype)


def _fake_ttnn(
    *,
    with_transformer: bool = True,
    instrumented: bool = True,
    with_create_sharded: bool = False,
) -> Any:
    module = types.SimpleNamespace()
    module.calls = [] if instrumented else _CallSink()
    module.__version__ = "fake-ttnn"
    module.__tt_metal_commit__ = "fake-tt-metal"
    module.bfloat16 = "ttnn.bfloat16"
    module.float32 = "ttnn.float32"
    module.int32 = "ttnn.int32"
    module.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"
    module.L1_MEMORY_CONFIG = "ttnn.L1_MEMORY_CONFIG"
    module.L1_HEIGHT_SHARDED_MEMORY_CONFIG = "ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG"
    module.DRAM_MEMORY_CONFIG = "ttnn.DRAM_MEMORY_CONFIG"
    module.TILE_SIZE = 32

    if with_create_sharded:
        module.ShardStrategy = types.SimpleNamespace(HEIGHT="HEIGHT")
        module.ShardOrientation = types.SimpleNamespace(ROW_MAJOR="ROW_MAJOR")

        class CoreGrid:
            def __init__(self, y, x):
                self.y = y
                self.x = x

        module.CoreGrid = CoreGrid

        def create_sharded_memory_config(**kwargs):
            module.calls.append({"op": "create_sharded_memory_config", "kwargs": dict(kwargs)})
            return f"sharded:{tuple(kwargs['shape'])}"

        module.create_sharded_memory_config = create_sharded_memory_config

    def record(op: str, **payload: Any) -> None:
        module.calls.append({"op": op, **payload})

    def from_torch(tensor, **kwargs):
        record("from_torch", name=getattr(tensor, "name", "torch_tensor"), shape=list(tensor.shape), kwargs=dict(kwargs))
        return FakeTTNNTensor(getattr(tensor, "name", "torch_tensor"), list(tensor.shape), dtype=str(kwargs.get("dtype")))

    def embedding(token_ids, weight, **kwargs):
        token_shape = list(getattr(token_ids, "shape", [1, 1]))
        weight_shape = list(getattr(weight, "shape", [1, 1, 16]))
        record("embedding", token_ids=getattr(token_ids, "name", token_ids), weight=getattr(weight, "name", weight), kwargs=dict(kwargs))
        return FakeTTNNTensor("embedding", [token_shape[0], token_shape[1], weight_shape[-1]])

    def linear(activation, weight, **kwargs):
        shape = list(activation.shape)
        shape[-1] = getattr(weight, "shape", [shape[-1]])[-1]
        record("linear", activation=getattr(activation, "name", activation), weight=getattr(weight, "name", weight), kwargs=dict(kwargs))
        return FakeTTNNTensor("linear", shape)

    def qkv_heads(fused_qkv, **kwargs):
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        physical = len(fused_qkv.shape) >= 4
        batch = fused_qkv.shape[-2] if physical else fused_qkv.shape[0]
        record("nlp_create_qkv_heads_decode", qkv=fused_qkv.name, kwargs=dict(kwargs))
        shapes = (
            [[1, batch, num_heads, head_dim], [1, batch, num_kv_heads, head_dim], [1, batch, num_kv_heads, head_dim]]
            if physical
            else [[batch, num_heads, 1, head_dim], [batch, num_kv_heads, 1, head_dim], [batch, num_kv_heads, 1, head_dim]]
        )
        return tuple(FakeTTNNTensor(name, shape) for name, shape in zip(("query", "key", "value"), shapes))

    def qkv_heads_prefill(fused_qkv, **kwargs):
        batch, seq_len = fused_qkv.shape[0], fused_qkv.shape[1]
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        record("split_query_key_value_and_split_heads", qkv=fused_qkv.name, kwargs=dict(kwargs))
        return (
            FakeTensor("query", [batch, num_heads, seq_len, head_dim]),
            FakeTensor("key", [batch, num_kv_heads, seq_len, head_dim]),
            FakeTensor("value", [batch, num_kv_heads, seq_len, head_dim]),
        )

    def rotary(tensor, cos, sin, transform, **kwargs):
        record("rotary_embedding_llama", tensor=tensor.name, kwargs=dict(kwargs))
        return type(tensor)(f"rotary:{tensor.name}", list(tensor.shape))

    def fused_rotary(q, k, cos, sin, transform, **kwargs):
        record("rotary_embedding_llama_fused_qk", q=q.name, kwargs=dict(kwargs))
        return q, k

    def sdpa(query, key, value, **kwargs):
        record("scaled_dot_product_attention", query=query.name, kwargs=dict(kwargs))
        return type(query)("attention", list(query.shape))

    def paged_sdpa(query, key_cache, value_cache, **kwargs):
        record("paged_scaled_dot_product_attention_decode", query=query.name, kwargs=dict(kwargs))
        return type(query)("attention", list(query.shape))

    def update_cache(cache, update, **kwargs):
        record("paged_update_cache", cache=getattr(cache, "name", cache), update=getattr(update, "name", update), kwargs=dict(kwargs))
        return type(cache)("cache", list(cache.shape))

    def fill_cache(cache, update, **kwargs):
        record("fill_cache", cache=getattr(cache, "name", cache), update=getattr(update, "name", update), kwargs=dict(kwargs))
        return type(cache)(getattr(cache, "name", "cache"), list(cache.shape))

    def paged_fill_cache(cache, update, page_table, **kwargs):
        record("paged_fill_cache", cache=getattr(cache, "name", cache), update=getattr(update, "name", update), kwargs=dict(kwargs))
        return type(cache)(getattr(cache, "name", "cache"), list(cache.shape))

    def concat_heads(attention, **kwargs):
        batch = attention.shape[1] if len(attention.shape) >= 4 else attention.shape[0]
        heads = int(kwargs.get("num_heads", 4))
        head_dim = attention.shape[-1]
        record("nlp_concat_heads_decode", attention=attention.name, kwargs=dict(kwargs))
        shape = [1, 1, batch, heads * head_dim] if len(attention.shape) >= 4 else [batch, 1, heads * head_dim]
        return type(attention)("concat_heads", shape)

    def concat_heads_prefill(attention, **kwargs):
        batch, _, seq_len, head_dim = attention.shape
        record("concatenate_heads", attention=attention.name, kwargs=dict(kwargs))
        return type(attention)("concat_heads", [batch, seq_len, 4 * head_dim])

    def rms_norm(hidden, **kwargs):
        record("rms_norm", hidden=getattr(hidden, "name", hidden), kwargs=dict(kwargs))
        return type(hidden)("rms_norm", list(hidden.shape))

    def reshape(tensor, logical_shape, padded_shape=None):
        record("reshape", tensor=tensor.name, logical_shape=list(logical_shape), padded_shape=padded_shape)
        return type(tensor)(f"reshape:{tensor.name}", list(logical_shape))

    def slice_tensor(tensor, starts, ends, steps=None):
        shape = [int(end) - int(start) for start, end in zip(starts, ends)]
        record("slice", tensor=tensor.name, starts=list(starts), ends=list(ends))
        return type(tensor)(f"slice:{tensor.name}", shape)

    def to_memory_config(tensor, **kwargs):
        record("to_memory_config", tensor=getattr(tensor, "name", tensor), kwargs=dict(kwargs))
        return type(tensor)(f"mem:{getattr(tensor, 'name', tensor)}", list(tensor.shape))

    def mul(lhs, rhs, **kwargs):
        record("mul", lhs=getattr(lhs, "name", lhs), rhs=getattr(rhs, "name", rhs), kwargs=dict(kwargs))
        return type(lhs)("mul", list(lhs.shape))

    def add(lhs, rhs, **kwargs):
        record("add", lhs=getattr(lhs, "name", lhs), rhs=getattr(rhs, "name", rhs), kwargs=dict(kwargs))
        return type(lhs)("add", list(lhs.shape))

    def concat(tensors, **kwargs):
        shape = list(tensors[0].shape)
        dim = int(kwargs.get("dim", -1))
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        record("concat", kwargs=dict(kwargs))
        return type(tensors[0])("concat", shape)

    def split(tensor, split_size, dim=-1, **kwargs):
        shape = list(tensor.shape)
        shape[dim] = int(split_size)
        record("split", tensor=tensor.name, split_size=int(split_size), dim=int(dim), kwargs=dict(kwargs))
        return type(tensor)("split:0", shape), type(tensor)("split:1", shape)

    def argmax(tensor, **kwargs):
        shape = list(tensor.shape)
        dim = int(kwargs.get("dim", -1))
        if dim < 0:
            dim += len(shape)
        del shape[dim]
        record("argmax", tensor=getattr(tensor, "name", tensor), kwargs=dict(kwargs))
        return type(tensor)("argmax", shape)

    def untilize(tensor, **kwargs):
        record("untilize", tensor=getattr(tensor, "name", tensor), kwargs=dict(kwargs))
        return type(tensor)(f"untilize:{getattr(tensor, 'name', tensor)}", list(tensor.shape))

    def topk(tensor, **kwargs):
        shape = list(tensor.shape)
        shape[int(kwargs.get("dim", -1))] = int(kwargs["k"])
        record("topk", tensor=getattr(tensor, "name", tensor), kwargs=dict(kwargs))
        return type(tensor)("topk_values", shape), type(tensor)("topk_indices", shape, dtype="ttnn.uint16")

    def typecast(tensor, dtype):
        record("typecast", tensor=getattr(tensor, "name", tensor), dtype=dtype)
        return type(tensor)(f"typecast:{getattr(tensor, 'name', tensor)}", list(tensor.shape), dtype=str(dtype))

    def gather(tensor, dim, index):
        record("gather", tensor=getattr(tensor, "name", tensor), dim=dim, index=getattr(index, "name", index))
        return type(tensor)("gather", list(index.shape), dtype=getattr(tensor, "dtype", "ttnn.uint32"))

    def to_torch(tensor):
        record("to_torch", tensor=getattr(tensor, "name", tensor))
        return getattr(tensor, "torch_value", None)

    def clone(tensor):
        record("clone", tensor=getattr(tensor, "name", tensor))
        return type(tensor)(f"clone:{getattr(tensor, 'name', tensor)}", list(tensor.shape))

    def copy(source, target):
        record("copy", source=getattr(source, "name", source), target=getattr(target, "name", target))

    def plus_one(tensor, **kwargs):
        record("plus_one", tensor=getattr(tensor, "name", tensor), kwargs=dict(kwargs))

    def unsqueeze_to_4d(tensor):
        shape = [1] * (4 - len(tensor.shape)) + list(tensor.shape)
        record("unsqueeze_to_4D", tensor=getattr(tensor, "name", tensor))
        return type(tensor)(f"unsqueeze:{getattr(tensor, 'name', tensor)}", shape)

    def transpose(tensor, dim0, dim1):
        shape = list(tensor.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        record("transpose", tensor=getattr(tensor, "name", tensor), dim0=dim0, dim1=dim1)
        return type(tensor)(f"transpose:{getattr(tensor, 'name', tensor)}", shape)

    def interleaved_to_sharded(tensor, memory_config):
        record("interleaved_to_sharded", tensor=getattr(tensor, "name", tensor))
        return type(tensor)(f"sharded:{getattr(tensor, 'name', tensor)}", list(tensor.shape))

    def begin_trace_capture(device, **kwargs):
        record("begin_trace_capture", device=device, kwargs=dict(kwargs))
        return "fake_trace_id"

    def end_trace_capture(device, trace_id, **kwargs):
        record("end_trace_capture", device=device, trace_id=trace_id, kwargs=dict(kwargs))

    def execute_trace(device, trace_id, **kwargs):
        record("execute_trace", device=device, trace_id=trace_id, kwargs=dict(kwargs))

    def release_trace(device, trace_id):
        record("release_trace", device=device, trace_id=trace_id)

    module.UnaryOpType = types.SimpleNamespace(SILU="SILU")
    module.UnaryWithParam = lambda op: ("UnaryWithParam", op)
    module.from_torch = from_torch
    module.linear = linear
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.reshape = reshape
    module.slice = slice_tensor
    module.to_memory_config = to_memory_config
    module.mul = mul
    module.add = add
    module.concat = concat
    module.split = split
    module.argmax = argmax
    module.untilize = untilize
    module.topk = topk
    module.typecast = typecast
    module.gather = gather
    module.to_torch = to_torch
    module.clone = clone
    module.copy = copy
    module.plus_one = plus_one
    module.unsqueeze_to_4D = unsqueeze_to_4d
    module.transpose = transpose
    module.interleaved_to_sharded = interleaved_to_sharded
    module.begin_trace_capture = begin_trace_capture
    module.end_trace_capture = end_trace_capture
    module.execute_trace = execute_trace
    module.release_trace = release_trace
    module.synchronize_device = lambda device: record("synchronize_device", device=device)
    module.experimental = types.SimpleNamespace(
        nlp_create_qkv_heads_decode=qkv_heads,
        rotary_embedding_llama=rotary,
        rotary_embedding_llama_fused_qk=fused_rotary,
        paged_update_cache=update_cache,
        paged_fused_update_cache=lambda *args, **kwargs: update_cache(args[0], args[1], **kwargs),
        paged_fill_cache=paged_fill_cache,
        nlp_concat_heads_decode=concat_heads,
    )
    module.transformer = types.SimpleNamespace(
        split_query_key_value_and_split_heads=qkv_heads_prefill,
        scaled_dot_product_attention=sdpa,
        paged_scaled_dot_product_attention_decode=paged_sdpa,
        concatenate_heads=concat_heads_prefill,
    ) if with_transformer else types.SimpleNamespace()
    module.kv_cache = types.SimpleNamespace(fill_cache_for_user_=fill_cache)
    return module


class _CallSink:
    def append(self, call: Any) -> None:
        return None


def _make_generate_fake_ttnn(**kwargs: Any) -> Any:
    module = _make_fake_ttnn(**kwargs)
    # Preserve the historical generate fake's regenerated-input fallback.
    module.plus_one = None

    def to_torch(tensor: Any) -> Any:
        module.calls.append({"op": "to_torch", "tensor": tensor.name})
        shape = list(tensor.shape)
        if len(shape) == 2 and shape[1] > 1:
            return [[17 for _ in range(shape[1])] for _ in range(shape[0])]
        if len(shape) == 2:
            return [[23] for _ in range(shape[0])]
        if len(shape) == 1:
            return [23 for _ in range(shape[0])]
        return [[23], [23]]

    module.to_torch = to_torch
    return module


def _make_fake_ttnn(**kwargs: Any) -> Any:
    return _fake_ttnn(**kwargs)


def _make_numeric_ttnn() -> Any:
    module = types.SimpleNamespace(
        calls=[],
        __version__="fake-numeric-ttnn",
        bfloat16="ttnn.bfloat16",
        float32="ttnn.float32",
        uint32="ttnn.uint32",
        TILE_LAYOUT="ttnn.TILE_LAYOUT",
        ROW_MAJOR_LAYOUT="ttnn.ROW_MAJOR_LAYOUT",
        L1_MEMORY_CONFIG="ttnn.L1_MEMORY_CONFIG",
    )
    module.UnaryOpType = types.SimpleNamespace(SILU="SILU")
    module.UnaryWithParam = lambda op: ("UnaryWithParam", op)

    def record(op: str, **payload: Any) -> None:
        module.calls.append({"op": op, **payload})

    def result(name: str, value: MiniTensor, **kwargs: Any) -> FakeTensor:
        return FakeTensor(
            name,
            dtype=kwargs.get("dtype", value.dtype),
            mem_config=kwargs.get("memory_config"),
            torch_value=value,
        )

    def from_torch(tensor: Any, **kwargs: Any) -> FakeTensor:
        record("from_torch", kwargs=dict(kwargs))
        value = tensor if isinstance(tensor, MiniTensor) else MiniTensor(tensor)
        return result("from_torch", value, **kwargs)

    def embedding(token_ids: Any, weight: Any, **kwargs: Any) -> FakeTensor:
        record("embedding", kwargs=dict(kwargs))
        weight_value = _to_mini(weight)
        if len(weight_value.shape) == 4:
            weight_value = weight_value.reshape(*weight_value.shape[-2:])
        value = _torch_embedding(
            _fake_numeric_torch(), _to_mini(token_ids), weight_value
        )
        return result("embedding", value, **kwargs)

    def rms_norm(hidden: Any, **kwargs: Any) -> FakeTensor:
        record("rms_norm", kwargs=dict(kwargs))
        weight = _to_mini(kwargs["weight"])
        if len(weight.shape) == 4:
            weight = weight.reshape(weight.shape[-1])
        value = _torch_rms_norm(
            _fake_numeric_torch(),
            _to_mini(hidden),
            weight,
            float(kwargs.get("epsilon", 1e-5)),
        )
        return result("rms_norm", value, **kwargs)

    def linear(activation: Any, weight: Any, **kwargs: Any) -> FakeTensor:
        record("linear", kwargs=dict(kwargs))
        weight_value = _to_mini(weight)
        if len(weight_value.shape) == 4:
            matrix = weight_value.reshape(*weight_value.shape[-2:])
            value = MiniTensor(_mini_array(activation) @ matrix.array)
        else:
            value = _torch_linear(
                _fake_numeric_torch(), _to_mini(activation), weight_value
            )
        return result("linear", value, **kwargs)

    def mul(lhs: Any, rhs: Any, **kwargs: Any) -> FakeTensor:
        record("mul", kwargs=dict(kwargs))
        left = _to_mini(lhs)
        if kwargs.get("input_tensor_a_activations"):
            left = _fake_numeric_torch().nn.functional.silu(left)
        return result("mul", left * _to_mini(rhs), **kwargs)

    def add(lhs: Any, rhs: Any, **kwargs: Any) -> FakeTensor:
        record("add", kwargs=dict(kwargs))
        right = _to_mini(rhs) if _has_numeric(rhs) else rhs
        return result("add", _to_mini(lhs) + right, **kwargs)

    def concat(tensors: list[Any], **kwargs: Any) -> FakeTensor:
        record("concat", kwargs=dict(kwargs))
        value = _fake_numeric_torch().cat(
            [_to_mini(tensor) for tensor in tensors],
            dim=int(kwargs.get("dim", -1)),
        )
        return result("concat", value, **kwargs)

    def reshape(tensor: Any, logical_shape: list[int], padded_shape=None):
        record("reshape", logical_shape=list(logical_shape))
        return result("reshape", _to_mini(tensor).reshape(*logical_shape))

    def to_memory_config(tensor: Any, **kwargs: Any) -> FakeTensor:
        record("to_memory_config", kwargs=dict(kwargs))
        return result("to_memory_config", _to_mini(tensor), **kwargs)

    def untilize(tensor: Any, **kwargs: Any) -> FakeTensor:
        record("untilize", kwargs=dict(kwargs))
        return result("untilize", _to_mini(tensor), **kwargs)

    def argmax(tensor: Any, **kwargs: Any) -> FakeTensor:
        record("argmax", kwargs=dict(kwargs))
        value = _fake_numeric_torch().argmax(
            _to_mini(tensor), dim=int(kwargs.get("dim", -1))
        )
        return result("argmax", value, **kwargs)

    module.from_torch = from_torch
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.linear = linear
    module.mul = mul
    module.add = add
    module.concat = concat
    module.reshape = reshape
    module.to_memory_config = to_memory_config
    module.untilize = untilize
    module.argmax = argmax
    module.to_torch = lambda tensor: _to_mini(tensor)
    module.synchronize_device = lambda device: record("synchronize_device")
    return module


def _fake_parameters(split_count: int, layer_count: int = 2) -> Any:
    def layer(layer_id: int) -> Any:
        return types.SimpleNamespace(
            attention=types.SimpleNamespace(
                wqkv_packed=types.SimpleNamespace(weight=FakeTensor(f"wqkv_{layer_id}", [1, 1, 16, 32])),
                o_proj=types.SimpleNamespace(weight=FakeTensor(f"o_proj_{layer_id}", [1, 1, 16, 16])),
                rotary=types.SimpleNamespace(
                    cos_matrix=FakeTensor(f"cos_{layer_id}", [1, 1, 8, 4]),
                    sin_matrix=FakeTensor(f"sin_{layer_id}", [1, 1, 8, 4]),
                    transformation_matrix=FakeTensor(f"transform_{layer_id}", [1, 1, 4, 4]),
                ),
            ),
            input_norm=types.SimpleNamespace(weight=FakeTensor(f"input_norm_{layer_id}", [1, 1, 1, 16])),
            post_attention_norm=types.SimpleNamespace(weight=FakeTensor(f"post_norm_{layer_id}", [1, 1, 1, 16])),
            mlp=types.SimpleNamespace(
                gate_proj=types.SimpleNamespace(weight=FakeTensor(f"gate_{layer_id}", [1, 1, 16, 32])),
                up_proj=types.SimpleNamespace(weight=FakeTensor(f"up_{layer_id}", [1, 1, 16, 32])),
                gate_up_proj=types.SimpleNamespace(weight=FakeTensor(f"gate_up_{layer_id}", [1, 1, 16, 64])),
                down_proj=types.SimpleNamespace(weight=FakeTensor(f"down_{layer_id}", [1, 1, 32, 16])),
            ),
        )

    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(weight=FakeTensor("embed_weight", [1, 1, 128, 16])),
        layers=[layer(index) for index in range(layer_count)],
        final_norm=types.SimpleNamespace(weight=FakeTensor("final_norm", [1, 1, 1, 16])),
        lm_head=types.SimpleNamespace(
            splits=[types.SimpleNamespace(shard_id=index, weight=FakeTensor(f"lm_head_{index}", [1, 1, 16, 16])) for index in range(split_count)]
        ),
    )


def _fake_numeric_parameters(split_count: int) -> Any:
    hidden_size, intermediate_size, vocab_size = 16, 32, 128
    shard_size = vocab_size // split_count

    def matrix(name: str, shape: tuple[int, int]) -> MiniTensor:
        values = (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1.0) / (1000.0 + len(name))
        return MiniTensor(values)

    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(weight=matrix("embedding", (vocab_size, hidden_size))),
        layers=[types.SimpleNamespace(
            input_norm=types.SimpleNamespace(weight=MiniTensor(np.ones(hidden_size))),
            post_attention_norm=types.SimpleNamespace(weight=MiniTensor(np.ones(hidden_size) * 0.5)),
            mlp=types.SimpleNamespace(
                gate_proj=types.SimpleNamespace(weight=matrix("gate", (intermediate_size, hidden_size))),
                up_proj=types.SimpleNamespace(weight=matrix("up", (intermediate_size, hidden_size))),
                down_proj=types.SimpleNamespace(weight=matrix("down", (hidden_size, intermediate_size))),
            ),
        )],
        final_norm=types.SimpleNamespace(weight=MiniTensor(np.ones(hidden_size) * 0.75)),
        lm_head=types.SimpleNamespace(splits=[types.SimpleNamespace(weight=matrix(f"lm_head_{index}", (shard_size, hidden_size))) for index in range(split_count)]),
    )


def _fake_physical_numeric_parameters(split_count: int) -> Any:
    params = _fake_numeric_parameters(split_count)
    for layer in params.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = getattr(layer.mlp, name).weight
            getattr(layer.mlp, name).weight = weight.transpose(0, 1).reshape(1, 1, *weight.shape[::-1])
    params.embedding.weight = params.embedding.weight.reshape(1, 1, 128, 16)
    params.final_norm.weight = params.final_norm.weight.reshape(1, 1, 1, 16)
    for split in params.lm_head.splits:
        split.weight = split.weight.transpose(0, 1).reshape(1, 1, *split.weight.shape[::-1])
    return params


def _fake_tokenizer_module(token_ids: list[int]) -> Any:
    class FakeTokenizer:
        def __call__(self, prompt: str, add_special_tokens: bool = True):
            return {"input_ids": list(token_ids)}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str):
            return FakeTokenizer()

    return types.SimpleNamespace(AutoTokenizer=AutoTokenizer)


def _set_program_full_logits(program_dir: Path) -> None:
    config_path = program_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["generation"].update({"template": "full_logits", "mode": "full_logits", "retain_logits": True})
    config["lm_head"]["retain_logits"] = True
    config["final"][-1] = "full_logits"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "_name_or_path": "fake-llama",
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
    }))


def _write_template_config(path: Path) -> None:
    path.write_text(json.dumps({
        "device": "p150a",
        "model": "llama3.1-8b",
        "batch_size": 32,
        "decode_seq_len": 1,
        "prefill_seq_len": 128,
        "max_cache_len": 1024,
        "attention_template": "official_paged_attention_decode",
        "mlp_template": "official_gated_mlp_decode",
        "lm_head_template": "official_split_lm_head",
        "kv_cache_template": "paged_kv_cache",
        "generation_template": "device_argmax_greedy",
        "lm_head_split_count": 8,
        "dtype_recipe": "official_like_performance_seed",
    }))


__all__ = [
    "FakeTensor", "FakeTorchTensor", "FakeTTNNTensor", "MiniTensor",
    "_fake_numeric_parameters", "_fake_numeric_torch", "_fake_parameters",
    "_fake_physical_numeric_parameters", "_fake_tokenizer_module", "_fake_torch",
    "_fake_ttnn", "_make_fake_ttnn", "_make_generate_fake_ttnn",
    "_make_numeric_ttnn",
    "_set_program_full_logits", "_torch_embedding", "_torch_linear", "_torch_rms_norm",
    "_write_fake_model_config", "_write_template_config",
]
