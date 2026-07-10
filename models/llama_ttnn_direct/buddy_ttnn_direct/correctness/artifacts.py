from __future__ import annotations

import hashlib
import struct
from typing import Any


def tensor_snapshot(
    name: str,
    tensor: Any,
    *,
    logical_shape: list[int] | None = None,
    sample_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = _float_cpu_tensor(tensor)
    values = [float(value) for value in host.reshape(-1).tolist()]
    return {
        "name": name,
        "logical_shape": logical_shape or _shape(tensor),
        "sample_shape": _shape(host),
        "sample_count": len(values),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "storage_dtype": "float32",
        "sample_policy": sample_policy or {"kind": "full_tensor"},
        "sha256": _float_sha256(values),
        "values": values,
    }


def last_token_vector(tensor: Any) -> Any:
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    ndim = int(getattr(tensor, "ndim", len(getattr(tensor, "shape", ()))))
    if ndim == 1:
        return tensor
    if ndim == 2:
        return tensor[-1]
    if ndim == 3:
        return tensor[0, -1]
    if ndim == 4 and int(tensor.shape[0]) == 1:
        return tensor[0, 0, -1]
    raise ValueError(f"unsupported hidden/logits tensor shape: {_shape(tensor)}")


def kv_cache_snapshot(
    name: str,
    tensor: Any,
    *,
    max_heads: int = 2,
    max_positions: int = 8,
    max_channels: int = 32,
) -> dict[str, Any]:
    shape = _shape(tensor)
    if len(shape) != 4:
        raise ValueError(f"expected [batch, heads, seq, dim] KV tensor: {shape}")
    head_ids = _evenly_spaced_indices(shape[1], max_heads)
    position_ids = _evenly_spaced_indices(shape[2], max_positions)
    channel_ids = _evenly_spaced_indices(shape[3], max_channels)
    sampled = tensor[0][head_ids][:, position_ids][:, :, channel_ids]
    return tensor_snapshot(
        name,
        sampled,
        logical_shape=shape,
        sample_policy={
            "kind": "kv_coordinates",
            "batch_id": 0,
            "head_ids": head_ids,
            "position_ids": position_ids,
            "channel_ids": channel_ids,
        },
    )


def _evenly_spaced_indices(size: int, limit: int) -> list[int]:
    if size <= 0:
        return []
    count = min(size, max(1, int(limit)))
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (size - 1) / (count - 1))
            for index in range(count)
        }
    )


def _float_cpu_tensor(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    to_float = getattr(tensor, "float", None)
    if callable(to_float):
        tensor = to_float()
    contiguous = getattr(tensor, "contiguous", None)
    if callable(contiguous):
        tensor = contiguous()
    if not callable(getattr(tensor, "reshape", None)):
        raise TypeError("tensor snapshot requires a torch-compatible tensor")
    return tensor


def _shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in getattr(tensor, "shape", ())]


def _float_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()
