from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Sequence


class TTNNObservationCollector:
    def __init__(
        self,
        *,
        ttnn: Any,
        torch: Any,
        capture_hidden: bool = True,
        capture_logits: bool = True,
        capture_kv_cache: bool = True,
        tensor_snapshot: Callable[..., dict[str, Any]],
        last_token_vector: Callable[[Any], Any],
        kv_cache_snapshot: Callable[..., dict[str, Any]],
    ) -> None:
        self.ttnn = ttnn
        self.torch = torch
        self.capture_hidden = bool(capture_hidden)
        self.capture_logits = bool(capture_logits)
        self.capture_kv_cache = bool(capture_kv_cache)
        self.tensor_snapshot = tensor_snapshot
        self.last_token_vector = last_token_vector
        self.kv_cache_snapshot = kv_cache_snapshot
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.diagnostics: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        name: str,
        tensor: Any,
        *,
        ops: Any,
        valid_seq_len: int | Sequence[int] | None = None,
        **_metadata: Any,
    ) -> None:
        if name.endswith(".logits"):
            if not self.capture_logits or not name.startswith("prefill."):
                return
        elif name.endswith("hidden"):
            if not self.capture_hidden or not name.startswith("prefill."):
                return
        elif name.endswith(("_pre_rope", "_post_rope")):
            if not name.startswith("prefill."):
                return
            self._observe_attention_tensor(
                name,
                tensor,
                ops=ops,
                valid_seq_len=valid_seq_len,
            )
            return
        else:
            return

        logical_shape = _shape(tensor)
        user_zero_seq_len = _user_zero_seq_len(valid_seq_len)
        sampled = tensor
        if name.startswith("prefill.layer.") and user_zero_seq_len is not None:
            sampled = ops.select_sequence_position(
                sampled,
                user_zero_seq_len - 1,
                op_name=f"correctness.{name}.position",
            )
        sampled = ops.slice_batch_user(
            sampled,
            0,
            op_name=f"correctness.{name}.user",
        )
        host = _to_torch(self.ttnn, sampled)
        self.checkpoints[name] = self.tensor_snapshot(
            name,
            self.last_token_vector(host),
            logical_shape=logical_shape,
            sample_policy={
                "kind": "last_token_vector",
                "batch_id": 0,
                "position": (
                    user_zero_seq_len - 1
                    if user_zero_seq_len is not None
                    else 0
                ),
                "source": "ttnn_device_slice",
            },
        )

    def observe_prefill_kv_cache(
        self,
        kv_cache: Any,
        *,
        effective_token_count: int,
    ) -> None:
        if not self.capture_kv_cache:
            return
        for layer_id, layer_cache in enumerate(kv_cache):
            for kind in ("key", "value"):
                tensor = getattr(layer_cache, kind[0])
                host = self._paged_user_zero_to_host(
                    tensor,
                    effective_token_count=effective_token_count,
                )
                name = f"prefill.layer.{layer_id}.{kind}_cache"
                self.checkpoints[name] = self.kv_cache_snapshot(name, host)

    def to_report(self) -> dict[str, Any]:
        captured = bool(self.checkpoints)
        return {
            "schema_version": 1,
            "kind": "ttnn_correctness_observations",
            "status": "captured" if captured else "empty",
            "passed": captured,
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": self.checkpoints,
            "diagnostic_count": len(self.diagnostics),
            "diagnostics": self.diagnostics,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status": "captured",
            "checkpoint_count": len(self.checkpoints),
            "checkpoint_names": sorted(self.checkpoints),
            "diagnostic_count": len(self.diagnostics),
            "diagnostic_names": sorted(self.diagnostics),
        }

    def _observe_attention_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        ops: Any,
        valid_seq_len: int | Sequence[int] | None,
    ) -> None:
        logical_shape = _shape(tensor)
        user_zero_seq_len = _user_zero_seq_len(valid_seq_len)
        sampled = tensor
        if user_zero_seq_len is not None:
            sampled = ops.select_sequence_position(
                sampled,
                user_zero_seq_len - 1,
                op_name=f"correctness.{name}.position",
            )
        sampled = ops.slice_batch_user(
            sampled,
            0,
            op_name=f"correctness.{name}.user",
        )
        host = _to_torch(self.ttnn, sampled)
        if len(host.shape) != 4:
            raise ValueError(
                f"expected [batch, heads, seq, dim] attention tensor: "
                f"{_shape(host)}"
            )
        vector = host[0, :, -1, :].reshape(-1)
        self.diagnostics[name] = self.tensor_snapshot(
            name,
            vector,
            logical_shape=logical_shape,
            sample_policy={
                "kind": "all_heads_last_token_vector",
                "batch_id": 0,
                "position": (
                    user_zero_seq_len - 1
                    if user_zero_seq_len is not None
                    else 0
                ),
                "source": "ttnn_device_slice",
            },
        )

    def _paged_user_zero_to_host(
        self,
        tensor: Any,
        *,
        effective_token_count: int,
    ) -> Any:
        shape = _shape(tensor)
        if len(shape) != 4:
            raise ValueError(f"expected paged KV cache rank 4, got {shape}")
        block_size = shape[2]
        block_count = max(1, math.ceil(effective_token_count / block_size))
        if block_count > shape[0]:
            raise ValueError(
                "effective token count requires more KV blocks than are "
                "available"
            )
        slice_op = getattr(self.ttnn, "slice", None)
        if not callable(slice_op):
            raise RuntimeError("TTNN correctness KV capture requires ttnn.slice")
        starts = [0, 0, 0, 0]
        ends = [block_count, shape[1], block_size, shape[3]]
        steps = [1, 1, 1, 1]
        try:
            sampled = slice_op(tensor, starts, ends, steps)
        except TypeError:
            sampled = slice_op(tensor, starts, ends)
        host = _to_torch(self.ttnn, sampled)
        host = host.permute(1, 0, 2, 3).reshape(
            1,
            shape[1],
            block_count * block_size,
            shape[3],
        )
        return host[:, :, :effective_token_count, :]


def _to_torch(ttnn: Any, tensor: Any) -> Any:
    to_torch = getattr(ttnn, "to_torch", None)
    if not callable(to_torch):
        raise RuntimeError("TTNN correctness capture requires ttnn.to_torch")
    return to_torch(tensor)


def _shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in getattr(tensor, "shape", ())]


def _user_zero_seq_len(value: int | Sequence[int] | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    values = list(value)
    return int(values[0]) if values else None
