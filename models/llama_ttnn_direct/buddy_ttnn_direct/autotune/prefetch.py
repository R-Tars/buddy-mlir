from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PREFETCH_SCHEMA_VERSION = 1
PREFETCH_REGION_PROMOTION_GAIN = 0.03
PREFETCH_FULL_MODEL_PROMOTION_GAIN = 0.01
PREFETCH_MAXIMUM_CV_PERCENT = 1.5
PREFETCH_SINGLE_DEVICE_L1_LIMIT_BYTES = 850_000
PREFETCH_SENDER_CORES = 8
PREFETCH_LEGAL_RECEIVERS_PER_SENDER = (1, 2, 3, 8, 10)
PREFETCH_REQUIRED_FULL_DECODE_SCOPES = (
    "mlp_only",
    "attention_projections",
    "all_major_linears",
)

_TILE_SIZE = 32
_MAX_CB_PAGES = 65_535
_BYTES_PER_TILE = {
    "bfloat4_b": 576,
    "bfloat8_b": 1088,
    "bfloat16": 2048,
}


class PrefetchAuditError(ValueError):
    """Raised when prefetch evidence or a candidate is malformed."""


@dataclass(frozen=True)
class PrefetchWeight:
    name: str
    k: int
    n: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise PrefetchAuditError("prefetch weight name must not be empty")
        if self.k <= 0 or self.n <= 0:
            raise PrefetchAuditError(
                "prefetch weight dimensions must be positive"
            )
        if self.dtype not in _BYTES_PER_TILE:
            raise PrefetchAuditError(
                f"unsupported prefetch weight dtype: {self.dtype}"
            )

    def to_dict(self, *, ring_size: int | None = None) -> dict[str, Any]:
        result = {
            "name": self.name,
            "shape": [self.k, self.n],
            "dtype": self.dtype,
        }
        if ring_size is not None:
            result["gcb_block_bytes"] = estimate_gcb_block_bytes(
                self, ring_size=ring_size
            )
        return result


LLAMA31_8B_PREFETCH_WEIGHTS = {
    "attention.qkv": PrefetchWeight(
        "attention.qkv", 4096, 6144, "bfloat8_b"
    ),
    "attention.o_proj": PrefetchWeight(
        "attention.o_proj", 4096, 4096, "bfloat8_b"
    ),
    "mlp.gate": PrefetchWeight("mlp.gate", 4096, 14336, "bfloat4_b"),
    "mlp.up": PrefetchWeight("mlp.up", 4096, 14336, "bfloat4_b"),
    "mlp.down": PrefetchWeight("mlp.down", 14336, 4096, "bfloat8_b"),
}


def estimate_gcb_block_bytes(
    weight: PrefetchWeight, *, ring_size: int
) -> int:
    """Match the current TTNN Prefetcher.insert_tensor block-size rule."""

    if ring_size <= 0:
        raise PrefetchAuditError("ring_size must be positive")
    h_tiles = math.ceil(weight.k / _TILE_SIZE)
    w_tiles = math.ceil(weight.n / _TILE_SIZE)
    h_tiles_padded = math.ceil(h_tiles / ring_size) * ring_size
    w_tiles_padded = math.ceil(w_tiles / ring_size) * ring_size
    tiles_per_receiver = (
        h_tiles_padded * w_tiles_padded
    ) // ring_size
    return tiles_per_receiver * _BYTES_PER_TILE[weight.dtype]


def llama31_8b_prefetch_scopes() -> dict[str, tuple[PrefetchWeight, ...]]:
    weights = LLAMA31_8B_PREFETCH_WEIGHTS
    attention = (weights["attention.qkv"], weights["attention.o_proj"])
    mlp = (weights["mlp.gate"], weights["mlp.up"], weights["mlp.down"])
    return {
        "mlp_only": mlp,
        "attention_projections": attention,
        "all_major_linears": attention + mlp,
    }


def audit_prefetch_scope(
    name: str,
    weights: Sequence[PrefetchWeight],
    *,
    receiver_cores_per_sender: int = 8,
    sender_cores: int = PREFETCH_SENDER_CORES,
    l1_limit_bytes: int = PREFETCH_SINGLE_DEVICE_L1_LIMIT_BYTES,
) -> dict[str, Any]:
    if receiver_cores_per_sender not in PREFETCH_LEGAL_RECEIVERS_PER_SENDER:
        raise PrefetchAuditError(
            "receiver_cores_per_sender is not legal for the current "
            "Blackhole prefetcher"
        )
    if sender_cores <= 0 or l1_limit_bytes <= 0:
        raise PrefetchAuditError("sender count and L1 limit must be positive")
    if not weights:
        raise PrefetchAuditError(
            "prefetch scope must contain at least one weight"
        )

    ring_size = receiver_cores_per_sender * sender_cores
    tensor_reports = [
        weight.to_dict(ring_size=ring_size) for weight in weights
    ]
    gcb_size = max(item["gcb_block_bytes"] for item in tensor_reports)
    incompatible = [
        weight.name
        for weight in weights
        if weight.k % ring_size or weight.n % ring_size
    ]
    return {
        "schema_version": PREFETCH_SCHEMA_VERSION,
        "scope": name,
        "receiver_cores_per_sender": receiver_cores_per_sender,
        "sender_cores": sender_cores,
        "ring_size": ring_size,
        "weight_request_order": [weight.name for weight in weights],
        "weights": tensor_reports,
        "gcb_size_bytes": gcb_size,
        "l1_limit_bytes": l1_limit_bytes,
        "l1_capacity_passed": gcb_size <= l1_limit_bytes,
        "ring_shard_compatible": not incompatible,
        "ring_incompatible_weights": incompatible,
        "static_eligible": gcb_size <= l1_limit_bytes and not incompatible,
        "stream_in1": True,
    }


def audit_official_llama31_8b_support(
    *, num_devices: int = 1, ring_size: int = 16
) -> dict[str, Any]:
    """Match current-source is_prefetcher_supported for Llama 3.1 8B."""

    if num_devices <= 0 or ring_size <= 0:
        raise PrefetchAuditError("num_devices and ring_size must be positive")
    if 14336 % num_devices:
        raise PrefetchAuditError(
            "Llama 3.1 8B hidden dimension must divide num_devices"
        )

    dim = 4096
    hidden_dim = 14336
    n_per_device = hidden_dim // num_devices
    n_per_core = math.ceil(n_per_device / ring_size)
    n_per_core_padded = math.ceil(n_per_core / _TILE_SIZE) * _TILE_SIZE
    n_padded = n_per_core_padded * ring_size
    h_tiles = math.ceil(dim / _TILE_SIZE)
    w_tiles = n_padded // _TILE_SIZE
    h_tiles_padded = math.ceil(h_tiles / ring_size) * ring_size
    tiles_per_core = h_tiles_padded * w_tiles // ring_size
    bytes_per_core = tiles_per_core * _BYTES_PER_TILE["bfloat8_b"]
    l1_limit = (
        1_000_000
        if num_devices in {4, 8}
        else PREFETCH_SINGLE_DEVICE_L1_LIMIT_BYTES
    )
    pages_passed = tiles_per_core <= _MAX_CB_PAGES
    l1_passed = bytes_per_core <= l1_limit
    kv_heads_divisible = 8 % num_devices == 0
    return {
        "schema_version": PREFETCH_SCHEMA_VERSION,
        "source_api": (
            "models.tt_transformers.tt.prefetcher."
            "is_prefetcher_supported"
        ),
        "model": "Llama-3.1-8B",
        "num_devices": num_devices,
        "ring_size": ring_size,
        "tiles_per_core": tiles_per_core,
        "maximum_cb_pages": _MAX_CB_PAGES,
        "pages_passed": pages_passed,
        "bytes_per_core": bytes_per_core,
        "l1_limit_bytes": l1_limit,
        "l1_capacity_passed": l1_passed,
        "kv_heads_divisible": kv_heads_divisible,
        "supported": pages_passed and l1_passed and kv_heads_divisible,
    }


def summarize_prefetch_samples(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise PrefetchAuditError(
            "prefetch samples must be non-empty, finite, and positive"
        )
    mean = statistics.fmean(values)
    return {
        "sample_count": len(values),
        "mean_ms": mean,
        "p50_ms": statistics.median(values),
        "stdev_ms": statistics.pstdev(values),
        "cv_percent": statistics.pstdev(values) / mean * 100.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def build_prefetch_region_ab(
    *,
    off_samples_ms: Sequence[float],
    on_samples_ms: Sequence[float],
    off_correctness_passed: bool,
    on_correctness_passed: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    off = summarize_prefetch_samples(off_samples_ms)
    on = summarize_prefetch_samples(on_samples_ms)
    gain = (off["p50_ms"] - on["p50_ms"]) / off["p50_ms"]
    correctness_passed = bool(
        off_correctness_passed and on_correctness_passed
    )
    stability_passed = (
        off["cv_percent"] <= PREFETCH_MAXIMUM_CV_PERCENT
        and on["cv_percent"] <= PREFETCH_MAXIMUM_CV_PERCENT
    )
    promotion_allowed = (
        correctness_passed
        and stability_passed
        and gain >= PREFETCH_REGION_PROMOTION_GAIN
    )
    return {
        "schema_version": PREFETCH_SCHEMA_VERSION,
        "status": (
            "passed" if correctness_passed and stability_passed else "failed"
        ),
        "scope": "mlp.gate_up",
        "statistic": "p50",
        "off": off,
        "on": on,
        "median_gain": gain,
        "correctness_passed": correctness_passed,
        "stability_passed": stability_passed,
        "promotion_threshold": PREFETCH_REGION_PROMOTION_GAIN,
        "enter_full_model": promotion_allowed,
        "metadata": dict(metadata or {}),
    }


def classify_prefetch_failure(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if (
        "circular buffers" in lowered
        and "clash" in lowered
    ) or "static circular buffer region" in lowered:
        return "l1_circular_buffer_conflict"
    if (
        "kernel group cores do not match sub device cores" in lowered
        or "num_intersections == num_cores" in lowered
    ):
        return "subdevice_core_set_mismatch"
    if "bytes_per_core" in lowered and "false" in lowered:
        return "official_l1_support_rejected"
    if "timed out" in lowered or "timeout" in lowered or "deadlock" in lowered:
        return "trace_lifecycle_timeout"
    return "runtime_error"


def build_full_decode_prefetch_assessment(
    *,
    scope: str,
    scope_audit: Mapping[str, Any],
    status: str,
    error: str | None = None,
    median_gain: float | None = None,
    cv_percent: float | None = None,
    correctness_passed: bool | None = None,
    evidence: Sequence[str] = (),
) -> dict[str, Any]:
    if scope not in PREFETCH_REQUIRED_FULL_DECODE_SCOPES:
        raise PrefetchAuditError(f"unknown full-decode prefetch scope: {scope}")
    if status not in {"measured", "static_rejected", "hardware_rejected"}:
        raise PrefetchAuditError(f"invalid full-decode status: {status}")

    promoted = False
    if status == "measured":
        if (
            median_gain is None
            or cv_percent is None
            or correctness_passed is None
        ):
            raise PrefetchAuditError(
                "measured full-decode assessment requires gain, CV, and "
                "correctness"
            )
        if not math.isfinite(median_gain) or not math.isfinite(cv_percent):
            raise PrefetchAuditError("full-decode metrics must be finite")
        promoted = (
            median_gain >= PREFETCH_FULL_MODEL_PROMOTION_GAIN
            and cv_percent <= PREFETCH_MAXIMUM_CV_PERCENT
            and correctness_passed
        )
    elif not error:
        raise PrefetchAuditError(
            "rejected full-decode assessment requires an error"
        )

    return {
        "schema_version": PREFETCH_SCHEMA_VERSION,
        "scope": scope,
        "status": status,
        "scope_audit": dict(scope_audit),
        "error_class": classify_prefetch_failure(error),
        "error": error,
        "median_gain": median_gain,
        "cv_percent": cv_percent,
        "correctness_passed": correctness_passed,
        "promotion_threshold": PREFETCH_FULL_MODEL_PROMOTION_GAIN,
        "maximum_cv_percent": PREFETCH_MAXIMUM_CV_PERCENT,
        "promoted": promoted,
        "evidence": list(evidence),
    }
