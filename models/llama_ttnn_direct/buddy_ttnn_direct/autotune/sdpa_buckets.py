from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .legality import DeviceDescriptor, WorkloadSpec
from .measurement import MeasurementCandidate
from .schema import PrecisionContract, sha256_json
from .sdpa import (
    SDPA_OPERATOR,
    SDPAEnumerationResult,
    enumerate_sdpa_programs,
    rank_sdpa_measurement_candidates,
)
from .space import SearchSpaceConfig

SDPA_BUCKET_SCHEMA_VERSION = 1
SDPA_BUCKET_PROMOTION_GAIN = 0.01
DEFAULT_SDPA_CONTEXT_BUCKET_BOUNDS = (
    (1, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
)


class SDPABucketError(ValueError):
    """Raised when context buckets or their measurements are inconsistent."""


@dataclass(frozen=True)
class SDPAContextBucket:
    name: str
    min_context_len: int
    max_context_len: int
    representative_context_len: int

    def __post_init__(self) -> None:
        if not self.name:
            raise SDPABucketError("SDPA context bucket name must be non-empty")
        if self.min_context_len <= 0:
            raise SDPABucketError("SDPA context lengths must be positive")
        if self.max_context_len < self.min_context_len:
            raise SDPABucketError("SDPA context bucket bounds are reversed")
        if not (
            self.min_context_len
            <= self.representative_context_len
            <= self.max_context_len
        ):
            raise SDPABucketError(
                "representative context length must be inside its bucket"
            )

    def contains(self, context_len: int) -> bool:
        return self.min_context_len <= int(context_len) <= self.max_context_len

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_context_len": self.min_context_len,
            "max_context_len": self.max_context_len,
            "representative_context_len": self.representative_context_len,
        }


@dataclass(frozen=True)
class SDPABucketEnumeration:
    bucket: SDPAContextBucket
    enumeration: SDPAEnumerationResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket.to_dict(),
            "enumeration": self.enumeration.to_dict(),
        }


def build_sdpa_context_buckets(
    max_cache_len: int,
    *,
    bounds: Sequence[tuple[int, int]] | None = None,
) -> tuple[SDPAContextBucket, ...]:
    capacity = int(max_cache_len)
    if capacity <= 0:
        raise SDPABucketError("max_cache_len must be positive")
    selected_bounds = tuple(bounds or DEFAULT_SDPA_CONTEXT_BUCKET_BOUNDS)
    buckets: list[SDPAContextBucket] = []
    expected_start = 1
    for start, end in selected_bounds:
        start = int(start)
        end = min(int(end), capacity)
        if start > capacity:
            break
        if start != expected_start:
            raise SDPABucketError(
                "SDPA context buckets must be contiguous and start at one"
            )
        bucket = SDPAContextBucket(
            name=f"context_{start}_{end}",
            min_context_len=start,
            max_context_len=end,
            representative_context_len=end,
        )
        buckets.append(bucket)
        expected_start = end + 1
        if end == capacity:
            break
    if not buckets or buckets[-1].max_context_len != capacity:
        raise SDPABucketError(
            f"SDPA context buckets do not cover cache capacity {capacity}"
        )
    return tuple(buckets)


def select_sdpa_context_bucket(
    buckets: Sequence[SDPAContextBucket], context_len: int
) -> SDPAContextBucket:
    for bucket in buckets:
        if bucket.contains(context_len):
            return bucket
    raise SDPABucketError(
        f"active context length {context_len} has no SDPA bucket"
    )


def enumerate_sdpa_context_buckets(
    *,
    runtime_config: Mapping[str, Any],
    device: DeviceDescriptor,
    precision_contract: PrecisionContract,
    bounds: Sequence[tuple[int, int]] | None = None,
    max_proposals_per_bucket: int = 256,
) -> tuple[SDPABucketEnumeration, ...]:
    physical_workload = WorkloadSpec.from_runtime_config(runtime_config)
    base_space = SearchSpaceConfig.from_runtime_config(runtime_config)
    buckets = build_sdpa_context_buckets(
        physical_workload.sdpa.cache_len,
        bounds=bounds,
    )
    return tuple(
        SDPABucketEnumeration(
            bucket=bucket,
            enumeration=enumerate_sdpa_programs(
                base_space=base_space,
                workload=physical_workload,
                device=device,
                precision_contract=precision_contract,
                max_proposals=max_proposals_per_bucket,
                active_context_len=bucket.representative_context_len,
            ),
        )
        for bucket in buckets
    )


def rank_sdpa_bucket_measurement_candidates(
    bucket: SDPABucketEnumeration,
) -> tuple[MeasurementCandidate, ...]:
    operator_group = f"{SDPA_OPERATOR}.{bucket.bucket.name}"
    ranked = []
    for candidate in rank_sdpa_measurement_candidates(bucket.enumeration):
        ranked.append(
            MeasurementCandidate.create(
                candidate_id=candidate.candidate_id,
                operator_name=operator_group,
                candidate_kind="sdpa_context_bucket",
                analytical_score=candidate.analytical_score,
                l1_bytes=candidate.l1_bytes,
                source=candidate.source,
                is_incumbent=candidate.is_incumbent,
                metadata={
                    **candidate.metadata,
                    "bucket": bucket.bucket.to_dict(),
                },
            )
        )
    return tuple(ranked)


def build_context_distribution(
    *,
    prompt_context_lengths: Sequence[int],
    generated_tokens_per_user: int | Sequence[int],
    buckets: Sequence[SDPAContextBucket],
) -> dict[str, Any]:
    if not prompt_context_lengths:
        raise SDPABucketError("prompt context distribution must not be empty")
    if isinstance(generated_tokens_per_user, int):
        generated = [int(generated_tokens_per_user)] * len(
            prompt_context_lengths
        )
    else:
        generated = [int(value) for value in generated_tokens_per_user]
    if len(generated) != len(prompt_context_lengths):
        raise SDPABucketError(
            "prompt and generated-token counts must have equal length"
        )
    counts = {bucket.name: 0 for bucket in buckets}
    transitions = {bucket.name: 0 for bucket in buckets}
    total = 0
    for prompt_len, token_count in zip(prompt_context_lengths, generated):
        prompt_len = int(prompt_len)
        if prompt_len <= 0 or token_count < 0:
            raise SDPABucketError(
                "prompt lengths must be positive and token counts non-negative"
            )
        previous = None
        for offset in range(token_count):
            context_len = prompt_len + offset
            bucket = select_sdpa_context_bucket(buckets, context_len)
            counts[bucket.name] += 1
            total += 1
            if previous is not None and previous != bucket.name:
                transitions[bucket.name] += 1
            previous = bucket.name
    if total <= 0:
        raise SDPABucketError("context distribution has no decode tokens")
    return {
        "prompt_count": len(prompt_context_lengths),
        "total_decode_tokens": total,
        "prompt_context_lengths": [
            int(value) for value in prompt_context_lengths
        ],
        "generated_tokens_per_user": generated,
        "buckets": {
            bucket.name: {
                **bucket.to_dict(),
                "token_count": counts[bucket.name],
                "weight": counts[bucket.name] / total,
                "entry_transition_count": transitions[bucket.name],
            }
            for bucket in buckets
        },
        "distribution_sha256": sha256_json(
            {
                "prompt_context_lengths": [
                    int(value) for value in prompt_context_lengths
                ],
                "generated_tokens_per_user": generated,
                "buckets": [bucket.to_dict() for bucket in buckets],
            }
        ),
    }


def select_sdpa_bucket_winner(
    bucket: SDPABucketEnumeration,
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    statistic: str = "p50",
) -> dict[str, Any]:
    official = bucket.enumeration.official_candidates[0]
    incumbent_latency = _measurement_latency(
        measurements.get(official.candidate_id), statistic
    )
    ranked = []
    for candidate in bucket.enumeration.candidates:
        latency = _measurement_latency(
            measurements.get(candidate.candidate_id), statistic
        )
        if latency is None:
            continue
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "latency_ms": latency,
                "is_official": candidate.is_official,
                "runtime_config": candidate.runtime_config(),
                "program": candidate.program.to_dict(),
                "kernel_output_memory": candidate.kernel_output_memory.to_dict(),
                "post_sdpa_output_memory": (
                    candidate.post_sdpa_output_memory.to_dict()
                ),
            }
        )
    ranked.sort(key=lambda item: (item["latency_ms"], item["candidate_id"]))
    winner = ranked[0] if ranked else None
    gain = None
    if incumbent_latency is not None and winner is not None:
        gain = (
            incumbent_latency - float(winner["latency_ms"])
        ) / incumbent_latency
    return {
        "schema_version": SDPA_BUCKET_SCHEMA_VERSION,
        "status": (
            "selected"
            if incumbent_latency is not None and winner is not None
            else "incomplete"
        ),
        "operator": SDPA_OPERATOR,
        "bucket": bucket.bucket.to_dict(),
        "statistic": statistic,
        "incumbent": {
            "candidate_id": official.candidate_id,
            "latency_ms": incumbent_latency,
        },
        "winner": copy.deepcopy(winner),
        "ranked_candidates": ranked,
        "bucket_gain": gain,
    }


def apply_sdpa_bucket_winners(
    runtime_config: Mapping[str, Any],
    bucket_enumerations: Sequence[SDPABucketEnumeration],
    winner_ids: Mapping[str, str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(runtime_config))
    entries = []
    for bucket_result in bucket_enumerations:
        bucket = bucket_result.bucket
        candidate_id = winner_ids.get(bucket.name)
        if candidate_id is None:
            raise SDPABucketError(
                f"missing winner for SDPA bucket {bucket.name}"
            )
        candidate = bucket_result.enumeration.candidate(candidate_id)
        runtime = candidate.runtime_config()
        config_key = sha256_json(
            {
                "bucket": bucket.to_dict(),
                "candidate_id": candidate.candidate_id,
                "runtime_config": runtime,
            }
        )
        entries.append(
            {
                **bucket.to_dict(),
                "candidate_id": candidate.candidate_id,
                "config_key": config_key,
                **runtime,
            }
        )
    result.setdefault("attention", {})["sdpa_context_buckets"] = entries
    result.setdefault("autotune", {})["context_aware_sdpa"] = {
        "schema_version": SDPA_BUCKET_SCHEMA_VERSION,
        "operator": SDPA_OPERATOR,
        "bucket_count": len(entries),
        "bucket_config_keys": [entry["config_key"] for entry in entries],
    }
    return result


def build_sdpa_bucket_phase_report(
    selections: Sequence[Mapping[str, Any]],
    *,
    context_distribution: Mapping[str, Any],
    trace_switch_overhead_ms: Sequence[float] = (),
    full_model_gain: float | None = None,
) -> dict[str, Any]:
    by_bucket = {
        str(item.get("bucket", {}).get("name")): copy.deepcopy(dict(item))
        for item in selections
    }
    distribution = context_distribution.get("buckets")
    if not isinstance(distribution, Mapping):
        raise SDPABucketError("context distribution has no bucket weights")
    if set(by_bucket) != set(distribution):
        raise SDPABucketError(
            "selection buckets do not match context distribution"
        )
    complete = all(
        item.get("status") == "selected" for item in by_bucket.values()
    )
    incumbent_weighted = _weighted_latency(by_bucket, distribution, "incumbent")
    winner_weighted = _weighted_latency(by_bucket, distribution, "winner")
    weighted_gain = None
    if incumbent_weighted is not None and winner_weighted is not None:
        weighted_gain = (
            incumbent_weighted - winner_weighted
        ) / incumbent_weighted
    switch_samples = [float(value) for value in trace_switch_overhead_ms]
    if any(not math.isfinite(value) or value < 0 for value in switch_samples):
        raise SDPABucketError("trace-switch overhead samples must be finite")
    weighted_gate_passed = (
        complete
        and weighted_gain is not None
        and weighted_gain >= SDPA_BUCKET_PROMOTION_GAIN
    )
    full_model_gate_passed = (
        full_model_gain is not None
        and math.isfinite(full_model_gain)
        and full_model_gain >= SDPA_BUCKET_PROMOTION_GAIN
    )
    return {
        "schema_version": SDPA_BUCKET_SCHEMA_VERSION,
        "stage": "context-aware-sdpa-bucket-tuning",
        "status": "passed" if complete else "incomplete",
        "operator": SDPA_OPERATOR,
        "bucket_selections": by_bucket,
        "context_distribution": copy.deepcopy(dict(context_distribution)),
        "full_decode_weighted_average": {
            "incumbent_latency_ms": incumbent_weighted,
            "winner_latency_ms": winner_weighted,
            "gain": weighted_gain,
        },
        "trace_switch_overhead": {
            "sample_count": len(switch_samples),
            "samples_ms": switch_samples,
            "mean_ms": (
                sum(switch_samples) / len(switch_samples)
                if switch_samples
                else None
            ),
            "max_ms": max(switch_samples) if switch_samples else None,
        },
        "promotion_threshold": SDPA_BUCKET_PROMOTION_GAIN,
        "weighted_distribution_gate_passed": weighted_gate_passed,
        "enter_full_model": weighted_gate_passed,
        "full_model_gain": full_model_gain,
        "full_model_gate_passed": full_model_gate_passed,
        "promotion_allowed": weighted_gate_passed and full_model_gate_passed,
        "phase_completed": complete,
    }


def _measurement_latency(
    measurement: Mapping[str, Any] | None, statistic: str
) -> float | None:
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("status") != "passed"
    ):
        return None
    statistics = measurement.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get(statistic) is None:
        return None
    value = float(statistics[statistic])
    return value if math.isfinite(value) and value > 0 else None


def _weighted_latency(
    selections: Mapping[str, Mapping[str, Any]],
    distribution: Mapping[str, Any],
    arm: str,
) -> float | None:
    total = 0.0
    for name, selection in selections.items():
        bucket_distribution = distribution.get(name)
        if not isinstance(bucket_distribution, Mapping):
            return None
        weight = float(bucket_distribution.get("weight", 0.0))
        selected = selection.get(arm)
        if (
            not isinstance(selected, Mapping)
            or selected.get("latency_ms") is None
        ):
            if weight == 0.0:
                continue
            return None
        total += weight * float(selected["latency_ms"])
    return total
