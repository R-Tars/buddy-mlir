from __future__ import annotations

from typing import Any

DECODE_STEP_AUTOTUNE_KNOBS = (
    "lm_head_split_count",
    "generation_template",
    "mlp_intermediate_dtype",
    "attention_sdpa_output_memory_config",
    "attention_concat_heads_output_memory_config",
)
from .profiling import (
    bottleneck_summary_complete as _bottleneck_summary_complete,
    bottleneck_summary_observed as _bottleneck_summary_observed,
    lm_head_profile_complete as _lm_head_profile_complete,
    lm_head_profile_observed as _lm_head_profile_observed,
    throughput_summary_complete as _throughput_summary_complete,
    throughput_summary_observed as _throughput_summary_observed,
)
from .runtime import (
    decode_output_shape_observed as _decode_output_shape_observed,
    decode_output_shapes_complete as _decode_output_shapes_complete,
)
from .schema import (
    contains_all as _contains_all,
    int_equal as _int_equal,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    path_exists_relative_to as _path_exists_relative_to,
    paths_exist_relative_to as _paths_exist_relative_to,
    positive_number as _positive_number,
    safe_int as _safe_int,
)


def autotune_knob_coverage_complete(
    coverage: Any,
    *,
    candidate_count: Any,
) -> bool:
    if not isinstance(coverage, dict):
        return False
    coverage_count = coverage.get("candidate_count")
    if not _positive_number(coverage_count):
        return False
    if candidate_count is not None and not _int_equal(
        coverage_count,
        candidate_count,
    ):
        return False
    if not _contains_all(
        coverage.get("knobs"),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    values = coverage.get("values")
    value_counts = coverage.get("value_counts")
    if not isinstance(values, dict) or not isinstance(value_counts, dict):
        return False
    for knob in DECODE_STEP_AUTOTUNE_KNOBS:
        knob_values = values.get(knob)
        if not isinstance(knob_values, list) or not knob_values:
            return False
        counts = value_counts.get(knob)
        if not isinstance(counts, dict) or not counts:
            return False
        try:
            count_total = sum(int(count) for count in counts.values())
        except (TypeError, ValueError):
            return False
        if not _int_equal(count_total, coverage_count):
            return False
    return True


def autotune_knob_coverage_observed(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "knobs": coverage.get("knobs"),
        "candidate_count": coverage.get("candidate_count"),
        "values": coverage.get("values"),
        "varied_knobs": coverage.get("varied_knobs"),
        "missing_varied_knobs": autotune_missing_varied_knobs(coverage),
        "all_knobs_varied": autotune_default_knobs_varied(coverage),
    }


def autotune_default_knobs_varied(coverage: Any) -> bool:
    return autotune_missing_varied_knobs(coverage) == []


def autotune_knob_variation_observed(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "varied_knobs": coverage.get("varied_knobs"),
        "missing_varied_knobs": autotune_missing_varied_knobs(coverage),
        "all_knobs_varied": autotune_default_knobs_varied(coverage),
    }


def autotune_missing_varied_knobs(coverage: Any) -> list[str]:
    if not isinstance(coverage, dict):
        return list(DECODE_STEP_AUTOTUNE_KNOBS)
    missing = coverage.get("missing_varied_knobs")
    if isinstance(missing, list):
        return [str(knob) for knob in missing]
    varied = coverage.get("varied_knobs")
    if not isinstance(varied, list):
        return list(DECODE_STEP_AUTOTUNE_KNOBS)
    return [
        knob
        for knob in DECODE_STEP_AUTOTUNE_KNOBS
        if knob not in varied
    ]


def autotune_output_kind_counts_complete(
    counts: Any,
    coverage: Any,
) -> bool:
    if not isinstance(counts, dict) or not isinstance(coverage, dict):
        return False
    expected_kinds = autotune_expected_output_kinds(coverage)
    if not expected_kinds:
        return True
    for kind in expected_kinds:
        if not _positive_number(counts.get(kind)):
            return False
    return True


def autotune_output_kind_counts_observed(
    counts: Any,
    coverage: Any,
) -> dict[str, Any]:
    return {
        "counts": counts if isinstance(counts, dict) else {},
        "expected_output_kinds": autotune_expected_output_kinds(coverage),
    }


def autotune_candidate_summaries(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    summaries = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        summaries.append(
            {
                "id": candidate.get("id"),
                "config": candidate.get("config"),
                "model": candidate.get("model"),
                "profile_metadata": candidate.get("profile_metadata", []),
                "profile_report": candidate.get("profile_report"),
                "knobs": candidate.get("knobs"),
                "status": candidate.get("status"),
                "passed": candidate.get("passed"),
                "metric": candidate.get("metric"),
                "output_kind": candidate.get("output_kind"),
                "output_shapes": candidate.get("output_shapes"),
                "lm_head_profile": candidate.get("lm_head_profile"),
                "throughput_summary": candidate.get("throughput_summary"),
                "bottleneck_summary": candidate.get("bottleneck_summary"),
                "parameter_source": candidate.get("parameter_source"),
                "trace_status": candidate.get("trace_status"),
                "reference_status": candidate.get("reference_status"),
                "reference_kind": candidate.get("reference_kind"),
                "reference_failed_checks": candidate.get(
                    "reference_failed_checks",
                    [],
                ),
                "error": candidate.get("error"),
            }
        )
    return summaries


def autotune_candidates_complete(
    candidates: Any,
    *,
    candidate_count: Any,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
    out_dir: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(candidates, list) or not candidates:
        return False
    if not _int_equal(len(candidates), candidate_count):
        return False
    candidate_ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False
        candidate_id = candidate.get("id")
        if not _non_empty_string(candidate_id):
            return False
        candidate_ids.append(candidate_id)
        if not autotune_candidate_complete(
            candidate,
            layer_count=layer_count,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            out_dir=out_dir,
            require_trace=require_trace,
        ):
            return False
    return len(candidate_ids) == len(set(candidate_ids))


def autotune_candidate_complete(
    candidate: dict[str, Any],
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
    out_dir: Any,
    require_trace: bool,
) -> bool:
    output_kind = candidate.get("output_kind")
    if output_kind not in {"token", "logits"}:
        return False
    knobs = candidate.get("knobs")
    if not isinstance(knobs, dict) or not _contains_all(
        list(knobs),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    if not (
        candidate.get("status") == "profiled"
        and candidate.get("passed") is True
        and candidate.get("parameter_source") == "hf_model"
        and candidate.get("reference_status") == "passed"
        and candidate.get("reference_failed_checks") == []
        and candidate.get("error") is None
        and _nonnegative_number(candidate.get("metric"))
        and _path_exists_relative_to(candidate.get("config"), out_dir)
        and _path_exists_relative_to(candidate.get("model"), out_dir)
        and _path_exists_relative_to(candidate.get("profile_report"), out_dir)
        and _paths_exist_relative_to(candidate.get("profile_metadata"), out_dir)
        and _lm_head_profile_complete(
            candidate.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and _throughput_summary_complete(candidate.get("throughput_summary"))
        and _bottleneck_summary_complete(candidate.get("bottleneck_summary"))
        and _decode_output_shapes_complete(
            candidate.get("output_shapes"),
            layer_count=layer_count,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
        )
    ):
        return False
    if require_trace:
        return candidate.get("trace_status") == "captured_and_executed"
    return True


def autotune_candidates_observed(
    candidates: Any,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    observed = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        observed.append(
            {
                "id": candidate.get("id"),
                "status": candidate.get("status"),
                "passed": candidate.get("passed"),
                "metric": candidate.get("metric"),
                "output_kind": candidate.get("output_kind"),
                "parameter_source": candidate.get("parameter_source"),
                "trace_status": candidate.get("trace_status"),
                "reference_status": candidate.get("reference_status"),
                "reference_failed_checks": candidate.get(
                    "reference_failed_checks"
                ),
                "error": candidate.get("error"),
                "knobs": candidate.get("knobs"),
                "output_shapes": _decode_output_shape_observed(
                    candidate.get("output_shapes")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    candidate.get("lm_head_profile")
                ),
                "throughput": _throughput_summary_observed(
                    candidate.get("throughput_summary")
                ),
                "bottleneck": _bottleneck_summary_observed(
                    candidate.get("bottleneck_summary")
                ),
            }
        )
    return observed


def autotune_leaderboard_complete(
    leaderboard: Any,
    *,
    candidate_count: Any,
    candidate_summaries: Any,
    best: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(leaderboard, list) or not leaderboard:
        return False
    expected_count = _safe_int(candidate_count)
    if expected_count is None or expected_count <= 0:
        return False
    if len(leaderboard) != expected_count:
        return False
    expected_ids = autotune_candidate_ids(candidate_summaries)
    if len(expected_ids) != expected_count:
        return False
    observed_ids = []
    for expected_rank, entry in enumerate(leaderboard, start=1):
        if not isinstance(entry, dict):
            return False
        if not _int_equal(entry.get("rank"), expected_rank):
            return False
        if not autotune_leaderboard_entry_complete(
            entry,
            require_trace=require_trace,
        ):
            return False
        observed_ids.append(str(entry.get("candidate_id")))
    if sorted(observed_ids) != sorted(expected_ids):
        return False
    if _non_empty_string(best):
        first = leaderboard[0]
        if first.get("candidate_id") != best:
            return False
    return len(observed_ids) == len(set(observed_ids))


def autotune_leaderboard_entry_complete(
    entry: dict[str, Any],
    *,
    require_trace: bool,
) -> bool:
    output_kind = entry.get("output_kind")
    if output_kind not in {"token", "logits"}:
        return False
    knobs = entry.get("knobs")
    if not isinstance(knobs, dict) or not _contains_all(
        list(knobs),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    if not (
        _non_empty_string(entry.get("candidate_id"))
        and entry.get("status") == "profiled"
        and entry.get("passed") is True
        and entry.get("parameter_source") == "hf_model"
        and entry.get("reference_status") == "passed"
        and entry.get("error") is None
        and _nonnegative_number(entry.get("metric_value"))
        and _non_empty_string(entry.get("profile_report"))
        and _throughput_summary_complete(entry.get("throughput_summary"))
        and _lm_head_profile_complete(
            entry.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and _bottleneck_summary_complete(entry.get("bottleneck_summary"))
    ):
        return False
    if require_trace:
        return entry.get("trace_status") == "captured_and_executed"
    return True


def autotune_best_candidate_summary_complete(
    summary: Any,
    *,
    best: Any,
    require_trace: bool,
) -> bool:
    if not isinstance(summary, dict) or not _non_empty_string(best):
        return False
    return (
        summary.get("candidate_id") == best
        and _int_equal(summary.get("rank"), 1)
        and autotune_leaderboard_entry_complete(
            summary,
            require_trace=require_trace,
        )
        and _non_empty_string(summary.get("config"))
        and _non_empty_string(summary.get("model"))
        and isinstance(summary.get("profile_metadata"), list)
    )


def autotune_candidate_ids(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        return []
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return []
        candidate_id = candidate.get("id") or candidate.get("candidate_id")
        if not _non_empty_string(candidate_id):
            return []
        ids.append(str(candidate_id))
    return ids


def autotune_leaderboard_observed(
    leaderboard: Any,
) -> list[dict[str, Any]]:
    if not isinstance(leaderboard, list):
        return []
    observed = []
    for entry in leaderboard:
        if not isinstance(entry, dict):
            continue
        observed.append(
            {
                "rank": entry.get("rank"),
                "candidate_id": entry.get("candidate_id"),
                "status": entry.get("status"),
                "passed": entry.get("passed"),
                "metric": entry.get("metric"),
                "metric_value": entry.get("metric_value"),
                "parameter_source": entry.get("parameter_source"),
                "trace_status": entry.get("trace_status"),
                "reference_status": entry.get("reference_status"),
                "output_kind": entry.get("output_kind"),
                "throughput": _throughput_summary_observed(
                    entry.get("throughput_summary")
                ),
                "bottleneck": _bottleneck_summary_observed(
                    entry.get("bottleneck_summary")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    entry.get("lm_head_profile")
                ),
            }
        )
    return observed


def autotune_best_candidate_summary_observed(
    summary: Any,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    observed = autotune_leaderboard_observed([summary])
    if not observed:
        return {}
    result = observed[0]
    result["config"] = summary.get("config")
    result["model"] = summary.get("model")
    metadata = summary.get("profile_metadata")
    result["profile_metadata_count"] = (
        len(metadata) if isinstance(metadata, list) else None
    )
    return result


def autotune_expected_output_kinds(coverage: Any) -> list[str]:
    if not isinstance(coverage, dict):
        return []
    values = coverage.get("values")
    if not isinstance(values, dict):
        return []
    generation_templates = values.get("generation_template")
    if not isinstance(generation_templates, list):
        return []
    kinds = []
    for template in generation_templates:
        kind = "token" if template == "device_argmax_greedy" else "logits"
        if kind not in kinds:
            kinds.append(kind)
    return kinds
