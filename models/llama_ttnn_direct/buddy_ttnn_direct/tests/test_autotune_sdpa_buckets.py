from __future__ import annotations

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    DeviceDescriptor,
    PrecisionContract,
    apply_sdpa_bucket_winners,
    build_context_distribution,
    build_sdpa_bucket_phase_report,
    build_sdpa_context_buckets,
    enumerate_sdpa_context_buckets,
    select_sdpa_bucket_winner,
    select_sdpa_context_bucket,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    load_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    PACKAGE_ROOT,
    _official_runtime_config,
)


@pytest.fixture(scope="module")
def bucket_enumerations():
    runtime = _official_runtime_config()
    precision = PrecisionContract.from_template_config(
        load_template_config(
            PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json"
        )
    )
    return enumerate_sdpa_context_buckets(
        runtime_config=runtime,
        device=DeviceDescriptor.p150a(),
        precision_contract=precision,
        max_proposals_per_bucket=96,
    )


def test_default_buckets_cover_cache_and_select_boundaries() -> None:
    buckets = build_sdpa_context_buckets(1024)

    assert [
        (item.min_context_len, item.max_context_len) for item in buckets
    ] == [
        (1, 128),
        (129, 256),
        (257, 512),
        (513, 1024),
    ]
    assert select_sdpa_context_bucket(buckets, 128).name == "context_1_128"
    assert select_sdpa_context_bucket(buckets, 129).name == "context_129_256"
    assert select_sdpa_context_bucket(buckets, 1024).name == "context_513_1024"
    with pytest.raises(ValueError, match="has no SDPA bucket"):
        select_sdpa_context_bucket(buckets, 1025)


def test_bucket_enumeration_keeps_physical_cache_and_context_identity(
    bucket_enumerations,
) -> None:
    assert len(bucket_enumerations) == 4
    candidate_sets = []
    for result in bucket_enumerations:
        report = result.enumeration.to_dict()
        assert report["status"] == "passed"
        assert report["workload"]["cache_len"] == 1024
        assert report["active_context_len"] == result.bucket.max_context_len
        assert len(report["searched_dimensions"]["post_sdpa_output_memory"]) > 1
        assert report["frozen_dimensions"] == {"exp_approx_mode": False}
        candidate_sets.append(
            {item.candidate_id for item in result.enumeration.candidates}
        )
    for left, right in zip(candidate_sets, candidate_sets[1:]):
        assert left.isdisjoint(right)


def test_context_distribution_uses_actual_prompt_positions() -> None:
    buckets = build_sdpa_context_buckets(1024)
    distribution = build_context_distribution(
        prompt_context_lengths=[127, 255, 700],
        generated_tokens_per_user=[3, 3, 2],
        buckets=buckets,
    )

    assert distribution["total_decode_tokens"] == 8
    assert distribution["buckets"]["context_1_128"]["token_count"] == 2
    assert distribution["buckets"]["context_129_256"]["token_count"] == 3
    assert distribution["buckets"]["context_257_512"]["token_count"] == 1
    assert distribution["buckets"]["context_513_1024"]["token_count"] == 2
    assert sum(
        item["weight"] for item in distribution["buckets"].values()
    ) == pytest.approx(1.0)


def test_winners_write_runtime_keys_and_weighted_report(
    bucket_enumerations,
) -> None:
    runtime = _official_runtime_config()
    selections = []
    winner_ids = {}
    for index, result in enumerate(bucket_enumerations):
        official = result.enumeration.official_candidates[0]
        challenger = next(
            item
            for item in result.enumeration.candidates
            if not item.is_official
        )
        measurements = {
            official.candidate_id: _measurement(1.0 + index * 0.1),
            challenger.candidate_id: _measurement(0.9 + index * 0.09),
        }
        selection = select_sdpa_bucket_winner(result, measurements)
        selections.append(selection)
        winner_ids[result.bucket.name] = selection["winner"]["candidate_id"]

    configured = apply_sdpa_bucket_winners(
        runtime, bucket_enumerations, winner_ids
    )
    entries = configured["attention"]["sdpa_context_buckets"]
    assert len(entries) == 4
    assert len({entry["config_key"] for entry in entries}) == 4
    assert all("post_sdpa_output_memory_config" in entry for entry in entries)
    assert (
        configured["attention"]["sdpa_program_config"]
        == runtime["attention"]["sdpa_program_config"]
    )

    distribution = build_context_distribution(
        prompt_context_lengths=[64, 200, 400, 800],
        generated_tokens_per_user=1,
        buckets=build_sdpa_context_buckets(1024),
    )
    report = build_sdpa_bucket_phase_report(
        selections,
        context_distribution=distribution,
        trace_switch_overhead_ms=[0.002, 0.003],
        full_model_gain=0.02,
    )
    assert report["phase_completed"] is True
    assert report["promotion_allowed"] is True
    assert report["full_decode_weighted_average"]["gain"] == pytest.approx(0.1)
    assert report["trace_switch_overhead"]["max_ms"] == 0.003


def _measurement(latency_ms: float) -> dict:
    return {
        "status": "passed",
        "passed": True,
        "statistics": {"p50": float(latency_ms)},
    }
