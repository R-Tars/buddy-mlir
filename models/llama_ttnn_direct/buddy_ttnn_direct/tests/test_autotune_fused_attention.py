from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.fused_attention import (
    FUSED_ATTENTION_REGION,
    apply_fused_attention_layout_candidate,
    build_fused_attention_phase_report,
    build_fused_attention_region_payload,
    enumerate_fused_attention_layout_candidates,
    rank_fused_attention_region_candidates,
    select_fused_attention_region_winner,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.legality import (
    DeviceDescriptor,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema import PrecisionContract
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.templates import (
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    KV_UPDATE_AXIS,
    ROPE_AXIS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.plans import (
    decode_step_plan,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    _official_runtime_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def precision() -> PrecisionContract:
    template = json.loads(
        (PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json").read_text()
    )
    return PrecisionContract.from_template_config(template)


@pytest.fixture(scope="module")
def enumeration(precision: PrecisionContract):
    return enumerate_fused_attention_layout_candidates(
        runtime_config=_official_runtime_config(),
        device=DeviceDescriptor.p150a(),
        precision_contract=precision,
    )


def test_joint_candidate_covers_producer_rope_constants_and_cache(
    enumeration,
) -> None:
    report = enumeration.to_dict()
    candidate = enumeration.candidates[0]

    assert enumeration.status == "passed"
    assert report["operator"] == FUSED_ATTENTION_REGION
    assert report["legal_candidate_count"] == 1
    assert report["conversion_bound_count"] == 1
    assert set(report["search_unit"]) == {
        "qkv_linear_output_sharding",
        "create_heads_output_layout",
        "q_k_core_placement",
        "persistent_rope_constant_placement",
        "fused_qk_rope",
        "cache_update_input_layout",
        "fused_kv_update",
    }
    assert candidate.q_core_ranges == candidate.v_core_ranges
    assert candidate.q_core_ranges != candidate.k_core_ranges
    assert candidate.persistent_core_ranges == (
        (0, 0, 7, 3),
        (0, 4, 7, 7),
    )
    assert candidate.added_conversion_count == 0
    assert candidate.removed_conversion_count == 3
    assert candidate.operation_count_reduction == 2
    assert candidate.qkv_program_config["per_core_N"] == 4
    assert candidate.legality.passed


def test_api_without_disjoint_producer_is_conversion_bound(
    precision: PrecisionContract,
) -> None:
    result = enumerate_fused_attention_layout_candidates(
        runtime_config=_official_runtime_config(),
        device=DeviceDescriptor.p150a(),
        precision_contract=precision,
        producer_supports_disjoint_qk=False,
    )

    assert result.status == "conversion_bound"
    assert result.candidates == ()
    assert any(
        reason["code"] == "CREATE_HEADS_DISJOINT_QK_API_UNAVAILABLE"
        for item in result.rejected
        for reason in item.get("reasons", [])
    )
    assert all(item.get("enter_full_model") is False for item in result.rejected)


def test_candidate_application_reaches_decode_plan_without_layout_drift(
    enumeration,
) -> None:
    runtime = _official_runtime_config()
    candidate = enumeration.candidates[0]
    configured = apply_fused_attention_layout_candidate(runtime, candidate)
    attention = configured["attention"]
    plan = decode_step_plan(
        layers=32,
        batch_size=32,
        cache_len=1024,
        config=configured,
    )

    assert configured["autotune"]["templates"][ROPE_AXIS] == FUSED_QK_ROPE
    assert (
        configured["autotune"]["templates"][KV_UPDATE_AXIS]
        == FUSED_PAGED_UPDATE
    )
    assert attention["qkv_heads_overlap_qk_coregrid"] is False
    assert attention["qkv_program_config"]["per_core_N"] == 4
    assert (
        attention["fused_k_memory_config"]
        == attention["fused_cache_key_memory_config"]
    )
    assert (
        attention["fused_q_memory_config"]
        == attention["fused_cache_value_memory_config"]
    )
    assert plan["templates"][ROPE_AXIS] == FUSED_QK_ROPE
    assert plan["templates"][KV_UPDATE_AXIS] == FUSED_PAGED_UPDATE
    assert (
        plan["rotary_memory_configs"]["cos_sin"]
        == attention["fused_rope_cos_sin_memory_config"]
    )
    assert (
        plan["rotary_memory_configs"]["transformation"]
        == attention["fused_rope_transform_memory_config"]
    )


def test_region_payload_and_ranking_include_both_arms(enumeration) -> None:
    runtime = _official_runtime_config()
    candidate = enumeration.candidates[0]
    ranked = rank_fused_attention_region_candidates(enumeration)
    incumbent_payload = build_fused_attention_region_payload(
        enumeration=enumeration,
        candidate_id=enumeration.incumbent_candidate_id,
        runtime_config=runtime,
        physical_cache_len=128,
    )
    challenger_payload = build_fused_attention_region_payload(
        enumeration=enumeration,
        candidate_id=candidate.candidate_id,
        runtime_config=runtime,
        physical_cache_len=128,
    )

    assert [item.candidate_id for item in ranked] == [
        enumeration.incumbent_candidate_id,
        candidate.candidate_id,
    ]
    assert ranked[0].is_incumbent
    assert incumbent_payload["mode"] == "incumbent"
    assert incumbent_payload["physical_cache_len"] == 128
    assert incumbent_payload["operation_count_reduction"] == 0
    assert incumbent_payload["operation_sequence"][0] == "linear.qkv_packed"
    assert incumbent_payload["incumbent_qkv_program_config"]["per_core_N"] == 6
    assert challenger_payload["mode"] == "challenger"
    assert challenger_payload["operation_count_reduction"] == 2
    assert challenger_payload["added_conversion_count"] == 0
    assert challenger_payload["hidden_input_memory"]["shard_shape"] == [32, 128]
    assert challenger_payload["weight_memory"] == {
        "kind": "ttnn_dram_sharded_memory_config",
        "k": 4096,
        "n": 6144,
        "dram_grid_width": 8,
    }
    assert challenger_payload["challenger_qkv_program_config"]["per_core_N"] == 4


def test_region_gate_uses_measured_net_latency(enumeration) -> None:
    candidate = enumeration.candidates[0]
    passing = select_fused_attention_region_winner(
        enumeration,
        {
            enumeration.incumbent_candidate_id: _measurement(1.0),
            candidate.candidate_id: _measurement(0.98),
        },
    )
    below_gate = select_fused_attention_region_winner(
        enumeration,
        {
            enumeration.incumbent_candidate_id: _measurement(1.0),
            candidate.candidate_id: _measurement(0.995),
        },
    )
    report = build_fused_attention_phase_report(
        passing,
        full_model_gain=0.003,
        full_model_evidence={"status": "profiled"},
    )

    assert passing["net_attention_subregion_gain"] == pytest.approx(0.02)
    assert passing["region_gate_passed"] is True
    assert passing["retained_candidate_id"] == candidate.candidate_id
    assert below_gate["region_gate_passed"] is False
    assert below_gate["retained_candidate_id"] is None
    assert report["phase_completed"] is True
    assert report["candidate_retained"] is True
    assert report["full_model_gain"] == pytest.approx(0.003)


def _measurement(latency_ms: float) -> dict:
    return {
        "status": "passed",
        "passed": True,
        "statistics": {"p50": float(latency_ms)},
    }
