from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    DEFAULT_LAYER_GROUP,
    OVERRIDE_LAYER_GROUP,
    PACKED_GATE_UP,
    PACKED_GATE_UP_OPERATOR,
    DeviceDescriptor,
    PrecisionContract,
    apply_packed_gate_up_candidate,
    apply_packed_gate_up_layer_group_candidates,
    build_packed_gate_up_phase_report,
    enumerate_packed_gate_up_candidates,
    rank_packed_gate_up_region_candidates,
    select_packed_gate_up_region_winner,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.templates import (
    GATE_UP_AXIS,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    _official_runtime_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class PackedGateUpTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = _official_runtime_config()
        template = json.loads(
            (PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json").read_text()
        )
        cls.precision = PrecisionContract.from_template_config(template)
        cls.device = DeviceDescriptor.p150a()
        cls.default = enumerate_packed_gate_up_candidates(
            runtime_config=cls.runtime,
            device=cls.device,
            precision_contract=cls.precision,
            representative_layer=0,
        )
        cls.override = enumerate_packed_gate_up_candidates(
            runtime_config=cls.runtime,
            device=cls.device,
            precision_contract=cls.precision,
            representative_layer=31,
        )

    def test_independent_operator_covers_program_and_layout_fields(self) -> None:
        for result, group, dtype in (
            (self.default, DEFAULT_LAYER_GROUP, "bfloat4_b"),
            (self.override, OVERRIDE_LAYER_GROUP, "bfloat8_b"),
        ):
            with self.subTest(group=group):
                report = result.to_dict()
                coverage = report["search_field_coverage"]
                self.assertEqual(result.status, "passed")
                self.assertEqual(report["operator"], PACKED_GATE_UP_OPERATOR)
                self.assertEqual(report["layer_group"], group)
                self.assertEqual(result.matmul.workloads[0].weight_dtype, dtype)
                self.assertGreaterEqual(len(result.candidates), 8)
                self.assertEqual(
                    set(coverage["program_family"]),
                    {"dram_sharded", "reuse_multicast_1d"},
                )
                self.assertGreaterEqual(len(coverage["in0_block_w"]), 2)
                self.assertGreaterEqual(len(coverage["per_core_N"]), 2)
                self.assertGreaterEqual(len(coverage["grid"]), 2)
                self.assertGreaterEqual(len(coverage["worker_cores"]), 2)
                self.assertEqual(set(coverage["split_strategy"]), {"split", "slice"})
                self.assertEqual(
                    set(coverage["split_output_layout"]),
                    {"interleaved"},
                )
                self.assertEqual(
                    set(coverage["mul_input_layout"]),
                    {"interleaved"},
                )
                self.assertEqual(
                    set(report["attempted_layouts"]["split_output"]),
                    {"width_sharded", "interleaved"},
                )
                self.assertTrue(
                    any(
                        item["field"] == "linear_output_memory"
                        for item in report["rejected"]
                        if item.get("field") is not None
                    )
                )

    def test_layer_precision_groups_have_distinct_candidate_identity(self) -> None:
        self.assertNotEqual(
            {item.candidate_id for item in self.default.candidates},
            {item.candidate_id for item in self.override.candidates},
        )
        self.assertNotEqual(
            self.default.incumbent_candidate_id,
            self.override.incumbent_candidate_id,
        )
        self.assertEqual(self.default.precision_contract_hash, self.precision.hash)
        self.assertEqual(self.override.precision_contract_hash, self.precision.hash)

    def test_candidate_application_preserves_independently_selected_program(
        self,
    ) -> None:
        candidate = next(
            item
            for item in self.default.candidates
            if item.program_family == "reuse_multicast_1d"
            and item.split_strategy == "slice"
            and not item.requires_mul_conversion
        )
        runtime = copy.deepcopy(self.runtime)
        runtime["attention"]["context_bucket_marker"] = "preserve-me"
        runtime["autotune"] = {
            "fused_attention_layout": {"candidate_id": "fused-winner"}
        }
        original_attention = copy.deepcopy(runtime["attention"])
        configured = apply_packed_gate_up_candidate(runtime, candidate)
        mlp = configured["mlp"]

        self.assertEqual(configured["attention"], original_attention)
        self.assertEqual(
            configured["autotune"]["fused_attention_layout"]["candidate_id"],
            "fused-winner",
        )
        self.assertEqual(
            configured["autotune"]["templates"][GATE_UP_AXIS], PACKED_GATE_UP
        )
        self.assertEqual(
            mlp["packed_gate_up_program_config"],
            candidate.matmul_candidate.programs[0].to_runtime_descriptor(),
        )
        self.assertEqual(mlp["packed_gate_up_split_strategy"], "slice")
        self.assertFalse(mlp["packed_gate_up_mul_conversion"])
        self.assertEqual(
            mlp["packed_gate_up_split_output_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )
        self.assertEqual(
            mlp["packed_gate_up_mul_input_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )

    def test_region_ranking_includes_incumbent_and_every_challenger(self) -> None:
        ranked = rank_packed_gate_up_region_candidates(self.default)

        self.assertEqual(ranked[0].candidate_id, self.default.incumbent_candidate_id)
        self.assertTrue(ranked[0].is_incumbent)
        self.assertEqual(len(ranked), len(self.default.candidates) + 1)
        self.assertEqual(
            {item.candidate_id for item in ranked[1:]},
            {item.candidate_id for item in self.default.candidates},
        )

    def test_full_model_config_preserves_layer_specific_split_strategy(self) -> None:
        default = next(
            item
            for item in self.default.candidates
            if item.program_family == "reuse_multicast_1d"
            and item.split_strategy == "split"
        )
        override = next(
            item
            for item in self.override.candidates
            if item.program_family == "reuse_multicast_1d"
            and item.split_strategy == "slice"
            and item.matmul_candidate.programs[0].to_dict()
            == default.matmul_candidate.programs[0].to_dict()
        )

        configured = apply_packed_gate_up_layer_group_candidates(
            self.runtime,
            {
                DEFAULT_LAYER_GROUP: default,
                OVERRIDE_LAYER_GROUP: override,
            },
        )
        mlp = configured["mlp"]
        layer_31 = mlp["layer_overrides"]["31"]
        self.assertEqual(mlp["packed_gate_up_split_strategy"], "split")
        self.assertEqual(layer_31["packed_gate_up_split_strategy"], "slice")
        self.assertEqual(
            layer_31["packed_gate_up_program_config"],
            override.matmul_candidate.programs[0].to_runtime_descriptor(),
        )
        self.assertIn("gate_up_compute_kernel_config", layer_31)
        self.assertEqual(
            configured["parameter_config"]["weight_memory_config"]["mlp_gate_up"][
                "name"
            ],
            "DRAM_MEMORY_CONFIG",
        )

    def test_region_gate_and_full_model_gate_are_kept_separate(self) -> None:
        selections = []
        for result in (self.default, self.override):
            winner = result.candidates[0]
            measurements = {
                result.incumbent_candidate_id: _measurement(1.0),
                winner.candidate_id: _measurement(0.96),
            }
            selection = select_packed_gate_up_region_winner(result, measurements)
            self.assertAlmostEqual(selection["representative_region_gain"], 0.04)
            self.assertTrue(selection["region_gate_passed"])
            self.assertFalse(selection["default_promotion_allowed"])
            selections.append(selection)

        region_only = build_packed_gate_up_phase_report(selections)
        promoted = build_packed_gate_up_phase_report(selections, full_model_gain=0.011)
        self.assertTrue(region_only["enter_full_model"])
        self.assertFalse(region_only["default_promotion_allowed"])
        self.assertTrue(promoted["default_promotion_allowed"])
        self.assertAlmostEqual(promoted["weighted_region"]["gain"], 0.04)

    def test_region_gain_below_three_percent_does_not_enter_full_model(self) -> None:
        result = self.default
        selection = select_packed_gate_up_region_winner(
            result,
            {
                result.incumbent_candidate_id: _measurement(1.0),
                result.candidates[0].candidate_id: _measurement(0.971),
            },
        )

        self.assertFalse(selection["region_gate_passed"])
        self.assertFalse(selection["enter_full_model"])


def _measurement(latency_ms: float) -> dict:
    return {
        "status": "passed",
        "statistics": {
            "mean": latency_ms,
            "p50": latency_ms,
            "p90": latency_ms,
        },
    }


if __name__ == "__main__":
    unittest.main()
