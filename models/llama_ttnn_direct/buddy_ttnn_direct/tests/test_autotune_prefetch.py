from __future__ import annotations

import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.prefetch import (
    LLAMA31_8B_PREFETCH_WEIGHTS,
    audit_official_llama31_8b_support,
    audit_prefetch_scope,
    build_full_decode_prefetch_assessment,
    build_prefetch_region_ab,
    classify_prefetch_failure,
    estimate_gcb_block_bytes,
    llama31_8b_prefetch_scopes,
)


class PrefetchCapacityAuditTest(unittest.TestCase):
    def test_gcb_sizes_match_current_ttnn_prefetcher(self) -> None:
        weights = LLAMA31_8B_PREFETCH_WEIGHTS
        self.assertEqual(
            estimate_gcb_block_bytes(weights["mlp.gate"], ring_size=64),
            516_096,
        )
        self.assertEqual(
            estimate_gcb_block_bytes(weights["mlp.down"], ring_size=64),
            974_848,
        )
        self.assertEqual(
            estimate_gcb_block_bytes(weights["attention.qkv"], ring_size=64),
            417_792,
        )
        self.assertEqual(
            estimate_gcb_block_bytes(
                weights["attention.o_proj"], ring_size=64
            ),
            278_528,
        )

    def test_current_official_single_device_guard_rejects_llama(self) -> None:
        default = audit_official_llama31_8b_support()
        ring64 = audit_official_llama31_8b_support(ring_size=64)

        self.assertEqual(default["bytes_per_core"], 3_899_392)
        self.assertFalse(default["supported"])
        self.assertEqual(ring64["bytes_per_core"], 974_848)
        self.assertFalse(ring64["supported"])

    def test_attention_fits_but_mlp_and_all_linears_do_not(self) -> None:
        scopes = llama31_8b_prefetch_scopes()
        audits = {
            name: audit_prefetch_scope(name, weights)
            for name, weights in scopes.items()
        }

        self.assertTrue(audits["attention_projections"]["static_eligible"])
        self.assertEqual(
            audits["attention_projections"]["gcb_size_bytes"], 417_792
        )
        self.assertFalse(audits["mlp_only"]["static_eligible"])
        self.assertFalse(audits["all_major_linears"]["static_eligible"])
        self.assertEqual(
            audits["all_major_linears"]["weight_request_order"],
            [
                "attention.qkv",
                "attention.o_proj",
                "mlp.gate",
                "mlp.up",
                "mlp.down",
            ],
        )


class PrefetchPromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scopes = llama31_8b_prefetch_scopes()
        self.audits = {
            name: audit_prefetch_scope(name, weights)
            for name, weights in self.scopes.items()
        }

    def test_region_ab_promotes_a_stable_three_percent_gain(self) -> None:
        report = build_prefetch_region_ab(
            off_samples_ms=[0.285, 0.286, 0.287, 0.286],
            on_samples_ms=[0.235, 0.236, 0.237, 0.236],
            off_correctness_passed=True,
            on_correctness_passed=True,
        )

        self.assertEqual(report["status"], "passed")
        self.assertGreater(report["median_gain"], 0.17)
        self.assertTrue(report["enter_full_model"])

    def test_failure_classifier_keeps_l1_and_subdevice_distinct(self) -> None:
        self.assertEqual(
            classify_prefetch_failure(
                "Statically allocated circular buffers clash with L1 buffers"
            ),
            "l1_circular_buffer_conflict",
        )
        self.assertEqual(
            classify_prefetch_failure(
                "Kernel group cores do not match sub device cores"
            ),
            "subdevice_core_set_mismatch",
        )

    def test_full_model_requires_gain_correctness_and_stability(self) -> None:
        candidate = build_full_decode_prefetch_assessment(
            scope="attention_projections",
            scope_audit=self.audits["attention_projections"],
            status="measured",
            median_gain=0.02,
            cv_percent=1.0,
            correctness_passed=True,
        )
        unstable = build_full_decode_prefetch_assessment(
            scope="attention_projections",
            scope_audit=self.audits["attention_projections"],
            status="measured",
            median_gain=0.02,
            cv_percent=1.6,
            correctness_passed=True,
        )

        self.assertTrue(candidate["promoted"])
        self.assertFalse(unstable["promoted"])

if __name__ == "__main__":
    unittest.main()
