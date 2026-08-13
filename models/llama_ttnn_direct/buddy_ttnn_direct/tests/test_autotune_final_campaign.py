from __future__ import annotations

import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.final_campaign import (
    FinalCampaignPolicy,
    build_final_campaign_report,
)


class FinalCampaignReportTest(unittest.TestCase):
    def test_stable_m1_result_is_partial_success_not_target_met(self) -> None:
        report = build_final_campaign_report(
            confirmation=_confirmation(
                incumbent=(33.9480, 33.9460, 33.9295),
                challenger=(35.5603, 35.6165, 35.5931),
                challenger_latency=(28.0696, 28.0273, 28.0520),
            ),
            correctness=_correctness(),
            generalization={"passed": True},
            search_evidence=_search_evidence(),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "partial_success")
        self.assertFalse(report["goal_achieved"])
        self.assertTrue(report["acceptance"]["promotion_allowed"])
        self.assertTrue(report["acceptance"]["partial_success"])
        self.assertFalse(report["acceptance"]["target_met"])
        self.assertEqual(
            report["performance"]["milestones"]["highest_passed"],
            "M1",
        )
        self.assertAlmostEqual(
            report["performance"]["relative_improvement"],
            0.0485,
            places=3,
        )

    def test_target_requires_throughput_latency_and_acceptance(self) -> None:
        report = build_final_campaign_report(
            confirmation=_confirmation(
                incumbent=(33.94, 33.95, 33.93),
                challenger=(37.50, 37.52, 37.48),
                challenger_latency=(26.65, 26.66, 26.64),
            ),
            correctness=_correctness(),
            generalization={"passed": True},
            search_evidence=_search_evidence(),
        )

        self.assertEqual(report["status"], "target_met")
        self.assertTrue(report["goal_achieved"])
        self.assertTrue(report["acceptance"]["target_met"])
        self.assertEqual(
            report["performance"]["milestones"]["highest_passed"],
            "M3",
        )

    def test_complete_campaign_without_gain_is_not_promoted(self) -> None:
        report = build_final_campaign_report(
            confirmation=_confirmation(
                incumbent=(33.94, 33.95, 33.93),
                challenger=(33.95, 33.94, 33.93),
                challenger_latency=(29.45, 29.46, 29.47),
            ),
            correctness=_correctness(),
            generalization={"passed": True},
            search_evidence=_search_evidence(),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "completed_no_promotion")
        self.assertFalse(report["goal_achieved"])
        self.assertFalse(report["acceptance"]["promotion_allowed"])
        self.assertFalse(report["acceptance"]["checks"]["minimum_relative_improvement"])

    def test_precision_mutation_and_bad_order_fail_acceptance(self) -> None:
        confirmation = _confirmation(
            incumbent=(33.94, 33.95, 33.93),
            challenger=(37.50, 37.52, 37.48),
            challenger_latency=(26.65, 26.66, 26.64),
        )
        confirmation["measurement_contract"]["order"] = [
            "incumbent_0",
            "challenger_0",
            "incumbent_1",
            "challenger_1",
            "incumbent_2",
            "challenger_2",
        ]
        correctness = _correctness()
        correctness["precision_contract"]["mutated"] = True

        report = build_final_campaign_report(
            confirmation=confirmation,
            correctness=correctness,
            generalization={"passed": True},
            search_evidence=_search_evidence(),
        )

        self.assertEqual(report["status"], "failed_acceptance")
        self.assertFalse(report["passed"])
        self.assertFalse(report["acceptance"]["promotion_allowed"])
        self.assertIn(
            "measurement_contract",
            report["acceptance"]["failed_checks"],
        )
        self.assertIn(
            "precision_contract_unchanged",
            report["acceptance"]["failed_checks"],
        )

    def test_policy_rejects_non_final_measurement_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed to 5 warmup"):
            FinalCampaignPolicy(iterations=99)


def _confirmation(
    *,
    incumbent: tuple[float, float, float],
    challenger: tuple[float, float, float],
    challenger_latency: tuple[float, float, float],
) -> dict[str, object]:
    records = {}
    for label, values, latencies in (
        ("incumbent", incumbent, (29.43, 29.44, 29.45)),
        ("challenger", challenger, challenger_latency),
    ):
        for index, (value, latency) in enumerate(zip(values, latencies)):
            records[f"{label}_{index}"] = {
                "passed": True,
                "tokens_per_second_per_user": value,
                "decode_step_ms_p50": latency,
                "device_seconds": 100.0,
                "report": f"{label}_{index}.json",
            }
    return {
        "measurement_contract": {
            "warmup": 5,
            "iterations": 100,
            "repetitions": 3,
            "order": [
                "incumbent_0",
                "challenger_0",
                "challenger_1",
                "incumbent_1",
                "incumbent_2",
                "challenger_2",
            ],
            "execution_mode": "trace",
            "runtime_input_mode": "persistent",
            "after_prefill": True,
            "isolated_subprocess": True,
        },
        "records": records,
    }


def _correctness() -> dict[str, object]:
    return {
        "full_depth_functional": {"passed": True},
        "all_bf16_regression": {"passed": True},
        "performance_recipe_quality": {"passed": True},
        "precision_contract": {
            "mutated": False,
            "expected_hash": "a" * 64,
            "observed_hash": "a" * 64,
        },
    }


def _search_evidence() -> dict[str, object]:
    return {
        "candidate_counts": {
            "matmul": {"enumerated": 126, "legal": 100, "measured": 60},
            "sdpa": {"enumerated": 226, "legal": 200, "measured": 30},
        },
        "device_seconds": 3600.0,
        "winner_discovery_path": ["enumerate", "microbench", "full_model"],
        "profiler_attribution": {"matmul_percent": 50.0},
        "contributions": {"linears_percent": 4.0},
        "prefetch_contribution": {"percent": 0.0},
        "ablation": [{"name": "without_linears", "delta_percent": -4.0}],
        "failures": [{"class": "runtime_api_error", "count": 1}],
    }


if __name__ == "__main__":
    unittest.main()
