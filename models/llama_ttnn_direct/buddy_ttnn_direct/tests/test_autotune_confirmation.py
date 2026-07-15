from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    CandidateConfig,
    ConfirmationArm,
    ConfirmationPolicy,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    confirm_matched_ab,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class MatchedABConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = json.loads(
            (PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json").read_text()
        )
        cls.precision = PrecisionContract.from_template_config(cls.template)
        cls.execution = ExecutionContract.from_template_config(
            cls.template,
            prompt_corpus_sha256=hashlib.sha256(b"phase8").hexdigest(),
        )

    def test_stable_two_percent_challenger_is_promoted(self) -> None:
        incumbent = self._arm(
            "incumbent",
            values=(100.0, 100.1, 99.9),
            tunable="official",
        )
        challenger = self._arm(
            "challenger",
            values=(102.0, 102.1, 101.9),
            tunable="candidate",
        )

        report = confirm_matched_ab(
            incumbent=incumbent,
            challenger=challenger,
        )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertTrue(report["matched_ab"])
        self.assertTrue(report["promotion"]["promoted"])
        self.assertEqual(report["promotion"]["selected_arm"], "challenger")
        self.assertAlmostEqual(report["relative_improvement"], 0.02)
        self.assertLess(report["arms"]["incumbent"]["cv"], 0.015)
        self.assertLess(report["arms"]["challenger"]["cv"], 0.015)

    def test_below_one_percent_retains_incumbent(self) -> None:
        report = confirm_matched_ab(
            incumbent=self._arm(
                "incumbent",
                values=(100.0, 100.0, 100.0),
                tunable="official",
            ),
            challenger=self._arm(
                "challenger",
                values=(100.5, 100.5, 100.5),
                tunable="candidate",
            ),
        )

        self.assertTrue(report["passed"])
        self.assertFalse(report["promotion"]["promoted"])
        self.assertEqual(report["promotion"]["selected_arm"], "incumbent")
        self.assertIn(
            "confirmation.minimum_relative_improvement",
            report["failed_checks"],
        )

    def test_contract_stability_and_quality_failures_are_classified(self) -> None:
        incumbent = self._arm(
            "incumbent",
            values=(100.0, 100.0, 100.0),
            tunable="official",
        )
        challenger = self._arm(
            "challenger",
            values=(80.0, 120.0, 106.0),
            tunable="candidate",
            correctness=False,
            quality=False,
            batch_size=1,
        )

        report = confirm_matched_ab(
            incumbent=incumbent,
            challenger=challenger,
            policy=ConfirmationPolicy(),
        )

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["passed"])
        self.assertFalse(report["promotion"]["promoted"])
        self.assertIn(
            "confirmation.frozen_candidate_contracts_match",
            report["failed_checks"],
        )
        self.assertIn("confirmation.challenger.cv", report["failed_checks"])
        self.assertIn(
            "confirmation.challenger.correctness",
            report["failed_checks"],
        )
        self.assertIn(
            "confirmation.challenger.quality",
            report["failed_checks"],
        )

    def test_report_measurement_counts_cannot_drift(self) -> None:
        incumbent = self._arm(
            "incumbent",
            values=(100.0, 100.0, 100.0),
            tunable="official",
        )
        reports = list(incumbent.reports)
        reports[1] = {**reports[1], "iterations": 99}
        invalid_incumbent = ConfirmationArm(
            label="incumbent",
            candidate=incumbent.candidate,
            reports=tuple(reports),
            correctness_passed=True,
            quality_passed=True,
        )

        report = confirm_matched_ab(
            incumbent=invalid_incumbent,
            challenger=self._arm(
                "challenger",
                values=(102.0, 102.0, 102.0),
                tunable="candidate",
            ),
        )

        self.assertFalse(report["passed"])
        self.assertIn(
            "confirmation.incumbent.reports_valid",
            report["failed_checks"],
        )

    def _arm(
        self,
        label: str,
        *,
        values: tuple[float, float, float],
        tunable: str,
        correctness: bool = True,
        quality: bool = True,
        batch_size: int = 32,
    ) -> ConfirmationArm:
        contract = MeasurementContract.final_confirmation()
        candidate = CandidateConfig.create(
            precision_contract=self.precision,
            expected_precision_hash=self.precision.hash,
            execution_contract=self.execution,
            measurement_contract=contract,
            semantic_graph_sha256="a" * 64,
            model_config_sha256="b" * 64,
            weights_recipe_sha256="c" * 64,
            runtime_commit="phase8-test-runtime",
            device_descriptor={
                "device": "p150a",
                "device_id": 0,
                "architecture": "blackhole",
            },
            target={
                "batch_size": batch_size,
                "cache_len": 1024,
                "page_block_size": 32,
            },
            tunable_state={"candidate_space_sha256": tunable},
        )
        reports = tuple(
            {
                "status": "profiled",
                "passed": True,
                "warmup": contract.warmup,
                "iterations": contract.iterations,
                "execution_mode": "trace",
                "runtime_input_mode": "persistent",
                "after_prefill": True,
                "tokens_per_second_per_user": value,
            }
            for value in values
        )
        return ConfirmationArm(
            label=label,
            candidate=candidate,
            reports=reports,
            correctness_passed=correctness,
            quality_passed=quality,
        )


if __name__ == "__main__":
    unittest.main()
