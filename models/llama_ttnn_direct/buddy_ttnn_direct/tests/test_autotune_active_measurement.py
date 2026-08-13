from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    MeasurementCandidate,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema import (
    MeasurementContract,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search import (
    ACTIVE_MEASUREMENT_MINIMUM_COVERAGE,
    SuccessiveHalvingPolicy,
    run_active_measurement_scheduler,
)


class ActiveMeasurementSchedulerTest(unittest.TestCase):
    def test_active_batches_successive_halving_and_distinct_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            groups, rejected = _campaign_candidates(extra=3)
            calls: list[tuple[str, str, str, tuple[int, int, int]]] = []
            failed_once = {"attention.qkv-candidate-1"}

            def measure(candidate, contract, round_name):
                calls.append(
                    (
                        candidate.operator_name,
                        candidate.candidate_id,
                        round_name,
                        (contract.warmup, contract.iterations, contract.repetitions),
                    )
                )
                failed = (
                    round_name == "round_1_short"
                    and candidate.candidate_id in failed_once
                )
                latency = 2.0 + candidate.analytical_score / 100.0
                return {
                    "status": "failed" if failed else "passed",
                    "passed": not failed,
                    "measurement_contract": contract.to_dict(),
                    "worker": {"isolated_subprocess": True},
                    "statistics": {
                        "mean": latency,
                        "p50": latency,
                        "p90": latency * 1.01,
                        "coefficient_of_variation": 0.001,
                        "coefficient_of_variation_percent": 0.1,
                        "sample_count": contract.iterations * contract.repetitions,
                    },
                    "cache": {"hit": False},
                    "device_seconds": 0.0,
                }

            out = root / "active_measurement_report.json"
            report = run_active_measurement_scheduler(
                candidate_groups=groups,
                statically_rejected=rejected,
                measurement_runner=measure,
                out=out,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertTrue(out.is_file())
            self.assertFalse(
                report["intermediate_selection"][
                    "minimum_relative_improvement_gate_applied"
                ]
            )
            expected_short = sum(dict(ACTIVE_MEASUREMENT_MINIMUM_COVERAGE).values())
            self.assertEqual(report["counts"]["measured_short"], expected_short)
            self.assertEqual(
                report["counts"]["measured_short_attempted"],
                expected_short + 1,
            )
            self.assertEqual(report["counts"]["statically_rejected"], 7)
            self.assertGreater(report["counts"]["analytically_pruned"], 0)
            self.assertEqual(report["counts"]["full_model_selected"], 14)
            self.assertEqual(report["counts"]["full_model_measured"], 0)
            self.assertTrue(
                report["acceptance"]["invariants"]["enumerated_not_used_as_measured"]
            )

            qkv = report["operators"]["attention.qkv"]
            self.assertEqual(qkv["counts"]["measured_short"], 8)
            self.assertEqual(qkv["counts"]["measured_short_attempted"], 9)
            self.assertEqual(qkv["rounds"][0]["passed_candidate_count"], 8)
            self.assertTrue(
                qkv["rounds"][0]["active_batches"][0]["ranking_update"][
                    "ranking_updated"
                ]
            )
            self.assertTrue(
                qkv["rounds"][0]["active_batches"][0]["ranking_update"][
                    "next_candidate_order"
                ]
            )
            self.assertEqual(qkv["rounds"][1]["passed_candidate_count"], 2)
            self.assertEqual(qkv["rounds"][2]["passed_candidate_count"], 2)
            self.assertEqual(qkv["full_model"]["measurement_status"], "not_run")
            incumbent = next(
                candidate.candidate_id
                for candidate in groups["attention.qkv"]
                if candidate.is_incumbent
            )
            self.assertIn(incumbent, qkv["proposal_measurements"])
            self.assertIn(
                "measured_confirmation",
                qkv["candidate_lifecycle"][incumbent]["states"],
            )
            self.assertIn(
                "statically_rejected",
                qkv["candidate_lifecycle"]["attention.qkv-rejected"]["states"],
            )
            sdpa = report["operators"]["attention.sdpa"]
            self.assertEqual(sdpa["counts"]["measured_short"], 16)
            self.assertEqual(sdpa["rounds"][1]["passed_candidate_count"], 4)
            self.assertEqual(sdpa["rounds"][2]["passed_candidate_count"], 4)

            observed_contracts = {call[3] for call in calls}
            self.assertEqual(
                observed_contracts,
                {(3, 10, 1), (5, 30, 1), (5, 50, 2)},
            )

            calls.clear()

            def must_not_run(*_args):
                calls.append(("unexpected", "unexpected", "unexpected", (0, 0, 0)))
                raise AssertionError("resume should reuse every scheduler record")

            resumed = run_active_measurement_scheduler(
                candidate_groups=groups,
                statically_rejected=rejected,
                measurement_runner=must_not_run,
                out=out,
            )
            self.assertTrue(resumed["passed"])
            self.assertEqual(calls, [])
            self.assertEqual(resumed["measurement_usage"]["new_measurement_count"], 0)
            self.assertGreater(
                resumed["measurement_usage"]["reused_measurement_count"], 0
            )

    def test_insufficient_legal_coverage_is_reported_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = SuccessiveHalvingPolicy(minimum_coverage=(("attention.qkv", 8),))
            candidates = {
                "attention.qkv": tuple(
                    _candidate("attention.qkv", index, count=5) for index in range(5)
                )
            }

            def measure(candidate, contract, round_name):
                return _passed_measurement(candidate, contract, round_name)

            report = run_active_measurement_scheduler(
                candidate_groups=candidates,
                statically_rejected={"attention.qkv": []},
                measurement_runner=measure,
                out=Path(tmpdir) / "report.json",
                policy=policy,
            )

            self.assertFalse(report["passed"])
            qkv = report["operators"]["attention.qkv"]
            self.assertEqual(qkv["counts"]["legal"], 5)
            self.assertEqual(qkv["counts"]["measured_short"], 5)
            self.assertFalse(qkv["acceptance"]["checks"]["minimum_short_coverage"])

    def test_full_model_measurements_have_their_own_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = SuccessiveHalvingPolicy(minimum_coverage=(("attention.qkv", 8),))
            groups = {
                "attention.qkv": tuple(
                    _candidate("attention.qkv", index, count=10) for index in range(10)
                )
            }

            def measure(candidate, contract, round_name):
                return _passed_measurement(candidate, contract, round_name)

            def full_model(candidate):
                return {
                    "status": "passed",
                    "passed": True,
                    "isolated_subprocess": True,
                    "tokens_per_second_per_user": 35.0
                    + candidate.analytical_score / 100.0,
                    "objective": {
                        "name": "tokens_per_second_per_user",
                        "value": 35.0 + candidate.analytical_score / 100.0,
                        "direction": "maximize",
                    },
                    "device_seconds": 0.0,
                }

            report = run_active_measurement_scheduler(
                candidate_groups=groups,
                statically_rejected={"attention.qkv": []},
                measurement_runner=measure,
                full_model_runner=full_model,
                out=Path(tmpdir) / "report.json",
                policy=policy,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"]["full_model_selected"], 2)
            self.assertEqual(report["counts"]["full_model_measured"], 2)


def _campaign_candidates(extra: int):
    groups = {}
    rejected = {}
    for operator, minimum in ACTIVE_MEASUREMENT_MINIMUM_COVERAGE:
        count = minimum + extra
        groups[operator] = tuple(
            _candidate(operator, index, count=count) for index in range(count)
        )
        rejected[operator] = [
            {
                "candidate_id": f"{operator}-rejected",
                "reason": "static_legality_failure",
            }
        ]
    return groups, rejected


def _candidate(operator: str, index: int, *, count: int) -> MeasurementCandidate:
    return MeasurementCandidate.create(
        candidate_id=f"{operator}-candidate-{index}",
        operator_name=operator,
        candidate_kind="sdpa" if operator == "attention.sdpa" else "matmul",
        analytical_score=float(index + 1),
        l1_bytes=(count - index) * 1024,
        source="official" if index == count - 1 else "analytical_fixture",
        is_incumbent=index == count - 1,
        metadata={"fixture_index": index},
    )


def _passed_measurement(candidate, contract, round_name):
    latency = 1.0 + candidate.analytical_score / 100.0
    return {
        "status": "passed",
        "passed": True,
        "measurement_contract": contract.to_dict(),
        "worker": {"isolated_subprocess": True},
        "statistics": {
            "mean": latency,
            "p50": latency,
            "p90": latency,
            "sample_count": contract.iterations * contract.repetitions,
        },
        "cache": {"hit": False},
        "device_seconds": 0.0,
        "round": round_name,
    }


if __name__ == "__main__":
    unittest.main()
