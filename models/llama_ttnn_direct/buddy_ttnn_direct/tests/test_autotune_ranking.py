from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    MeasurementCandidate,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.ranking import (
    HardwareRankingModel,
    RankingExample,
    build_profiler_ranking_features,
    build_ranking_phase_report,
    evaluate_ranking_leave_one_campaign_out,
    extract_ranking_features,
    ranking_examples_from_active_report,
    train_hardware_ranking_model,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search import (
    SuccessiveHalvingPolicy,
    run_active_measurement_scheduler,
)


class RankingFeatureAndModelTest(unittest.TestCase):
    def test_requested_candidate_and_profiler_features_are_extracted(self) -> None:
        profiler = build_profiler_ranking_features(_profiler_report())
        features = extract_ranking_features(
            _candidate_mapping("candidate-0", score=8.0),
            profiler_features=profiler,
        )

        self.assertEqual(features["operator_kind"], "mlp_projection")
        self.assertEqual(features["program_family"], "reuse_multicast_1d")
        self.assertIn("in0_block_w", features["program_signature"])
        self.assertGreater(features["work_tiles_per_core_log"], 0.0)
        self.assertGreater(features["estimated_dram_bytes_log"], 0.0)
        self.assertEqual(features["fpu_util_percent"], 72.0)
        self.assertEqual(features["cb_wait_fraction"], 0.1)

    def test_model_round_trip_preserves_predictions_and_order(self) -> None:
        examples = _ranking_examples(campaigns=3, candidates=10)
        model = train_hardware_ranking_model(examples, ridge_strength=0.1)
        restored = HardwareRankingModel.from_dict(model.to_dict())
        candidates = [
            _candidate_mapping(f"candidate-{index}", score=float(index + 1))
            for index in reversed(range(10))
        ]

        self.assertAlmostEqual(
            model.predict_candidate(candidates[0]),
            restored.predict_candidate(candidates[0]),
        )
        self.assertEqual(
            [item["candidate_id"] for item in restored.rank_candidates(candidates)],
            [f"candidate-{index}" for index in range(10)],
        )
        metadata = restored.scheduler_metadata()
        self.assertEqual(
            metadata["target"],
            "within_campaign_hardware_rank_percentile",
        )
        self.assertEqual(metadata["hardware_model_weight"], 0.5)
        self.assertTrue(metadata["final_hardware_measurement_required"])

    def test_leave_one_campaign_out_recovers_winners_and_reduces_measurements(
        self,
    ) -> None:
        examples = _ranking_examples(campaigns=4, candidates=10)
        model = train_hardware_ranking_model(examples, ridge_strength=0.1)
        evaluation = evaluate_ranking_leave_one_campaign_out(
            examples,
            ridge_strength=0.1,
        )
        report = build_ranking_phase_report(
            examples=examples,
            model=model,
            evaluation=evaluation,
            historical_sources=["fixture"],
        )

        self.assertTrue(evaluation["all_winners_recovered"])
        self.assertEqual(evaluation["top_k_winner_recall"], 1.0)
        self.assertGreaterEqual(evaluation["measurement_reduction"], 0.3)
        self.assertGreater(evaluation["mean_spearman_rank_correlation"], 0.9)
        self.assertTrue(report["phase_completed"])
        self.assertFalse(report["usage"]["substitutes_final_hardware_measurement"])

    def test_historical_scheduler_report_becomes_training_examples(self) -> None:
        report = {
            "operators": {
                "mlp.gate": {
                    "analytical_ranking": [
                        _candidate_mapping("a", score=1.0),
                        _candidate_mapping("b", score=2.0),
                    ],
                    "rounds": [
                        {
                            "measurements": {
                                "a": {
                                    "passed": True,
                                    "latency_ms": 0.2,
                                },
                                "b": {
                                    "passed": True,
                                    "statistics": {"p50": 0.3},
                                },
                            }
                        }
                    ],
                }
            }
        }
        examples = ranking_examples_from_active_report(
            report,
            campaign_name="phase3",
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].campaign_id, "phase3:mlp.gate")
        self.assertEqual({item.latency_ms for item in examples}, {0.2, 0.3})


class RankingSchedulerIntegrationTest(unittest.TestCase):
    def test_model_orders_round_one_without_replacing_measurement(self) -> None:
        candidates = tuple(
            MeasurementCandidate.create(
                candidate_id=f"candidate-{index}",
                operator_name="mlp.gate",
                candidate_kind="matmul",
                analytical_score=float(index),
                l1_bytes=1024,
                source="fixture",
                is_incumbent=index == 5,
            )
            for index in range(6)
        )

        class ReverseRanker:
            def rank_candidates(self, values):
                return sorted(
                    values,
                    key=lambda item: item.analytical_score,
                    reverse=True,
                )

            def scheduler_metadata(self):
                return {
                    "model_kind": "fixture",
                    "fingerprint": "fixture-ranking-model",
                    "final_hardware_measurement_required": True,
                }

        calls = []

        def measure(candidate, contract, round_name):
            calls.append((candidate.candidate_id, round_name))
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
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_active_measurement_scheduler(
                candidate_groups={"mlp.gate": candidates},
                statically_rejected={},
                measurement_runner=measure,
                ranking_model=ReverseRanker(),
                policy=SuccessiveHalvingPolicy(
                    minimum_coverage=(("mlp.gate", 3),),
                    short_batch_size=3,
                ),
                out=Path(tmpdir) / "report.json",
            )

        first_batch = report["operators"]["mlp.gate"]["rounds"][0]["active_batches"][0]
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["algorithm"], "hardware_ranked_active_successive_halving"
        )
        self.assertEqual(
            first_batch["candidate_ids"],
            ["candidate-5", "candidate-4", "candidate-3"],
        )
        self.assertEqual(
            first_batch["ranking_update"]["method"],
            "hardware_calibrated_ranking_model",
        )
        self.assertTrue(calls)


def _ranking_examples(*, campaigns: int, candidates: int):
    result = []
    for campaign in range(campaigns):
        scale = 1.0 + campaign / 10.0
        for index in range(candidates):
            candidate = _candidate_mapping(
                f"campaign-{campaign}-candidate-{index}",
                score=float(index + 1),
                incumbent=index == candidates - 1,
            )
            result.append(
                RankingExample.create(
                    campaign_id=f"campaign-{campaign}",
                    candidate_id=candidate["candidate_id"],
                    operator_name="mlp.gate",
                    latency_ms=scale * (0.2 + index / 100.0),
                    features=extract_ranking_features(candidate),
                )
            )
    return result


def _candidate_mapping(
    candidate_id: str,
    *,
    score: float,
    incumbent: bool = False,
):
    return {
        "candidate_id": candidate_id,
        "operator": "mlp.gate",
        "candidate_kind": "matmul",
        "analytical_score": score,
        "l1_bytes": 131072,
        "source": "fixture",
        "is_incumbent": incumbent,
        "metadata": {
            "program_family": "reuse_multicast_1d",
            "workloads": [
                {
                    "m_tiles": 1,
                    "k_tiles": 128,
                    "n_tiles": 448,
                    "active_cores": 64,
                    "in0_block_w": score,
                }
            ],
            "candidate": {
                "operator": "mlp.gate",
                "program_family": "reuse_multicast_1d",
                "hot_path_conversion_count": 0,
                "worker_core_counts": [64],
                "programs": [
                    {
                        "program_family": "reuse_multicast_1d",
                        "compute_grid": [8, 8],
                        "in0_block_w": score,
                        "per_core_m": 1,
                        "per_core_n": 7,
                        "out_block_h": 1,
                        "out_block_w": 7,
                        "out_subblock_h": 1,
                        "out_subblock_w": 7,
                    }
                ],
            },
        },
    }


def _profiler_report():
    return {
        "regions": [
            {
                "region": "gate_linear",
                "device_kernel_latency_ms": 1.0,
                "hardware_metrics": {
                    "fpu_util_percent": 72.0,
                    "cb_wait_front_ms": 0.1,
                    "noc_congestion_impact_percent": 4.0,
                    "packer_stall_ms": 0.02,
                    "unpacker_stall_ms": 0.03,
                    "achieved_weight_bandwidth_gbps": 100.0,
                },
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
