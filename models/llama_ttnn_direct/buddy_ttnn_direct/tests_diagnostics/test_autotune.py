from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import campaign
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    candidate_fingerprint,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.model_evaluator import (
    ModelCandidateEvaluator,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema import (
    MeasurementContract,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search import (
    PIPELINE_STAGES,
    SearchBudget,
    SearchCallbacks,
    SearchCandidate,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json"


class CanonicalAutotuneCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model_path = self.root / "model"
        _write_fake_model_config(self.model_path)
        self.out = self.root / "autotune.json"

    def test_cli_dry_run_uses_canonical_pipeline_and_resumes(self) -> None:
        arguments = [
            "diagnose",
            "--stage",
            "autotune",
            "--model-path",
            str(self.model_path),
            "--config",
            str(CONFIG_PATH),
            "--layers",
            "2",
            "--dry-run",
            "--out",
            str(self.out),
        ]
        with patch.object(
            campaign.ModelCandidateEvaluator,
            "__init__",
            side_effect=AssertionError("dry-run constructed a hardware evaluator"),
        ), patch.object(
            campaign,
            "run_active_measurement_scheduler",
            side_effect=AssertionError("dry-run entered active measurement"),
        ):
            self.assertEqual(main(arguments), 0)
            first = json.loads(self.out.read_text())
            self.assertEqual(main(arguments), 0)
            resumed = json.loads(self.out.read_text())

        self.assertEqual(first["status"], "dry_run")
        self.assertTrue(first["passed"])
        self.assertEqual(first["algorithm"], "hierarchical_constrained_beam_search")
        self.assertEqual(first["pipeline_stages"], list(PIPELINE_STAGES))
        self.assertEqual(first["search_space_schema_version"], 2)
        self.assertFalse(first["cartesian_exhaustive_search"])
        self.assertNotIn("levels", first)
        self.assertNotIn("strategy", first)
        self.assertTrue(first["precision_contract"]["frozen"])
        self.assertEqual(
            {
                key: first["execution_contract"][key]
                for key in (
                    "execution_mode",
                    "runtime_input_mode",
                    "after_prefill",
                    "sampling",
                    "page_table",
                )
            },
            {
                "execution_mode": "trace",
                "runtime_input_mode": "persistent",
                "after_prefill": True,
                "sampling": "force_argmax",
                "page_table": "fixed",
            },
        )
        self.assertEqual(
            [stage["name"] for stage in first["search_report"]["stages"]],
            list(PIPELINE_STAGES),
        )
        template_groups = [
            group
            for group in first["search_report"]["proposal_groups"]
            if group["stage"] == "template"
        ]
        self.assertEqual(len(template_groups), 4)
        self.assertTrue(all(group["proposal_count"] > 0 for group in template_groups))
        self.assertEqual(
            first["confirmation_policy"],
            {
                "metric": "tokens_per_second_per_user",
                "metric_direction": "maximize",
                "minimum_relative_improvement": 0.01,
                "maximum_cv": 0.015,
                "warmup": 5,
                "iterations": 100,
                "repetitions": 3,
            },
        )
        self.assertEqual(
            first["resume_identity"]["run_id"],
            resumed["resume_identity"]["run_id"],
        )
        self.assertTrue(resumed["search_report"]["resume"]["completed_report_reused"])

    def test_schema_v2_candidate_identity_freezes_precision_and_execution(self) -> None:
        context = campaign._context(
            self.model_path,
            CONFIG_PATH,
            "identity prompt",
            2,
            None,
            None,
            None,
            "p150a",
            0,
        )
        evaluator = ModelCandidateEvaluator(
            context={
                **context,
                "prompt": "identity prompt",
                "tokenizer_path": None,
                "runtime_commit": "phase5-test-runtime",
            },
            output_root=self.root / "measurements",
            candidate_spaces={},
        )
        candidate = evaluator.candidate_config(
            SearchCandidate(space=context["base_space"]),
            MeasurementContract.final_confirmation(),
        )
        identity = candidate.to_dict()

        self.assertEqual(identity["schema_version"], 2)
        self.assertEqual(identity["tunable_state"]["space"]["schema_version"], 2)
        self.assertTrue(identity["precision_contract"]["frozen"])
        self.assertEqual(
            identity["precision_contract"]["hash"], context["precision"].hash
        )
        self.assertEqual(identity["execution_contract"], context["execution"].to_dict())
        self.assertEqual(
            identity["measurement_contract"],
            MeasurementContract.final_confirmation().to_dict(),
        )
        tunable_json = json.dumps(identity["tunable_state"], sort_keys=True)
        for marker in (
            "math_fidelity",
            "fp32_dest_acc",
            "math_approx_mode",
            "packer_l1_acc",
            "dtype_recipe",
        ):
            self.assertNotIn(marker, tunable_json)
        self.assertEqual(len(candidate_fingerprint(candidate)), 64)

        with self.assertRaisesRegex(ValueError, "frozen to 'bf16'"):
            campaign.run_autotune_campaign(
                model_path=self.model_path,
                config_path=CONFIG_PATH,
                out=self.root / "invalid-precision.json",
                prompt=None,
                layers=2,
                dtype_seed="fp32",
                dry_run=True,
            )

    def test_mock_measurement_runs_active_search_and_confirmation(self) -> None:
        calls: dict[str, list[object]] = {
            "active": [],
            "candidate": [],
            "layer": [],
            "full": [],
            "confirmation": [],
        }

        def active(candidate, contract, round_name):
            calls["active"].append((candidate.candidate_id, round_name))
            latency = max(0.001, float(candidate.analytical_score) + 1.0)
            return {
                "status": "passed",
                "passed": True,
                "statistics": {
                    "mean": latency,
                    "p50": latency,
                    "p90": latency,
                    "sample_count": contract.iterations,
                },
                "measurement_contract": contract.to_dict(),
                "worker": {"isolated_subprocess": True},
                "device_seconds": 0.0,
            }

        def active_full(candidate):
            value = 100.0 + float(candidate.analytical_score) / 100.0
            return _passed_result(value)

        def candidate_evaluator(stage, candidate):
            calls["candidate"].append((stage, candidate.candidate_id))
            return _passed_result(100.0 + len(candidate.mutation_ids))

        def layer_evaluator(candidate):
            calls["layer"].append(candidate.candidate_id)
            return _passed_result(100.0 + len(candidate.mutation_ids))

        def full_model_evaluator(candidate):
            calls["full"].append(candidate.candidate_id)
            return _passed_result(100.0 + len(candidate.mutation_ids))

        def confirmation_runner(candidate, arm, repetition, contract):
            calls["confirmation"].append((candidate.candidate_id, arm, repetition))
            value = (100.0 if arm == "incumbent" else 102.0) + (0.01 * repetition)
            return {
                **_passed_result(value),
                "warmup": contract.warmup,
                "iterations": contract.iterations,
                "execution_mode": "trace",
                "runtime_input_mode": "persistent",
                "after_prefill": True,
            }

        report = campaign.run_autotune_campaign(
            model_path=self.model_path,
            config_path=CONFIG_PATH,
            out=self.out,
            prompt="mock campaign prompt",
            layers=2,
            resume=False,
            budget=SearchBudget(
                max_candidates=512,
                max_device_minutes=1.0,
                beam_width=4,
                template_top_k=2,
                microbench_top_k=2,
                full_model_top_k=2,
            ),
            measurement_runner=active,
            active_full_model_runner=active_full,
            callbacks=SearchCallbacks(
                candidate_evaluator=candidate_evaluator,
                layer_evaluator=layer_evaluator,
                full_model_evaluator=full_model_evaluator,
                confirmation_runner=confirmation_runner,
            ),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["active_measurement"]["status"], "passed")
        search = report["search_report"]
        self.assertEqual(search["status"], "passed")
        self.assertEqual(
            [stage["name"] for stage in search["stages"]], list(PIPELINE_STAGES)
        )
        self.assertTrue(search["matched_ab_confirmation"]["promotion"]["promoted"])
        self.assertEqual(
            search["matched_ab_confirmation"]["policy"]["iterations"], 100
        )
        self.assertGreater(len(calls["active"]), 0)
        self.assertGreater(len(calls["candidate"]), 0)
        self.assertEqual(len(calls["confirmation"]), 6)

    def test_failure_report_is_persisted(self) -> None:
        report = campaign.run_autotune_campaign(
            model_path=self.root / "missing-model",
            config_path=CONFIG_PATH,
            out=self.out,
            prompt=None,
            layers=2,
            dry_run=True,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["failure_report_written"])
        self.assertEqual(report, json.loads(self.out.read_text()))
        self.assertTrue(report["error"]["type"])

    def test_non_dry_run_without_prompt_fails_before_device_work(self) -> None:
        exit_code = main(
            [
                "diagnose",
                "--stage",
                "autotune",
                "--model-path",
                str(self.model_path),
                "--config",
                str(CONFIG_PATH),
                "--layers",
                "2",
                "--out",
                str(self.out),
            ]
        )
        report = json.loads(self.out.read_text())
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertIn("requires --prompt", report["error"])


def _passed_result(value: float) -> dict[str, object]:
    return {
        "status": "passed",
        "passed": True,
        "tokens_per_second_per_user": value,
        "objective": {
            "name": "tokens_per_second_per_user",
            "value": value,
            "direction": "maximize",
        },
        "correctness_passed": True,
        "quality_passed": True,
        "isolated_subprocess": True,
        "device_seconds": 0.0,
    }


def _write_fake_model_config(model_root: Path) -> None:
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "phase5-fake-llama",
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 128256,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "max_position_embeddings": 131072,
                "tie_word_embeddings": False,
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
