from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import models.llama_ttnn_direct.buddy_ttnn_direct.autotune.campaign as campaign
import models.llama_ttnn_direct.buddy_ttnn_direct.autotune.model_evaluator as model_evaluator
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.confirmation import (
    ConfirmationArm,
    ConfirmationPolicy,
    confirm_matched_ab,
)
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
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics import (
    candidate_quality_gate,
)

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
        self.assertEqual(first["pipeline_stage_status"]["layout_beam"], "skipped")
        self.assertEqual(first["pipeline_stage_status"]["template_search"], "passed")
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
        reproduce = Path(first["search_report"]["artifacts"]["reproduce_build"])
        self.assertIn(".cli build ", reproduce.read_text())

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

    def test_default_evaluator_preserves_contract_and_fails_closed(self) -> None:
        context = campaign._context(
            self.model_path, CONFIG_PATH, "default evaluator prompt", 2,
            None, None, None, "p150a", 0,
        )
        evaluator = _make_evaluator(self.root / "default-measurement", context)
        with _fake_profile_process():
            report = evaluator.full_model_evaluator(SearchCandidate(space=context["base_space"]))

        self.assertTrue(report["passed"])
        self.assertEqual(report["warmup"], 5)
        self.assertEqual(report["iterations"], 50)
        self.assertEqual(report["execution_mode"], "trace")
        self.assertEqual(report["runtime_input_mode"], "persistent")
        self.assertTrue(report["after_prefill"])
        self.assertEqual(report["sampling"], "force_argmax")
        self.assertEqual(report["page_table"], "fixed")
        self.assertTrue(Path(report["program_dir"]).is_dir())
        self.assertEqual(report["gate_status"], "not_evaluated")
        self.assertFalse(report["correctness_passed"])
        self.assertFalse(report["quality_passed"])

    def test_default_confirmation_uses_contract_and_passed_gate(self) -> None:
        context = campaign._context(
            self.model_path, CONFIG_PATH, "confirmation prompt", 2,
            None, None, None, "p150a", 0,
        )
        evidence = self.root / "gate-evidence.json"

        def gate(**values: object) -> dict[str, object]:
            evidence.write_text("evidence")
            return {
                "status": "passed", "passed": True,
                "correctness_passed": True, "quality_passed": True,
                "evidence_path": str(evidence), "evidence_sha256": "a" * 64,
                "cache_identity": {"candidate_fingerprint": values["candidate_fingerprint"]},
            }

        evaluator = _make_evaluator(
            self.root / "confirmation-measurement", context, candidate_gate_runner=gate
        )
        candidate = SearchCandidate(space=context["base_space"])
        policy = ConfirmationPolicy()
        with _fake_profile_process():
            full = evaluator.full_model_evaluator(candidate)
            reports = {
                "incumbent": tuple(
                    evaluator.confirmation_runner(candidate, "incumbent", index, policy.measurement_contract)
                    for index in range(3)
                ),
                "challenger": tuple(
                    evaluator.confirmation_runner(candidate, "challenger", index, policy.measurement_contract)
                    for index in range(3)
                ),
            }
        candidate_config = evaluator.candidate_config(candidate, policy.measurement_contract)
        confirmation = confirm_matched_ab(
            incumbent=ConfirmationArm(
                label="incumbent", candidate=candidate_config,
                reports=reports["incumbent"],
                correctness_passed=full["correctness_passed"],
                quality_passed=full["quality_passed"],
            ),
            challenger=ConfirmationArm(
                label="challenger", candidate=candidate_config,
                reports=reports["challenger"],
                correctness_passed=full["correctness_passed"],
                quality_passed=full["quality_passed"],
            ),
            policy=policy,
        )
        self.assertTrue(full["correctness_passed"])
        self.assertTrue(full["quality_passed"])
        self.assertTrue(all(item["reports_valid"] for item in confirmation["arms"].values()))
        self.assertTrue(confirmation["passed"])

    def test_failed_quality_gate_blocks_faster_challenger(self) -> None:
        context = campaign._context(
                self.model_path, CONFIG_PATH, "gate prompt", 2,
                None, None, None, "p150a", 0,
        )
        candidate = SearchCandidate(space=context["base_space"])
        policy = ConfirmationPolicy()
        candidate_config = _make_evaluator(
            self.root / "gate-measurement", context
        ).candidate_config(candidate, policy.measurement_contract)
        reports = tuple(
            {
                "status": "passed", "passed": True,
                "warmup": 5, "iterations": 100,
                "execution_mode": "trace", "runtime_input_mode": "persistent",
                "after_prefill": True,
                "tokens_per_second_per_user": 100.0 + (5.0 if label == "challenger" else 0.0),
            }
            for label in ("incumbent", "challenger")
            for _ in range(3)
        )
        result = confirm_matched_ab(
            incumbent=ConfirmationArm(
                label="incumbent", candidate=candidate_config, reports=reports[:3],
                correctness_passed=True, quality_passed=True,
            ),
            challenger=ConfirmationArm(
                label="challenger", candidate=candidate_config, reports=reports[3:],
                correctness_passed=False, quality_passed=False,
            ),
            policy=policy,
        )
        self.assertEqual(result["relative_improvement"], 0.05)
        self.assertFalse(result["promotion"]["promoted"])
        self.assertEqual(result["promotion"]["selected_arm"], "incumbent")
        self.assertIn("confirmation.challenger.correctness", result["failed_checks"])

    def test_missing_official_root_fails_before_campaign_dispatch(self) -> None:
        arguments = [
            "diagnose", "--stage", "autotune", "--model-path", str(self.model_path),
            "--config", str(CONFIG_PATH), "--prompt", "preflight", "--layers", "2",
            "--out", str(self.out),
        ]
        with patch.object(campaign, "run_autotune_campaign") as run_campaign:
            self.assertEqual(main(arguments), 1)
        run_campaign.assert_not_called()
        report = json.loads(self.out.read_text())
        self.assertIn("--official-tt-metal-root", report["error"])

    def test_candidate_quality_gate_uses_acceptance_checks_and_cache_identity(self) -> None:
        official = self.root / "official"
        reference = official / candidate_quality_gate.DEFAULT_REFERENCE
        reference.parent.mkdir(parents=True)
        reference.write_bytes(b"reference")
        gate_root = self.root / "gates"
        candidate = SimpleNamespace(runtime_commit="runtime")
        fingerprint = "b" * 64
        values = {
            "candidate": candidate, "candidate_fingerprint": fingerprint,
            "program_dir": self.root / "program", "model_path": self.model_path,
            "tokenizer_path": self.model_path, "layers": 2, "batch_size": 32,
            "prefill_len": 256, "cache_len": 1024, "device": "p150a", "device_id": 0,
        }
        acceptance = {
            "status": "passed", "passed": True,
            "checks": {
                name: True for name in (
                    *candidate_quality_gate.CORRECTNESS_CHECKS,
                    "official_buddy_min_user_greedy_agreement",
                )
            },
        }
        with patch.object(candidate_quality_gate, "_git_commit", return_value="official"), patch.object(
            candidate_quality_gate, "run_performance_correctness",
            return_value={"status": "passed", "passed": True, "acceptance": acceptance},
        ):
            gate = candidate_quality_gate.build_candidate_quality_gate(
                official_tt_metal_root=official, official_python=None,
                accuracy_tokens=500, gate_root=gate_root,
            )
            first = gate(**values)
            second = gate(**values)
        self.assertTrue(first["passed"])
        self.assertTrue(first["correctness_passed"])
        self.assertTrue(first["quality_passed"])
        self.assertFalse(first["resumed"])
        self.assertTrue(second["resumed"])
        self.assertEqual(first["cache_identity"], second["cache_identity"])

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

def _make_evaluator(
    root: Path,
    context: dict[str, object],
    candidate_gate_runner=None,
) -> ModelCandidateEvaluator:
    return ModelCandidateEvaluator(
        context={
            **context,
            "prompt": "test evaluator prompt",
            "tokenizer_path": None,
            "runtime_commit": "phase5.1-test-runtime",
        },
        output_root=root,
        candidate_spaces={},
        candidate_gate_runner=candidate_gate_runner,
    )


@contextmanager
def _fake_profile_process():
    def write_bundle(**kwargs: object) -> None:
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        output = Path(command[command.index("--out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        value = 105.0 if "challenger" in output.as_posix() else 100.0
        output.write_text(
            json.dumps({
                "status": "passed", "passed": True,
                "decode_step_ms_p50": 10.0,
                "decode_step_ms_mean": 10.0,
                "decode_step_ms_max": 10.0,
                "decode_step_ms_samples": [10.0] * 5,
                "tokens_per_second_per_user": value,
            })
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch.object(model_evaluator, "write_decode_program_bundle", write_bundle),
        patch.object(model_evaluator.subprocess, "run", run),
    ):
        yield


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
