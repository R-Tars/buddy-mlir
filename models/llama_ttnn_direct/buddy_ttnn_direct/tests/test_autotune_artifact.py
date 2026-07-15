from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    REQUIRED_ABLATIONS,
    build_paper_artifact,
    verify_paper_artifact,
)


class PaperArtifactTest(unittest.TestCase):
    def test_builds_complete_reproducible_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = self._fixture(root)

            artifact = build_paper_artifact(spec, out_dir=root / "artifact")

            self.assertTrue(artifact["passed"])
            metrics = artifact["metrics"]
            self.assertEqual(metrics["enumeration"]["total_search_entities"], 15)
            self.assertEqual(metrics["enumeration"]["program_candidate_total"], 7)
            self.assertEqual(metrics["pruning"]["program_candidates_rejected"], 2)
            self.assertAlmostEqual(
                metrics["pruning"]["program_candidate_pruning_ratio"], 2 / 7
            )
            self.assertEqual(metrics["microbench"]["experiment_count"], 5)
            self.assertEqual(metrics["full_model"]["profile_count"], 8)
            self.assertEqual(metrics["best_performance"]["value"], 34.0)
            self.assertAlmostEqual(
                metrics["best_performance"]["official_ratio"], 34.0 / 33.0
            )
            self.assertEqual(
                {row["name"] for row in artifact["ablations"]},
                set(REQUIRED_ABLATIONS),
            )
            transfer_on = next(
                row
                for row in artifact["ablations"]
                if row["name"] == "representative_layer_transfer_on"
            )
            transfer_off = next(
                row
                for row in artifact["ablations"]
                if row["name"] == "representative_layer_transfer_off"
            )
            self.assertEqual(
                transfer_off["planned_layer_evaluation_count"],
                transfer_on["planned_layer_evaluation_count"] * 16,
            )
            for filename in (
                "paper_artifact.json",
                "ablation.csv",
                "RESULTS.md",
                "evidence_summary.json",
                "spec.json",
                "source_manifest.json",
                "artifact_manifest.json",
                "reproduce.sh",
            ):
                self.assertTrue((root / "artifact" / filename).is_file())
            verification = verify_paper_artifact(root / "artifact")
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["checked_artifact_file_count"], 7)
            self.assertEqual(verification["checked_source_count"], 13)

    def test_missing_required_ablation_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = self._fixture(root)
            spec["ablations"] = spec["ablations"][:-1]

            report = build_paper_artifact(spec, out_dir=root / "artifact")

            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["type"], "PaperArtifactError")
            self.assertIn("ablation matrix mismatch", report["error"]["message"])
            self.assertTrue((root / "artifact/artifact_failure.json").is_file())

    def test_verifier_detects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_root = root / "artifact"
            report = build_paper_artifact(self._fixture(root), out_dir=artifact_root)
            self.assertTrue(report["passed"])
            (artifact_root / "RESULTS.md").write_text("tampered\n")

            verification = verify_paper_artifact(artifact_root)

            self.assertFalse(verification["passed"])
            self.assertIn(
                "artifact file hash mismatch: RESULTS.md",
                verification["errors"],
            )

    def test_verifier_detects_source_evidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_root = root / "artifact"
            report = build_paper_artifact(self._fixture(root), out_dir=artifact_root)
            self.assertTrue(report["passed"])
            template = root / "template.json"
            payload = json.loads(template.read_text())
            payload["new_field"] = "changed"
            _write_json(template, payload)

            verification = verify_paper_artifact(artifact_root)

            self.assertFalse(verification["passed"])
            self.assertIn(
                "source evidence hash mismatch: template_acceptance",
                verification["errors"],
            )

    def test_failure_report_invalidates_stale_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_root = root / "artifact"
            spec = self._fixture(root)
            first = build_paper_artifact(spec, out_dir=artifact_root)
            self.assertTrue(first["passed"])
            spec["ablations"] = spec["ablations"][:-1]
            second = build_paper_artifact(spec, out_dir=artifact_root)
            self.assertFalse(second["passed"])

            verification = verify_paper_artifact(artifact_root)

            self.assertFalse(verification["passed"])
            self.assertIn("artifact_failure.json is present", verification["errors"])

    def test_spec_file_resolves_relative_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = self._fixture(root)
            spec["sources"] = {
                key: (
                    [Path(path).name for path in value]
                    if key == "layout_search_reports"
                    else Path(value).name
                )
                for key, value in spec["sources"].items()
            }
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)

            artifact = build_paper_artifact(spec_path, out_dir=root / "artifact")

            self.assertTrue(artifact["passed"])

    def _fixture(self, root: Path) -> dict:
        best_config = root / "best_config.json"
        _write_json(best_config, {"schema_version": 1, "autotune": {}})
        reports = {
            "template_acceptance": (
                "template.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "templates": [
                        {"name": f"template-{index}", "status": "passed"}
                        for index in range(4)
                    ],
                },
            ),
            "microbench_acceptance": (
                "microbench.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "transfer_report": {
                        "layer_count": 32,
                        "representative_layers": [0, 31],
                        "groups": [
                            {"transfer_count": 30},
                            {"transfer_count": 0},
                        ],
                    },
                },
            ),
            "matmul_acceptance": (
                "matmul.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "candidate_counts": {"attention.qkv": {"legal": 2, "rejected": 1}},
                    "microbenchmark_selection": {
                        "ranked": [
                            {"candidate_id": "one"},
                            {"candidate_id": "two"},
                        ]
                    },
                },
            ),
            "sdpa_enumeration": (
                "sdpa_enumeration.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "proposal_count": 4,
                    "legal_candidate_count": 3,
                    "rejected_candidate_count": 1,
                },
            ),
            "sdpa_acceptance": (
                "sdpa.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "correctness": {
                        "official": {"passed": True},
                        "challenger": {"passed": True},
                    },
                },
            ),
            "layout_acceptance": (
                "layout.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "conversion_measurement": {"p50_ms": 0.03},
                },
            ),
            "search_report": (
                "search.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "passed": True,
                    "artifacts": {"best_config": str(best_config)},
                    "budget_usage": {
                        "device_seconds": 120.0,
                        "evaluation_count": 10,
                        "reused_evaluation_count": 0,
                    },
                    "stages": [
                        {
                            "name": "full_model_trace",
                            "evaluations": [{"passed": True}, {"passed": True}],
                        }
                    ],
                    "matched_ab_confirmation": {
                        "measurement_contract": {
                            "warmup": 5,
                            "iterations": 100,
                            "repetitions": 3,
                        },
                        "arms": {
                            "incumbent": {"median": 34.0, "cv": 0.001},
                            "challenger": {"median": 33.9, "cv": 0.002},
                        },
                        "promotion": {
                            "selected_arm": "incumbent",
                            "selected_candidate_fingerprint": "a" * 64,
                        },
                        "execution_order": [
                            {"arm": "incumbent"},
                            {"arm": "challenger"},
                        ]
                        * 3,
                    },
                },
            ),
            "generalization_report": (
                "generalization.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "passed": True,
                    "acceptance": {
                        "passed": True,
                        "model_count": 1,
                        "workload_count": 3,
                        "workloads_per_model": {"llama": 3},
                    },
                },
            ),
            "correctness_evidence": (
                "correctness.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "target": {"published_decode_tokens_per_second_per_user": 32.0},
                    "compiler_correctness": {
                        "passed": True,
                        "logits_pcc": 0.999,
                        "minimum_hidden_pcc": 0.995,
                        "minimum_sampled_kv_pcc": 0.992,
                    },
                    "performance_recipe_correctness": {
                        "passed": True,
                        "evaluated_tokens": 500,
                        "official_top1_accuracy": 0.91,
                        "buddy_min_user_top1_accuracy": 0.908,
                        "official_top5_accuracy": 0.98,
                        "buddy_min_user_top5_accuracy": 0.98,
                        "buddy_min_user_greedy_agreement_with_official": 0.97,
                    },
                    "final_decode_parity": {"official_release_median_tpsu": 33.0},
                },
            ),
        }
        source_paths = {}
        for label, (filename, payload) in reports.items():
            path = root / filename
            _write_json(path, payload)
            source_paths[label] = str(path)
        layout_paths = []
        for index, (expanded, rejected) in enumerate(((2, 1), (1, 0), (1, 0))):
            path = root / f"layout_search_{index}.json"
            _write_json(
                path,
                {
                    "schema_version": 1,
                    "status": "passed",
                    "search_statistics": {"expanded_state_count": expanded},
                    "legality": {"illegal_or_unmeasured_transition_count": rejected},
                },
            )
            layout_paths.append(str(path))
        source_paths["layout_search_reports"] = layout_paths
        return {
            "schema_version": 1,
            "campaign": {
                "name": "unit-test-campaign",
                "model": "llama",
                "device": "p150a",
                "workload": "b32-cache1024",
            },
            "sources": source_paths,
            "ablations": _ablations(),
        }


def _ablations() -> list[dict]:
    axes = {
        "config_only": (True, False, False),
        "template_only": (False, True, False),
        "layout_only": (False, False, True),
        "template_config": (True, True, False),
        "template_config_layout": (True, True, True),
        "representative_layer_transfer_on": (True, True, True),
        "representative_layer_transfer_off": (True, True, True),
        "analytical_pruning_on": (True, True, True),
        "analytical_pruning_off": (True, True, True),
    }
    rows = []
    for name in REQUIRED_ABLATIONS:
        config, template, layout = axes[name]
        measured = name in {"config_only", "template_config_layout"}
        mode = (
            "measured"
            if measured
            else (
                "analytical"
                if name.startswith("analytical_") or name.startswith("representative_")
                else "reused_equivalent"
            )
        )
        rows.append(
            {
                "name": name,
                "mode": mode,
                "enabled": {
                    "config_search": config,
                    "template_search": template,
                    "layout_search": layout,
                    "representative_layer_transfer": not name.endswith("transfer_off"),
                    "analytical_pruning": not name.endswith("pruning_off"),
                },
                "equivalent_to": (None if measured else "template_config_layout"),
                "selection": "retain_incumbent",
                "full_model_profile_count": 8 if measured else 3,
                "evidence_labels": ["search_report", "search_best_config"],
                "notes": ["unit-test evidence"],
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
