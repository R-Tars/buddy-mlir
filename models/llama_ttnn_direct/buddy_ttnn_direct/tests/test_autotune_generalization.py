from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.generalization import (
    GeneralizationWorkload,
    run_generalization_suite,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search import (
    SearchProposal,
    SearchProposalGroup,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_search import (
    _base_space,
    _proposal_groups,
    _write_fake_model_config,
)


class GeneralizationSuiteTest(unittest.TestCase):
    def test_one_model_three_cache_shapes_passes_publication_breadth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = self._cases(root, cache_lengths=(512, 1024, 2048))

            report = run_generalization_suite(
                workloads=cases,
                out_dir=root / "suite",
                dry_run=True,
            )

            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["acceptance"]["model_count"], 1)
            self.assertEqual(report["acceptance"]["workload_count"], 3)
            self.assertEqual(
                report["acceptance"]["workloads_per_model"],
                {"llama-3.1-8b": 3},
            )
            self.assertTrue(
                report["acceptance"]["checks"][
                    "publication_breadth_2_models_or_1x3_workloads"
                ]
            )
            self.assertEqual(
                {record["target"]["cache_len"] for record in report["workloads"]},
                {512, 1024, 2048},
            )
            self.assertEqual(
                len({record["shape_sha256"] for record in report["workloads"]}),
                3,
            )

    def test_two_workloads_do_not_claim_paper_breadth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = run_generalization_suite(
                workloads=self._cases(root, cache_lengths=(512, 1024)),
                out_dir=root / "suite",
                dry_run=True,
            )

            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])
            self.assertEqual(
                report["acceptance"]["failed_checks"],
                ["publication_breadth_2_models_or_1x3_workloads"],
            )

    def test_two_model_ids_satisfy_alternate_breadth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = list(self._cases(root, cache_lengths=(512, 1024)))
            cases[1] = dataclasses.replace(cases[1], model_id="llama-3.2-3b")

            report = run_generalization_suite(
                workloads=cases,
                out_dir=root / "suite",
                dry_run=True,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["acceptance"]["model_count"], 2)

    def test_official_hand_seed_challenger_is_rejected_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = list(self._cases(root, cache_lengths=(512, 1024, 2048)))
            groups = list(cases[0].proposal_groups)
            proposal = groups[0].proposals[0]
            forbidden = SearchProposal.create(
                mutation=proposal.mutation,
                score=proposal.score,
                proposal_id=proposal.proposal_id,
                source="official_hand_seed",
                evidence=proposal.evidence,
            )
            groups[0] = SearchProposalGroup(
                stage=groups[0].stage,
                name=groups[0].name,
                proposals=(forbidden,),
                incumbent_score=groups[0].incumbent_score,
            )
            cases[0] = dataclasses.replace(
                cases[0],
                proposal_groups=tuple(groups),
            )

            report = run_generalization_suite(
                workloads=cases,
                out_dir=root / "suite",
                dry_run=True,
            )

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["type"], "GeneralizationError")
            self.assertIn("forbidden challenger", report["error"]["message"])
            self.assertTrue(report["failure_report_written"])
            self.assertTrue((root / "suite/generalization_report.json").is_file())

    def test_completed_suite_resume_skips_workload_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = self._cases(root, cache_lengths=(512, 1024, 2048))
            kwargs = {
                "workloads": cases,
                "out_dir": root / "suite",
                "dry_run": True,
            }

            first = run_generalization_suite(**kwargs)
            second = run_generalization_suite(**kwargs)

            self.assertTrue(first["passed"])
            self.assertTrue(second["passed"])
            self.assertTrue(second["resume"]["completed_report_reused"])

    def _cases(
        self,
        root: Path,
        *,
        cache_lengths: tuple[int, ...],
    ) -> tuple[GeneralizationWorkload, ...]:
        model_root = root / "model"
        _write_fake_model_config(model_root)
        base, template = _base_space(model_root)
        groups = _proposal_groups(base)
        cases = []
        for cache_len in cache_lengths:
            case_template = {**template, "max_cache_len": cache_len}
            cases.append(
                GeneralizationWorkload.create(
                    name=f"llama31-b32-cache{cache_len}",
                    model_id="llama-3.1-8b",
                    base_space=base,
                    proposal_groups=groups,
                    template_config=case_template,
                    target={
                        "batch_size": 32,
                        "prefill_len": 256,
                        "cache_len": cache_len,
                        "num_layers": 32,
                        "hidden_size": 4096,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                    },
                    run_identity=f"generalization-cache-{cache_len}",
                    seed_policy="baseline_only",
                )
            )
        return tuple(cases)


if __name__ == "__main__":
    unittest.main()
