from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    CandidateConfig,
    CoreGrid,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    ProposalScore,
    SearchBudget,
    SearchCallbacks,
    SearchProposalGroup,
    SearchSpaceConfig,
    buildable_template_config,
    proposal_from_space,
    run_hierarchical_search,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.config import (
    build_codegen_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates import (
    build_execution_plan,
    load_template_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json"


class HierarchicalSearchTest(unittest.TestCase):
    def test_local_mutations_compose_across_all_three_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            _write_fake_model_config(model_root)
            base, _ = _base_space(model_root)
            groups = _proposal_groups(base)

            candidate = groups[0].proposals[0].mutation.apply(base)
            candidate = groups[1].proposals[0].mutation.apply(candidate)
            candidate = groups[2].proposals[0].mutation.apply(candidate)

            self.assertEqual(
                candidate.templates["kv_update"],
                "paged_fused_update_cache",
            )
            self.assertEqual(
                candidate.operators["attention.sdpa"]["grid"],
                [8, 4],
            )
            self.assertEqual(
                candidate.memory_configs["lm_head.shard_output_memory_config"][
                    "runtime_name"
                ],
                "DRAM_MEMORY_CONFIG",
            )

    def test_pipeline_matched_ab_and_best_config_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "model"
            out_dir = root / "search"
            program_dir = root / "program"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)
            callbacks, calls = _callbacks(template, base)

            report = run_hierarchical_search(
                base_space=base,
                proposal_groups=_proposal_groups(base),
                template_config=template,
                out_dir=out_dir,
                callbacks=callbacks,
                budget=SearchBudget(
                    max_candidates=128,
                    max_device_minutes=1.0,
                    beam_width=4,
                    template_top_k=2,
                    microbench_top_k=4,
                    full_model_top_k=2,
                ),
                run_identity="phase8-unit",
            )

            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["passed"])
            self.assertFalse(report["cartesian_exhaustive_search"])
            self.assertEqual(
                [stage["name"] for stage in report["stages"]],
                [
                    "template_search",
                    "op_search",
                    "layout_beam",
                    "layer_confirmation",
                    "full_model_trace",
                    "long_confirmation",
                ],
            )
            self.assertTrue(report["matched_ab_confirmation"]["promotion"]["promoted"])
            self.assertEqual(len(calls["confirmation"]), 6)
            self.assertEqual(
                [item[1] for item in calls["confirmation"]],
                [
                    "incumbent",
                    "challenger",
                    "challenger",
                    "incumbent",
                    "incumbent",
                    "challenger",
                ],
            )
            for stage in report["stages"][:3]:
                for group in stage["groups"]:
                    self.assertLessEqual(group["output_beam_size"], 4)

            best_config_path = Path(report["artifacts"]["best_config"])
            self.assertTrue(best_config_path.is_file())
            self.assertEqual(
                json.loads(best_config_path.read_text())["autotune"]["schema_version"],
                2,
            )
            self.assertEqual(
                main(
                    [
                        "build",
                        "--model-path",
                        str(model_root),
                        "--config",
                        str(best_config_path),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )
            generated = json.loads((program_dir / "config.json").read_text())
            self.assertEqual(generated["autotune"]["schema_version"], 2)
            self.assertEqual(
                generated["attention"]["sdpa_program_config"]["core_grid"],
                [8, 4],
            )

    def test_budget_exhaustion_still_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "model"
            out_dir = root / "search"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)
            callbacks, _ = _callbacks(template, base)

            report = run_hierarchical_search(
                base_space=base,
                proposal_groups=_proposal_groups(base),
                template_config=template,
                out_dir=out_dir,
                callbacks=callbacks,
                budget=SearchBudget(
                    max_candidates=1,
                    max_device_minutes=1.0,
                    beam_width=4,
                    template_top_k=2,
                    microbench_top_k=2,
                    full_model_top_k=2,
                ),
                run_identity="phase8-budget",
            )

            self.assertEqual(report["status"], "budget_exhausted")
            self.assertFalse(report["passed"])
            self.assertTrue(report["failure_report_written"])
            persisted = json.loads((out_dir / "search_report.json").read_text())
            self.assertEqual(persisted["error"]["type"], "SearchBudgetExhausted")

            resumed = run_hierarchical_search(
                base_space=base,
                proposal_groups=_proposal_groups(base),
                template_config=template,
                out_dir=out_dir,
                callbacks=callbacks,
                budget=SearchBudget(
                    max_candidates=128,
                    max_device_minutes=1.0,
                    beam_width=4,
                    template_top_k=2,
                    microbench_top_k=2,
                    full_model_top_k=2,
                ),
                run_identity="phase8-budget",
            )
            self.assertEqual(resumed["status"], "passed")
            self.assertTrue(resumed["resume"]["checkpoint_loaded"])
            self.assertGreaterEqual(
                resumed["budget_usage"]["reused_evaluation_count"],
                1,
            )

    def test_device_minute_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "model"
            out_dir = root / "search"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)

            def expensive(_stage, _candidate):
                return {
                    "status": "passed",
                    "passed": True,
                    "latency_ms": 1.0,
                    "device_seconds": 1.0,
                }

            report = run_hierarchical_search(
                base_space=base,
                proposal_groups=_proposal_groups(base),
                template_config=template,
                out_dir=out_dir,
                callbacks=SearchCallbacks(candidate_evaluator=expensive),
                budget=SearchBudget(
                    max_candidates=128,
                    max_device_minutes=0.001,
                    beam_width=4,
                    template_top_k=2,
                    microbench_top_k=2,
                    full_model_top_k=2,
                ),
                dry_run=True,
                run_identity="phase8-device-budget",
            )

            self.assertEqual(report["status"], "budget_exhausted")
            self.assertGreaterEqual(
                report["budget_usage"]["device_minutes"],
                0.001,
            )

    def test_completed_run_resumes_without_callback_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "model"
            out_dir = root / "search"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)
            callbacks, calls = _callbacks(template, base)
            kwargs = {
                "base_space": base,
                "proposal_groups": _proposal_groups(base),
                "template_config": template,
                "out_dir": out_dir,
                "callbacks": callbacks,
                "budget": SearchBudget(
                    max_candidates=128,
                    max_device_minutes=1.0,
                    beam_width=4,
                    template_top_k=2,
                    microbench_top_k=4,
                    full_model_top_k=2,
                ),
                "run_identity": "phase8-resume",
            }

            first = run_hierarchical_search(**kwargs)
            counts = {name: len(values) for name, values in calls.items()}
            second = run_hierarchical_search(**kwargs)

            self.assertEqual(first["status"], "passed")
            self.assertEqual(second["status"], "passed")
            self.assertTrue(second["resume"]["completed_report_reused"])
            self.assertEqual(
                counts,
                {name: len(values) for name, values in calls.items()},
            )

    def test_callback_exception_is_persisted_per_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "model"
            out_dir = root / "search"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)

            def fail(_stage, _candidate):
                raise RuntimeError("classified callback failure")

            report = run_hierarchical_search(
                base_space=base,
                proposal_groups=_proposal_groups(base),
                template_config=template,
                out_dir=out_dir,
                callbacks=SearchCallbacks(candidate_evaluator=fail),
                dry_run=True,
                run_identity="phase8-failure",
            )

            self.assertEqual(report["status"], "failed")
            candidate_reports = sorted((out_dir / "candidates").glob("*/*.json"))
            self.assertGreaterEqual(len(candidate_reports), 1)
            saved = json.loads(candidate_reports[0].read_text())
            self.assertFalse(saved["passed"])
            self.assertEqual(saved["result"]["error"]["type"], "RuntimeError")
            self.assertTrue((out_dir / "search_report.json").is_file())

    def test_buildable_config_helper_preserves_template_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            _write_fake_model_config(model_root)
            base, template = _base_space(model_root)
            result = buildable_template_config(template, base)
            self.assertEqual(result["batch_size"], template["batch_size"])
            self.assertEqual(result["dtype_recipe"], template["dtype_recipe"])
            self.assertEqual(result["autotune"], base.to_dict())


def _proposal_groups(base: SearchSpaceConfig) -> tuple[SearchProposalGroup, ...]:
    template_payload = base.to_dict()
    template_payload["templates"]["kv_update"] = "paged_fused_update_cache"
    template_space = SearchSpaceConfig.from_dict(template_payload)
    template_proposal = proposal_from_space(
        stage="template",
        group="kv_update",
        label="paged_fused_update_cache",
        base_space=base,
        candidate_space=template_space,
        score=ProposalScore(latency_ms=9.0),
        proposal_id="template-fused-kv",
    )

    op_space = base.with_sdpa_grid(CoreGrid(8, 4))
    op_proposal = proposal_from_space(
        stage="op",
        group="attention.sdpa",
        label="sdpa-grid-8x4",
        base_space=base,
        candidate_space=op_space,
        score=ProposalScore(
            latency_ms=8.0,
            l1_bytes=1024,
            conversion_latency_ms=0.1,
        ),
        proposal_id="op-sdpa-8x4",
    )

    layout_space = base.with_lm_head_dram_concat()
    layout_proposal = proposal_from_space(
        stage="layout",
        group="lm_head",
        label="dram-concat",
        base_space=base,
        candidate_space=layout_space,
        score=ProposalScore(latency_ms=7.0),
        proposal_id="layout-lm-head-dram",
    )
    return (
        SearchProposalGroup(
            stage="template",
            name="kv_update",
            proposals=(template_proposal,),
            incumbent_score=ProposalScore(latency_ms=10.0),
        ),
        SearchProposalGroup(
            stage="op",
            name="attention.sdpa",
            proposals=(op_proposal,),
            incumbent_score=ProposalScore(latency_ms=10.0, l1_bytes=2048),
        ),
        SearchProposalGroup(
            stage="layout",
            name="lm_head",
            proposals=(layout_proposal,),
            incumbent_score=ProposalScore(latency_ms=10.0),
        ),
    )


def _callbacks(template: dict, base: SearchSpaceConfig):
    calls: dict[str, list] = {
        "candidate": [],
        "layer": [],
        "full": [],
        "confirmation": [],
    }
    precision = PrecisionContract.from_template_config(template)
    execution = ExecutionContract.from_template_config(
        template,
        prompt_corpus_sha256=hashlib.sha256(b"phase8-search").hexdigest(),
    )
    incumbent_sha = hashlib.sha256(
        json.dumps(base.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def mutation_count(candidate) -> int:
        return len(candidate.mutation_ids)

    def candidate_evaluator(stage, candidate):
        calls["candidate"].append((stage, candidate.candidate_id))
        return {
            "status": "passed",
            "passed": True,
            "objective": {
                "name": "latency_ms",
                "value": 20.0 - mutation_count(candidate),
                "direction": "minimize",
            },
            "correctness_passed": True,
            "quality_passed": True,
            "isolated_subprocess": True,
            "device_seconds": 0.0,
        }

    def layer_evaluator(candidate):
        calls["layer"].append(candidate.candidate_id)
        return {
            "status": "passed",
            "passed": True,
            "latency_ms": 20.0 - mutation_count(candidate),
            "correctness_passed": True,
            "quality_passed": True,
            "isolated_subprocess": True,
            "device_seconds": 0.0,
        }

    def full_model_evaluator(candidate):
        calls["full"].append(candidate.candidate_id)
        return {
            "status": "profiled",
            "passed": True,
            "tokens_per_second_per_user": 100.0 + mutation_count(candidate),
            "correctness_passed": True,
            "quality_passed": True,
            "isolated_subprocess": True,
            "device_seconds": 0.0,
        }

    def confirmation_runner(candidate, arm, repetition, contract):
        calls["confirmation"].append((candidate.candidate_id, arm, repetition))
        baseline = 100.0 if arm == "incumbent" else 102.0
        jitter = (0.0, 0.05, -0.05)[repetition]
        return {
            "status": "profiled",
            "passed": True,
            "warmup": contract.warmup,
            "iterations": contract.iterations,
            "execution_mode": "trace",
            "runtime_input_mode": "persistent",
            "after_prefill": True,
            "tokens_per_second_per_user": baseline + jitter,
            "correctness_passed": True,
            "quality_passed": True,
            "isolated_subprocess": True,
            "device_seconds": 0.0,
        }

    def candidate_factory(candidate, contract):
        return CandidateConfig.create(
            precision_contract=precision,
            expected_precision_hash=precision.hash,
            execution_contract=execution,
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
                "batch_size": 32,
                "cache_len": 1024,
                "page_block_size": 32,
            },
            tunable_state={
                "candidate_space_sha256": candidate.space_sha256,
                "is_incumbent": candidate.space_sha256 == incumbent_sha,
            },
        )

    return (
        SearchCallbacks(
            candidate_evaluator=candidate_evaluator,
            layer_evaluator=layer_evaluator,
            full_model_evaluator=full_model_evaluator,
            confirmation_runner=confirmation_runner,
            candidate_config_factory=candidate_factory,
        ),
        calls,
    )


def _base_space(model_root: Path) -> tuple[SearchSpaceConfig, dict]:
    template = load_template_config(CONFIG_PATH)
    graph = import_hf_llama(
        model_root,
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
        generation_mode="greedy",
    )
    plan = build_execution_plan(graph, template)
    runtime = build_codegen_config(plan)
    return SearchSpaceConfig.from_runtime_config(runtime), template


def _write_fake_model_config(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "phase8-fake-llama",
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
