from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_diff import (
    PARITY_SECTIONS,
    REQUIRED_PARITY_PATHS,
    build_config_parity_view,
    default_official_config_path,
    diff_official_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.config import (
    build_codegen_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.official_config import (
    P150A_LLAMA31_8B_B32_PERFORMANCE,
    load_official_config_profile,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
)


class ConfigDiffTest(unittest.TestCase):
    def test_product_docs_keep_artifacts_in_buddy_build_tree(self) -> None:
        model_root = Path(__file__).resolve().parents[2]
        for path in (
            model_root / "README.md",
            model_root / "docs" / "commands.md",
        ):
            text = path.read_text()
            self.assertNotIn("/tmp", text, path)
            self.assertIn("$BUDDY_BUILD/models/llama31_ttnn_direct", text)

    def test_latest_evidence_summary_is_compact_and_current(self) -> None:
        evidence_dir = (
            Path(__file__).resolve().parents[2] / "docs" / "evidence"
        )
        evidence_files = sorted(path.name for path in evidence_dir.glob("*.json"))
        self.assertEqual(evidence_files, ["latest_summary.json"])

        summary_text = (evidence_dir / "latest_summary.json").read_text()
        summary = json.loads(summary_text)
        self.assertLessEqual(len(summary_text.splitlines()), 300)
        self.assertTrue(summary["performance"]["passed"])
        self.assertEqual(summary["performance"]["warmup"], 5)
        self.assertEqual(summary["performance"]["iterations"], 100)
        self.assertEqual(summary["performance"]["repetitions"], 3)
        self.assertTrue(summary["correctness"]["full_depth_functional_passed"])
        self.assertTrue(summary["correctness"]["all_bf16_passed"])
        self.assertTrue(
            summary["correctness"]["performance_recipe_quality_passed"]
        )
        self.assertGreaterEqual(
            summary["correctness"]["all_bf16_minimum_pcc"],
            summary["correctness"]["all_bf16_pcc_threshold"],
        )
        for key, value in summary["source_identity"].items():
            if key.endswith("_commit"):
                self.assertGreaterEqual(len(value), 8, key)
            else:
                self.assertEqual(len(value), 64, key)
        self.assertFalse(summary["raw_artifacts"]["tracked_raw_artifacts"])

    def test_normalized_parity_config_matches_itself(self) -> None:
        official = json.loads(default_official_config_path().read_text())

        diff = diff_official_config(official, official)

        self.assertEqual(diff["status"], "match")
        self.assertEqual(diff["summary"]["issue_count"], 0)
        self.assertGreater(diff["summary"]["matching_count"], 0)
        self.assertEqual(diff["summary"]["section_count"], len(PARITY_SECTIONS))
        self.assertEqual(diff["summary"]["sections_with_issues"], [])
        self.assertEqual(diff["gap_summary"]["status"], "match")
        self.assertEqual(diff["gap_summary"]["issue_count"], 0)
        self.assertEqual(diff["gap_summary"]["sections_with_issues"], [])
        self.assertEqual(diff["gap_summary"]["top_issue_paths"], [])
        coverage = diff["required_field_coverage"]
        self.assertEqual(coverage["required_fields"], list(REQUIRED_PARITY_PATHS))
        self.assertEqual(coverage["official"]["status"], "complete")
        self.assertEqual(coverage["official"]["missing_required_paths"], [])
        self.assertEqual(
            coverage["official"]["required_field_count"],
            len(REQUIRED_PARITY_PATHS),
        )
        self.assertEqual(
            coverage["official"]["present_required_count"],
            len(REQUIRED_PARITY_PATHS),
        )
        self.assertEqual(set(diff["sections"]), set(PARITY_SECTIONS))
        self.assertTrue(
            all(
                section["status"] == "match"
                for section in diff["sections"].values()
            )
        )

    def test_generated_config_reports_known_official_gaps(self) -> None:
        ours = _fake_generated_config()
        official = json.loads(default_official_config_path().read_text())

        diff = diff_official_config(ours, official)

        self.assertEqual(diff["status"], "diff_found")
        missing_paths = {field["path"] for field in diff["missing_fields"]}
        mismatch_paths = {field["path"] for field in diff["mismatched_fields"]}
        self.assertIn("memory_config.attention_qkv", missing_paths)
        self.assertIn("program_config.attention_sdpa", missing_paths)
        self.assertIn(
            "compute_fidelity.mlp.gate_up_default",
            missing_paths,
        )
        self.assertIn("core_grid.attention", missing_paths)
        self.assertNotIn("lm_head.argmax_strategy", mismatch_paths)
        self.assertIn("paged_attention.scale", mismatch_paths)
        self.assertEqual(diff["gap_summary"]["status"], "diff_found")
        self.assertIn(
            "memory_config",
            diff["gap_summary"]["sections_with_issues"],
        )
        self.assertGreater(
            diff["gap_summary"]["issue_counts_by_section"]["memory_config"],
            0,
        )
        self.assertTrue(diff["gap_summary"]["top_issue_paths"])
        self.assertEqual(
            diff["sections"]["memory_config"]["status"],
            "diff_found",
        )
        self.assertIn(
            "memory_config.attention_qkv",
            diff["sections"]["memory_config"]["missing_paths"],
        )
        self.assertEqual(
            diff["ours"]["parity_config"]["lm_head"]["split_count"],
            8,
        )
        self.assertEqual(
            diff["required_field_coverage"]["official"]["status"],
            "complete",
        )
        self.assertEqual(
            diff["required_field_coverage"]["ours"]["status"],
            "incomplete",
        )
        self.assertIn(
            "memory_config.attention_qkv",
            diff["required_field_coverage"]["ours"]["missing_required_paths"],
        )

    def test_imported_official_profile_has_exact_generated_parity(self) -> None:
        official = load_official_config_profile(
            P150A_LLAMA31_8B_B32_PERFORMANCE
        )
        generated = _fake_generated_config(
            official_profile=P150A_LLAMA31_8B_B32_PERFORMANCE
        )

        self.assertEqual(official["schema_version"], 2)
        self.assertEqual(
            official["source"]["kind"],
            "extracted_tt_transformers_model_args",
        )
        self.assertEqual(official["source"]["device_name"], "P150")
        self.assertEqual(
            official["source"]["tt_metal_git_commit"],
            "61e690c25202111b52cbc1fbc9148b6524070c6f",
        )
        self.assertEqual(
            official["source"]["runtime_adaptations"],
            [
                "prefill_qkv_and_wo_disable_fuse_batch_for_"
                "buddy_batch32_tensor_shape"
            ],
        )
        self.assertEqual(
            generated["official_config_profile"],
            P150A_LLAMA31_8B_B32_PERFORMANCE,
        )
        self.assertEqual(
            generated["attention"]["qkv_program_config"]["kind"],
            "ttnn_matmul_dram_sharded_program_config",
        )
        self.assertEqual(
            generated["rms_norm"]["attention"][
                "input_memory_config"
            ]["shard_shape"],
            [32, 128],
        )
        self.assertEqual(
            generated["mlp"]["layer_overrides"]["31"][
                "parameter_intermediate_dtype"
            ],
            "bfloat8_b",
        )
        self.assertEqual(
            generated["lm_head"]["shard_output_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )
        diff = diff_official_config(generated, official)
        self.assertEqual(diff["status"], "match")
        self.assertEqual(diff["summary"]["issue_count"], 0)

    def test_official_required_field_coverage_reports_missing_seed_field(
        self,
    ) -> None:
        ours = _fake_generated_config()
        official = json.loads(default_official_config_path().read_text())
        official["parity_config"]["program_config"].pop(
            "attention_sdpa",
        )
        official["parity_config"]["memory_config"]["attention_sdpa"] = None

        diff = diff_official_config(ours, official)

        coverage = diff["required_field_coverage"]["official"]
        self.assertEqual(coverage["status"], "incomplete")
        self.assertEqual(coverage["missing_required_count"], 2)
        self.assertIn(
            "program_config.attention_sdpa",
            coverage["missing_required_paths"],
        )
        self.assertIn(
            "memory_config.attention_sdpa",
            coverage["missing_required_paths"],
        )
        self.assertEqual(
            coverage["sections_missing_required_fields"],
            ["program_config", "memory_config"],
        )

    def test_inspect_includes_official_config_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = root / "program"
            out = root / "inspect.json"
            program_dir.mkdir()
            (program_dir / "config.json").write_text(
                json.dumps(_fake_generated_config())
            )
            (program_dir / "execution_plan.json").write_text("{}")
            for artifact in (
                "README.md",
                "model.py",
                "run_decode.py",
                "semantic_graph.json",
                "weights_manifest.json",
            ):
                (program_dir / artifact).write_text("{}")

            exit_code = main(
                [
                    "inspect",
                    "--program-dir",
                    str(program_dir),
                    "--official-config",
                    str(default_official_config_path()),
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            inspect_report = json.loads(out.read_text())
            self.assertTrue(inspect_report["passed"])
            report = inspect_report["official_config_diff"]
            self.assertEqual(report["status"], "diff_found")
            self.assertGreater(report["summary"]["issue_count"], 0)
            self.assertEqual(report["gap_summary"]["status"], "diff_found")
            self.assertIn(
                "program_config",
                report["gap_summary"]["sections_with_issues"],
            )

    def test_build_config_parity_view_accepts_generated_config(self) -> None:
        view = build_config_parity_view(_fake_generated_config())

        self.assertEqual(view["source_format"], "generated_ttnn_direct_config")
        self.assertEqual(
            view["parity_config"]["dtype_recipe"]["recipe"],
            "official_like_performance_seed",
        )
        self.assertEqual(view["parity_config"]["lm_head"]["shard_count"], 8)


def _fake_generated_config(
    *, official_profile: str | None = None
) -> dict[str, object]:
    graph = import_hf_llama(
        "/tmp/fake-config-diff",
        config={
            "_name_or_path": "fake-config-diff",
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        },
        state_dict_metadata=[],
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )
    plan = build_execution_plan(
        graph,
        {
            "device": "p150a",
            "model": "llama3.1-8b",
            "batch_size": 32,
            "decode_seq_len": 1,
            "prefill_seq_len": 128,
            "max_cache_len": 1024,
            "attention_template": "official_paged_attention_decode",
            "mlp_template": "official_gated_mlp_decode",
            "lm_head_template": "official_split_lm_head",
            "kv_cache_template": "paged_kv_cache",
            "generation_template": "device_argmax_greedy",
            "lm_head_split_count": 8,
            "dtype_recipe": "official_like_performance_seed",
            "official_config_profile": official_profile,
        },
    )
    return build_codegen_config(plan)


if __name__ == "__main__":
    unittest.main()
