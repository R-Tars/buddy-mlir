from __future__ import annotations

import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.reports.evidence import (
    reference_summary,
    step_names_with_status,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.performance import (
    PROFILE_GENERATE_MILESTONE_IDS,
    performance_gap_summary,
    resolve_performance_baseline,
    throughput_baseline_summary,
    validate_real_generate_milestones,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.runtime import (
    decode_runtime_inputs_complete,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.reports.runtime_diagnostics import (
    runtime_error_findings,
)


class ReportHelperTest(unittest.TestCase):
    def test_runtime_diagnostics_classifies_firmware_failure(self) -> None:
        findings = runtime_error_findings(
            {"step": {"error": "Failed to initialize FW; try resetting the board"}}
        )

        self.assertEqual(findings[0]["kind"], "tenstorrent_firmware_init_failed")
        self.assertEqual(findings[0]["path"], "step.error")

    def test_evidence_summary_helpers_report_failed_items(self) -> None:
        results = {"ok": "pass", "pending": "pending", "bad": "runtime_error"}
        self.assertEqual(step_names_with_status(results, failing=True), ["bad"])
        self.assertEqual(
            reference_summary(
                {
                    "reference": {
                        "status": "failed",
                        "kind": "sample",
                        "checks": [
                            {"name": "ok", "passed": True},
                            {"name": "bad", "passed": False},
                        ],
                    }
                }
            )["reference_failed_checks"],
            ["bad"],
        )

    def test_official_performance_baseline_resolves(self) -> None:
        baseline = resolve_performance_baseline(
            "tt_metal_official_llama31_8b_b32"
        )

        self.assertEqual(baseline["model"], "Llama 3.1 8B")
        self.assertEqual(baseline["role"], "official_8b_target")
        self.assertEqual(baseline["batch_size"], 32)
        self.assertEqual(baseline["decode_tokens_per_second_per_user"], 33.1)
        self.assertTrue(baseline["baseline_file"].endswith(
            "performance_baselines.json"
        ))

    def test_performance_summaries_preserve_ratios_and_bottleneck(self) -> None:
        report = {
            "baseline_tokens_per_second_per_user": 10.0,
            "baseline_reference": "sample",
            "min_baseline_ratio": 0.5,
            "baseline_reference_entry": {
                "id": "sample",
                "role": "official_8b_target",
            },
        }
        profile = {
            "throughput_summary": {
                "status": "measured",
                "tokens_per_second_per_user": 6.0,
            },
            "bottleneck_summary": {
                "max_section": "argmax",
                "max_section_ms": 3.0,
                "sections_ms": {"argmax": 3.0, "decode": 1.0},
            },
        }

        throughput = throughput_baseline_summary(report, profile)
        gap = performance_gap_summary(report, profile)

        self.assertEqual(throughput["ratio"], 0.6)
        self.assertTrue(throughput["passed"])
        self.assertEqual(gap["shortfall_to_baseline"], 4.0)
        self.assertEqual(gap["bottleneck"]["max_section_share"], 0.75)

    def test_generate_depth_sweep_can_complete_m1_milestone(self) -> None:
        profile_generate = {
            "profile_generate_report": "/tmp/profile.json",
            "generate_report": "/tmp/generate.json",
            "performance_milestones": {
                "official_reference": {"id": "official"},
                "observed": {},
                "milestones": [
                    {
                        "id": milestone_id,
                        "name": milestone_id,
                        "passed": milestone_id == "M0",
                        "status": "passed" if milestone_id == "M0" else "failed",
                    }
                    for milestone_id in PROFILE_GENERATE_MILESTONE_IDS
                ],
            },
        }
        depth_sweep = {
            "generate_depth_sweep_report": "/tmp/depth.json",
            "status": "pass",
            "covered_full_depth": True,
            "max_depth": 32,
            "passed_depth_count": 6,
            "failed_depths": [],
            "acceptance": {"passed": True},
        }

        summary = validate_real_generate_milestones(
            profile_generate,
            depth_sweep,
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["highest_passed"], "M1")
        self.assertEqual(summary["next_milestone"]["id"], "M2")
        self.assertEqual(summary["milestones"][1]["evidence_source"], "generate_depth_sweep")

    def test_decode_runtime_inputs_accept_shared_prompt_rotary(self) -> None:
        step = {
            "input_source": "prompt_runtime",
            "input_shapes": {
                "token_ids": [32, 1],
                "page_table": [32, 32],
                "cache_position": [32],
                "key_cache": [1024, 8, 32, 128],
                "value_cache": [1024, 8, 32, 128],
            },
            "kv_cache": {
                "physical_shape": [1024, 8, 32, 128],
                "logical_shape": [32, 1024, 8, 128],
                "page_block_size": 32,
                "page_count": 32,
                "max_num_blocks": 1024,
            },
            "synthetic_runtime_input_tensor_count": 0,
            "prompt_runtime_input_tensor_count": 1,
            "decode_runtime_state_input_tensor_count": 2,
            "rotary_runtime_input_tensor_count": 3,
            "rotary_runtime_state": {
                "shared_across_layers": True,
                "tensor_count": 3,
            },
            "kv_cache_runtime_input_tensor_count": 64,
            "synthetic_rotary_tensor_count": 0,
        }

        self.assertTrue(
            decode_runtime_inputs_complete(
                step,
                layer_count=32,
                batch_size=32,
                seq_len=1,
                cache_len=1024,
                num_kv_heads=8,
                head_dim=128,
                page_block_size=32,
            )
        )


if __name__ == "__main__":
    unittest.main()
