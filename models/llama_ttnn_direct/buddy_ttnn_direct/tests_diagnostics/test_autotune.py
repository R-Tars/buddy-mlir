from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.ttnn_tensorizer import (
    load_parameter_config_from_program,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.autotune import (
    _confirmation_promotion_decision,
    _select_winner,
    run_layered_autotune,
)


class LayeredAutotuneTest(unittest.TestCase):
    def test_confirmation_rejects_subthreshold_winner(self) -> None:
        decision = _confirmation_promotion_decision(
            challenger={"passed": True, "metric_value": 100.6},
            incumbent={"passed": True, "metric_value": 100.0},
            min_relative_improvement=0.01,
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["status"], "rejected")

    def test_runtime_dtype_seed_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, config_path = _write_inputs(root)
            with self.assertRaisesRegex(ValueError, "frozen to 'bf16'"):
                run_layered_autotune(
                    model_path=model_path,
                    config_path=config_path,
                    out=root / "autotune.json",
                    prompt=None,
                    layers=2,
                    dtype_seed="fp32",
                    dry_run=True,
                )

    def test_selection_keeps_incumbent_below_improvement_threshold(self) -> None:
        incumbent = {
            "candidate_id": "incumbent",
            "passed": True,
            "metric_value": 100.0,
        }
        challenger = {
            "candidate_id": "challenger",
            "passed": True,
            "metric_value": 100.9,
        }
        winner = _select_winner(
            [incumbent, challenger],
            dry_run=False,
            min_relative_improvement=0.01,
        )
        self.assertIs(winner, incumbent)

    def test_dry_run_builds_three_precision_frozen_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, config_path = _write_inputs(root)
            report = run_layered_autotune(
                model_path=model_path,
                config_path=config_path,
                out=root / "autotune.json",
                prompt=None,
                layers=2,
                dry_run=True,
            )

            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(
                [level["name"] for level in report["levels"]],
                [
                    "lm_head_split_count",
                    "memory_config_layout",
                    "program_config_core_grid",
                ],
            )
            self.assertEqual(report["unique_measurement_count"], 4)
            self.assertTrue(
                all(level["candidate_count"] == 2 for level in report["levels"])
            )

            split16 = next(
                candidate
                for candidate in report["levels"][0]["candidates"]
                if candidate["varied_value"] == 16
            )
            program_dir = Path(split16["program_dir"])
            runtime_config = json.loads((program_dir / "config.json").read_text())
            self.assertEqual(runtime_config["lm_head"]["split_count"], 16)
            self.assertEqual(
                len(runtime_config["lm_head"]["program_configs"]),
                16,
            )
            self.assertEqual(
                {
                    item["per_core_N"]
                    for item in runtime_config["lm_head"]["program_configs"]
                },
                {4},
            )
            parameter_config = load_parameter_config_from_program(program_dir)
            lm_head_weight = next(
                value
                for value in parameter_config["weights"].values()
                if value["role"] == "lm_head"
            )
            self.assertEqual(lm_head_weight["memory_config"]["n"], 8016)

            candidate_payload = json.loads(
                (program_dir.parent / "candidate.json").read_text()
            )
            self.assertNotIn("dtype_recipe", candidate_payload["state"])
            self.assertEqual(candidate_payload["state"]["space"]["schema_version"], 2)
            self.assertNotIn(
                "exp_approx_mode",
                candidate_payload["state"]["space"]["operators"]["attention.sdpa"],
            )
            self.assertTrue(candidate_payload["config"]["precision_contract"]["frozen"])
            self.assertEqual(
                candidate_payload["config"]["execution_contract"]["execution_mode"],
                "trace",
            )

    def test_real_selection_reuses_previous_level_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, config_path = _write_inputs(root)
            calls = []

            def fake_profile(**kwargs):
                runtime_config = json.loads(
                    (Path(kwargs["program_dir"]) / "config.json").read_text()
                )
                state = {
                    "lm_head_split_count": runtime_config["lm_head"]["split_count"],
                    "memory_layout": (
                        "lm_head_dram_concat"
                        if runtime_config["lm_head"]["concat_memory_config"]["name"]
                        == "DRAM_MEMORY_CONFIG"
                        else "official_l1_sharded"
                    ),
                    "program_config": (
                        "sdpa_grid_8x4"
                        if runtime_config["attention"]["sdpa_program_config"][
                            "core_grid"
                        ]
                        == [8, 4]
                        else "official"
                    ),
                }
                self.assertEqual(runtime_config["autotune"]["schema_version"], 2)
                self.assertNotIn("memory_layout", runtime_config["autotune"])
                self.assertNotIn("program_config", runtime_config["autotune"])
                self.assertEqual(
                    runtime_config["template_config"]["dtype_recipe"],
                    "official_like_performance_seed",
                )
                self.assertTrue(kwargs["after_prefill"])
                self.assertEqual(kwargs["execution_mode"], "trace")
                self.assertEqual(kwargs["runtime_input_mode"], "persistent")
                score = 10.0
                score += state["lm_head_split_count"] == 16
                score += state["memory_layout"] == "lm_head_dram_concat"
                score += state["program_config"] == "sdpa_grid_8x4"
                calls.append(state)
                report = {
                    "status": "profiled",
                    "passed": True,
                    "tokens_per_second_per_user": score,
                    "aggregate_tokens_per_second": score * 32,
                    "decode_step_ms_p50": 1000.0 / score,
                    "decode_step_ms_mean": 1000.0 / score,
                    "setup_ms": 1.0,
                    "prefill_ms": 1.0,
                    "error": None,
                }
                Path(kwargs["out"]).write_text(json.dumps(report))
                return report

            with patch(
                "models.llama_ttnn_direct.buddy_ttnn_direct."
                "diagnostics.autotune.candidate.check_device_ownership",
                return_value={"checked": True, "available": True},
            ):
                report = run_layered_autotune(
                    model_path=model_path,
                    config_path=config_path,
                    out=root / "autotune.json",
                    prompt="test prompt",
                    layers=2,
                    warmup=1,
                    iterations=2,
                    confirm_warmup=1,
                    confirm_iterations=3,
                    resume=False,
                    profile_runner=fake_profile,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["unique_measurement_count"], 4)
            self.assertEqual(len(calls), 6)
            self.assertEqual(
                report["selected_state"],
                {
                    "lm_head_split_count": 16,
                    "memory_layout": "lm_head_dram_concat",
                    "program_config": "sdpa_grid_8x4",
                },
            )
            self.assertTrue(
                all(
                    level["candidates"][0]["measurement_reused"]
                    for level in report["levels"][1:]
                )
            )
            self.assertEqual(
                report["confirmation"]["iterations"],
                3,
            )
            self.assertTrue(report["promotion_decision"]["promoted"])
            self.assertEqual(
                report["invocation"]["execution_contract"]["runtime_input_mode"],
                "persistent",
            )


def _write_inputs(root: Path) -> tuple[Path, Path]:
    model_path = root / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-llama-autotune",
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
    config_path = root / "seed.json"
    config_path.write_text(
        json.dumps(
            {
                "device": "p150a",
                "model": "llama3.1-8b",
                "batch_size": 32,
                "decode_seq_len": 1,
                "prefill_seq_len": 128,
                "max_cache_len": 1024,
                "runtime_input_mode": "persistent",
                "attention_template": "official_paged_attention_decode",
                "mlp_template": "official_gated_mlp_decode",
                "lm_head_template": "official_split_lm_head",
                "kv_cache_template": "paged_kv_cache",
                "generation_template": "device_argmax_greedy",
                "lm_head_argmax_strategy": ("full_logits_untilize_multicore_argmax"),
                "lm_head_split_count": 8,
                "dtype_recipe": "official_like_performance_seed",
                "official_config_profile": ("p150a_llama31_8b_b32_performance"),
            }
        )
    )
    return model_path, config_path


if __name__ == "__main__":
    unittest.main()
