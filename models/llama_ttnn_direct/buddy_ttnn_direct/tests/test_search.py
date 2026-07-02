from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.search.decode_step_autotune import (
    run_decode_step_autotune,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.search.space import (
    candidate_id,
    enumerate_candidate_configs,
    load_search_space,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.dump import (
    dump_graph_json,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_single_layer_decode import (
    _make_fake_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)


class SearchTest(unittest.TestCase):
    def test_lm_head_search_space_enumerates_deterministic_candidates(
        self,
    ) -> None:
        candidates = enumerate_candidate_configs(
            _seed_config(),
            {
                "lm_head_split_count": [1, 2, 8],
                "generation_template": [
                    "full_logits",
                    "device_argmax_greedy",
                ],
            },
        )

        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            [
                "lm_split_1_full_logits",
                "lm_split_1_argmax",
                "lm_split_2_full_logits",
                "lm_split_2_argmax",
                "lm_split_8_full_logits",
                "lm_split_8_argmax",
            ],
        )
        self.assertEqual(
            candidate_id(16, "device_argmax_greedy"),
            "lm_split_16_argmax",
        )
        self.assertEqual(candidates[0]["config"]["lm_head_split_count"], 1)
        self.assertEqual(
            candidates[0]["config"]["generation_template"], "full_logits"
        )

    def test_decode_step_search_space_adds_optional_knob_suffixes(self) -> None:
        candidates = enumerate_candidate_configs(
            _seed_config(),
            {
                "lm_head_split_count": [2],
                "generation_template": ["device_argmax_greedy"],
                "mlp_intermediate_dtype": [None, "bfloat8_b"],
                "attention_sdpa_output_memory_config": [None, "l1"],
                "attention_concat_heads_output_memory_config": [None],
            },
        )

        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            [
                "lm_split_2_argmax",
                "lm_split_2_argmax_sdpa_l1",
                "lm_split_2_argmax_mlp_bfloat8_b",
                "lm_split_2_argmax_mlp_bfloat8_b_sdpa_l1",
            ],
        )
        self.assertEqual(
            candidate_id(
                4,
                "full_logits",
                mlp_intermediate_dtype="bfloat8_b",
                attention_sdpa_output_memory_config="l1",
                attention_concat_heads_output_memory_config="l1",
            ),
            "lm_split_4_full_logits_mlp_bfloat8_b_sdpa_l1_concat_l1",
        )

    def test_cli_search_dry_run_generates_configs_plans_and_codegen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_json = root / "semantic.json"
            base_config = root / "base_config.json"
            space_json = root / "space.json"
            report_json = root / "search_report.json"
            dump_graph_json(_fake_graph(), semantic_json)
            base_config.write_text(json.dumps(_seed_config()))
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [1, 2],
                        "generation_template": [
                            "full_logits",
                            "device_argmax_greedy",
                        ],
                    }
                )
            )

            exit_code = main(
                [
                    "search",
                    "--semantic-json",
                    str(semantic_json),
                    "--base-config",
                    str(base_config),
                    "--space",
                    str(space_json),
                    "--metric",
                    "latency_ms",
                    "--out",
                    str(report_json),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["metric"], "latency_ms")
            self.assertEqual(report["candidate_count"], 4)
            self.assertIsNone(report["best"])
            self.assertEqual(
                [candidate["status"] for candidate in report["candidates"]],
                ["dry_run_generated"] * 4,
            )

            candidate_root = root / report["candidates_dir"]
            full_logits = candidate_root / "lm_split_2_full_logits"
            argmax = candidate_root / "lm_split_2_argmax"
            self.assertTrue((full_logits / "config.json").is_file())
            self.assertTrue((full_logits / "execution_plan.json").is_file())
            self.assertTrue((full_logits / "codegen" / "model.py").is_file())
            self.assertTrue((argmax / "codegen" / "config.json").is_file())

            full_plan = json.loads(
                (full_logits / "execution_plan.json").read_text()
            )
            argmax_plan = json.loads(
                (argmax / "execution_plan.json").read_text()
            )
            self.assertEqual(full_plan["final"][-1], "full_logits")
            self.assertEqual(
                argmax_plan["final"][-1], "device_argmax_greedy"
            )
            generated_config = json.loads(
                (full_logits / "codegen" / "config.json").read_text()
            )
            self.assertEqual(generated_config["lm_head"]["split_count"], 2)
            self.assertTrue(generated_config["generation"]["retain_logits"])
            self.assertIn(
                "official_split_lm_head + full_logits",
                (full_logits / "codegen" / "model.py").read_text(),
            )

    def test_default_space_file_loads_expected_candidate_count(self) -> None:
        repo_root = Path(__file__).parents[4]
        space = load_search_space(
            repo_root
            / "models"
            / "llama_ttnn_direct"
            / "buddy_ttnn_direct"
            / "search"
            / "spaces"
            / "lm_head_minimal.json"
        )

        self.assertEqual(len(space["lm_head_split_count"]), 5)
        self.assertEqual(len(space["generation_template"]), 2)
        self.assertEqual(
            len(enumerate_candidate_configs(_seed_config(), space)), 10
        )

    def test_cli_autotune_decode_step_dry_run_generates_candidate_configs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            space_json = root / "decode_step_space.json"
            report_json = root / "decode_step_autotune.json"
            _write_fake_model_config(model_dir)
            config_json.write_text(json.dumps(_seed_config()))
            space_json.write_text(
                json.dumps(
                    {
                        "lm_head_split_count": [2],
                        "generation_template": ["device_argmax_greedy"],
                        "mlp_intermediate_dtype": [None, "bfloat8_b"],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    }
                )
            )
            self.assertEqual(
                main(
                    [
                        "build-program",
                        "--model-path",
                        str(model_dir),
                        "--config",
                        str(config_json),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )

            exit_code = main(
                [
                    "autotune-decode-step",
                    "--program-dir",
                    str(program_dir),
                    "--space",
                    str(space_json),
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["search"], "decode_step_minimal")
            self.assertEqual(report["candidate_count"], 2)
            self.assertIsNone(report["best"])
            candidate_root = root / report["candidates_dir"]
            tuned_config = json.loads(
                (
                    candidate_root
                    / "lm_split_2_argmax_mlp_bfloat8_b"
                    / "config.json"
                ).read_text()
            )
            self.assertEqual(tuned_config["lm_head"]["split_count"], 2)
            self.assertEqual(
                tuned_config["mlp"]["intermediate_dtype"],
                "bfloat8_b",
            )
            self.assertFalse(tuned_config["lm_head"]["retain_logits"])
            self.assertTrue(
                (
                    candidate_root
                    / "lm_split_2_argmax_mlp_bfloat8_b"
                    / "semantic_graph.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    candidate_root
                    / "lm_split_2_argmax_mlp_bfloat8_b"
                    / "weights_manifest.json"
                ).is_file()
            )

    def test_decode_step_autotune_profiles_fake_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_autotune.json"
            _write_fake_model_config(model_dir)
            config_json.write_text(json.dumps(_seed_config()))
            self.assertEqual(
                main(
                    [
                        "build-program",
                        "--model-path",
                        str(model_dir),
                        "--config",
                        str(config_json),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )

            report = run_decode_step_autotune(
                program_dir=program_dir,
                space={
                    "lm_head_split_count": [2],
                    "generation_template": ["device_argmax_greedy"],
                    "mlp_intermediate_dtype": [None],
                    "attention_sdpa_output_memory_config": [None],
                    "attention_concat_heads_output_memory_config": [None],
                },
                out=report_json,
                layers=1,
                batch_size=2,
                cache_len=16,
                ttnn_module=_make_fake_ttnn(),
                torch_module=_fake_torch(),
            )

            self.assertFalse(report["dry_run"])
            self.assertEqual(report["candidate_count"], 1)
            self.assertIsNotNone(report["best"])
            self.assertEqual(report["best"]["status"], "profiled")
            self.assertIsInstance(report["best"]["metric"], float)
            self.assertEqual(
                report["candidates"][0]["knobs"]["lm_head_split_count"],
                2,
            )

    def test_decode_step_autotune_profiles_fake_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_autotune.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            config_json.write_text(json.dumps(_seed_config()))
            self.assertEqual(
                main(
                    [
                        "build-program",
                        "--model-path",
                        str(model_dir),
                        "--config",
                        str(config_json),
                        "--out-dir",
                        str(program_dir),
                    ]
                ),
                0,
            )

            with _fake_torch_and_safetensors():
                report = run_decode_step_autotune(
                    program_dir=program_dir,
                    model_path=model_dir,
                    space={
                        "lm_head_split_count": [2],
                        "generation_template": ["device_argmax_greedy"],
                        "mlp_intermediate_dtype": [None],
                        "attention_sdpa_output_memory_config": [None],
                        "attention_concat_heads_output_memory_config": [None],
                    },
                    out=report_json,
                    layers=1,
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertFalse(report["dry_run"])
            self.assertEqual(report["model_path"], str(model_dir))
            self.assertEqual(report["candidate_count"], 1)
            self.assertIsNotNone(report["best"])
            self.assertEqual(report["best"]["status"], "profiled")
            self.assertEqual(report["best"]["parameter_source"], "hf_model")
            self.assertEqual(
                report["best"]["parameter_setup"]["tensorization"]["tensor_count"],
                11,
            )
            candidate = report["candidates"][0]
            self.assertEqual(candidate["parameter_source"], "hf_model")
            self.assertEqual(
                candidate["parameter_setup"]["materialization"]["tensor_count"],
                15,
            )
            profile_report = json.loads(
                (root / candidate["profile_report"]).read_text()
            )
            self.assertEqual(profile_report["parameter_source"], "hf_model")
            self.assertEqual(profile_report["input_source"], "synthetic")
            self.assertEqual(profile_report["tensor_conversion_count"], 19)
            self.assertEqual(
                profile_report["parameter_setup"]["tensorization"]["tensor_count"],
                11,
            )
            self.assertEqual(
                profile_report["parameter_setup"]["synthetic_rotary_tensor_count"],
                3,
            )
            self.assertEqual(
                profile_report["parameter_setup"][
                    "synthetic_runtime_input_tensor_count"
                ],
                5,
            )
            candidate_root = root / report["candidates_dir"] / candidate["id"]
            self.assertTrue((candidate_root / "semantic_graph.json").is_file())
            self.assertTrue((candidate_root / "weights_manifest.json").is_file())


def _fake_graph():
    return import_hf_llama(
        "/tmp/fake-llama-search",
        config={
            "_name_or_path": "fake-llama-search",
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


def _seed_config() -> dict[str, object]:
    return {
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
    }


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-search",
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
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
