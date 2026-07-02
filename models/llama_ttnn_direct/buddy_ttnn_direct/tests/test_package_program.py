from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.package import (
    PACKAGE_BACKEND,
    PACKAGE_PROGRAM_TYPE,
    package_dry_run_report,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.program import (
    PROGRAM_ARTIFACTS,
)


class PackageProgramTest(unittest.TestCase):
    def test_cli_package_program_writes_direct_package_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            package_dir = root / "package"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

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
            self.assertEqual(
                main(
                    [
                        "package-program",
                        "--program-dir",
                        str(program_dir),
                        "--out-dir",
                        str(package_dir),
                    ]
                ),
                0,
            )

            expected_files = set(PROGRAM_ARTIFACTS)
            expected_files.update({"manifest.json", "PACKAGE_README.md"})
            self.assertEqual(
                {path.name for path in package_dir.iterdir()},
                expected_files,
            )
            manifest = json.loads((package_dir / "manifest.json").read_text())
            self.assertEqual(manifest["backend"], PACKAGE_BACKEND)
            self.assertEqual(manifest["program_type"], PACKAGE_PROGRAM_TYPE)
            self.assertEqual(manifest["entrypoint"], "model.py")
            self.assertEqual(manifest["semantic_graph"], "semantic_graph.json")
            self.assertEqual(
                manifest["execution_plan"], "execution_plan.json"
            )
            self.assertEqual(
                manifest["weights_manifest"], "weights_manifest.json"
            )
            self.assertFalse(manifest["runtime"]["buddy_cli_supported"])
            self.assertTrue(manifest["runtime"]["python_runner_supported"])
            self.assertEqual(manifest["runtime"]["python_runner"], "run_decode.py")
            self.assertEqual(
                manifest["runtime"]["runner_modes"],
                ["inspect", "smoke", "profile", "validate-real"],
            )
            self.assertTrue(manifest["runtime"]["dry_run_supported"])
            self.assertTrue(
                manifest["runtime"]["real_weight_validation_supported"]
            )
            self.assertEqual(manifest["model_name"], "fake-package-program")
            self.assertEqual(manifest["num_layers"], 2)
            package_readme = (package_dir / "PACKAGE_README.md").read_text()
            self.assertIn("python run_decode.py --mode smoke", package_readme)
            self.assertIn("validate-real", package_readme)

    def test_package_program_dry_run_reports_manifest_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            package_dir = root / "package"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
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

            self.assertEqual(
                main(
                    [
                        "package-program",
                        "--program-dir",
                        str(program_dir),
                        "--out-dir",
                        str(package_dir),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertFalse(package_dir.exists())

            report = package_dry_run_report(program_dir, package_dir)

            self.assertTrue(report["dry_run"])
            self.assertEqual(report["backend"], PACKAGE_BACKEND)
            self.assertEqual(report["program_type"], PACKAGE_PROGRAM_TYPE)
            self.assertEqual(
                report["manifest"]["entrypoint"],
                "model.py",
            )
            self.assertTrue(
                report["manifest"]["runtime"]["python_runner_supported"]
            )
            self.assertIn("manifest.json", report["artifacts"])

    def test_cmake_target_is_additive_and_separate_from_llama31_tt_rax(
        self,
    ) -> None:
        repo_root = Path(__file__).parents[4]
        top_cmake = (repo_root / "CMakeLists.txt").read_text()
        models_cmake = (repo_root / "models" / "CMakeLists.txt").read_text()
        direct_cmake = (
            repo_root / "models" / "llama_ttnn_direct" / "CMakeLists.txt"
        ).read_text()

        self.assertIn("BUDDY_BUILD_LLAMA31_TTNN_DIRECT_MODEL", top_cmake)
        self.assertIn("add_subdirectory(llama_ttnn_direct)", models_cmake)
        self.assertIn("llama31_ttnn_direct_program", direct_cmake)
        self.assertIn("llama31_ttnn_direct_package", direct_cmake)
        self.assertNotIn("llama31_tt_rax", direct_cmake)


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-package-program",
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


def _write_template_config(path: Path) -> None:
    path.write_text(
        json.dumps(
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
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
