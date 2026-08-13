from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    build_baseline_artifact,
    verify_baseline_artifact,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    baseline as baseline_module,
)


class BaselineArtifactTest(unittest.TestCase):
    def test_builds_and_verifies_phase_zero_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))

            report = self._build(fixture)

            self.assertTrue(report["passed"])
            self.assertEqual(len(report["repetitions"]), 3)
            self.assertEqual(
                [len(row["decode_step_ms_samples"]) for row in report["repetitions"]],
                [100, 100, 100],
            )
            summary = report["summary"]["tokens_per_second_per_user"]
            self.assertEqual(summary["median"], 34.0)
            self.assertLess(summary["coefficient_of_variation"], 0.015)
            for filename in (
                "baseline_decode.json",
                "baseline_op_graph.json",
                "baseline_config.json",
                "baseline_runtime_identity.json",
                "baseline_source_manifest.json",
                "baseline_artifact_manifest.json",
            ):
                self.assertTrue((fixture["out_dir"] / filename).is_file())
            verification = verify_baseline_artifact(fixture["out_dir"])
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["checked_artifact_file_count"], 5)
            self.assertEqual(verification["checked_source_count"], 9)

    def test_contract_violation_writes_classified_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            path = fixture["reports"][1]
            payload = json.loads(path.read_text())
            payload["iterations"] = 99
            _write_json(path, payload)

            report = self._build(fixture)

            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["type"], "BaselineArtifactError")
            self.assertIn("violates the baseline contract", report["error"]["message"])
            self.assertTrue(
                (fixture["out_dir"] / "baseline_failure.json").is_file()
            )

    def test_shared_decode_contract_rejects_host_token_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            path = fixture["reports"][0]
            payload = json.loads(path.read_text())
            payload["runtime_inputs"]["token_update"] = "host_copy"
            _write_json(path, payload)

            report = self._build(fixture)

            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["type"], "BaselineArtifactError")
            self.assertIn("runtime_inputs.token_update", report["error"]["message"])

    def test_verifier_detects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            self.assertTrue(self._build(fixture)["passed"])
            config_path = fixture["out_dir"] / "baseline_config.json"
            config = json.loads(config_path.read_text())
            config["runtime_input_mode"] = "eager"
            _write_json(config_path, config)

            verification = verify_baseline_artifact(fixture["out_dir"])

            self.assertFalse(verification["passed"])
            self.assertIn("baseline config hash mismatch", verification["errors"])
            self.assertTrue(
                any("artifact file hash mismatch" in row for row in verification["errors"])
            )

    def test_verifier_detects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            self.assertTrue(self._build(fixture)["passed"])
            fixture["prompt_corpus"].write_text("changed\n")

            verification = verify_baseline_artifact(fixture["out_dir"])

            self.assertFalse(verification["passed"])
            self.assertTrue(
                any(
                    "source prompt_corpus file hash mismatch" in row
                    for row in verification["errors"]
                )
            )

    def _build(self, fixture: dict) -> dict:
        with mock.patch.object(
            baseline_module, "_git_commit", return_value="a" * 40
        ):
            return build_baseline_artifact(
                fixture["reports"],
                program_dir=fixture["program_dir"],
                seed_config=fixture["seed_config"],
                prompt_corpus=fixture["prompt_corpus"],
                ttnn_binary=fixture["ttnn_binary"],
                out_dir=fixture["out_dir"],
                repo_root=fixture["buddy_repo"],
                buddy_commit="b" * 40,
                tt_metal_root=fixture["tt_metal_repo"],
            )

    def _fixture(self, root: Path) -> dict:
        program_dir = root / "program"
        program_dir.mkdir()
        seed_config = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "p150a_llama31_8b_b32.json"
        )
        seed = json.loads(seed_config.read_text())
        _write_json(
            program_dir / "config.json",
            {
                "schema_version": 1,
                "num_layers": 32,
                "official_config_profile": seed["official_config_profile"],
                "runtime_input_mode": "persistent",
                "generation": {
                    "template": "device_argmax_greedy",
                    "mode": "greedy",
                },
                "autotune": {
                    "templates": {"attention.rope": "separate_qk_rope"},
                    "operators": {"attention.qkv": {"kind": "matmul"}},
                    "edges": {"qkv_to_rope": {"conversion": "none"}},
                    "memory_configs": {},
                    "core_grids": {},
                    "extra_program_configs": {},
                },
            },
        )
        _write_json(
            program_dir / "execution_plan.json",
            {"schema_version": 1, "mode": "decode", "layers": []},
        )
        (program_dir / "model.py").write_text("MODEL = 'fixture'\n")
        prompt_corpus = root / "prompts.json"
        _write_json(prompt_corpus, ["fixture prompt"] * 32)
        ttnn_binary = root / "_ttnn.so"
        ttnn_binary.write_bytes(b"fixture-binary")
        buddy_repo = root / "buddy"
        buddy_repo.mkdir()
        tt_metal_repo = root / "tt-metal"
        tt_metal_repo.mkdir()
        reports: list[Path] = []
        for index, throughput in enumerate((33.9, 34.0, 34.1)):
            path = root / f"rep{index}.json"
            samples = [1000.0 / throughput + offset * 0.0001 for offset in range(100)]
            _write_json(
                path,
                {
                    "schema_version": 1,
                    "status": "profiled",
                    "passed": True,
                    "acceptance": {"status": "passed", "passed": True},
                    "mode": "decode-steady",
                    "layers": 32,
                    "batch_size": 32,
                    "prefill_len": 256,
                    "cache_len": 1024,
                    "warmup": 5,
                    "iterations": 100,
                    "execution_mode": "trace",
                    "runtime_input_mode": "persistent",
                    "after_prefill": True,
                    "prefill_execution_mode": "eager",
                    "device": "p150a",
                    "device_id": 0,
                    "trace_capture_count": 1,
                    "trace_execute_count": 105,
                    "persistent_input_count": 7,
                    "program_compile_count_after_capture": 0,
                    "program_dir": str(program_dir.resolve()),
                    "tokens_per_second_per_user": throughput,
                    "decode_step_ms_samples": samples,
                    "warmup_step_ms_samples": samples[:5],
                    "trace_key": {
                        "device_id": 0,
                        "program_config_hash": "c" * 64,
                        "layer_count": 32,
                        "batch_size": 32,
                        "cache_len": 1024,
                        "dtype_recipe": "official_like_performance_seed",
                        "argmax_strategy": "full_logits_untilize_multicore_argmax",
                    },
                    "runtime_context": {
                        "decode_token_runtime_handoff": "device_tensor_direct",
                        "decode_token_host_roundtrip_per_step": False,
                    },
                    "runtime_inputs": {
                        "new_device_tensors_per_decode_step": 0,
                        "host_to_device_updates_per_decode_step": 0,
                        "page_table_reused": True,
                        "token_update": "captured_device_to_device_copy",
                    },
                    "ttnn_environment": {
                        "tt_metal_home": str(tt_metal_repo.resolve()),
                        "tt_metal_git_commit": "a" * 40,
                    },
                },
            )
            reports.append(path)
        return {
            "reports": reports,
            "program_dir": program_dir,
            "seed_config": seed_config,
            "prompt_corpus": prompt_corpus,
            "ttnn_binary": ttnn_binary,
            "out_dir": root / "artifact",
            "buddy_repo": buddy_repo,
            "tt_metal_repo": tt_metal_repo,
        }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
