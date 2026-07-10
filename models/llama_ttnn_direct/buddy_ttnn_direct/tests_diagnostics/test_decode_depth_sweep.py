from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.search.decode_depth_sweep import (
    resolve_decode_depths,
    run_decode_depth_sweep,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_single_layer_decode import (
    _make_fake_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_validate_direct import (
    _write_fake_model_config,
    _write_template_config,
)


class DecodeDepthSweepTest(unittest.TestCase):
    def test_resolve_decode_depths_defaults_to_review_progression(self) -> None:
        self.assertEqual(
            resolve_decode_depths(None, program_num_layers=2),
            [1, 2],
        )
        self.assertEqual(
            resolve_decode_depths(None, program_num_layers=6),
            [1, 2, 4, 6],
        )
        self.assertEqual(
            resolve_decode_depths("1,2,full", program_num_layers=2),
            [1, 2],
        )
        with self.assertRaisesRegex(ValueError, "num_layers"):
            resolve_decode_depths("1,4", program_num_layers=2)

    def test_cli_decode_depth_sweep_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "depth_sweep.json"
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

            exit_code = main(
                [
                    "decode-depth-sweep",
                    "--program-dir",
                    str(program_dir),
                    "--depths",
                    "1,2,full",
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
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["command"], "decode-depth-sweep")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["program_num_layers"], 2)
            self.assertEqual(report["depths"], [1, 2])
            self.assertTrue(report["covered_full_depth"])
            self.assertEqual(report["status_counts"], {"dry_run": 2})
            self.assertEqual(report["passed_depth_count"], 2)
            self.assertEqual(report["failed_depths"], [])
            self.assertEqual(report["acceptance"]["status"], "passed")
            self.assertTrue(report["acceptance"]["passed"])
            self.assertTrue(
                all(check["passed"] for check in report["acceptance"]["checks"])
            )
            self.assertEqual(
                [record["layer_profile_ids"] for record in report["records"]],
                [[0], [0, 1]],
            )
            self.assertIn(
                "embedding_ms",
                report["records"][0]["section_latency_ms"],
            )
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in report["records"][1]["layer_profiles"]
                ],
                [0, 1],
            )
            self.assertIn(
                "argmax_status",
                report["records"][0]["lm_head_profile"],
            )
            self.assertIn(
                "decode_depth_sweep.profile_breakdown",
                [
                    check["name"]
                    for check in report["acceptance"]["checks"]
                ],
            )
            for depth in (1, 2):
                self.assertTrue(
                    (
                        root
                        / "depth_sweep_profiles"
                        / f"profile_depth_{depth}.json"
                    ).is_file()
                )

    def test_run_decode_depth_sweep_profiles_fake_real_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "depth_sweep.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
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

            with _fake_torch_and_safetensors():
                report = run_decode_depth_sweep(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    depths=[1, "full"],
                    batch_size=2,
                    cache_len=16,
                    device="p150a",
                    trace=True,
                    trace_iterations=2,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )

            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["passed"])
            self.assertEqual(report["depths"], [1, 2])
            self.assertEqual(report["status_counts"], {"profiled": 2})
            self.assertEqual(
                report["reference_status_counts"],
                {"passed": 2},
            )
            self.assertEqual(
                report["trace_status_counts"],
                {"captured_and_executed": 2},
            )
            self.assertEqual(report["failed_depths"], [])
            depth_one, depth_two = report["records"]
            self.assertEqual(depth_one["layer_profile_ids"], [0])
            self.assertEqual(depth_two["layer_profile_ids"], [0, 1])
            self.assertIn("embedding_ms", depth_two["section_latency_ms"])
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in depth_two["layer_profiles"]
                ],
                [0, 1],
            )
            self.assertEqual(
                depth_two["lm_head_profile"]["argmax_status"],
                "profiled",
            )
            self.assertEqual(depth_two["output_shapes"]["token"], [2, 1])
            self.assertEqual(
                [
                    layer["layer_id"]
                    for layer in depth_two["output_shapes"]["kv_cache_layers"]
                ],
                [0, 1],
            )
            self.assertGreater(depth_two["tokens_per_second_per_user"], 0.0)
            self.assertEqual(depth_two["trace_iterations"], 2)
            self.assertEqual(report["acceptance"]["failed_checks"], [])

    def test_run_decode_depth_sweep_accepts_partial_depth_when_allowed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "depth_sweep.json"
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

            report = run_decode_depth_sweep(
                out=report_json,
                program_dir=program_dir,
                depths=[1],
                batch_size=2,
                cache_len=16,
                dry_run=True,
                require_full_depth=False,
            )

            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertFalse(report["covered_full_depth"])
            self.assertFalse(report["require_full_depth"])
            self.assertEqual(report["depths"], [1])
            self.assertEqual(report["acceptance"]["failed_checks"], [])
            self.assertNotIn(
                "decode_depth_sweep.full_depth",
                [
                    check["name"]
                    for check in report["acceptance"]["checks"]
                ],
            )

    def test_decode_depth_sweep_reports_profile_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "depth_sweep.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
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

            import models.llama_ttnn_direct.buddy_ttnn_direct.search.decode_depth_sweep as sweep_module

            original_profile = sweep_module.profile_decode_step

            def profile_with_depth_two_failure(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                if kwargs["layers"] == 2:
                    profile["passed"] = False
                    profile["status"] = "reference_mismatch"
                    profile["reference"]["status"] = "failed"
                    profile["error"] = "forced depth two mismatch"
                    out = kwargs.get("out")
                    if out is not None:
                        Path(out).write_text(
                            json.dumps(profile, indent=2) + "\n"
                        )
                return profile

            with patch.object(
                sweep_module,
                "profile_decode_step",
                side_effect=profile_with_depth_two_failure,
            ):
                with _fake_torch_and_safetensors():
                    report = run_decode_depth_sweep(
                        out=report_json,
                        program_dir=program_dir,
                        model_path=model_dir,
                        depths=[1, 2],
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["passed"])
            self.assertEqual(report["failed_depths"], [2])
            self.assertEqual(report["status_counts"]["reference_mismatch"], 1)
            self.assertIn(
                "decode_depth_sweep.all_depths_passed",
                report["acceptance"]["failed_checks"],
            )
            self.assertEqual(
                report["records"][1]["error"],
                "forced depth two mismatch",
            )

    def test_decode_depth_sweep_fails_without_profile_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "depth_sweep.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
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

            import models.llama_ttnn_direct.buddy_ttnn_direct.search.decode_depth_sweep as sweep_module

            original_profile = sweep_module.profile_decode_step

            def profile_without_lm_head_breakdown(*args, **kwargs):
                profile = original_profile(*args, **kwargs)
                profile.pop("lm_head_profile", None)
                out = kwargs.get("out")
                if out is not None:
                    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
                return profile

            with patch.object(
                sweep_module,
                "profile_decode_step",
                side_effect=profile_without_lm_head_breakdown,
            ):
                with _fake_torch_and_safetensors():
                    report = run_decode_depth_sweep(
                        out=report_json,
                        program_dir=program_dir,
                        model_path=model_dir,
                        depths=[1],
                        batch_size=2,
                        cache_len=16,
                        device="p150a",
                        ttnn_module=_make_fake_ttnn(),
                        torch_module=_fake_torch(),
                        require_full_depth=False,
                    )

            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["passed"])
            self.assertIn(
                "decode_depth_sweep.profile_breakdown",
                report["acceptance"]["failed_checks"],
            )
            breakdown_check = [
                check
                for check in report["acceptance"]["checks"]
                if check["name"] == "decode_depth_sweep.profile_breakdown"
            ][0]
            self.assertEqual(
                breakdown_check["observed"][0]["lm_head_profile"],
                {},
            )


if __name__ == "__main__":
    unittest.main()
