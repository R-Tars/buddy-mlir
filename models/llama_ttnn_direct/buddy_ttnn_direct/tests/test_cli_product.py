from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import (
    build_parser,
    main,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)


class ProductCliTest(unittest.TestCase):
    def test_top_level_help_shows_product_commands_only(self) -> None:
        help_text = build_parser().format_help()

        for command in (
            "build",
            "generate",
            "profile",
            "validate",
            "inspect",
            "diagnose",
        ):
            self.assertIn(command, help_text)
        for legacy in (
            "build-program",
            "profile-generate",
            "validate-direct",
            "smoke-prefill",
            "generate-depth-sweep",
            "search",
        ):
            self.assertNotIn(legacy, help_text)

    def test_build_alias_writes_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)

            exit_code = main(
                [
                    "build",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(program_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((program_dir / "model.py").is_file())
            self.assertTrue((program_dir / "run_decode.py").is_file())

    def test_profile_alias_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            report_json = root / "profile.json"
            underlying_generate = root / "profile_generate.json"

            exit_code = main(
                [
                    "profile",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "2",
                    "--prefill-len",
                    "8",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--generate-report",
                    str(underlying_generate),
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "profile-generate")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])

    def test_validate_dryrun_program_dir_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            out_dir = root / "validate"

            exit_code = main(
                [
                    "validate",
                    "--suite",
                    "dryrun",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "2",
                    "--prefill-len",
                    "8",
                    "--layers",
                    "1",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "validation_report.json").read_text())
            self.assertEqual(report["command"], "validate")
            self.assertEqual(report["suite"], "dryrun")
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["failed_checks"], [])
            self.assertTrue((out_dir / "generate_dryrun.json").is_file())
            self.assertTrue((out_dir / "profile_dryrun.json").is_file())

    def test_inspect_writes_program_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program_dir = self._build_program(root)
            report_json = root / "inspect.json"

            exit_code = main(
                [
                    "inspect",
                    "--program-dir",
                    str(program_dir),
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["command"], "inspect")
            self.assertTrue(report["passed"])
            self.assertEqual(report["artifacts"]["missing"], [])

    def test_diagnose_mlp_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = Path(tmpdir) / "diagnose_mlp.json"

            exit_code = main(
                [
                    "diagnose",
                    "--stage",
                    "mlp",
                    "--batch-size",
                    "2",
                    "--hidden-size",
                    "16",
                    "--intermediate-size",
                    "32",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])

    def _build_program(self, root: Path) -> Path:
        model_dir = root / "fake_model"
        config_json = root / "template_config.json"
        program_dir = root / "program"
        _write_fake_model_config(model_dir)
        _write_template_config(config_json)
        self.assertEqual(
            main(
                [
                    "build",
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
        return program_dir


if __name__ == "__main__":
    unittest.main()
