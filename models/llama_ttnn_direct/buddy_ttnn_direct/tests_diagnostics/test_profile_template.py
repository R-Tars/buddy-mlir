from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.profile_template import (
    MLP_PROFILE_OPS,
    profile_template,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_mlp import (
    NO_TTNN_DEVICE_MESSAGE,
)


class ProfileTemplateTest(unittest.TestCase):
    def test_profile_template_dry_run_writes_no_trace_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            out = root / "profile.json"
            _write_config(config)

            report = profile_template(
                template="mlp_decode",
                config_path=config,
                out=out,
                warmup=2,
                iterations=4,
                trace=False,
                dry_run=True,
            )

            dumped = json.loads(out.read_text())
            self.assertEqual(report, dumped)
            self.assertEqual(dumped["status"], "dry_run")
            self.assertFalse(dumped["trace_enabled"])
            self.assertEqual(dumped["trace"], {"requested": False, "status": "disabled"})
            self.assertEqual(dumped["latency_ms"], {"mean": 0.0, "p50": 0.0, "p90": 0.0})
            self.assertEqual(dumped["ops"], MLP_PROFILE_OPS)
            self.assertEqual(dumped["device"], "p150a")
            self.assertEqual(dumped["batch_size"], 32)
            self.assertEqual(dumped["hidden_size"], 16)
            self.assertEqual(dumped["intermediate_size"], 32)

    def test_profile_template_dry_run_writes_trace_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            out = root / "profile_trace.json"
            _write_config(config)

            profile_template(
                template="mlp_decode",
                config_path=config,
                out=out,
                warmup=5,
                iterations=20,
                trace=True,
                dry_run=True,
            )

            dumped = json.loads(out.read_text())
            self.assertTrue(dumped["trace_enabled"])
            self.assertEqual(dumped["warmup"], 5)
            self.assertEqual(dumped["iterations"], 20)
            self.assertEqual(dumped["trace"], {"requested": True, "status": "dry_run"})
            self.assertEqual(dumped["latency_ms"]["mean"], 0.0)

    def test_profile_template_without_device_writes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            out = root / "profile_no_device.json"
            _write_config(config)

            report = profile_template(
                template="mlp_decode",
                config_path=config,
                out=out,
                warmup=0,
                iterations=2,
                trace=True,
                ttnn_module=types.SimpleNamespace(),
                torch_module=types.SimpleNamespace(),
            )

            self.assertEqual(report["status"], "no_device")
            self.assertEqual(report["error"], NO_TTNN_DEVICE_MESSAGE)
            self.assertEqual(report["latency_ms"], {"mean": 0.0, "p50": 0.0, "p90": 0.0})
            self.assertEqual(json.loads(out.read_text()), report)

    def test_cli_profile_template_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            out = root / "profile.json"
            _write_config(config)

            exit_code = main(
                [
                    "profile-template",
                    "--template",
                    "mlp_decode",
                    "--config",
                    str(config),
                    "--warmup",
                    "1",
                    "--iterations",
                    "3",
                    "--trace",
                    "--dry-run",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            dumped = json.loads(out.read_text())
            self.assertEqual(dumped["template"], "mlp_decode")
            self.assertEqual(dumped["trace"], {"requested": True, "status": "dry_run"})
            self.assertEqual(dumped["ops"], MLP_PROFILE_OPS)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "fake-llama-profile",
                "template_config": {"device": "p150a"},
                "batch_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
