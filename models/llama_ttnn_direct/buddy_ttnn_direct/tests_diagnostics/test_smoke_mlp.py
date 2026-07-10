from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_mlp import (
    MLP_SMOKE_OPS,
    NO_TTNN_DEVICE_MESSAGE,
    run_smoke_mlp,
)


class SmokeMLPTest(unittest.TestCase):
    def test_smoke_mlp_dry_run_writes_report_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mlp_smoke_report.json"

            report = run_smoke_mlp(
                out=out,
                device="p150a",
                batch_size=2,
                hidden_size=16,
                intermediate_size=32,
                dtype_seed="bf16",
                dry_run=True,
            )

            dumped = json.loads(out.read_text())
            self.assertEqual(report, dumped)
            self.assertTrue(dumped["passed"])
            self.assertEqual(dumped["status"], "dry_run")
            self.assertTrue(dumped["dry_run"])
            self.assertIsNone(dumped["pcc"])
            self.assertEqual(dumped["latency_ms"], 0.0)
            self.assertEqual(dumped["ttnn_ops"], MLP_SMOKE_OPS)

    def test_smoke_mlp_without_device_writes_graceful_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mlp_smoke_report.json"

            report = run_smoke_mlp(
                out=out,
                device="p150a",
                batch_size=2,
                hidden_size=16,
                intermediate_size=32,
                dtype_seed="bf16",
                ttnn_module=types.SimpleNamespace(),
                torch_module=types.SimpleNamespace(),
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "no_device")
            self.assertEqual(report["error"], NO_TTNN_DEVICE_MESSAGE)
            self.assertEqual(json.loads(out.read_text()), report)

    def test_cli_smoke_mlp_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mlp_smoke_report.json"

            exit_code = main(
                [
                    "smoke-mlp",
                    "--device",
                    "p150a",
                    "--batch-size",
                    "2",
                    "--hidden-size",
                    "16",
                    "--intermediate-size",
                    "32",
                    "--dtype-seed",
                    "bf16",
                    "--dry-run",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["template"], "mlp_decode")
            self.assertTrue(report["passed"])
            self.assertEqual(report["ttnn_ops"], MLP_SMOKE_OPS)


if __name__ == "__main__":
    unittest.main()
