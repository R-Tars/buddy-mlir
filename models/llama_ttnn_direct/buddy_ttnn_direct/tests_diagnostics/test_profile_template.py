from __future__ import annotations

import json
import tempfile
import types
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics import template_profile
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_mlp import NO_TTNN_DEVICE_MESSAGE


class ProfileTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config, self.out = self.root / "config.json", self.root / "profile.json"
        _write_config(self.config)

    def profile(self, **overrides):
        args = dict(template="mlp_decode", config_path=self.config, out=self.out, warmup=1, iterations=2)
        args.update(overrides)
        return template_profile.profile_template(**args)

    def hardware_profile(self, fake, *, trace):
        with patch.object(template_profile, "managed_mlp_device", fake.managed), \
             patch.object(template_profile, "prepare_mlp_smoke_on_device", fake.prepare):
            return self.profile(trace=trace, ttnn_module=fake, torch_module=object())

    def test_profile_template_dry_run_modes(self) -> None:
        for trace, status in ((False, "disabled"), (True, "dry_run")):
            with self.subTest(trace=trace):
                report = self.profile(trace=trace, dry_run=True)
                self.assertEqual(report, json.loads(self.out.read_text()))
                self.assertEqual(report["status"], "dry_run")
                self.assertEqual(report["trace"], {"requested": trace, "status": status})
                self.assertEqual(report["latency_ms"], {"mean": 0.0, "p50": 0.0, "p90": 0.0})
                self.assertEqual(report["ops"], template_profile.MLP_PROFILE_OPS)
                self.assertEqual(
                    (report["device"], report["batch_size"], report["hidden_size"],
                     report["intermediate_size"]), ("p150a", 32, 16, 32))

    def test_profile_template_without_device_writes_schema(self) -> None:
        report = self.profile(trace=True, ttnn_module=types.SimpleNamespace(), torch_module=types.SimpleNamespace())
        self.assertEqual(report["status"], "no_device")
        self.assertEqual(report["error"], NO_TTNN_DEVICE_MESSAGE)
        self.assertEqual(report["latency_ms"], {"mean": 0.0, "p50": 0.0, "p90": 0.0})
        self.assertEqual(json.loads(self.out.read_text()), report)

    def test_cli_profile_template_dry_run(self) -> None:
        exit_code = main([
            "diagnose", "--stage", "template-profile", "--template", "mlp_decode", "--config", str(self.config),
            "--warmup", "1", "--iterations", "3",
            "--trace", "--dry-run", "--out", str(self.out),
        ])
        report = json.loads(self.out.read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["template"], "mlp_decode")
        self.assertEqual(report["trace"], {"requested": True, "status": "dry_run"})

    def test_eager_reuses_one_device_and_persistent_tensors(self) -> None:
        fake = _FakeTTNN(trace_api=False)
        report = self.hardware_profile(fake, trace=False)
        self.assertEqual(fake.calls, Counter(enter=1, prepare=1, execute=3, synchronize=3, pcc=1, exit=1))
        self.assertEqual(report["trace"], {"requested": False, "status": "disabled"})
        self.assertEqual(report["device_lifecycle"]["open_count"], 1)
        self.assertTrue(report["tensor_lifecycle"]["persistent_tensor_reuse"])

    def test_trace_captures_replays_and_releases_once(self) -> None:
        fake = _FakeTTNN()
        report = self.hardware_profile(fake, trace=True)
        self.assertEqual((fake.calls["enter"], fake.calls["exit"]), (1, 1))
        self.assertEqual(fake.calls["execute"], 2)  # compile plus capture
        self.assertEqual(fake.calls["begin"], fake.calls["end"], fake.calls["release"])
        self.assertEqual(fake.calls["begin"], 1)
        self.assertEqual(fake.calls["execute_trace"], 3)
        self.assertEqual(report["trace"], {"requested": True, "status": "captured", "capture_count": 1,
            "warmup_execute_count": 1, "measured_execute_count": 2, "release_count": 1,
        })
        self.assertEqual(fake.calls["pcc"], 1)
        self.assertLess(fake.events.index("pcc"), fake.events.index("release"))

    def test_trace_fallbacks_are_explicit_and_eager(self) -> None:
        cases = ((False, False, "trace_api_unavailable_fell_back_to_eager", 3),
                 (True, True, "trace_failed_fell_back_to_eager", 4))
        for trace_api, capture_error, status, executions in cases:
            with self.subTest(status=status):
                fake = _FakeTTNN(trace_api=trace_api, capture_error=capture_error)
                report = self.hardware_profile(fake, trace=True)
                self.assertEqual(report["trace"]["status"], status)
                self.assertEqual(fake.calls["execute"], executions)
                self.assertEqual((fake.calls["enter"], fake.calls["exit"]), (1, 1))
                self.assertEqual(fake.calls["pcc"], 1)


class _FakeTTNN:
    def __init__(self, *, trace_api=True, capture_error=False):
        self.calls, self.events = Counter(), []
        self.capture_error = capture_error
        if not trace_api:
            for name in template_profile.TRACE_APIS:
                setattr(self, name, None)

    @contextmanager
    def managed(self, _ttnn, _device_id):
        self.calls["enter"] += 1
        try:
            yield object()
        finally:
            self.calls["exit"] += 1

    def prepare(self, **_kwargs):
        self.calls["prepare"] += 1
        def execute():
            self.calls["execute"] += 1
            return object()
        def pcc(_output):
            self.calls["pcc"] += 1
            self.events.append("pcc")
            return 1.0
        return execute, pcc

    def synchronize_device(self, _device):
        self.calls["synchronize"] += 1

    def begin_trace_capture(self, _device, **_kwargs):
        self.calls["begin"] += 1
        if self.capture_error:
            raise RuntimeError("capture failed")
        return 7

    def end_trace_capture(self, _device, _trace_id, **_kwargs):
        self.calls["end"] += 1

    def execute_trace(self, _device, _trace_id, **_kwargs):
        self.calls["execute_trace"] += 1

    def release_trace(self, _device, _trace_id):
        self.calls["release"] += 1
        self.events.append("release")


def _write_config(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": 1, "model_name": "fake-llama-profile",
        "template_config": {"device": "p150a"}, "batch_size": 32,
        "hidden_size": 16, "intermediate_size": 32,
    }))


if __name__ == "__main__":
    unittest.main()
