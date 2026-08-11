from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.legacy_decode_loop import (
    run_prompt_decode_loop,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import (
    _make_generate_fake_ttnn,
    _fake_torch,
    _fake_tokenizer_module,
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)


class PromptDecodeLoopTest(unittest.TestCase):
    def test_cli_prompt_decode_loop_dry_run_uses_generate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prompt_decode_loop_report.json"
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

            exit_code = main(
                [
                    "diagnose",
                    "--stage",
                    "decode-loop-legacy",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "3",
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
            self.assertEqual(report["template"], "prompt_decode_loop")
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["decode_steps"], 3)
            self.assertEqual(report["legacy_requested_decode_steps"], 3)
            self.assertEqual(report["max_new_tokens"], 4)
            self.assertEqual(report["prefill_status"], "dry_run")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(
                report["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertTrue(report["planned_decode_loop_runtime_owned"])
            self.assertFalse(report["decode_loop_runtime_owned"])
            self.assertEqual(report["runtime_owner"], "TTNNDirectRuntimeContext")
            self.assertEqual(report["generated_token_ids"], [])
            self.assertEqual(report["step_reports"], [])
            self.assertEqual(report["throughput_summary"], {})
            self.assertEqual(
                report["legacy_adapter"]["runtime_api"],
                "runtime.generate.run_generate",
            )

    def test_prompt_decode_loop_reuses_prompt_conditioned_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prompt_decode_loop_report.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
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

            fake_ttnn = _make_generate_fake_ttnn()
            with _fake_torch_and_safetensors():
                report = run_prompt_decode_loop(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    decode_steps=2,
                    layers=1,
                    device="p150a",
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=fake_ttnn,
                    torch_module=_fake_torch(),
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["decode_loop_runtime_owned"])
            self.assertEqual(report["runtime_owner"], "TTNNDirectRuntimeContext")
            self.assertEqual(report["runtime_session_owner"], "runtime.generate.run_generate")
            self.assertEqual(report["decode_steps"], 2)
            self.assertEqual(report["max_new_tokens"], 3)
            self.assertEqual(report["prefill_status"], "passed")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(
                report["model_semantics"],
                "prompt_conditioned_prefill_decode",
            )
            self.assertEqual(len(report["step_reports"]), 2)
            self.assertEqual(
                report["generated_token_ids"],
                [[23, 23, 23], [23, 23, 23]],
            )
            self.assertEqual(
                report["throughput_summary"]["generated_tokens_per_user"],
                3,
            )
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(json.loads(report_json.read_text()), report)


if __name__ == "__main__":
    unittest.main()
