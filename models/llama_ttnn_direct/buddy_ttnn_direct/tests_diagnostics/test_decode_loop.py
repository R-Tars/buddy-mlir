from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.legacy_decode_loop import (
    run_prompt_decode_loop,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import (
    _fake_tokenizer_module,
    _fake_torch,
    _make_generate_fake_ttnn,
    _write_fake_model_config,
    _write_template_config,
)


def _build_program(root: Path, *, weights: bool = False) -> tuple[Path, Path]:
    model, config, program = root / "fake_model", root / "config.json", root / "program"
    _write_fake_model_config(model)
    if weights:
        _write_fake_model_weights(model, _fake_weight_specs())
    _write_template_config(config)
    result = main(
        [
            "build", "--model-path", str(model), "--config", str(config),
            "--out-dir", str(program),
        ]
    )
    if result != 0:
        raise AssertionError(f"test program build failed with exit code {result}")
    return model, program


class PromptDecodeLoopTest(unittest.TestCase):
    def assertFields(self, report: dict[str, object], **expected: object) -> None:
        self.assertEqual({key: report[key] for key in expected}, expected)

    def test_cli_prompt_decode_loop_dry_run_uses_generate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, program = _build_program(root)
            output = root / "prompt_decode_loop_report.json"
            exit_code = main(
                [
                    "diagnose", "--stage", "decode-loop-legacy",
                    "--program-dir", str(program), "--max-new-tokens", "3",
                    "--layers", "1", "--batch-size", "2", "--cache-len", "16",
                    "--dry-run", "--out", str(output),
                ]
            )
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertFields(
            report,
            template="prompt_decode_loop",
            status="dry_run",
            passed=True,
            decode_steps=3,
            legacy_requested_decode_steps=3,
            max_new_tokens=4,
            prefill_status="dry_run",
            kv_cache_source="prefill",
            model_semantics="prompt_conditioned_prefill_decode",
            planned_decode_loop_runtime_owned=True,
            decode_loop_runtime_owned=False,
            runtime_owner="TTNNDirectRuntimeContext",
            generated_token_ids=[],
            step_reports=[],
            throughput_summary={},
        )
        self.assertEqual(
            report["legacy_adapter"]["runtime_api"], "runtime.generate.run_generate"
        )

    def test_prompt_decode_loop_reuses_prompt_conditioned_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model, program = _build_program(root, weights=True)
            output = root / "prompt_decode_loop_report.json"
            with _fake_torch_and_safetensors():
                report = run_prompt_decode_loop(
                    out=output,
                    program_dir=program,
                    model_path=model,
                    prompt="hello tenstorrent",
                    tokenizer_path=model,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    decode_steps=2,
                    layers=1,
                    device="p150a",
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=_make_generate_fake_ttnn(),
                    torch_module=_fake_torch(),
                )
            persisted = json.loads(output.read_text())

        self.assertFields(
            report,
            passed=True,
            status="passed",
            decode_loop_runtime_owned=True,
            runtime_owner="TTNNDirectRuntimeContext",
            runtime_session_owner="runtime.generate.run_generate",
            decode_steps=2,
            max_new_tokens=3,
            prefill_status="passed",
            kv_cache_source="prefill",
            model_semantics="prompt_conditioned_prefill_decode",
        )
        self.assertEqual(len(report["step_reports"]), 2)
        self.assertEqual(report["generated_token_ids"], [[23, 23, 23], [23, 23, 23]])
        self.assertEqual(report["throughput_summary"]["generated_tokens_per_user"], 3)
        self.assertEqual(report["reference"]["status"], "passed")
        self.assertEqual(persisted, report)


if __name__ == "__main__":
    unittest.main()
