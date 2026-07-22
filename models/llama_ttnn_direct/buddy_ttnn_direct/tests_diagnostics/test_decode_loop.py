from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.decode_loop import (
    _normalize_token_ids,
    run_prompt_decode_loop,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters_tensorizer import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_single_layer_decode import (
    _fake_tokenizer_module,
    _make_fake_ttnn,
)


class PromptDecodeLoopTest(unittest.TestCase):
    def test_normalize_token_ids_accepts_physical_decode_shape(self) -> None:
        self.assertEqual(
            _normalize_token_ids([[[[13], [17]]]], batch_size=2),
            [[13], [17]],
        )

    def test_cli_prompt_decode_loop_dry_run(self) -> None:
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
            self.assertEqual(report["decode_steps"], 3)
            self.assertEqual(report["max_new_tokens"], 3)
            self.assertEqual(report["prefill_status"], "not_run")
            self.assertEqual(report["kv_cache_source"], "empty_initialized")
            self.assertEqual(
                report["model_semantics"],
                "decode_only_empty_or_uninitialized_kv",
            )
            self.assertTrue(report["planned_decode_loop_runtime_owned"])
            self.assertFalse(report["decode_loop_runtime_owned"])
            self.assertEqual(report["generated_token_ids"], [])
            self.assertEqual(report["generated_text"], "")
            self.assertEqual(report["generated_text_status"], "not_run")
            self.assertEqual(report["per_step_token_metadata"], [])
            self.assertEqual(
                report["decode_runtime_state_input_tensor_count"],
                6,
            )
            self.assertEqual(report["rotary_runtime_input_tensor_count"], 9)
            self.assertEqual(report["kv_cache_runtime_input_tensor_count"], 2)

    def test_prompt_decode_loop_executes_two_runtime_owned_steps(self) -> None:
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

            fake_ttnn = _make_fake_ttnn()
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
            self.assertEqual(report["runtime_owner"], "prompt_decode_loop")
            self.assertEqual(report["input_source"], "prompt_decode_loop")
            self.assertEqual(report["parameter_source"], "hf_model")
            self.assertEqual(report["decode_steps"], 2)
            self.assertEqual(report["max_new_tokens"], 2)
            self.assertEqual(report["prefill_status"], "not_run")
            self.assertEqual(report["kv_cache_source"], "empty_initialized")
            self.assertEqual(
                report["model_semantics"],
                "decode_only_empty_or_uninitialized_kv",
            )
            self.assertIn("does not run prefill", report["semantic_disclaimer"])
            self.assertEqual(len(report["step_reports"]), 2)
            self.assertEqual(
                report["prompt_tokenization"]["selected_token_id"],
                42,
            )
            self.assertEqual(
                report["step_reports"][0]["cache_position_value"],
                2,
            )
            self.assertEqual(
                report["step_reports"][1]["cache_position_value"],
                3,
            )
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(report["generated_token_ids"], [[-1, -1], [-1, -1]])
            self.assertEqual(
                report["generated_token_id_source"],
                "placeholder_unmaterialized",
            )
            self.assertEqual(report["token_materialization_status"], "unavailable")
            self.assertEqual(report["generated_text_status"], "placeholder")
            self.assertEqual(
                report["generated_text_source"],
                "unmaterialized_token_placeholder",
            )
            self.assertEqual(
                report["generated_text"],
                "<unmaterialized-token> <unmaterialized-token>",
            )
            self.assertEqual(
                report["generated_text_by_user"],
                [
                    "<unmaterialized-token> <unmaterialized-token>",
                    "<unmaterialized-token> <unmaterialized-token>",
                ],
            )
            self.assertEqual(len(report["per_step_token_metadata"]), 2)
            self.assertEqual(
                report["per_step_token_metadata"][0]["cache_position_value"],
                2,
            )
            self.assertEqual(
                report["per_step_token_metadata"][1]["cache_position_value"],
                3,
            )
            self.assertEqual(
                report["per_step_token_metadata"][0]["token_ids_by_user"],
                [[-1], [-1]],
            )
            self.assertEqual(
                report["step_reports"][0]["generated_token_ids"],
                [[-1], [-1]],
            )
            self.assertEqual(
                report["step_reports"][0]["token_materialization"]["status"],
                "unavailable",
            )
            self.assertEqual(report["prompt_runtime_input_tensor_count"], 1)
            self.assertEqual(
                report["decode_runtime_state_input_tensor_count"],
                4,
            )
            self.assertEqual(report["rotary_runtime_input_tensor_count"], 6)
            self.assertEqual(report["kv_cache_runtime_input_tensor_count"], 2)
            self.assertEqual(
                report["synthetic_runtime_input_tensor_count"],
                0,
            )
            self.assertEqual(report["synthetic_rotary_tensor_count"], 0)
            self.assertEqual(report["tensor_conversion_count"], 30)
            self.assertEqual(
                report["throughput_summary"]["generated_tokens_per_user"],
                2,
            )
            self.assertEqual(
                report["reference"]["status"],
                "passed",
            )
            ops = [call["op"] for call in fake_ttnn.calls]
            self.assertEqual(ops.count("untilize"), 2)
            self.assertEqual(ops.count("argmax"), 2)
            self.assertEqual(ops.count("topk"), 0)
            self.assertEqual(ops.count("gather"), 0)
            self.assertEqual(json.loads(report_json.read_text()), report)


if __name__ == "__main__":
    unittest.main()
