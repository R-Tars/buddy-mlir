from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.generate import run_generate
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_prefill import (
    FakeTensor,
    _make_fake_ttnn as _make_prefill_fake_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_single_layer_decode import (
    _fake_tokenizer_module,
)


class GenerateTest(unittest.TestCase):
    def test_cli_generate_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_report.json"
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
                    "generate",
                    "--program-dir",
                    str(program_dir),
                    "--max-new-tokens",
                    "3",
                    "--prefill-len",
                    "8",
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
            self.assertEqual(report["mode"], "generate")
            self.assertEqual(report["template"], "prefill_then_decode_generate")
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["prefill_status"], "dry_run")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(report["max_new_tokens"], 3)
            self.assertEqual(report["decode_steps"], 2)
            self.assertTrue(report["planned_decode_loop_runtime_owned"])
            self.assertFalse(report["decode_loop_runtime_owned"])
            self.assertEqual(report["generated_token_ids"], [])
            self.assertEqual(report["generated_text"], "")

    def test_generate_runs_prefill_then_decode_with_prefilled_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "generate_report.json"
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

            fake_ttnn = _make_generate_fake_ttnn()
            with _fake_torch_and_safetensors():
                report = run_generate(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    prompt="hello tenstorrent",
                    tokenizer_path=model_dir,
                    tokenizer_module=_fake_tokenizer_module([7, 11, 42]),
                    max_new_tokens=3,
                    layers=1,
                    prefill_len=8,
                    device="p150a",
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=fake_ttnn,
                    torch_module=_fake_torch(),
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["mode"], "generate")
            self.assertEqual(report["prefill_status"], "passed")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertTrue(report["generate_runtime_owned"])
            self.assertTrue(report["decode_loop_runtime_owned"])
            self.assertEqual(report["decode_steps"], 2)
            self.assertEqual(report["prefill"]["status"], "passed")
            self.assertEqual(report["prefill"]["cache_population"][0]["status"], "filled")
            self.assertEqual(
                report["prefill"]["first_token"]["token_ids_by_user"],
                [[17], [17]],
            )
            self.assertEqual(report["generated_token_ids"], [[17, 23, 23], [17, 23, 23]])
            self.assertEqual(report["generated_text_status"], "fallback")
            self.assertEqual(report["generated_text"], "<tok:17> <tok:23> <tok:23>")
            self.assertEqual(
                report["prompt_tokenization"]["effective_token_count"],
                3,
            )
            self.assertEqual(report["prompt_tokenization"]["prefill_len"], 8)
            self.assertEqual(
                report["step_reports"][0]["cache_position_value"],
                3,
            )
            self.assertEqual(
                report["step_reports"][1]["cache_position_value"],
                4,
            )
            self.assertEqual(
                report["parameter_setup"]["synthetic_runtime_input_tensor_count"],
                0,
            )
            self.assertEqual(
                report["parameter_setup"]["synthetic_rotary_tensor_count"],
                0,
            )
            self.assertEqual(
                report["parameter_setup"]["synthetic_kv_cache_tensor_count"],
                0,
            )
            ops = [call["op"] for call in fake_ttnn.calls]
            self.assertIn("scaled_dot_product_attention", ops)
            self.assertIn("fill_cache", ops)
            self.assertIn("paged_scaled_dot_product_attention_decode", ops)
            self.assertEqual(json.loads(report_json.read_text()), report)


def _make_generate_fake_ttnn():
    module = _make_prefill_fake_ttnn()
    module.DRAM_MEMORY_CONFIG = "ttnn.DRAM_MEMORY_CONFIG"

    def from_torch(tensor, **kwargs):
        module.calls.append(
            {
                "op": "from_torch",
                "shape": list(tensor.shape),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            getattr(tensor, "name", "torch_tensor"),
            list(tensor.shape),
            dtype=str(kwargs.get("dtype", "ttnn.bfloat16")),
        )

    def to_torch(tensor):
        module.calls.append({"op": "to_torch", "tensor": tensor.name})
        shape = list(tensor.shape)
        if len(shape) == 2 and shape[1] > 1:
            return [[17 for _ in range(shape[1])] for _ in range(shape[0])]
        if len(shape) == 2:
            return [[23] for _ in range(shape[0])]
        if len(shape) == 1:
            return [23 for _ in range(shape[0])]
        return [[23], [23]]

    def nlp_create_qkv_heads_decode(fused_qkv, **kwargs):
        batch = fused_qkv.shape[0]
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        module.calls.append(
            {"op": "nlp_create_qkv_heads_decode", "kwargs": dict(kwargs)}
        )
        return (
            FakeTensor("query_decode", [batch, num_heads, 1, head_dim]),
            FakeTensor("key_decode", [batch, num_kv_heads, 1, head_dim]),
            FakeTensor("value_decode", [batch, num_kv_heads, 1, head_dim]),
        )

    def paged_update_cache(cache, update, **kwargs):
        module.calls.append({"op": "paged_update_cache", "kwargs": dict(kwargs)})
        return FakeTensor(cache.name, cache.shape)

    def paged_scaled_dot_product_attention_decode(q, k_cache, v_cache, **kwargs):
        module.calls.append(
            {
                "op": "paged_scaled_dot_product_attention_decode",
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("decode_attention", q.shape)

    def nlp_concat_heads_decode(attention, **kwargs):
        batch, _, _, head_dim = attention.shape
        num_heads = int(kwargs["num_heads"])
        module.calls.append({"op": "nlp_concat_heads_decode", "kwargs": dict(kwargs)})
        return FakeTensor("decode_concat_heads", [batch, 1, num_heads * head_dim])

    def to_memory_config(tensor, **kwargs):
        module.calls.append({"op": "to_memory_config", "kwargs": dict(kwargs)})
        return FakeTensor(f"mem:{tensor.name}", tensor.shape)

    module.from_torch = from_torch
    module.to_torch = to_torch
    module.to_memory_config = to_memory_config
    module.experimental = types.SimpleNamespace(
        rotary_embedding_llama=module.experimental.rotary_embedding_llama,
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        paged_update_cache=paged_update_cache,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
    module.transformer.paged_scaled_dot_product_attention_decode = (
        paged_scaled_dot_product_attention_decode
    )
    return module


if __name__ == "__main__":
    unittest.main()
