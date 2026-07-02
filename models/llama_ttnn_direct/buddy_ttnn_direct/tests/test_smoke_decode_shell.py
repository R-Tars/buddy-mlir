from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_decode_shell import (
    DECODE_SHELL_OPS,
    run_smoke_decode_shell,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)


class SmokeDecodeShellTest(unittest.TestCase):
    def test_cli_smoke_decode_shell_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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
                    "smoke-decode-shell",
                    "--program-dir",
                    str(program_dir),
                    "--layers",
                    "1",
                    "--disable-attention",
                    "--device",
                    "p150a",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["ttnn_ops"], DECODE_SHELL_OPS)
            self.assertEqual(report["input_shapes"]["token_ids"], [32, 1])
            self.assertEqual(report["input_source"], "planned")
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_run_smoke_decode_shell_executes_fake_attention_disabled_shell(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_decode_shell(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake_ttnn,
                parameters=_fake_parameters(split_count=8),
                token_ids="token_ids",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["layers"][0]["attention"], "disabled")
            self.assertEqual(report["input_source"], "injected")
            self.assertEqual(report["runtime_input_tensor_count"], 0)
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(report["reference"]["kind"], "structural_shape_dtype")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )
            self.assertEqual(report["reference"]["observed_ops"][0], "embedding")
            self.assertTrue(
                all(check["passed"] for check in report["reference"]["checks"])
            )
            self.assertEqual(json.loads(report_json.read_text()), report)
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "embedding",
                    "rms_norm",
                    "linear",
                    "linear",
                    "mul",
                    "linear",
                    "add",
                    "rms_norm",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "concat",
                    "argmax",
                ],
            )
            self.assertNotIn(
                "paged_scaled_dot_product_attention_decode",
                [call["op"] for call in fake_ttnn.calls],
            )

    def test_run_smoke_decode_shell_synthesizes_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_decode_shell(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
                parameters=_fake_parameters(split_count=8),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["input_source"], "synthetic")
            self.assertEqual(report["input_shapes"]["token_ids"], [32, 1])
            self.assertEqual(report["runtime_input_tensor_count"], 1)
            self.assertEqual(fake_ttnn.calls[0]["op"], "from_torch")
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"]["layout"],
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(fake_ttnn.calls[1]["op"], "embedding")
            self.assertIsInstance(fake_ttnn.calls[1]["token_ids"], FakeTensor)
            self.assertEqual(json.loads(report_json.read_text()), report)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: list[int] | None = None,
        dtype: str = "bf16",
        mem_config: str | None = None,
    ) -> None:
        self.name = name
        self.shape = shape or [32, 1, 16]
        self.dtype = dtype
        self._mem_config = mem_config

    def memory_config(self) -> str | None:
        return self._mem_config

    def __repr__(self) -> str:
        return f"FakeTensor({self.name})"


def _fake_parameters(split_count: int):
    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(weight="embed_weight"),
        layers=[
            types.SimpleNamespace(
                input_norm=types.SimpleNamespace(weight="input_norm_weight"),
                post_attention_norm=types.SimpleNamespace(
                    weight="post_attention_norm_weight"
                ),
                mlp=types.SimpleNamespace(
                    gate_proj=types.SimpleNamespace(weight="gate_weight"),
                    up_proj=types.SimpleNamespace(weight="up_weight"),
                    down_proj=types.SimpleNamespace(weight="down_weight"),
                ),
            )
        ],
        final_norm=types.SimpleNamespace(weight="final_norm_weight"),
        lm_head=types.SimpleNamespace(
            splits=[
                types.SimpleNamespace(weight=f"lm_head_{index}")
                for index in range(split_count)
            ]
        ),
    )


def _make_fake_ttnn():
    module = types.ModuleType("ttnn")
    module.calls = []
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"

    class UnaryOpType:
        SILU = "SILU"

    def unary_with_param(op):
        return ("UnaryWithParam", op)

    def embedding(token_ids, weight, **kwargs):
        module.calls.append(
            {
                "op": "embedding",
                "token_ids": token_ids,
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(f"embedding:{weight}", mem_config=kwargs.get("memory_config"))

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
            dtype=str(getattr(tensor, "dtype", None)),
        )

    def rms_norm(hidden, **kwargs):
        module.calls.append(
            {
                "op": "rms_norm",
                "hidden": getattr(hidden, "name", hidden),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            f"rms_norm:{kwargs.get('weight')}",
            mem_config=kwargs.get("memory_config"),
        )

    def linear(activation, weight, **kwargs):
        module.calls.append(
            {
                "op": "linear",
                "activation": getattr(activation, "name", activation),
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(f"linear:{weight}", mem_config=kwargs.get("memory_config"))

    def mul(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "mul",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("mul_out", mem_config=kwargs.get("memory_config"))

    def add(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "add",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("add_out", mem_config=kwargs.get("memory_config"))

    def concat(tensors, **kwargs):
        names = [getattr(tensor, "name", tensor) for tensor in tensors]
        module.calls.append(
            {
                "op": "concat",
                "tensors": names,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("concat:" + ",".join(names))

    def argmax(tensor, **kwargs):
        module.calls.append(
            {
                "op": "argmax",
                "tensor": getattr(tensor, "name", tensor),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(f"argmax:{getattr(tensor, 'name', tensor)}", [32])

    module.UnaryOpType = UnaryOpType
    module.UnaryWithParam = unary_with_param
    module.from_torch = from_torch
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.linear = linear
    module.mul = mul
    module.add = add
    module.concat = concat
    module.argmax = argmax
    return module


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-decode-shell",
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "tie_word_embeddings": False,
            }
        )
    )


def _write_template_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "device": "p150a",
                "model": "llama3.1-8b",
                "batch_size": 32,
                "decode_seq_len": 1,
                "prefill_seq_len": 128,
                "max_cache_len": 1024,
                "attention_template": "official_paged_attention_decode",
                "mlp_template": "official_gated_mlp_decode",
                "lm_head_template": "official_split_lm_head",
                "kv_cache_template": "paged_kv_cache",
                "generation_template": "device_argmax_greedy",
                "lm_head_split_count": 8,
                "dtype_recipe": "official_like_performance_seed",
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
