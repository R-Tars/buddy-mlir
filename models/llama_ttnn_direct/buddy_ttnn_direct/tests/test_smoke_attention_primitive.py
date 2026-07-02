from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
    run_smoke_attention_primitive,
)


class SmokeAttentionPrimitiveTest(unittest.TestCase):
    def test_cli_smoke_attention_primitive_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"

            exit_code = main(
                [
                    "smoke-attention-primitive",
                    "--primitive",
                    "paged_scaled_dot_product_attention_decode",
                    "--device",
                    "p150a",
                    "--batch-size",
                    "32",
                    "--hidden-size",
                    "4096",
                    "--num-heads",
                    "32",
                    "--num-kv-heads",
                    "8",
                    "--head-dim",
                    "128",
                    "--dry-run",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(out.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(
                report["primitive"],
                "paged_scaled_dot_product_attention_decode",
            )
            self.assertEqual(
                report["input_shapes"]["query"],
                [32, 32, 1, 128],
            )
            self.assertEqual(report["layout"], "tile")
            self.assertEqual(report["dtype"], "bfloat16")
            self.assertEqual(report["memory_config"], "default_or_l1")
            self.assertEqual(report["tensor_conversion_count"], 5)
            self.assertEqual(
                report["ttnn_environment"]["module_available"],
                False,
            )
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_each_attention_primitive_runs_with_fake_ttnn(self) -> None:
        all_called_ops = []
        for primitive in ATTENTION_PRIMITIVES:
            with self.subTest(primitive=primitive):
                with tempfile.TemporaryDirectory() as tmpdir:
                    out = Path(tmpdir) / f"{primitive}.json"
                    fake_ttnn = _fake_ttnn()
                    report = run_smoke_attention_primitive(
                        out=out,
                        primitive=primitive,
                        device="p150a",
                        batch_size=2,
                        hidden_size=16,
                        num_heads=4,
                        num_kv_heads=2,
                        head_dim=4,
                        max_cache_len=16,
                        ttnn_module=fake_ttnn,
                        torch_module=_fake_torch(),
                    )

                    self.assertTrue(report["passed"])
                    self.assertEqual(report["status"], "passed")
                    self.assertIsNotNone(report["output_shapes"])
                    self.assertEqual(report["reference"]["status"], "passed")
                    self.assertEqual(report["reference"]["kind"], "structural_shape")
                    self.assertTrue(
                        all(
                            check["passed"]
                            for check in report["reference"]["checks"]
                        )
                    )
                    self.assertEqual(
                        set(report["output_shapes"]),
                        set(report["expected_output_shapes"]),
                    )
                    self.assertEqual(
                        report["tensor_conversion_count"],
                        len(report["input_shapes"]),
                    )
                    self.assertEqual(report["dtype"], "bfloat16")
                    self.assertEqual(report["layout"], "tile")
                    self.assertEqual(report["memory_config"], "default_or_l1")
                    self.assertEqual(
                        report["ttnn_environment"]["version"],
                        "fake-ttnn",
                    )
                    self.assertEqual(
                        report["ttnn_environment"]["tt_metal_git_commit"],
                        "fake-tt-metal",
                    )
                    self.assertEqual(json.loads(out.read_text()), report)
                    all_called_ops.extend(
                        call["op"] for call in fake_ttnn.calls
                    )

        self.assertIn("nlp_create_qkv_heads_decode", all_called_ops)
        self.assertIn(
            "paged_scaled_dot_product_attention_decode",
            all_called_ops,
        )

    def test_api_mismatch_is_reported_without_silent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"
            fake_ttnn = _fake_ttnn(with_transformer=False)

            report = run_smoke_attention_primitive(
                out=out,
                primitive="paged_scaled_dot_product_attention_decode",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                max_cache_len=16,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "api_mismatch")
            self.assertIn(
                "paged_scaled_dot_product_attention_decode",
                report["error"],
            )
            self.assertEqual(json.loads(out.read_text()), report)


class FakeTorchTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str | None = None,
        name: str = "torch_tensor",
    ) -> None:
        self.shape = list(shape)
        self.dtype = dtype
        self.name = name


class FakeTTNNTensor:
    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str | None = "ttnn.bfloat16",
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype


def _fake_torch():
    module = types.SimpleNamespace()
    module.bfloat16 = "torch.bfloat16"
    module.float32 = "torch.float32"
    module.int32 = "torch.int32"

    def randn(shape, dtype=None):
        return FakeTorchTensor(tuple(shape), dtype=dtype, name="randn")

    def zeros(shape, dtype=None):
        return FakeTorchTensor(tuple(shape), dtype=dtype, name="zeros")

    module.randn = randn
    module.zeros = zeros
    return module


def _fake_ttnn(*, with_transformer: bool = True):
    module = types.SimpleNamespace(calls=[])
    module.__version__ = "fake-ttnn"
    module.__tt_metal_commit__ = "fake-tt-metal"
    module.bfloat16 = "ttnn.bfloat16"
    module.float32 = "ttnn.float32"
    module.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    module.L1_MEMORY_CONFIG = "ttnn.L1_MEMORY_CONFIG"

    def from_torch(tensor, **kwargs):
        module.calls.append(
            {
                "op": "from_torch",
                "shape": list(tensor.shape),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor(
            tensor.name,
            list(tensor.shape),
            dtype=str(kwargs.get("dtype")),
        )

    def linear(activation, weight, **kwargs):
        out_shape = list(activation.shape)
        out_shape[-1] = weight.shape[-1]
        module.calls.append(
            {
                "op": "linear",
                "activation": activation.name,
                "weight": weight.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor("linear", out_shape)

    def nlp_create_qkv_heads_decode(fused_qkv, **kwargs):
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        batch = fused_qkv.shape[0]
        module.calls.append(
            {
                "op": "nlp_create_qkv_heads_decode",
                "qkv": fused_qkv.name,
                "kwargs": dict(kwargs),
            }
        )
        return (
            FakeTTNNTensor("query", [batch, num_heads, 1, head_dim]),
            FakeTTNNTensor("key", [batch, num_kv_heads, 1, head_dim]),
            FakeTTNNTensor("value", [batch, num_kv_heads, 1, head_dim]),
        )

    def rotary_embedding_llama(tensor, cos, sin, transform, **kwargs):
        module.calls.append(
            {
                "op": "rotary_embedding_llama",
                "tensor": tensor.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor(f"rotary:{tensor.name}", tensor.shape)

    def paged_update_cache(cache, update, **kwargs):
        module.calls.append(
            {
                "op": "paged_update_cache",
                "cache": cache.name,
                "update": update.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor("cache", cache.shape)

    def paged_scaled_dot_product_attention_decode(q, k_cache, v_cache, **kwargs):
        module.calls.append(
            {
                "op": "paged_scaled_dot_product_attention_decode",
                "query": q.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor("attention", q.shape)

    def to_memory_config(tensor, **kwargs):
        module.calls.append(
            {
                "op": "to_memory_config",
                "tensor": tensor.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor(f"mem:{tensor.name}", tensor.shape)

    def nlp_concat_heads_decode(attention, **kwargs):
        num_heads = int(kwargs["num_heads"])
        batch = attention.shape[0]
        head_dim = attention.shape[-1]
        module.calls.append(
            {
                "op": "nlp_concat_heads_decode",
                "attention": attention.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor("concat_heads", [batch, 1, num_heads * head_dim])

    module.from_torch = from_torch
    module.linear = linear
    module.to_memory_config = to_memory_config
    module.experimental = types.SimpleNamespace(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        rotary_embedding_llama=rotary_embedding_llama,
        paged_update_cache=paged_update_cache,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
    if with_transformer:
        module.transformer = types.SimpleNamespace(
            paged_scaled_dot_product_attention_decode=(
                paged_scaled_dot_product_attention_decode
            )
        )
    return module


if __name__ == "__main__":
    unittest.main()
