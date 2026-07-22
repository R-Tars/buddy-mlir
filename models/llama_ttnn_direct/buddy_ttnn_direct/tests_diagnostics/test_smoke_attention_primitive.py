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
    PRIMITIVE_EXPECTED_OBSERVED_OPS,
    run_smoke_attention_primitive,
)


class SmokeAttentionPrimitiveTest(unittest.TestCase):
    def test_cli_smoke_attention_primitive_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"

            exit_code = main(
                [
                    "diagnose",
                    "--stage",
                    "attention-primitive",
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
                [1, 32, 32, 128],
            )
            self.assertEqual(
                report["input_shapes"]["key_cache"],
                [1024, 8, 32, 128],
            )
            self.assertEqual(
                report["input_shapes"]["page_table"],
                [32, 32],
            )
            self.assertEqual(report["layout"], "tile")
            self.assertEqual(report["dtype"], "bfloat16")
            self.assertEqual(report["memory_config"], "default_or_l1")
            self.assertEqual(
                report["input_tensor_contracts"]["cache_position"],
                {
                    "dtype": "int32",
                    "layout": "row_major",
                    "memory_config": "default_or_dram",
                },
            )
            self.assertEqual(
                report["input_tensor_contracts"]["query"][
                    "memory_config"
                ],
                "height_sharded_l1",
            )
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
                    self.assertEqual(
                        report["reference"]["kind"],
                        "structural_shape_op_sequence",
                    )
                    self.assertEqual(
                        report["reference"]["planned_ops"],
                        PRIMITIVE_EXPECTED_OBSERVED_OPS[primitive],
                    )
                    self.assertEqual(
                        report["reference"]["observed_ops_source"],
                        "ttnn_module_instrumentation",
                    )
                    self.assertIn(
                        "observed_op_sequence",
                        [
                            check["name"]
                            for check in report["reference"]["checks"]
                        ],
                    )
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

    def test_successful_primitive_without_op_instrumentation_uses_direct_call_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"
            report = run_smoke_attention_primitive(
                out=out,
                primitive="qkv_linear",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                max_cache_len=16,
                ttnn_module=_fake_ttnn(instrumented=False),
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(
                report["reference"]["observed_ops"],
                PRIMITIVE_EXPECTED_OBSERVED_OPS["qkv_linear"],
            )
            self.assertEqual(
                report["reference"]["observed_ops_source"],
                "direct_primitive_call",
            )
            self.assertEqual(json.loads(out.read_text()), report)

    def test_runtime_index_tensors_use_int32_row_major(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"
            fake_ttnn = _fake_ttnn()

            report = run_smoke_attention_primitive(
                out=out,
                primitive="paged_update_cache",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                max_cache_len=64,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            by_name = {
                call["name"]: call
                for call in fake_ttnn.calls
                if call["op"] == "from_torch"
            }
            self.assertEqual(
                by_name["page_table"]["kwargs"]["dtype"],
                "ttnn.int32",
            )
            self.assertEqual(
                by_name["page_table"]["kwargs"]["layout"],
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(
                by_name["page_table"]["values"],
                [[0, 1], [2, 3]],
            )
            self.assertEqual(
                by_name["cache_position"]["kwargs"]["dtype"],
                "ttnn.int32",
            )
            self.assertEqual(
                by_name["cache_position"]["values"],
                [0, 0],
            )
            self.assertEqual(
                by_name["update"]["kwargs"]["memory_config"],
                "ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG",
            )

    def test_decode_sharded_primitives_request_height_sharded_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"
            fake_ttnn = _fake_ttnn(with_create_sharded=True)

            report = run_smoke_attention_primitive(
                out=out,
                primitive="rotary_embedding_decode",
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
            by_name = {
                call["name"]: call
                for call in fake_ttnn.calls
                if call["op"] == "from_torch"
            }
            self.assertEqual(
                by_name["query"]["kwargs"]["memory_config"],
                "sharded:(32, 4)",
            )
            self.assertEqual(
                by_name["key"]["kwargs"]["memory_config"],
                "sharded:(32, 4)",
            )
            self.assertEqual(
                by_name["cos_matrix"]["kwargs"]["memory_config"],
                "sharded:(32, 4)",
            )
            self.assertEqual(
                by_name["sin_matrix"]["kwargs"]["memory_config"],
                "sharded:(32, 4)",
            )
            self.assertEqual(
                by_name["transformation_matrix"]["kwargs"]["memory_config"],
                "sharded:(32, 32)",
            )
            self.assertEqual(by_name["cos_matrix"]["shape"], [1, 2, 1, 4])
            self.assertEqual(by_name["sin_matrix"]["shape"], [1, 2, 1, 4])
            self.assertEqual(
                by_name["transformation_matrix"]["shape"],
                [1, 1, 64, 32],
            )

            concat_out = Path(tmpdir) / "concat_report.json"
            run_smoke_attention_primitive(
                out=concat_out,
                primitive="nlp_concat_heads_decode",
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
            to_memory_calls = [
                call
                for call in fake_ttnn.calls
                if call["op"] == "to_memory_config"
            ]
            self.assertTrue(to_memory_calls)
            self.assertEqual(
                to_memory_calls[-1]["kwargs"]["memory_config"],
                "sharded:(32, 4)",
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

    def test_observed_op_mismatch_fails_structural_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "primitive_report.json"
            fake_ttnn = _fake_ttnn()

            def wrong_sdpa(q, k_cache, v_cache, **kwargs):
                fake_ttnn.calls.append(
                    {
                        "op": "wrong_sdpa_decode",
                        "query": q.name,
                        "kwargs": dict(kwargs),
                    }
                )
                return FakeTTNNTensor("attention", q.shape)

            fake_ttnn.transformer.paged_scaled_dot_product_attention_decode = (
                wrong_sdpa
            )

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
            self.assertEqual(report["status"], "reference_mismatch")
            self.assertEqual(report["reference"]["status"], "failed")
            failed_checks = [
                check
                for check in report["reference"]["checks"]
                if not check["passed"]
            ]
            self.assertEqual(
                [check["name"] for check in failed_checks],
                ["observed_op_sequence"],
            )
            self.assertEqual(
                failed_checks[0]["missing_from_ordered_coverage"],
                ["paged_scaled_dot_product_attention_decode"],
            )
            self.assertEqual(json.loads(out.read_text()), report)


class FakeTorchTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str | None = None,
        name: str = "torch_tensor",
        values: Any | None = None,
    ) -> None:
        self.shape = list(shape)
        self.dtype = dtype
        self.name = name
        self.values = values


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

    def tensor(values, dtype=None):
        return FakeTorchTensor(
            tuple(_nested_shape(values)),
            dtype=dtype,
            name="tensor",
            values=values,
        )

    module.randn = randn
    module.zeros = zeros
    module.tensor = tensor
    return module


def _nested_shape(values: Any) -> list[int]:
    shape = []
    current = values
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


class _CallSink:
    def append(self, call: Any) -> None:
        pass


def _fake_ttnn(
    *,
    with_transformer: bool = True,
    instrumented: bool = True,
    with_create_sharded: bool = False,
):
    module = types.SimpleNamespace(calls=[] if instrumented else _CallSink())
    module.__version__ = "fake-ttnn"
    module.__tt_metal_commit__ = "fake-tt-metal"
    module.bfloat16 = "ttnn.bfloat16"
    module.float32 = "ttnn.float32"
    module.int32 = "ttnn.int32"
    module.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"
    module.L1_MEMORY_CONFIG = "ttnn.L1_MEMORY_CONFIG"
    module.L1_HEIGHT_SHARDED_MEMORY_CONFIG = (
        "ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG"
    )
    module.DRAM_MEMORY_CONFIG = "ttnn.DRAM_MEMORY_CONFIG"
    module.TILE_SIZE = 32

    if with_create_sharded:
        module.ShardStrategy = types.SimpleNamespace(HEIGHT="HEIGHT")
        module.ShardOrientation = types.SimpleNamespace(ROW_MAJOR="ROW_MAJOR")

        class CoreGrid:
            def __init__(self, y, x):
                self.y = y
                self.x = x

        module.CoreGrid = CoreGrid

        def create_sharded_memory_config(**kwargs):
            module.calls.append(
                {
                    "op": "create_sharded_memory_config",
                    "kwargs": dict(kwargs),
                }
            )
            return f"sharded:{tuple(kwargs['shape'])}"

        module.create_sharded_memory_config = create_sharded_memory_config

    def from_torch(tensor, **kwargs):
        module.calls.append(
            {
                "op": "from_torch",
                "name": tensor.name,
                "shape": list(tensor.shape),
                "values": tensor.values,
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
        physical_decode = len(fused_qkv.shape) >= 4
        batch = fused_qkv.shape[-2] if physical_decode else fused_qkv.shape[0]
        module.calls.append(
            {
                "op": "nlp_create_qkv_heads_decode",
                "qkv": fused_qkv.name,
                "kwargs": dict(kwargs),
            }
        )
        return (
            FakeTTNNTensor(
                "query",
                [1, batch, num_heads, head_dim]
                if physical_decode
                else [batch, num_heads, 1, head_dim],
            ),
            FakeTTNNTensor(
                "key",
                [1, batch, num_kv_heads, head_dim]
                if physical_decode
                else [batch, num_kv_heads, 1, head_dim],
            ),
            FakeTTNNTensor(
                "value",
                [1, batch, num_kv_heads, head_dim]
                if physical_decode
                else [batch, num_kv_heads, 1, head_dim],
            ),
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
        physical_decode = len(attention.shape) >= 4
        batch = attention.shape[1] if physical_decode else attention.shape[0]
        head_dim = attention.shape[-1]
        module.calls.append(
            {
                "op": "nlp_concat_heads_decode",
                "attention": attention.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTTNNTensor(
            "concat_heads",
            [1, 1, batch, num_heads * head_dim]
            if physical_decode
            else [batch, 1, num_heads * head_dim],
        )

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
