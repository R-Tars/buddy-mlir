from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_prefill import (
    run_smoke_prefill,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)


class SmokePrefillTest(unittest.TestCase):
    def test_cli_smoke_prefill_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prefill_smoke_report.json"
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
                    "smoke-prefill",
                    "--program-dir",
                    str(program_dir),
                    "--layers",
                    "1",
                    "--prefill-len",
                    "8",
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
            self.assertEqual(report["template"], "prefill_smoke")
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["prefill_status"], "dry_run")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(report["input_shapes"]["token_ids"], [2, 8])
            self.assertIn("scaled_dot_product_attention", report["op_sequence"])
            self.assertEqual(report["cache_population"][0]["status"], "planned")
            self.assertEqual(
                report["expected_output_shapes"]["key_cache"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                report["cache_population"][0]["write_policy"],
                "fill_cache_per_user",
            )
            self.assertEqual(
                report["cache_population"][0]["update_shape_layout"],
                "batch_heads_seq_head_dim",
            )
            self.assertEqual(
                report["cache_population"][0]["planned_user_count"],
                2,
            )

    def test_run_smoke_prefill_executes_generated_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "prefill_smoke_report.json"
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
            report = run_smoke_prefill(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                prefill_len=8,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake_ttnn,
                torch_module=object(),
                parameters=_fake_parameters(split_count=8),
                token_ids=FakeTensor("prefill_token_ids", [2, 8]),
                kv_cache=[
                    types.SimpleNamespace(
                        k=FakeTensor("key_cache", [2, 2, 32, 4]),
                        v=FakeTensor("value_cache", [2, 2, 32, 4]),
                    )
                ],
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["prefill_status"], "passed")
            self.assertEqual(report["kv_cache_source"], "prefill")
            self.assertEqual(report["output_shapes"]["token"], [2, 8])
            self.assertEqual(report["output_shapes"]["key_cache"], [2, 2, 32, 4])
            self.assertEqual(report["cache_population"][0]["status"], "filled")
            self.assertEqual(
                report["cache_population"][0]["write_policy"],
                "fill_cache_per_user",
            )
            self.assertEqual(
                report["cache_population"][0]["update_shape_layout"],
                "batch_heads_seq_head_dim",
            )
            self.assertEqual(
                report["cache_population"][0]["filled_user_count"],
                2,
            )
            self.assertEqual(
                report["cache_population"][0]["key_cache_shape"],
                [2, 2, 32, 4],
            )
            self.assertEqual(
                [
                    user["user_id"]
                    for user in report["cache_population"][0]["users"]
                ],
                [0, 1],
            )
            self.assertEqual(
                report["cache_population"][0]["users"][0]["key_update_shape"],
                [1, 2, 8, 4],
            )
            fill_calls = [
                call for call in fake_ttnn.calls if call["op"] == "fill_cache"
            ]
            self.assertEqual(
                [call["kwargs"].get("user_id") for call in fill_calls],
                [0, 0, 1, 1],
            )
            self.assertEqual(
                len([call for call in fake_ttnn.calls if call["op"] == "slice"]),
                4,
            )
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertIn(
                "fill_cache.k",
                report["reference"]["observed_ops"],
            )
            self.assertEqual(json.loads(report_json.read_text()), report)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str = "ttnn.bfloat16",
        mem_config: str | None = None,
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype
        self._mem_config = mem_config

    def memory_config(self) -> str | None:
        return self._mem_config

    def __repr__(self) -> str:
        return f"FakeTensor({self.name})"


def _fake_parameters(split_count: int):
    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(
            weight=FakeTensor("embed_weight", [1, 1, 128, 16])
        ),
        layers=[
            types.SimpleNamespace(
                attention=types.SimpleNamespace(
                    wqkv_packed=types.SimpleNamespace(
                        weight=FakeTensor("wqkv_weight", [1, 1, 16, 32])
                    ),
                    o_proj=types.SimpleNamespace(
                        weight=FakeTensor("o_proj_weight", [1, 1, 16, 16])
                    ),
                    rotary=types.SimpleNamespace(
                        cos_matrix=FakeTensor("cos_matrix", [1, 1, 8, 4]),
                        sin_matrix=FakeTensor("sin_matrix", [1, 1, 8, 4]),
                        transformation_matrix=FakeTensor(
                            "transformation_matrix",
                            [1, 1, 4, 4],
                        ),
                    ),
                ),
                input_norm=types.SimpleNamespace(
                    weight=FakeTensor("input_norm_weight", [1, 1, 1, 16])
                ),
                post_attention_norm=types.SimpleNamespace(
                    weight=FakeTensor(
                        "post_attention_norm_weight",
                        [1, 1, 1, 16],
                    )
                ),
                mlp=types.SimpleNamespace(
                    gate_proj=types.SimpleNamespace(
                        weight=FakeTensor("gate_weight", [1, 1, 16, 32])
                    ),
                    up_proj=types.SimpleNamespace(
                        weight=FakeTensor("up_weight", [1, 1, 16, 32])
                    ),
                    down_proj=types.SimpleNamespace(
                        weight=FakeTensor("down_weight", [1, 1, 32, 16])
                    ),
                ),
            )
        ],
        final_norm=types.SimpleNamespace(
            weight=FakeTensor("final_norm_weight", [1, 1, 1, 16])
        ),
        lm_head=types.SimpleNamespace(
            splits=[
                types.SimpleNamespace(
                    shard_id=index,
                    weight=FakeTensor(f"lm_head_{index}", [1, 1, 16, 16]),
                )
                for index in range(split_count)
            ]
        ),
    )


def _make_fake_ttnn():
    module = types.ModuleType("ttnn")
    module.calls = []
    module.__version__ = "fake-ttnn"
    module.__tt_metal_commit__ = "fake-tt-metal"
    module.bfloat16 = "ttnn.bfloat16"
    module.float32 = "ttnn.float32"
    module.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"
    module.L1_MEMORY_CONFIG = "ttnn.L1_MEMORY_CONFIG"
    module.DRAM_MEMORY_CONFIG = "ttnn.DRAM_MEMORY_CONFIG"

    class UnaryOpType:
        SILU = "SILU"

    def unary_with_param(op):
        return ("UnaryWithParam", op)

    def embedding(token_ids, weight, **kwargs):
        module.calls.append({"op": "embedding", "kwargs": dict(kwargs)})
        return FakeTensor("embedding", [token_ids.shape[0], token_ids.shape[1], weight.shape[-1]])

    def rms_norm(hidden, **kwargs):
        module.calls.append({"op": "rms_norm", "kwargs": dict(kwargs)})
        return FakeTensor("rms_norm", hidden.shape)

    def linear(activation, weight, **kwargs):
        shape = list(activation.shape)
        shape[-1] = weight.shape[-1]
        module.calls.append({"op": "linear", "kwargs": dict(kwargs)})
        return FakeTensor(f"linear:{weight.name}", shape)

    def split_query_key_value_and_split_heads(fused_qkv, **kwargs):
        batch, seq_len = fused_qkv.shape[0], fused_qkv.shape[1]
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        module.calls.append(
            {
                "op": "split_query_key_value_and_split_heads",
                "kwargs": dict(kwargs),
            }
        )
        return (
            FakeTensor("query", [batch, num_heads, seq_len, head_dim]),
            FakeTensor("key", [batch, num_kv_heads, seq_len, head_dim]),
            FakeTensor("value", [batch, num_kv_heads, seq_len, head_dim]),
        )

    def rotary_embedding_llama(tensor, cos, sin, transform, **kwargs):
        module.calls.append({"op": "rotary_embedding_llama", "kwargs": dict(kwargs)})
        return FakeTensor(f"rotary:{tensor.name}", tensor.shape)

    def scaled_dot_product_attention(query, key, value, **kwargs):
        module.calls.append(
            {"op": "scaled_dot_product_attention", "kwargs": dict(kwargs)}
        )
        return FakeTensor("attention", query.shape)

    def fill_cache(cache, update, **kwargs):
        module.calls.append({"op": "fill_cache", "kwargs": dict(kwargs)})
        return FakeTensor(cache.name, cache.shape)

    def to_memory_config(tensor, **kwargs):
        module.calls.append({"op": "to_memory_config", "kwargs": dict(kwargs)})
        return FakeTensor(f"mem:{tensor.name}", tensor.shape)

    def paged_fill_cache(cache, update, page_table, **kwargs):
        module.calls.append(
            {
                "op": "paged_fill_cache",
                "page_table": page_table.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(cache.name, cache.shape)

    def slice_tensor(tensor, starts, ends, steps=None):
        output_shape = [int(end) - int(start) for start, end in zip(starts, ends)]
        module.calls.append(
            {
                "op": "slice",
                "tensor": tensor.name,
                "starts": list(starts),
                "ends": list(ends),
                "steps": list(steps) if steps is not None else None,
            }
        )
        return FakeTensor(f"slice:{tensor.name}:{starts[0]}", output_shape)

    def concatenate_heads(attention, **kwargs):
        batch, _, seq_len, head_dim = attention.shape
        module.calls.append({"op": "concatenate_heads", "kwargs": dict(kwargs)})
        return FakeTensor("concat_heads", [batch, seq_len, 4 * head_dim])

    def mul(lhs, rhs, **kwargs):
        module.calls.append({"op": "mul", "kwargs": dict(kwargs)})
        return FakeTensor("mul", lhs.shape)

    def add(lhs, rhs, **kwargs):
        module.calls.append({"op": "add", "kwargs": dict(kwargs)})
        return FakeTensor("add", lhs.shape)

    def concat(tensors, **kwargs):
        shape = list(tensors[0].shape)
        dim = int(kwargs.get("dim", -1))
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        module.calls.append({"op": "concat", "kwargs": dict(kwargs)})
        return FakeTensor("concat", shape)

    def argmax(tensor, **kwargs):
        shape = list(tensor.shape)
        dim = int(kwargs.get("dim", -1))
        if dim < 0:
            dim += len(shape)
        del shape[dim]
        module.calls.append({"op": "argmax", "kwargs": dict(kwargs)})
        return FakeTensor("argmax", shape)

    module.UnaryOpType = UnaryOpType
    module.UnaryWithParam = unary_with_param
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.linear = linear
    module.to_memory_config = to_memory_config
    module.mul = mul
    module.add = add
    module.concat = concat
    module.argmax = argmax
    module.slice = slice_tensor
    module.experimental = types.SimpleNamespace(
        rotary_embedding_llama=rotary_embedding_llama,
        paged_fill_cache=paged_fill_cache,
    )
    module.transformer = types.SimpleNamespace(
        split_query_key_value_and_split_heads=(
            split_query_key_value_and_split_heads
        ),
        scaled_dot_product_attention=scaled_dot_product_attention,
        concatenate_heads=concatenate_heads,
    )
    module.kv_cache = types.SimpleNamespace(fill_cache_for_user_=fill_cache)
    return module


if __name__ == "__main__":
    unittest.main()
