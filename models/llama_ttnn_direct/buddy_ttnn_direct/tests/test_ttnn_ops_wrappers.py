from __future__ import annotations

import types
import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.templates import ttnn_ops
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.ttnn_ops import (
    UnsupportedTTNNOp,
)


class TTNNOpsWrapperTest(unittest.TestCase):
    def test_module_import_does_not_require_ttnn(self) -> None:
        self.assertTrue(callable(ttnn_ops.nlp_create_qkv_heads_decode))
        self.assertTrue(issubclass(UnsupportedTTNNOp, RuntimeError))

    def test_qkv_heads_wrapper_calls_experimental_api(self) -> None:
        fake = _fake_ttnn()

        q, k, v = ttnn_ops.nlp_create_qkv_heads_decode(
            fake,
            "fused_qkv",
            num_heads=32,
            num_kv_heads=8,
            memory_config="heads_mem",
        )

        self.assertEqual((q, k, v), ("q", "k", "v"))
        self.assertEqual(
            fake.calls,
            [
                (
                    "nlp_create_qkv_heads_decode",
                    "fused_qkv",
                    {
                        "num_heads": 32,
                        "num_kv_heads": 8,
                        "memory_config": "heads_mem",
                    },
                )
            ],
        )

    def test_rotary_wrapper_calls_llama_api_for_q_and_k(self) -> None:
        fake = _fake_ttnn()

        q, k = ttnn_ops.rotary_embedding_decode(
            fake,
            "q_pre",
            "k_pre",
            cos_matrix="cos",
            sin_matrix="sin",
            transformation_matrix="trans",
        )

        self.assertEqual((q, k), ("rotary:q_pre", "rotary:k_pre"))
        self.assertEqual(
            fake.calls,
            [
                (
                    "rotary_embedding_llama",
                    "q_pre",
                    "cos",
                    "sin",
                    "trans",
                    {"is_decode_mode": True},
                ),
                (
                    "rotary_embedding_llama",
                    "k_pre",
                    "cos",
                    "sin",
                    "trans",
                    {"is_decode_mode": True},
                ),
            ],
        )

    def test_paged_update_cache_wrapper_calls_experimental_api(self) -> None:
        fake = _fake_ttnn()

        out = ttnn_ops.paged_update_cache(
            fake,
            "cache",
            "update",
            update_idxs_tensor="pos",
            page_table="page_table",
        )

        self.assertEqual(out, "updated:cache")
        self.assertEqual(
            fake.calls,
            [
                (
                    "paged_update_cache",
                    "cache",
                    "update",
                    {
                        "update_idxs_tensor": "pos",
                        "page_table": "page_table",
                    },
                )
            ],
        )

    def test_paged_sdpa_wrapper_calls_transformer_api(self) -> None:
        fake = _fake_ttnn()

        out = ttnn_ops.paged_sdpa_decode(
            fake,
            "q",
            "k_cache",
            "v_cache",
            "page_table",
            "pos",
            scale=0.125,
            memory_config="sdpa_mem",
            program_config="sdpa_pc",
            compute_kernel_config="sdpa_ck",
        )

        self.assertEqual(out, "sdpa:q")
        self.assertEqual(
            fake.calls,
            [
                (
                    "paged_scaled_dot_product_attention_decode",
                    "q",
                    "k_cache",
                    "v_cache",
                    {
                        "cur_pos_tensor": "pos",
                        "page_table_tensor": "page_table",
                        "scale": 0.125,
                        "memory_config": "sdpa_mem",
                        "program_config": "sdpa_pc",
                        "compute_kernel_config": "sdpa_ck",
                    },
                )
            ],
        )

    def test_concat_heads_wrapper_calls_memory_config_then_decode_api(
        self,
    ) -> None:
        fake = _fake_ttnn()

        out = ttnn_ops.nlp_concat_heads_decode(
            fake,
            "attn",
            num_heads=32,
            memory_config="concat_mem",
        )

        self.assertEqual(out, "concat:mem:attn")
        self.assertEqual(
            fake.calls,
            [
                ("to_memory_config", "attn", {"memory_config": "concat_mem"}),
                (
                    "nlp_concat_heads_decode",
                    "mem:attn",
                    {"num_heads": 32},
                ),
            ],
        )

    def test_missing_api_raises_clear_unsupported_error(self) -> None:
        with self.assertRaises(UnsupportedTTNNOp) as ctx:
            ttnn_ops.paged_sdpa_decode(
                types.SimpleNamespace(),
                "q",
                "k_cache",
                "v_cache",
                "page_table",
                "pos",
        )

        message = str(ctx.exception)
        self.assertIn("TTNN Direct", message)
        self.assertIn("paged_scaled_dot_product_attention_decode", message)
        self.assertIn(
            "ttnn.transformer.paged_scaled_dot_product_attention_decode",
            message,
        )

    def test_rotary_and_cache_missing_api_raise_clear_errors(self) -> None:
        missing = types.SimpleNamespace()

        with self.assertRaisesRegex(
            UnsupportedTTNNOp, "rotary_embedding_llama"
        ):
            ttnn_ops.rotary_embedding_decode(
                missing,
                "q",
                "k",
                cos_matrix="cos",
                sin_matrix="sin",
                transformation_matrix="trans",
            )
        with self.assertRaisesRegex(UnsupportedTTNNOp, "paged_update_cache"):
            ttnn_ops.paged_update_cache(
                missing,
                "cache",
                "update",
                update_idxs_tensor="pos",
            )


def _fake_ttnn():
    module = types.SimpleNamespace(calls=[])

    def nlp_create_qkv_heads_decode(fused_qkv, **kwargs):
        module.calls.append(
            ("nlp_create_qkv_heads_decode", fused_qkv, dict(kwargs))
        )
        return "q", "k", "v"

    def rotary_embedding_llama(
        tensor,
        cos_matrix,
        sin_matrix,
        transformation_matrix,
        **kwargs,
    ):
        module.calls.append(
            (
                "rotary_embedding_llama",
                tensor,
                cos_matrix,
                sin_matrix,
                transformation_matrix,
                dict(kwargs),
            )
        )
        return f"rotary:{tensor}"

    def paged_update_cache(cache_tensor, update_tensor, **kwargs):
        module.calls.append(
            ("paged_update_cache", cache_tensor, update_tensor, dict(kwargs))
        )
        return f"updated:{cache_tensor}"

    def paged_scaled_dot_product_attention_decode(
        query,
        key_cache,
        value_cache,
        **kwargs,
    ):
        module.calls.append(
            (
                "paged_scaled_dot_product_attention_decode",
                query,
                key_cache,
                value_cache,
                dict(kwargs),
            )
        )
        return f"sdpa:{query}"

    def to_memory_config(tensor, **kwargs):
        module.calls.append(("to_memory_config", tensor, dict(kwargs)))
        return f"mem:{tensor}"

    def nlp_concat_heads_decode(attention, **kwargs):
        module.calls.append(
            ("nlp_concat_heads_decode", attention, dict(kwargs))
        )
        return f"concat:{attention}"

    module.experimental = types.SimpleNamespace(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        rotary_embedding_llama=rotary_embedding_llama,
        paged_update_cache=paged_update_cache,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
    module.transformer = types.SimpleNamespace(
        paged_scaled_dot_product_attention_decode=(
            paged_scaled_dot_product_attention_decode
        )
    )
    module.to_memory_config = to_memory_config
    return module


if __name__ == "__main__":
    unittest.main()
