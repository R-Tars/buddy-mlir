from __future__ import annotations

import types
import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.config_runtime import (
    TTNNConfigResolutionError,
    realize_ttnn_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates import (
    ttnn_ops as legacy_ttnn_ops,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import (
    TTNNCompatOps,
    UnsupportedTTNNOp,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import (
    ops as ttnn_ops,
)


class TTNNOpsWrapperTest(unittest.TestCase):
    def test_runtime_config_realizes_official_ttnn_descriptors(self) -> None:
        fake = _fake_config_ttnn()
        config = {
            "metadata": {"kind": "official_profile", "enabled": True},
            "dtype": {"kind": "ttnn_dtype", "name": "bfloat8_b"},
            "memory": {
                "kind": "ttnn_sharded_memory_config",
                "strategy": "width",
                "core_grid": [8, 4],
                "shard_shape": [32, 128],
            },
            "explicit_memory": {
                "kind": "ttnn_sharded_memory_config",
                "strategy": "height",
                "core_grid": [8, 8],
                "core_ranges": [[0, 0, 7, 3], [0, 4, 7, 7]],
                "shard_shape": [32, 128],
            },
            "weight_memory": {
                "kind": "ttnn_dram_sharded_memory_config",
                "k": 4096,
                "n": 6144,
                "dram_grid_width": 8,
            },
            "programs": [
                {
                    "kind": "ttnn_matmul_dram_sharded_program_config",
                    "in0_block_w": 4,
                    "per_core_M": 1,
                    "per_core_N": 6,
                    "fused_activation": "silu",
                },
                {
                    "kind": "ttnn_sdpa_program_config",
                    "core_grid": [8, 8],
                    "sub_core_grids": [[1, 0, 8, 7]],
                    "q_chunk_size": 0,
                    "k_chunk_size": 0,
                    "exp_approx_mode": False,
                    "max_cores_per_head_batch": 16,
                },
                {
                    "kind": "ttnn_layer_norm_program_config",
                    "core_grid": [8, 4],
                    "subblock_w": 4,
                    "block_h": 1,
                    "block_w": 4,
                    "inplace": False,
                },
                {
                    "kind": ("ttnn_matmul_multicore_reuse_mcast_program_config"),
                    "core_grid": [8, 10],
                    "in0_block_w": 1,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 24,
                    "fuse_batch": True,
                },
                {
                    "kind": "ttnn_matmul_multicore_reuse_program_config",
                    "core_grid": [8, 8],
                    "in0_block_w": 4,
                    "out_subblock_h": 1,
                    "out_subblock_w": 4,
                    "per_core_M": 1,
                    "per_core_N": 128,
                },
                {
                    "kind": "ttnn_matmul_multicore_reuse_mcast_1d_program_config",
                    "core_grid": [11, 10],
                    "in0_block_w": 4,
                    "out_subblock_h": 1,
                    "out_subblock_w": 4,
                    "out_block_h": 1,
                    "out_block_w": 4,
                    "per_core_M": 1,
                    "per_core_N": 4,
                    "fuse_batch": True,
                    "mcast_in0": True,
                    "gather_in0": False,
                    "untilize_out": False,
                },
                {
                    "kind": "ttnn_matmul_multi_core_reuse_multi_cast_dram_sharded_program_config",
                    "in0_block_w": 4,
                    "per_core_M": 1,
                    "per_core_N": 4,
                },
            ],
            "compute": {
                "kind": "ttnn_wormhole_compute_kernel_config",
                "math_fidelity": "HiFi2",
                "math_approx_mode": True,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": True,
            },
        }

        resolved = realize_ttnn_config(config, fake)

        self.assertEqual(
            resolved["metadata"],
            {"kind": "official_profile", "enabled": True},
        )
        self.assertEqual(resolved["dtype"], "dtype:bfloat8_b")
        self.assertEqual(resolved["memory"]["constructor"], "sharded")
        self.assertEqual(resolved["memory"]["shape"], (32, 128))
        self.assertEqual(
            resolved["memory"]["core_grid"],
            {"constructor": "core_grid", "x": 8, "y": 4},
        )
        self.assertEqual(
            resolved["explicit_memory"]["core_grid"],
            (
                "core_range_set",
                (
                    ("core_range", (0, 0), (7, 3)),
                    ("core_range", (0, 4), (7, 7)),
                ),
            ),
        )
        self.assertEqual(
            resolved["weight_memory"],
            {
                "constructor": "memory_config",
                "memory_layout": "width_sharded",
                "buffer_type": "dram",
                "shard_spec": {
                    "constructor": "shard_spec",
                    "grid": (
                        "core_range_set",
                        (("core_range", (0, 0), (7, 0)),),
                    ),
                    "shape": (4096, 768),
                    "orientation": "row_major",
                },
            },
        )
        self.assertEqual(resolved["programs"][0]["constructor"], "matmul")
        self.assertEqual(
            resolved["programs"][0]["fused_activation"],
            "unary:silu",
        )
        self.assertEqual(resolved["programs"][1]["constructor"], "sdpa")
        self.assertEqual(
            resolved["programs"][1]["sub_core_grids"],
            (
                "core_range_set",
                (("core_range", (1, 0), (8, 7)),),
            ),
        )
        self.assertEqual(resolved["programs"][2]["constructor"], "norm")
        self.assertEqual(
            resolved["programs"][3]["constructor"],
            "matmul_mcast",
        )
        self.assertEqual(
            resolved["programs"][3]["compute_with_storage_grid_size"],
            (8, 10),
        )
        self.assertEqual(resolved["programs"][3]["out_block_h"], 1)
        self.assertEqual(resolved["programs"][3]["out_block_w"], 24)
        self.assertEqual(resolved["programs"][4]["constructor"], "matmul_reuse")
        self.assertEqual(resolved["programs"][5]["constructor"], "matmul_mcast_1d")
        self.assertTrue(resolved["programs"][5]["mcast_in0"])
        self.assertEqual(
            resolved["programs"][6]["constructor"],
            "matmul_batched_dram",
        )
        self.assertEqual(resolved["compute"]["constructor"], "compute")
        self.assertEqual(resolved["compute"]["math_fidelity"], "HiFi2")

    def test_runtime_config_rejects_unknown_ttnn_descriptor(self) -> None:
        with self.assertRaisesRegex(
            TTNNConfigResolutionError,
            "unsupported TTNN config descriptor kind",
        ):
            realize_ttnn_config(
                {"kind": "ttnn_not_a_real_descriptor"},
                types.SimpleNamespace(),
            )

    def test_module_import_does_not_require_ttnn(self) -> None:
        self.assertTrue(callable(ttnn_ops.nlp_create_qkv_heads_decode))
        self.assertTrue(callable(TTNNCompatOps))
        self.assertTrue(issubclass(UnsupportedTTNNOp, RuntimeError))
        self.assertIs(
            legacy_ttnn_ops.nlp_create_qkv_heads_decode,
            ttnn_ops.nlp_create_qkv_heads_decode,
        )
        self.assertIs(
            legacy_ttnn_ops.UnsupportedTTNNOp,
            UnsupportedTTNNOp,
        )

    def test_split_last_dim_honors_strategy_and_output_memory(self) -> None:
        calls = []

        def split(tensor, split_size, *, dim, memory_config):
            calls.append(("split", split_size, dim, memory_config))
            return ("gate", "up")

        def slice_op(tensor, starts, ends, **kwargs):
            calls.append(("slice", tuple(starts), tuple(ends), kwargs))
            return tuple(ends)

        fake = types.SimpleNamespace(
            split=split,
            slice=slice_op,
            L1_MEMORY_CONFIG="l1-interleaved",
        )
        ops = TTNNCompatOps(fake, record_ops=True)
        tensor = types.SimpleNamespace(shape=(1, 1, 32, 64))

        self.assertEqual(
            ops.split_last_dim(
                tensor,
                split_size=32,
                strategy="split",
                memory_config="L1_MEMORY_CONFIG",
            ),
            ("gate", "up"),
        )
        ops.split_last_dim(
            tensor,
            split_size=32,
            strategy="slice",
            memory_config="L1_MEMORY_CONFIG",
        )

        self.assertEqual(calls[0], ("split", 32, -1, "l1-interleaved"))
        self.assertEqual(calls[1][-1]["memory_config"], "l1-interleaved")
        self.assertEqual(calls[2][-1]["memory_config"], "l1-interleaved")

    def test_linear_and_split_errors_include_operation_name(self) -> None:
        class FailingTTNN:
            @staticmethod
            def linear(*args, **kwargs):
                raise RuntimeError("linear failed")

            @staticmethod
            def split(*args, **kwargs):
                raise RuntimeError("split failed")

        ops = TTNNCompatOps(FailingTTNN())
        with self.assertRaisesRegex(RuntimeError, "named_linear: linear failed"):
            ops.linear("input", "weight", op_name="named_linear")
        with self.assertRaisesRegex(RuntimeError, "named_split: split failed"):
            ops.split_last_dim(
                object(),
                split_size=4,
                strategy="split",
                op_name="named_split",
            )

    def test_model_ops_selects_requested_sequence_position(self) -> None:
        calls = []

        def slice_op(tensor, starts, ends, steps):
            calls.append((tensor, starts, ends, steps))
            return "selected"

        ops = TTNNCompatOps(types.SimpleNamespace(slice=slice_op), record_ops=True)
        tensor = types.SimpleNamespace(shape=(2, 8, 16))

        result = ops.select_sequence_position(tensor, 2)

        self.assertEqual(result, "selected")
        self.assertEqual(
            calls,
            [(tensor, [0, 2, 0], [2, 3, 16], [1, 1, 1])],
        )
        self.assertEqual(ops.op_log, ["select_sequence_position"])

    def test_model_ops_selects_each_users_sequence_position(self) -> None:
        slices = []

        def slice_op(tensor, starts, ends, steps):
            slices.append((starts, ends, steps))
            return f"user{starts[0]}"

        def concat(tensors, *, dim):
            return (list(tensors), dim)

        ops = TTNNCompatOps(
            types.SimpleNamespace(slice=slice_op, concat=concat),
            record_ops=True,
        )
        tensor = types.SimpleNamespace(shape=(2, 8, 16))

        result = ops.select_sequence_positions(tensor, [2, 5])

        self.assertEqual(result, (["user0", "user1"], 0))
        self.assertEqual(
            slices,
            [
                ([0, 2, 0], [1, 3, 16], [1, 1, 1]),
                ([1, 5, 0], [2, 6, 16], [1, 1, 1]),
            ],
        )
        self.assertEqual(ops.op_log, ["select_sequence_positions"])

    def test_model_ops_normalizes_prefill_hidden_to_residual_shape(self) -> None:
        calls = []

        def reshape(tensor, logical_shape, padded_shape):
            calls.append((tensor, logical_shape, padded_shape))
            return types.SimpleNamespace(shape=logical_shape)

        ops = TTNNCompatOps(types.SimpleNamespace(reshape=reshape), record_ops=True)
        hidden = types.SimpleNamespace(shape=(1, 32, 256, 4096))
        residual = types.SimpleNamespace(shape=(32, 1, 256, 4096))

        result = ops.reshape_prefill_hidden_like(hidden, residual)

        self.assertEqual(result.shape, (32, 1, 256, 4096))
        self.assertEqual(
            calls,
            [
                (
                    hidden,
                    (32, 1, 256, 4096),
                    (32, 1, 256, 4096),
                )
            ],
        )
        self.assertEqual(ops.op_log, ["reshape_prefill_hidden_like"])

    def test_model_ops_canonicalizes_prefill_layer_input(self) -> None:
        calls = []

        def reshape(tensor, logical_shape, padded_shape):
            calls.append((tensor, logical_shape, padded_shape))
            return types.SimpleNamespace(shape=logical_shape)

        ops = TTNNCompatOps(types.SimpleNamespace(reshape=reshape))
        hidden = types.SimpleNamespace(shape=(32, 256, 4096))

        result = ops.reshape_prefill_hidden_for_layer(hidden)

        self.assertEqual(result.shape, (1, 32, 256, 4096))
        self.assertEqual(
            calls,
            [(hidden, (1, 32, 256, 4096), (1, 32, 256, 4096))],
        )

    def test_model_ops_local_global_argmax_stays_on_device(self) -> None:
        calls = []

        def topk(tensor, **kwargs):
            calls.append(("topk", tensor, dict(kwargs)))
            return f"values:{tensor}", f"indices:{tensor}"

        def typecast(tensor, dtype):
            calls.append(("typecast", tensor, dtype))
            return f"uint32:{tensor}"

        def add(tensor, scalar):
            calls.append(("add", tensor, scalar))
            return f"offset:{tensor}:{scalar}"

        def concat(tensors, **kwargs):
            calls.append(("concat", list(tensors), dict(kwargs)))
            return "concat:" + ",".join(tensors)

        def gather(tensor, dim, index):
            calls.append(("gather", tensor, dim, index))
            return "global-token"

        module = types.SimpleNamespace(
            topk=topk,
            typecast=typecast,
            add=add,
            concat=concat,
            gather=gather,
            uint32="uint32",
        )
        ops = TTNNCompatOps(module)

        values0, indices0 = ops.local_argmax("logits0", vocab_start=0)
        values1, indices1 = ops.local_argmax("logits1", vocab_start=64)
        token = ops.global_argmax(
            [values0, values1],
            [indices0, indices1],
        )

        self.assertEqual(token, "global-token")
        self.assertEqual(calls[-1][0], "gather")
        self.assertEqual(calls[-1][2], -1)
        self.assertIn(("add", "uint32:indices:logits1", 64), calls)
        self.assertNotIn(("add", "uint32:indices:logits0", 0), calls)
        self.assertEqual(
            [call[0] for call in calls],
            [
                "topk",
                "typecast",
                "topk",
                "typecast",
                "add",
                "concat",
                "concat",
                "topk",
                "gather",
            ],
        )

    def test_model_ops_normalizes_prefill_topk_token_shape(self) -> None:
        calls = []

        def reshape(tensor, logical_shape, padded_shape=None):
            calls.append((logical_shape, padded_shape))
            return "normalized"

        ops = TTNNCompatOps(types.SimpleNamespace(reshape=reshape))
        token = types.SimpleNamespace(shape=(1, 32, 1, 1))

        result = ops.normalize_decode_token(token, batch_size=32)

        self.assertEqual(result, "normalized")
        self.assertEqual(calls, [((32, 1), (32, 1))])

    def test_model_ops_skips_tile_layout_when_already_tiled(self) -> None:
        calls = []
        module = types.SimpleNamespace(
            TILE_LAYOUT="tile",
            to_layout=lambda *args: calls.append(args) or "converted",
        )
        ops = TTNNCompatOps(module)
        tensor = types.SimpleNamespace(layout="tile")

        result = ops.ensure_tile_layout(tensor, op_name="to_layout.tile")

        self.assertIs(result, tensor)
        self.assertEqual(calls, [])

    def test_model_ops_skips_equal_memory_config(self) -> None:
        calls = []
        module = types.SimpleNamespace(
            to_memory_config=(
                lambda *args, **kwargs: calls.append((args, kwargs)) or "converted"
            )
        )
        ops = TTNNCompatOps(module)
        tensor = types.SimpleNamespace(memory_config=lambda: "l1")

        result = ops.to_memory_config(tensor, memory_config="l1")

        self.assertIs(result, tensor)
        self.assertEqual(calls, [])

    def test_model_ops_force_argmax_uses_official_multicore_path(self) -> None:
        calls = []

        def untilize(tensor, **kwargs):
            calls.append(("untilize", tensor, dict(kwargs)))
            return "row-major-logits"

        def argmax(tensor, **kwargs):
            calls.append(("argmax", tensor, dict(kwargs)))
            return "tokens"

        ops = TTNNCompatOps(types.SimpleNamespace(untilize=untilize, argmax=argmax))

        result = ops.force_argmax("tiled-logits")

        self.assertEqual(result, "tokens")
        self.assertEqual(
            calls,
            [
                ("untilize", "tiled-logits", {"use_multicore": True}),
                (
                    "argmax",
                    "row-major-logits",
                    {
                        "dim": -1,
                        "keepdim": False,
                        "use_multicore": True,
                    },
                ),
            ],
        )

    def test_model_ops_force_argmax_normalizes_prefill_logits(self) -> None:
        calls = []

        def reshape(tensor, logical_shape, padded_shape=None):
            calls.append(("reshape", logical_shape, padded_shape))
            return types.SimpleNamespace(shape=logical_shape)

        def untilize(tensor, **kwargs):
            calls.append(("untilize", tuple(tensor.shape), dict(kwargs)))
            return tensor

        def argmax(tensor, **kwargs):
            calls.append(("argmax", tuple(tensor.shape), dict(kwargs)))
            return "tokens"

        ops = TTNNCompatOps(
            types.SimpleNamespace(
                reshape=reshape,
                untilize=untilize,
                argmax=argmax,
            )
        )
        logits = types.SimpleNamespace(shape=(1, 32, 1, 128256))

        result = ops.force_argmax(logits)

        self.assertEqual(result, "tokens")
        self.assertEqual(
            calls[0],
            ("reshape", (1, 1, 32, 128256), (1, 1, 32, 128256)),
        )
        self.assertEqual(calls[1][1], (1, 1, 32, 128256))

    def test_qkv_heads_wrapper_calls_experimental_api(self) -> None:
        fake = _fake_ttnn()

        q, k, v = ttnn_ops.nlp_create_qkv_heads_decode(
            fake,
            "fused_qkv",
            num_heads=32,
            num_kv_heads=8,
            overlap_qk_coregrid=False,
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
                        "overlap_qk_coregrid": False,
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

    def test_fused_rope_and_cache_wrappers_call_experimental_apis(self) -> None:
        fake = _fake_ttnn()

        q, k = ttnn_ops.rotary_embedding_fused_qk(
            fake,
            "q_pre",
            "k_pre",
            cos_matrix="cos",
            sin_matrix="sin",
            transformation_matrix="trans",
            compute_kernel_config="compute",
        )
        key_cache, value_cache = ttnn_ops.paged_fused_update_cache(
            fake,
            "key_cache",
            q,
            "value_cache",
            "value",
            update_idxs_tensor="pos",
            page_table="page_table",
        )

        self.assertEqual((q, k), ("fused_rotary:q_pre", "fused_rotary:k_pre"))
        self.assertEqual(
            (key_cache, value_cache),
            ("updated:key_cache", "updated:value_cache"),
        )
        self.assertIn(
            (
                "rotary_embedding_llama_fused_qk",
                "q_pre",
                "k_pre",
                "cos",
                "sin",
                "trans",
                {"compute_kernel_config": "compute"},
            ),
            fake.calls,
        )
        self.assertIn(
            (
                "paged_fused_update_cache",
                "key_cache",
                "fused_rotary:q_pre",
                "value_cache",
                "value",
                {
                    "update_idxs_tensor": "pos",
                    "page_table": "page_table",
                },
            ),
            fake.calls,
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

    def test_prefill_wrappers_call_transformer_and_cache_apis(self) -> None:
        fake = _fake_ttnn()

        q, k, v = ttnn_ops.split_qkv_heads_prefill(
            fake,
            "fused_qkv",
            num_heads=32,
            num_kv_heads=8,
            memory_config="heads_mem",
        )
        q, k = ttnn_ops.rotary_embedding_prefill(
            fake,
            q,
            k,
            cos_matrix="cos",
            sin_matrix="sin",
            transformation_matrix="trans",
        )
        attn = ttnn_ops.scaled_dot_product_attention(
            fake,
            q,
            k,
            v,
            scale=0.125,
            memory_config="sdpa_mem",
        )
        cache = ttnn_ops.fill_cache(fake, "cache", "key", user_id=0)
        paged_cache = ttnn_ops.paged_fill_cache(
            fake,
            "paged_cache",
            "paged_key",
            "page_table",
            batch_idx=3,
        )
        out = ttnn_ops.concat_heads_prefill(fake, attn)

        self.assertEqual(
            (q, k, v), ("rotary:q_prefill", "rotary:k_prefill", "v_prefill")
        )
        self.assertEqual(cache, "filled:cache")
        self.assertEqual(paged_cache, "paged_filled:paged_cache")
        self.assertEqual(out, "concat_prefill:prefill_attn")
        self.assertIn(
            (
                "split_query_key_value_and_split_heads",
                "fused_qkv",
                {
                    "num_heads": 32,
                    "num_kv_heads": 8,
                    "transpose_key": False,
                    "memory_config": "heads_mem",
                },
            ),
            fake.calls,
        )
        self.assertIn(
            (
                "scaled_dot_product_attention",
                "rotary:q_prefill",
                "rotary:k_prefill",
                "v_prefill",
                {
                    "is_causal": True,
                    "scale": 0.125,
                    "memory_config": "sdpa_mem",
                },
            ),
            fake.calls,
        )

    def test_fill_cache_falls_back_to_positional_batch_index(self) -> None:
        calls = []

        def fill_cache_for_user_(cache_tensor, update_tensor, batch_index):
            calls.append((cache_tensor, update_tensor, batch_index))
            return f"filled:{cache_tensor}:{batch_index}"

        fake = types.SimpleNamespace(
            kv_cache=types.SimpleNamespace(fill_cache_for_user_=fill_cache_for_user_)
        )

        out = ttnn_ops.fill_cache(fake, "cache", "key", user_id=3)

        self.assertEqual(out, "filled:cache:3")
        self.assertEqual(calls, [("cache", "key", 3)])

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

        with self.assertRaisesRegex(UnsupportedTTNNOp, "rotary_embedding_llama"):
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
        module.calls.append(("nlp_create_qkv_heads_decode", fused_qkv, dict(kwargs)))
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

    def rotary_embedding_llama_fused_qk(
        query,
        key,
        cos_matrix,
        sin_matrix,
        transformation_matrix,
        **kwargs,
    ):
        module.calls.append(
            (
                "rotary_embedding_llama_fused_qk",
                query,
                key,
                cos_matrix,
                sin_matrix,
                transformation_matrix,
                dict(kwargs),
            )
        )
        return f"fused_rotary:{query}", f"fused_rotary:{key}"

    def paged_update_cache(cache_tensor, update_tensor, **kwargs):
        module.calls.append(
            ("paged_update_cache", cache_tensor, update_tensor, dict(kwargs))
        )
        return f"updated:{cache_tensor}"

    def paged_fused_update_cache(
        key_cache,
        key,
        value_cache,
        value,
        **kwargs,
    ):
        module.calls.append(
            (
                "paged_fused_update_cache",
                key_cache,
                key,
                value_cache,
                value,
                dict(kwargs),
            )
        )
        return f"updated:{key_cache}", f"updated:{value_cache}"

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

    def split_query_key_value_and_split_heads(fused_qkv, **kwargs):
        module.calls.append(
            (
                "split_query_key_value_and_split_heads",
                fused_qkv,
                dict(kwargs),
            )
        )
        return "q_prefill", "k_prefill", "v_prefill"

    def scaled_dot_product_attention(query, key, value, **kwargs):
        module.calls.append(
            (
                "scaled_dot_product_attention",
                query,
                key,
                value,
                dict(kwargs),
            )
        )
        return "prefill_attn"

    def fill_cache(cache_tensor, update_tensor, **kwargs):
        module.calls.append(("fill_cache", cache_tensor, update_tensor, dict(kwargs)))
        return f"filled:{cache_tensor}"

    def paged_fill_cache(cache_tensor, update_tensor, page_table, **kwargs):
        module.calls.append(
            (
                "paged_fill_cache",
                cache_tensor,
                update_tensor,
                page_table,
                dict(kwargs),
            )
        )
        return f"paged_filled:{cache_tensor}"

    def concatenate_heads(attention, **kwargs):
        module.calls.append(("concatenate_heads", attention, dict(kwargs)))
        return f"concat_prefill:{attention}"

    def to_memory_config(tensor, **kwargs):
        module.calls.append(("to_memory_config", tensor, dict(kwargs)))
        return f"mem:{tensor}"

    def nlp_concat_heads_decode(attention, **kwargs):
        module.calls.append(("nlp_concat_heads_decode", attention, dict(kwargs)))
        return f"concat:{attention}"

    module.experimental = types.SimpleNamespace(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        rotary_embedding_llama=rotary_embedding_llama,
        rotary_embedding_llama_fused_qk=rotary_embedding_llama_fused_qk,
        paged_update_cache=paged_update_cache,
        paged_fused_update_cache=paged_fused_update_cache,
        paged_fill_cache=paged_fill_cache,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
    module.transformer = types.SimpleNamespace(
        paged_scaled_dot_product_attention_decode=(
            paged_scaled_dot_product_attention_decode
        ),
        split_query_key_value_and_split_heads=(split_query_key_value_and_split_heads),
        scaled_dot_product_attention=scaled_dot_product_attention,
        concatenate_heads=concatenate_heads,
    )
    module.kv_cache = types.SimpleNamespace(fill_cache_for_user_=fill_cache)
    module.to_memory_config = to_memory_config
    return module


def _fake_config_ttnn():
    def constructor(name):
        def build(**kwargs):
            return {"constructor": name, **kwargs}

        return build

    return types.SimpleNamespace(
        bfloat8_b="dtype:bfloat8_b",
        ShardStrategy=types.SimpleNamespace(WIDTH="width", HEIGHT="height"),
        ShardOrientation=types.SimpleNamespace(ROW_MAJOR="row_major"),
        TensorMemoryLayout=types.SimpleNamespace(WIDTH_SHARDED="width_sharded"),
        BufferType=types.SimpleNamespace(DRAM="dram"),
        MathFidelity=types.SimpleNamespace(HiFi2="HiFi2"),
        UnaryOpType=types.SimpleNamespace(SILU="unary:silu"),
        CoreCoord=lambda x, y: (x, y),
        CoreRange=lambda start, end: ("core_range", start, end),
        CoreRangeSet=lambda ranges: (
            "core_range_set",
            tuple(sorted(ranges)),
        ),
        ShardSpec=lambda grid, shape, orientation: {
            "constructor": "shard_spec",
            "grid": grid,
            "shape": shape,
            "orientation": orientation,
        },
        MemoryConfig=lambda memory_layout, buffer_type, shard_spec: {
            "constructor": "memory_config",
            "memory_layout": memory_layout,
            "buffer_type": buffer_type,
            "shard_spec": shard_spec,
        },
        CoreGrid=constructor("core_grid"),
        create_sharded_memory_config=constructor("sharded"),
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig=constructor("matmul"),
        MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig=constructor(
            "matmul_batched_dram"
        ),
        MatmulMultiCoreReuseProgramConfig=constructor("matmul_reuse"),
        MatmulMultiCoreReuseMultiCastProgramConfig=constructor("matmul_mcast"),
        MatmulMultiCoreReuseMultiCast1DProgramConfig=constructor("matmul_mcast_1d"),
        SDPAProgramConfig=constructor("sdpa"),
        LayerNormShardedMultiCoreProgramConfig=constructor("norm"),
        WormholeComputeKernelConfig=constructor("compute"),
    )


if __name__ == "__main__":
    unittest.main()
