from __future__ import annotations

import importlib.util
import inspect
import json
import py_compile
import sys
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen import (
    python_ttnn as legacy_python_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler import (
    build_codegen_config,
    render_python_ttnn_model,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.attention_decode import (
    official_paged_attention_decode_op_sequence,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    dump_execution_plan,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import TTNNCompatOps


def _fake_plan(num_layers: int = 2, lm_head_split_count: int = 8) -> dict[str, object]:
    graph = import_hf_llama(
        "/tmp/fake-llama-codegen",
        config={
            "_name_or_path": "fake-llama-codegen",
            "model_type": "llama",
            "num_hidden_layers": num_layers,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        },
        state_dict_metadata=[],
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )
    return build_execution_plan(
        graph,
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
            "lm_head_split_count": lm_head_split_count,
            "dtype_recipe": "official_like_performance_seed",
        },
    )


class PythonTTNNSkeletonCodegenTest(unittest.TestCase):
    def test_codegen_compatibility_facade_reexports_compiler(self) -> None:
        self.assertIs(
            legacy_python_ttnn.build_codegen_config,
            build_codegen_config,
        )
        self.assertIs(
            legacy_python_ttnn.render_python_ttnn_model,
            render_python_ttnn_model,
        )

    def test_codegen_python_writes_skeleton_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            plan = _fake_plan(num_layers=2)
            dump_execution_plan(plan, plan_json)

            exit_code = main(
                [
                    "codegen-python",
                    "--plan-json",
                    str(plan_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "model.py").is_file())
            self.assertTrue((out_dir / "config.json").is_file())
            self.assertTrue((out_dir / "plan.json").is_file())
            self.assertTrue((out_dir / "README.md").is_file())

            source = (out_dir / "model.py").read_text()
            self.assertIn("import ttnn", source)
            self.assertIn("ttnn_compat.model_ops import", source)
            self.assertIn("class BuddyLlama31TTNN", source)
            self.assertIn("def decode_step", source)
            self.assertIn("def decode_layer", source)
            self.assertNotIn("class TTNNCompatOps", source)
            self.assertIn("self.op_log = []", inspect.getsource(TTNNCompatOps))
            self.assertIn('op_name="residual_add.attn"', source)
            self.assertIn('op_name="residual_add.mlp"', source)
            self.assertIn("self.ops.linear", source)
            self.assertIn("self.ops.mul_silu", source)
            self.assertIn("self.ops.concat", source)
            self.assertIn("self.ops.local_argmax", source)
            self.assertIn("self.ops.global_argmax", source)
            self.assertIn("self.ops.embedding", source)
            self.assertIn("self.ops.rms_norm", source)
            self.assertIn("layer_params.wqkv_packed.weight", source)
            self.assertIn("self.ops.nlp_create_qkv_heads_decode", source)
            self.assertIn("self.rotary_embedding_decode", source)
            self.assertIn("self.paged_update_kv_cache", source)
            self.assertIn("self.ops.paged_sdpa_decode", source)
            self.assertIn("self.ops.nlp_concat_heads_decode", source)
            self.assertIn("layer_params.o_proj.weight", source)
            self.assertIn("GENERATED_LM_HEAD_SPLIT_COUNT = 8", source)
            self.assertNotIn("raise NotImplementedError", source)

            py_compile.compile(
                str(out_dir / "model.py"),
                doraise=True,
            )

            config = json.loads((out_dir / "config.json").read_text())
            self.assertEqual(config["num_layers"], 2)
            self.assertEqual(config["hidden_size"], 16)
            self.assertEqual(config["intermediate_size"], 32)
            self.assertEqual(config["num_attention_heads"], 4)
            self.assertEqual(config["num_key_value_heads"], 2)
            self.assertEqual(config["head_dim"], 4)
            self.assertEqual(config["vocab_size"], 128)
            self.assertEqual(config["template_config"]["device"], "p150a")
            self.assertEqual(config["embedding"]["output_memory_config"], None)
            self.assertEqual(config["rms_norm"]["eps"], 1e-5)
            self.assertEqual(config["mlp"]["template"], "official_gated_mlp_decode")
            self.assertEqual(
                config["attention"]["op_sequence"],
                official_paged_attention_decode_op_sequence(),
            )
            self.assertEqual(config["attention"]["scale"], 0.5)
            self.assertEqual(config["lm_head"]["split_count"], 8)
            self.assertEqual(len(config["lm_head"]["splits"]), 8)
            self.assertEqual(
                config["lm_head"]["splits"][0],
                {"shard_id": 0, "vocab_start": 0, "vocab_end": 16},
            )
            self.assertEqual(
                config["lm_head"]["splits"][-1],
                {"shard_id": 7, "vocab_start": 112, "vocab_end": 128},
            )
            self.assertEqual(config["generation"]["mode"], "greedy")
            self.assertEqual(config["kv_cache"]["template"], "paged_kv_cache")
            self.assertEqual(config["kv_cache"]["policy"], "paged")
            self.assertEqual(config["kv_cache"]["page_block_size"], 32)
            self.assertEqual(config["kv_cache"]["dtype"], "bfloat8_b")
            self.assertEqual(config["kv_cache"]["max_cache_len"], 1024)
            self.assertEqual(config["kv_cache"]["num_kv_heads"], 2)
            self.assertEqual(config["kv_cache"]["head_dim"], 4)
            copied_plan = json.loads((out_dir / "plan.json").read_text())
            self.assertEqual(copied_plan["layers"], plan["layers"])

    def test_generated_embedding_and_norm_use_ttnn_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            parameters = _ns(
                embedding=_ns(weight="embed_weight"),
                layers=[
                    _ns(
                        input_norm=_ns(weight="attn_norm_weight"),
                        post_attention_norm=_ns(weight="mlp_norm_weight"),
                    )
                ],
                final_norm=_ns(weight="final_norm_weight"),
            )
            config = _ns(
                num_layers=1,
                embedding=_ns(
                    output_memory_config="embed_mem",
                    output_dtype="bf16",
                ),
                rms_norm=_ns(
                    eps=1e-5,
                    output_memory_config="norm_mem",
                    output_dtype="bf16",
                ),
            )
            model = generated.BuddyLlama31TTNN(
                device=None,
                parameters=parameters,
                config=config,
            )

            embedded = model.embed("token_ids")
            attn_norm = model.rmsnorm(_FakeTensor("hidden"), 0, kind="attn")
            mlp_norm = model.rmsnorm(_FakeTensor("hidden"), 0, kind="mlp")
            final_norm = model.final_norm(_FakeTensor("hidden"))

            self.assertEqual(embedded.name, "embedding:embed_weight")
            self.assertEqual(attn_norm.name, "rms_norm:attn_norm_weight")
            self.assertEqual(mlp_norm.name, "rms_norm:mlp_norm_weight")
            self.assertEqual(final_norm.name, "rms_norm:final_norm_weight")
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                ["embedding", "rms_norm", "rms_norm", "rms_norm"],
            )
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"],
                {"memory_config": "embed_mem", "dtype": "bf16"},
            )
            self.assertEqual(
                fake_ttnn.calls[1]["kwargs"],
                {
                    "weight": "attn_norm_weight",
                    "epsilon": 1e-5,
                    "memory_config": "norm_mem",
                    "dtype": "bf16",
                },
            )

    def test_generated_rms_norm_converts_hidden_to_tile_layout(self) -> None:
        plan = _fake_plan(num_layers=1)
        source = render_python_ttnn_model(plan)
        compat_source = inspect.getsource(TTNNCompatOps)
        self.assertIn("def ensure_tile_layout", compat_source)
        self.assertIn("to_layout.tile.{op_name}", compat_source)

        fake_ttnn = _make_fake_ttnn_module()
        fake_ttnn.TILE_LAYOUT = "ttnn.TILE_LAYOUT"

        def to_layout(tensor, layout):
            fake_ttnn.calls.append(
                {
                    "op": "to_layout",
                    "tensor": getattr(tensor, "name", tensor),
                    "layout": layout,
                }
            )
            return _FakeTensor(
                f"tile:{getattr(tensor, 'name', tensor)}",
                getattr(tensor, "_mem_config", None),
            )

        fake_ttnn.to_layout = to_layout
        sys.modules["ttnn"] = fake_ttnn
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_py = Path(tmpdir) / "model.py"
                model_py.write_text(source)
                generated = _load_generated_model(model_py)
        finally:
            sys.modules.pop("ttnn", None)

        model = generated.BuddyLlama31TTNN(
            device=None,
            parameters=_ns(
                layers=[_ns(input_norm=_ns(weight="attn_norm_weight"))],
            ),
            config=_ns(
                rms_norm=_ns(
                    eps=1e-5,
                    output_memory_config="norm_mem",
                    output_dtype="bf16",
                ),
            ),
        )

        out = model.rmsnorm(_FakeTensor("hidden"), 0, kind="attn")

        self.assertEqual(out.name, "rms_norm:attn_norm_weight")
        self.assertEqual(
            [call["op"] for call in fake_ttnn.calls],
            ["to_layout", "rms_norm"],
        )
        self.assertEqual(fake_ttnn.calls[0]["tensor"], "hidden")
        self.assertEqual(fake_ttnn.calls[0]["layout"], "ttnn.TILE_LAYOUT")
        self.assertEqual(fake_ttnn.calls[1]["hidden"], "tile:hidden")

    def test_lm_head_split_count_changes_codegen_and_config(self) -> None:
        configs = {}
        sources = {}
        for split_count in (1, 2, 8):
            plan = _fake_plan(num_layers=1, lm_head_split_count=split_count)
            configs[split_count] = build_codegen_config(plan)
            sources[split_count] = render_python_ttnn_model(plan)

        self.assertNotEqual(sources[1], sources[2])
        self.assertNotEqual(sources[2], sources[8])
        self.assertIn("GENERATED_LM_HEAD_SPLIT_COUNT = 1", sources[1])
        self.assertIn("GENERATED_LM_HEAD_SPLIT_COUNT = 2", sources[2])
        self.assertIn("GENERATED_LM_HEAD_SPLIT_COUNT = 8", sources[8])

        self.assertEqual(
            configs[1]["lm_head"]["splits"],
            [{"shard_id": 0, "vocab_start": 0, "vocab_end": 128}],
        )
        self.assertEqual(
            configs[2]["lm_head"]["splits"],
            [
                {"shard_id": 0, "vocab_start": 0, "vocab_end": 64},
                {"shard_id": 1, "vocab_start": 64, "vocab_end": 128},
            ],
        )
        self.assertEqual(configs[8]["lm_head"]["splits"][3]["vocab_end"], 64)

    def test_generated_mlp_decode_uses_mockable_ttnn_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            parameters = _ns(
                layers=[
                    _ns(
                        mlp=_ns(
                            gate_proj=_ns(weight="gate_weight"),
                            up_proj=_ns(weight="up_weight"),
                            down_proj=_ns(weight="down_weight"),
                        )
                    )
                ]
            )
            config = _ns(
                num_layers=1,
                mlp=_ns(
                    gate_output_memory_config="gate_mem",
                    gate_program_config="gate_pc",
                    up_output_memory_config="up_mem",
                    up_program_config="up_pc",
                    down_output_memory_config="down_mem",
                    down_program_config="down_pc",
                    compute_kernel_config="compute_cfg",
                    intermediate_dtype="bf16",
                    output_dtype="bf8",
                ),
            )
            model = generated.BuddyLlama31TTNN(
                device=None,
                parameters=parameters,
                config=config,
            )

            out = model.mlp_decode(0, _FakeTensor("hidden", "hidden_mem"))

            self.assertEqual(out.name, "linear:down_weight")
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                ["linear", "linear", "mul", "linear"],
            )
            self.assertEqual(fake_ttnn.calls[0]["weight"], "gate_weight")
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"],
                {
                    "memory_config": "gate_mem",
                    "program_config": "gate_pc",
                    "compute_kernel_config": "compute_cfg",
                    "dtype": "bf16",
                },
            )
            self.assertEqual(fake_ttnn.calls[1]["weight"], "up_weight")
            self.assertEqual(fake_ttnn.calls[2]["lhs"], "linear:gate_weight")
            self.assertEqual(fake_ttnn.calls[2]["rhs"], "linear:up_weight")
            self.assertEqual(
                fake_ttnn.calls[2]["kwargs"],
                {
                    "input_tensor_a_activations": [("UnaryWithParam", "SILU")],
                    "memory_config": "gate_mem",
                    "dtype": "bf16",
                },
            )
            self.assertEqual(fake_ttnn.calls[3]["weight"], "down_weight")
            self.assertEqual(
                fake_ttnn.calls[3]["kwargs"],
                {
                    "memory_config": "down_mem",
                    "program_config": "down_pc",
                    "compute_kernel_config": "compute_cfg",
                    "dtype": "bf8",
                },
            )

    def test_generated_attention_wrappers_call_official_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            ops = generated.TTNNCompatOps(fake_ttnn)
            q, k, v = ops.nlp_create_qkv_heads_decode(
                _FakeTensor("qkv"),
                num_heads=4,
                num_kv_heads=2,
                memory_config="heads_mem",
            )
            attn = ops.paged_sdpa_decode(
                q,
                "k_cache",
                "v_cache",
                "page_table",
                "cache_pos",
                scale=0.5,
                memory_config="sdpa_mem",
                program_config="sdpa_pc",
                compute_kernel_config="sdpa_ck",
            )
            out = ops.nlp_concat_heads_decode(
                attn,
                num_heads=4,
                memory_config="concat_heads_mem",
            )

            self.assertEqual(out.name, "concat_heads:mem:sdpa:q")
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "nlp_create_qkv_heads_decode",
                    "paged_scaled_dot_product_attention_decode",
                    "to_memory_config",
                    "nlp_concat_heads_decode",
                ],
            )
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"],
                {
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "memory_config": "heads_mem",
                },
            )
            self.assertEqual(fake_ttnn.calls[1]["query"], "q")
            self.assertEqual(
                fake_ttnn.calls[1]["kwargs"],
                {
                    "cur_pos_tensor": "cache_pos",
                    "page_table_tensor": "page_table",
                    "scale": 0.5,
                    "memory_config": "sdpa_mem",
                    "program_config": "sdpa_pc",
                    "compute_kernel_config": "sdpa_ck",
                },
            )
            self.assertEqual(
                fake_ttnn.calls[2]["kwargs"],
                {"memory_config": "concat_heads_mem"},
            )
            self.assertEqual(
                fake_ttnn.calls[3]["kwargs"],
                {"num_heads": 4},
            )

    def test_generated_prefill_qkv_reshape_squeezes_batch_or_unit_axis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            ops = generated.TTNNCompatOps(fake_ttnn)
            qkv_official = _FakeTensor(
                "qkv_official",
                shape=(1, 32, 128, 64),
            )
            qkv_runtime = _FakeTensor(
                "qkv_runtime",
                shape=(32, 1, 128, 64),
            )

            official = ops.reshape_prefill_qkv_for_heads(qkv_official)
            runtime = ops.reshape_prefill_qkv_for_heads(qkv_runtime)

            self.assertEqual(official.shape, (32, 128, 64))
            self.assertEqual(runtime.shape, (32, 128, 64))
            self.assertEqual(
                [call["dim"] for call in fake_ttnn.calls],
                [0, 1],
            )

    def test_generated_decode_hidden_normalizes_layer_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=2), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            source = (out_dir / "model.py").read_text()
            self.assertIn("reshape_hidden_decode", source)
            self.assertIn(
                "def reshape_decode_hidden_for_layer",
                inspect.getsource(TTNNCompatOps),
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            self.assertIs(generated.TTNNCompatOps, TTNNCompatOps)
            ops = generated.TTNNCompatOps(fake_ttnn)
            hidden_3d = _FakeTensor(
                "hidden_3d",
                shape=(32, 1, 4096),
            )
            hidden_4d = _FakeTensor(
                "hidden_4d",
                shape=(1, 32, 1, 4096),
            )

            normalized_3d = ops.reshape_decode_hidden_for_layer(hidden_3d)
            normalized_4d = ops.reshape_decode_hidden_for_layer(hidden_4d)

            self.assertEqual(normalized_3d.shape, (1, 1, 32, 4096))
            self.assertEqual(normalized_4d.shape, (1, 1, 32, 4096))
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls[-2:]],
                ["reshape", "reshape"],
            )
            self.assertEqual(
                fake_ttnn.calls[-2]["logical_shape"],
                (1, 1, 32, 4096),
            )
            self.assertEqual(
                fake_ttnn.calls[-1]["padded_shape"],
                (1, 1, 32, 4096),
            )

    def test_generated_argmax_token_normalizes_decode_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=2), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            source = (out_dir / "model.py").read_text()
            self.assertIn("self.ops.normalize_decode_token", source)
            self.assertIn(
                'self._record(f"{op_name}.reshape")',
                inspect.getsource(TTNNCompatOps),
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            ops = generated.TTNNCompatOps(fake_ttnn)
            token = _FakeTensor(
                "argmax_token",
                shape=(1, 1, 32),
            )

            normalized = ops.normalize_decode_token(token, batch_size=32)

            self.assertEqual(normalized.shape, (32, 1))
            self.assertEqual(fake_ttnn.calls[-1]["op"], "reshape")
            self.assertEqual(fake_ttnn.calls[-1]["logical_shape"], (32, 1))
            self.assertEqual(fake_ttnn.calls[-1]["padded_shape"], (32, 1))

    def test_generated_prefill_cache_fill_slices_each_batch_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            model = generated.BuddyLlama31TTNN(
                device=None,
                parameters=_ns(),
                config=_ns(batch_size=2),
            )
            kv_cache = [
                _ns(
                    k=_FakeTensor("key_cache", shape=(2, 2, 32, 4)),
                    v=_FakeTensor("value_cache", shape=(2, 2, 32, 4)),
                )
            ]

            _, report = model.fill_prefill_kv_cache(
                0,
                _FakeTensor("key_update", shape=(2, 2, 8, 4)),
                _FakeTensor("value_update", shape=(2, 2, 8, 4)),
                kv_cache,
            )

            self.assertEqual(report["write_policy"], "fill_cache_per_user")
            self.assertEqual(
                report["update_shape_layout"],
                "batch_heads_seq_head_dim",
            )
            self.assertEqual(report["filled_user_count"], 2)
            self.assertEqual(
                [user["user_id"] for user in report["users"]],
                [0, 1],
            )
            self.assertEqual(
                report["users"][0]["key_update_shape"],
                [1, 2, 8, 4],
            )
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "slice",
                    "slice",
                    "fill_cache",
                    "fill_cache",
                    "slice",
                    "slice",
                    "fill_cache",
                    "fill_cache",
                ],
            )
            fill_calls = [
                call for call in fake_ttnn.calls if call["op"] == "fill_cache"
            ]
            self.assertEqual(
                [call["kwargs"].get("user_id") for call in fill_calls],
                [0, 0, 1, 1],
            )

    def test_generated_prefill_cache_fill_uses_paged_cache_with_page_table(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            model = generated.BuddyLlama31TTNN(
                device=None,
                parameters=_ns(),
                config=_ns(batch_size=2),
            )
            kv_cache = [
                _ns(
                    k=_FakeTensor("key_cache", shape=(4, 2, 32, 4)),
                    v=_FakeTensor("value_cache", shape=(4, 2, 32, 4)),
                )
            ]

            _, report = model.fill_prefill_kv_cache(
                0,
                _FakeTensor("key_update", shape=(2, 2, 128, 4)),
                _FakeTensor("value_update", shape=(2, 2, 128, 4)),
                kv_cache,
                page_table=_FakeTensor("page_table", shape=(2, 2)),
            )

            self.assertEqual(report["write_policy"], "paged_fill_cache_per_user")
            self.assertEqual(report["page_table_shape"], [2, 2])
            self.assertEqual(
                report["update_shape_layout"],
                "batch_heads_seq_head_dim",
            )
            self.assertEqual(report["filled_user_count"], 2)
            paged_calls = [
                call for call in fake_ttnn.calls if call["op"] == "paged_fill_cache"
            ]
            self.assertEqual(len(paged_calls), 4)
            self.assertEqual(
                [call["kwargs"].get("batch_idx") for call in paged_calls],
                [0, 0, 1, 1],
            )
            self.assertEqual(
                [call["update_shape"] for call in paged_calls],
                [
                    [1, 2, 128, 4],
                    [1, 2, 128, 4],
                    [1, 2, 128, 4],
                    [1, 2, 128, 4],
                ],
            )
            self.assertNotIn(
                "fill_cache",
                [call["op"] for call in fake_ttnn.calls],
            )

    def test_generated_lm_head_argmax_uses_local_global_reduction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "generated"
            dump_execution_plan(
                _fake_plan(num_layers=1, lm_head_split_count=2), plan_json
            )
            self.assertEqual(
                main(
                    [
                        "codegen-python",
                        "--plan-json",
                        str(plan_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                ),
                0,
            )

            fake_ttnn = _make_fake_ttnn_module()
            sys.modules["ttnn"] = fake_ttnn
            try:
                generated = _load_generated_model(out_dir / "model.py")
            finally:
                sys.modules.pop("ttnn", None)

            parameters = _ns(
                lm_head=_ns(
                    splits=[
                        _ns(weight="lm_head_shard0"),
                        _ns(weight="lm_head_shard1"),
                    ]
                )
            )
            config = _ns(
                lm_head=_ns(
                    split_count=2,
                    output_memory_config="lm_mem",
                    concat_memory_config="concat_mem",
                    output_dtype="bf8",
                    compute_kernel_config="compute_cfg",
                    program_configs=["pc0", "pc1"],
                    argmax_strategy="local_global_argmax",
                    splits=[
                        _ns(vocab_start=0, vocab_end=64),
                        _ns(vocab_start=64, vocab_end=128),
                    ],
                    retain_logits=False,
                ),
                generation=_ns(mode="greedy"),
            )
            model = generated.BuddyLlama31TTNN(
                device=None,
                parameters=parameters,
                config=config,
            )

            out = model.lm_head_argmax(_FakeTensor("hidden", "hidden_mem"))

            self.assertTrue(out.name.startswith("gather:"))
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "linear",
                    "topk",
                    "typecast",
                    "linear",
                    "topk",
                    "typecast",
                    "add",
                    "concat",
                    "concat",
                    "topk",
                    "gather",
                ],
            )
            self.assertEqual(fake_ttnn.calls[0]["weight"], "lm_head_shard0")
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"],
                {
                    "memory_config": "lm_mem",
                    "program_config": "pc0",
                    "compute_kernel_config": "compute_cfg",
                    "dtype": "bf8",
                },
            )
            linear_calls = [call for call in fake_ttnn.calls if call["op"] == "linear"]
            self.assertEqual(linear_calls[1]["weight"], "lm_head_shard1")
            self.assertEqual(linear_calls[1]["kwargs"]["program_config"], "pc1")
            concat_calls = [call for call in fake_ttnn.calls if call["op"] == "concat"]
            self.assertEqual(len(concat_calls), 2)
            self.assertTrue(
                all(
                    call["kwargs"] == {"dim": -1, "memory_config": "concat_mem"}
                    for call in concat_calls
                )
            )
            self.assertNotIn(
                ["linear:lm_head_shard0", "linear:lm_head_shard1"],
                [call["tensors"] for call in concat_calls],
            )

    def test_codegen_python_dry_run_does_not_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            out_dir = root / "dry-run-output"
            dump_execution_plan(_fake_plan(num_layers=1), plan_json)

            exit_code = main(
                [
                    "codegen-python",
                    "--plan-json",
                    str(plan_json),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(out_dir.exists())


class _FakeTensor:
    def __init__(
        self,
        name: str,
        mem_config: str | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ):
        self.name = name
        self._mem_config = mem_config
        if shape is not None:
            self.shape = shape

    def memory_config(self):
        return self._mem_config


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _make_fake_ttnn_module():
    module = types.ModuleType("ttnn")
    module.calls = []
    module.uint32 = "ttnn.uint32"

    class UnaryOpType:
        SILU = "SILU"

    def unary_with_param(op):
        return ("UnaryWithParam", op)

    def linear(activation, weight, **kwargs):
        module.calls.append(
            {
                "op": "linear",
                "activation": getattr(activation, "name", activation),
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(f"linear:{weight}", kwargs.get("memory_config"))

    def embedding(token_ids, weight, **kwargs):
        module.calls.append(
            {
                "op": "embedding",
                "token_ids": token_ids,
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(f"embedding:{weight}", kwargs.get("memory_config"))

    def rms_norm(hidden, **kwargs):
        module.calls.append(
            {
                "op": "rms_norm",
                "hidden": getattr(hidden, "name", hidden),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(
            f"rms_norm:{kwargs.get('weight')}",
            kwargs.get("memory_config"),
        )

    def mul(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "mul",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor("mul_out", kwargs.get("memory_config"))

    def add(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "add",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor("add_out", kwargs.get("memory_config"))

    def concat(tensors, **kwargs):
        tensor_names = [getattr(tensor, "name", tensor) for tensor in tensors]
        module.calls.append(
            {
                "op": "concat",
                "tensors": tensor_names,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(
            "concat:" + ",".join(tensor_names),
            kwargs.get("memory_config"),
        )

    def to_memory_config(tensor, **kwargs):
        tensor_name = getattr(tensor, "name", tensor)
        module.calls.append(
            {
                "op": "to_memory_config",
                "tensor": tensor_name,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(
            f"mem:{tensor_name}",
            kwargs.get("memory_config"),
        )

    def argmax(tensor, **kwargs):
        module.calls.append(
            {
                "op": "argmax",
                "tensor": getattr(tensor, "name", tensor),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(f"argmax:{getattr(tensor, 'name', tensor)}")

    def untilize(tensor, **kwargs):
        name = getattr(tensor, "name", tensor)
        module.calls.append(
            {"op": "untilize", "tensor": name, "kwargs": dict(kwargs)}
        )
        return _FakeTensor(f"untilize:{name}")

    def topk(tensor, **kwargs):
        name = getattr(tensor, "name", tensor)
        module.calls.append({"op": "topk", "tensor": name, "kwargs": dict(kwargs)})
        return _FakeTensor(f"topk_values:{name}"), _FakeTensor(f"topk_indices:{name}")

    def typecast(tensor, dtype):
        name = getattr(tensor, "name", tensor)
        module.calls.append({"op": "typecast", "tensor": name, "dtype": dtype})
        return _FakeTensor(f"typecast:{name}")

    def gather(tensor, dim, index):
        tensor_name = getattr(tensor, "name", tensor)
        index_name = getattr(index, "name", index)
        module.calls.append(
            {
                "op": "gather",
                "tensor": tensor_name,
                "dim": dim,
                "index": index_name,
            }
        )
        return _FakeTensor(f"gather:{tensor_name}:{index_name}")

    def squeeze(tensor, dim):
        shape = list(getattr(tensor, "shape", ()))
        if shape:
            shape.pop(int(dim))
        module.calls.append(
            {
                "op": "squeeze",
                "tensor": getattr(tensor, "name", tensor),
                "dim": dim,
            }
        )
        return _FakeTensor(
            f"squeeze:{getattr(tensor, 'name', tensor)}",
            getattr(tensor, "_mem_config", None),
            shape=tuple(shape),
        )

    def reshape(tensor, logical_shape, padded_shape=None):
        module.calls.append(
            {
                "op": "reshape",
                "tensor": getattr(tensor, "name", tensor),
                "logical_shape": tuple(logical_shape),
                "padded_shape": (
                    tuple(padded_shape) if padded_shape is not None else None
                ),
            }
        )
        return _FakeTensor(
            f"reshape:{getattr(tensor, 'name', tensor)}",
            getattr(tensor, "_mem_config", None),
            shape=tuple(logical_shape),
        )

    def slice_tensor(tensor, starts, ends, steps=None):
        shape = tuple(int(end) - int(start) for start, end in zip(starts, ends))
        module.calls.append(
            {
                "op": "slice",
                "tensor": getattr(tensor, "name", tensor),
                "starts": list(starts),
                "ends": list(ends),
                "steps": list(steps) if steps is not None else None,
            }
        )
        return _FakeTensor(
            f"slice:{getattr(tensor, 'name', tensor)}:{starts[0]}",
            getattr(tensor, "_mem_config", None),
            shape=shape,
        )

    def fill_cache(cache, update, *args, **kwargs):
        if args and "user_id" not in kwargs:
            kwargs["user_id"] = args[0]
        module.calls.append(
            {
                "op": "fill_cache",
                "cache": getattr(cache, "name", cache),
                "update": getattr(update, "name", update),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(
            getattr(cache, "name", "cache"),
            getattr(cache, "_mem_config", None),
            shape=getattr(cache, "shape", None),
        )

    def paged_fill_cache(cache, update, page_table, **kwargs):
        module.calls.append(
            {
                "op": "paged_fill_cache",
                "cache": getattr(cache, "name", cache),
                "update": getattr(update, "name", update),
                "update_shape": list(getattr(update, "shape", [])),
                "page_table": getattr(page_table, "name", page_table),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(
            getattr(cache, "name", "cache"),
            getattr(cache, "_mem_config", None),
            shape=getattr(cache, "shape", None),
        )

    def nlp_create_qkv_heads_decode(qkv, **kwargs):
        module.calls.append(
            {
                "op": "nlp_create_qkv_heads_decode",
                "qkv": getattr(qkv, "name", qkv),
                "kwargs": dict(kwargs),
            }
        )
        return (
            _FakeTensor("q"),
            _FakeTensor("k"),
            _FakeTensor("v"),
        )

    def paged_scaled_dot_product_attention_decode(
        query,
        key_cache,
        value_cache,
        **kwargs,
    ):
        module.calls.append(
            {
                "op": "paged_scaled_dot_product_attention_decode",
                "query": getattr(query, "name", query),
                "key_cache": getattr(key_cache, "name", key_cache),
                "value_cache": getattr(value_cache, "name", value_cache),
                "page_table": kwargs.get("page_table_tensor"),
                "cache_position": kwargs.get("cur_pos_tensor"),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(f"sdpa:{getattr(query, 'name', query)}")

    def nlp_concat_heads_decode(attn, **kwargs):
        module.calls.append(
            {
                "op": "nlp_concat_heads_decode",
                "attn": getattr(attn, "name", attn),
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(f"concat_heads:{getattr(attn, 'name', attn)}")

    module.UnaryOpType = UnaryOpType
    module.UnaryWithParam = unary_with_param
    module.linear = linear
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.mul = mul
    module.add = add
    module.concat = concat
    module.to_memory_config = to_memory_config
    module.argmax = argmax
    module.untilize = untilize
    module.topk = topk
    module.typecast = typecast
    module.gather = gather
    module.squeeze = squeeze
    module.reshape = reshape
    module.slice = slice_tensor
    module.experimental = _ns(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
        paged_fill_cache=paged_fill_cache,
    )
    module.kv_cache = _ns(fill_cache_for_user_=fill_cache)
    module.transformer = _ns(
        paged_scaled_dot_product_attention_decode=(
            paged_scaled_dot_product_attention_decode
        )
    )
    return module


def _load_generated_model(path: Path):
    module_name = "generated_buddy_ttnn_model"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


if __name__ == "__main__":
    unittest.main()
