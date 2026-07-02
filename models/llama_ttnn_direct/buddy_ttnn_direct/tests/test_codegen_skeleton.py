from __future__ import annotations

import json
import py_compile
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.python_ttnn import (
    CustomFusedRegionNotImplemented,
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
    CUSTOM_FUSED_LM_HEAD_TEMPLATE,
    CUSTOM_FUSED_MLP_TEMPLATE,
    build_execution_plan,
    dump_execution_plan,
)


def _fake_plan(
    num_layers: int = 2, lm_head_split_count: int = 8
) -> dict[str, object]:
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
            self.assertIn("templates import ttnn_ops", source)
            self.assertIn("class BuddyLlama31TTNN", source)
            self.assertIn("def decode_step", source)
            self.assertIn("def decode_layer", source)
            self.assertIn("class TTNNCompatOps", source)
            self.assertIn("self.op_log = []", source)
            self.assertIn('op_name="residual_add.attn"', source)
            self.assertIn('op_name="residual_add.mlp"', source)
            self.assertIn("self.ops.linear", source)
            self.assertIn("self.ops.mul_silu", source)
            self.assertIn("self.ops.concat", source)
            self.assertIn("self.ops.argmax", source)
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

    def test_custom_fused_templates_fail_codegen_explicitly(self) -> None:
        plan = _fake_plan(num_layers=1)
        plan["layers"][0]["templates"][4] = CUSTOM_FUSED_MLP_TEMPLATE
        plan["final"][1] = CUSTOM_FUSED_LM_HEAD_TEMPLATE

        with self.assertRaisesRegex(
            CustomFusedRegionNotImplemented,
            "CustomFusedRegionNotImplemented",
        ) as ctx:
            render_python_ttnn_model(plan)

        message = str(ctx.exception)
        self.assertIn(CUSTOM_FUSED_MLP_TEMPLATE, message)
        self.assertIn(CUSTOM_FUSED_LM_HEAD_TEMPLATE, message)

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
                    "input_tensor_a_activations": [
                        ("UnaryWithParam", "SILU")
                    ],
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

    def test_generated_lm_head_argmax_uses_split_linear_concat_argmax(
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

            self.assertEqual(
                out.name,
                "argmax:concat:linear:lm_head_shard0,linear:lm_head_shard1",
            )
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                ["linear", "linear", "concat", "argmax"],
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
            self.assertEqual(fake_ttnn.calls[1]["weight"], "lm_head_shard1")
            self.assertEqual(fake_ttnn.calls[1]["kwargs"]["program_config"], "pc1")
            self.assertEqual(
                fake_ttnn.calls[2]["tensors"],
                ["linear:lm_head_shard0", "linear:lm_head_shard1"],
            )
            self.assertEqual(
                fake_ttnn.calls[2]["kwargs"],
                {"dim": -1, "memory_config": "concat_mem"},
            )
            self.assertEqual(
                fake_ttnn.calls[3]["kwargs"],
                {"dim": -1},
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
    def __init__(self, name: str, mem_config: str | None = None):
        self.name = name
        self._mem_config = mem_config

    def memory_config(self):
        return self._mem_config


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _make_fake_ttnn_module():
    module = types.ModuleType("ttnn")
    module.calls = []

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
    module.experimental = _ns(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
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
