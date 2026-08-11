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
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.compiler import (
    build_codegen_config,
    render_python_ttnn_model,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.codegen import (
    dry_run_report,
    write_python_ttnn_skeleton,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import import_hf_llama
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.attention_decode import official_paged_attention_decode_op_sequence
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import build_execution_plan
from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import TTNNCompatOps


def _fake_plan(num_layers: int = 2, lm_head_split_count: int = 8) -> dict[str, object]:
    graph = import_hf_llama("/tmp/fake-llama-codegen", config={
        "_name_or_path": "fake-llama-codegen", "model_type": "llama", "num_hidden_layers": num_layers,
        "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 128, "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0, "tie_word_embeddings": False,
    }, state_dict_metadata=[], mode="decode", batch_size=32, seq_len=1, max_cache_len=1024)
    return build_execution_plan(graph, {
        "device": "p150a", "model": "llama3.1-8b", "batch_size": 32, "decode_seq_len": 1,
        "prefill_seq_len": 128, "max_cache_len": 1024,
        "attention_template": "official_paged_attention_decode", "mlp_template": "official_gated_mlp_decode",
        "lm_head_template": "official_split_lm_head", "kv_cache_template": "paged_kv_cache",
        "generation_template": "device_argmax_greedy", "lm_head_split_count": lm_head_split_count,
        "dtype_recipe": "official_like_performance_seed",
    })


class _FakeTensor:
    def __init__(self, name: str, mem_config: str | None = None, *, shape: tuple[int, ...] | None = None):
        self.name, self._mem_config = name, mem_config
        if shape is not None:
            self.shape = shape

    def memory_config(self):
        return self._mem_config


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _make_fake_ttnn_module():
    module = types.ModuleType("ttnn")
    module.calls, module.uint32 = [], "ttnn.uint32"
    module.UnaryOpType = _ns(SILU="SILU")
    module.UnaryWithParam = lambda op: ("UnaryWithParam", op)

    def name(value):
        return getattr(value, "name", value)

    def result(op, result_name, *, mem=None, shape=None, **payload):
        module.calls.append({"op": op, **payload})
        return _FakeTensor(result_name, mem, shape=shape)

    def linear(activation, weight, **kwargs):
        return result("linear", f"linear:{weight}", mem=kwargs.get("memory_config"), activation=name(activation), weight=weight, kwargs=dict(kwargs))

    def embedding(token_ids, weight, **kwargs):
        return result("embedding", f"embedding:{weight}", mem=kwargs.get("memory_config"), token_ids=token_ids, weight=weight, kwargs=dict(kwargs))

    def rms_norm(hidden, **kwargs):
        return result("rms_norm", f"rms_norm:{kwargs.get('weight')}", mem=kwargs.get("memory_config"), hidden=name(hidden), kwargs=dict(kwargs))

    def binary(op, lhs, rhs, **kwargs):
        return result(op, f"{op}_out", mem=kwargs.get("memory_config"), lhs=name(lhs), rhs=name(rhs), kwargs=dict(kwargs))

    def concat(tensors, **kwargs):
        names = [name(item) for item in tensors]
        return result("concat", "concat:" + ",".join(names), mem=kwargs.get("memory_config"), tensors=names, kwargs=dict(kwargs))

    def to_memory_config(tensor, **kwargs):
        return result("to_memory_config", f"mem:{name(tensor)}", mem=kwargs.get("memory_config"), tensor=name(tensor), kwargs=dict(kwargs))

    def unary(op, tensor, **kwargs):
        return result(op, f"{op}:{name(tensor)}", tensor=name(tensor), kwargs=dict(kwargs))

    def topk(tensor, **kwargs):
        module.calls.append({"op": "topk", "tensor": name(tensor), "kwargs": dict(kwargs)})
        return _FakeTensor(f"topk_values:{name(tensor)}"), _FakeTensor(f"topk_indices:{name(tensor)}")

    def typecast(tensor, dtype):
        return result("typecast", f"typecast:{name(tensor)}", tensor=name(tensor), dtype=dtype)

    def gather(tensor, dim, index):
        return result("gather", f"gather:{name(tensor)}:{name(index)}", tensor=name(tensor), dim=dim, index=name(index))

    def squeeze(tensor, dim):
        shape = list(tensor.shape); shape.pop(int(dim))
        return result("squeeze", f"squeeze:{name(tensor)}", mem=tensor._mem_config, shape=tuple(shape), tensor=name(tensor), dim=dim)

    def reshape(tensor, logical_shape, padded_shape=None):
        padded = tuple(padded_shape) if padded_shape is not None else None
        return result("reshape", f"reshape:{name(tensor)}", mem=tensor._mem_config, shape=tuple(logical_shape), tensor=name(tensor), logical_shape=tuple(logical_shape), padded_shape=padded)

    def slice_tensor(tensor, starts, ends, steps=None):
        shape = tuple(int(end) - int(start) for start, end in zip(starts, ends))
        return result("slice", f"slice:{name(tensor)}:{starts[0]}", mem=tensor._mem_config, shape=shape, tensor=name(tensor), starts=list(starts), ends=list(ends), steps=list(steps) if steps else None)

    def fill_cache(cache, update, *args, **kwargs):
        if args and "user_id" not in kwargs:
            kwargs["user_id"] = args[0]
        return result("fill_cache", name(cache), mem=cache._mem_config, shape=getattr(cache, "shape", None), cache=name(cache), update=name(update), kwargs=dict(kwargs))

    def paged_fill_cache(cache, update, page_table, **kwargs):
        return result("paged_fill_cache", name(cache), mem=cache._mem_config, shape=getattr(cache, "shape", None), cache=name(cache), update=name(update), update_shape=list(update.shape), page_table=name(page_table), kwargs=dict(kwargs))

    def qkv(qkv, **kwargs):
        module.calls.append({"op": "nlp_create_qkv_heads_decode", "qkv": name(qkv), "kwargs": dict(kwargs)})
        return _FakeTensor("q"), _FakeTensor("k"), _FakeTensor("v")

    def sdpa(query, key_cache, value_cache, **kwargs):
        return result("paged_scaled_dot_product_attention_decode", f"sdpa:{name(query)}", query=name(query), key_cache=name(key_cache), value_cache=name(value_cache), page_table=kwargs.get("page_table_tensor"), cache_position=kwargs.get("cur_pos_tensor"), kwargs=dict(kwargs))

    def concat_heads(attn, **kwargs):
        return result("nlp_concat_heads_decode", f"concat_heads:{name(attn)}", attn=name(attn), kwargs=dict(kwargs))

    module.linear, module.embedding, module.rms_norm = linear, embedding, rms_norm
    module.mul = lambda lhs, rhs, **kwargs: binary("mul", lhs, rhs, **kwargs)
    module.add = lambda lhs, rhs, **kwargs: binary("add", lhs, rhs, **kwargs)
    module.concat, module.to_memory_config = concat, to_memory_config
    module.argmax = lambda tensor, **kwargs: unary("argmax", tensor, **kwargs)
    module.untilize = lambda tensor, **kwargs: unary("untilize", tensor, **kwargs)
    module.topk, module.typecast, module.gather = topk, typecast, gather
    module.squeeze, module.reshape, module.slice = squeeze, reshape, slice_tensor
    module.experimental = _ns(nlp_create_qkv_heads_decode=qkv, nlp_concat_heads_decode=concat_heads, paged_fill_cache=paged_fill_cache)
    module.kv_cache = _ns(fill_cache_for_user_=fill_cache)
    module.transformer = _ns(paged_scaled_dot_product_attention_decode=sdpa)
    return module


def _load_generated_model(path: Path):
    name = "generated_buddy_ttnn_model"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class PythonTTNNSkeletonCodegenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def generated(self, *, layers: int = 1, splits: int = 8):
        source = render_python_ttnn_model(_fake_plan(layers, splits))
        path = self.root / f"model_{layers}_{splits}.py"
        path.write_text(source)
        fake = _make_fake_ttnn_module()
        with patch.dict(sys.modules, {"ttnn": fake}):
            module = _load_generated_model(path)
        return module, fake, source

    def test_codegen_python_writes_skeleton_artifacts(self) -> None:
        out = self.root / "generated"
        plan = _fake_plan(2)
        write_python_ttnn_skeleton(plan, out)
        self.assertEqual({path.name for path in out.iterdir()}, {"model.py", "config.json", "plan.json", "README.md"})
        source = (out / "model.py").read_text()
        markers = (
            "import ttnn", "ttnn_compat.model_ops import", "class BuddyLlama31TTNN", "def decode_step",
            "def decode_layer", 'op_name="residual_add.attn"', 'op_name="residual_add.mlp"',
            "self.ops.linear", "self.ops.mul_silu", "self.ops.concat", "self.ops.local_argmax",
            "self.ops.global_argmax", "self.ops.embedding", "self.ops.rms_norm", "layer_params.wqkv_packed.weight",
            "self.ops.nlp_create_qkv_heads_decode", "self.rotary_embedding_decode", "self.paged_update_kv_cache",
            "self.ops.paged_sdpa_decode", "self.ops.nlp_concat_heads_decode", "layer_params.o_proj.weight",
            "GENERATED_LM_HEAD_SPLIT_COUNT = 8",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        self.assertNotIn("class TTNNCompatOps", source)
        self.assertNotIn("raise NotImplementedError", source)
        self.assertIn("self.op_log = [] if record_ops else None", inspect.getsource(TTNNCompatOps))
        py_compile.compile(str(out / "model.py"), doraise=True)
        config = json.loads((out / "config.json").read_text())
        expected = {"num_layers": 2, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "vocab_size": 128}
        self.assertEqual({key: config[key] for key in expected}, expected)
        self.assertEqual(config["attention"]["op_sequence"], official_paged_attention_decode_op_sequence())
        self.assertEqual((config["attention"]["scale"], config["mlp"]["template"], config["generation"]["mode"]), (0.5, "official_gated_mlp_decode", "greedy"))
        self.assertEqual((config["lm_head"]["split_count"], len(config["lm_head"]["splits"]), config["lm_head"]["splits"][0], config["lm_head"]["splits"][-1]), (8, 8, {"shard_id": 0, "vocab_start": 0, "vocab_end": 16}, {"shard_id": 7, "vocab_start": 112, "vocab_end": 128}))
        kv = config["kv_cache"]
        self.assertEqual({key: kv[key] for key in ("template", "policy", "page_block_size", "dtype", "max_cache_len", "num_kv_heads", "head_dim")}, {"template": "paged_kv_cache", "policy": "paged", "page_block_size": 32, "dtype": "bfloat8_b", "max_cache_len": 1024, "num_kv_heads": 2, "head_dim": 4})
        self.assertEqual(json.loads((out / "plan.json").read_text())["layers"], plan["layers"])

    def test_generated_embedding_and_norm_use_ttnn_wrappers(self) -> None:
        generated, fake, _ = self.generated()
        params = _ns(embedding=_ns(weight="embed_weight"), layers=[_ns(input_norm=_ns(weight="attn_norm_weight"), post_attention_norm=_ns(weight="mlp_norm_weight"))], final_norm=_ns(weight="final_norm_weight"))
        config = _ns(num_layers=1, embedding=_ns(output_memory_config="embed_mem", output_dtype="bf16"), rms_norm=_ns(eps=1e-5, output_memory_config="norm_mem", output_dtype="bf16"))
        model = generated.BuddyLlama31TTNN(device=None, parameters=params, config=config)
        outputs = (model.embed("token_ids"), model.rmsnorm(_FakeTensor("hidden"), 0, kind="attn"), model.rmsnorm(_FakeTensor("hidden"), 0, kind="mlp"), model.final_norm(_FakeTensor("hidden")))
        self.assertEqual([item.name for item in outputs], ["embedding:embed_weight", "rms_norm:attn_norm_weight", "rms_norm:mlp_norm_weight", "rms_norm:final_norm_weight"])
        self.assertEqual([item["op"] for item in fake.calls], ["embedding", "rms_norm", "rms_norm", "rms_norm"])
        self.assertEqual(fake.calls[0]["kwargs"], {"memory_config": "embed_mem", "dtype": "bf16"})
        self.assertEqual(fake.calls[1]["kwargs"], {"weight": "attn_norm_weight", "epsilon": 1e-5, "memory_config": "norm_mem"})

    def test_generated_rms_norm_converts_hidden_to_tile_layout(self) -> None:
        generated, fake, _ = self.generated()
        fake.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
        fake.to_layout = lambda tensor, layout: (fake.calls.append({"op": "to_layout", "tensor": tensor.name, "layout": layout}) or _FakeTensor(f"tile:{tensor.name}"))
        model = generated.BuddyLlama31TTNN(device=None, parameters=_ns(layers=[_ns(input_norm=_ns(weight="attn_norm_weight"))]), config=_ns(rms_norm=_ns(eps=1e-5, output_memory_config="norm_mem", output_dtype="bf16")))
        out = model.rmsnorm(_FakeTensor("hidden"), 0, kind="attn")
        self.assertEqual((out.name, [item["op"] for item in fake.calls], fake.calls[1]["hidden"]), ("rms_norm:attn_norm_weight", ["to_layout", "rms_norm"], "tile:hidden"))
        self.assertIn("to_layout.tile.{op_name}", inspect.getsource(TTNNCompatOps))

    def test_lm_head_split_count_changes_codegen_and_config(self) -> None:
        configs = {count: build_codegen_config(_fake_plan(1, count)) for count in (1, 2, 8)}
        sources = {count: render_python_ttnn_model(_fake_plan(1, count)) for count in (1, 2, 8)}
        self.assertEqual(len(set(sources.values())), 3)
        for count in (1, 2, 8):
            self.assertIn(f"GENERATED_LM_HEAD_SPLIT_COUNT = {count}", sources[count])
        self.assertEqual(configs[1]["lm_head"]["splits"], [{"shard_id": 0, "vocab_start": 0, "vocab_end": 128}])
        self.assertEqual(configs[2]["lm_head"]["splits"], [{"shard_id": 0, "vocab_start": 0, "vocab_end": 64}, {"shard_id": 1, "vocab_start": 64, "vocab_end": 128}])
        self.assertEqual(configs[8]["lm_head"]["splits"][3]["vocab_end"], 64)

    def test_generated_mlp_decode_uses_mockable_ttnn_wrappers(self) -> None:
        generated, fake, _ = self.generated()
        params = _ns(layers=[_ns(mlp=_ns(gate_proj=_ns(weight="gate_weight"), up_proj=_ns(weight="up_weight"), down_proj=_ns(weight="down_weight")))])
        config = _ns(num_layers=1, mlp=_ns(gate_output_memory_config="gate_mem", gate_program_config="gate_pc", up_output_memory_config="up_mem", up_program_config="up_pc", down_output_memory_config="down_mem", down_program_config="down_pc", compute_kernel_config="compute_cfg", intermediate_dtype="bf16", output_dtype="bf8"))
        out = generated.BuddyLlama31TTNN(device=None, parameters=params, config=config).mlp_decode(0, _FakeTensor("hidden", "hidden_mem"))
        self.assertEqual((out.name, [item["op"] for item in fake.calls]), ("linear:down_weight", ["linear", "linear", "mul", "linear"]))
        self.assertEqual([fake.calls[index]["weight"] for index in (0, 1, 3)], ["gate_weight", "up_weight", "down_weight"])
        self.assertEqual(fake.calls[0]["kwargs"], {"memory_config": "gate_mem", "program_config": "gate_pc", "compute_kernel_config": "compute_cfg", "dtype": "bf16"})
        self.assertEqual(fake.calls[2]["kwargs"], {"input_tensor_a_activations": [("UnaryWithParam", "SILU")], "memory_config": "gate_mem", "dtype": "bf16"})

    def test_generated_attention_wrappers_call_official_ops(self) -> None:
        generated, fake, _ = self.generated()
        ops = generated.TTNNCompatOps(fake)
        q, _, _ = ops.nlp_create_qkv_heads_decode(_FakeTensor("qkv"), num_heads=4, num_kv_heads=2, memory_config="heads_mem")
        attn = ops.paged_sdpa_decode(q, "k_cache", "v_cache", "page_table", "cache_pos", scale=0.5, memory_config="sdpa_mem", program_config="sdpa_pc", compute_kernel_config="sdpa_ck")
        out = ops.nlp_concat_heads_decode(attn, num_heads=4, memory_config="concat_heads_mem")
        self.assertEqual((out.name, [item["op"] for item in fake.calls]), ("concat_heads:mem:sdpa:q", ["nlp_create_qkv_heads_decode", "paged_scaled_dot_product_attention_decode", "to_memory_config", "nlp_concat_heads_decode"]))
        self.assertEqual(fake.calls[1]["kwargs"], {"cur_pos_tensor": "cache_pos", "page_table_tensor": "page_table", "scale": 0.5, "memory_config": "sdpa_mem", "program_config": "sdpa_pc", "compute_kernel_config": "sdpa_ck"})
        self.assertEqual((fake.calls[2]["kwargs"], fake.calls[3]["kwargs"]), ({"memory_config": "concat_heads_mem"}, {"num_heads": 4}))

    def test_generated_prefill_qkv_reshape_squeezes_batch_or_unit_axis(self) -> None:
        generated, fake, _ = self.generated()
        ops = generated.TTNNCompatOps(fake)
        outputs = [ops.reshape_prefill_qkv_for_heads(_FakeTensor(name, shape=shape)) for name, shape in (("official", (1, 32, 128, 64)), ("runtime", (32, 1, 128, 64)))]
        self.assertEqual([item.shape for item in outputs], [(32, 128, 64), (32, 128, 64)])
        self.assertEqual([item["dim"] for item in fake.calls], [0, 1])

    def test_generated_decode_hidden_normalizes_layer_layout(self) -> None:
        generated, fake, source = self.generated(layers=2)
        self.assertIn("reshape_hidden_decode", source)
        self.assertIs(generated.TTNNCompatOps, TTNNCompatOps)
        ops = generated.TTNNCompatOps(fake)
        outputs = [ops.reshape_decode_hidden_for_layer(_FakeTensor(name, shape=shape)) for name, shape in (("3d", (32, 1, 4096)), ("4d", (1, 32, 1, 4096)))]
        self.assertEqual([item.shape for item in outputs], [(1, 1, 32, 4096)] * 2)
        self.assertEqual([item["op"] for item in fake.calls[-2:]], ["reshape", "reshape"])
        self.assertEqual((fake.calls[-2]["logical_shape"], fake.calls[-1]["padded_shape"]), ((1, 1, 32, 4096), (1, 1, 32, 4096)))

    def test_generated_argmax_token_normalizes_decode_layout(self) -> None:
        generated, fake, source = self.generated(layers=2)
        self.assertIn("self.ops.normalize_decode_token", source)
        token = generated.TTNNCompatOps(fake).normalize_decode_token(_FakeTensor("token", shape=(1, 1, 32)), batch_size=32)
        self.assertEqual((token.shape, fake.calls[-1]["op"], fake.calls[-1]["logical_shape"], fake.calls[-1]["padded_shape"]), ((32, 1), "reshape", (32, 1), (32, 1)))
        self.assertIn('self._record(f"{op_name}.reshape")', inspect.getsource(TTNNCompatOps))

    def test_generated_prefill_cache_fill_slices_each_batch_user(self) -> None:
        generated, fake, _ = self.generated()
        model = generated.BuddyLlama31TTNN(device=None, parameters=_ns(), config=_ns(batch_size=2))
        cache = [_ns(k=_FakeTensor("key_cache", shape=(2, 2, 32, 4)), v=_FakeTensor("value_cache", shape=(2, 2, 32, 4)))]
        _, report = model.fill_prefill_kv_cache(0, _FakeTensor("key_update", shape=(2, 2, 8, 4)), _FakeTensor("value_update", shape=(2, 2, 8, 4)), cache)
        self.assertEqual((report["write_policy"], report["update_shape_layout"], report["filled_user_count"], [item["user_id"] for item in report["users"]]), ("fill_cache_per_user", "batch_heads_seq_head_dim", 2, [0, 1]))
        self.assertEqual(report["users"][0]["key_update_shape"], [1, 2, 8, 4])
        self.assertEqual([item["op"] for item in fake.calls], ["slice", "slice", "fill_cache", "fill_cache"] * 2)
        self.assertEqual([item["kwargs"]["user_id"] for item in fake.calls if item["op"] == "fill_cache"], [0, 0, 1, 1])

    def test_generated_prefill_cache_fill_uses_paged_cache_with_page_table(self) -> None:
        generated, fake, _ = self.generated()
        model = generated.BuddyLlama31TTNN(device=None, parameters=_ns(), config=_ns(batch_size=2))
        cache = [_ns(k=_FakeTensor("key_cache", shape=(4, 2, 32, 4)), v=_FakeTensor("value_cache", shape=(4, 2, 32, 4)))]
        _, report = model.fill_prefill_kv_cache(0, _FakeTensor("key_update", shape=(2, 2, 128, 4)), _FakeTensor("value_update", shape=(2, 2, 128, 4)), cache, page_table=_FakeTensor("page_table", shape=(2, 2)))
        self.assertEqual((report["write_policy"], report["page_table_shape"], report["update_shape_layout"], report["filled_user_count"]), ("paged_fill_cache_per_user", [2, 2], "batch_heads_seq_head_dim", 2))
        calls = [item for item in fake.calls if item["op"] == "paged_fill_cache"]
        self.assertEqual(([item["kwargs"]["batch_idx"] for item in calls], [item["update_shape"] for item in calls]), ([0, 0, 1, 1], [[1, 2, 128, 4]] * 4))
        self.assertNotIn("fill_cache", [item["op"] for item in fake.calls])

    def test_generated_lm_head_argmax_uses_local_global_reduction(self) -> None:
        generated, fake, _ = self.generated(splits=2)
        params = _ns(lm_head=_ns(splits=[_ns(weight="lm_head_shard0"), _ns(weight="lm_head_shard1")]))
        config = _ns(lm_head=_ns(split_count=2, output_memory_config="lm_mem", concat_memory_config="concat_mem", output_dtype="bf8", compute_kernel_config="compute_cfg", program_configs=["pc0", "pc1"], argmax_strategy="local_global_argmax", splits=[_ns(vocab_start=0, vocab_end=64), _ns(vocab_start=64, vocab_end=128)], retain_logits=False), generation=_ns(mode="greedy"))
        out = generated.BuddyLlama31TTNN(device=None, parameters=params, config=config).lm_head_argmax(_FakeTensor("hidden", "hidden_mem"))
        self.assertTrue(out.name.startswith("gather:"))
        self.assertEqual([item["op"] for item in fake.calls], ["linear", "topk", "typecast", "linear", "topk", "typecast", "add", "concat", "concat", "topk", "gather"])
        linears = [item for item in fake.calls if item["op"] == "linear"]
        self.assertEqual([(item["weight"], item["kwargs"]["program_config"]) for item in linears], [("lm_head_shard0", "pc0"), ("lm_head_shard1", "pc1")])
        concats = [item for item in fake.calls if item["op"] == "concat"]
        self.assertEqual(len(concats), 2)
        self.assertTrue(all(item["kwargs"] == {"dim": -1, "memory_config": "concat_mem"} for item in concats))
        self.assertNotIn(["linear:lm_head_shard0", "linear:lm_head_shard1"], [item["tensors"] for item in concats])

    def test_codegen_python_dry_run_does_not_write_artifacts(self) -> None:
        out = self.root / "dry-run"
        report = dry_run_report(_fake_plan(1), out)
        self.assertTrue(report["dry_run"])
        self.assertFalse(out.exists())
