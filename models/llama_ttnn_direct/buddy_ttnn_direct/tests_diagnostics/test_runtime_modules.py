from __future__ import annotations

import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime import decode as runtime_decode
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime import prefill as runtime_prefill
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime import reports as runtime_reports
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.context import TTNNDirectRuntimeContext
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.decode import (
    _normalize_teacher_forcing,
    _teacher_forcing_token_tensor,
    build_decode_runtime_for_position,
    materialize_generate_token_events,
    prefill_token_direct_handoff,
    run_decode_loop,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.device import GenerateDeviceSession, maybe_generate_device
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.generate import run_generate
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.inputs import (
    build_decode_kv_cache_runtime_state,
    build_decode_rotary_runtime_state,
    build_decode_runtime_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.kv_cache import build_prompt_decode_kv_cache_tensors
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.prefill import (
    attach_prefill_rotary_parameters,
    build_prefill_page_table_tensor,
    prefill_token_ids_tensor,
    run_prefill_prompt,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.profile import GenerateSectionProfiler, run_profile_generate
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.state import GENERATE_RUNTIME_OWNER, build_generate_state
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.tokenizer import (
    detokenize_generated_token_ids,
    tokenize_prompt_for_decode,
    tokenize_prompt_for_prefill,
)


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


class _SyncTTNN:
    def __init__(self):
        self.sync_count = 0

    def synchronize_device(self, _device):
        self.sync_count += 1


class _GeneratedModel:
    def __init__(self):
        self.ops = NS(argmax=lambda value: value)

    def prefill_prompt(self):
        self.embed(); self.attention_prefill(0); self.mlp_decode(0); return "prefilled"

    def decode_step(self):
        self.attention_decode(0); self.mlp_decode(0); self.final_norm(); self.lm_head_argmax(); return "decoded"

    def embed(self): return "embedded"
    def attention_prefill(self, _layer): return "prefill_attention"
    def attention_decode(self, _layer): return "decode_attention"
    def mlp_decode(self, _layer): return "mlp"
    def final_norm(self): return "normed"
    def lm_head_argmax(self): self.ops.argmax("logits"); return "token"


class _PrefillModel:
    def __init__(self):
        self.prefill_args = None
        self.prefill_token = _Tensor((2, 1))
        self.kv_cache = [NS(k=_Tensor((4, 8)), v=_Tensor((4, 8)))]
        self.cache_reports = [{"layer_id": 0, "status": "filled"}]

    def prefill_prompt(self, token_ids, kv_cache, page_table, *, valid_seq_len=None):
        self.prefill_args = (token_ids, kv_cache, page_table, valid_seq_len)
        return self.prefill_token, self.kv_cache, self.cache_reports


class _Tokenizer:
    pad_token_id, eos_token_id = 0, 2

    def __call__(self, prompt, add_special_tokens=True):
        return {"input_ids": [len(word) for word in prompt.split()]}

    def batch_decode(self, rows, skip_special_tokens=True):
        return [" ".join(f"tok{token}" for token in row) for row in rows]


class _TokenizerModule:
    AutoTokenizer = NS(from_pretrained=lambda _path: _Tokenizer())


class _HostTensor:
    def __init__(self, values=None, *, shape=None, dtype=None):
        self.values, self.shape, self.dtype, self.name = values, shape or self._shape(values), dtype, None

    @classmethod
    def _shape(cls, value):
        if not isinstance(value, list): return ()
        return (len(value), *cls._shape(value[0])) if value else (0,)


class _Torch:
    int32, bfloat16, float32 = "torch.int32", "torch.bfloat16", "torch.float32"
    tensor = staticmethod(lambda values, dtype=None: _HostTensor(values, dtype=dtype))
    zeros = staticmethod(lambda shape, dtype=None: _HostTensor(shape=shape, dtype=dtype))
    randn = zeros


class _TTNN:
    int32, uint32, bfloat16, float32 = "ttnn.int32", "ttnn.uint32", "ttnn.bfloat16", "ttnn.float32"
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = "row_major", "tile", "dram_memory"

    def __init__(self):
        self.from_torch_calls = []

    def from_torch(self, tensor, **kwargs):
        self.from_torch_calls.append((tensor, kwargs))
        return NS(source=tensor.name, shape=list(tensor.shape), dtype=tensor.dtype, values=tensor.values, kwargs=kwargs)


class RuntimeModuleTest(unittest.TestCase):
    def test_teacher_forcing_validates_and_builds_batch_token_tensor(self) -> None:
        self.assertEqual(_normalize_teacher_forcing([[11, 12], [13, 14]], decode_step_count=2, batch_size=2), [[11, 12], [13, 14]])
        with self.assertRaisesRegex(ValueError, "batch width"):
            _normalize_teacher_forcing([[11]], decode_step_count=1, batch_size=2)
        token = _teacher_forcing_token_tensor(ttnn=_TTNN(), torch=_Torch(), device="device0", token_ids=[11, 12], step_index=3)
        self.assertEqual((token.source, token.values, token.shape), ("teacher_forcing_token_ids_3", [[11], [12]], [2, 1]))

    def test_canonical_generate_modules_expose_runtime_entries(self) -> None:
        self.assertTrue(all(callable(value) for value in (run_generate, run_profile_generate, build_generate_state)))
        self.assertEqual(GENERATE_RUNTIME_OWNER, "TTNNDirectRuntimeContext")

    def test_generate_device_session_uses_injected_device(self) -> None:
        session = maybe_generate_device(ttnn=object(), device_id=7, injected=object())
        self.assertIsInstance(session, GenerateDeviceSession)
        with session as device:
            self.assertEqual(device, "fake-device:7")

    def test_runtime_context_report_schema_is_preserved(self) -> None:
        context = TTNNDirectRuntimeContext(
            parameters=object(), prefill_token_ids=_Tensor((2, 8)), prefill_page_table=_Tensor((2, 1)),
            kv_cache=[object(), object()], tensor_conversion_count=7, parameter_source="real_model", input_source="prompt",
            prefill_tokenization={"effective_token_count": 8}, prefill_page_table_runtime_state={"page_count": 1},
            kv_cache_runtime_state={"layers": 2}, prefill_prompt_runtime_input_tensor_count=1,
            prefill_page_table_runtime_input_tensor_count=1, prefill_rotary_runtime_input_tensor_count=3,
            parameter_setup={"tensorized_role_count": 5}, tokenizer_path=Path("/tmp/tokenizer"), tokenizer_module=None,
        )
        context.install_generated_model(generated_module=NS(), generated_model=object())
        context.install_decode_runtime(NS(page_table=_Tensor((2, 1)), cache_position=_Tensor((2,)), decode_runtime_state={"cache_len": 16}, rotary_runtime_state={"rotary": "ready"}))
        context.update_decode_token(_Tensor((2, 1))); context.update_kv_cache([object()])
        report = context.to_report(decode_step_count=3)
        expected = {"class": "TTNNDirectRuntimeContext", "status": "built", "parameter_source": "real_model", "current_kv_cache_layers": 1, "prefill_page_table_shape": [2, 1], "current_page_table_shape": [2, 1], "current_cache_position_shape": [2], "decode_step_count": 3, "generated_model_initialization_count": 1}
        self.assertEqual({key: report[key] for key in expected}, expected)

    def test_generate_section_profiler_records_wrapped_model_sections(self) -> None:
        ttnn, model = _SyncTTNN(), _GeneratedModel()
        profiler = GenerateSectionProfiler(ttnn=ttnn, device=object()); profiler.install(model)
        self.assertEqual((model.prefill_prompt(), model.decode_step()), ("prefilled", "decoded"))
        report = profiler.to_report(host_copy_profile={"total_ms": "1.25"})
        self.assertEqual((report["status"], report["basis"], report["host_copy_ms"]), ("measured", "generated model method wrappers", 1.25))
        self.assertGreater(ttnn.sync_count, 0)
        self.assertEqual((report["prefill_layer_profiles"][0]["layer_id"], report["decode_layer_profiles"][0]["layer_id"]), (0, 0))
        self.assertGreaterEqual(report["sections_ms"]["argmax_ms"], 0.0)

    def test_runtime_tokenizer_module_preserves_reports(self) -> None:
        prefill = tokenize_prompt_for_prefill(prompt="hello ttnn direct", batch_size=2, prefill_len=5, tokenizer_path="/tmp/tokenizer", vocab_size=128, tokenizer_module=_TokenizerModule)
        self.assertEqual((prefill.selected_token_id, prefill.token_ids, prefill.to_report()["source"]), (6, [[5, 4, 6, 0, 0]] * 2, "prompt_tokenizer_prefill"))
        decode = tokenize_prompt_for_decode(prompt="hello ttnn direct", batch_size=2, tokenizer_path="/tmp/tokenizer", vocab_size=128, tokenizer_module=_TokenizerModule)
        self.assertEqual(decode.token_ids, [[6], [6]])
        text = detokenize_generated_token_ids(token_ids_by_user=[[4, 6], [5]], tokenizer_path="/tmp/tokenizer", tokenizer_module=_TokenizerModule)
        self.assertEqual((text["status"], text["generated_text_by_user"]), ("decoded", ["tok4 tok6", "tok5"]))

    def test_runtime_inputs_module_preserves_reports(self) -> None:
        state = build_decode_runtime_state(batch_size=2, cache_len=10, page_block_size=4, prompt_token_count=6)
        self.assertEqual((state.page_count, state.max_num_blocks, state.cache_position, state.page_table), (3, 6, [5, 5], [[0, 1, 2], [3, 4, 5]]))
        rotary = build_decode_rotary_runtime_state(layer_count=2, batch_size=2, head_dim=64, cache_position_value=5)
        self.assertEqual((rotary.tensor_count, rotary.to_report()["source"]), (6, "rotary_runtime_state"))
        kv = build_decode_kv_cache_runtime_state(layer_count=2, batch_size=2, cache_len=10, page_block_size=4, num_kv_heads=8, head_dim=64)
        self.assertEqual((kv.physical_shape, kv.logical_shape), ([6, 8, 4, 64], [2, 10, 8, 64]))

    def test_runtime_prefill_builds_page_table_tensor_report(self) -> None:
        result = build_prefill_page_table_tensor(ttnn=_TTNN(), torch=_Torch(), device="device0", batch_size=2, cache_len=10, page_block_size=4, prompt_token_count=6)
        table, state = result.page_table, result.prefill_page_table_runtime_state
        self.assertEqual((result.tensor_conversion_count, table.source, table.shape, table.values), (1, "prefill_page_table", [2, 3], [[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(table.kwargs, {"device": "device0", "dtype": "ttnn.int32", "layout": "row_major"})
        self.assertEqual((state["source"], state["page_count"], state["cache_position_value"]), ("prefill_page_table_runtime_state", 3, 5))

    def test_runtime_prefill_token_ids_tensor_uses_runtime_int_tensor(self) -> None:
        tensor = prefill_token_ids_tensor(ttnn=_TTNN(), torch=_Torch(), device="device0", token_ids=[[11, 12], [13, 14]])
        self.assertEqual((tensor.source, tensor.shape, tensor.values, tensor.kwargs), ("prefill_prompt_token_ids", [2, 2], [[11, 12], [13, 14]], {"device": "device0", "dtype": "ttnn.uint32", "layout": "row_major"}))

    def test_runtime_prefill_attaches_rotary_parameters(self) -> None:
        ttnn, parameters = _TTNN(), NS(layers=[NS(), NS()])
        result = attach_prefill_rotary_parameters(parameters=parameters, ttnn=ttnn, torch=_Torch(), device="device0", dtype_seed="bf16", plan={
            "layers": 2, "prefill_len": 128, "rotary": {"theta": 500000.0, "scaling": None},
            "layer_parameter_shapes": {"rotary_cos_matrix": [1, 1, 128, 128], "rotary_sin_matrix": [1, 1, 128, 128], "rotary_transformation_matrix": [1, 1, 32, 32]},
        })
        rotary = parameters.layers[0].attention.rotary
        self.assertEqual((result.tensor_conversion_count, result.rotary_runtime_state["source"], len(ttnn.from_torch_calls)), (3, "hf_rope_config", 3))
        self.assertEqual((rotary.cos_matrix.source, rotary.sin_matrix.source, rotary.transformation_matrix.source), ("runtime.shared.prefill_rotary_cos", "runtime.shared.prefill_rotary_sin", "runtime.shared.prefill_rotary_transform"))
        self.assertEqual((rotary.cos_matrix.shape, rotary.cos_matrix.kwargs), ([1, 1, 128, 128], {"device": "device0", "dtype": "ttnn.bfloat16", "layout": "tile"}))
        self.assertIs(parameters.layers[1].attention.rotary, rotary)

    def test_runtime_prefill_prompt_helper_updates_context_and_event(self) -> None:
        model = _PrefillModel()
        context = NS(generated_model=model, prefill_token_ids="prefill-token-ids", kv_cache=["old-cache"], prefill_page_table="prefill-page-table", prefill_tokenization={"effective_token_count": 4})
        context.update_kv_cache = lambda value: setattr(context, "kv_cache", value)
        context.update_decode_token = lambda value: setattr(context, "token_ids", value)
        with patch.object(runtime_prefill, "_observed_cache_population", return_value=[{"layer_id": 0, "status": "filled"}]), patch.object(runtime_prefill, "_prefill_reference", return_value={"status": "passed", "passed": True}), patch.object(runtime_prefill, "_generated_observed_op_sequence", return_value=["prefill_prompt"]):
            result = run_prefill_prompt(context=context, ttnn=_SyncTTNN(), device="device0", prefill_plan={"layers": 1}, layer_count=1)
        self.assertEqual(model.prefill_args, ("prefill-token-ids", ["old-cache"], "prefill-page-table", 4))
        self.assertIs(context.kv_cache, model.kv_cache); self.assertIs(context.token_ids, model.prefill_token)
        self.assertEqual((result.output_shapes["token"], result.cache_population), ([2, 1], [{"layer_id": 0, "status": "filled"}]))
        self.assertEqual(result.reference, {"status": "passed", "passed": True, "observed_ops_source": "runtime_instrumentation"})
        self.assertEqual(result.first_token.runtime_handoff, "device_tensor_direct")
        self.assertEqual(result.generated_token_events[0], {"step_index": "prefill", "token": model.prefill_token, "runtime_handoff": "device_tensor_direct", "runtime_host_roundtrip": False, "cache_position_value": 3, "token_shape": [2, 1]})

    def test_runtime_kv_cache_builds_paged_dram_cache_tensors(self) -> None:
        result = build_prompt_decode_kv_cache_tensors(ttnn=_TTNN(), torch=_Torch(), device="device0", dtype_seed="bf16", layer_count=2, batch_size=2, cache_len=10, page_block_size=4, num_kv_heads=8, head_dim=64)
        cache, state = result.kv_cache[0], result.kv_cache_runtime_state
        self.assertEqual((result.tensor_conversion_count, len(result.kv_cache), cache.k.source, cache.v.source, cache.k.shape), (4, 2, "runtime.layers.0.key_cache", "runtime.layers.0.value_cache", [6, 8, 4, 64]))
        self.assertEqual(cache.k.kwargs, {"device": "device0", "dtype": "ttnn.bfloat16", "memory_config": "dram_memory", "layout": "tile"})
        self.assertEqual({key: state[key] for key in ("source", "physical_shape", "logical_shape", "memory_config", "ttnn_memory_config")}, {"source": "kv_cache_runtime_state", "physical_shape": [6, 8, 4, 64], "logical_shape": [2, 10, 8, 64], "memory_config": "dram", "ttnn_memory_config": "dram_memory"})

    def test_runtime_reports_helpers_preserve_generate_report_fields(self) -> None:
        budget = runtime_reports.generated_token_budget(max_new_tokens=3, decode_steps=2)
        self.assertEqual((budget["prefill_first_token_count"], budget["total_planned_generated_tokens"]), (1, 3))
        cache = runtime_reports.cache_population_summary([{"status": "passed", "layer_id": 0, "write_policy": "paged_fill_cache_per_user", "update_shape_layout": "tile", "key_cache_shape": [2, 4], "value_cache_shape": [2, 4], "filled_user_count": 2, "planned_user_count": 2}])
        self.assertEqual((cache["status_counts"], cache["write_policies"], cache["filled_user_count_total"]), ({"passed": 1}, ["paged_fill_cache_per_user"], 2))
        host = runtime_reports.host_copy_profile(first_token_materialization_ms=1.5, step_reports=[{"token_materialization_ms": "2.5"}, {"token_materialization_ms": None}])
        self.assertEqual((host["status"], host["total_ms"], host["runtime_host_roundtrip_present"]), ("measured", 4.0, False))
        throughput = runtime_reports.generate_throughput_summary(latency_ms=100.0, batch_size=2, max_new_tokens=3)
        self.assertEqual((throughput["status"], throughput["tokens_per_second_per_user"], throughput["aggregate_tokens_per_second"]), ("measured", 30.0, 60.0))
        self.assertIn("argmax_ms", runtime_reports.section_profile_not_run("dry_run")["sections_ms"])
        dry = runtime_reports.generate_dry_run_report(program_dir=Path("/tmp/program"), program_num_layers=1, layers=1, max_new_tokens=2, decode_steps=1, prefill_len=8, device="p150a", device_id=0, batch_size=2, cache_len=16, dtype_seed="bf16", decode_plan={"tensor_conversion_count": 5, "op_sequence": ["decode"]}, prefill_plan={"layers": 1, "batch_size": 2, "tensor_conversion_count": 7, "op_sequence": ["prefill"], "expected_output_shapes": {"key_cache": [2, 8, 16], "value_cache": [2, 8, 16]}})
        self.assertEqual((dry["status"], dry["passed"], dry["tensor_conversion_count"], dry["prefill_cache_population_summary"]["layer_count"], dry["end_to_end_contract"]["passed"]), ("dry_run", True, 12, 1, True))

    def test_runtime_decode_helpers_preserve_token_handoff_reports(self) -> None:
        token = [[11], [12]]
        handoff = prefill_token_direct_handoff(prefill_token=token)
        self.assertEqual((handoff.status, handoff.token_ids, handoff.runtime_host_roundtrip), ("device_tensor_direct", token, False))
        steps = [{"step_index": 0, "passed": True}]
        materialized = materialize_generate_token_events([
            {"step_index": "prefill", "token": token, "runtime_handoff": "device_tensor_direct", "runtime_host_roundtrip": False, "cache_position_value": 2, "page_table_shape": [2, 1], "token_shape": [2, 1]},
            {"step_index": 0, "token": [[13], [14]], "runtime_handoff": "device_tensor_direct", "runtime_host_roundtrip": False, "cache_position_value": 3, "page_table_shape": [2, 1], "token_shape": [2, 1]},
        ], step_reports=steps, ttnn=NS(), batch_size=2)
        self.assertEqual((materialized.generated_token_ids_by_user, materialized.first_token_ids_by_user, steps[0]["generated_token_ids"], steps[0]["token_materialization"]["source"]), ([[11, 13], [12, 14]], [[11], [12]], [[13], [14]], "tensor_value"))

    def test_runtime_decode_builder_composes_runtime_state_helpers(self) -> None:
        def state(**kwargs):
            self.assertEqual((kwargs["page_block_size"], kwargs["prompt_token_count"]), (4, 8))
            return NS(page_table="page-table", cache_position="cache-position", decode_runtime_state={"cache_position_value": 7}, tensor_conversion_count=2)

        def rotary(**kwargs):
            self.assertEqual(kwargs["cache_position_value"], 7)
            return NS(rotary_runtime_state={"rotary": "ready"}, tensor_conversion_count=3)

        with patch.object(runtime_decode, "_build_prompt_decode_runtime_state_tensors", side_effect=state), patch.object(runtime_decode, "attach_decode_rotary_parameters", side_effect=rotary):
            result = build_decode_runtime_for_position(ttnn=object(), torch=object(), device=object(), dtype_seed="bf16", parameters=object(), decode_plan={"kv_cache": {"page_block_size": 4}}, batch_size=2, cache_len=16, prefill_effective_token_count=6, generated_token_index=1)
        self.assertEqual((result.page_table, result.cache_position, result.rotary_runtime_state, result.tensor_conversion_count, result.decode_runtime_state_input_tensor_count, result.rotary_runtime_input_tensor_count), ("page-table", "cache-position", {"rotary": "ready"}, 5, 2, 3))

    def test_runtime_decode_loop_updates_context_and_counts_runtime_tensors(self) -> None:
        indexes, tokens = [], [_Tensor((2, 1)), _Tensor((2, 1))]

        def build(**kwargs):
            index = int(kwargs["generated_token_index"]); indexes.append(index)
            return NS(page_table=f"page-table-{index}", cache_position=f"cache-position-{index}", decode_runtime_state={"cache_position_value": 5 + index}, rotary_runtime_state={"rotary_index": index}, tensor_conversion_count=5, decode_runtime_state_input_tensor_count=2, rotary_runtime_input_tensor_count=3)

        def decode(**_kwargs):
            index = len(indexes) - 1
            return tokens[index], [NS(k=_Tensor((1,)), v=_Tensor((1,)))], 1.25 + index

        context = NS(parameters=object(), prefill_tokenization={"effective_token_count": 4}, generated_model=object(), token_ids="prefill-token", page_table=None, cache_position=None, kv_cache=["initial-cache"])
        context.install_decode_runtime = lambda state: vars(context).update(page_table=state.page_table, cache_position=state.cache_position, decode_runtime_state=state.decode_runtime_state, rotary_state=state.rotary_runtime_state)
        context.update_kv_cache = lambda value: setattr(context, "kv_cache", value)
        context.update_decode_token = lambda value: setattr(context, "token_ids", value)
        with patch.object(runtime_decode, "build_decode_runtime_for_position", side_effect=build), patch.object(runtime_decode, "_time_decode_step", side_effect=decode), patch.object(runtime_decode, "_loop_input_shapes", return_value={"page_table": [2, 1]}), patch.object(runtime_decode, "_loop_output_shapes", return_value={"token": [2, 1]}), patch.object(runtime_decode, "_decode_step_reference", return_value={"status": "passed", "passed": True}), patch.object(runtime_decode, "_generated_observed_op_sequence", return_value=["decode_step"]):
            result = run_decode_loop(context=context, ttnn=object(), torch=object(), device="device0", dtype_seed="bf16", decode_plan={"kv_cache": {"page_block_size": 4}}, batch_size=2, cache_len=16, layer_count=1, decode_step_count=2, generated_token_events=[{"step_index": "prefill", "token": "t0"}], initial_tensor_conversion_count=10)
        self.assertEqual((indexes, result.tensor_conversion_count, result.decode_runtime_state_input_tensor_count, result.decode_rotary_runtime_input_tensor_count), ([0, 1], 20, 4, 6))
        self.assertEqual((len(result.step_reports), [item["passed"] for item in result.step_reports]), (2, [True, True]))
        self.assertEqual([item["step_index"] for item in result.generated_token_events[1:]], [0, 1])
        self.assertIs(context.token_ids, tokens[-1])
        self.assertEqual(result.decode_runtime_state, {"cache_position_value": 6})
