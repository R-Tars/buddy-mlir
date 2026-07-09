from __future__ import annotations

import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.generate import (
    GenerateSectionProfiler as GenerateCompatProfiler,
    TTNNDirectRuntimeContext as GenerateCompatContext,
    run_generate,
    run_profile_generate,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.context import (
    TTNNDirectRuntimeContext,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime import (
    decode as runtime_decode,
    reports as runtime_reports,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.decode import (
    build_decode_runtime_for_position,
    materialize_generate_token_events,
    prefill_token_direct_handoff,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.generate import (
    build_generate_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.inputs import (
    build_decode_kv_cache_runtime_state,
    build_decode_rotary_runtime_state,
    build_decode_runtime_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.kv_cache import (
    build_prompt_decode_kv_cache_tensors,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.profile import (
    GenerateSectionProfiler,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.prefill import (
    attach_prefill_rotary_parameters,
    build_prefill_page_table_tensor,
    prefill_token_ids_tensor,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.tokenizer import (
    detokenize_generated_token_ids,
    tokenize_prompt_for_decode,
    tokenize_prompt_for_prefill,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime_inputs import (
    build_decode_runtime_state as compat_build_decode_runtime_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime_inputs import (
    tokenize_prompt_for_prefill as compat_tokenize_prompt_for_prefill,
)


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeTTNN:
    def __init__(self) -> None:
        self.sync_count = 0

    def synchronize_device(self, _device: object) -> None:
        self.sync_count += 1


class _FakeOps:
    def argmax(self, value: object) -> object:
        return value


class _FakeGeneratedModel:
    def __init__(self) -> None:
        self.ops = _FakeOps()

    def prefill_prompt(self) -> str:
        self.embed()
        self.attention_prefill(0)
        self.mlp_decode(0)
        return "prefilled"

    def decode_step(self) -> str:
        self.attention_decode(0)
        self.mlp_decode(0)
        self.final_norm()
        self.lm_head_argmax()
        return "decoded"

    def embed(self) -> str:
        return "embedded"

    def attention_prefill(self, _layer_id: int) -> str:
        return "prefill_attention"

    def attention_decode(self, _layer_id: int) -> str:
        return "decode_attention"

    def mlp_decode(self, _layer_id: int) -> str:
        return "mlp"

    def final_norm(self) -> str:
        return "normed"

    def lm_head_argmax(self) -> str:
        self.ops.argmax("logits")
        return "token"


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        _ = add_special_tokens
        return {"input_ids": [len(word) for word in prompt.split()]}

    def batch_decode(
        self,
        rows: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        _ = skip_special_tokens
        return [" ".join(f"tok{token_id}" for token_id in row) for row in rows]


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(_path: str) -> _FakeTokenizer:
        return _FakeTokenizer()


class _FakeTokenizerModule:
    AutoTokenizer = _FakeAutoTokenizer


class _FakeHostTensor:
    def __init__(
        self,
        values: object | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: object | None = None,
    ) -> None:
        self.values = values
        self.shape = shape if shape is not None else self._infer_shape(values)
        self.dtype = dtype
        self.name: str | None = None

    @classmethod
    def _infer_shape(cls, values: object | None) -> tuple[int, ...]:
        if isinstance(values, list):
            if values and isinstance(values[0], list):
                return (len(values), *cls._infer_shape(values[0]))
            return (len(values),)
        return ()


class _FakeTorchForPrefill:
    int32 = "torch.int32"
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"

    def tensor(
        self,
        values: object,
        dtype: object | None = None,
    ) -> _FakeHostTensor:
        return _FakeHostTensor(values, dtype=dtype)

    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> _FakeHostTensor:
        return _FakeHostTensor(shape=shape, dtype=dtype)

    def randn(
        self,
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> _FakeHostTensor:
        return _FakeHostTensor(shape=shape, dtype=dtype)


class _FakeTTNNForPrefill:
    int32 = "ttnn.int32"
    uint32 = "ttnn.uint32"
    bfloat16 = "ttnn.bfloat16"
    float32 = "ttnn.float32"
    ROW_MAJOR_LAYOUT = "row_major"
    TILE_LAYOUT = "tile"
    DRAM_MEMORY_CONFIG = "dram_memory"

    def __init__(self) -> None:
        self.from_torch_calls: list[tuple[_FakeHostTensor, dict[str, object]]] = []

    def from_torch(self, tensor: _FakeHostTensor, **kwargs: object) -> object:
        self.from_torch_calls.append((tensor, kwargs))
        return types.SimpleNamespace(
            source=tensor.name,
            shape=list(tensor.shape),
            dtype=tensor.dtype,
            values=tensor.values,
            kwargs=kwargs,
        )


class RuntimeModuleTest(unittest.TestCase):
    def test_generate_module_reexports_runtime_classes(self) -> None:
        self.assertIs(GenerateCompatContext, TTNNDirectRuntimeContext)
        self.assertIs(GenerateCompatProfiler, GenerateSectionProfiler)
        self.assertTrue(callable(run_generate))
        self.assertTrue(callable(run_profile_generate))
        self.assertTrue(callable(build_generate_state))

    def test_runtime_context_report_schema_is_preserved(self) -> None:
        context = TTNNDirectRuntimeContext(
            parameters=object(),
            prefill_token_ids=_FakeTensor((2, 8)),
            prefill_page_table=_FakeTensor((2, 1)),
            kv_cache=[object(), object()],
            tensor_conversion_count=7,
            parameter_source="real_model",
            input_source="prompt",
            prefill_tokenization={"effective_token_count": 8},
            prefill_page_table_runtime_state={"page_count": 1},
            kv_cache_runtime_state={"layers": 2},
            prefill_prompt_runtime_input_tensor_count=1,
            prefill_page_table_runtime_input_tensor_count=1,
            prefill_rotary_runtime_input_tensor_count=3,
            parameter_setup={"tensorized_role_count": 5},
            tokenizer_path=Path("/tmp/tokenizer"),
            tokenizer_module=None,
        )
        context.install_generated_model(
            generated_module=types.SimpleNamespace(),
            generated_model=object(),
        )
        context.install_decode_runtime(
            types.SimpleNamespace(
                page_table=_FakeTensor((2, 1)),
                cache_position=_FakeTensor((2,)),
                decode_runtime_state={"cache_len": 16},
                rotary_runtime_state={"rotary": "ready"},
            )
        )
        context.update_decode_token(_FakeTensor((2, 1)))
        context.update_kv_cache([object()])

        report = context.to_report(decode_step_count=3)

        self.assertEqual(report["class"], "TTNNDirectRuntimeContext")
        self.assertEqual(report["status"], "built")
        self.assertEqual(report["parameter_source"], "real_model")
        self.assertEqual(report["current_kv_cache_layers"], 1)
        self.assertEqual(report["prefill_page_table_shape"], [2, 1])
        self.assertEqual(report["current_page_table_shape"], [2, 1])
        self.assertEqual(report["current_cache_position_shape"], [2])
        self.assertEqual(report["decode_step_count"], 3)
        self.assertEqual(report["generated_model_initialization_count"], 1)

    def test_generate_section_profiler_records_wrapped_model_sections(self) -> None:
        ttnn = _FakeTTNN()
        model = _FakeGeneratedModel()
        profiler = GenerateSectionProfiler(ttnn=ttnn, device=object())

        profiler.install(model)
        self.assertEqual(model.prefill_prompt(), "prefilled")
        self.assertEqual(model.decode_step(), "decoded")
        report = profiler.to_report(host_copy_profile={"total_ms": "1.25"})

        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["basis"], "generated model method wrappers")
        self.assertEqual(report["host_copy_ms"], 1.25)
        self.assertGreater(ttnn.sync_count, 0)
        self.assertEqual(report["prefill_layer_profiles"][0]["layer_id"], 0)
        self.assertEqual(report["decode_layer_profiles"][0]["layer_id"], 0)
        self.assertGreaterEqual(report["sections_ms"]["argmax_ms"], 0.0)

    def test_runtime_tokenizer_module_preserves_reports_and_compat_imports(self) -> None:
        tokenization = tokenize_prompt_for_prefill(
            prompt="hello ttnn direct",
            batch_size=2,
            prefill_len=5,
            tokenizer_path="/tmp/tokenizer",
            vocab_size=128,
            tokenizer_module=_FakeTokenizerModule,
        )

        self.assertIs(compat_tokenize_prompt_for_prefill, tokenize_prompt_for_prefill)
        self.assertEqual(tokenization.selected_token_id, 6)
        self.assertEqual(tokenization.token_ids, [[5, 4, 6, 0, 0], [5, 4, 6, 0, 0]])
        self.assertEqual(tokenization.to_report()["source"], "prompt_tokenizer_prefill")

        decode_tokenization = tokenize_prompt_for_decode(
            prompt="hello ttnn direct",
            batch_size=2,
            tokenizer_path="/tmp/tokenizer",
            vocab_size=128,
            tokenizer_module=_FakeTokenizerModule,
        )
        self.assertEqual(decode_tokenization.token_ids, [[6], [6]])

        text = detokenize_generated_token_ids(
            token_ids_by_user=[[4, 6], [5]],
            tokenizer_path="/tmp/tokenizer",
            tokenizer_module=_FakeTokenizerModule,
        )
        self.assertEqual(text["status"], "decoded")
        self.assertEqual(text["generated_text_by_user"], ["tok4 tok6", "tok5"])

    def test_runtime_inputs_module_preserves_reports_and_compat_imports(self) -> None:
        runtime_state = build_decode_runtime_state(
            batch_size=2,
            cache_len=10,
            page_block_size=4,
            prompt_token_count=6,
        )
        self.assertIs(compat_build_decode_runtime_state, build_decode_runtime_state)
        self.assertEqual(runtime_state.page_count, 3)
        self.assertEqual(runtime_state.max_num_blocks, 6)
        self.assertEqual(runtime_state.cache_position, [5, 5])
        self.assertEqual(runtime_state.page_table, [[0, 1, 2], [3, 4, 5]])

        rotary_state = build_decode_rotary_runtime_state(
            layer_count=2,
            batch_size=2,
            head_dim=64,
            cache_position_value=5,
        )
        self.assertEqual(rotary_state.tensor_count, 6)
        self.assertEqual(rotary_state.to_report()["source"], "rotary_runtime_state")

        kv_state = build_decode_kv_cache_runtime_state(
            layer_count=2,
            batch_size=2,
            cache_len=10,
            page_block_size=4,
            num_kv_heads=8,
            head_dim=64,
        )
        self.assertEqual(kv_state.physical_shape, [6, 8, 4, 64])
        self.assertEqual(kv_state.logical_shape, [2, 10, 8, 64])

    def test_runtime_prefill_builds_page_table_tensor_report(self) -> None:
        ttnn = _FakeTTNNForPrefill()
        result = build_prefill_page_table_tensor(
            ttnn=ttnn,
            torch=_FakeTorchForPrefill(),
            device="device0",
            batch_size=2,
            cache_len=10,
            page_block_size=4,
            prompt_token_count=6,
        )

        self.assertEqual(result.tensor_conversion_count, 1)
        self.assertEqual(result.page_table.source, "prefill_page_table")
        self.assertEqual(result.page_table.shape, [2, 3])
        self.assertEqual(result.page_table.values, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(
            result.page_table.kwargs,
            {"device": "device0", "dtype": "ttnn.int32", "layout": "row_major"},
        )
        self.assertEqual(
            result.prefill_page_table_runtime_state["source"],
            "prefill_page_table_runtime_state",
        )
        self.assertEqual(result.prefill_page_table_runtime_state["page_count"], 3)
        self.assertEqual(
            result.prefill_page_table_runtime_state["cache_position_value"],
            5,
        )

    def test_runtime_prefill_token_ids_tensor_uses_runtime_int_tensor(self) -> None:
        ttnn = _FakeTTNNForPrefill()
        tensor = prefill_token_ids_tensor(
            ttnn=ttnn,
            torch=_FakeTorchForPrefill(),
            device="device0",
            token_ids=[[11, 12], [13, 14]],
        )

        self.assertEqual(tensor.source, "prefill_prompt_token_ids")
        self.assertEqual(tensor.shape, [2, 2])
        self.assertEqual(tensor.values, [[11, 12], [13, 14]])
        self.assertEqual(
            tensor.kwargs,
            {"device": "device0", "dtype": "ttnn.uint32", "layout": "row_major"},
        )

    def test_runtime_prefill_attaches_rotary_parameters(self) -> None:
        ttnn = _FakeTTNNForPrefill()
        parameters = types.SimpleNamespace(
            layers=[types.SimpleNamespace(), types.SimpleNamespace()],
        )

        result = attach_prefill_rotary_parameters(
            parameters=parameters,
            ttnn=ttnn,
            torch=_FakeTorchForPrefill(),
            device="device0",
            dtype_seed="bf16",
            prefill_plan={
                "layers": 2,
                "layer_parameter_shapes": {
                    "rotary_cos_matrix": [1, 32, 128, 128],
                    "rotary_sin_matrix": [1, 32, 128, 128],
                    "rotary_transformation_matrix": [1, 1, 32, 32],
                },
            },
        )

        self.assertEqual(result.tensor_conversion_count, 6)
        first_rotary = parameters.layers[0].attention.rotary
        self.assertEqual(first_rotary.cos_matrix.source, "prefill.layers.0.rotary_cos")
        self.assertEqual(first_rotary.sin_matrix.source, "prefill.layers.0.rotary_sin")
        self.assertEqual(
            first_rotary.transformation_matrix.source,
            "prefill.layers.0.rotary_transform",
        )
        self.assertEqual(first_rotary.cos_matrix.shape, [1, 32, 128, 128])
        self.assertEqual(
            first_rotary.cos_matrix.kwargs,
            {"device": "device0", "dtype": "ttnn.bfloat16", "layout": "tile"},
        )
        second_rotary = parameters.layers[1].attention.rotary
        self.assertEqual(second_rotary.cos_matrix.source, "prefill.layers.1.rotary_cos")
        self.assertEqual(len(ttnn.from_torch_calls), 6)

    def test_runtime_kv_cache_builds_paged_dram_cache_tensors(self) -> None:
        ttnn = _FakeTTNNForPrefill()
        result = build_prompt_decode_kv_cache_tensors(
            ttnn=ttnn,
            torch=_FakeTorchForPrefill(),
            device="device0",
            dtype_seed="bf16",
            layer_count=2,
            batch_size=2,
            cache_len=10,
            page_block_size=4,
            num_kv_heads=8,
            head_dim=64,
        )

        self.assertEqual(result.tensor_conversion_count, 4)
        self.assertEqual(len(result.kv_cache), 2)
        self.assertEqual(result.kv_cache[0].k.source, "runtime.layers.0.key_cache")
        self.assertEqual(result.kv_cache[0].v.source, "runtime.layers.0.value_cache")
        self.assertEqual(result.kv_cache[0].k.shape, [6, 8, 4, 64])
        self.assertEqual(
            result.kv_cache[0].k.kwargs,
            {
                "device": "device0",
                "dtype": "ttnn.bfloat16",
                "memory_config": "dram_memory",
                "layout": "tile",
            },
        )
        self.assertEqual(result.kv_cache_runtime_state["source"], "kv_cache_runtime_state")
        self.assertEqual(result.kv_cache_runtime_state["physical_shape"], [6, 8, 4, 64])
        self.assertEqual(result.kv_cache_runtime_state["logical_shape"], [2, 10, 8, 64])
        self.assertEqual(result.kv_cache_runtime_state["memory_config"], "dram")
        self.assertEqual(
            result.kv_cache_runtime_state["ttnn_memory_config"],
            "dram_memory",
        )

    def test_runtime_reports_helpers_preserve_generate_report_fields(self) -> None:
        budget = runtime_reports.generated_token_budget(
            max_new_tokens=3,
            decode_steps=2,
        )
        self.assertEqual(budget["prefill_first_token_count"], 1)
        self.assertEqual(budget["total_planned_generated_tokens"], 3)

        cache_summary = runtime_reports.cache_population_summary(
            [
                {
                    "status": "passed",
                    "layer_id": 0,
                    "write_policy": "paged_fill_cache_per_user",
                    "update_shape_layout": "tile",
                    "key_cache_shape": [2, 4],
                    "value_cache_shape": [2, 4],
                    "filled_user_count": 2,
                    "planned_user_count": 2,
                }
            ]
        )
        self.assertEqual(cache_summary["status_counts"], {"passed": 1})
        self.assertEqual(
            cache_summary["write_policies"],
            ["paged_fill_cache_per_user"],
        )
        self.assertEqual(cache_summary["filled_user_count_total"], 2)

        host_copy = runtime_reports.host_copy_profile(
            first_token_materialization_ms=1.5,
            step_reports=[
                {"token_materialization_ms": "2.5"},
                {"token_materialization_ms": None},
            ],
        )
        self.assertEqual(host_copy["status"], "measured")
        self.assertEqual(host_copy["total_ms"], 4.0)
        self.assertFalse(host_copy["runtime_host_roundtrip_present"])

        throughput = runtime_reports.generate_throughput_summary(
            latency_ms=100.0,
            batch_size=2,
            max_new_tokens=3,
        )
        self.assertEqual(throughput["status"], "measured")
        self.assertEqual(throughput["tokens_per_second_per_user"], 30.0)
        self.assertEqual(throughput["aggregate_tokens_per_second"], 60.0)

        fallback_profile = runtime_reports.section_profile_not_run("dry_run")
        self.assertEqual(fallback_profile["status"], "not_run")
        self.assertIn("argmax_ms", fallback_profile["sections_ms"])

    def test_runtime_decode_helpers_preserve_token_handoff_reports(self) -> None:
        prefill_token = [[11], [12]]
        handoff = prefill_token_direct_handoff(prefill_token=prefill_token)
        self.assertEqual(handoff.status, "device_tensor_direct")
        self.assertIs(handoff.token_ids, prefill_token)
        self.assertFalse(handoff.runtime_host_roundtrip)

        step_reports = [{"step_index": 0, "passed": True}]
        materialized = materialize_generate_token_events(
            [
                {
                    "step_index": "prefill",
                    "token": [[11], [12]],
                    "runtime_handoff": "device_tensor_direct",
                    "runtime_host_roundtrip": False,
                    "cache_position_value": 2,
                    "page_table_shape": [2, 1],
                    "token_shape": [2, 1],
                },
                {
                    "step_index": 0,
                    "token": [[13], [14]],
                    "runtime_handoff": "device_tensor_direct",
                    "runtime_host_roundtrip": False,
                    "cache_position_value": 3,
                    "page_table_shape": [2, 1],
                    "token_shape": [2, 1],
                },
            ],
            step_reports=step_reports,
            ttnn=types.SimpleNamespace(),
            batch_size=2,
        )

        self.assertEqual(materialized.generated_token_ids_by_user, [[11, 13], [12, 14]])
        self.assertEqual(materialized.first_token_ids_by_user, [[11], [12]])
        self.assertEqual(step_reports[0]["generated_token_ids"], [[13], [14]])
        self.assertEqual(
            step_reports[0]["token_materialization"]["source"],
            "tensor_value",
        )

    def test_runtime_decode_builder_composes_runtime_state_helpers(self) -> None:
        original_runtime_builder = (
            runtime_decode._build_prompt_decode_runtime_state_tensors
        )
        original_rotary_builder = runtime_decode._attach_runtime_rotary_parameters

        def fake_runtime_builder(**kwargs: object) -> types.SimpleNamespace:
            self.assertEqual(kwargs["page_block_size"], 4)
            self.assertEqual(kwargs["prompt_token_count"], 8)
            return types.SimpleNamespace(
                page_table="page-table",
                cache_position="cache-position",
                decode_runtime_state={"cache_position_value": 7},
                tensor_conversion_count=2,
            )

        def fake_rotary_builder(**kwargs: object) -> types.SimpleNamespace:
            self.assertEqual(kwargs["cache_position_value"], 7)
            return types.SimpleNamespace(
                rotary_runtime_state={"rotary": "ready"},
                tensor_conversion_count=3,
            )

        try:
            runtime_decode._build_prompt_decode_runtime_state_tensors = fake_runtime_builder
            runtime_decode._attach_runtime_rotary_parameters = fake_rotary_builder
            runtime_state = build_decode_runtime_for_position(
                ttnn=object(),
                torch=object(),
                device=object(),
                dtype_seed="bf16",
                parameters=object(),
                decode_plan={"kv_cache": {"page_block_size": 4}},
                batch_size=2,
                cache_len=16,
                prefill_effective_token_count=6,
                generated_token_index=1,
            )
        finally:
            runtime_decode._build_prompt_decode_runtime_state_tensors = (
                original_runtime_builder
            )
            runtime_decode._attach_runtime_rotary_parameters = original_rotary_builder

        self.assertEqual(runtime_state.page_table, "page-table")
        self.assertEqual(runtime_state.cache_position, "cache-position")
        self.assertEqual(runtime_state.rotary_runtime_state, {"rotary": "ready"})
        self.assertEqual(runtime_state.tensor_conversion_count, 5)
        self.assertEqual(runtime_state.decode_runtime_state_input_tensor_count, 2)
        self.assertEqual(runtime_state.rotary_runtime_input_tensor_count, 3)


if __name__ == "__main__":
    unittest.main()
