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
    prefill as runtime_prefill,
    reports as runtime_reports,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.decode import (
    _normalize_teacher_forcing,
    _teacher_forcing_token_tensor,
    build_decode_runtime_for_position,
    materialize_generate_token_events,
    prefill_token_direct_handoff,
    run_decode_loop,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.device import (
    GenerateDeviceSession,
    maybe_generate_device,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.generate import (
    build_generate_state as compat_build_generate_state,
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
    run_prefill_prompt,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.state import (
    GENERATE_RUNTIME_OWNER,
    build_generate_state,
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


class _FakePrefillModel:
    def __init__(self) -> None:
        self.prefill_args: tuple[object, ...] | None = None
        self.prefill_token = _FakeTensor((2, 1))
        self.kv_cache = [
            types.SimpleNamespace(k=_FakeTensor((4, 8)), v=_FakeTensor((4, 8)))
        ]
        self.cache_reports = [{"layer_id": 0, "status": "filled"}]

    def prefill_prompt(
        self,
        prefill_token_ids: object,
        kv_cache: object,
        prefill_page_table: object,
        *,
        valid_seq_len: int | None = None,
    ) -> tuple[object, list[object], list[dict[str, object]]]:
        self.prefill_args = (
            prefill_token_ids,
            kv_cache,
            prefill_page_table,
            valid_seq_len,
        )
        return self.prefill_token, self.kv_cache, self.cache_reports


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self, prompt: str, add_special_tokens: bool = True
    ) -> dict[str, list[int]]:
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
    def test_teacher_forcing_validates_and_builds_batch_token_tensor(self) -> None:
        normalized = _normalize_teacher_forcing(
            [[11, 12], [13, 14]],
            decode_step_count=2,
            batch_size=2,
        )
        self.assertEqual(normalized, [[11, 12], [13, 14]])
        with self.assertRaisesRegex(ValueError, "batch width"):
            _normalize_teacher_forcing(
                [[11]],
                decode_step_count=1,
                batch_size=2,
            )

        ttnn = _FakeTTNNForPrefill()
        token = _teacher_forcing_token_tensor(
            ttnn=ttnn,
            torch=_FakeTorchForPrefill(),
            device="device0",
            token_ids=[11, 12],
            step_index=3,
        )
        self.assertEqual(token.source, "teacher_forcing_token_ids_3")
        self.assertEqual(token.values, [[11], [12]])
        self.assertEqual(token.shape, [2, 1])

    def test_generate_module_reexports_runtime_classes(self) -> None:
        self.assertIs(GenerateCompatContext, TTNNDirectRuntimeContext)
        self.assertIs(GenerateCompatProfiler, GenerateSectionProfiler)
        self.assertTrue(callable(run_generate))
        self.assertTrue(callable(run_profile_generate))
        self.assertIs(compat_build_generate_state, build_generate_state)
        self.assertTrue(callable(build_generate_state))
        self.assertEqual(GENERATE_RUNTIME_OWNER, "TTNNDirectRuntimeContext")

    def test_generate_device_session_uses_injected_device(self) -> None:
        session = maybe_generate_device(
            ttnn=object(),
            device_id=7,
            injected=object(),
        )

        self.assertIsInstance(session, GenerateDeviceSession)
        with session as device:
            self.assertEqual(device, "fake-device:7")

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

    def test_runtime_tokenizer_module_preserves_reports_and_compat_imports(
        self,
    ) -> None:
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
            plan={
                "layers": 2,
                "prefill_len": 128,
                "rotary": {
                    "theta": 500000.0,
                    "scaling": None,
                },
                "layer_parameter_shapes": {
                    "rotary_cos_matrix": [1, 1, 128, 128],
                    "rotary_sin_matrix": [1, 1, 128, 128],
                    "rotary_transformation_matrix": [1, 1, 32, 32],
                },
            },
        )

        self.assertEqual(result.tensor_conversion_count, 3)
        self.assertEqual(result.rotary_runtime_state["source"], "hf_rope_config")
        first_rotary = parameters.layers[0].attention.rotary
        self.assertEqual(
            first_rotary.cos_matrix.source,
            "runtime.shared.prefill_rotary_cos",
        )
        self.assertEqual(
            first_rotary.sin_matrix.source,
            "runtime.shared.prefill_rotary_sin",
        )
        self.assertEqual(
            first_rotary.transformation_matrix.source,
            "runtime.shared.prefill_rotary_transform",
        )
        self.assertEqual(first_rotary.cos_matrix.shape, [1, 1, 128, 128])
        self.assertEqual(
            first_rotary.cos_matrix.kwargs,
            {"device": "device0", "dtype": "ttnn.bfloat16", "layout": "tile"},
        )
        second_rotary = parameters.layers[1].attention.rotary
        self.assertIs(second_rotary, first_rotary)
        self.assertEqual(len(ttnn.from_torch_calls), 3)

    def test_runtime_prefill_prompt_helper_updates_context_and_event(self) -> None:
        original_cache_population = runtime_prefill._observed_cache_population
        original_prefill_reference = runtime_prefill._prefill_reference
        original_observed_ops = runtime_prefill._generated_observed_op_sequence
        model = _FakePrefillModel()
        context = types.SimpleNamespace(
            generated_model=model,
            prefill_token_ids="prefill-token-ids",
            kv_cache=["old-cache"],
            prefill_page_table="prefill-page-table",
            prefill_tokenization={"effective_token_count": 4},
        )

        def update_kv_cache(kv_cache: object) -> None:
            context.kv_cache = kv_cache

        def update_decode_token(token_ids: object) -> None:
            context.token_ids = token_ids

        context.update_kv_cache = update_kv_cache
        context.update_decode_token = update_decode_token

        try:
            runtime_prefill._observed_cache_population = lambda **_: [
                {"layer_id": 0, "status": "filled"}
            ]
            runtime_prefill._prefill_reference = lambda **_: {
                "status": "passed",
                "passed": True,
            }
            runtime_prefill._generated_observed_op_sequence = lambda *_: [
                "prefill_prompt"
            ]
            result = run_prefill_prompt(
                context=context,
                ttnn=_FakeTTNN(),
                device="device0",
                prefill_plan={"layers": 1},
                layer_count=1,
            )
        finally:
            runtime_prefill._observed_cache_population = original_cache_population
            runtime_prefill._prefill_reference = original_prefill_reference
            runtime_prefill._generated_observed_op_sequence = original_observed_ops

        self.assertEqual(
            model.prefill_args,
            (
                "prefill-token-ids",
                ["old-cache"],
                "prefill-page-table",
                4,
            ),
        )
        self.assertIs(context.kv_cache, model.kv_cache)
        self.assertIs(context.token_ids, model.prefill_token)
        self.assertEqual(result.output_shapes["token"], [2, 1])
        self.assertEqual(result.cache_population, [{"layer_id": 0, "status": "filled"}])
        self.assertEqual(
            result.reference,
            {
                "status": "passed",
                "passed": True,
                "observed_ops_source": "runtime_instrumentation",
            },
        )
        self.assertEqual(result.first_token.runtime_handoff, "device_tensor_direct")
        self.assertEqual(
            result.generated_token_events,
            [
                {
                    "step_index": "prefill",
                    "token": model.prefill_token,
                    "runtime_handoff": "device_tensor_direct",
                    "runtime_host_roundtrip": False,
                    "cache_position_value": 3,
                    "token_shape": [2, 1],
                }
            ],
        )

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
        self.assertEqual(
            result.kv_cache_runtime_state["source"], "kv_cache_runtime_state"
        )
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

        dry_run_report = runtime_reports.generate_dry_run_report(
            program_dir=Path("/tmp/program"),
            program_num_layers=1,
            layers=1,
            max_new_tokens=2,
            decode_steps=1,
            prefill_len=8,
            device="p150a",
            device_id=0,
            batch_size=2,
            cache_len=16,
            dtype_seed="bf16",
            decode_plan={
                "tensor_conversion_count": 5,
                "op_sequence": ["decode"],
            },
            prefill_plan={
                "layers": 1,
                "batch_size": 2,
                "tensor_conversion_count": 7,
                "op_sequence": ["prefill"],
                "expected_output_shapes": {
                    "key_cache": [2, 8, 16],
                    "value_cache": [2, 8, 16],
                },
            },
        )
        self.assertEqual(dry_run_report["status"], "dry_run")
        self.assertTrue(dry_run_report["passed"])
        self.assertEqual(dry_run_report["tensor_conversion_count"], 12)
        self.assertEqual(
            dry_run_report["prefill_cache_population_summary"]["layer_count"],
            1,
        )
        self.assertTrue(dry_run_report["end_to_end_contract"]["passed"])

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
        original_rotary_builder = runtime_decode.attach_decode_rotary_parameters

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
            runtime_decode._build_prompt_decode_runtime_state_tensors = (
                fake_runtime_builder
            )
            runtime_decode.attach_decode_rotary_parameters = fake_rotary_builder
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
            runtime_decode.attach_decode_rotary_parameters = original_rotary_builder

        self.assertEqual(runtime_state.page_table, "page-table")
        self.assertEqual(runtime_state.cache_position, "cache-position")
        self.assertEqual(runtime_state.rotary_runtime_state, {"rotary": "ready"})
        self.assertEqual(runtime_state.tensor_conversion_count, 5)
        self.assertEqual(runtime_state.decode_runtime_state_input_tensor_count, 2)
        self.assertEqual(runtime_state.rotary_runtime_input_tensor_count, 3)

    def test_runtime_decode_loop_updates_context_and_counts_runtime_tensors(
        self,
    ) -> None:
        original_builder = runtime_decode.build_decode_runtime_for_position
        original_time_decode = runtime_decode._time_decode_step
        original_input_shapes = runtime_decode._loop_input_shapes
        original_output_shapes = runtime_decode._loop_output_shapes
        original_reference = runtime_decode._decode_step_reference
        original_observed_ops = runtime_decode._generated_observed_op_sequence
        builder_indexes: list[int] = []

        def fake_builder(**kwargs: object) -> types.SimpleNamespace:
            generated_token_index = int(kwargs["generated_token_index"])
            builder_indexes.append(generated_token_index)
            return types.SimpleNamespace(
                page_table=f"page-table-{generated_token_index}",
                cache_position=f"cache-position-{generated_token_index}",
                decode_runtime_state={
                    "cache_position_value": 5 + generated_token_index
                },
                rotary_runtime_state={"rotary_index": generated_token_index},
                tensor_conversion_count=5,
                decode_runtime_state_input_tensor_count=2,
                rotary_runtime_input_tensor_count=3,
            )

        decode_tokens = [_FakeTensor((2, 1)), _FakeTensor((2, 1))]

        def fake_time_decode_step(
            **kwargs: object,
        ) -> tuple[object, list[object], float]:
            step_index = len(builder_indexes) - 1
            return (
                decode_tokens[step_index],
                [types.SimpleNamespace(k=_FakeTensor((1,)), v=_FakeTensor((1,)))],
                1.25 + step_index,
            )

        context = types.SimpleNamespace(
            parameters=object(),
            prefill_tokenization={"effective_token_count": 4},
            generated_model=object(),
            token_ids="prefill-token",
            page_table=None,
            cache_position=None,
            kv_cache=["initial-cache"],
        )

        def install_decode_runtime(runtime_state: types.SimpleNamespace) -> None:
            context.page_table = runtime_state.page_table
            context.cache_position = runtime_state.cache_position
            context.decode_runtime_state = runtime_state.decode_runtime_state
            context.rotary_state = runtime_state.rotary_runtime_state

        context.install_decode_runtime = install_decode_runtime
        context.update_kv_cache = lambda kv_cache: setattr(
            context, "kv_cache", kv_cache
        )
        context.update_decode_token = lambda token: setattr(context, "token_ids", token)

        try:
            runtime_decode.build_decode_runtime_for_position = fake_builder
            runtime_decode._time_decode_step = fake_time_decode_step
            runtime_decode._loop_input_shapes = lambda **_: {"page_table": [2, 1]}
            runtime_decode._loop_output_shapes = lambda **_: {"token": [2, 1]}
            runtime_decode._decode_step_reference = lambda **_: {
                "status": "passed",
                "passed": True,
            }
            runtime_decode._generated_observed_op_sequence = lambda *_: ["decode_step"]
            result = run_decode_loop(
                context=context,
                ttnn=object(),
                torch=object(),
                device="device0",
                dtype_seed="bf16",
                decode_plan={"kv_cache": {"page_block_size": 4}},
                batch_size=2,
                cache_len=16,
                layer_count=1,
                decode_step_count=2,
                generated_token_events=[{"step_index": "prefill", "token": "t0"}],
                initial_tensor_conversion_count=10,
            )
        finally:
            runtime_decode.build_decode_runtime_for_position = original_builder
            runtime_decode._time_decode_step = original_time_decode
            runtime_decode._loop_input_shapes = original_input_shapes
            runtime_decode._loop_output_shapes = original_output_shapes
            runtime_decode._decode_step_reference = original_reference
            runtime_decode._generated_observed_op_sequence = original_observed_ops

        self.assertEqual(builder_indexes, [0, 1])
        self.assertEqual(result.tensor_conversion_count, 20)
        self.assertEqual(result.decode_runtime_state_input_tensor_count, 4)
        self.assertEqual(result.decode_rotary_runtime_input_tensor_count, 6)
        self.assertEqual(len(result.step_reports), 2)
        self.assertTrue(all(report["passed"] for report in result.step_reports))
        self.assertEqual(result.generated_token_events[1]["step_index"], 0)
        self.assertEqual(result.generated_token_events[2]["step_index"], 1)
        self.assertIs(context.token_ids, decode_tokens[-1])
        self.assertEqual(result.decode_runtime_state, {"cache_position_value": 6})


if __name__ == "__main__":
    unittest.main()
