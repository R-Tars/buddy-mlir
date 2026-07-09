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
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.inputs import (
    build_decode_kv_cache_runtime_state,
    build_decode_rotary_runtime_state,
    build_decode_runtime_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.profile import (
    GenerateSectionProfiler,
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


class RuntimeModuleTest(unittest.TestCase):
    def test_generate_module_reexports_runtime_classes(self) -> None:
        self.assertIs(GenerateCompatContext, TTNNDirectRuntimeContext)
        self.assertIs(GenerateCompatProfiler, GenerateSectionProfiler)
        self.assertTrue(callable(run_generate))
        self.assertTrue(callable(run_profile_generate))

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


if __name__ == "__main__":
    unittest.main()
