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
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.profile import (
    GenerateSectionProfiler,
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


if __name__ == "__main__":
    unittest.main()
