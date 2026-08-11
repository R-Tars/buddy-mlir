from __future__ import annotations

import json
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics import prefill as prefill_diagnostic
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.attention_layer import (
    ATTENTION_LAYER_OPS,
    run_smoke_attention_layer,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.attention_primitive import (
    ATTENTION_PRIMITIVES,
    PRIMITIVE_EXPECTED_OBSERVED_OPS,
    run_smoke_attention_primitive,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.cli import DIAGNOSE_STAGES
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.decode_shell import (
    DECODE_SHELL_OPS,
    run_smoke_decode_shell,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.decode_step import (
    SINGLE_LAYER_DECODE_OPS,
    profile_decode_step,
    run_smoke_decode_step,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.mlp import (
    MLP_SMOKE_OPS,
    run_smoke_mlp,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.prefill import (
    run_smoke_prefill,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.device import (
    managed_ttnn_device,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import (
    FakeTensor,
    MiniTensor,
    _fake_numeric_parameters,
    _fake_numeric_torch,
    _fake_parameters,
    _fake_physical_numeric_parameters,
    _fake_tokenizer_module,
    _fake_torch,
    _fake_ttnn,
    _make_fake_ttnn,
    _make_numeric_ttnn,
    _write_fake_model_config,
    _write_template_config,
)


EXPECTED_STAGES = (
    "mlp",
    "attention-primitive",
    "attention-layer",
    "prefill",
    "decode-shell",
    "decode-step",
    "decode-step-profile",
    "decode-loop-legacy",
    "depth-sweep",
    "generate-depth-sweep",
    "autotune",
    "autotune-profiler-audit",
    "benchmark-parity",
    "execution-graph-diff",
    "performance-correctness",
    "template-profile",
)


class DiagnosticContractTest(unittest.TestCase):
    def _program(self, root: Path) -> Path:
        model_dir = root / "fake_model"
        config_json = root / "template_config.json"
        program_dir = root / "program"
        _write_fake_model_config(model_dir)
        _write_template_config(config_json)
        self.assertEqual(
            main(
                [
                    "build",
                    "--model-path",
                    str(model_dir),
                    "--config",
                    str(config_json),
                    "--out-dir",
                    str(program_dir),
                ]
            ),
            0,
        )
        return program_dir

    @staticmethod
    def _cache() -> list[types.SimpleNamespace]:
        return [
            types.SimpleNamespace(
                k=FakeTensor("key_cache", [2, 2, 32, 4]),
                v=FakeTensor("value_cache", [2, 2, 32, 4]),
            )
        ]

    @staticmethod
    def _attention_args() -> dict[str, object]:
        return {
            "device": "p150a",
            "batch_size": 2,
            "hidden_size": 16,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "max_cache_len": 16,
        }

    def _injected_decode_state(self) -> dict[str, object]:
        return {
            "parameters": _fake_parameters(8),
            "token_ids": FakeTensor("token_ids", [2, 1]),
            "page_table": FakeTensor("page_table", [2, 1]),
            "cache_position": FakeTensor("cache_position", [2]),
            "kv_cache": self._cache(),
        }

    def test_stage_contract_is_exact_and_ordered(self) -> None:
        self.assertEqual(DIAGNOSE_STAGES, EXPECTED_STAGES)

    def test_mlp_dry_run_and_no_device_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dry = run_smoke_mlp(
                out=root / "mlp-dry.json",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                intermediate_size=32,
                dry_run=True,
            )
            self.assertEqual(dry["status"], "dry_run")
            self.assertEqual(dry["ttnn_ops"], MLP_SMOKE_OPS)
            unavailable = run_smoke_mlp(
                out=root / "mlp-no-device.json",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                intermediate_size=32,
                ttnn_module=types.SimpleNamespace(),
                torch_module=types.SimpleNamespace(),
            )
            self.assertEqual(unavailable["status"], "no_device")
            self.assertFalse(unavailable["passed"])

    def test_attention_primitive_dry_run_and_fake_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = {
                "primitive": "paged_scaled_dot_product_attention_decode",
                "device": "p150a",
                "batch_size": 2,
                "hidden_size": 16,
                "num_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 4,
                "max_cache_len": 16,
            }
            dry = run_smoke_attention_primitive(
                **common, out=root / "attention.json", dry_run=True
            )
            self.assertEqual(dry["status"], "dry_run")
            self.assertIn(dry["primitive"], ATTENTION_PRIMITIVES)
            fake = _fake_ttnn()
            passed = run_smoke_attention_primitive(
                **common,
                out=root / "attention-fake.json",
                ttnn_module=fake,
                torch_module=_fake_torch(),
            )
            self.assertEqual(passed["status"], "passed")
            self.assertTrue(passed["reference"]["passed"])
            self.assertTrue(passed["output_shapes"])

    def test_each_attention_primitive_preserves_fake_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for primitive in ATTENTION_PRIMITIVES:
                with self.subTest(primitive=primitive):
                    report = run_smoke_attention_primitive(
                        **self._attention_args(),
                        primitive=primitive,
                        out=root / f"{primitive}.json",
                        ttnn_module=_fake_ttnn(),
                        torch_module=_fake_torch(),
                    )
                    self.assertEqual(report["status"], "passed")
                    self.assertEqual(
                        report["reference"]["planned_ops"],
                        PRIMITIVE_EXPECTED_OBSERVED_OPS[primitive],
                    )
                    self.assertTrue(report["reference"]["passed"])
                    self.assertEqual(
                        set(report["output_shapes"]),
                        set(report["expected_output_shapes"]),
                    )

            direct = run_smoke_attention_primitive(
                **self._attention_args(),
                primitive="qkv_linear",
                out=root / "qkv-direct-evidence.json",
                ttnn_module=_fake_ttnn(instrumented=False),
                torch_module=_fake_torch(),
            )
            self.assertEqual(
                direct["reference"]["observed_ops_source"],
                "direct_primitive_call",
            )

    def test_attention_index_and_sharded_memory_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            update_fake = _fake_ttnn()
            update = run_smoke_attention_primitive(
                **self._attention_args(),
                primitive="paged_update_cache",
                out=root / "update.json",
                ttnn_module=update_fake,
                torch_module=_fake_torch(),
            )
            self.assertEqual(
                update["input_tensor_contracts"]["cache_position"],
                {
                    "dtype": "int32",
                    "layout": "row_major",
                    "memory_config": "default_or_dram",
                },
            )
            index_calls = {
                call["name"]: call["kwargs"]
                for call in update_fake.calls
                if call["op"] == "from_torch"
                and call["name"] in {"page_table", "cache_position"}
            }
            self.assertEqual(index_calls["page_table"]["dtype"], "ttnn.int32")
            self.assertEqual(
                index_calls["cache_position"]["layout"],
                "ttnn.ROW_MAJOR_LAYOUT",
            )

            rotary_fake = _fake_ttnn(with_create_sharded=True)
            run_smoke_attention_primitive(
                **self._attention_args(),
                primitive="rotary_embedding_decode",
                out=root / "rotary.json",
                ttnn_module=rotary_fake,
                torch_module=_fake_torch(),
            )
            sharded = [
                call
                for call in rotary_fake.calls
                if call["op"] == "create_sharded_memory_config"
            ]
            self.assertTrue(sharded)
            self.assertIn((32, 4), [tuple(call["kwargs"]["shape"]) for call in sharded])

    def test_attention_failures_remain_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = {
                **self._attention_args(),
                "primitive": "paged_scaled_dot_product_attention_decode",
            }
            unavailable = run_smoke_attention_primitive(
                **common,
                out=root / "api-mismatch.json",
                ttnn_module=_fake_ttnn(with_transformer=False),
                torch_module=_fake_torch(),
            )
            self.assertEqual(unavailable["status"], "api_mismatch")

            fake = _fake_ttnn()

            def wrong_sdpa(query, key_cache, value_cache, **kwargs):
                fake.calls.append({"op": "wrong_sdpa_decode"})
                return type(query)("attention", list(query.shape))

            fake.transformer.paged_scaled_dot_product_attention_decode = wrong_sdpa
            mismatch = run_smoke_attention_primitive(
                **common,
                out=root / "reference-mismatch.json",
                ttnn_module=fake,
                torch_module=_fake_torch(),
            )
            self.assertEqual(mismatch["status"], "reference_mismatch")
            self.assertEqual(
                [
                    check["name"]
                    for check in mismatch["reference"]["checks"]
                    if not check["passed"]
                ],
                ["observed_op_sequence"],
            )

            program = self._program(root)
            layer_api = run_smoke_attention_layer(
                out=root / "layer-api-mismatch.json",
                program_dir=program,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_fake_ttnn(with_transformer=False),
                torch_module=_fake_torch(),
            )
            self.assertEqual(layer_api["status"], "api_mismatch")

            layer_fake = _fake_ttnn()

            def wrong_layer_sdpa(query, key_cache, value_cache, **kwargs):
                layer_fake.calls.append({"op": "wrong_sdpa_decode"})
                return type(query)("attention", list(query.shape))

            layer_fake.transformer.paged_scaled_dot_product_attention_decode = (
                wrong_layer_sdpa
            )
            layer_mismatch = run_smoke_attention_layer(
                out=root / "layer-reference-mismatch.json",
                program_dir=program,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=layer_fake,
                torch_module=_fake_torch(),
            )
            self.assertEqual(layer_mismatch["status"], "reference_mismatch")

    def test_attention_layer_and_prefill_plans_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            layer = run_smoke_attention_layer(
                out=root / "attention-layer.json",
                program_dir=program,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                dry_run=True,
            )
            self.assertEqual(layer["status"], "dry_run")
            self.assertEqual(layer["op_sequence"], ATTENTION_LAYER_OPS)
            prefill = run_smoke_prefill(
                out=root / "prefill.json",
                program_dir=program,
                layers=1,
                prefill_len=8,
                device="p150a",
                batch_size=2,
                cache_len=16,
                dry_run=True,
            )
            self.assertEqual(prefill["status"], "dry_run")
            self.assertEqual(prefill["prefill_status"], "dry_run")
            self.assertEqual(prefill["kv_cache_source"], "prefill")

    def test_prefill_model_path_reuses_canonical_runtime_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            context = types.SimpleNamespace(
                tensor_conversion_count=7,
                parameter_source="hf_model",
                input_source="prompt_prefill",
            )
            execution = {
                "status": "passed",
                "passed": True,
                "prefill_status": "passed",
                "kv_cache_source": "prefill",
            }
            with mock.patch.object(
                prefill_diagnostic,
                "_build_runtime_session",
                return_value=types.SimpleNamespace(context=context),
            ) as build_session, mock.patch.object(
                prefill_diagnostic,
                "_run_generated_prefill",
                return_value=execution,
            ) as run_prefill:
                report = run_smoke_prefill(
                    out=root / "prefill-model.json",
                    program_dir=program,
                    model_path=root / "model",
                    prompt="hello",
                    layers=1,
                    prefill_len=8,
                    device="p150a",
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=_make_fake_ttnn(),
                    torch_module=_fake_torch(),
                )
            self.assertEqual(report["parameter_source"], "hf_model")
            self.assertEqual(report["input_source"], "prompt_prefill")
            self.assertIs(run_prefill.call_args.kwargs["runtime_context"], context)
            self.assertEqual(build_session.call_args.kwargs["layer_count"], 1)

    def test_attention_layer_and_prefill_execute_with_injected_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            layer = run_smoke_attention_layer(
                out=root / "attention-layer-fake.json",
                program_dir=program,
                layer=0,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_fake_ttnn(),
                torch_module=_fake_torch(),
            )
            self.assertEqual(layer["status"], "passed")
            self.assertTrue(layer["reference"]["passed"])
            self.assertEqual(len(layer["primitive_reports"]), 8)

            fake = _make_fake_ttnn()
            prefill = run_smoke_prefill(
                out=root / "prefill-fake.json",
                program_dir=program,
                layers=1,
                prefill_len=8,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake,
                torch_module=object(),
                parameters=_fake_parameters(8),
                token_ids=FakeTensor("prefill_token_ids", [2, 8]),
                kv_cache=self._cache(),
            )
            self.assertEqual(prefill["status"], "passed")
            self.assertEqual(prefill["cache_population"][0]["status"], "filled")
            self.assertEqual(prefill["reference"]["status"], "passed")
            self.assertEqual(
                [
                    call["kwargs"].get("user_id")
                    for call in fake.calls
                    if call["op"] == "fill_cache"
                ],
                [0, 0, 1, 1],
            )

    def test_decode_shell_is_mlp_only_and_decode_step_eager_is_structural(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            shell = run_smoke_decode_shell(
                out=root / "shell.json",
                program_dir=program,
                layers=1,
                disable_attention=True,
                device="p150a",
                batch_size=2,
                cache_len=16,
                dry_run=True,
            )
            self.assertEqual(shell["status"], "dry_run")
            self.assertEqual(shell["ttnn_ops"], DECODE_SHELL_OPS)
            decode = run_smoke_decode_step(
                out=root / "decode.json",
                program_dir=program,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_make_fake_ttnn(),
                parameters=_fake_parameters(8),
                token_ids=FakeTensor("token_ids", [2, 1]),
                page_table=FakeTensor("page_table", [2, 1]),
                cache_position=FakeTensor("cache_position", [2]),
                kv_cache=self._cache(),
            )
            self.assertEqual(decode["status"], "passed")
            self.assertEqual(decode["reference"]["status"], "passed")
            self.assertEqual(decode["op_sequence"], SINGLE_LAYER_DECODE_OPS)

    def test_decode_shell_injected_execution_remains_mlp_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            fake = _make_fake_ttnn()
            report = run_smoke_decode_shell(
                out=root / "shell-fake.json",
                program_dir=program,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake,
                parameters=_fake_parameters(8),
                token_ids=FakeTensor("token_ids", [32, 1]),
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["input_source"], "injected")
            self.assertEqual(report["reference"]["status"], "passed")
            ops = [call["op"] for call in fake.calls]
            self.assertIn("mul", ops)
            self.assertNotIn("paged_scaled_dot_product_attention_decode", ops)

    def test_decode_shell_numeric_reference_preserves_pcc_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            parameters = _fake_numeric_parameters(8)
            report = run_smoke_decode_shell(
                out=root / "shell-numeric.json",
                program_dir=program,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=_make_numeric_ttnn(),
                torch_module=_fake_numeric_torch(),
                parameters=parameters,
                reference_parameters=parameters,
                token_ids=MiniTensor([[0] for _ in range(32)], dtype="int64"),
            )
            numeric = report["reference"]["numeric_reference"]
            self.assertEqual(report["status"], "passed")
            self.assertEqual(numeric["status"], "passed")
            self.assertGreaterEqual(numeric["pcc"], 0.999999)

            physical = _fake_physical_numeric_parameters(8)
            physical_report = run_smoke_decode_shell(
                out=root / "shell-physical-numeric.json",
                program_dir=program,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=_make_numeric_ttnn(),
                torch_module=_fake_numeric_torch(),
                parameters=physical,
                reference_parameters=physical,
                token_ids=MiniTensor([[0] for _ in range(32)], dtype="int64"),
            )
            self.assertEqual(
                physical_report["reference"]["numeric_reference"]["status"],
                "passed",
            )

    def test_decode_trace_profile_and_cli_dispatch_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            fake = _make_fake_ttnn()
            trace = run_smoke_decode_step(
                out=root / "decode-trace.json",
                program_dir=program,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                trace=True,
                trace_iterations=2,
                ttnn_module=fake,
                parameters=_fake_parameters(8),
                token_ids=FakeTensor("token_ids", [2, 1]),
                page_table=FakeTensor("page_table", [2, 1]),
                cache_position=FakeTensor("cache_position", [2]),
                kv_cache=self._cache(),
            )
            self.assertEqual(trace["trace"]["status"], "captured_and_executed")
            ops = [call["op"] for call in fake.calls]
            self.assertEqual(ops.count("begin_trace_capture"), 1)
            self.assertEqual(ops.count("end_trace_capture"), 1)
            self.assertEqual(ops.count("release_trace"), 1)
            self.assertEqual(ops.count("execute_trace"), 2)
            profile = profile_decode_step(
                out=root / "profile.json",
                program_dir=program,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                trace=True,
                dry_run=True,
            )
            self.assertEqual(profile["status"], "dry_run")
            self.assertEqual(profile["trace"]["status"], "dry_run")
            cli_report = root / "cli.json"
            self.assertEqual(
                main(
                    [
                        "diagnose",
                        "--stage",
                        "decode-step",
                        "--program-dir",
                        str(program),
                        "--device",
                        "p150a",
                        "--batch-size",
                        "2",
                        "--cache-len",
                        "16",
                        "--dry-run",
                        "--out",
                        str(cli_report),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(cli_report.read_text())["status"], "dry_run")

    def test_decode_profile_preserves_segmented_fake_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = self._program(root)
            report = profile_decode_step(
                out=root / "profile-fake.json",
                program_dir=program,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                trace=True,
                trace_iterations=2,
                ttnn_module=_make_fake_ttnn(),
                **self._injected_decode_state(),
            )
            self.assertEqual(report["status"], "profiled")
            self.assertEqual(report["trace"]["status"], "captured_and_executed")
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(len(report["layer_profiles"]), 1)
            self.assertEqual(report["lm_head_profile"]["argmax_status"], "profiled")
            self.assertGreater(
                report["throughput_summary"]["tokens_per_second_per_user"],
                0.0,
            )

            mismatch = run_smoke_decode_step(
                out=root / "decode-api-mismatch.json",
                program_dir=program,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_make_fake_ttnn(with_transformer=False),
                **self._injected_decode_state(),
            )
            self.assertEqual(mismatch["status"], "api_mismatch")

    def test_managed_device_closes_once_on_success_and_exception(self) -> None:
        for raises in (False, True):
            with self.subTest(raises=raises):
                calls = []
                device = object()
                ttnn = types.SimpleNamespace(
                    open_device=lambda **kwargs: calls.append(("open", kwargs)) or device,
                    close_device=lambda value: calls.append(("close", value)),
                )
                with self.assertRaises(RuntimeError) if raises else nullcontext():
                    with managed_ttnn_device(ttnn, 3) as opened:
                        self.assertIs(opened, device)
                        if raises:
                            raise RuntimeError("injected failure")
                self.assertEqual(calls[0], ("open", {"device_id": 3}))
                self.assertEqual(calls[1], ("close", device))
                self.assertEqual(len(calls), 2)

    def test_prompt_tokenization_adapter_is_used_for_model_backed_setup(self) -> None:
        self.assertEqual(
            _fake_tokenizer_module([7, 11]).AutoTokenizer.from_pretrained("fake")(
                "hello"
            )["input_ids"],
            [7, 11],
        )


if __name__ == "__main__":
    unittest.main()
