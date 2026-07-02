from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_single_layer_decode import (
    SINGLE_LAYER_DECODE_OPS,
    profile_decode_step,
    run_smoke_decode_step,
    run_smoke_single_layer_decode,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_decode_shell import (
    _write_fake_model_config,
    _write_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_parameters import (
    _fake_torch_and_safetensors,
    _fake_weight_specs,
    _write_fake_model_weights,
)


class SmokeSingleLayerDecodeTest(unittest.TestCase):
    def test_cli_smoke_single_layer_decode_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "single_layer_decode_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            exit_code = main(
                [
                    "smoke-single-layer-decode",
                    "--program-dir",
                    str(program_dir),
                    "--device",
                    "p150a",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["op_sequence"], SINGLE_LAYER_DECODE_OPS)
            self.assertEqual(report["input_shapes"]["token_ids"], [2, 1])
            self.assertEqual(
                report["expected_output_shapes"]["key_cache"],
                [2, 16, 2, 4],
            )
            self.assertEqual(report["parameter_shapes"]["attention_wqkv"], [16, 32])
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_run_smoke_single_layer_decode_executes_generated_decode_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "single_layer_decode_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_single_layer_decode(
                out=report_json,
                program_dir=program_dir,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake_ttnn,
                parameters=_fake_parameters(split_count=8),
                token_ids=FakeTensor("token_ids", [2, 1]),
                page_table=FakeTensor("page_table", [2, 1]),
                cache_position=FakeTensor("cache_position", [2]),
                kv_cache=[
                    types.SimpleNamespace(
                        k=FakeTensor("key_cache", [2, 16, 2, 4]),
                        v=FakeTensor("value_cache", [2, 16, 2, 4]),
                    )
                ],
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(report["output_shapes"]["key_cache"], [2, 16, 2, 4])
            self.assertEqual(report["tensor_conversion_count"], 0)
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(report["reference"]["kind"], "structural_shape_dtype")
            self.assertTrue(
                all(check["passed"] for check in report["reference"]["checks"])
            )
            self.assertIn("embedding", report["reference"]["observed_ops"])
            self.assertEqual(json.loads(report_json.read_text()), report)
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "embedding",
                    "rms_norm",
                    "linear",
                    "nlp_create_qkv_heads_decode",
                    "rotary_embedding_llama",
                    "rotary_embedding_llama",
                    "paged_update_cache",
                    "paged_update_cache",
                    "paged_scaled_dot_product_attention_decode",
                    "nlp_concat_heads_decode",
                    "linear",
                    "add",
                    "rms_norm",
                    "linear",
                    "linear",
                    "mul",
                    "linear",
                    "add",
                    "rms_norm",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                    "concat",
                    "argmax",
                ],
            )

    def test_run_smoke_single_layer_decode_synthesizes_fake_ttnn_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "single_layer_decode_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_single_layer_decode(
                out=report_json,
                program_dir=program_dir,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["tensor_conversion_count"], 25)
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls[:25]],
                ["from_torch"] * 25,
            )
            self.assertIn(
                "paged_scaled_dot_product_attention_decode",
                [call["op"] for call in fake_ttnn.calls],
            )

    def test_run_smoke_single_layer_decode_can_materialize_fake_model_weights(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "single_layer_decode_report.json"
            _write_fake_model_config(model_dir)
            _write_fake_model_weights(model_dir, _fake_weight_specs())
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            with _fake_torch_and_safetensors():
                report = run_smoke_single_layer_decode(
                    out=report_json,
                    program_dir=program_dir,
                    model_path=model_dir,
                    device="p150a",
                    batch_size=2,
                    cache_len=16,
                    ttnn_module=fake_ttnn,
                    torch_module=_fake_torch(),
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["parameter_source"], "hf_model")
            self.assertEqual(report["input_source"], "synthetic")
            self.assertEqual(report["tensor_conversion_count"], 25)
            self.assertEqual(
                report["parameter_setup"]["materialization"][
                    "materialized_layer_ids"
                ],
                [0],
            )
            self.assertEqual(
                report["parameter_setup"]["materialization"]["tensor_count"],
                21,
            )
            self.assertEqual(
                report["parameter_setup"]["tensorization"]["roles"],
                ["embedding", "norm", "attention", "mlp", "lm_head"],
            )
            self.assertEqual(
                report["parameter_setup"]["tensorization"]["tensor_count"],
                17,
            )
            self.assertEqual(
                report["parameter_setup"]["tensorization"][
                    "memory_config_counts"
                ],
                {"dram": 17},
            )
            self.assertEqual(
                report["parameter_setup"]["tensorization"][
                    "ttnn_memory_config_counts"
                ],
                {"ttnn.DRAM_MEMORY_CONFIG": 17},
            )
            self.assertEqual(
                report["parameter_setup"]["tensorization"]["key_tensors"][
                    "layers.0.attention.wqkv_packed.weight"
                ]["ttnn_memory_config"],
                "ttnn.DRAM_MEMORY_CONFIG",
            )
            self.assertEqual(
                report["parameter_setup"]["synthetic_rotary_tensor_count"],
                3,
            )
            self.assertEqual(
                report["parameter_setup"]["synthetic_runtime_input_tensor_count"],
                5,
            )
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls[:17]],
                ["from_torch"] * 17,
            )
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"]["layout"],
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"]["memory_config"],
                "ttnn.DRAM_MEMORY_CONFIG",
            )
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_cli_smoke_decode_step_dry_run_two_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            exit_code = main(
                [
                    "smoke-decode-step",
                    "--program-dir",
                    str(program_dir),
                    "--layers",
                    "2",
                    "--device",
                    "p150a",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--trace",
                    "--trace-iterations",
                    "2",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["template"], "generated_decode_step")
            self.assertEqual(report["layers"], 2)
            self.assertEqual(report["op_sequence"].count("qkv_linear"), 2)
            self.assertEqual(report["op_sequence"].count("rms_norm.final"), 1)
            self.assertEqual(report["tensor_conversion_count"], 37)
            self.assertTrue(report["trace_enabled"])
            self.assertEqual(report["trace_iterations"], 2)
            self.assertEqual(report["trace"]["status"], "dry_run")
            self.assertEqual(report["reference"]["status"], "dry_run")

    def test_run_smoke_decode_step_trace_executes_fake_trace_apis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_trace_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_decode_step(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                device="p150a",
                batch_size=2,
                cache_len=16,
                trace=True,
                trace_iterations=2,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["trace"]["status"], "captured_and_executed")
            self.assertEqual(report["trace"]["iterations"], 2)
            self.assertEqual(len(report["trace"]["execute_samples_ms"]), 2)
            self.assertEqual(report["reference"]["status"], "passed")
            trace_ops = [
                call["op"]
                for call in fake_ttnn.calls
                if call["op"]
                in {
                    "begin_trace_capture",
                    "end_trace_capture",
                    "execute_trace",
                    "release_trace",
                }
            ]
            self.assertEqual(
                trace_ops,
                [
                    "begin_trace_capture",
                    "end_trace_capture",
                    "execute_trace",
                    "execute_trace",
                    "release_trace",
                ],
            )

    def test_cli_profile_decode_step_dry_run_two_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_profile_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            exit_code = main(
                [
                    "profile-decode-step",
                    "--program-dir",
                    str(program_dir),
                    "--layers",
                    "2",
                    "--batch-size",
                    "2",
                    "--cache-len",
                    "16",
                    "--trace",
                    "--trace-iterations",
                    "2",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["template"], "generated_decode_step_profile")
            self.assertEqual(len(report["layer_profiles"]), 2)
            self.assertIn("embedding_ms", report["profile_sections"])
            self.assertIn(
                "tensor_conversion_ms",
                report["bottleneck_summary"]["sections_ms"],
            )
            self.assertEqual(report["throughput_summary"]["status"], "dry_run")
            self.assertEqual(
                report["throughput_summary"]["tokens_per_second_per_user"],
                0.0,
            )
            self.assertEqual(report["trace"]["status"], "dry_run")
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_profile_decode_step_reports_fake_bottleneck_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_profile_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            report = profile_decode_step(
                out=report_json,
                program_dir=program_dir,
                layers=2,
                device="p150a",
                batch_size=2,
                cache_len=16,
                trace=True,
                trace_iterations=2,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "profiled")
            self.assertEqual(report["layers"], 2)
            self.assertEqual(len(report["layer_profiles"]), 2)
            self.assertEqual(report["tensor_conversion_count"], 37)
            self.assertGreaterEqual(report["tensor_conversion_ms"], 0.0)
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(report["reference"]["kind"], "structural_shape_dtype")
            self.assertTrue(
                all(check["passed"] for check in report["reference"]["checks"])
            )
            self.assertEqual(report["lm_head_profile"]["split_count"], 8)
            self.assertEqual(report["lm_head_profile"]["argmax_status"], "profiled")
            sections = report["bottleneck_summary"]["sections_ms"]
            for name in (
                "embedding_ms",
                "per_layer_attention_ms",
                "per_layer_mlp_ms",
                "lm_head_ms",
                "argmax_ms",
                "host_copy_ms",
                "trace_execute_ms",
                "tensor_conversion_ms",
            ):
                self.assertIn(name, sections)
            self.assertEqual(report["trace"]["status"], "captured_and_executed")
            self.assertEqual(report["trace"]["iterations"], 2)
            throughput = report["throughput_summary"]
            self.assertEqual(throughput["status"], "measured")
            self.assertEqual(throughput["batch_size"], 2)
            self.assertGreater(throughput["tokens_per_second_per_user"], 0.0)
            self.assertAlmostEqual(
                throughput["aggregate_tokens_per_second"],
                throughput["tokens_per_second_per_user"] * 2,
            )
            self.assertGreater(
                throughput["trace_execute_tokens_per_second_per_user"],
                0.0,
            )
            self.assertEqual(report["ttnn_environment"]["version"], "fake-ttnn")
            self.assertEqual(
                report["ttnn_environment"]["tt_metal_git_commit"],
                "fake-tt-metal",
            )
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_run_smoke_decode_step_executes_two_generated_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_step_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            fake_ttnn = _make_fake_ttnn()
            report = run_smoke_decode_step(
                out=report_json,
                program_dir=program_dir,
                layers=2,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["layers"], 2)
            self.assertEqual(report["tensor_conversion_count"], 37)
            self.assertEqual(report["output_shapes"]["token"], [2, 1])
            self.assertEqual(len(report["output_shapes"]["kv_cache_layers"]), 2)
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(report["ttnn_environment"]["version"], "fake-ttnn")
            self.assertEqual(
                report["reference"]["planned_ops"].count("qkv_linear"),
                2,
            )
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls[:37]],
                ["from_torch"] * 37,
            )
            generated_ops = [call["op"] for call in fake_ttnn.calls[37:]]
            self.assertEqual(
                generated_ops.count("paged_scaled_dot_product_attention_decode"),
                2,
            )
            self.assertEqual(generated_ops.count("add"), 4)
            self.assertEqual(generated_ops.count("rms_norm"), 5)
            self.assertEqual(generated_ops.count("linear"), 18)
            self.assertEqual(generated_ops.count("argmax"), 1)

    def test_single_layer_decode_api_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "single_layer_decode_report.json"
            _write_fake_model_config(model_dir)
            _write_template_config(config_json)
            self.assertEqual(
                main(
                    [
                        "build-program",
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

            report = run_smoke_single_layer_decode(
                out=report_json,
                program_dir=program_dir,
                device="p150a",
                batch_size=2,
                cache_len=16,
                ttnn_module=_make_fake_ttnn(with_transformer=False),
                parameters=_fake_parameters(split_count=8),
                token_ids=FakeTensor("token_ids", [2, 1]),
                page_table=FakeTensor("page_table", [2, 1]),
                cache_position=FakeTensor("cache_position", [2]),
                kv_cache=[
                    types.SimpleNamespace(
                        k=FakeTensor("key_cache", [2, 16, 2, 4]),
                        v=FakeTensor("value_cache", [2, 16, 2, 4]),
                    )
                ],
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "api_mismatch")
            self.assertIn(
                "paged_scaled_dot_product_attention_decode",
                report["error"],
            )


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str = "ttnn.bfloat16",
        mem_config: str | None = None,
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype
        self._mem_config = mem_config

    def memory_config(self) -> str | None:
        return self._mem_config

    def __repr__(self) -> str:
        return f"FakeTensor({self.name})"


def _fake_parameters(split_count: int):
    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(
            weight=FakeTensor("embed_weight", [128, 16])
        ),
        layers=[
            types.SimpleNamespace(
                attention=types.SimpleNamespace(
                    wqkv_packed=types.SimpleNamespace(
                        weight=FakeTensor("wqkv_weight", [16, 32])
                    ),
                    o_proj=types.SimpleNamespace(
                        weight=FakeTensor("o_proj_weight", [16, 16])
                    ),
                    rotary=types.SimpleNamespace(
                        cos_matrix=FakeTensor("cos_matrix", [1, 1, 4, 4]),
                        sin_matrix=FakeTensor("sin_matrix", [1, 1, 4, 4]),
                        transformation_matrix=FakeTensor(
                            "transformation_matrix",
                            [1, 1, 4, 4],
                        ),
                    ),
                ),
                input_norm=types.SimpleNamespace(
                    weight=FakeTensor("input_norm_weight", [16])
                ),
                post_attention_norm=types.SimpleNamespace(
                    weight=FakeTensor("post_attention_norm_weight", [16])
                ),
                mlp=types.SimpleNamespace(
                    gate_proj=types.SimpleNamespace(
                        weight=FakeTensor("gate_weight", [16, 32])
                    ),
                    up_proj=types.SimpleNamespace(
                        weight=FakeTensor("up_weight", [16, 32])
                    ),
                    down_proj=types.SimpleNamespace(
                        weight=FakeTensor("down_weight", [32, 16])
                    ),
                ),
            )
        ],
        final_norm=types.SimpleNamespace(
            weight=FakeTensor("final_norm_weight", [16])
        ),
        lm_head=types.SimpleNamespace(
            splits=[
                types.SimpleNamespace(
                    shard_id=index,
                    weight=FakeTensor(f"lm_head_{index}", [16, 16]),
                )
                for index in range(split_count)
            ]
        ),
    )


def _make_fake_ttnn(*, with_transformer: bool = True):
    module = types.ModuleType("ttnn")
    module.calls = []
    module.__version__ = "fake-ttnn"
    module.__tt_metal_commit__ = "fake-tt-metal"
    module.bfloat16 = "ttnn.bfloat16"
    module.float32 = "ttnn.float32"
    module.TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"
    module.DRAM_MEMORY_CONFIG = "ttnn.DRAM_MEMORY_CONFIG"

    class UnaryOpType:
        SILU = "SILU"

    def unary_with_param(op):
        return ("UnaryWithParam", op)

    def embedding(token_ids, weight, **kwargs):
        batch = token_ids.shape[0]
        hidden_size = weight.shape[-1]
        module.calls.append(
            {
                "op": "embedding",
                "token_ids": token_ids.name,
                "weight": weight.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("embedding", [batch, 1, hidden_size])

    def from_torch(tensor, **kwargs):
        module.calls.append(
            {
                "op": "from_torch",
                "shape": list(tensor.shape),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            getattr(tensor, "name", "torch_tensor"),
            list(tensor.shape),
            dtype=str(kwargs.get("dtype")),
        )

    def rms_norm(hidden, **kwargs):
        module.calls.append(
            {
                "op": "rms_norm",
                "hidden": hidden.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("rms_norm", hidden.shape)

    def linear(activation, weight, **kwargs):
        out_shape = list(activation.shape)
        out_shape[-1] = weight.shape[-1]
        module.calls.append(
            {
                "op": "linear",
                "activation": activation.name,
                "weight": weight.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(f"linear:{weight.name}", out_shape)

    def nlp_create_qkv_heads_decode(fused_qkv, **kwargs):
        num_heads = int(kwargs["num_heads"])
        num_kv_heads = int(kwargs["num_kv_heads"])
        head_dim = fused_qkv.shape[-1] // (num_heads + 2 * num_kv_heads)
        batch = fused_qkv.shape[0]
        module.calls.append(
            {
                "op": "nlp_create_qkv_heads_decode",
                "qkv": fused_qkv.name,
                "kwargs": dict(kwargs),
            }
        )
        return (
            FakeTensor("query", [batch, num_heads, 1, head_dim]),
            FakeTensor("key", [batch, num_kv_heads, 1, head_dim]),
            FakeTensor("value", [batch, num_kv_heads, 1, head_dim]),
        )

    def rotary_embedding_llama(tensor, cos, sin, transform, **kwargs):
        module.calls.append(
            {
                "op": "rotary_embedding_llama",
                "tensor": tensor.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(f"rotary:{tensor.name}", tensor.shape)

    def paged_update_cache(cache, update, **kwargs):
        module.calls.append(
            {
                "op": "paged_update_cache",
                "cache": cache.name,
                "update": update.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("cache", cache.shape)

    def paged_scaled_dot_product_attention_decode(q, k_cache, v_cache, **kwargs):
        module.calls.append(
            {
                "op": "paged_scaled_dot_product_attention_decode",
                "query": q.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("attention", q.shape)

    def nlp_concat_heads_decode(attention, **kwargs):
        num_heads = int(kwargs["num_heads"])
        batch = attention.shape[0]
        head_dim = attention.shape[-1]
        module.calls.append(
            {
                "op": "nlp_concat_heads_decode",
                "attention": attention.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("concat_heads", [batch, 1, num_heads * head_dim])

    def mul(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "mul",
                "lhs": lhs.name,
                "rhs": rhs.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("mul", lhs.shape)

    def add(lhs, rhs, **kwargs):
        module.calls.append(
            {
                "op": "add",
                "lhs": lhs.name,
                "rhs": rhs.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("add", lhs.shape)

    def concat(tensors, **kwargs):
        shape = list(tensors[0].shape)
        dim = int(kwargs.get("dim", -1))
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        module.calls.append(
            {
                "op": "concat",
                "tensors": [tensor.name for tensor in tensors],
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("concat", shape)

    def argmax(tensor, **kwargs):
        dim = int(kwargs.get("dim", -1))
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        del shape[dim]
        module.calls.append(
            {
                "op": "argmax",
                "tensor": tensor.name,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor("argmax", shape)

    def begin_trace_capture(device, **kwargs):
        module.calls.append(
            {
                "op": "begin_trace_capture",
                "device": device,
                "kwargs": dict(kwargs),
            }
        )
        return "fake_trace_id"

    def end_trace_capture(device, trace_id, **kwargs):
        module.calls.append(
            {
                "op": "end_trace_capture",
                "device": device,
                "trace_id": trace_id,
                "kwargs": dict(kwargs),
            }
        )

    def execute_trace(device, trace_id, **kwargs):
        module.calls.append(
            {
                "op": "execute_trace",
                "device": device,
                "trace_id": trace_id,
                "kwargs": dict(kwargs),
            }
        )

    def release_trace(device, trace_id):
        module.calls.append(
            {
                "op": "release_trace",
                "device": device,
                "trace_id": trace_id,
            }
        )

    module.UnaryOpType = UnaryOpType
    module.UnaryWithParam = unary_with_param
    module.from_torch = from_torch
    module.embedding = embedding
    module.rms_norm = rms_norm
    module.linear = linear
    module.mul = mul
    module.add = add
    module.concat = concat
    module.argmax = argmax
    module.begin_trace_capture = begin_trace_capture
    module.end_trace_capture = end_trace_capture
    module.execute_trace = execute_trace
    module.release_trace = release_trace
    module.experimental = types.SimpleNamespace(
        nlp_create_qkv_heads_decode=nlp_create_qkv_heads_decode,
        rotary_embedding_llama=rotary_embedding_llama,
        paged_update_cache=paged_update_cache,
        nlp_concat_heads_decode=nlp_concat_heads_decode,
    )
    if with_transformer:
        module.transformer = types.SimpleNamespace(
            paged_scaled_dot_product_attention_decode=(
                paged_scaled_dot_product_attention_decode
            )
        )
    return module


if __name__ == "__main__":
    unittest.main()
