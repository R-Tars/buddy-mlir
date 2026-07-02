from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.parameters import (
    load_llama_parameters_from_manifests,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.ttnn_tensorizer import (
    load_parameter_config_from_program,
    to_ttnn_parameters,
)


class ParameterMaterializerTest(unittest.TestCase):
    def test_loads_layer_limited_parameters_and_packs_qkv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
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

            with _fake_torch_and_safetensors():
                params = load_llama_parameters_from_manifests(
                    model_path=model_dir,
                    weights_manifest=program_dir / "weights_manifest.json",
                    config=program_dir / "config.json",
                    tensor_backend="torch",
                    layers=[0],
                )

            self.assertEqual(params.metadata["materialized_layer_ids"], [0])
            self.assertIsNotNone(params.layers[0])
            self.assertIsNone(params.layers[1])
            self.assertEqual(
                params.layers[0].attention.wqkv_packed.shape,
                [32, 16],
            )
            self.assertEqual(
                params.layers[0].attention.wqkv_packed.source_keys,
                [
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                ],
            )
            self.assertEqual(
                params.layers[0].mlp.gate_proj.shape,
                [32, 16],
            )
            self.assertEqual(len(params.lm_head.splits), 8)
            self.assertEqual(params.lm_head.splits[0].weight.shape, [16, 16])
            self.assertEqual(params.metadata["tensor_count"], 21)

    def test_cli_materialize_parameters_writes_shape_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
            report_json = root / "parameter_report.json"
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

            with _fake_torch_and_safetensors():
                exit_code = main(
                    [
                        "materialize-parameters",
                        "--model-path",
                        str(model_dir),
                        "--program-dir",
                        str(program_dir),
                        "--backend",
                        "torch",
                        "--layers",
                        "0",
                        "--out",
                        str(report_json),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["backend"], "torch")
            self.assertEqual(report["materialized_layer_ids"], [0])
            self.assertEqual(report["lm_head"]["split_count"], 8)
            self.assertEqual(
                report["tensors"]["layers.0.attention.wqkv_packed.weight"][
                    "shape"
                ],
                [32, 16],
            )
            self.assertEqual(
                report["tensors"]["lm_head.splits.0.weight"]["shape"],
                [16, 16],
            )

    def test_cli_tensorize_parameters_dry_run_reports_dtype_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
            report_json = root / "tensorize_report.json"
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

            exit_code = main(
                [
                    "tensorize-parameters",
                    "--program-dir",
                    str(program_dir),
                    "--roles",
                    "mlp,lm_head",
                    "--layers",
                    "0",
                    "--device",
                    "p150a",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["tensor_count"], 11)
            records = {record["path"]: record for record in report["tensors"]}
            self.assertEqual(
                records["layers.0.mlp.gate_proj.weight"]["target_dtype"],
                "bfloat4_b",
            )
            self.assertEqual(
                records["layers.0.mlp.down_proj.weight"]["target_dtype"],
                "bfloat8_b",
            )
            self.assertEqual(
                records["lm_head.splits.0.weight"]["layout"],
                "tile",
            )

    def test_cli_tensorize_parameters_dry_run_reports_full_decode_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
            report_json = root / "tensorize_report.json"
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

            exit_code = main(
                [
                    "tensorize-parameters",
                    "--program-dir",
                    str(program_dir),
                    "--roles",
                    "embedding,norm,attention,mlp,lm_head",
                    "--layers",
                    "0",
                    "--device",
                    "p150a",
                    "--dry-run",
                    "--out",
                    str(report_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_json.read_text())
            self.assertEqual(
                report["roles"],
                ["embedding", "norm", "attention", "mlp", "lm_head"],
            )
            self.assertEqual(report["tensor_count"], 17)
            records = {record["path"]: record for record in report["tensors"]}
            self.assertEqual(
                records["embedding.weight"]["target_dtype"],
                "bfloat16",
            )
            self.assertEqual(
                records["layers.0.input_norm.weight"]["layout"],
                "row_major",
            )
            self.assertEqual(
                records["layers.0.attention.wqkv_packed.weight"][
                    "source_keys"
                ],
                [
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                ],
            )
            self.assertEqual(
                records["layers.0.attention.o_proj.weight"]["target_dtype"],
                "bfloat8_b",
            )
            self.assertEqual(
                records["final_norm.weight"]["layout"],
                "row_major",
            )

    def test_to_ttnn_parameters_uses_role_dtype_and_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
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

            with _fake_torch_and_safetensors():
                torch_params = load_llama_parameters_from_manifests(
                    model_path=model_dir,
                    weights_manifest=program_dir / "weights_manifest.json",
                    config=program_dir / "config.json",
                    tensor_backend="torch",
                    layers=[0],
                )

            fake_ttnn = FakeTTNN()
            result = to_ttnn_parameters(
                torch_params,
                device="device0",
                parameter_config=load_parameter_config_from_program(program_dir),
                roles=["mlp", "lm_head"],
                layers=[0],
                ttnn_module=fake_ttnn,
            )

            self.assertEqual(result.report["status"], "pass")
            self.assertEqual(result.report["tensor_count"], 11)
            self.assertEqual(len(fake_ttnn.calls), 11)
            self.assertEqual(
                result.parameters.layers[0].mlp.gate_proj.weight.dtype,
                "ttnn.bfloat4_b",
            )
            self.assertEqual(
                result.parameters.layers[0].mlp.down_proj.weight.dtype,
                "ttnn.bfloat8_b",
            )
            self.assertEqual(
                result.parameters.lm_head.splits[0].weight.layout,
                "ttnn.TILE_LAYOUT",
            )

    def test_to_ttnn_parameters_covers_generated_decode_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            program_dir = root / "program"
            config_json = root / "template_config.json"
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

            with _fake_torch_and_safetensors():
                torch_params = load_llama_parameters_from_manifests(
                    model_path=model_dir,
                    weights_manifest=program_dir / "weights_manifest.json",
                    config=program_dir / "config.json",
                    tensor_backend="torch",
                    layers=[0],
                )

            fake_ttnn = FakeTTNN()
            result = to_ttnn_parameters(
                torch_params,
                device="device0",
                parameter_config=load_parameter_config_from_program(program_dir),
                roles=["embedding", "norm", "attention", "mlp", "lm_head"],
                layers=[0],
                ttnn_module=fake_ttnn,
            )

            self.assertEqual(result.report["status"], "pass")
            self.assertEqual(result.report["tensor_count"], 17)
            self.assertEqual(len(fake_ttnn.calls), 17)
            self.assertIsNone(result.parameters.layers[1])
            self.assertEqual(
                result.parameters.embedding.weight.dtype,
                "ttnn.bfloat16",
            )
            self.assertEqual(
                result.parameters.embedding.weight.layout,
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(
                result.parameters.layers[0].input_norm.weight.layout,
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(
                result.parameters.layers[0].attention.wqkv_packed.weight.dtype,
                "ttnn.bfloat8_b",
            )
            self.assertEqual(
                result.parameters.layers[0].attention.o_proj.weight.layout,
                "ttnn.TILE_LAYOUT",
            )
            self.assertEqual(
                result.parameters.final_norm.weight.dtype,
                "ttnn.bfloat16",
            )


class FakeTensor:
    def __init__(self, name: str, shape: list[int], dtype: str):
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype

    def __getitem__(self, key: Any) -> "FakeTensor":
        if not isinstance(key, tuple) or len(key) != 2:
            raise AssertionError(f"unexpected fake tensor slice: {key!r}")
        vocab_slice, hidden_slice = key
        if hidden_slice != slice(None, None, None):
            raise AssertionError(f"unexpected hidden slice: {hidden_slice!r}")
        start = int(vocab_slice.start or 0)
        stop = int(vocab_slice.stop)
        return FakeTensor(
            f"{self.name}[{start}:{stop}]",
            [stop - start, self.shape[1]],
            self.dtype,
        )


class FakeTTNNTensor:
    def __init__(self, source: FakeTensor, dtype: str, layout: str, device: str):
        self.source = source
        self.shape = list(source.shape)
        self.dtype = dtype
        self.layout = layout
        self.device = device


class FakeTTNN:
    bfloat4_b = "ttnn.bfloat4_b"
    bfloat8_b = "ttnn.bfloat8_b"
    bfloat16 = "ttnn.bfloat16"
    TILE_LAYOUT = "ttnn.TILE_LAYOUT"
    ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def from_torch(
        self,
        tensor: FakeTensor,
        *,
        device: str,
        dtype: str,
        layout: str,
    ) -> FakeTTNNTensor:
        self.calls.append(
            {
                "tensor": tensor.name,
                "device": device,
                "dtype": dtype,
                "layout": layout,
            }
        )
        return FakeTTNNTensor(tensor, dtype, layout, device)


class FakeSafeOpen:
    def __init__(self, path: str, *_args: Any, **_kwargs: Any):
        self.path = Path(path)
        self.header = _read_fake_safetensors_header(self.path)

    def __enter__(self) -> "FakeSafeOpen":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def get_tensor(self, key: str) -> FakeTensor:
        entry = self.header[key]
        return FakeTensor(
            key,
            [int(dim) for dim in entry["shape"]],
            f"torch.{entry['dtype']}",
        )


@contextmanager
def _fake_torch_and_safetensors():
    old_modules = {
        name: sys.modules.get(name)
        for name in ("torch", "safetensors", "safetensors.torch")
    }

    def fake_cat(tensors: list[FakeTensor], *, dim: int) -> FakeTensor:
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        return FakeTensor("cat", shape, tensors[0].dtype)

    torch_module = types.ModuleType("torch")
    torch_module.cat = fake_cat
    safetensors_module = types.ModuleType("safetensors")
    safetensors_module.safe_open = FakeSafeOpen
    safetensors_torch_module = types.ModuleType("safetensors.torch")
    safetensors_torch_module.load_file = None

    sys.modules["torch"] = torch_module
    sys.modules["safetensors"] = safetensors_module
    sys.modules["safetensors.torch"] = safetensors_torch_module
    try:
        yield
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-parameter-materializer",
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "tie_word_embeddings": False,
            }
        )
    )


def _fake_weight_specs() -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {"dtype": "F32", "shape": [128, 16]},
        "model.norm.weight": {"dtype": "F32", "shape": [16]},
        "lm_head.weight": {"dtype": "F32", "shape": [128, 16]},
    }
    for layer_id in range(2):
        prefix = f"model.layers.{layer_id}"
        specs.update(
            {
                f"{prefix}.input_layernorm.weight": {
                    "dtype": "F32",
                    "shape": [16],
                },
                f"{prefix}.self_attn.q_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 16],
                },
                f"{prefix}.self_attn.k_proj.weight": {
                    "dtype": "F32",
                    "shape": [8, 16],
                },
                f"{prefix}.self_attn.v_proj.weight": {
                    "dtype": "F32",
                    "shape": [8, 16],
                },
                f"{prefix}.self_attn.o_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 16],
                },
                f"{prefix}.post_attention_layernorm.weight": {
                    "dtype": "F32",
                    "shape": [16],
                },
                f"{prefix}.mlp.gate_proj.weight": {
                    "dtype": "F32",
                    "shape": [32, 16],
                },
                f"{prefix}.mlp.up_proj.weight": {
                    "dtype": "F32",
                    "shape": [32, 16],
                },
                f"{prefix}.mlp.down_proj.weight": {
                    "dtype": "F32",
                    "shape": [16, 32],
                },
            }
        )
    return specs


def _write_template_config(path: Path) -> None:
    path.write_text(
        json.dumps(
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
                "lm_head_split_count": 8,
                "dtype_recipe": "official_like_performance_seed",
            }
        )
    )


def _write_fake_model_weights(
    model_dir: Path,
    specs: dict[str, dict[str, object]],
) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: shard_name for key in specs},
            }
        )
    )
    _write_fake_safetensors(model_dir / shard_name, specs)


def _write_fake_safetensors(
    path: Path,
    specs: dict[str, dict[str, object]],
) -> None:
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for key, spec in specs.items():
        shape = [int(dim) for dim in spec["shape"]]
        dtype = str(spec["dtype"])
        byte_size = _numel(shape) * _dtype_size(dtype)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * offset
    )


def _read_fake_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    return header


def _dtype_size(dtype: str) -> int:
    return {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "U64": 8,
        "I32": 4,
        "U32": 4,
        "I16": 2,
        "U16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }[dtype]


def _numel(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


if __name__ == "__main__":
    unittest.main()
