from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_decode_shell import (
    DECODE_SHELL_OPS,
    run_smoke_decode_shell,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_smoke_attention_primitive import (
    _fake_torch,
)


class SmokeDecodeShellTest(unittest.TestCase):
    def test_cli_smoke_decode_shell_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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
                    "smoke-decode-shell",
                    "--program-dir",
                    str(program_dir),
                    "--layers",
                    "1",
                    "--disable-attention",
                    "--device",
                    "p150a",
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
            self.assertEqual(report["ttnn_ops"], DECODE_SHELL_OPS)
            self.assertEqual(report["input_shapes"]["token_ids"], [32, 1])
            self.assertEqual(report["input_source"], "planned")
            self.assertEqual(report["reference"]["status"], "dry_run")
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )

    def test_run_smoke_decode_shell_executes_fake_attention_disabled_shell(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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
            report = run_smoke_decode_shell(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake_ttnn,
                parameters=_fake_parameters(split_count=8),
                token_ids="token_ids",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["layers"][0]["attention"], "disabled")
            self.assertEqual(report["input_source"], "injected")
            self.assertEqual(report["runtime_input_tensor_count"], 0)
            self.assertEqual(report["reference"]["status"], "passed")
            self.assertEqual(
                report["reference"]["kind"],
                "structural_shape_dtype_op_sequence",
            )
            self.assertEqual(
                report["reference"]["numeric_reference"]["status"],
                "not_run",
            )
            self.assertEqual(report["reference"]["observed_ops"][0], "embedding")
            op_check = next(
                check
                for check in report["reference"]["checks"]
                if check["name"] == "observed_op_sequence"
            )
            self.assertTrue(op_check["passed"])
            self.assertEqual(op_check["expected"], DECODE_SHELL_OPS)
            self.assertIn("mlp_gate", report["reference"]["observed_ops"])
            self.assertTrue(
                all(check["passed"] for check in report["reference"]["checks"])
            )
            self.assertEqual(json.loads(report_json.read_text()), report)
            self.assertEqual(
                [call["op"] for call in fake_ttnn.calls],
                [
                    "embedding",
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
            self.assertNotIn(
                "paged_scaled_dot_product_attention_decode",
                [call["op"] for call in fake_ttnn.calls],
            )

    def test_run_smoke_decode_shell_reports_torch_numeric_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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
            parameters = _fake_numeric_parameters(split_count=8)
            report = run_smoke_decode_shell(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake_ttnn,
                torch_module=_fake_numeric_torch(),
                parameters=parameters,
                reference_parameters=parameters,
                token_ids=MiniTensor([[0] for _ in range(32)], dtype="int64"),
            )

            self.assertTrue(report["passed"])
            numeric = report["reference"]["numeric_reference"]
            self.assertEqual(numeric["status"], "passed")
            self.assertEqual(numeric["kind"], "torch_decode_shell")
            self.assertGreaterEqual(numeric["pcc"], 0.999999)
            self.assertTrue(all(check["passed"] for check in numeric["checks"]))
            self.assertIn("mlp_down", report["reference"]["observed_ops"])
            self.assertEqual(json.loads(report_json.read_text()), report)

    def test_run_smoke_decode_shell_synthesizes_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "fake_model"
            config_json = root / "template_config.json"
            program_dir = root / "program"
            report_json = root / "decode_shell_report.json"
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
            report = run_smoke_decode_shell(
                out=report_json,
                program_dir=program_dir,
                layers=1,
                disable_attention=True,
                device="p150a",
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
                parameters=_fake_parameters(split_count=8),
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["input_source"], "synthetic")
            self.assertEqual(report["input_shapes"]["token_ids"], [32, 1])
            self.assertEqual(report["runtime_input_tensor_count"], 1)
            self.assertEqual(fake_ttnn.calls[0]["op"], "from_torch")
            self.assertEqual(
                fake_ttnn.calls[0]["kwargs"]["layout"],
                "ttnn.ROW_MAJOR_LAYOUT",
            )
            self.assertEqual(fake_ttnn.calls[1]["op"], "embedding")
            self.assertIsInstance(fake_ttnn.calls[1]["token_ids"], FakeTensor)
            self.assertEqual(json.loads(report_json.read_text()), report)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: list[int] | None = None,
        dtype: str = "bf16",
        mem_config: str | None = None,
        torch_value: Any | None = None,
    ) -> None:
        self.name = name
        self.torch_value = torch_value
        self.shape = (
            list(torch_value.shape)
            if torch_value is not None
            else shape or [32, 1, 16]
        )
        self.dtype = str(torch_value.dtype) if torch_value is not None else dtype
        self._mem_config = mem_config

    def memory_config(self) -> str | None:
        return self._mem_config

    def __repr__(self) -> str:
        return f"FakeTensor({self.name})"


class MiniTensor:
    def __init__(self, values: Any, dtype: str = "float32") -> None:
        np_dtype = np.int64 if dtype.startswith("int") else np.float64
        self.array = np.array(values, dtype=np_dtype)
        self.dtype = dtype
        self.shape = list(self.array.shape)

    def transpose(self, dim0: int, dim1: int) -> "MiniTensor":
        return MiniTensor(np.swapaxes(self.array, dim0, dim1), self.dtype)

    def mean(self, *, dim: int, keepdim: bool) -> "MiniTensor":
        return MiniTensor(
            np.mean(self.array, axis=dim, keepdims=keepdim),
            self.dtype,
        )

    def reshape(self, *shape: int) -> "MiniTensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return MiniTensor(np.reshape(self.array, shape), self.dtype)

    def tolist(self) -> list[Any]:
        return self.array.tolist()

    def detach(self) -> "MiniTensor":
        return self

    def cpu(self) -> "MiniTensor":
        return self

    def __matmul__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array @ _mini_array(other), self.dtype)

    def __add__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array + _mini_array(other), self.dtype)

    def __radd__(self, other: Any) -> "MiniTensor":
        return self.__add__(other)

    def __mul__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array * _mini_array(other), self.dtype)

    def __rmul__(self, other: Any) -> "MiniTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "MiniTensor":
        return MiniTensor(self.array / _mini_array(other), self.dtype)

    def __rtruediv__(self, other: Any) -> "MiniTensor":
        return MiniTensor(_mini_array(other) / self.array, self.dtype)

    def __neg__(self) -> "MiniTensor":
        return MiniTensor(-self.array, self.dtype)


def _mini_array(value: Any) -> np.ndarray:
    if isinstance(value, MiniTensor):
        return value.array
    if isinstance(value, FakeTensor) and value.torch_value is not None:
        return value.torch_value.array
    return np.array(value)


def _fake_parameters(split_count: int):
    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(weight="embed_weight"),
        layers=[
            types.SimpleNamespace(
                input_norm=types.SimpleNamespace(weight="input_norm_weight"),
                post_attention_norm=types.SimpleNamespace(
                    weight="post_attention_norm_weight"
                ),
                mlp=types.SimpleNamespace(
                    gate_proj=types.SimpleNamespace(weight="gate_weight"),
                    up_proj=types.SimpleNamespace(weight="up_weight"),
                    down_proj=types.SimpleNamespace(weight="down_weight"),
                ),
            )
        ],
        final_norm=types.SimpleNamespace(weight="final_norm_weight"),
        lm_head=types.SimpleNamespace(
            splits=[
                types.SimpleNamespace(weight=f"lm_head_{index}")
                for index in range(split_count)
            ]
        ),
    )


def _fake_numeric_parameters(split_count: int):
    hidden_size = 16
    intermediate_size = 32
    vocab_size = 128
    shard_size = vocab_size // split_count

    def matrix(name: str, shape: tuple[int, int]) -> MiniTensor:
        values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        values = (values + 1.0) / (1000.0 + len(name))
        return MiniTensor(values)

    return types.SimpleNamespace(
        embedding=types.SimpleNamespace(
            weight=matrix("embedding", (vocab_size, hidden_size)),
        ),
        layers=[
            types.SimpleNamespace(
                input_norm=types.SimpleNamespace(
                    weight=MiniTensor(np.ones(hidden_size)),
                ),
                post_attention_norm=types.SimpleNamespace(
                    weight=MiniTensor(np.ones(hidden_size) * 0.5),
                ),
                mlp=types.SimpleNamespace(
                    gate_proj=types.SimpleNamespace(
                        weight=matrix("gate", (intermediate_size, hidden_size)),
                    ),
                    up_proj=types.SimpleNamespace(
                        weight=matrix("up", (intermediate_size, hidden_size)),
                    ),
                    down_proj=types.SimpleNamespace(
                        weight=matrix("down", (hidden_size, intermediate_size)),
                    ),
                ),
            )
        ],
        final_norm=types.SimpleNamespace(
            weight=MiniTensor(np.ones(hidden_size) * 0.75),
        ),
        lm_head=types.SimpleNamespace(
            splits=[
                types.SimpleNamespace(
                    weight=matrix(f"lm_head_{index}", (shard_size, hidden_size)),
                )
                for index in range(split_count)
            ]
        ),
    )


def _fake_numeric_torch():
    module = types.SimpleNamespace()
    module.float32 = "float32"
    module.int64 = "int64"

    def embedding(token_ids: MiniTensor, weight: MiniTensor) -> MiniTensor:
        return MiniTensor(weight.array[token_ids.array.astype(np.int64)])

    def silu(tensor: MiniTensor) -> MiniTensor:
        return MiniTensor(tensor.array / (1.0 + np.exp(-tensor.array)))

    module.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(embedding=embedding, silu=silu)
    )
    module.rsqrt = lambda tensor: MiniTensor(1.0 / np.sqrt(tensor.array))
    module.cat = lambda tensors, dim: MiniTensor(
        np.concatenate([tensor.array for tensor in tensors], axis=dim)
    )
    module.argmax = lambda tensor, dim: MiniTensor(
        np.argmax(tensor.array, axis=dim),
        dtype="int64",
    )
    return module


def _make_fake_ttnn():
    module = types.ModuleType("ttnn")
    module.calls = []
    module.ROW_MAJOR_LAYOUT = "ttnn.ROW_MAJOR_LAYOUT"

    class UnaryOpType:
        SILU = "SILU"

    def unary_with_param(op):
        return ("UnaryWithParam", op)

    def embedding(token_ids, weight, **kwargs):
        torch_value = None
        if _has_numeric(token_ids) and _has_numeric(weight):
            torch_value = _fake_numeric_torch().nn.functional.embedding(
                _to_mini(token_ids),
                _to_mini(weight),
            )
        module.calls.append(
            {
                "op": "embedding",
                "token_ids": token_ids,
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            f"embedding:{weight}",
            mem_config=kwargs.get("memory_config"),
            torch_value=torch_value,
        )

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
            dtype=str(getattr(tensor, "dtype", None)),
            torch_value=tensor if isinstance(tensor, MiniTensor) else None,
        )

    def rms_norm(hidden, **kwargs):
        torch_value = None
        if _has_numeric(hidden) and _has_numeric(kwargs.get("weight")):
            hidden_value = _to_mini(hidden)
            weight_value = _to_mini(kwargs["weight"])
            variance = (hidden_value * hidden_value).mean(
                dim=-1,
                keepdim=True,
            )
            torch_value = (
                hidden_value
                * _fake_numeric_torch().rsqrt(
                    variance + float(kwargs.get("epsilon", 1e-5))
                )
                * weight_value
            )
        module.calls.append(
            {
                "op": "rms_norm",
                "hidden": getattr(hidden, "name", hidden),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            f"rms_norm:{kwargs.get('weight')}",
            mem_config=kwargs.get("memory_config"),
            torch_value=torch_value,
        )

    def linear(activation, weight, **kwargs):
        torch_value = None
        if _has_numeric(activation) and _has_numeric(weight):
            torch_value = _to_mini(activation) @ _to_mini(weight).transpose(-1, -2)
        module.calls.append(
            {
                "op": "linear",
                "activation": getattr(activation, "name", activation),
                "weight": weight,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            f"linear:{weight}",
            mem_config=kwargs.get("memory_config"),
            torch_value=torch_value,
        )

    def mul(lhs, rhs, **kwargs):
        torch_value = None
        if _has_numeric(lhs) and _has_numeric(rhs):
            lhs_value = _to_mini(lhs)
            if kwargs.get("input_tensor_a_activations"):
                lhs_value = _fake_numeric_torch().nn.functional.silu(lhs_value)
            torch_value = lhs_value * _to_mini(rhs)
        module.calls.append(
            {
                "op": "mul",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            "mul_out",
            mem_config=kwargs.get("memory_config"),
            torch_value=torch_value,
        )

    def add(lhs, rhs, **kwargs):
        torch_value = None
        if _has_numeric(lhs) and _has_numeric(rhs):
            torch_value = _to_mini(lhs) + _to_mini(rhs)
        module.calls.append(
            {
                "op": "add",
                "lhs": getattr(lhs, "name", lhs),
                "rhs": getattr(rhs, "name", rhs),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            "add_out",
            mem_config=kwargs.get("memory_config"),
            torch_value=torch_value,
        )

    def concat(tensors, **kwargs):
        names = [getattr(tensor, "name", tensor) for tensor in tensors]
        torch_value = None
        if all(_has_numeric(tensor) for tensor in tensors):
            torch_value = _fake_numeric_torch().cat(
                [_to_mini(tensor) for tensor in tensors],
                dim=int(kwargs.get("dim", -1)),
            )
        module.calls.append(
            {
                "op": "concat",
                "tensors": names,
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            "concat:" + ",".join(names),
            torch_value=torch_value,
        )

    def argmax(tensor, **kwargs):
        torch_value = None
        if _has_numeric(tensor):
            torch_value = _fake_numeric_torch().argmax(
                _to_mini(tensor),
                dim=int(kwargs.get("dim", -1)),
            )
        module.calls.append(
            {
                "op": "argmax",
                "tensor": getattr(tensor, "name", tensor),
                "kwargs": dict(kwargs),
            }
        )
        return FakeTensor(
            f"argmax:{getattr(tensor, 'name', tensor)}",
            [32],
            torch_value=torch_value,
        )

    def to_torch(tensor):
        if isinstance(tensor, FakeTensor):
            return tensor.torch_value
        if isinstance(tensor, MiniTensor):
            return tensor
        return None

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
    module.to_torch = to_torch
    return module


def _has_numeric(value: Any) -> bool:
    return isinstance(value, MiniTensor) or (
        isinstance(value, FakeTensor) and value.torch_value is not None
    )


def _to_mini(value: Any) -> MiniTensor:
    if isinstance(value, MiniTensor):
        return value
    if isinstance(value, FakeTensor) and value.torch_value is not None:
        return value.torch_value
    raise TypeError(f"expected numeric fake tensor, got {type(value).__name__}")


def _write_fake_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "fake-decode-shell",
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


if __name__ == "__main__":
    unittest.main()
