from __future__ import annotations

import importlib
import json
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

from ..codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
)
from ..codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
    to_ttnn_parameters,
)
from ..runtime.tokenizer import PromptTokenizationError, tokenize_prompt_for_decode
from ..runtime.config_runtime import realize_ttnn_config
from ..runtime.device import managed_ttnn_device as _maybe_managed_device
from ..runtime.model_loader import (
    load_generated_model as _runtime_load_generated_model,
    to_namespace as _runtime_to_namespace,
)
from ..runtime.model_setup import materialization_summary, tensorization_summary
from ..runtime.plans import DECODE_PARAMETER_ROLES
from ..runtime.tensor_meta import (
    runtime_int_tensor as _runtime_runtime_int_tensor,
    tensor_dtype as _runtime_tensor_dtype,
    tensor_shape as _runtime_tensor_shape,
)
from ..ttnn_compat import UnsupportedTTNNOp
from .support import (
    NO_TTNN_DEVICE_MESSAGE,
    NoTTNNDeviceError,
    dtype as _dtype,
    dtype_check as _dtype_check,
    dry_run_reference as _dry_run_reference,
    failed_diagnostic_report as _failed_diagnostic_report,
    generated_model as _load_generated_model,
    generated_observed_op_sequence as _generated_observed_op_sequence,
    observed_op_sequence as _observed_op_sequence,
    op_sequence_coverage_check as _op_sequence_coverage_check,
    runtime_int_tensor as _runtime_int_tensor,
    shape as _shape,
    shape_check as _shape_check,
    tensor_shape as _tensor_shape,
    to_namespace as _to_namespace,
    value_check as _value_check,
    write_report as _write_report,
)
DECODE_SHELL_OPS = [
    "embedding",
    "rms_norm.mlp",
    "mlp_gate",
    "mlp_up",
    "mul_silu",
    "mlp_down",
    "residual_add",
    "rms_norm.final",
    "split_lm_head",
    "argmax_or_sampling",
]


NUMERIC_REFERENCE_NOT_RUN_REASON = (
    "No torch numeric reference is executed by this smoke path yet; "
    "the reference evidence is limited to generated-path structure, "
    "shape, and dtype checks."
)


def run_smoke_decode_shell(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int,
    disable_attention: bool,
    device: str,
    device_id: int = 0,
    model_path: str | Path | None = None,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    parameters: Any | None = None,
    reference_parameters: Any | None = None,
    token_ids: Any | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
    pcc_threshold: float = 0.99,
) -> dict[str, Any]:
    if layers <= 0:
        raise ValueError("layers must be positive")
    if not disable_attention:
        raise ValueError(
            "smoke-decode-shell PR-D supports --disable-attention only"
        )

    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    config = dict(config)
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config["batch_size"] = int(batch_size)
    if cache_len is not None:
        if cache_len <= 0:
            raise ValueError("cache_len must be positive")
        config["max_cache_len"] = int(cache_len)
    layer_count = min(layers, int(config["num_layers"]))
    if dry_run:
        report = _base_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            dry_run=True,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "reference": _dry_run_reference("decode_shell"),
                "input_source": "planned",
                "input_shapes": _decode_shell_input_shapes(config),
                "runtime_input_tensor_count": 0,
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_report(out, report)
        return report

    try:
        ttnn = (
            ttnn_module
            if ttnn_module is not None
            else importlib.import_module("ttnn")
        )
    except ImportError as err:
        report = _no_device_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            shell_params = parameters
            host_reference_params = reference_parameters
            parameter_setup = None
            parameter_source = "injected" if shell_params is not None else "hf_model"
            if shell_params is None:
                if model_path is None:
                    raise ValueError(
                        "--model-path is required for non-dry-run decode shell"
                    )
                host_params = load_llama_parameters_from_manifests(
                    model_path=model_path,
                    weights_manifest=program_root / "weights_manifest.json",
                    config=program_root / "config.json",
                    tensor_backend="torch",
                    layers=range(layer_count),
                )
                if host_reference_params is None:
                    host_reference_params = host_params
                tensorization_result = _tensorize_shell_parameters(
                    ttnn,
                    host_params,
                    load_parameter_config_from_program(program_root),
                    ttnn_device,
                    layer_count,
                )
                assert tensorization_result.parameters is not None
                shell_params = tensorization_result.parameters
                parameter_setup = {
                    "materialization": materialization_summary(host_params),
                    "tensorization": tensorization_summary(
                        tensorization_result.report
                    ),
                }
            runtime_token_ids = token_ids
            runtime_input_tensor_count = 0
            prompt_runtime_input_tensor_count = 0
            prompt_tokenization = None
            input_source = "injected"
            if runtime_token_ids is None:
                torch = _load_torch_for_runtime_inputs(torch_module)
                if prompt is not None:
                    prompt_runtime = _build_prompt_decode_shell_inputs(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        config=config,
                        prompt=prompt,
                        tokenizer_path=(
                            tokenizer_path
                            or model_path
                            or program_root
                        ),
                        tokenizer_module=tokenizer_module,
                    )
                    runtime_token_ids = prompt_runtime.token_ids
                    prompt_runtime_input_tensor_count = (
                        prompt_runtime.tensor_conversion_count
                    )
                    prompt_tokenization = prompt_runtime.prompt_tokenization
                    input_source = "prompt_runtime"
                else:
                    runtime_input = _build_synthetic_decode_shell_inputs(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        config=config,
                    )
                    runtime_token_ids = runtime_input.token_ids
                    runtime_input_tensor_count = (
                        runtime_input.tensor_conversion_count
                    )
                    input_source = "synthetic"
            report = _run_generated_decode_shell(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=shell_params,
                config=config,
                device=ttnn_device,
                layer_count=layer_count,
                token_ids=runtime_token_ids,
                torch_module=torch_module,
                reference_parameters=host_reference_params,
                pcc_threshold=pcc_threshold,
                realize_config=ttnn_module is None,
            )
            report["parameter_source"] = parameter_source
            report["input_source"] = input_source
            report["input_shapes"] = {
                "token_ids": (
                    _shape(runtime_token_ids)
                    or _decode_shell_input_shapes(config)["token_ids"]
                )
            }
            report["runtime_input_tensor_count"] = runtime_input_tensor_count
            report["synthetic_runtime_input_tensor_count"] = (
                runtime_input_tensor_count
            )
            report["prompt_runtime_input_tensor_count"] = (
                prompt_runtime_input_tensor_count
            )
            if prompt_tokenization is not None:
                report["prompt_tokenization"] = prompt_tokenization
            if parameter_setup is not None:
                report["parameter_setup"] = parameter_setup
            report.update(
                {
                    **_base_report(
                        program_dir=program_root,
                        device=device,
                        device_id=device_id,
                        layers=layer_count,
                        disable_attention=disable_attention,
                        dry_run=False,
                    ),
                    **report,
                }
            )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            detail=str(err),
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        ValueError,
        PromptTokenizationError,
    ) as err:
        report = _failed_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            status="configuration_error",
            message=str(err),
        )
    except Exception as err:
        report = _failed_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
        )

    _write_report(out, report)
    return report


def _run_generated_decode_shell(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    device: Any,
    layer_count: int,
    token_ids: Any | None,
    torch_module: Any | None,
    reference_parameters: Any | None,
    pcc_threshold: float,
    realize_config: bool = False,
) -> dict[str, Any]:
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    model_config = realize_ttnn_config(config, ttnn) if realize_config else config
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(model_config),
    )
    model.ops.enable_recording()
    if token_ids is None:
        raise ValueError("token_ids must be provided for decode shell execution")

    start = time.perf_counter()
    hidden = model.embed(token_ids)
    layer_reports = []
    for layer_id in range(layer_count):
        hidden = model.ops.reshape_decode_hidden_for_layer(
            hidden,
            op_name="reshape_hidden_decode",
        )
        residual = hidden
        normalized = model.rmsnorm(hidden, layer_id, kind="mlp")
        mlp_out = model.mlp_decode(layer_id, normalized)
        hidden = model.ops.add(residual, mlp_out)
        layer_reports.append(
            {
                "layer_id": layer_id,
                "attention": "disabled",
                "output_shape": _shape(hidden),
                "output_dtype": _dtype(hidden),
            }
        )
    hidden = model.final_norm(hidden)
    final_hidden = hidden
    token = model.lm_head_argmax(hidden)
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    output = {
        "shape": _shape(token),
        "dtype": _dtype(token),
        "repr": repr(token),
    }
    reference = _decode_shell_reference(
        ttnn=ttnn,
        torch_module=torch_module,
        config=config,
        layer_reports=layer_reports,
        output=output,
        final_hidden=final_hidden,
        token=token,
        reference_parameters=reference_parameters,
        token_ids=token_ids,
        observed_ops=_generated_observed_op_sequence(model, ttnn),
        layer_count=layer_count,
        pcc_threshold=pcc_threshold,
    )
    passed = bool(reference["passed"])

    return {
        "passed": passed,
        "status": "passed" if passed else "reference_mismatch",
        "latency_ms": latency_ms,
        "layers": layer_reports,
        "output": output,
        "reference": reference,
        "error": None if passed else "decode shell structural reference mismatch",
    }


def _tensorize_shell_parameters(
    ttnn: Any,
    host_params: Any,
    parameter_config: dict[str, Any],
    device: Any,
    layer_count: int,
) -> Any:
    return to_ttnn_parameters(
        host_params,
        device,
        parameter_config,
        roles=DECODE_PARAMETER_ROLES,
        layers=range(layer_count),
        ttnn_module=ttnn,
    )


def _load_torch_for_runtime_inputs(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        return importlib.import_module("torch")
    except ImportError as err:
        raise ValueError(
            "torch is required to synthesize decode shell token_ids"
        ) from err


def _build_synthetic_decode_shell_inputs(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    config: dict[str, Any],
) -> SimpleNamespace:
    if not hasattr(ttnn, "from_torch"):
        raise ValueError(
            "ttnn module must provide from_torch to synthesize token_ids"
        )
    shape = _decode_shell_input_shapes(config)["token_ids"]
    host_tensor = _zeros_token_ids(torch, shape)
    try:
        host_tensor.name = "token_ids"
    except AttributeError:
        pass
    kwargs = {"device": device}
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    token_ids = ttnn.from_torch(host_tensor, **kwargs)
    return SimpleNamespace(
        token_ids=token_ids,
        tensor_conversion_count=1,
    )


def _build_prompt_decode_shell_inputs(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    config: dict[str, Any],
    prompt: str,
    tokenizer_path: str | Path,
    tokenizer_module: Any | None,
) -> SimpleNamespace:
    if not hasattr(ttnn, "from_torch"):
        raise ValueError(
            "ttnn module must provide from_torch to build prompt token_ids"
        )
    tokenization = tokenize_prompt_for_decode(
        prompt=prompt,
        batch_size=int(config["batch_size"]),
        tokenizer_path=tokenizer_path,
        vocab_size=int(config.get("vocab_size", 0)) or None,
        tokenizer_module=tokenizer_module,
    )
    host_tensor = _token_ids_tensor(
        torch,
        tokenization.token_ids,
        name="prompt_token_ids",
    )
    kwargs = {"device": device}
    dtype = getattr(
        ttnn,
        "uint32",
        getattr(ttnn, "int32", getattr(ttnn, "bfloat16", None)),
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    token_ids = ttnn.from_torch(host_tensor, **kwargs)
    return SimpleNamespace(
        token_ids=token_ids,
        tensor_conversion_count=1,
        prompt_tokenization=tokenization.to_report(),
    )


def _zeros_token_ids(torch: Any, shape: list[int]) -> Any:
    zeros = getattr(torch, "zeros", None)
    if not callable(zeros):
        raise ValueError("torch module must provide zeros")
    dtype = getattr(torch, "int32", None)
    if dtype is None:
        dtype = getattr(torch, "long", None)
    if dtype is None:
        return zeros(shape)
    try:
        return zeros(shape, dtype=dtype)
    except TypeError:
        return zeros(shape)


def _token_ids_tensor(
    torch: Any,
    token_ids: list[list[int]],
    *,
    name: str,
) -> Any:
    return _runtime_int_tensor(torch, token_ids, name=name)


def _decode_shell_input_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "token_ids": [int(config["batch_size"]), int(config["seq_len"])]
    }


def _decode_shell_reference(
    *,
    ttnn: Any,
    torch_module: Any | None,
    config: dict[str, Any],
    layer_reports: list[dict[str, Any]],
    output: dict[str, Any],
    final_hidden: Any,
    token: Any,
    reference_parameters: Any | None,
    token_ids: Any,
    observed_ops: list[str] | None,
    layer_count: int,
    pcc_threshold: float,
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    seq_len = int(config["seq_len"])
    hidden_size = int(config["hidden_size"])
    expected_hidden_shape = [batch_size, seq_len, hidden_size]
    accepted_hidden_shapes = [
        expected_hidden_shape,
        [1, batch_size, seq_len, hidden_size],
        [1, seq_len, batch_size, hidden_size],
    ]
    accepted_token_shapes = [
        [batch_size, seq_len],
        [batch_size],
        [1, batch_size, seq_len],
        [1, batch_size],
    ]
    checks: list[dict[str, Any]] = [
        _value_check("layer_count", len(layer_reports), layer_count),
        _shape_check(
            "output.token",
            output.get("shape"),
            accepted=accepted_token_shapes,
        ),
        _dtype_check("output.token", output.get("dtype")),
    ]
    for layer_report in layer_reports:
        layer_id = int(layer_report["layer_id"])
        checks.extend(
            [
                _value_check(
                    f"layers.{layer_id}.attention",
                    layer_report.get("attention"),
                    "disabled",
                ),
                _shape_check(
                    f"layers.{layer_id}.output",
                    layer_report.get("output_shape"),
                    expected=expected_hidden_shape,
                    accepted=accepted_hidden_shapes,
                ),
                _dtype_check(
                    f"layers.{layer_id}.output",
                    layer_report.get("output_dtype"),
                ),
            ]
        )

    numeric_reference = _decode_shell_numeric_reference(
        ttnn=ttnn,
        torch_module=torch_module,
        config=config,
        reference_parameters=reference_parameters,
        token_ids=token_ids,
        final_hidden=final_hidden,
        token=token,
        layer_count=layer_count,
        pcc_threshold=pcc_threshold,
    )
    passed = all(check["passed"] for check in checks)
    if numeric_reference.get("passed") is False:
        passed = False
    checks.append(
        _op_sequence_coverage_check(
            planned_ops=list(DECODE_SHELL_OPS),
            observed_ops=observed_ops,
        )
    )
    passed = passed and checks[-1]["passed"]
    return {
        "kind": "structural_shape_dtype_op_sequence",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": numeric_reference,
        "planned_ops": list(DECODE_SHELL_OPS),
        "observed_ops": observed_ops,
        "checks": checks,
    }


def _decode_shell_numeric_reference(
    *,
    ttnn: Any,
    torch_module: Any | None,
    config: dict[str, Any],
    reference_parameters: Any | None,
    token_ids: Any,
    final_hidden: Any,
    token: Any,
    layer_count: int,
    pcc_threshold: float,
) -> dict[str, Any]:
    if reference_parameters is None:
        return _numeric_reference_not_run(
            "Host torch parameters are not available for this smoke run."
        )
    try:
        torch = _load_torch_for_runtime_inputs(torch_module)
        reference_token_ids = _to_torch_tensor(ttnn, token_ids)
        observed_hidden = _to_torch_tensor(ttnn, final_hidden)
        if reference_token_ids is None:
            return _numeric_reference_not_run(
                "token_ids could not be converted to a torch tensor."
            )
        if observed_hidden is None:
            return _numeric_reference_not_run(
                "final hidden output could not be converted to a torch tensor."
            )
        expected = _torch_decode_shell_reference(
            torch=torch,
            parameters=reference_parameters,
            token_ids=reference_token_ids,
            config=config,
            layer_count=layer_count,
        )
        pcc = _pearson_corr(
            observed_hidden,
            expected.final_hidden,
        )
        expected_hidden_shape = _shape(expected.final_hidden)
        accepted_hidden_shapes = [expected_hidden_shape]
        if expected_hidden_shape is not None and len(expected_hidden_shape) == 3:
            accepted_hidden_shapes.extend(
                [
                    [1, *expected_hidden_shape],
                    [
                        1,
                        expected_hidden_shape[1],
                        expected_hidden_shape[0],
                        expected_hidden_shape[2],
                    ],
                ]
            )
        checks = [
            {
                "name": "final_hidden.pcc",
                "type": "pcc",
                "actual": pcc,
                "threshold": pcc_threshold,
                "passed": pcc is not None and pcc >= pcc_threshold,
            },
            _shape_check(
                "final_hidden",
                _shape(observed_hidden),
                expected=expected_hidden_shape,
                accepted=accepted_hidden_shapes,
            ),
            _dtype_check(
                "final_hidden",
                _dtype(observed_hidden),
            ),
        ]
        observed_token = _to_torch_tensor(ttnn, token)
        if observed_token is not None:
            checks.append(
                {
                    "name": "token.match",
                    "type": "value",
                    "actual": _flatten_values(observed_token),
                    "expected": _flatten_values(expected.token),
                    "passed": _flatten_values(observed_token)
                    == _flatten_values(expected.token),
                }
            )
        passed = all(check["passed"] for check in checks)
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "kind": "torch_decode_shell",
            "pcc": pcc,
            "pcc_threshold": pcc_threshold,
            "checks": checks,
        }
    except Exception as err:
        return _numeric_reference_not_run(
            f"{type(err).__name__}: {err}"
        )


def _numeric_reference_not_run(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "passed": None,
        "reason": reason,
    }


def _torch_decode_shell_reference(
    *,
    torch: Any,
    parameters: Any,
    token_ids: Any,
    config: dict[str, Any],
    layer_count: int,
) -> SimpleNamespace:
    eps = float(config.get("rms_norm", {}).get("eps") or 1e-5)
    hidden = _torch_embedding(
        torch,
        token_ids,
        _parameter_weight(parameters.embedding),
    )
    for layer_id in range(layer_count):
        layer = parameters.layers[layer_id]
        residual = hidden
        hidden = _torch_rms_norm(
            torch,
            hidden,
            _parameter_weight(layer.post_attention_norm),
            eps,
        )
        mlp = layer.mlp
        gate = _torch_linear(torch, hidden, _parameter_weight(mlp.gate_proj))
        up = _torch_linear(torch, hidden, _parameter_weight(mlp.up_proj))
        hidden = _torch_silu(torch, gate) * up
        hidden = _torch_linear(torch, hidden, _parameter_weight(mlp.down_proj))
        hidden = residual + hidden
    final_hidden = _torch_rms_norm(
        torch,
        hidden,
        _parameter_weight(parameters.final_norm),
        eps,
    )
    logits = _torch_cat(
        torch,
        [
            _torch_linear(torch, final_hidden, _parameter_weight(split))
            for split in parameters.lm_head.splits
        ],
        dim=-1,
    )
    token = _torch_argmax(torch, logits, dim=-1)
    return SimpleNamespace(final_hidden=final_hidden, token=token)


def _parameter_weight(parameter: Any) -> Any:
    weight = getattr(parameter, "weight", parameter)
    torch_value = getattr(weight, "torch_value", None)
    return torch_value if torch_value is not None else weight


def _torch_embedding(torch: Any, token_ids: Any, weight: Any) -> Any:
    weight = _torch_embedding_weight(weight)
    to_long = getattr(token_ids, "long", None)
    if callable(to_long):
        token_ids = to_long()
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    embedding = getattr(functional, "embedding", None)
    if callable(embedding):
        return embedding(token_ids, weight)
    return weight[token_ids]


def _torch_linear(torch: Any, activation: Any, weight: Any) -> Any:
    return activation @ _torch_linear_weight_for_matmul(weight)


def _torch_rms_norm(torch: Any, hidden: Any, weight: Any, eps: float) -> Any:
    weight = _torch_norm_weight(weight, hidden)
    variance = (hidden * hidden).mean(dim=-1, keepdim=True)
    return hidden * torch.rsqrt(variance + eps) * weight


def _torch_silu(torch: Any, tensor: Any) -> Any:
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    silu = getattr(functional, "silu", None)
    if callable(silu):
        return silu(tensor)
    return tensor / (1 + torch.exp(-tensor))


def _torch_cat(torch: Any, tensors: list[Any], *, dim: int) -> Any:
    return torch.cat(tensors, dim=dim)


def _torch_argmax(torch: Any, tensor: Any, *, dim: int) -> Any:
    return torch.argmax(tensor, dim=dim)


def _torch_embedding_weight(weight: Any) -> Any:
    shape = _shape(weight)
    if _is_physical_4d_weight(shape):
        return _reshape_reference_tensor(weight, [shape[2], shape[3]])
    return weight


def _torch_linear_weight_for_matmul(weight: Any) -> Any:
    shape = _shape(weight)
    if _is_physical_4d_weight(shape):
        return _reshape_reference_tensor(weight, [shape[2], shape[3]])
    transpose = getattr(weight, "transpose", None)
    if callable(transpose):
        return transpose(-1, -2)
    raise TypeError("linear reference weight must provide transpose")


def _torch_norm_weight(weight: Any, hidden: Any) -> Any:
    shape = _shape(weight)
    if _is_physical_4d_weight(shape):
        hidden_shape = _shape(hidden)
        hidden_size = hidden_shape[-1] if hidden_shape else shape[2] * shape[3]
        return _reshape_reference_tensor(weight, [hidden_size])
    return weight


def _is_physical_4d_weight(shape: list[int] | None) -> bool:
    return (
        isinstance(shape, list)
        and len(shape) == 4
        and shape[0] == 1
        and shape[1] == 1
    )


def _reshape_reference_tensor(tensor: Any, shape: list[int]) -> Any:
    reshape = getattr(tensor, "reshape", None)
    if callable(reshape):
        return reshape(*shape)
    raise TypeError("reference tensor must provide reshape for physical 4D weights")


def _to_torch_tensor(ttnn: Any, tensor: Any) -> Any | None:
    torch_value = getattr(tensor, "torch_value", None)
    if torch_value is not None:
        return torch_value
    to_torch = getattr(ttnn, "to_torch", None)
    if callable(to_torch):
        return to_torch(tensor)
    return tensor if _looks_like_torch_tensor(tensor) else None


def _looks_like_torch_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "tolist")


def _pearson_corr(observed: Any, expected: Any) -> float | None:
    lhs = [float(value) for value in _flatten_values(observed)]
    rhs = [float(value) for value in _flatten_values(expected)]
    if len(lhs) != len(rhs) or not lhs:
        return None
    if len(lhs) == 1:
        return 1.0 if lhs[0] == rhs[0] else 0.0
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    lhs_centered = [value - lhs_mean for value in lhs]
    rhs_centered = [value - rhs_mean for value in rhs]
    numerator = sum(
        left * right for left, right in zip(lhs_centered, rhs_centered)
    )
    lhs_norm = sum(value * value for value in lhs_centered) ** 0.5
    rhs_norm = sum(value * value for value in rhs_centered) ** 0.5
    denominator = lhs_norm * rhs_norm
    if denominator == 0.0:
        return 1.0 if lhs == rhs else 0.0
    return numerator / denominator


def _flatten_values(tensor: Any) -> list[Any]:
    value = tensor
    for name in ("detach", "cpu"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            value = reshape(-1)
        except TypeError:
            value = reshape((-1,))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        flattened = tolist()
    else:
        flattened = value
    return _flatten_nested(flattened)


def _flatten_nested(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for element in value:
            items.extend(_flatten_nested(element))
        return items
    return [value]


def _base_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "decode_shell",
        "program_dir": str(kwargs["program_dir"]),
        "device": kwargs["device"],
        "device_id": kwargs["device_id"],
        "layers_requested": kwargs["layers"],
        "disable_attention": kwargs["disable_attention"],
        "dry_run": kwargs["dry_run"],
        "ttnn_ops": list(DECODE_SHELL_OPS),
    }


def _error_report(**kwargs: Any) -> dict[str, Any]:
    error_keys = {"status", "message", "detail", "ttnn_version", "ttnn_module"}
    base = _base_report(
        **{key: value for key, value in kwargs.items() if key not in error_keys},
        dry_run=False,
    )
    return _failed_diagnostic_report(
        base,
        status=kwargs["status"],
        message=kwargs["message"],
        detail=kwargs.get("detail", kwargs["message"]),
        ttnn_version=kwargs.get("ttnn_version"),
        ttnn_module=kwargs.get("ttnn_module"),
    )


def _no_device_report(**kwargs: Any) -> dict[str, Any]:
    kwargs.update(status="no_device", message=NO_TTNN_DEVICE_MESSAGE)
    return _error_report(**kwargs)


def _failed_report(**kwargs: Any) -> dict[str, Any]:
    return _error_report(**kwargs)
