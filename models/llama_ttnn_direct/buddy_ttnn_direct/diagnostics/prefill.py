from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..runtime_environment import collect_ttnn_environment
from ..runtime.plans import decode_step_plan as _runtime_decode_step_plan
from ..runtime.plans import prefill_plan as _runtime_prefill_plan
from ..runtime.prefill import run_prefill_prompt as _runtime_run_prefill_prompt
from ..runtime.reports import planned_cache_population as _planned_cache_population
from ..runtime.session import build_runtime_session as _build_runtime_session

from ..runtime.errors import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from ..runtime.device import managed_ttnn_device as _maybe_managed_device
from ..ttnn_compat import UnsupportedTTNNOp
from .support import synthetic_tensor_factory as _synthetic_tensor_factory
from .support import (
    dtype as _dtype,
    dry_run_reference as _dry_run_reference,
    failed_diagnostic_report as _failed_diagnostic_report,
    generated_model as _load_generated_model,
    runtime_int_tensor as _runtime_int_tensor,
    shape as _shape,
    to_namespace as _to_namespace,
    write_report as _write_report,
)

def run_smoke_prefill(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int = 1,
    prefill_len: int | None = None,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    parameters: Any | None = None,
    token_ids: Any | None = None,
    kv_cache: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers)
    num_layers = int(config["num_layers"])
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if layer_count > num_layers:
        raise ValueError(
            f"layers must be <= generated config num_layers ({num_layers})"
        )
    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config["seq_len"]
    )
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")
    plan = _runtime_prefill_plan(
        layers=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        config=config,
    )

    if dry_run:
        report = _base_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            plan=plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "prefill_status": "dry_run",
                "kv_cache_source": "prefill",
                "latency_ms": 0.0,
                "output_shapes": None,
                "cache_population": _planned_cache_population(plan),
                "tensor_conversion_count": plan["tensor_conversion_count"],
                "error": None,
                "ttnn_version": None,
                "reference": _dry_run_reference("prefill_smoke"),
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
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="no_device",
            message=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        torch = (
            torch_module
            if torch_module is not None
            else importlib.import_module("torch")
        )
    except ImportError as err:
        if parameters is None or token_ids is None or kv_cache is None:
            report = _unavailable_report(
                program_dir=program_root,
                layers=layer_count,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                prefill_len=prefill_len,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                plan=plan,
                status="missing_torch",
                message="torch is required to synthesize prefill tensors.",
                detail=str(err),
                ttnn_version=getattr(ttnn, "__version__", None),
                ttnn_module=ttnn,
            )
            _write_report(out, report)
            return report
        torch = None

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            runtime_context = None
            if model_path is not None:
                if prompt is None:
                    raise ValueError("prompt is required with model_path")
                session = _build_runtime_session(
                    ttnn=ttnn,
                    torch=torch,
                    device=ttnn_device,
                    dtype_seed=dtype_seed,
                    decode_plan=_runtime_decode_step_plan(
                        layers=layer_count,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        config=config,
                    ),
                    prefill_plan=plan,
                    program_dir=program_root,
                    model_path=Path(model_path),
                    prompt=prompt,
                    tokenizer_path=tokenizer_path or model_path,
                    tokenizer_module=tokenizer_module,
                    config=config,
                    layer_count=layer_count,
                    batch_size=batch_size,
                    cache_len=cache_len,
                    prefill_len=prefill_len,
                )
                runtime_context = session.context
                tensor_conversion_count = runtime_context.tensor_conversion_count
                parameter_source = runtime_context.parameter_source
                input_source = runtime_context.input_source
            elif parameters is None:
                assert torch is not None
                state = _build_synthetic_prefill_state(
                    ttnn=ttnn,
                    torch=torch,
                    device=ttnn_device,
                    dtype_seed=dtype_seed,
                    plan=plan,
                )
                parameters = state.parameters
                token_ids = state.token_ids
                kv_cache = state.kv_cache
                tensor_conversion_count = state.tensor_conversion_count
                parameter_source = "synthetic"
                input_source = "synthetic"
            else:
                if token_ids is None or kv_cache is None:
                    raise ValueError(
                        "token_ids and kv_cache are required with injected "
                        "prefill parameters"
                    )
                tensor_conversion_count = 0
                parameter_source = "injected"
                input_source = "injected"

            report = _run_generated_prefill(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=parameters,
                config=config,
                layer_count=layer_count,
                device=ttnn_device,
                token_ids=token_ids,
                kv_cache=kv_cache,
                tensor_conversion_count=tensor_conversion_count,
                plan=plan,
                runtime_context=runtime_context,
            )
            report.update(
                {
                    **_base_report(
                        program_dir=program_root,
                        layers=layer_count,
                        device=device,
                        device_id=device_id,
                        batch_size=batch_size,
                        prefill_len=prefill_len,
                        cache_len=cache_len,
                        dtype_seed=dtype_seed,
                        dry_run=False,
                        plan=plan,
                    ),
                    **report,
                    "parameter_source": parameter_source,
                    "input_source": input_source,
                }
            )
    except NoTTNNDeviceError as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="no_device",
            message=NO_TTNN_DEVICE_MESSAGE,
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
    except UnsupportedTTNNOp as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )
    except Exception as err:
        report = _unavailable_report(
            program_dir=program_root,
            layers=layer_count,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            prefill_len=prefill_len,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            plan=plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_version=getattr(ttnn, "__version__", None),
            ttnn_module=ttnn,
        )

    _write_report(out, report)
    return report


class _DiagnosticPrefillContext:
    """Minimal context adapter required by the canonical prefill runtime."""

    def __init__(self, model: Any, token_ids: Any, kv_cache: Any, prefill_len: int) -> None:
        self.generated_model = model
        self.prefill_token_ids = token_ids
        self.prefill_page_table = None
        self.kv_cache = kv_cache
        self.prefill_tokenization = {
            "effective_token_count": int(prefill_len),
        }

    def update_kv_cache(self, kv_cache: Any) -> None:
        self.kv_cache = kv_cache

    def update_decode_token(self, token_ids: Any) -> None:
        self.token_ids = token_ids


def _run_generated_prefill(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    layer_count: int,
    device: Any,
    token_ids: Any,
    kv_cache: Any,
    tensor_conversion_count: int,
    plan: dict[str, Any],
    runtime_context: Any | None = None,
) -> dict[str, Any]:
    if runtime_context is None:
        generated = _load_generated_model(program_dir / "model.py", ttnn)
        prefill_config = dict(config)
        prefill_config.update(
            num_layers=layer_count,
            batch_size=int(plan["batch_size"]),
            max_cache_len=int(plan["cache_len"]),
        )
        prefill_config["prefill"] = dict(prefill_config.get("prefill") or {})
        prefill_config["prefill"]["seq_len"] = int(plan["prefill_len"])
        model = generated.BuddyLlama31TTNN(
            device=device,
            parameters=parameters,
            config=_to_namespace(prefill_config),
        )
        runtime_context = _DiagnosticPrefillContext(
            model, token_ids, kv_cache, int(plan["prefill_len"])
        )
    else:
        model = runtime_context.generated_model
    model.ops.enable_recording()
    execution = _runtime_run_prefill_prompt(
        context=runtime_context,
        ttnn=ttnn,
        device=device,
        prefill_plan=plan,
        layer_count=layer_count,
    )
    token = execution.prefill_token
    kv_cache = execution.kv_cache
    reference = execution.reference
    passed = bool(reference["passed"])
    return {
        "passed": passed,
        "status": "passed" if passed else "reference_mismatch",
        "prefill_status": "passed" if passed else "reference_mismatch",
        "kv_cache_source": "prefill",
        "latency_ms": execution.latency_ms,
        "output_shapes": execution.output_shapes,
        "cache_population": execution.cache_population,
        "output": {
            "kind": "token",
            "shape": _shape(token),
            "dtype": _dtype(token),
            "repr": repr(token),
        },
        "tensor_conversion_count": tensor_conversion_count,
        "error": None if passed else "prefill structural reference mismatch",
        "ttnn_version": getattr(ttnn, "__version__", None),
        "ttnn_environment": collect_ttnn_environment(ttnn),
        "prefill_execution": execution.execution_report,
        "reference": reference,
    }


def _build_synthetic_prefill_state(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
) -> SimpleNamespace:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    inputs = plan["input_shapes"]
    params = plan["parameter_shapes"]
    layer_params = plan["layer_parameter_shapes"]
    lm_head_splits = []
    for shard_id, shape in enumerate(params["lm_head_splits"]):
        lm_head_splits.append(
            SimpleNamespace(
                shard_id=shard_id,
                weight=tensor(shape, name=f"lm_head_{shard_id}"),
            )
        )

    parameters = SimpleNamespace(
        embedding=SimpleNamespace(
            weight=tensor(params["embedding"], name="embedding")
        ),
        layers=[],
        final_norm=SimpleNamespace(
            weight=tensor(params["final_norm"], name="final_norm")
        ),
        lm_head=SimpleNamespace(splits=lm_head_splits),
    )
    for layer_id in range(int(plan["layers"])):
        parameters.layers.append(
            SimpleNamespace(
                attention=SimpleNamespace(
                    wqkv_packed=SimpleNamespace(
                        weight=tensor(
                            layer_params["attention_wqkv"],
                            name=f"layers.{layer_id}.attention_wqkv",
                        )
                    ),
                    o_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["attention_o_proj"],
                            name=f"layers.{layer_id}.o_proj",
                        )
                    ),
                    rotary=SimpleNamespace(
                        cos_matrix=tensor(
                            layer_params["rotary_cos_matrix"],
                            name=f"layers.{layer_id}.rotary_cos",
                        ),
                        sin_matrix=tensor(
                            layer_params["rotary_sin_matrix"],
                            name=f"layers.{layer_id}.rotary_sin",
                        ),
                        transformation_matrix=tensor(
                            layer_params["rotary_transformation_matrix"],
                            name=f"layers.{layer_id}.rotary_transform",
                        ),
                    ),
                ),
                input_norm=SimpleNamespace(
                    weight=tensor(
                        layer_params["input_norm"],
                        name=f"layers.{layer_id}.input_norm",
                    )
                ),
                post_attention_norm=SimpleNamespace(
                    weight=tensor(
                        layer_params["post_attention_norm"],
                        name=f"layers.{layer_id}.post_attention_norm",
                    )
                ),
                mlp=SimpleNamespace(
                    gate_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_gate"],
                            name=f"layers.{layer_id}.mlp_gate",
                        )
                    ),
                    up_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_up"],
                            name=f"layers.{layer_id}.mlp_up",
                        )
                    ),
                    down_proj=SimpleNamespace(
                        weight=tensor(
                            layer_params["mlp_down"],
                            name=f"layers.{layer_id}.mlp_down",
                        )
                    ),
                ),
            )
        )
    host_token_ids = _runtime_int_tensor(
        torch,
        [[0 for _ in range(int(plan["prefill_len"]))] for _ in range(int(plan["batch_size"]))],
        name="prefill_token_ids",
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
    token_ids = ttnn.from_torch(host_token_ids, **kwargs)
    kv_cache = []
    for layer_id in range(int(plan["layers"])):
        kv_cache.append(
            SimpleNamespace(
                k=tensor(
                    inputs["key_cache"],
                    name=f"layers.{layer_id}.key_cache",
                    zeros=True,
                ),
                v=tensor(
                    inputs["value_cache"],
                    name=f"layers.{layer_id}.value_cache",
                    zeros=True,
                ),
            )
        )
    return SimpleNamespace(
        parameters=parameters,
        token_ids=token_ids,
        kv_cache=kv_cache,
        tensor_conversion_count=tensor_count() + 1,
    )


def _base_report(**kwargs: Any) -> dict[str, Any]:
    plan = kwargs["plan"]
    dtype_seed = kwargs["dtype_seed"]
    return {
        "schema_version": 1,
        "template": "prefill_smoke",
        "program_dir": str(kwargs["program_dir"]),
        "layers": kwargs["layers"],
        "device": kwargs["device"],
        "device_id": kwargs["device_id"],
        "batch_size": kwargs["batch_size"],
        "prefill_len": kwargs["prefill_len"],
        "cache_len": kwargs["cache_len"],
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": kwargs["dry_run"],
        "op_sequence": plan["op_sequence"],
        "input_shapes": plan["input_shapes"],
        "parameter_shapes": plan["parameter_shapes"],
        "layer_parameter_shapes": plan["layer_parameter_shapes"],
        "expected_intermediate_shapes": plan["expected_intermediate_shapes"],
        "expected_output_shapes": plan["expected_output_shapes"],
        "kv_cache": plan["kv_cache"],
        "model_semantics": "prefill_populates_kv_cache_for_decode",
        "ttnn_environment": collect_ttnn_environment(None),
    }


def _unavailable_report(**kwargs: Any) -> dict[str, Any]:
    error_keys = {
        "status", "message", "detail", "ttnn_version", "ttnn_module"
    }
    base_kwargs = {
        key: value for key, value in kwargs.items() if key not in error_keys
    }
    plan = kwargs["plan"]
    base = _base_report(**base_kwargs, dry_run=False)
    return _failed_diagnostic_report(
        base,
        status=kwargs["status"],
        message=kwargs["message"],
        detail=kwargs["detail"],
        ttnn_version=kwargs.get("ttnn_version"),
        ttnn_module=kwargs.get("ttnn_module"),
        extra={
            "prefill_status": kwargs["status"],
            "kv_cache_source": "prefill",
            "output_shapes": None,
            "cache_population": _planned_cache_population(plan),
            "tensor_conversion_count": 0,
            "reference": {
                "kind": "prefill_smoke",
                "status": "not_run",
                "passed": False,
                "checks": [],
            },
        },
    )
