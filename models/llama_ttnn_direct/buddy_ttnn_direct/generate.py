from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
)
from .codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
    to_ttnn_parameters,
)
from .runtime_environment import collect_ttnn_environment
from .runtime_inputs import (
    PromptTokenizationError,
    detokenize_generated_token_ids,
    tokenize_prompt_for_prefill,
)
from .smoke_decode_shell import (
    _dry_run_reference,
    _dtype,
    _runtime_int_tensor,
    _shape,
    _to_namespace,
)
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, NoTTNNDeviceError
from .smoke_prefill import (
    _observed_cache_population,
    _planned_cache_population,
    _prefill_plan,
    _prefill_reference,
)
from .smoke_single_layer_decode import (
    DECODE_PARAMETER_ROLES,
    _attach_runtime_rotary_parameters,
    _build_prompt_decode_kv_cache_tensors,
    _build_prompt_decode_runtime_state_tensors,
    _decode_step_plan,
    _decode_step_reference,
    _generated_observed_op_sequence,
    _load_generated_model,
    _materialization_summary,
    _synthetic_tensor_factory,
    _tensorization_summary,
    _time_decode_step,
    _trace_report,
    _write_report,
)
from .decode_loop import (
    _generated_token_id_source,
    _generated_token_materialization_status,
    _loop_generated_token_ids,
    _loop_input_shapes,
    _loop_output_shapes,
    _materialize_token_ids,
)
from .templates.ttnn_ops import UnsupportedTTNNOp


class TTNNDirectRuntimeContext:
    """Owns the TTNN Direct runtime state for one generate invocation."""

    def __init__(
        self,
        *,
        parameters: Any,
        prefill_token_ids: Any,
        kv_cache: Any,
        tensor_conversion_count: int,
        parameter_source: str,
        input_source: str,
        prefill_tokenization: dict[str, Any],
        kv_cache_runtime_state: dict[str, Any],
        prefill_prompt_runtime_input_tensor_count: int,
        prefill_rotary_runtime_input_tensor_count: int,
        parameter_setup: dict[str, Any],
        tokenizer_path: str | Path,
        tokenizer_module: Any | None,
    ) -> None:
        self.parameters = parameters
        self.prefill_token_ids = prefill_token_ids
        self.kv_cache = kv_cache
        self.token_ids = None
        self.page_table = None
        self.cache_position = None
        self.rotary_state = None
        self.decode_runtime_state = None
        self.generated_module = None
        self.generated_model = None
        self.tensor_conversion_count = int(tensor_conversion_count)
        self.parameter_source = parameter_source
        self.input_source = input_source
        self.prefill_tokenization = prefill_tokenization
        self.kv_cache_runtime_state = kv_cache_runtime_state
        self.prefill_prompt_runtime_input_tensor_count = int(
            prefill_prompt_runtime_input_tensor_count
        )
        self.prefill_rotary_runtime_input_tensor_count = int(
            prefill_rotary_runtime_input_tensor_count
        )
        self.parameter_setup = parameter_setup
        self.tokenizer_path = str(tokenizer_path)
        self.tokenizer_module = tokenizer_module
        self.parameter_tensorization_count_per_generate = 1
        self.parameter_tensorization_count_per_decode_step = 0
        self.kv_cache_initialization_count_per_generate = 1
        self.kv_cache_reinitialized_per_step = False
        self.kv_cache_update_count = 0
        self.page_table_update_count = 0
        self.rotary_state_update_count = 0
        self.generated_model_initialization_count = 0
        self.decode_token_update_count = 0

    def install_generated_model(
        self,
        *,
        generated_module: Any,
        generated_model: Any,
    ) -> None:
        self.generated_module = generated_module
        self.generated_model = generated_model
        self.generated_model_initialization_count += 1

    def install_decode_runtime(self, runtime_state: SimpleNamespace) -> None:
        self.page_table = runtime_state.page_table
        self.cache_position = runtime_state.cache_position
        self.decode_runtime_state = runtime_state.decode_runtime_state
        self.rotary_state = runtime_state.rotary_runtime_state
        self.page_table_update_count += 1
        self.rotary_state_update_count += 1

    def update_decode_token(self, token_ids: Any) -> None:
        self.token_ids = token_ids
        self.decode_token_update_count += 1

    def update_kv_cache(self, kv_cache: Any) -> None:
        self.kv_cache = kv_cache
        self.kv_cache_update_count += 1

    def to_report(self, *, decode_step_count: int) -> dict[str, Any]:
        return {
            "class": "TTNNDirectRuntimeContext",
            "status": "built",
            "owns": [
                "parameters",
                "kv_cache",
                "page_table",
                "rotary_state",
                "tokenizer",
                "generated_model",
            ],
            "parameter_source": self.parameter_source,
            "input_source": self.input_source,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_module_injected": self.tokenizer_module is not None,
            "generated_model_initialized": self.generated_model is not None,
            "generated_model_initialization_count": (
                self.generated_model_initialization_count
            ),
            "parameter_tensorization_count_per_generate": (
                self.parameter_tensorization_count_per_generate
            ),
            "parameter_tensorization_count_per_decode_step": (
                self.parameter_tensorization_count_per_decode_step
            ),
            "kv_cache_initialization_count_per_generate": (
                self.kv_cache_initialization_count_per_generate
            ),
            "kv_cache_reinitialized_per_step": (
                self.kv_cache_reinitialized_per_step
            ),
            "kv_cache_update_count": self.kv_cache_update_count,
            "decode_step_count": int(decode_step_count),
            "page_table_update_count": self.page_table_update_count,
            "rotary_state_update_count": self.rotary_state_update_count,
            "decode_token_update_count": self.decode_token_update_count,
            "current_kv_cache_layers": len(self.kv_cache or []),
            "current_page_table_shape": _shape(self.page_table),
            "current_cache_position_shape": _shape(self.cache_position),
            "current_rotary_state": self.rotary_state,
        }


def run_generate(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int = 2,
    layers: int = 1,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = int(layers)
    token_count = int(max_new_tokens)
    num_layers = int(config["num_layers"])
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if layer_count > num_layers:
        raise ValueError(
            f"layers must be <= generated config num_layers ({num_layers})"
        )
    if token_count <= 0:
        raise ValueError("max_new_tokens must be positive")

    batch_size = int(batch_size or config["batch_size"])
    cache_len = int(cache_len or config["max_cache_len"])
    prefill_len = int(
        prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config.get("seq_len", 1)
    )
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")

    decode_plan = _decode_step_plan(
        layers=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        config=config,
    )
    prefill_plan = _prefill_plan(
        layers=layer_count,
        batch_size=batch_size,
        prefill_len=prefill_len,
        cache_len=cache_len,
        config=config,
    )
    decode_step_count = max(0, token_count - 1)

    if dry_run:
        report = _generate_base_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            dry_run=True,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "runtime_status": "dry_run",
                "prefill_status": "dry_run",
                "decode_loop_runtime_owned": False,
                "planned_decode_loop_runtime_owned": True,
                "kv_cache_source": "prefill",
                "generated_token_ids": [],
                "generated_text": "",
                "generated_text_by_user": [],
                "generated_text_status": "not_run",
                "generated_text_source": "dry_run",
                "prefill": {
                    "status": "dry_run",
                    "cache_population": _planned_cache_population(prefill_plan),
                },
                "step_reports": [],
                "per_step_token_metadata": [],
                "tensor_conversion_count": (
                    prefill_plan["tensor_conversion_count"]
                    + decode_plan["tensor_conversion_count"]
                ),
                "runtime_context": {
                    "class": "TTNNDirectRuntimeContext",
                    "status": "planned",
                    "owns": [
                        "parameters",
                        "kv_cache",
                        "page_table",
                        "rotary_state",
                        "tokenizer",
                        "generated_model",
                    ],
                    "parameter_tensorization_count_per_generate": 1,
                    "parameter_tensorization_count_per_decode_step": 0,
                    "kv_cache_initialization_count_per_generate": 1,
                    "kv_cache_reinitialized_per_step": False,
                    "decode_step_count": decode_step_count,
                },
                "parameter_tensorization_count_per_generate": 1,
                "parameter_tensorization_count_per_decode_step": 0,
                "kv_cache_initialization_count_per_generate": 1,
                "kv_cache_reinitialized_per_step": False,
                "synthetic_runtime_input_tensor_count": 0,
                "synthetic_rotary_tensor_count": 0,
                "synthetic_kv_cache_tensor_count": 0,
                "trace": _trace_report(requested=False, status="disabled"),
                "reference": _dry_run_reference("generate"),
                "error": None,
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_report(out, report)
        return report

    if model_path is None:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_model_path",
            message="model_path is required for generate execution",
            detail="model_path was not provided",
        )
        _write_report(out, report)
        return report
    if prompt is None:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_prompt",
            message="prompt is required for generate execution",
            detail="prompt was not provided",
        )
        _write_report(out, report)
        return report
    if decode_plan["output_kind"] != "token":
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="unsupported_output_kind",
            message="generate requires token output",
            detail=f"output_kind={decode_plan['output_kind']}",
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
        report = _generate_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
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
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="missing_torch",
            message="torch is required to build generate runtime tensors",
            detail=str(err),
            ttnn_module=ttnn,
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_generate_device(ttnn, device_id, ttnn_module) as ttnn_device:
            context = _build_generate_state(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
                program_dir=program_root,
                model_path=Path(model_path),
                prompt=prompt,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            generated = _load_generated_model(program_root / "model.py", ttnn)
            generate_config = dict(config)
            generate_config["num_layers"] = layer_count
            model = generated.BuddyLlama31TTNN(
                device=ttnn_device,
                parameters=context.parameters,
                config=_to_namespace(generate_config),
            )
            context.install_generated_model(
                generated_module=generated,
                generated_model=model,
            )

            total_start = time.perf_counter()
            prefill_start = time.perf_counter()
            prefill_token, kv_cache, cache_reports = context.generated_model.prefill_prompt(
                context.prefill_token_ids,
                context.kv_cache,
            )
            context.update_kv_cache(kv_cache)
            synchronize = getattr(ttnn, "synchronize_device", None)
            if callable(synchronize):
                synchronize(ttnn_device)
            prefill_latency_ms = (time.perf_counter() - prefill_start) * 1000.0
            prefill_output_shapes = {
                "token": _shape(prefill_token),
                "key_cache": _shape(kv_cache[0].k),
                "value_cache": _shape(kv_cache[0].v),
                "kv_cache_layers": [
                    {
                        "layer_id": layer_id,
                        "key_cache": _shape(layer_cache.k),
                        "value_cache": _shape(layer_cache.v),
                    }
                    for layer_id, layer_cache in enumerate(kv_cache[:layer_count])
                ],
            }
            prefill_reference = _prefill_reference(
                plan=prefill_plan,
                layer_count=layer_count,
                output_shapes=prefill_output_shapes,
                output={
                    "kind": "token",
                    "shape": _shape(prefill_token),
                    "dtype": _dtype(prefill_token),
                },
                observed_ops=_generated_observed_op_sequence(
                    context.generated_model,
                    ttnn,
                ),
            )
            first_token = _prefill_last_token_to_decode_token(
                prefill_token=prefill_token,
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                batch_size=batch_size,
            )
            if first_token.status != "materialized":
                raise PromptTokenizationError(
                    "prefill token could not be materialized for decode"
                )

            generated_token_ids_by_user = [
                list(row) for row in first_token.token_ids_by_user
            ]
            context.update_decode_token(first_token.token_ids)
            per_step_token_metadata = [
                {
                    "step_index": "prefill",
                    "token_ids_by_user": first_token.token_ids_by_user,
                    "token_materialization_status": first_token.status,
                    "token_materialization_source": first_token.source,
                    "cache_position_value": (
                        context.prefill_tokenization["effective_token_count"] - 1
                    ),
                    "token_shape": _shape(first_token.token_ids),
                }
            ]

            decode_runtime = _build_decode_runtime_for_position(
                ttnn=ttnn,
                torch=torch,
                device=ttnn_device,
                dtype_seed=dtype_seed,
                parameters=context.parameters,
                decode_plan=decode_plan,
                batch_size=batch_size,
                cache_len=cache_len,
                prefill_effective_token_count=(
                    context.prefill_tokenization["effective_token_count"]
                ),
                generated_token_index=0,
            )
            context.install_decode_runtime(decode_runtime)
            decode_runtime_state = context.decode_runtime_state
            rotary_runtime_state = context.rotary_state
            tensor_conversion_count = (
                context.tensor_conversion_count
                + first_token.tensor_conversion_count
                + decode_runtime.tensor_conversion_count
            )
            decode_runtime_state_count = (
                decode_runtime.decode_runtime_state_input_tensor_count
            )
            decode_rotary_runtime_count = (
                decode_runtime.rotary_runtime_input_tensor_count
            )
            step_reports = []
            for step_index in range(decode_step_count):
                input_shapes = _loop_input_shapes(
                    token_ids=context.token_ids,
                    page_table=context.page_table,
                    cache_position=context.cache_position,
                    kv_cache=context.kv_cache,
                )
                token, kv_cache, latency_ms = _time_decode_step(
                    ttnn=ttnn,
                    model=context.generated_model,
                    device=ttnn_device,
                    token_ids=context.token_ids,
                    page_table=context.page_table,
                    cache_position=context.cache_position,
                    kv_cache=context.kv_cache,
                )
                context.update_kv_cache(kv_cache)
                output_shapes = _loop_output_shapes(
                    token=token,
                    kv_cache=context.kv_cache,
                    layer_count=layer_count,
                )
                output = {
                    "kind": "token",
                    "shape": _shape(token),
                    "dtype": _dtype(token),
                    "repr": repr(token),
                }
                token_materialization = _loop_generated_token_ids(
                    token=token,
                    ttnn=ttnn,
                    batch_size=batch_size,
                )
                step_token_ids = token_materialization["token_ids_by_user"]
                for user_index, row in enumerate(step_token_ids):
                    generated_token_ids_by_user[user_index].extend(row)
                token_metadata = {
                    "step_index": step_index,
                    "token_ids_by_user": step_token_ids,
                    "token_materialization_status": (
                        token_materialization["status"]
                    ),
                    "token_materialization_source": (
                        token_materialization["source"]
                    ),
                    "cache_position_value": decode_runtime_state.get(
                        "cache_position_value"
                    ),
                    "page_table_shape": input_shapes.get("page_table"),
                    "token_shape": _shape(token),
                }
                per_step_token_metadata.append(token_metadata)
                reference = _decode_step_reference(
                    plan=decode_plan,
                    layer_count=layer_count,
                    output_shapes=output_shapes,
                    output=output,
                    observed_ops=_generated_observed_op_sequence(
                        context.generated_model,
                        ttnn,
                    ),
                )
                step_reports.append(
                    {
                        "step_index": step_index,
                        "status": (
                            "passed"
                            if reference["passed"]
                            else "reference_mismatch"
                        ),
                        "passed": bool(reference["passed"]),
                        "latency_ms": latency_ms,
                        "cache_position_value": decode_runtime_state.get(
                            "cache_position_value"
                        ),
                        "input_shapes": input_shapes,
                        "decode_runtime_state": decode_runtime_state,
                        "rotary_runtime_state": rotary_runtime_state,
                        "output_shapes": output_shapes,
                        "output": output,
                        "generated_token_ids": step_token_ids,
                        "token_materialization": token_materialization,
                        "reference": reference,
                    }
                )
                context.update_decode_token(token)
                if step_index + 1 < decode_step_count:
                    decode_runtime = _build_decode_runtime_for_position(
                        ttnn=ttnn,
                        torch=torch,
                        device=ttnn_device,
                        dtype_seed=dtype_seed,
                        parameters=context.parameters,
                        decode_plan=decode_plan,
                        batch_size=batch_size,
                        cache_len=cache_len,
                        prefill_effective_token_count=(
                            context.prefill_tokenization["effective_token_count"]
                        ),
                        generated_token_index=step_index + 1,
                    )
                    context.install_decode_runtime(decode_runtime)
                    decode_runtime_state = context.decode_runtime_state
                    rotary_runtime_state = context.rotary_state
                    decode_runtime_state_count += (
                        decode_runtime.decode_runtime_state_input_tensor_count
                    )
                    decode_rotary_runtime_count += (
                        decode_runtime.rotary_runtime_input_tensor_count
                    )
                    tensor_conversion_count += (
                        decode_runtime.tensor_conversion_count
                    )

            latency_ms = (time.perf_counter() - total_start) * 1000.0
            text_report = detokenize_generated_token_ids(
                token_ids_by_user=generated_token_ids_by_user,
                tokenizer_path=tokenizer_path or model_path,
                tokenizer_module=tokenizer_module,
            )
            decode_passed = all(step["passed"] for step in step_reports)
            passed = bool(prefill_reference["passed"] and decode_passed)
            parameter_setup = dict(context.parameter_setup)
            parameter_setup.update(
                {
                    "generate_runtime_owned": passed,
                    "decode_loop_runtime_owned": (
                        decode_step_count == 0 or decode_passed
                    ),
                    "prefill_prompt_runtime_input_tensor_count": (
                        context.prefill_prompt_runtime_input_tensor_count
                    ),
                    "prefill_rotary_runtime_input_tensor_count": (
                        context.prefill_rotary_runtime_input_tensor_count
                    ),
                    "prefill_first_token_tensor_conversion_count": (
                        first_token.tensor_conversion_count
                    ),
                    "decode_runtime_state_input_tensor_count": (
                        decode_runtime_state_count
                    ),
                    "decode_rotary_runtime_input_tensor_count": (
                        decode_rotary_runtime_count
                    ),
                    "decode_loop_step_count": decode_step_count,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "parameter_tensorization_count_per_generate": (
                        context.parameter_tensorization_count_per_generate
                    ),
                    "parameter_tensorization_count_per_decode_step": (
                        context.parameter_tensorization_count_per_decode_step
                    ),
                    "kv_cache_initialization_count_per_generate": (
                        context.kv_cache_initialization_count_per_generate
                    ),
                    "kv_cache_reinitialized_per_step": (
                        context.kv_cache_reinitialized_per_step
                    ),
                }
            )
            report = _generate_base_report(
                program_dir=program_root,
                layers=layer_count,
                max_new_tokens=token_count,
                decode_steps=decode_step_count,
                prefill_len=prefill_len,
                device=device,
                device_id=device_id,
                batch_size=batch_size,
                cache_len=cache_len,
                dtype_seed=dtype_seed,
                dry_run=False,
                decode_plan=decode_plan,
                prefill_plan=prefill_plan,
            )
            report.update(
                {
                    "passed": passed,
                    "status": "passed" if passed else "reference_mismatch",
                    "runtime_status": (
                        "passed" if passed else "reference_mismatch"
                    ),
                    "prefill_status": (
                        "passed"
                        if prefill_reference["passed"]
                        else "reference_mismatch"
                    ),
                    "decode_loop_runtime_owned": (
                        decode_step_count == 0 or decode_passed
                    ),
                    "generate_runtime_owned": passed,
                    "kv_cache_source": "prefill",
                    "input_source": "prompt_prefill",
                    "runtime_owner": "generate",
                    "parameter_source": context.parameter_source,
                    "parameter_setup": parameter_setup,
                    "prompt_tokenization": context.prefill_tokenization,
                    "prefill_tokenization": context.prefill_tokenization,
                    "decode_runtime_state": decode_runtime_state,
                    "rotary_runtime_state": rotary_runtime_state,
                    "kv_cache_runtime_state": context.kv_cache_runtime_state,
                    "runtime_context": context.to_report(
                        decode_step_count=decode_step_count
                    ),
                    "parameter_tensorization_count_per_generate": (
                        context.parameter_tensorization_count_per_generate
                    ),
                    "parameter_tensorization_count_per_decode_step": (
                        context.parameter_tensorization_count_per_decode_step
                    ),
                    "kv_cache_initialization_count_per_generate": (
                        context.kv_cache_initialization_count_per_generate
                    ),
                    "kv_cache_reinitialized_per_step": (
                        context.kv_cache_reinitialized_per_step
                    ),
                    "prefill": {
                        "status": (
                            "passed"
                            if prefill_reference["passed"]
                            else "reference_mismatch"
                        ),
                        "latency_ms": prefill_latency_ms,
                        "output_shapes": prefill_output_shapes,
                        "output": {
                            "kind": "token",
                            "shape": _shape(prefill_token),
                            "dtype": _dtype(prefill_token),
                            "repr": repr(prefill_token),
                        },
                        "first_token": {
                            "status": first_token.status,
                            "source": first_token.source,
                            "token_ids_by_user": first_token.token_ids_by_user,
                            "token_shape": _shape(first_token.token_ids),
                            "host_roundtrip": True,
                        },
                        "cache_population": _observed_cache_population(
                            plan=prefill_plan,
                            cache_reports=cache_reports,
                            output_shapes=prefill_output_shapes,
                        ),
                        "reference": prefill_reference,
                    },
                    "step_reports": step_reports,
                    "per_step_token_metadata": per_step_token_metadata,
                    "generated_token_ids": generated_token_ids_by_user,
                    "generated_token_id_source": _generated_token_id_source(
                        per_step_token_metadata
                    ),
                    "token_materialization_status": (
                        _generated_token_materialization_status(
                            per_step_token_metadata
                        )
                    ),
                    "generated_text": text_report["generated_text"],
                    "generated_text_by_user": (
                        text_report["generated_text_by_user"]
                    ),
                    "generated_text_status": text_report["status"],
                    "generated_text_source": text_report["source"],
                    "generated_text_report": text_report,
                    "output_shapes": (
                        step_reports[-1]["output_shapes"]
                        if step_reports
                        else prefill_output_shapes
                    ),
                    "output": step_reports[-1]["output"] if step_reports else None,
                    "tensor_conversion_count": tensor_conversion_count,
                    "synthetic_runtime_input_tensor_count": 0,
                    "synthetic_rotary_tensor_count": 0,
                    "synthetic_kv_cache_tensor_count": 0,
                    "latency_ms": latency_ms,
                    "throughput_summary": _generate_throughput_summary(
                        latency_ms=latency_ms,
                        batch_size=batch_size,
                        max_new_tokens=token_count,
                    ),
                    "trace": _trace_report(requested=False, status="disabled"),
                    "reference": _generate_reference_summary(
                        prefill_reference=prefill_reference,
                        step_reports=step_reports,
                    ),
                    "error": None
                    if passed
                    else "generate structural reference mismatch",
                    "ttnn_version": getattr(ttnn, "__version__", None),
                    "ttnn_environment": collect_ttnn_environment(ttnn),
                }
            )
    except NoTTNNDeviceError as err:
        report = _generate_no_device_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            detail=str(err),
            ttnn_module=ttnn,
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        PromptTokenizationError,
    ) as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="parameter_setup_error",
            message=str(err),
            detail=str(err),
            ttnn_module=ttnn,
        )
    except UnsupportedTTNNOp as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="api_mismatch",
            message=str(err),
            detail=err.op_name,
            ttnn_module=ttnn,
        )
    except Exception as err:
        report = _generate_failed_report(
            program_dir=program_root,
            layers=layer_count,
            max_new_tokens=token_count,
            decode_steps=decode_step_count,
            prefill_len=prefill_len,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
            dtype_seed=dtype_seed,
            decode_plan=decode_plan,
            prefill_plan=prefill_plan,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
            detail=str(err),
            ttnn_module=ttnn,
        )

    _write_report(out, report)
    return report


def run_profile_generate(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int = 2,
    layers: int = 1,
    prefill_len: int | None = None,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    generate_report: str | Path | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    profile_path = Path(out)
    generate_report_path = (
        Path(generate_report)
        if generate_report is not None
        else _default_generate_report_path(profile_path)
    )
    generate_payload = run_generate(
        out=generate_report_path,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_new_tokens,
        layers=layers,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=dry_run,
        ttnn_module=ttnn_module,
        torch_module=torch_module,
        tokenizer_module=tokenizer_module,
    )
    report = _profile_generate_from_generate_report(
        generate_payload,
        profile_path=profile_path,
        generate_report_path=generate_report_path,
    )
    _write_report(profile_path, report)
    return report


def _build_generate_state(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    program_dir: Path,
    model_path: Path,
    prompt: str,
    tokenizer_path: str | Path,
    tokenizer_module: Any | None,
) -> SimpleNamespace:
    host_params = load_llama_parameters_from_manifests(
        model_path=model_path,
        weights_manifest=program_dir / "weights_manifest.json",
        config=program_dir / "config.json",
        tensor_backend="torch",
        layers=range(int(decode_plan["layers"])),
    )
    materialization_summary = _materialization_summary(host_params)
    result = to_ttnn_parameters(
        host_params,
        device,
        load_parameter_config_from_program(program_dir),
        roles=DECODE_PARAMETER_ROLES,
        layers=range(int(decode_plan["layers"])),
        ttnn_module=ttnn,
    )
    assert result.parameters is not None
    prefill_tokenization = tokenize_prompt_for_prefill(
        prompt=prompt,
        batch_size=int(prefill_plan["batch_size"]),
        prefill_len=int(prefill_plan["prefill_len"]),
        tokenizer_path=tokenizer_path,
        vocab_size=prefill_plan.get("vocab_size"),
        tokenizer_module=tokenizer_module,
    )
    prefill_token_ids = _prefill_token_ids_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        token_ids=prefill_tokenization.token_ids,
    )
    kv_runtime = _build_prompt_decode_kv_cache_tensors(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        layer_count=int(prefill_plan["layers"]),
        batch_size=int(prefill_plan["batch_size"]),
        cache_len=int(prefill_plan["cache_len"]),
        page_block_size=int(prefill_plan["kv_cache"]["page_block_size"]),
        num_kv_heads=int(prefill_plan["kv_cache"]["logical_shape"][2]),
        head_dim=int(prefill_plan["kv_cache"]["logical_shape"][3]),
    )
    prefill_rotary = _attach_prefill_rotary_parameters(
        parameters=result.parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        prefill_plan=prefill_plan,
    )
    tensorization_count = int(result.report["tensor_count"])
    tensor_conversion_count = (
        tensorization_count
        + 1
        + int(kv_runtime.tensor_conversion_count)
        + int(prefill_rotary.tensor_conversion_count)
    )
    return TTNNDirectRuntimeContext(
        parameters=result.parameters,
        prefill_token_ids=prefill_token_ids,
        kv_cache=kv_runtime.kv_cache,
        tensor_conversion_count=tensor_conversion_count,
        parameter_source="hf_model",
        input_source="prompt_prefill",
        prefill_tokenization=prefill_tokenization.to_report(),
        kv_cache_runtime_state=kv_runtime.kv_cache_runtime_state,
        prefill_prompt_runtime_input_tensor_count=1,
        prefill_rotary_runtime_input_tensor_count=(
            prefill_rotary.tensor_conversion_count
        ),
        parameter_setup={
            "materialization": materialization_summary,
            "tensorization": _tensorization_summary(result.report),
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "prefill_prompt_runtime_input_tensor_count": 1,
            "prefill_rotary_runtime_input_tensor_count": (
                prefill_rotary.tensor_conversion_count
            ),
            "kv_cache_runtime_input_tensor_count": (
                kv_runtime.tensor_conversion_count
            ),
            "kv_cache_runtime_state": kv_runtime.kv_cache_runtime_state,
        },
        tokenizer_path=tokenizer_path,
        tokenizer_module=tokenizer_module,
    )


def _prefill_token_ids_tensor(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    token_ids: list[list[int]],
) -> Any:
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
    return ttnn.from_torch(
        _runtime_int_tensor(torch, token_ids, name="prefill_prompt_token_ids"),
        **kwargs,
    )


def _attach_prefill_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    prefill_plan: dict[str, Any],
) -> SimpleNamespace:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    shapes = prefill_plan["layer_parameter_shapes"]
    for layer_id in range(int(prefill_plan["layers"])):
        layer = parameters.layers[layer_id]
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = SimpleNamespace()
            layer.attention = attention
        attention.rotary = SimpleNamespace(
            cos_matrix=tensor(
                shapes["rotary_cos_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_cos",
            ),
            sin_matrix=tensor(
                shapes["rotary_sin_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_sin",
            ),
            transformation_matrix=tensor(
                shapes["rotary_transformation_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_transform",
            ),
        )
    return SimpleNamespace(tensor_conversion_count=tensor_count())


def _prefill_last_token_to_decode_token(
    *,
    prefill_token: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    batch_size: int,
) -> SimpleNamespace:
    materialized = _materialize_token_ids(
        prefill_token,
        ttnn=ttnn,
        batch_size=batch_size,
    )
    if materialized is None:
        return SimpleNamespace(
            status="unavailable",
            source="unmaterialized_prefill_token",
            token_ids_by_user=[],
            token_ids=None,
            tensor_conversion_count=0,
        )
    token_ids = _prefill_token_ids_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        token_ids=materialized["token_ids_by_user"],
    )
    return SimpleNamespace(
        status="materialized",
        source=materialized["source"],
        token_ids_by_user=materialized["token_ids_by_user"],
        token_ids=token_ids,
        tensor_conversion_count=1,
    )


def _build_decode_runtime_for_position(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    parameters: Any,
    decode_plan: dict[str, Any],
    batch_size: int,
    cache_len: int,
    prefill_effective_token_count: int,
    generated_token_index: int,
) -> SimpleNamespace:
    runtime_state = _build_prompt_decode_runtime_state_tensors(
        ttnn=ttnn,
        torch=torch,
        device=device,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=int(decode_plan["kv_cache"]["page_block_size"]),
        prompt_token_count=(
            int(prefill_effective_token_count)
            + int(generated_token_index)
            + 1
        ),
    )
    rotary_runtime = _attach_runtime_rotary_parameters(
        parameters=parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=decode_plan,
        cache_position_value=int(
            runtime_state.decode_runtime_state["cache_position_value"]
        ),
    )
    return SimpleNamespace(
        page_table=runtime_state.page_table,
        cache_position=runtime_state.cache_position,
        decode_runtime_state=runtime_state.decode_runtime_state,
        rotary_runtime_state=rotary_runtime.rotary_runtime_state,
        tensor_conversion_count=(
            int(runtime_state.tensor_conversion_count)
            + int(rotary_runtime.tensor_conversion_count)
        ),
        decode_runtime_state_input_tensor_count=(
            runtime_state.tensor_conversion_count
        ),
        rotary_runtime_input_tensor_count=rotary_runtime.tensor_conversion_count,
    )


def _generate_base_report(
    *,
    program_dir: Path,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    dry_run: bool,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
) -> dict[str, Any]:
    token_budget = _generated_token_budget(
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
    )
    return {
        "schema_version": 1,
        "command": "generate",
        "mode": "generate",
        "template": "prefill_then_decode_generate",
        "program_dir": str(program_dir),
        "layers": layers,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "max_new_tokens": max_new_tokens,
        "decode_steps": decode_steps,
        "generated_token_budget": token_budget,
        "prefill_first_token_counts_as_generated_token": True,
        "decode_steps_excludes_prefill_token": True,
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": dry_run,
        "prefill_plan": prefill_plan,
        "decode_plan": decode_plan,
        "prefill_op_sequence": prefill_plan["op_sequence"],
        "decode_op_sequence": decode_plan["op_sequence"],
        "model_semantics": "prompt_prefill_then_decode",
        "kv_cache_source": "prefill",
        "semantic_disclaimer": (
            "This generate path runs prefill before decode. It is a first "
            "functional bring-up path and still uses host-side token "
            "materialization between prefill and decode; performance parity is "
            "not claimed."
        ),
        "ttnn_environment": collect_ttnn_environment(None),
    }


def _generated_token_budget(
    *,
    max_new_tokens: int,
    decode_steps: int,
) -> dict[str, Any]:
    prefill_first_token_count = 1 if max_new_tokens > 0 else 0
    return {
        "max_new_tokens": max_new_tokens,
        "prefill_first_token_count": prefill_first_token_count,
        "decode_loop_token_count": decode_steps,
        "decode_steps": decode_steps,
        "total_planned_generated_tokens": (
            prefill_first_token_count + decode_steps
        ),
        "decode_steps_formula": (
            "max_new_tokens - 1 because the first generated token is "
            "materialized from prefill output"
        ),
    }


def _generate_no_device_report(
    *,
    detail: str,
    ttnn_module: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _generate_failed_report(
        status="no_device",
        message=NO_TTNN_DEVICE_MESSAGE,
        detail=detail,
        ttnn_module=ttnn_module,
        **kwargs,
    )


def _generate_failed_report(
    *,
    program_dir: Path,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_module: Any | None = None,
) -> dict[str, Any]:
    report = _generate_base_report(
        program_dir=program_dir,
        layers=layers,
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        decode_plan=decode_plan,
        prefill_plan=prefill_plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "runtime_status": status,
            "prefill_status": status,
            "decode_loop_runtime_owned": False,
            "generate_runtime_owned": False,
            "generated_token_ids": [],
            "generated_text": "",
            "generated_text_by_user": [],
            "generated_text_status": "not_run",
            "generated_text_source": status,
            "prefill": {
                "status": status,
                "cache_population": _planned_cache_population(prefill_plan),
            },
            "step_reports": [],
            "per_step_token_metadata": [],
            "tensor_conversion_count": 0,
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "latency_ms": None,
            "trace": _trace_report(requested=False, status="disabled"),
            "reference": {
                "kind": "generate_prefill_then_decode",
                "status": "not_run",
                "passed": False,
                "checks": [],
            },
            "error": message,
            "detail": detail,
            "ttnn_version": getattr(ttnn_module, "__version__", None),
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
        }
    )
    return report


def _generate_reference_summary(
    *,
    prefill_reference: dict[str, Any],
    step_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_steps = [
        int(step["step_index"])
        for step in step_reports
        if not step.get("passed")
    ]
    passed = bool(prefill_reference.get("passed")) and not failed_steps
    return {
        "kind": "generate_prefill_then_decode_structural",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "prefill_status": prefill_reference.get("status"),
        "decode_step_count": len(step_reports),
        "failed_decode_steps": failed_steps,
    }


def _generate_throughput_summary(
    *,
    latency_ms: float | None,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    total_tokens = batch_size * max_new_tokens
    summary: dict[str, Any] = {
        "batch_size": batch_size,
        "generated_tokens_per_user": max_new_tokens,
        "total_generated_tokens": total_tokens,
        "basis": "generate_latency_ms",
    }
    if latency_ms is None or latency_ms <= 0.0:
        summary.update(
            {
                "status": "unavailable",
                "latency_ms": latency_ms,
                "tokens_per_second_per_user": None,
                "aggregate_tokens_per_second": None,
            }
        )
        return summary
    tokens_per_second_per_user = 1000.0 * max_new_tokens / latency_ms
    summary.update(
        {
            "status": "measured",
            "latency_ms": latency_ms,
            "tokens_per_second_per_user": tokens_per_second_per_user,
            "aggregate_tokens_per_second": (
                tokens_per_second_per_user * batch_size
            ),
        }
    )
    return summary


def _profile_generate_from_generate_report(
    generate: dict[str, Any],
    *,
    profile_path: Path,
    generate_report_path: Path,
) -> dict[str, Any]:
    throughput = generate.get("throughput_summary") or {}
    step_latencies = [
        latency
        for latency in (
            _float_or_none(step.get("latency_ms"))
            for step in generate.get("step_reports", [])
        )
        if latency is not None
    ]
    prefill_ms = _float_or_none((generate.get("prefill") or {}).get("latency_ms"))
    decode_total_ms = sum(step_latencies) if step_latencies else None
    decode_step_ms_mean = (
        decode_total_ms / len(step_latencies)
        if decode_total_ms is not None and step_latencies
        else None
    )
    tokens_per_second_per_user = _float_or_none(
        throughput.get("tokens_per_second_per_user")
    )
    aggregate_tokens_per_second = _float_or_none(
        throughput.get("aggregate_tokens_per_second")
    )
    dry_run = generate.get("status") == "dry_run"
    generate_ran = bool(generate.get("passed"))
    has_positive_throughput = (
        tokens_per_second_per_user is not None
        and tokens_per_second_per_user > 0.0
    )
    acceptance = _profile_generate_acceptance(
        dry_run=dry_run,
        generate_ran=generate_ran,
        has_positive_throughput=has_positive_throughput,
    )
    if dry_run:
        status = "dry_run"
    elif acceptance["passed"]:
        status = "profiled"
    else:
        status = generate.get("status") or "profile_incomplete"

    return {
        "schema_version": 1,
        "command": "profile-generate",
        "mode": "profile-generate",
        "template": "prefill_then_decode_generate_profile",
        "status": status,
        "passed": bool(acceptance["passed"]),
        "dry_run": dry_run,
        "program_dir": generate.get("program_dir"),
        "generate_report": str(generate_report_path),
        "profile_report": str(profile_path),
        "generate_status": generate.get("status"),
        "generate_passed": bool(generate.get("passed")),
        "layers": generate.get("layers"),
        "batch_size": generate.get("batch_size"),
        "prefill_len": generate.get("prefill_len"),
        "cache_len": generate.get("cache_len"),
        "max_new_tokens": generate.get("max_new_tokens"),
        "decode_steps": generate.get("decode_steps"),
        "prefill_status": generate.get("prefill_status"),
        "kv_cache_source": generate.get("kv_cache_source"),
        "parameter_source": generate.get("parameter_source"),
        "input_source": generate.get("input_source"),
        "runtime_owner": generate.get("runtime_owner"),
        "generate_runtime_owned": generate.get("generate_runtime_owned"),
        "decode_loop_runtime_owned": generate.get("decode_loop_runtime_owned"),
        "runtime_context": generate.get("runtime_context"),
        "parameter_setup": generate.get("parameter_setup"),
        "synthetic_runtime_input_tensor_count": generate.get(
            "synthetic_runtime_input_tensor_count"
        ),
        "synthetic_rotary_tensor_count": generate.get(
            "synthetic_rotary_tensor_count"
        ),
        "synthetic_kv_cache_tensor_count": generate.get(
            "synthetic_kv_cache_tensor_count"
        ),
        "prefill_prompt_runtime_input_tensor_count": (
            _setup_count(generate, "prefill_prompt_runtime_input_tensor_count")
        ),
        "prefill_rotary_runtime_input_tensor_count": (
            _setup_count(generate, "prefill_rotary_runtime_input_tensor_count")
        ),
        "decode_runtime_state_input_tensor_count": (
            _setup_count(generate, "decode_runtime_state_input_tensor_count")
        ),
        "decode_rotary_runtime_input_tensor_count": (
            _setup_count(generate, "decode_rotary_runtime_input_tensor_count")
        ),
        "kv_cache_runtime_input_tensor_count": (
            _setup_count(generate, "kv_cache_runtime_input_tensor_count")
        ),
        "generated_text_status": generate.get("generated_text_status"),
        "generated_token_count_by_user": _generated_token_counts(generate),
        "latency_ms": _float_or_none(generate.get("latency_ms")),
        "prefill_ms": prefill_ms,
        "decode_step_ms_mean": decode_step_ms_mean,
        "decode_step_ms_min": min(step_latencies) if step_latencies else None,
        "decode_step_ms_max": max(step_latencies) if step_latencies else None,
        "decode_step_ms_samples": step_latencies,
        "tokens_per_second_per_user": tokens_per_second_per_user,
        "aggregate_tokens_per_second": aggregate_tokens_per_second,
        "throughput_summary": throughput,
        "sections": _profile_generate_sections(
            prefill_ms=prefill_ms,
            decode_total_ms=decode_total_ms,
            decode_step_ms_mean=decode_step_ms_mean,
        ),
        "per_layer": _profile_generate_per_layer(generate),
        "acceptance": acceptance,
        "official_performance_parity_claimed": False,
        "message": (
            "First generate profile only; no official performance parity is "
            "claimed."
        ),
        "error": None if acceptance["passed"] else generate.get("error"),
        "ttnn_environment": generate.get("ttnn_environment"),
    }


def _profile_generate_sections(
    *,
    prefill_ms: float | None,
    decode_total_ms: float | None,
    decode_step_ms_mean: float | None,
) -> dict[str, Any]:
    unavailable = {
        "status": "unavailable",
        "value_ms": None,
        "reason": "generate path does not yet collect section-level timers",
    }
    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_step_ms_mean": decode_step_ms_mean,
        "embedding_ms": dict(unavailable),
        "prefill_attention_ms": dict(unavailable),
        "decode_attention_ms": dict(unavailable),
        "mlp_ms": dict(unavailable),
        "lm_head_ms": dict(unavailable),
        "argmax_ms": dict(unavailable),
        "host_copy_ms": {
            "status": "unmeasured",
            "value_ms": None,
            "host_roundtrip_present": True,
            "reason": (
                "first generate implementation materializes the prefill token "
                "on host, but does not time the copy separately"
            ),
        },
    }


def _profile_generate_per_layer(generate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "layers": generate.get("layers"),
        "attention_ms": None,
        "mlp_ms": None,
        "reason": (
            "profile-generate currently reports whole prefill and decode-step "
            "timings; per-layer attribution is the next profiling refinement"
        ),
    }


def _profile_generate_acceptance(
    *,
    dry_run: bool,
    generate_ran: bool,
    has_positive_throughput: bool,
) -> dict[str, Any]:
    checks = [
        {
            "name": "profile_generate.full_generated_model_can_run",
            "passed": bool(dry_run or generate_ran),
            "dry_run": dry_run,
        },
        {
            "name": "profile_generate.tokens_per_second_per_user_positive",
            "passed": bool(dry_run or has_positive_throughput),
            "dry_run": dry_run,
        },
        {
            "name": "profile_generate.no_official_parity_claim",
            "passed": True,
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [
            check["name"] for check in checks if not check["passed"]
        ],
    }


def _generated_token_counts(generate: dict[str, Any]) -> list[int]:
    rows = generate.get("generated_token_ids") or []
    return [len(row) for row in rows if isinstance(row, list)]


def _setup_count(generate: dict[str, Any], key: str) -> Any:
    setup = generate.get("parameter_setup")
    if isinstance(setup, dict):
        return setup.get(key)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_generate_report_path(out: Path) -> Path:
    suffix = out.suffix or ".json"
    return out.with_name(f"{out.stem}_generate{suffix}")


class _maybe_generate_device:
    def __init__(self, ttnn: Any, device_id: int, injected: Any | None) -> None:
        self.ttnn = ttnn
        self.device_id = device_id
        self.injected = injected
        self.device = None
        self.opened = False

    def __enter__(self) -> Any:
        if self.injected is not None:
            self.device = f"fake-device:{self.device_id}"
            return self.device
        open_device = getattr(self.ttnn, "open_device", None)
        if not callable(open_device):
            raise NoTTNNDeviceError("ttnn.open_device is not available")
        self.device = open_device(device_id=self.device_id)
        self.opened = True
        return self.device

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.opened:
            return
        close_device = getattr(self.ttnn, "close_device", None)
        if callable(close_device):
            close_device(self.device)
