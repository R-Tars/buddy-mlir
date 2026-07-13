from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codegen.parameters import load_llama_parameters_from_manifests
from ..codegen.ttnn_tensorizer import (
    load_parameter_config_from_program,
    to_ttnn_parameters,
)
from ..smoke_single_layer_decode import (
    DECODE_PARAMETER_ROLES,
    _materialization_summary,
    _tensorization_summary,
)
from .context import TTNNDirectRuntimeContext
from .kv_cache import build_prompt_decode_kv_cache_tensors
from .prefill import (
    attach_prefill_rotary_parameters,
    build_prefill_page_table_tensor,
    prefill_token_ids_tensor,
)
from .tokenizer import PrefillPromptTokenization, tokenize_prompt_for_prefill


GENERATE_RUNTIME_OWNER = "TTNNDirectRuntimeContext"


def build_generate_state(
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
    prefill_tokenization: PrefillPromptTokenization | None = None,
) -> TTNNDirectRuntimeContext:
    if prefill_tokenization is None:
        prefill_tokenization = tokenize_prompt_for_prefill(
            prompt=prompt,
            batch_size=int(prefill_plan["batch_size"]),
            prefill_len=int(prefill_plan["prefill_len"]),
            tokenizer_path=tokenizer_path,
            vocab_size=prefill_plan.get("vocab_size"),
            tokenizer_module=tokenizer_module,
        )
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
    prefill_token_ids = prefill_token_ids_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        token_ids=prefill_tokenization.token_ids,
    )
    prefill_page_table = build_prefill_page_table_tensor(
        ttnn=ttnn,
        torch=torch,
        device=device,
        batch_size=int(prefill_plan["batch_size"]),
        cache_len=int(prefill_plan["cache_len"]),
        page_block_size=int(prefill_plan["kv_cache"]["page_block_size"]),
        prompt_token_count=int(prefill_tokenization.effective_token_count),
    )
    kv_runtime = build_prompt_decode_kv_cache_tensors(
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
    prefill_rotary = attach_prefill_rotary_parameters(
        parameters=result.parameters,
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        plan=prefill_plan,
    )
    tensorization_count = int(result.report["tensor_count"])
    tensor_conversion_count = (
        tensorization_count
        + 1
        + int(prefill_page_table.tensor_conversion_count)
        + int(kv_runtime.tensor_conversion_count)
        + int(prefill_rotary.tensor_conversion_count)
    )
    return TTNNDirectRuntimeContext(
        parameters=result.parameters,
        prefill_token_ids=prefill_token_ids,
        prefill_page_table=prefill_page_table.page_table,
        kv_cache=kv_runtime.kv_cache,
        tensor_conversion_count=tensor_conversion_count,
        parameter_source="hf_model",
        input_source="prompt_prefill",
        prefill_tokenization=prefill_tokenization.to_report(),
        prefill_page_table_runtime_state=(
            prefill_page_table.prefill_page_table_runtime_state
        ),
        kv_cache_runtime_state=kv_runtime.kv_cache_runtime_state,
        prefill_prompt_runtime_input_tensor_count=1,
        prefill_page_table_runtime_input_tensor_count=(
            prefill_page_table.tensor_conversion_count
        ),
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
            "prefill_page_table_runtime_input_tensor_count": (
                prefill_page_table.tensor_conversion_count
            ),
            "prefill_page_table_runtime_state": (
                prefill_page_table.prefill_page_table_runtime_state
            ),
            "prefill_rotary_runtime_input_tensor_count": (
                prefill_rotary.tensor_conversion_count
            ),
            "prefill_rotary_runtime_state": (
                prefill_rotary.rotary_runtime_state
            ),
            "kv_cache_runtime_input_tensor_count": (
                kv_runtime.tensor_conversion_count
            ),
            "kv_cache_runtime_state": kv_runtime.kv_cache_runtime_state,
        },
        tokenizer_path=tokenizer_path,
        tokenizer_module=tokenizer_module,
    )
