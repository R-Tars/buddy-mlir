from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


class TTNNDirectRuntimeContext:
    """Owns the TTNN Direct runtime state for one generate invocation."""

    def __init__(
        self,
        *,
        parameters: Any,
        prefill_token_ids: Any,
        prefill_page_table: Any,
        kv_cache: Any,
        tensor_conversion_count: int,
        parameter_source: str,
        input_source: str,
        prefill_tokenization: dict[str, Any],
        prefill_page_table_runtime_state: dict[str, Any],
        kv_cache_runtime_state: dict[str, Any],
        prefill_prompt_runtime_input_tensor_count: int,
        prefill_page_table_runtime_input_tensor_count: int,
        prefill_rotary_runtime_input_tensor_count: int,
        parameter_setup: dict[str, Any],
        tokenizer_path: str | Path,
        tokenizer_module: Any | None,
    ) -> None:
        self.parameters = parameters
        self.prefill_token_ids = prefill_token_ids
        self.prefill_page_table = prefill_page_table
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
        self.prefill_page_table_runtime_state = prefill_page_table_runtime_state
        self.kv_cache_runtime_state = kv_cache_runtime_state
        self.prefill_prompt_runtime_input_tensor_count = int(
            prefill_prompt_runtime_input_tensor_count
        )
        self.prefill_page_table_runtime_input_tensor_count = int(
            prefill_page_table_runtime_input_tensor_count
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
        self.cache_position_update_count = 0
        self.rotary_state_update_count = 0
        self.generated_model_initialization_count = 0
        self.decode_token_update_count = 0
        self.decode_token_runtime_handoff = "device_tensor_direct"
        self.decode_token_host_roundtrip_per_step = False
        self.host_token_materialization_for_reporting_only = True
        self.runtime_input_report = None

    def install_generated_model(
        self,
        *,
        generated_module: Any,
        generated_model: Any,
    ) -> None:
        self.generated_module = generated_module
        self.generated_model = generated_model
        self.generated_model_initialization_count += 1

    def install_decode_runtime(
        self,
        runtime_state: SimpleNamespace,
        *,
        page_table_updated: bool = True,
        cache_position_updated: bool = True,
        rotary_state_updated: bool = True,
    ) -> None:
        self.page_table = runtime_state.page_table
        self.cache_position = runtime_state.cache_position
        self.decode_runtime_state = runtime_state.decode_runtime_state
        self.rotary_state = runtime_state.rotary_runtime_state
        self.page_table_update_count += int(page_table_updated)
        self.cache_position_update_count += int(cache_position_updated)
        self.rotary_state_update_count += int(rotary_state_updated)

    def set_runtime_input_report(self, report: dict[str, Any]) -> None:
        self.runtime_input_report = dict(report)

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
            "cache_position_update_count": self.cache_position_update_count,
            "rotary_state_update_count": self.rotary_state_update_count,
            "decode_token_update_count": self.decode_token_update_count,
            "decode_token_runtime_handoff": (
                self.decode_token_runtime_handoff
            ),
            "decode_token_host_roundtrip_per_step": (
                self.decode_token_host_roundtrip_per_step
            ),
            "host_token_materialization_for_reporting_only": (
                self.host_token_materialization_for_reporting_only
            ),
            "current_kv_cache_layers": len(self.kv_cache or []),
            "prefill_page_table_shape": _shape(self.prefill_page_table),
            "prefill_page_table_runtime_state": (
                self.prefill_page_table_runtime_state
            ),
            "current_page_table_shape": _shape(self.page_table),
            "current_cache_position_shape": _shape(self.cache_position),
            "current_rotary_state": self.rotary_state,
            "runtime_inputs": self.runtime_input_report,
        }


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]
