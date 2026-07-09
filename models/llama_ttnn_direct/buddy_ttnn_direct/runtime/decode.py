from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from ..decode_loop import _loop_generated_token_ids
from ..smoke_single_layer_decode import (
    _attach_runtime_rotary_parameters,
    _build_prompt_decode_runtime_state_tensors,
)


def prefill_token_direct_handoff(*, prefill_token: Any) -> SimpleNamespace:
    return SimpleNamespace(
        status="device_tensor_direct",
        source="prefill_output_tensor",
        token_ids=prefill_token,
        tensor_conversion_count=0,
        runtime_handoff="device_tensor_direct",
        runtime_host_roundtrip=False,
    )


def build_decode_runtime_for_position(
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


def materialize_generate_token_events(
    token_events: list[dict[str, Any]],
    *,
    step_reports: list[dict[str, Any]],
    ttnn: Any,
    batch_size: int,
) -> SimpleNamespace:
    generated_token_ids_by_user = [[] for _ in range(batch_size)]
    per_step_token_metadata: list[dict[str, Any]] = []
    first_token_materialization_ms = 0.0
    first_token_status = "not_run"
    first_token_source = "none"
    first_token_ids_by_user: list[list[int]] = []
    step_reports_by_index = {
        int(report["step_index"]): report
        for report in step_reports
        if isinstance(report.get("step_index"), int)
    }
    for event in token_events:
        materialization_start = time.perf_counter()
        materialization = _loop_generated_token_ids(
            token=event["token"],
            ttnn=ttnn,
            batch_size=batch_size,
        )
        materialization_ms = (
            time.perf_counter() - materialization_start
        ) * 1000.0
        token_ids = materialization["token_ids_by_user"]
        for user_index, row in enumerate(token_ids):
            generated_token_ids_by_user[user_index].extend(row)

        metadata = {
            "step_index": event["step_index"],
            "token_ids_by_user": token_ids,
            "token_materialization_status": materialization["status"],
            "token_materialization_source": materialization["source"],
            "token_materialization_ms": materialization_ms,
            "token_materialization_phase": "reporting_after_decode_loop",
            "runtime_handoff": event.get("runtime_handoff"),
            "runtime_host_roundtrip": bool(
                event.get("runtime_host_roundtrip")
            ),
            "cache_position_value": event.get("cache_position_value"),
            "page_table_shape": event.get("page_table_shape"),
            "token_shape": event.get("token_shape"),
        }
        per_step_token_metadata.append(metadata)
        if event["step_index"] == "prefill":
            first_token_materialization_ms = materialization_ms
            first_token_status = materialization["status"]
            first_token_source = materialization["source"]
            first_token_ids_by_user = token_ids
            continue
        report = step_reports_by_index.get(int(event["step_index"]))
        if report is None:
            continue
        report["generated_token_ids"] = token_ids
        report["token_materialization"] = materialization
        report["token_materialization_ms"] = materialization_ms
        report["token_materialization_phase"] = "reporting_after_decode_loop"

    return SimpleNamespace(
        generated_token_ids_by_user=generated_token_ids_by_user,
        per_step_token_metadata=per_step_token_metadata,
        first_token_materialization_ms=first_token_materialization_ms,
        first_token_status=first_token_status,
        first_token_source=first_token_source,
        first_token_ids_by_user=first_token_ids_by_user,
    )
