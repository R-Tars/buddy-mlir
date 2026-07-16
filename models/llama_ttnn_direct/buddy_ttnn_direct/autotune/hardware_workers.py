from __future__ import annotations

import importlib
import math
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from ..runtime.config_runtime import realize_ttnn_config
from ..ttnn_compat.ops import paged_sdpa_decode
from .microbench import MicrobenchmarkError, make_worker_response


def matmul_trace_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """Measure one representative MatMul config through an isolated TTNN trace."""

    payload = _mapping(request.get("payload"), "payload")
    candidate = _mapping(request.get("candidate"), "candidate")
    measurement = _mapping(
        candidate.get("measurement_contract"), "measurement_contract"
    )
    execution = _mapping(
        candidate.get("execution_contract"), "execution_contract"
    )
    _validate_execution_contract(execution)

    ttnn = importlib.import_module("ttnn")
    torch = importlib.import_module("torch")
    device_id = int(payload.get("device_id", 0))
    shape = _mapping(payload.get("shape"), "shape")
    m, k, n = (int(shape[name]) for name in ("m", "k", "n"))
    if min(m, k, n) <= 0 or any(value % 32 for value in (m, k, n)):
        raise MicrobenchmarkError(
            "MatMul worker shapes must be positive tile multiples"
        )

    fill_value = float(payload.get("fill_value", 0.015625))
    if not math.isfinite(fill_value) or fill_value == 0:
        raise MicrobenchmarkError(
            "MatMul fill_value must be finite and nonzero"
        )
    warmup = int(measurement["warmup"])
    iterations = int(measurement["iterations"])

    with _managed_device(ttnn, device_id) as device:
        input_memory = realize_ttnn_config(
            _mapping(payload.get("input_memory"), "input_memory"), ttnn
        )
        weight_memory = realize_ttnn_config(
            _mapping(payload.get("weight_memory"), "weight_memory"), ttnn
        )
        output_memory = realize_ttnn_config(
            _mapping(payload.get("output_memory"), "output_memory"), ttnn
        )
        program_config = realize_ttnn_config(
            _mapping(payload.get("program_config"), "program_config"), ttnn
        )
        compute_kernel_config = realize_ttnn_config(
            _mapping(
                payload.get("compute_kernel_config"), "compute_kernel_config"
            ),
            ttnn,
        )
        input_dtype = _dtype(ttnn, str(payload.get("input_dtype", "bfloat16")))
        weight_dtype = _dtype(
            ttnn, str(payload.get("weight_dtype", "bfloat8_b"))
        )
        output_dtype = _dtype(
            ttnn, str(payload.get("output_dtype", "bfloat16"))
        )

        input_tensor = ttnn.full(
            (1, 1, m, k),
            fill_value,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_memory,
        )
        weight_tensor = ttnn.full(
            (1, 1, k, n),
            fill_value,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_memory,
        )

        def linear() -> Any:
            return ttnn.linear(
                input_tensor,
                weight_tensor,
                memory_config=output_memory,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
            )

        output = linear()
        _synchronize(ttnn, device)
        correctness = _constant_matmul_correctness(
            torch=torch,
            output=ttnn.to_torch(output).to(torch.float32),
            expected=k * fill_value * fill_value,
        )
        if not correctness["passed"]:
            raise MicrobenchmarkError(
                "MatMul correctness gate failed: " + str(correctness)
            )

        cache_before_capture = _program_cache_count(device)
        trace_id = None
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            linear()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            cache_after_capture = _program_cache_count(device)

            for _ in range(warmup):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

            samples: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        finally:
            if trace_id is not None:
                release = getattr(ttnn, "release_trace", None)
                if callable(release):
                    release(device, trace_id)

        return make_worker_response(
            request,
            samples,
            program_cache_count=cache_after_capture,
            trace_capture_count=1,
            new_tensor_allocations=0,
            metadata={
                "device_id": device_id,
                "shape": {"m": m, "k": k, "n": n},
                "program_family": payload.get("program_family"),
                "program_cache_before_capture": cache_before_capture,
                "program_cache_after_capture": cache_after_capture,
                "new_programs_after_capture": (
                    cache_after_capture - cache_before_capture
                ),
                "trace_replay": True,
                "persistent_inputs": True,
                "correctness": correctness,
            },
        )


def packed_gate_up_region_trace_worker(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure separate or packed gate/up through the complete MLP subregion."""

    payload = _mapping(request.get("payload"), "payload")
    candidate = _mapping(request.get("candidate"), "candidate")
    measurement = _mapping(
        candidate.get("measurement_contract"), "measurement_contract"
    )
    execution = _mapping(
        candidate.get("execution_contract"), "execution_contract"
    )
    _validate_execution_contract(execution)

    mode = str(payload.get("mode"))
    if mode not in {"incumbent", "challenger"}:
        raise MicrobenchmarkError(
            "packed gate/up region mode must be 'incumbent' or 'challenger'"
        )
    split_strategy = str(payload.get("split_strategy", "split"))
    if split_strategy not in {"split", "slice"}:
        raise MicrobenchmarkError(
            "packed gate/up split strategy must be 'split' or 'slice'"
        )

    ttnn = importlib.import_module("ttnn")
    torch = importlib.import_module("torch")
    device_id = int(payload.get("device_id", 0))
    shape = _mapping(payload.get("shape"), "shape")
    m, k, n = (int(shape[name]) for name in ("m", "k", "n"))
    if min(m, k, n) <= 0 or any(value % 32 for value in (m, k, n)):
        raise MicrobenchmarkError(
            "packed gate/up worker shapes must be positive tile multiples"
        )
    fill_value = float(payload.get("fill_value", 0.015625))
    if not math.isfinite(fill_value) or fill_value == 0:
        raise MicrobenchmarkError(
            "packed gate/up fill_value must be finite and nonzero"
        )
    warmup = int(measurement["warmup"])
    iterations = int(measurement["iterations"])

    with _managed_device(ttnn, device_id) as device:
        input_memory = realize_ttnn_config(
            _mapping(payload.get("input_memory"), "input_memory"), ttnn
        )
        weight_memory = realize_ttnn_config(
            _mapping(payload.get("weight_memory"), "weight_memory"), ttnn
        )
        output_memory = realize_ttnn_config(
            _mapping(payload.get("output_memory"), "output_memory"), ttnn
        )
        split_memory = realize_ttnn_config(
            _mapping(payload.get("split_memory"), "split_memory"), ttnn
        )
        mul_memory = realize_ttnn_config(
            _mapping(payload.get("mul_memory"), "mul_memory"), ttnn
        )
        compute_kernel_config = realize_ttnn_config(
            _mapping(
                payload.get("compute_kernel_config"), "compute_kernel_config"
            ),
            ttnn,
        )
        input_dtype = _dtype(ttnn, str(payload.get("input_dtype", "bfloat16")))
        weight_dtype = _dtype(
            ttnn, str(payload.get("weight_dtype", "bfloat4_b"))
        )
        output_dtype = _dtype(
            ttnn, str(payload.get("output_dtype", "bfloat16"))
        )

        input_tensor = ttnn.full(
            (1, 1, m, k),
            fill_value,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_memory,
        )
        weight_width = 2 * n if mode == "challenger" else n
        first_weight = ttnn.full(
            (1, 1, k, weight_width),
            fill_value,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_memory,
        )
        second_weight = None
        if mode == "incumbent":
            second_weight = ttnn.full(
                (1, 1, k, n),
                fill_value,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=weight_memory,
            )

        if mode == "incumbent":
            gate_program = realize_ttnn_config(
                _mapping(
                    payload.get("gate_program_config"), "gate_program_config"
                ),
                ttnn,
            )
            up_program = realize_ttnn_config(
                _mapping(payload.get("up_program_config"), "up_program_config"),
                ttnn,
            )
        else:
            packed_program = realize_ttnn_config(
                _mapping(payload.get("program_config"), "program_config"), ttnn
            )

        activation = _silu_activation(ttnn)
        requires_conversion = bool(
            payload.get("requires_mul_conversion", False)
        )

        def split_packed(packed: Any) -> tuple[Any, Any]:
            try:
                if split_strategy == "split":
                    result = ttnn.split(
                        packed,
                        n,
                        dim=-1,
                        memory_config=split_memory,
                    )
                    if len(result) != 2:
                        raise MicrobenchmarkError(
                            "packed gate/up split did not produce two tensors"
                        )
                    return result[0], result[1]
                starts = [0, 0, 0, 0]
                middle = [1, 1, m, n]
                second = [0, 0, 0, n]
                end = [1, 1, m, 2 * n]
                return (
                    ttnn.slice(
                        packed, starts, middle, memory_config=split_memory
                    ),
                    ttnn.slice(packed, second, end, memory_config=split_memory),
                )
            except Exception as exc:
                raise MicrobenchmarkError(
                    f"packed gate/up {split_strategy} stage failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        def mul_silu(gate: Any, up: Any) -> Any:
            try:
                if requires_conversion:
                    gate = ttnn.to_memory_config(gate, mul_memory)
                    up = ttnn.to_memory_config(up, mul_memory)
                kwargs = {
                    "memory_config": mul_memory,
                    "dtype": output_dtype,
                }
                if activation is not None:
                    kwargs["input_tensor_a_activations"] = [activation]
                return ttnn.mul(gate, up, **kwargs)
            except Exception as exc:
                raise MicrobenchmarkError(
                    "packed gate/up mul_silu stage failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        def region() -> Any:
            if mode == "incumbent":
                gate = ttnn.linear(
                    input_tensor,
                    first_weight,
                    memory_config=output_memory,
                    program_config=gate_program,
                    compute_kernel_config=compute_kernel_config,
                    dtype=output_dtype,
                )
                up = ttnn.linear(
                    input_tensor,
                    second_weight,
                    memory_config=output_memory,
                    program_config=up_program,
                    compute_kernel_config=compute_kernel_config,
                    dtype=output_dtype,
                )
            else:
                try:
                    packed = ttnn.linear(
                        input_tensor,
                        first_weight,
                        memory_config=output_memory,
                        program_config=packed_program,
                        compute_kernel_config=compute_kernel_config,
                        dtype=output_dtype,
                    )
                except Exception as exc:
                    raise MicrobenchmarkError(
                        "packed gate/up linear stage failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                gate, up = split_packed(packed)
            return mul_silu(gate, up)

        output = region()
        _synchronize(ttnn, device)
        projected = k * fill_value * fill_value
        expected = (projected / (1.0 + math.exp(-projected))) * projected
        correctness = _constant_tensor_correctness(
            torch=torch,
            output=ttnn.to_torch(output).to(torch.float32),
            expected=expected,
        )
        if not correctness["passed"]:
            raise MicrobenchmarkError(
                "packed gate/up region correctness gate failed: "
                + str(correctness)
            )

        cache_before_capture = _program_cache_count(device)
        trace_id = None
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            region()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            cache_after_capture = _program_cache_count(device)
            for _ in range(warmup):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        finally:
            if trace_id is not None:
                release = getattr(ttnn, "release_trace", None)
                if callable(release):
                    release(device, trace_id)

        if mode == "incumbent":
            operations = ["linear.gate", "linear.up", "mul_silu"]
        else:
            operations = ["linear.gate_up_packed", split_strategy]
            if requires_conversion:
                operations.extend(
                    ["to_memory_config.gate", "to_memory_config.up"]
                )
            operations.append("mul_silu")
        return make_worker_response(
            request,
            samples,
            program_cache_count=cache_after_capture,
            trace_capture_count=1,
            new_tensor_allocations=0,
            metadata={
                "device_id": device_id,
                "mode": mode,
                "shape": {"m": m, "k": k, "n": n},
                "representative_layer": payload.get("representative_layer"),
                "layer_group": payload.get("layer_group"),
                "program_family": payload.get("program_family"),
                "split_strategy": split_strategy,
                "operation_sequence": operations,
                "linear_count": 2 if mode == "incumbent" else 1,
                "split_operation_count": 0 if mode == "incumbent" else 1,
                "added_conversion_count": 2 if requires_conversion else 0,
                "program_cache_before_capture": cache_before_capture,
                "program_cache_after_capture": cache_after_capture,
                "new_programs_after_capture": (
                    cache_after_capture - cache_before_capture
                ),
                "trace_replay": True,
                "persistent_inputs": True,
                "correctness": correctness,
            },
        )


def sdpa_context_region_trace_worker(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure paged decode SDPA plus its post-kernel memory transition."""

    payload = _mapping(request.get("payload"), "payload")
    candidate = _mapping(request.get("candidate"), "candidate")
    measurement = _mapping(
        candidate.get("measurement_contract"), "measurement_contract"
    )
    execution = _mapping(
        candidate.get("execution_contract"), "execution_contract"
    )
    _validate_execution_contract(execution)

    ttnn = importlib.import_module("ttnn")
    torch = importlib.import_module("torch")
    device_id = int(payload.get("device_id", 0))
    batch_size = int(payload.get("batch_size", 32))
    num_heads = int(payload.get("num_heads", 32))
    num_kv_heads = int(payload.get("num_kv_heads", 8))
    head_dim = int(payload.get("head_dim", 128))
    physical_cache_len = int(payload.get("physical_cache_len", 1024))
    active_context_len = int(
        payload.get("active_context_len", physical_cache_len)
    )
    page_block_size = int(payload.get("page_block_size", 32))
    if min(batch_size, num_heads, num_kv_heads, head_dim) <= 0:
        raise MicrobenchmarkError("SDPA dimensions must be positive")
    if not 1 <= active_context_len <= physical_cache_len:
        raise MicrobenchmarkError("active SDPA context exceeds physical cache")
    if physical_cache_len % page_block_size:
        raise MicrobenchmarkError("physical cache must be page-block aligned")
    warmup = int(measurement["warmup"])
    iterations = int(measurement["iterations"])

    with _managed_device(ttnn, device_id) as device:
        program_config = realize_ttnn_config(
            _mapping(payload.get("program_config"), "program_config"), ttnn
        )
        kernel_memory = realize_ttnn_config(
            _mapping(
                payload.get("kernel_output_memory"), "kernel_output_memory"
            ),
            ttnn,
        )
        post_memory = realize_ttnn_config(
            _mapping(
                payload.get("post_sdpa_output_memory"), "post output memory"
            ),
            ttnn,
        )
        official_program = realize_ttnn_config(
            _mapping(
                payload.get("official_program_config"), "official program"
            ),
            ttnn,
        )
        official_kernel_memory = realize_ttnn_config(
            _mapping(
                payload.get("official_kernel_output_memory"),
                "official kernel memory",
            ),
            ttnn,
        )
        official_post_memory = realize_ttnn_config(
            _mapping(
                payload.get("official_post_sdpa_output_memory"),
                "official post memory",
            ),
            ttnn,
        )
        query_memory = realize_ttnn_config(
            _mapping(payload.get("query_memory"), "query memory"), ttnn
        )
        dtype = _dtype(ttnn, str(payload.get("dtype", "bfloat16")))
        page_count = physical_cache_len // page_block_size
        cache_shape = (
            batch_size * page_count,
            num_kv_heads,
            page_block_size,
            head_dim,
        )
        generator = torch.Generator().manual_seed(int(payload.get("seed", 17)))
        query_host = torch.randn(
            (1, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16,
            generator=generator,
        )
        key_host = torch.randn(
            cache_shape,
            dtype=torch.bfloat16,
            generator=generator,
        )
        value_host = torch.randn(
            cache_shape,
            dtype=torch.bfloat16,
            generator=generator,
        )
        page_table_host = torch.arange(
            batch_size * page_count, dtype=torch.int32
        ).reshape(batch_size, page_count)
        cache_position_host = torch.full(
            (batch_size,), active_context_len - 1, dtype=torch.int32
        )
        query = ttnn.from_torch(
            query_host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=query_memory,
        )
        key_cache = ttnn.from_torch(
            key_host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        value_cache = ttnn.from_torch(
            value_host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        page_table = ttnn.from_torch(
            page_table_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cache_position = ttnn.from_torch(
            cache_position_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        scale = float(payload.get("scale", head_dim**-0.5))

        def region(program: Any, kernel: Any, post: Any) -> Any:
            output = paged_sdpa_decode(
                ttnn,
                query,
                key_cache,
                value_cache,
                page_table,
                cache_position,
                scale=scale,
                memory_config=kernel,
                program_config=program,
            )
            memory_getter = getattr(output, "memory_config", None)
            output_memory = memory_getter() if callable(memory_getter) else None
            if output_memory != post:
                output = ttnn.to_memory_config(output, memory_config=post)
            return output

        official_output = region(
            official_program, official_kernel_memory, official_post_memory
        )
        candidate_output = region(program_config, kernel_memory, post_memory)
        _synchronize(ttnn, device)
        correctness = _tensor_pair_correctness(
            torch=torch,
            reference=ttnn.to_torch(official_output).to(torch.float32),
            candidate=ttnn.to_torch(candidate_output).to(torch.float32),
        )
        if not correctness["passed"]:
            raise MicrobenchmarkError(
                "SDPA correctness gate failed: " + str(correctness)
            )

        cache_before_capture = _program_cache_count(device)
        trace_id = None
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            region(program_config, kernel_memory, post_memory)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            cache_after_capture = _program_cache_count(device)
            for _ in range(warmup):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        finally:
            if trace_id is not None:
                release = getattr(ttnn, "release_trace", None)
                if callable(release):
                    release(device, trace_id)

        return make_worker_response(
            request,
            samples,
            program_cache_count=cache_after_capture,
            trace_capture_count=1,
            new_tensor_allocations=0,
            metadata={
                "device_id": device_id,
                "bucket_name": payload.get("bucket_name"),
                "physical_cache_len": physical_cache_len,
                "active_context_len": active_context_len,
                "batch_size": batch_size,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "operation_sequence": [
                    "paged_scaled_dot_product_attention_decode",
                    "to_memory_config.post_sdpa",
                ],
                "program_cache_before_capture": cache_before_capture,
                "program_cache_after_capture": cache_after_capture,
                "new_programs_after_capture": (
                    cache_after_capture - cache_before_capture
                ),
                "trace_replay": True,
                "persistent_inputs": True,
                "correctness": correctness,
            },
        )


def _constant_matmul_correctness(
    *,
    torch: Any,
    output: Any,
    expected: float,
) -> dict[str, Any]:
    observed = float(output.mean().item())
    maximum_deviation = float((output - observed).abs().max().item())
    finite = bool(
        torch.isfinite(output).all().item()
        and math.isfinite(observed)
        and math.isfinite(maximum_deviation)
    )
    relative_error = abs(observed - expected) / max(abs(expected), 1e-12)
    passed = (
        finite
        and relative_error <= 0.15
        and maximum_deviation <= max(abs(observed) * 0.01, 1e-3)
    )
    return {
        "passed": passed,
        "finite": finite,
        "expected_mean": expected,
        "observed_mean": observed,
        "relative_error": relative_error,
        "maximum_deviation": maximum_deviation,
    }


def _tensor_pair_correctness(
    *,
    torch: Any,
    reference: Any,
    candidate: Any,
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "passed": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference = reference.flatten()
    candidate = candidate.flatten()
    finite = bool(
        torch.isfinite(reference).all().item()
        and torch.isfinite(candidate).all().item()
    )
    difference = (reference - candidate).abs()
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    reference_centered = reference - reference.mean()
    candidate_centered = candidate - candidate.mean()
    denominator = torch.sqrt(
        (reference_centered.square().sum())
        * (candidate_centered.square().sum())
    )
    denominator_value = float(denominator.item())
    pcc = (
        float(
            (
                (reference_centered * candidate_centered).sum() / denominator
            ).item()
        )
        if denominator_value > 0
        else (1.0 if max_abs_error == 0.0 else 0.0)
    )
    return {
        "passed": finite and pcc >= 0.999 and max_abs_error <= 0.1,
        "shape_match": True,
        "finite": finite,
        "pcc": pcc,
        "pcc_threshold": 0.999,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
    }


def _constant_tensor_correctness(
    *,
    torch: Any,
    output: Any,
    expected: float,
) -> dict[str, Any]:
    observed = float(output.mean().item())
    maximum_deviation = float((output - observed).abs().max().item())
    finite = bool(
        torch.isfinite(output).all().item()
        and math.isfinite(observed)
        and math.isfinite(maximum_deviation)
    )
    relative_error = abs(observed - expected) / max(abs(expected), 1e-12)
    passed = (
        finite
        and relative_error <= 0.2
        and maximum_deviation <= max(abs(observed) * 0.02, 1e-3)
    )
    return {
        "passed": passed,
        "finite": finite,
        "expected_mean": expected,
        "observed_mean": observed,
        "relative_error": relative_error,
        "maximum_deviation": maximum_deviation,
    }


def _silu_activation(ttnn: Any) -> Any | None:
    unary_with_param = getattr(ttnn, "UnaryWithParam", None)
    unary_op_type = getattr(ttnn, "UnaryOpType", None)
    silu = (
        getattr(unary_op_type, "SILU", None)
        if unary_op_type is not None
        else None
    )
    return (
        unary_with_param(silu)
        if unary_with_param is not None and silu is not None
        else None
    )


def _validate_execution_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "after_prefill": True,
        "sampling": "force_argmax",
        "page_table": "fixed",
    }
    mismatches = {
        key: {"expected": wanted, "observed": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise MicrobenchmarkError(
            f"MatMul worker execution contract mismatch: {mismatches}"
        )


def _dtype(ttnn: Any, value: str) -> Any:
    aliases = {
        "bf16": "bfloat16",
        "bfp8": "bfloat8_b",
        "bfp4": "bfloat4_b",
        "fp32": "float32",
    }
    name = aliases.get(value, value)
    dtype = getattr(ttnn, name, None)
    if dtype is None:
        raise MicrobenchmarkError(f"TTNN does not expose dtype {name!r}")
    return dtype


def _program_cache_count(device: Any) -> int:
    getter = getattr(device, "num_program_cache_entries", None)
    return int(getter()) if callable(getter) else 0


def _synchronize(ttnn: Any, device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)


@contextmanager
def _managed_device(ttnn: Any, device_id: int):
    open_device = getattr(ttnn, "open_device", None)
    close_device = getattr(ttnn, "close_device", None)
    if not callable(open_device) or not callable(close_device):
        raise MicrobenchmarkError("TTNN device open/close APIs are unavailable")
    try:
        device = open_device(device_id=device_id)
    except TypeError:
        device = open_device(device_id)
    try:
        yield device
    finally:
        close_device(device)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MicrobenchmarkError(f"{label} must be an object")
    return dict(value)
