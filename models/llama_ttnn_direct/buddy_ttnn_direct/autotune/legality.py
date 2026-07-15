from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .space import (
    CoreGrid,
    MatmulProgramConfig,
    MemoryConfig,
    SDPAProgramConfig,
    SearchSpaceConfig,
)
from .templates import (
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    KV_UPDATE_AXIS,
    ROPE_AXIS,
    template_choice_from_runtime_config,
    validate_template_selection,
)

LEGALITY_SCHEMA_VERSION = 1
TILE_HEIGHT = 32
TILE_WIDTH = 32
DEFAULT_L1_SAFETY_FACTOR = 0.8

ERROR_CLASSES = (
    "shape_incompatible",
    "l1_overflow",
    "unsupported_layout",
    "invalid_program_config",
    "invalid_core_grid",
    "api_unavailable",
    "compile_error",
    "runtime_error",
)

_SHARDED_LAYOUTS = {"width_sharded", "height_sharded", "block_sharded"}
_TILE_BYTES = {
    "bf16": 2048,
    "bfloat16": 2048,
    "bfp8": 1088,
    "bfloat8_b": 1088,
    "bfp4": 576,
    "bfloat4_b": 576,
    "fp32": 4096,
    "float32": 4096,
}


@dataclass(frozen=True)
class DeviceDescriptor:
    name: str
    architecture: str
    compute_grid: CoreGrid
    worker_core_count: int
    l1_bytes_per_core: int
    reserved_l1_bytes_per_core: int = 0
    max_cb_pages: int = 256
    dram_grid_width: int | None = None

    def __post_init__(self) -> None:
        if self.worker_core_count <= 0:
            raise ValueError("worker_core_count must be positive")
        if self.worker_core_count > self.compute_grid.x * self.compute_grid.y:
            raise ValueError("worker_core_count exceeds the rectangular compute grid")
        if self.l1_bytes_per_core <= 0:
            raise ValueError("l1_bytes_per_core must be positive")
        if not 0 <= self.reserved_l1_bytes_per_core < self.l1_bytes_per_core:
            raise ValueError("reserved L1 must be smaller than total L1")
        if self.max_cb_pages <= 0:
            raise ValueError("max_cb_pages must be positive")

    @classmethod
    def p150a(cls) -> "DeviceDescriptor":
        # P150A is a 2-column-harvested Blackhole: 11 x 10 logical workers.
        return cls(
            name="p150a",
            architecture="blackhole",
            compute_grid=CoreGrid(11, 10),
            worker_core_count=110,
            l1_bytes_per_core=1_572_864,
            dram_grid_width=8,
        )

    @property
    def available_l1_bytes_per_core(self) -> int:
        return self.l1_bytes_per_core - self.reserved_l1_bytes_per_core

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "compute_grid": self.compute_grid.to_list(),
            "worker_core_count": self.worker_core_count,
            "l1_bytes_per_core": self.l1_bytes_per_core,
            "reserved_l1_bytes_per_core": self.reserved_l1_bytes_per_core,
            "available_l1_bytes_per_core": self.available_l1_bytes_per_core,
            "max_cb_pages": self.max_cb_pages,
            "dram_grid_width": self.dram_grid_width,
        }


@dataclass(frozen=True)
class TensorSpec:
    name: str
    logical_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    dtype: str
    layout: str
    memory: MemoryConfig
    cores: tuple[tuple[int, int], ...] = ()
    device: str = "p150a:0"

    def __post_init__(self) -> None:
        if not self.logical_shape or any(value <= 0 for value in self.logical_shape):
            raise ValueError(f"{self.name} logical shape must be positive")
        if len(self.logical_shape) != len(self.padded_shape) or any(
            value <= 0 for value in self.padded_shape
        ):
            raise ValueError(f"{self.name} padded shape is invalid")
        if any(
            logical > padded
            for logical, padded in zip(self.logical_shape, self.padded_shape)
        ):
            raise ValueError(f"{self.name} logical shape exceeds padded shape")
        if self.layout not in {"tile", "row_major"}:
            raise ValueError(f"{self.name} has unsupported tensor layout")
        if len(set(self.cores)) != len(self.cores):
            raise ValueError(f"{self.name} core coordinates must be unique")

    @property
    def width(self) -> int:
        return self.padded_shape[-1]

    @property
    def flattened_height(self) -> int:
        return math.prod(self.padded_shape[:-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "logical_shape": list(self.logical_shape),
            "padded_shape": list(self.padded_shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "memory": self.memory.to_dict(),
            "cores": [list(core) for core in self.cores],
            "device": self.device,
        }


@dataclass(frozen=True)
class MatmulWorkload:
    name: str
    m: int
    k: int
    n: int
    input_memory: MemoryConfig
    output_memory: MemoryConfig
    weight_memory: MemoryConfig
    input_dtype: str = "bf16"
    weight_dtype: str = "bfloat8_b"
    output_dtype: str = "bf16"
    intermediate_dtype: str = "bf16"
    batch_count: int = 1
    input_memory_path: str | None = None
    output_memory_path: str | None = None

    def __post_init__(self) -> None:
        if min(self.m, self.k, self.n, self.batch_count) <= 0:
            raise ValueError(f"{self.name} dimensions must be positive")
        for dtype in (
            self.input_dtype,
            self.weight_dtype,
            self.output_dtype,
            self.intermediate_dtype,
        ):
            _tile_bytes(dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": {"m": self.m, "k": self.k, "n": self.n},
            "batch_count": self.batch_count,
            "input_memory": self.input_memory.to_dict(),
            "output_memory": self.output_memory.to_dict(),
            "weight_memory": self.weight_memory.to_dict(),
            "input_dtype": self.input_dtype,
            "weight_dtype": self.weight_dtype,
            "output_dtype": self.output_dtype,
            "intermediate_dtype": self.intermediate_dtype,
            "input_memory_path": self.input_memory_path,
            "output_memory_path": self.output_memory_path,
        }


@dataclass(frozen=True)
class SDPAWorkload:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    cache_len: int
    page_block_size: int = 32
    query_dtype: str = "bf16"
    cache_dtype: str = "bfloat8_b"
    output_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if (
            min(
                self.batch_size,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.cache_len,
                self.page_block_size,
            )
            <= 0
        ):
            raise ValueError("SDPA dimensions must be positive")
        for dtype in (self.query_dtype, self.cache_dtype, self.output_dtype):
            _tile_bytes(dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "query_dtype": self.query_dtype,
            "cache_dtype": self.cache_dtype,
            "output_dtype": self.output_dtype,
        }


@dataclass(frozen=True)
class PagedFusedUpdateWorkload:
    key_input: TensorSpec
    value_input: TensorSpec
    key_cache: TensorSpec
    value_cache: TensorSpec
    page_table: TensorSpec
    update_indices: TensorSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_input": self.key_input.to_dict(),
            "value_input": self.value_input.to_dict(),
            "key_cache": self.key_cache.to_dict(),
            "value_cache": self.value_cache.to_dict(),
            "page_table": self.page_table.to_dict(),
            "update_indices": self.update_indices.to_dict(),
        }


@dataclass(frozen=True)
class FusedQKRoPEWorkload:
    q: TensorSpec
    k: TensorSpec
    cos: TensorSpec
    sin: TensorSpec
    transformation: TensorSpec
    q_output_memory: MemoryConfig
    k_output_memory: MemoryConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q.to_dict(),
            "k": self.k.to_dict(),
            "cos": self.cos.to_dict(),
            "sin": self.sin.to_dict(),
            "transformation": self.transformation.to_dict(),
            "q_output_memory": self.q_output_memory.to_dict(),
            "k_output_memory": self.k_output_memory.to_dict(),
        }


@dataclass(frozen=True)
class WorkloadSpec:
    matmuls: tuple[MatmulWorkload, ...]
    sdpa: SDPAWorkload
    paged_fused_update: PagedFusedUpdateWorkload | None = None
    fused_qk_rope: FusedQKRoPEWorkload | None = None

    @classmethod
    def from_runtime_config(cls, config: Mapping[str, Any]) -> "WorkloadSpec":
        batch = int(config["batch_size"])
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config["num_key_value_heads"])
        head_dim = int(config["head_dim"])
        cache_len = int(config["max_cache_len"])
        qkv_width = (num_heads + 2 * num_kv_heads) * head_dim
        decode_m = _round_up(batch, TILE_HEIGHT)

        def memory(path: str, fallback: str) -> MemoryConfig:
            value = _get_path(config, path)
            if isinstance(value, Mapping):
                return MemoryConfig.from_runtime_descriptor(value)
            return MemoryConfig.named(fallback)

        def weight(k: int, n: int) -> MemoryConfig:
            return MemoryConfig.from_runtime_descriptor(
                {
                    "kind": "ttnn_dram_sharded_memory_config",
                    "k": k,
                    "n": n,
                    "dram_grid_width": 8,
                }
            )

        input_width = "L1_WIDTH_SHARDED_MEMORY_CONFIG"
        matmuls = [
            MatmulWorkload(
                "attention.qkv",
                decode_m,
                hidden,
                qkv_width,
                memory("rms_norm.attention.output_memory_config", input_width),
                memory("attention.qkv_output_memory_config", input_width),
                weight(hidden, qkv_width),
                weight_dtype="bfloat8_b",
                input_memory_path="rms_norm.attention.output_memory_config",
                output_memory_path="attention.qkv_output_memory_config",
            ),
            MatmulWorkload(
                "attention.o_proj",
                decode_m,
                hidden,
                hidden,
                MemoryConfig.named(input_width),
                memory("attention.o_proj_output_memory_config", input_width),
                weight(hidden, hidden),
                weight_dtype="bfloat8_b",
                output_memory_path="attention.o_proj_output_memory_config",
            ),
            MatmulWorkload(
                "mlp.gate",
                decode_m,
                hidden,
                intermediate,
                memory("rms_norm.mlp.output_memory_config", input_width),
                memory("mlp.gate_output_memory_config", input_width),
                weight(hidden, intermediate),
                weight_dtype="bfloat4_b",
                input_memory_path="rms_norm.mlp.output_memory_config",
                output_memory_path="mlp.gate_output_memory_config",
            ),
            MatmulWorkload(
                "mlp.up",
                decode_m,
                hidden,
                intermediate,
                memory("rms_norm.mlp.output_memory_config", input_width),
                memory("mlp.up_output_memory_config", input_width),
                weight(hidden, intermediate),
                weight_dtype="bfloat4_b",
                input_memory_path="rms_norm.mlp.output_memory_config",
                output_memory_path="mlp.up_output_memory_config",
            ),
            MatmulWorkload(
                "mlp.down",
                decode_m,
                intermediate,
                hidden,
                memory("mlp.gate_output_memory_config", input_width),
                memory("mlp.down_output_memory_config", input_width),
                weight(intermediate, hidden),
                weight_dtype="bfloat8_b",
                input_memory_path="mlp.gate_output_memory_config",
                output_memory_path="mlp.down_output_memory_config",
            ),
        ]
        lm_head = config.get("lm_head") or {}
        splits = lm_head.get("splits") or []
        split_count = int(lm_head.get("split_count", len(splits) or 1))
        if splits:
            shard_widths = [
                int(split["vocab_end"]) - int(split["vocab_start"]) for split in splits
            ]
        else:
            vocab_size = int(config["vocab_size"])
            shard_widths = [math.ceil(vocab_size / split_count)] * split_count
        for index, shard_width in enumerate(shard_widths):
            matmuls.append(
                MatmulWorkload(
                    f"lm_head.shards[{index}]",
                    decode_m,
                    hidden,
                    shard_width,
                    memory("lm_head.input_memory_config", input_width),
                    memory("lm_head.output_memory_config", input_width),
                    weight(hidden, shard_width),
                    weight_dtype="bfloat8_b",
                    output_dtype="bfloat8_b",
                    input_memory_path="lm_head.input_memory_config",
                    output_memory_path="lm_head.output_memory_config",
                )
            )
        kv_update_template = template_choice_from_runtime_config(config, KV_UPDATE_AXIS)
        rope_template = template_choice_from_runtime_config(config, ROPE_AXIS)
        return cls(
            matmuls=tuple(matmuls),
            sdpa=SDPAWorkload(
                batch_size=batch,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_len=cache_len,
            ),
            paged_fused_update=(
                _infer_paged_fused_update_workload(config)
                if kv_update_template == FUSED_PAGED_UPDATE
                else None
            ),
            fused_qk_rope=(
                _infer_fused_qk_rope_workload(config)
                if rope_template == FUSED_QK_ROPE
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "matmuls": [item.to_dict() for item in self.matmuls],
            "sdpa": self.sdpa.to_dict(),
            "paged_fused_update": (
                self.paged_fused_update.to_dict()
                if self.paged_fused_update is not None
                else None
            ),
            "fused_qk_rope": (
                self.fused_qk_rope.to_dict() if self.fused_qk_rope is not None else None
            ),
        }


def _infer_paged_fused_update_workload(
    config: Mapping[str, Any],
) -> PagedFusedUpdateWorkload:
    batch = int(config["batch_size"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    cache_len = int(config["max_cache_len"])
    kv_config = config.get("kv_cache") or {}
    page_block_size = int(kv_config.get("page_block_size", 32))
    pages_per_user = math.ceil(cache_len / page_block_size)
    padded_heads = _round_up(num_kv_heads, TILE_HEIGHT)
    attention = config.get("attention") or {}
    key_descriptor = _required_memory_descriptor(
        attention,
        "fused_cache_key_memory_config",
    )
    value_descriptor = _required_memory_descriptor(
        attention,
        "fused_cache_value_memory_config",
    )
    key_memory = MemoryConfig.from_runtime_descriptor(key_descriptor)
    value_memory = MemoryConfig.from_runtime_descriptor(value_descriptor)
    key_input = TensorSpec(
        name="key_input",
        logical_shape=(1, batch, num_kv_heads, head_dim),
        padded_shape=(1, batch, padded_heads, head_dim),
        dtype="bf16",
        layout="tile",
        memory=key_memory,
        cores=_descriptor_cores(key_descriptor),
    )
    value_input = replace(
        key_input,
        name="value_input",
        memory=value_memory,
        cores=_descriptor_cores(value_descriptor),
    )
    cache_memory = MemoryConfig.named("DRAM_MEMORY_CONFIG")
    cache_dtype = _dtype_name(kv_config.get("dtype"), "bfloat8_b")
    cache_shape = (
        batch * pages_per_user,
        num_kv_heads,
        page_block_size,
        head_dim,
    )
    key_cache = TensorSpec(
        name="key_cache",
        logical_shape=cache_shape,
        padded_shape=cache_shape,
        dtype=cache_dtype,
        layout="tile",
        memory=cache_memory,
    )
    return PagedFusedUpdateWorkload(
        key_input=key_input,
        value_input=value_input,
        key_cache=key_cache,
        value_cache=replace(key_cache, name="value_cache"),
        page_table=TensorSpec(
            name="page_table",
            logical_shape=(batch, pages_per_user),
            padded_shape=(batch, pages_per_user),
            dtype="int32",
            layout="row_major",
            memory=cache_memory,
        ),
        update_indices=TensorSpec(
            name="update_indices",
            logical_shape=(batch,),
            padded_shape=(batch,),
            dtype="int32",
            layout="row_major",
            memory=cache_memory,
        ),
    )


def _infer_fused_qk_rope_workload(
    config: Mapping[str, Any],
) -> FusedQKRoPEWorkload:
    batch = int(config["batch_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    attention = config.get("attention") or {}
    q_descriptor = _required_memory_descriptor(attention, "fused_q_memory_config")
    k_descriptor = _required_memory_descriptor(attention, "fused_k_memory_config")
    cos_descriptor = _required_memory_descriptor(
        attention, "fused_rope_cos_sin_memory_config"
    )
    transform_descriptor = _required_memory_descriptor(
        attention, "fused_rope_transform_memory_config"
    )
    q_memory = MemoryConfig.from_runtime_descriptor(q_descriptor)
    k_memory = MemoryConfig.from_runtime_descriptor(k_descriptor)
    cos_memory = MemoryConfig.from_runtime_descriptor(cos_descriptor)
    transform_memory = MemoryConfig.from_runtime_descriptor(transform_descriptor)
    q = TensorSpec(
        name="q",
        logical_shape=(1, batch, num_heads, head_dim),
        padded_shape=(1, batch, _round_up(num_heads, TILE_HEIGHT), head_dim),
        dtype="bf16",
        layout="tile",
        memory=q_memory,
        cores=_descriptor_cores(q_descriptor),
    )
    k = TensorSpec(
        name="k",
        logical_shape=(1, batch, num_kv_heads, head_dim),
        padded_shape=(
            1,
            batch,
            _round_up(num_kv_heads, TILE_HEIGHT),
            head_dim,
        ),
        dtype="bf16",
        layout="tile",
        memory=k_memory,
        cores=_descriptor_cores(k_descriptor),
    )
    cos = TensorSpec(
        name="cos",
        logical_shape=(1, 2 * batch, 1, head_dim),
        padded_shape=(1, 2 * batch, TILE_HEIGHT, head_dim),
        dtype="bf16",
        layout="tile",
        memory=cos_memory,
        cores=_descriptor_cores(cos_descriptor),
    )
    transformation = TensorSpec(
        name="transformation",
        logical_shape=(1, 1, 2 * batch * TILE_HEIGHT, TILE_WIDTH),
        padded_shape=(1, 1, 2 * batch * TILE_HEIGHT, TILE_WIDTH),
        dtype="bf16",
        layout="tile",
        memory=transform_memory,
        cores=_descriptor_cores(transform_descriptor),
    )
    return FusedQKRoPEWorkload(
        q=q,
        k=k,
        cos=cos,
        sin=replace(cos, name="sin"),
        transformation=transformation,
        q_output_memory=q_memory,
        k_output_memory=k_memory,
    )


def _required_memory_descriptor(
    owner: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = owner.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"fused template requires runtime descriptor {name}")
    return value


def _descriptor_cores(
    descriptor: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    ranges = descriptor.get("core_ranges")
    if not isinstance(ranges, list):
        return ()
    selected: list[tuple[int, int]] = []
    for rectangle in ranges:
        coordinates = _rectangle_coordinates(rectangle)
        if coordinates is None:
            raise ValueError(f"invalid core range: {rectangle!r}")
        selected.extend(sorted(coordinates, key=lambda item: (item[1], item[0])))
    return tuple(selected)


def _dtype_name(value: Any, fallback: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name", fallback))
    if value is None:
        return fallback
    return str(value)


@dataclass(frozen=True)
class LegalityIssue:
    code: str
    error_class: str
    path: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.error_class not in ERROR_CLASSES:
            raise ValueError(f"unsupported legality error class: {self.error_class}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "error_class": self.error_class,
            "path": self.path,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class L1Estimate:
    path: str
    components: Mapping[str, int]
    cb_pages: Mapping[str, int]
    total_bytes: int
    available_bytes: int
    safety_factor: float
    limit_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "components": dict(self.components),
            "cb_pages": dict(self.cb_pages),
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "safety_factor": self.safety_factor,
            "limit_bytes": self.limit_bytes,
            "utilization_of_available": self.total_bytes / self.available_bytes,
            "within_limit": self.total_bytes <= self.limit_bytes,
        }


@dataclass(frozen=True)
class CompileValidationRequest:
    command: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("compile validation command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("compile validation timeout must be positive")


@dataclass(frozen=True)
class CompileValidationResult:
    status: str
    passed: bool
    error_class: str | None
    detail_code: str | None
    return_code: int | None
    elapsed_seconds: float
    command: tuple[str, ...]
    cwd: str | None
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "error_class": self.error_class,
            "detail_code": self.detail_code,
            "return_code": self.return_code,
            "elapsed_seconds": self.elapsed_seconds,
            "command": list(self.command),
            "cwd": self.cwd,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass(frozen=True)
class LegalityReport:
    candidate_id: str
    status: str
    passed: bool
    static_passed: bool
    error_class: str | None
    issues: tuple[LegalityIssue, ...]
    l1_estimates: tuple[L1Estimate, ...]
    device: DeviceDescriptor
    safety_factor: float
    compile_validation: CompileValidationResult | None = None
    schema_version: int = LEGALITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "passed": self.passed,
            "static_passed": self.static_passed,
            "error_class": self.error_class,
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count": len(self.issues),
            "l1_estimates": [estimate.to_dict() for estimate in self.l1_estimates],
            "device": self.device.to_dict(),
            "safety_factor": self.safety_factor,
            "compile_validation": (
                self.compile_validation.to_dict()
                if self.compile_validation is not None
                else None
            ),
        }


def validate_candidate(
    space: SearchSpaceConfig,
    workload: WorkloadSpec,
    device: DeviceDescriptor,
    *,
    candidate_id: str = "candidate",
    safety_factor: float = DEFAULT_L1_SAFETY_FACTOR,
    compile_request: CompileValidationRequest | None = None,
) -> LegalityReport:
    if not 0 < safety_factor <= 1:
        raise ValueError("L1 safety_factor must be in (0, 1]")

    issues: list[LegalityIssue] = []
    estimates: list[L1Estimate] = []
    _validate_declared_grids(space, device, issues)
    _validate_declared_memory(space, device, issues)

    workloads_by_name = {item.name: item for item in workload.matmuls}
    for operator_name in sorted(space.operators):
        operator = space.operators[operator_name]
        if operator.get("kind") == "matmul":
            matching = _matching_matmul_workloads(operator_name, workloads_by_name)
            if not matching:
                _issue(
                    issues,
                    "MATMUL_SHAPE_MISSING",
                    "shape_incompatible",
                    operator_name,
                    "no representative matmul shape was supplied",
                )
                continue
            programs = operator.get("programs") or []
            if operator_name == "lm_head.shards" and len(programs) != len(matching):
                _issue(
                    issues,
                    "MATMUL_PROGRAM_COUNT_MISMATCH",
                    "invalid_program_config",
                    operator_name,
                    "LM-head program count must match the number of shards",
                    programs=len(programs),
                    shards=len(matching),
                )
            for index, shape in enumerate(matching):
                if not programs:
                    break
                shape = _resolve_matmul_memory(shape, space)
                program_value = programs[min(index, len(programs) - 1)]
                program = MatmulProgramConfig.from_dict(program_value)
                path = f"{operator_name}.programs[{min(index, len(programs) - 1)}]"
                estimate = _validate_matmul(
                    path,
                    program,
                    shape,
                    device,
                    safety_factor,
                    issues,
                )
                estimates.append(estimate)
        elif operator.get("kind") == "sdpa":
            program = SDPAProgramConfig.from_dict(operator)
            estimate = _validate_sdpa(
                "attention.sdpa",
                program,
                operator,
                workload.sdpa,
                device,
                safety_factor,
                issues,
            )
            estimates.append(estimate)

    templates = space.templates
    for constraint in validate_template_selection(templates):
        _issue(
            issues,
            constraint.code,
            constraint.error_class,
            constraint.path,
            constraint.message,
        )
    if templates.get(KV_UPDATE_AXIS) == FUSED_PAGED_UPDATE:
        if workload.paged_fused_update is None:
            _issue(
                issues,
                "FUSED_CACHE_CONTEXT_MISSING",
                "shape_incompatible",
                "templates.attention.kv_update",
                "paged fused update requires representative tensor metadata",
            )
        else:
            _validate_paged_fused_update(
                workload.paged_fused_update,
                device,
                issues,
            )
    if templates.get(ROPE_AXIS) == FUSED_QK_ROPE:
        if workload.fused_qk_rope is None:
            _issue(
                issues,
                "FUSED_QK_ROPE_CONTEXT_MISSING",
                "shape_incompatible",
                "templates.attention.rope",
                "fused QK RoPE requires representative tensor metadata",
            )
        else:
            _validate_fused_qk_rope(workload.fused_qk_rope, device, issues)

    static_passed = not issues
    compile_result = None
    if static_passed and compile_request is not None:
        compile_result = run_compile_validation(compile_request)
    if not static_passed:
        status = "static_rejected"
        passed = False
        error_class = issues[0].error_class
    elif compile_result is None:
        status = "static_legal"
        passed = True
        error_class = None
    elif compile_result.passed:
        status = "compile_legal"
        passed = True
        error_class = None
    else:
        status = "compile_rejected"
        passed = False
        error_class = compile_result.error_class
    return LegalityReport(
        candidate_id=candidate_id,
        status=status,
        passed=passed,
        static_passed=static_passed,
        error_class=error_class,
        issues=tuple(issues),
        l1_estimates=tuple(estimates),
        device=device,
        safety_factor=safety_factor,
        compile_validation=compile_result,
    )


def run_compile_validation(
    request: CompileValidationRequest,
) -> CompileValidationResult:
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in request.env.items()})
    try:
        process = subprocess.Popen(
            list(request.command),
            cwd=str(request.cwd) if request.cwd is not None else None,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        elapsed = time.monotonic() - started
        error_class = (
            "api_unavailable" if isinstance(exc, FileNotFoundError) else "runtime_error"
        )
        return CompileValidationResult(
            status="launch_failed",
            passed=False,
            error_class=error_class,
            detail_code="compile_command_unavailable",
            return_code=None,
            elapsed_seconds=elapsed,
            command=request.command,
            cwd=str(request.cwd) if request.cwd is not None else None,
            stdout_tail="",
            stderr_tail=f"{type(exc).__name__}: {exc}",
        )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=request.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    stdout_tail = _tail(stdout)
    stderr_tail = _tail(stderr)
    if timed_out:
        return CompileValidationResult(
            status="timeout",
            passed=False,
            error_class="runtime_error",
            detail_code="compile_timeout",
            return_code=None,
            elapsed_seconds=elapsed,
            command=request.command,
            cwd=str(request.cwd) if request.cwd is not None else None,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
    if process.returncode == 0:
        return CompileValidationResult(
            status="passed",
            passed=True,
            error_class=None,
            detail_code=None,
            return_code=0,
            elapsed_seconds=elapsed,
            command=request.command,
            cwd=str(request.cwd) if request.cwd is not None else None,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
    error_class, detail_code = classify_validation_error("\n".join((stdout, stderr)))
    return CompileValidationResult(
        status="failed",
        passed=False,
        error_class=error_class,
        detail_code=detail_code,
        return_code=process.returncode,
        elapsed_seconds=elapsed,
        command=request.command,
        cwd=str(request.cwd) if request.cwd is not None else None,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def classify_validation_error(output: str) -> tuple[str, str]:
    text = output.lower()
    patterns = (
        (
            "l1_overflow",
            "l1_overflow",
            (
                "l1 overflow",
                "l1 buffer overflow",
                "not enough l1",
                "circular buffer allocation",
            ),
        ),
        (
            "invalid_core_grid",
            "invalid_core_grid",
            ("invalid core grid", "core grid", "core range", "worker core"),
        ),
        (
            "invalid_program_config",
            "invalid_program_config",
            ("program config", "in0_block_w", "per_core_", "out_subblock", "out_block"),
        ),
        (
            "unsupported_layout",
            "unsupported_layout",
            (
                "unsupported layout",
                "memory layout",
                "shard orientation",
                "must be sharded",
            ),
        ),
        (
            "shape_incompatible",
            "shape_incompatible",
            (
                "shape incompatible",
                "shape mismatch",
                "tile aligned",
                "must be divisible",
                "dimension",
            ),
        ),
        (
            "api_unavailable",
            "api_unavailable",
            (
                "api unavailable",
                "has no attribute",
                "undefined symbol",
                "no module named",
            ),
        ),
        (
            "runtime_error",
            "runtime_error",
            (
                "runtime error",
                "device unavailable",
                "device is busy",
                "segmentation fault",
                "core dump",
            ),
        ),
        (
            "compile_error",
            "compile_error",
            (
                "compile error",
                "compilation failed",
                "kernel build",
                "failed to compile",
            ),
        ),
    )
    for error_class, detail_code, markers in patterns:
        if any(marker in text for marker in markers):
            return error_class, detail_code
    return "compile_error", "unclassified_compile_error"


def write_legality_report(
    path: str | Path, reports: Sequence[LegalityReport]
) -> dict[str, Any]:
    completed = bool(reports)
    all_candidates_accepted = completed and all(report.passed for report in reports)
    payload = {
        "schema_version": LEGALITY_SCHEMA_VERSION,
        "status": "completed" if completed else "empty",
        "passed": completed,
        "all_candidates_processed": completed,
        "all_candidates_accepted": all_candidates_accepted,
        "candidate_count": len(reports),
        "accepted_count": sum(report.passed for report in reports),
        "rejected_count": sum(not report.passed for report in reports),
        "error_class_counts": {
            error_class: sum(report.error_class == error_class for report in reports)
            for error_class in ERROR_CLASSES
            if any(report.error_class == error_class for report in reports)
        },
        "candidates": [report.to_dict() for report in reports],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    return payload


def _validate_declared_grids(
    space: SearchSpaceConfig,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    for path, value in sorted(space.core_grids.items()):
        _validate_grid(path, CoreGrid.from_value(value), device, issues)
    for name, raw in sorted(space.operators.items()):
        grid_value = raw.get("compute_grid") if isinstance(raw, Mapping) else None
        if grid_value is not None:
            _validate_grid(
                f"operators.{name}.compute_grid",
                CoreGrid.from_value(grid_value),
                device,
                issues,
            )


def _validate_declared_memory(
    space: SearchSpaceConfig,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    for path, value in sorted(space.memory_configs.items()):
        _validate_memory_config(
            f"memory_configs.{path}",
            MemoryConfig.from_dict(value),
            device,
            issues,
        )


def _validate_grid(
    path: str,
    grid: CoreGrid,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    if grid.x > device.compute_grid.x or grid.y > device.compute_grid.y:
        _issue(
            issues,
            "GRID_OUT_OF_BOUNDS",
            "invalid_core_grid",
            path,
            "grid exceeds the physical compute grid",
            grid=grid.to_list(),
            physical_grid=device.compute_grid.to_list(),
        )
    if grid.x * grid.y > device.worker_core_count:
        _issue(
            issues,
            "GRID_WORKER_COUNT_EXCEEDED",
            "invalid_core_grid",
            path,
            "grid requests more workers than the device exposes",
            requested=grid.x * grid.y,
            available=device.worker_core_count,
        )


def _validate_memory_config(
    path: str,
    memory: MemoryConfig,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
    *,
    tensor_shape: tuple[int, int] | None = None,
) -> None:
    if memory.layout == "interleaved" and any(
        value is not None
        for value in (memory.grid, memory.shard_shape, memory.orientation)
    ):
        _issue(
            issues,
            "INTERLEAVED_HAS_SHARD_SPEC",
            "unsupported_layout",
            path,
            "interleaved memory cannot carry a shard specification",
        )
    explicit_sharded = memory.runtime_kind == "ttnn_sharded_memory_config"
    if explicit_sharded and (memory.grid is None or memory.shard_shape is None):
        _issue(
            issues,
            "SHARDED_SPEC_INCOMPLETE",
            "unsupported_layout",
            path,
            "explicit sharded memory requires both grid and shard_shape",
        )
        return
    if memory.grid is not None:
        _validate_grid(f"{path}.grid", memory.grid, device, issues)
    if memory.orientation is not None and memory.orientation.lower() not in {
        "row_major",
        "col_major",
    }:
        _issue(
            issues,
            "SHARD_ORIENTATION_UNSUPPORTED",
            "unsupported_layout",
            path,
            "shard orientation must be row_major or col_major",
            orientation=memory.orientation,
        )
    if tensor_shape is None or memory.grid is None or memory.shard_shape is None:
        return
    height, width = tensor_shape
    shard_h, shard_w = memory.shard_shape
    cores = memory.grid.x * memory.grid.y
    compatible = False
    if memory.layout == "width_sharded":
        compatible = shard_h == height and shard_w * cores == width
    elif memory.layout == "height_sharded":
        compatible = shard_w == width and shard_h * cores == height
    elif memory.layout == "block_sharded":
        compatible = shard_h * shard_w * cores == height * width
    if not compatible:
        _issue(
            issues,
            "SHARD_SHAPE_MISMATCH",
            "shape_incompatible",
            path,
            "shards do not exactly cover the padded tensor shape",
            tensor_shape=list(tensor_shape),
            shard_shape=list(memory.shard_shape),
            core_count=cores,
            layout=memory.layout,
        )


def _matching_matmul_workloads(
    operator_name: str, workloads: Mapping[str, MatmulWorkload]
) -> list[MatmulWorkload]:
    if operator_name == "lm_head.shards":
        return [
            workloads[name]
            for name in sorted(workloads)
            if name.startswith("lm_head.shards[")
        ]
    item = workloads.get(operator_name)
    return [item] if item is not None else []


def _resolve_matmul_memory(
    workload: MatmulWorkload, space: SearchSpaceConfig
) -> MatmulWorkload:
    replacements: dict[str, MemoryConfig] = {}
    for field_name, path in (
        ("input_memory", workload.input_memory_path),
        ("output_memory", workload.output_memory_path),
    ):
        if path is None:
            continue
        value = space.memory_configs.get(path)
        if value is not None:
            replacements[field_name] = MemoryConfig.from_dict(value)
    return replace(workload, **replacements) if replacements else workload


def _validate_matmul(
    path: str,
    program: MatmulProgramConfig,
    shape: MatmulWorkload,
    device: DeviceDescriptor,
    safety_factor: float,
    issues: list[LegalityIssue],
) -> L1Estimate:
    for label, value in (("M", shape.m), ("K", shape.k), ("N", shape.n)):
        if value % TILE_WIDTH != 0:
            _issue(
                issues,
                "MATMUL_TILE_ALIGNMENT",
                "shape_incompatible",
                path,
                f"matmul padded {label} dimension is not tile aligned",
                dimension=label,
                value=value,
                tile=TILE_WIDTH,
            )
    params = program.parameters
    required = {"in0_block_w", "per_core_m", "per_core_n"}
    if program.program_family in {"reuse", "reuse_multicast", "reuse_multicast_1d"}:
        required.update({"out_subblock_h", "out_subblock_w"})
    if program.program_family in {"reuse_multicast", "reuse_multicast_1d"}:
        required.update({"out_block_h", "out_block_w"})
    missing = sorted(required - set(params))
    if missing:
        _issue(
            issues,
            "MATMUL_PROGRAM_FIELDS_MISSING",
            "invalid_program_config",
            path,
            "matmul program is missing required fields",
            missing=missing,
            family=program.program_family,
        )
    values = {name: _positive_int(params.get(name)) for name in required}
    invalid = sorted(name for name, value in values.items() if value is None)
    if invalid:
        _issue(
            issues,
            "MATMUL_PROGRAM_FIELD_INVALID",
            "invalid_program_config",
            path,
            "matmul program fields must be positive integers",
            fields=invalid,
        )
    in0_block_w = values.get("in0_block_w") or 1
    per_core_m = values.get("per_core_m") or 1
    per_core_n = values.get("per_core_n") or 1
    m_tiles = math.ceil(shape.m / TILE_HEIGHT)
    k_tiles = math.ceil(shape.k / TILE_WIDTH)
    n_tiles = math.ceil(shape.n / TILE_WIDTH)
    if k_tiles % in0_block_w:
        _issue(
            issues,
            "MATMUL_K_BLOCK_DIVISIBILITY",
            "shape_incompatible",
            path,
            "K tiles must be divisible by in0_block_w",
            k_tiles=k_tiles,
            in0_block_w=in0_block_w,
        )
    if m_tiles % per_core_m:
        _issue(
            issues,
            "MATMUL_M_CORE_DIVISIBILITY",
            "shape_incompatible",
            path,
            "M tiles must be divisible by per_core_M",
            m_tiles=m_tiles,
            per_core_m=per_core_m,
        )

    if program.compute_grid is not None:
        _validate_grid(f"{path}.compute_grid", program.compute_grid, device, issues)
    elif program.program_family in {"reuse", "reuse_multicast", "reuse_multicast_1d"}:
        _issue(
            issues,
            "MATMUL_GRID_REQUIRED",
            "invalid_core_grid",
            path,
            "this matmul family requires a compute grid",
            family=program.program_family,
        )

    if program.program_family in {"dram_sharded", "batched_dram_sharded"}:
        _validate_dram_sharded_matmul(
            path,
            program,
            shape,
            m_tiles,
            k_tiles,
            per_core_m,
            issues,
        )
    else:
        _validate_reuse_matmul(
            path,
            program,
            values,
            m_tiles,
            n_tiles,
            device,
            issues,
        )
    _validate_memory_config(
        f"{path}.input_memory",
        shape.input_memory,
        device,
        issues,
        tensor_shape=(shape.m, shape.k),
    )
    _validate_memory_config(
        f"{path}.output_memory",
        shape.output_memory,
        device,
        issues,
        tensor_shape=(shape.m, _round_up(shape.n, TILE_WIDTH)),
    )
    estimate = _estimate_matmul_l1(
        path,
        shape,
        per_core_m,
        per_core_n,
        in0_block_w,
        device,
        safety_factor,
    )
    _validate_l1_estimate(estimate, device, issues)
    _validate_allowed_workers(path, program.allowed_worker_cores, device, issues)
    return estimate


def _validate_dram_sharded_matmul(
    path: str,
    program: MatmulProgramConfig,
    shape: MatmulWorkload,
    m_tiles: int,
    k_tiles: int,
    per_core_m: int,
    issues: list[LegalityIssue],
) -> None:
    expected_input_layout = (
        "height_sharded"
        if program.program_family == "batched_dram_sharded"
        else "width_sharded"
    )
    expected_weight_layout = expected_input_layout
    if shape.input_memory.layout != expected_input_layout:
        _issue(
            issues,
            "DRAM_SHARDED_INPUT_LAYOUT",
            "unsupported_layout",
            path,
            "input activation layout is incompatible with the DRAM-sharded family",
            expected=expected_input_layout,
            observed=shape.input_memory.layout,
        )
    if shape.output_memory.layout != expected_input_layout:
        _issue(
            issues,
            "DRAM_SHARDED_OUTPUT_LAYOUT",
            "unsupported_layout",
            path,
            "output layout is incompatible with the DRAM-sharded family",
            expected=expected_input_layout,
            observed=shape.output_memory.layout,
        )
    if (
        shape.weight_memory.buffer != "dram"
        or shape.weight_memory.layout != expected_weight_layout
    ):
        _issue(
            issues,
            "DRAM_SHARDED_WEIGHT_LAYOUT",
            "unsupported_layout",
            path,
            "weight must use the matching DRAM-sharded layout",
            expected={"buffer": "dram", "layout": expected_weight_layout},
            observed={
                "buffer": shape.weight_memory.buffer,
                "layout": shape.weight_memory.layout,
            },
        )
    if program.program_family == "dram_sharded" and (m_tiles != 1 or per_core_m != 1):
        _issue(
            issues,
            "DRAM_SHARDED_DECODE_M",
            "shape_incompatible",
            path,
            "decode DRAM-sharded matmul requires one M tile per core",
            m_tiles=m_tiles,
            per_core_m=per_core_m,
        )
    if program.program_family == "batched_dram_sharded" and shape.batch_count <= 1:
        _issue(
            issues,
            "BATCHED_DRAM_SHARDED_REQUIRES_BATCH",
            "shape_incompatible",
            path,
            "batched DRAM-sharded matmul requires batch_count > 1",
            batch_count=shape.batch_count,
        )
    extra = shape.weight_memory.to_dict().get("extra_fields") or {}
    for key, expected in (("k", shape.k), ("n", shape.n)):
        observed = extra.get(key)
        if observed is not None and int(observed) != expected:
            _issue(
                issues,
                "DRAM_WEIGHT_SHAPE_MISMATCH",
                "shape_incompatible",
                path,
                "DRAM-sharded weight descriptor does not match matmul shape",
                dimension=key,
                expected=expected,
                observed=observed,
            )
    if k_tiles <= 0:
        _issue(
            issues,
            "DRAM_WEIGHT_K_EMPTY",
            "shape_incompatible",
            path,
            "DRAM-sharded weight has no K tiles",
        )


def _validate_reuse_matmul(
    path: str,
    program: MatmulProgramConfig,
    values: Mapping[str, int | None],
    m_tiles: int,
    n_tiles: int,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    per_core_m = values.get("per_core_m") or 1
    per_core_n = values.get("per_core_n") or 1
    out_subblock_h = values.get("out_subblock_h") or 1
    out_subblock_w = values.get("out_subblock_w") or 1
    if per_core_m % out_subblock_h or per_core_n % out_subblock_w:
        _issue(
            issues,
            "MATMUL_SUBBLOCK_DIVISIBILITY",
            "invalid_program_config",
            path,
            "subblocks must divide per-core M and N",
            per_core_m=per_core_m,
            per_core_n=per_core_n,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
        )
    if out_subblock_h * out_subblock_w > 8:
        _issue(
            issues,
            "MATMUL_DEST_REGISTER_OVERFLOW",
            "invalid_program_config",
            path,
            "output subblock exceeds the fixed non-FP32 destination register budget",
            subblock_tiles=out_subblock_h * out_subblock_w,
            max_tiles=8,
        )
    out_block_h = values.get("out_block_h")
    out_block_w = values.get("out_block_w")
    if out_block_h is not None and (
        per_core_m % out_block_h or out_block_h % out_subblock_h
    ):
        _issue(
            issues,
            "MATMUL_OUT_BLOCK_H_INCOMPATIBLE",
            "invalid_program_config",
            path,
            "out_block_h must divide per_core_M and be divisible by out_subblock_h",
        )
    if out_block_w is not None and (
        per_core_n % out_block_w or out_block_w % out_subblock_w
    ):
        _issue(
            issues,
            "MATMUL_OUT_BLOCK_W_INCOMPATIBLE",
            "invalid_program_config",
            path,
            "out_block_w must divide per_core_N and be divisible by out_subblock_w",
        )
    required_cores = math.ceil(m_tiles / per_core_m) * math.ceil(n_tiles / per_core_n)
    available_cores = (
        program.compute_grid.x * program.compute_grid.y
        if program.compute_grid is not None
        else device.worker_core_count
    )
    if required_cores > available_cores:
        _issue(
            issues,
            "MATMUL_CORE_CAPACITY",
            "invalid_core_grid",
            path,
            "program grid cannot cover all output blocks",
            required_cores=required_cores,
            available_cores=available_cores,
        )


def _estimate_matmul_l1(
    path: str,
    shape: MatmulWorkload,
    per_core_m: int,
    per_core_n: int,
    in0_block_w: int,
    device: DeviceDescriptor,
    safety_factor: float,
) -> L1Estimate:
    input_tiles = per_core_m * in0_block_w
    weight_tiles = per_core_n * in0_block_w
    output_tiles = per_core_m * per_core_n
    input_bytes = input_tiles * _tile_bytes(shape.input_dtype)
    weight_bytes = weight_tiles * _tile_bytes(shape.weight_dtype)
    output_bytes = output_tiles * _tile_bytes(shape.output_dtype)
    intermediate_bytes = output_tiles * _tile_bytes(shape.intermediate_dtype)
    components = {
        "input_buffers": input_bytes + weight_bytes,
        "output_buffers": output_bytes,
        "double_buffered_cbs": input_bytes + weight_bytes,
        "intermediate_tiles": intermediate_bytes,
        "program_local_scratch": 64 * 1024,
    }
    cb_pages = {
        "input_0": input_tiles * 2,
        "input_1": weight_tiles * 2,
        "output": output_tiles,
        "intermediate": output_tiles,
    }
    return _l1_estimate(path, components, cb_pages, device, safety_factor)


def _validate_sdpa(
    path: str,
    program: SDPAProgramConfig,
    operator: Mapping[str, Any],
    workload: SDPAWorkload,
    device: DeviceDescriptor,
    safety_factor: float,
    issues: list[LegalityIssue],
) -> L1Estimate:
    _validate_grid(f"{path}.grid", program.grid, device, issues)
    sub_core_count = _validate_sub_core_grids(path, program, device, issues)
    for name, value in (
        ("q_chunk_size", program.q_chunk_size),
        ("k_chunk_size", program.k_chunk_size),
    ):
        if value and value % TILE_WIDTH:
            _issue(
                issues,
                "SDPA_CHUNK_TILE_ALIGNMENT",
                "shape_incompatible",
                f"{path}.{name}",
                "SDPA chunk sizes must be zero/auto or tile aligned",
                value=value,
            )
    grid_cores = sub_core_count or program.grid.x * program.grid.y
    if program.max_cores_per_head_batch > grid_cores:
        _issue(
            issues,
            "SDPA_CORE_ALLOCATION_EXCEEDED",
            "invalid_core_grid",
            path,
            "max_cores_per_head_batch exceeds the SDPA grid",
            requested=program.max_cores_per_head_batch,
            grid_cores=grid_cores,
        )
    if workload.num_heads % workload.num_kv_heads:
        _issue(
            issues,
            "SDPA_GQA_HEAD_RATIO",
            "shape_incompatible",
            path,
            "query head count must be divisible by KV head count",
        )
    if workload.head_dim % TILE_WIDTH:
        _issue(
            issues,
            "SDPA_HEAD_DIM_ALIGNMENT",
            "shape_incompatible",
            path,
            "SDPA head_dim must be tile aligned",
            head_dim=workload.head_dim,
        )
    if workload.cache_len % workload.page_block_size:
        _issue(
            issues,
            "SDPA_PAGE_BLOCK_DIVISIBILITY",
            "shape_incompatible",
            path,
            "cache length must be divisible by page block size",
            cache_len=workload.cache_len,
            page_block_size=workload.page_block_size,
        )
    for key in ("kernel_output_memory", "output_memory"):
        value = operator.get(key)
        if value is not None:
            memory = MemoryConfig.from_dict(value)
            _validate_memory_config(
                f"{path}.{key}",
                memory,
                device,
                issues,
                tensor_shape=(
                    workload.batch_size * workload.num_heads,
                    workload.head_dim,
                ),
            )
            if (
                key == "kernel_output_memory"
                and workload.num_heads != workload.num_kv_heads
                and memory.layout != "interleaved"
            ):
                _issue(
                    issues,
                    "SDPA_GQA_SHARDED_OUTPUT_UNSUPPORTED",
                    "unsupported_layout",
                    f"{path}.{key}",
                    "TTNN SDPA decode does not support sharded output for GQA",
                    num_heads=workload.num_heads,
                    num_kv_heads=workload.num_kv_heads,
                    output_layout=memory.layout,
                )
    q_chunk = program.q_chunk_size or TILE_HEIGHT
    k_chunk = program.k_chunk_size or min(workload.cache_len, 128)
    head_tiles = math.ceil(workload.head_dim / TILE_WIDTH)
    q_tiles = math.ceil(q_chunk / TILE_HEIGHT) * head_tiles
    kv_tiles = 2 * math.ceil(k_chunk / TILE_HEIGHT) * head_tiles
    score_tiles = math.ceil(q_chunk / TILE_HEIGHT) * math.ceil(k_chunk / TILE_WIDTH)
    components = {
        "input_buffers": q_tiles * _tile_bytes(workload.query_dtype)
        + kv_tiles * _tile_bytes(workload.cache_dtype),
        "output_buffers": q_tiles * _tile_bytes(workload.output_dtype),
        "double_buffered_cbs": q_tiles * _tile_bytes(workload.query_dtype)
        + kv_tiles * _tile_bytes(workload.cache_dtype),
        "intermediate_tiles": score_tiles * _tile_bytes("bf16"),
        "program_local_scratch": 96 * 1024,
    }
    cb_pages = {
        "query": q_tiles * 2,
        "key_value": kv_tiles * 2,
        "output": q_tiles,
        "attention_scores": score_tiles,
    }
    estimate = _l1_estimate(path, components, cb_pages, device, safety_factor)
    _validate_l1_estimate(estimate, device, issues)
    return estimate


def _validate_paged_fused_update(
    workload: PagedFusedUpdateWorkload,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    path = "templates.attention.kv_update"
    inputs = (workload.key_input, workload.value_input)
    caches = (workload.key_cache, workload.value_cache)
    for index, (tensor, cache) in enumerate(zip(inputs, caches)):
        tensor_path = f"{path}.inputs[{index}]"
        _validate_tensor_cores(tensor, tensor_path, device, issues)
        if tensor.padded_shape[0] != 1:
            _issue(
                issues,
                "FUSED_CACHE_INPUT_DIM0",
                "shape_incompatible",
                tensor_path,
                "paged fused update input dim 0 must be one",
            )
        if tensor.dtype.lower() not in {"bf16", "bfloat16", "fp32", "float32"}:
            _issue(
                issues,
                "FUSED_CACHE_INPUT_DTYPE",
                "unsupported_layout",
                tensor_path,
                "paged fused update inputs must be BF16 or FP32",
            )
        if tensor.memory.layout == "width_sharded" or not tensor.cores:
            _issue(
                issues,
                "FUSED_CACHE_INPUT_SHARDING",
                "unsupported_layout",
                tensor_path,
                "paged fused update inputs must be non-width sharded",
            )
        if (tensor.memory.orientation or "").lower() != "row_major":
            _issue(
                issues,
                "FUSED_CACHE_ORIENTATION",
                "unsupported_layout",
                tensor_path,
                "paged fused update requires row-major shard orientation",
            )
        if (
            tensor.memory.shard_shape is None
            or tensor.memory.shard_shape[1] != tensor.width
        ):
            _issue(
                issues,
                "FUSED_CACHE_SHARD_WIDTH",
                "shape_incompatible",
                tensor_path,
                "input shard width must equal the padded tensor width",
            )
        if cache.layout != "tile" or cache.memory.layout != "interleaved":
            _issue(
                issues,
                "FUSED_CACHE_CACHE_LAYOUT",
                "unsupported_layout",
                f"{path}.caches[{index}]",
                "cache tensors must be tiled and interleaved",
            )
        if cache.dtype.lower() not in {
            "fp32",
            "float32",
            "bf16",
            "bfloat16",
            "bfp8",
            "bfloat8_b",
            "bfp4",
            "bfloat4_b",
        }:
            _issue(
                issues,
                "FUSED_CACHE_CACHE_DTYPE",
                "unsupported_layout",
                f"{path}.caches[{index}]",
                "cache dtype is not supported by paged fused update",
            )
        if tensor.width != cache.width:
            _issue(
                issues,
                "FUSED_CACHE_HEAD_DIM_MISMATCH",
                "shape_incompatible",
                tensor_path,
                "input and cache head dimensions must match",
            )
        if tensor.device != cache.device:
            _issue(
                issues,
                "FUSED_CACHE_DEVICE_MISMATCH",
                "runtime_error",
                tensor_path,
                "input and cache tensors must be on the same device",
            )
    if inputs[0].layout != inputs[1].layout:
        _issue(
            issues,
            "FUSED_CACHE_LAYOUT_PAIR",
            "unsupported_layout",
            path,
            "K and V inputs must both be tiled or both row-major",
        )
    if inputs[0].padded_shape != inputs[1].padded_shape:
        _issue(
            issues,
            "FUSED_CACHE_INPUT_SHAPE_PAIR",
            "shape_incompatible",
            path,
            "K and V input padded shapes must match",
        )
    if caches[0].padded_shape != caches[1].padded_shape:
        _issue(
            issues,
            "FUSED_CACHE_CACHE_SHAPE_PAIR",
            "shape_incompatible",
            path,
            "K and V cache padded shapes must match",
        )
    if len({tensor.device for tensor in (*inputs, *caches)}) != 1:
        _issue(
            issues,
            "FUSED_CACHE_PAIR_DEVICE",
            "runtime_error",
            path,
            "all K/V inputs and caches must use the same device",
        )
    if len(inputs[0].cores) != len(inputs[1].cores):
        _issue(
            issues,
            "FUSED_CACHE_CORE_COUNT",
            "invalid_core_grid",
            path,
            "K and V inputs must use the same number of cores",
        )
    if set(inputs[0].cores) & set(inputs[1].cores):
        _issue(
            issues,
            "FUSED_CACHE_CORE_OVERLAP",
            "invalid_core_grid",
            path,
            "K and V input core ranges must not overlap",
        )
    batch = inputs[0].padded_shape[1]
    page_table = workload.page_table
    if page_table.layout != "row_major" or page_table.padded_shape[0] != batch:
        _issue(
            issues,
            "FUSED_CACHE_PAGE_TABLE",
            "shape_incompatible",
            f"{path}.page_table",
            "interleaved page table must be row-major with one row per user",
            batch=batch,
        )
    if page_table.padded_shape[-1] > caches[0].padded_shape[0]:
        _issue(
            issues,
            "FUSED_CACHE_PAGE_TABLE_CAPACITY",
            "shape_incompatible",
            f"{path}.page_table",
            "page table references more blocks per user than the cache owns",
            page_table_blocks=page_table.padded_shape[-1],
            cache_blocks=caches[0].padded_shape[0],
        )
    if page_table.dtype.lower() not in {"int32", "uint16"}:
        _issue(
            issues,
            "FUSED_CACHE_PAGE_TABLE_DTYPE",
            "unsupported_layout",
            f"{path}.page_table",
            "page table must be INT32 or sharded UINT16",
        )
    update = workload.update_indices
    if update.layout != "row_major" or update.dtype.lower() != "int32":
        _issue(
            issues,
            "FUSED_CACHE_UPDATE_INDICES",
            "unsupported_layout",
            f"{path}.update_indices",
            "update indices must be row-major INT32",
        )
    if math.prod(update.logical_shape) != batch:
        _issue(
            issues,
            "FUSED_CACHE_UPDATE_COUNT",
            "shape_incompatible",
            f"{path}.update_indices",
            "update index count must equal batch size",
            expected=batch,
            observed=math.prod(update.logical_shape),
        )


def _validate_fused_qk_rope(
    workload: FusedQKRoPEWorkload,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    path = "templates.attention.rope"
    tensors = (
        workload.q,
        workload.k,
        workload.cos,
        workload.sin,
        workload.transformation,
    )
    for tensor in tensors:
        _validate_tensor_cores(tensor, f"{path}.{tensor.name}", device, issues)
        if tensor.dtype.lower() not in {"bf16", "bfloat16"}:
            _issue(
                issues,
                "FUSED_QK_ROPE_DTYPE",
                "unsupported_layout",
                f"{path}.{tensor.name}",
                "fused QK RoPE requires BF16 tensors",
            )
        if tensor.memory.layout != "height_sharded":
            _issue(
                issues,
                "FUSED_QK_ROPE_SHARDING",
                "unsupported_layout",
                f"{path}.{tensor.name}",
                "fused QK RoPE requires height-sharded tensors",
            )
        if tensor.device != workload.q.device:
            _issue(
                issues,
                "FUSED_QK_ROPE_DEVICE",
                "runtime_error",
                f"{path}.{tensor.name}",
                "all fused QK RoPE tensors must use the same device",
            )
    q, k = workload.q, workload.k
    if not (q.layout == k.layout == workload.cos.layout == workload.sin.layout):
        _issue(
            issues,
            "FUSED_QK_ROPE_LAYOUT_PAIR",
            "unsupported_layout",
            path,
            "Q, K, cos, and sin must use the same tensor layout",
        )
    if q.logical_shape[0] != 1 or k.logical_shape[0] != 1:
        _issue(
            issues,
            "FUSED_QK_ROPE_DECODE_ONLY",
            "shape_incompatible",
            path,
            "fused QK RoPE only supports decode sequence length one",
        )
    if q.width != k.width or q.width % TILE_WIDTH:
        _issue(
            issues,
            "FUSED_QK_ROPE_HEAD_DIM",
            "shape_incompatible",
            path,
            "Q and K head dimensions must match and be tile aligned",
        )
    q_batch = q.logical_shape[1]
    k_batch = k.logical_shape[1]
    if q_batch != k_batch or q_batch > 32:
        _issue(
            issues,
            "FUSED_QK_ROPE_BATCH",
            "shape_incompatible",
            path,
            "Q and K batches must match and not exceed 32",
            q_batch=q_batch,
            k_batch=k_batch,
        )
    if len(q.cores) + len(k.cores) > 64:
        _issue(
            issues,
            "FUSED_QK_ROPE_CORE_LIMIT",
            "invalid_core_grid",
            path,
            "Q and K together may use at most 64 cores",
        )
    if set(q.cores) & set(k.cores):
        _issue(
            issues,
            "FUSED_QK_ROPE_CORE_OVERLAP",
            "invalid_core_grid",
            path,
            "Q and K core ranges must not overlap",
        )
    if workload.cos.logical_shape != workload.sin.logical_shape:
        _issue(
            issues,
            "FUSED_QK_ROPE_COS_SIN_SHAPE",
            "shape_incompatible",
            path,
            "cos and sin shapes must match",
        )
    elif workload.cos.logical_shape[1] != q_batch + k_batch:
        _issue(
            issues,
            "FUSED_QK_ROPE_COS_SIN_BATCH",
            "shape_incompatible",
            path,
            "cos and sin batch must equal Q batch plus K batch",
        )
    transform = workload.transformation
    if transform.layout != "tile" or transform.memory.shard_shape != (32, 32):
        _issue(
            issues,
            "FUSED_QK_ROPE_TRANSFORM_SHARD",
            "unsupported_layout",
            path,
            "transformation matrix must be tiled with one 32x32 tile per shard",
        )
    if len(transform.cores) < len(q.cores) + len(k.cores):
        _issue(
            issues,
            "FUSED_QK_ROPE_TRANSFORM_CORES",
            "invalid_core_grid",
            path,
            "transformation matrix must cover all Q and K cores",
        )
    if (
        workload.q_output_memory.layout != q.memory.layout
        or workload.k_output_memory.layout != k.memory.layout
    ):
        _issue(
            issues,
            "FUSED_QK_ROPE_OUTPUT_LAYOUT",
            "unsupported_layout",
            path,
            "Q/K output memory layouts must match their respective inputs",
        )


def _validate_tensor_cores(
    tensor: TensorSpec,
    path: str,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    _validate_memory_config(
        f"{path}.memory",
        tensor.memory,
        device,
        issues,
        tensor_shape=(tensor.flattened_height, tensor.width),
    )
    if (
        tensor.cores
        and tensor.memory.grid is not None
        and len(tensor.cores) != tensor.memory.grid.x * tensor.memory.grid.y
    ):
        _issue(
            issues,
            "TENSOR_SHARD_CORE_COUNT",
            "invalid_core_grid",
            path,
            "explicit tensor cores do not match the memory grid core count",
            explicit_core_count=len(tensor.cores),
            memory_grid_core_count=tensor.memory.grid.x * tensor.memory.grid.y,
        )
    for x, y in tensor.cores:
        if x < 0 or y < 0 or x >= device.compute_grid.x or y >= device.compute_grid.y:
            _issue(
                issues,
                "TENSOR_CORE_OUT_OF_BOUNDS",
                "invalid_core_grid",
                path,
                "tensor shard core lies outside the physical grid",
                core=[x, y],
                physical_grid=device.compute_grid.to_list(),
            )


def _validate_sub_core_grids(
    path: str,
    program: SDPAProgramConfig,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> int:
    value = program.sub_core_grids
    if value is None:
        return 0
    if not isinstance(value, list) or not value:
        _issue(
            issues,
            "SDPA_SUB_CORE_FORMAT",
            "invalid_core_grid",
            f"{path}.sub_core_grids",
            "SDPA sub_core_grids must be a non-empty list of rectangles",
        )
        return 0
    selected: set[tuple[int, int]] = set()
    for index, rectangle in enumerate(value):
        coordinates = _rectangle_coordinates(rectangle)
        if coordinates is None:
            _issue(
                issues,
                "SDPA_SUB_CORE_FORMAT",
                "invalid_core_grid",
                f"{path}.sub_core_grids[{index}]",
                "sub-core rectangle must be [x0, y0, x1, y1]",
            )
            continue
        if selected & coordinates:
            _issue(
                issues,
                "SDPA_SUB_CORE_OVERLAP",
                "invalid_core_grid",
                f"{path}.sub_core_grids[{index}]",
                "SDPA sub-core rectangles must not overlap",
            )
        selected.update(coordinates)
    for x, y in selected:
        if x >= device.compute_grid.x or y >= device.compute_grid.y:
            _issue(
                issues,
                "SDPA_SUB_CORE_OUT_OF_BOUNDS",
                "invalid_core_grid",
                f"{path}.sub_core_grids",
                "SDPA sub-core lies outside the physical device grid",
                core=[x, y],
                device_grid=device.compute_grid.to_list(),
            )
    expected = program.grid.x * program.grid.y
    if len(selected) != expected:
        _issue(
            issues,
            "SDPA_SUB_CORE_COUNT_MISMATCH",
            "invalid_core_grid",
            f"{path}.sub_core_grids",
            "SDPA sub-core set must contain exactly the logical grid core count",
            selected_core_count=len(selected),
            expected_core_count=expected,
            program_grid=program.grid.to_list(),
        )
    return len(selected)


def _validate_allowed_workers(
    path: str,
    value: Any,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        _issue(
            issues,
            "ALLOWED_WORKERS_FORMAT",
            "invalid_core_grid",
            path,
            "allowed_worker_cores must be a list of [x, y] coordinates",
        )
        return
    coordinates: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            _issue(
                issues,
                "ALLOWED_WORKERS_FORMAT",
                "invalid_core_grid",
                path,
                "allowed_worker_cores must be a list of [x, y] coordinates",
            )
            return
        coordinates.append((int(item[0]), int(item[1])))
    if len(set(coordinates)) != len(coordinates):
        _issue(
            issues,
            "ALLOWED_WORKERS_DUPLICATE",
            "invalid_core_grid",
            path,
            "allowed_worker_cores contains duplicate coordinates",
        )
    for coordinate in coordinates:
        if not (
            0 <= coordinate[0] < device.compute_grid.x
            and 0 <= coordinate[1] < device.compute_grid.y
        ):
            _issue(
                issues,
                "ALLOWED_WORKER_OUT_OF_BOUNDS",
                "invalid_core_grid",
                path,
                "allowed worker lies outside the physical grid",
                core=list(coordinate),
            )


def _l1_estimate(
    path: str,
    components: Mapping[str, int],
    cb_pages: Mapping[str, int],
    device: DeviceDescriptor,
    safety_factor: float,
) -> L1Estimate:
    available = device.available_l1_bytes_per_core
    return L1Estimate(
        path=path,
        components=dict(components),
        cb_pages=dict(cb_pages),
        total_bytes=sum(components.values()),
        available_bytes=available,
        safety_factor=safety_factor,
        limit_bytes=math.floor(available * safety_factor),
    )


def _validate_l1_estimate(
    estimate: L1Estimate,
    device: DeviceDescriptor,
    issues: list[LegalityIssue],
) -> None:
    if estimate.total_bytes > estimate.limit_bytes:
        _issue(
            issues,
            "L1_SAFETY_LIMIT_EXCEEDED",
            "l1_overflow",
            estimate.path,
            "estimated per-core L1 footprint exceeds the safety limit",
            estimated_l1=estimate.total_bytes,
            limit=estimate.limit_bytes,
            available_l1=estimate.available_bytes,
            safety_factor=estimate.safety_factor,
        )
    oversized = {
        name: count
        for name, count in estimate.cb_pages.items()
        if count > device.max_cb_pages
    }
    if oversized:
        _issue(
            issues,
            "CB_PAGE_COUNT_EXCEEDED",
            "l1_overflow",
            estimate.path,
            "one or more circular buffers exceed the conservative page limit",
            cb_pages=oversized,
            max_cb_pages=device.max_cb_pages,
        )


def _issue(
    issues: list[LegalityIssue],
    code: str,
    error_class: str,
    path: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(
        LegalityIssue(
            code=code,
            error_class=error_class,
            path=path,
            message=message,
            details=details,
        )
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _rectangle_coordinates(value: Any) -> set[tuple[int, int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if min(x0, y0) < 0 or x1 < x0 or y1 < y0:
        return None
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}


def _tile_bytes(dtype: str) -> int:
    try:
        return _TILE_BYTES[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported tile dtype for L1 estimate: {dtype}") from exc


def _round_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _tail(value: str, limit: int = 16_384) -> str:
    return value[-limit:]
