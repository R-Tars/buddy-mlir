from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .schema import AUTOTUNE_SCHEMA_VERSION, canonical_json
from .templates import (
    ACTIVATION_AXIS,
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    GATE_UP_AXIS,
    KV_UPDATE_AXIS,
    MUL_FUSED_SILU,
    PACKED_GATE_UP,
    ROPE_AXIS,
    SEPARATE_GATE_UP,
    SEPARATE_PAGED_UPDATE,
    SEPARATE_QK_ROPE,
    TemplateSelectionError,
    apply_template_selection,
    normalize_template_selection,
)

OFFICIAL_LINEAR_OUTPUTS = "official_l1_sharded"
LM_HEAD_DRAM_CONCAT = "lm_head_dram_concat"
OFFICIAL_PROGRAM_CONFIG = "official"
SDPA_GRID_8X4_PROGRAM_CONFIG = "sdpa_grid_8x4"

_NAMED_MEMORY = {
    "DRAM_MEMORY_CONFIG": ("dram", "interleaved"),
    "L1_MEMORY_CONFIG": ("l1", "interleaved"),
    "L1_WIDTH_SHARDED_MEMORY_CONFIG": ("l1", "width_sharded"),
    "L1_HEIGHT_SHARDED_MEMORY_CONFIG": ("l1", "height_sharded"),
    "L1_BLOCK_SHARDED_MEMORY_CONFIG": ("l1", "block_sharded"),
}
_MATMUL_FAMILIES = {
    "ttnn_matmul_multicore_reuse_program_config": "reuse",
    "ttnn_matmul_multicore_reuse_mcast_program_config": "reuse_multicast",
    "ttnn_matmul_multicore_reuse_mcast_1d_program_config": ("reuse_multicast_1d"),
    "ttnn_matmul_dram_sharded_program_config": "dram_sharded",
    "ttnn_matmul_multi_core_reuse_multi_cast_dram_sharded_program_config": (
        "batched_dram_sharded"
    ),
}
_RUNTIME_PARAMETER_NAMES = {
    "per_core_m": "per_core_M",
    "per_core_n": "per_core_N",
}
_SCHEMA_PARAMETER_NAMES = {
    value: key for key, value in _RUNTIME_PARAMETER_NAMES.items()
}
_MATMUL_PATHS = {
    "attention.qkv": {
        "section": "attention",
        "program": "qkv_program_config",
        "output_memory": "qkv_output_memory_config",
    },
    "attention.o_proj": {
        "section": "attention",
        "program": "o_proj_program_config",
        "output_memory": "o_proj_output_memory_config",
    },
    "mlp.gate": {
        "section": "mlp",
        "program": "gate_program_config",
        "output_memory": "gate_output_memory_config",
    },
    "mlp.up": {
        "section": "mlp",
        "program": "up_program_config",
        "output_memory": "up_output_memory_config",
    },
    "mlp.down": {
        "section": "mlp",
        "program": "down_program_config",
        "output_memory": "down_output_memory_config",
    },
    "lm_head.shards": {
        "section": "lm_head",
        "programs": "program_configs",
        "output_memory": "shard_output_memory_config",
    },
}
_TYPED_PROGRAM_PATHS = {
    "attention.qkv_program_config",
    "attention.o_proj_program_config",
    "attention.sdpa_program_config",
    "mlp.gate_program_config",
    "mlp.up_program_config",
    "mlp.down_program_config",
}
_EDGE_PATHS = {
    "sdpa_to_concat_heads": (
        "attention.sdpa_output_memory_config",
        "attention.concat_heads_input_memory_config",
    ),
    "final_norm_to_lm_head": (
        "rms_norm.final.output_memory_config",
        "lm_head.input_memory_config",
    ),
    "lm_head_shards_to_concat": (
        "lm_head.shard_output_memory_config",
        "lm_head.concat_memory_config",
    ),
}


class SpaceSchemaError(ValueError):
    """Raised when a structured search-space document is malformed."""


@dataclass(frozen=True)
class CoreGrid:
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x <= 0 or self.y <= 0:
            raise SpaceSchemaError("core grid dimensions must be positive")

    @classmethod
    def from_value(cls, value: Any) -> "CoreGrid":
        if isinstance(value, Mapping):
            return cls(x=int(value["x"]), y=int(value["y"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(x=int(value[0]), y=int(value[1]))
        raise SpaceSchemaError(f"invalid core grid: {value!r}")

    def to_list(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True)
class MemoryConfig:
    buffer: str
    layout: str
    runtime_kind: str
    runtime_name: str | None = None
    grid: CoreGrid | None = None
    shard_shape: tuple[int, int] | None = None
    orientation: str | None = None
    _extra_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if self.buffer not in {"dram", "l1"}:
            raise SpaceSchemaError(f"unsupported memory buffer: {self.buffer}")
        if self.layout not in {
            "interleaved",
            "width_sharded",
            "height_sharded",
            "block_sharded",
        }:
            raise SpaceSchemaError(f"unsupported memory layout: {self.layout}")
        if self.shard_shape is not None and any(
            value <= 0 for value in self.shard_shape
        ):
            raise SpaceSchemaError("shard shape dimensions must be positive")
        _decode_object(self._extra_json, "memory extra_fields")
        if self.runtime_name in _NAMED_MEMORY:
            expected = _NAMED_MEMORY[self.runtime_name]
            if (self.buffer, self.layout) != expected:
                raise SpaceSchemaError(
                    f"{self.runtime_name} implies {expected}, observed "
                    f"{(self.buffer, self.layout)}"
                )

    @classmethod
    def from_runtime_descriptor(cls, value: Mapping[str, Any]) -> "MemoryConfig":
        descriptor = copy.deepcopy(dict(value))
        runtime_kind = str(descriptor.pop("kind", ""))
        runtime_name = descriptor.pop("name", None)
        grid_value = descriptor.pop("core_grid", None)
        shard_value = descriptor.pop("shard_shape", None)
        orientation = descriptor.pop("orientation", None)
        if runtime_name in _NAMED_MEMORY:
            buffer, layout = _NAMED_MEMORY[str(runtime_name)]
        elif runtime_kind == "ttnn_sharded_memory_config":
            buffer = str(descriptor.pop("buffer", "l1")).lower()
            strategy = str(descriptor.pop("strategy", ""))
            layout = f"{strategy}_sharded"
        elif runtime_kind == "ttnn_dram_sharded_memory_config":
            buffer, layout = "dram", "width_sharded"
        else:
            raise SpaceSchemaError(f"unsupported runtime memory descriptor: {value!r}")
        shard_shape = None
        if shard_value is not None:
            if not isinstance(shard_value, (list, tuple)) or len(shard_value) != 2:
                raise SpaceSchemaError(f"invalid shard shape: {shard_value!r}")
            shard_shape = (int(shard_value[0]), int(shard_value[1]))
        return cls(
            buffer=buffer,
            layout=layout,
            runtime_kind=runtime_kind,
            runtime_name=str(runtime_name) if runtime_name is not None else None,
            grid=CoreGrid.from_value(grid_value) if grid_value is not None else None,
            shard_shape=shard_shape,
            orientation=str(orientation) if orientation is not None else None,
            _extra_json=canonical_json(descriptor),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryConfig":
        shard_value = value.get("shard_shape")
        shard_shape = (
            (int(shard_value[0]), int(shard_value[1]))
            if shard_value is not None
            else None
        )
        grid_value = value.get("grid")
        return cls(
            buffer=str(value["buffer"]),
            layout=str(value["layout"]),
            runtime_kind=str(value["runtime_kind"]),
            runtime_name=(
                str(value["runtime_name"])
                if value.get("runtime_name") is not None
                else None
            ),
            grid=CoreGrid.from_value(grid_value) if grid_value is not None else None,
            shard_shape=shard_shape,
            orientation=(
                str(value["orientation"])
                if value.get("orientation") is not None
                else None
            ),
            _extra_json=canonical_json(value.get("extra_fields") or {}),
        )

    @classmethod
    def named(cls, name: str) -> "MemoryConfig":
        try:
            buffer, layout = _NAMED_MEMORY[name]
        except KeyError as exc:
            raise SpaceSchemaError(f"unsupported named memory config: {name}") from exc
        return cls(
            buffer=buffer,
            layout=layout,
            runtime_kind="ttnn_memory_config",
            runtime_name=name,
        )

    def to_runtime_descriptor(self) -> dict[str, Any]:
        result = _decode_object(self._extra_json, "memory extra_fields")
        result["kind"] = self.runtime_kind
        if self.runtime_name is not None:
            result["name"] = self.runtime_name
        if self.grid is not None:
            result["core_grid"] = self.grid.to_list()
        if self.shard_shape is not None:
            result["shard_shape"] = list(self.shard_shape)
        if self.orientation is not None:
            result["orientation"] = self.orientation
        if self.runtime_kind == "ttnn_sharded_memory_config":
            result["strategy"] = self.layout.removesuffix("_sharded")
            if self.buffer != "l1":
                result["buffer"] = self.buffer
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "layout": self.layout,
            "runtime_kind": self.runtime_kind,
            "runtime_name": self.runtime_name,
            "grid": self.grid.to_list() if self.grid is not None else None,
            "shard_shape": (
                list(self.shard_shape) if self.shard_shape is not None else None
            ),
            "orientation": self.orientation,
            "extra_fields": _decode_object(self._extra_json, "memory extra_fields"),
        }


@dataclass(frozen=True)
class MatmulProgramConfig:
    program_family: str
    runtime_kind: str
    compute_grid: CoreGrid | None = None
    _allowed_worker_cores_json: str = field(default="null", repr=False)
    _parameters_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if self.program_family not in set(_MATMUL_FAMILIES.values()):
            raise SpaceSchemaError(
                f"unsupported matmul program family: {self.program_family}"
            )
        _decode_json(self._allowed_worker_cores_json, "allowed_worker_cores")
        _decode_object(self._parameters_json, "matmul parameters")

    @classmethod
    def from_runtime_descriptor(cls, value: Mapping[str, Any]) -> "MatmulProgramConfig":
        descriptor = copy.deepcopy(dict(value))
        runtime_kind = str(descriptor.pop("kind", ""))
        try:
            family = _MATMUL_FAMILIES[runtime_kind]
        except KeyError as exc:
            raise SpaceSchemaError(
                f"unsupported matmul descriptor kind: {runtime_kind}"
            ) from exc
        grid_value = descriptor.pop("core_grid", None)
        workers = descriptor.pop("allowed_worker_cores", None)
        parameters = {
            _SCHEMA_PARAMETER_NAMES.get(key, key): child
            for key, child in descriptor.items()
        }
        return cls(
            program_family=family,
            runtime_kind=runtime_kind,
            compute_grid=(
                CoreGrid.from_value(grid_value) if grid_value is not None else None
            ),
            _allowed_worker_cores_json=canonical_json(workers),
            _parameters_json=canonical_json(parameters),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatmulProgramConfig":
        grid_value = value.get("compute_grid")
        reserved = {
            "kind",
            "program_family",
            "runtime_kind",
            "compute_grid",
            "allowed_worker_cores",
        }
        parameters = {key: child for key, child in value.items() if key not in reserved}
        return cls(
            program_family=str(value["program_family"]),
            runtime_kind=str(value["runtime_kind"]),
            compute_grid=(
                CoreGrid.from_value(grid_value) if grid_value is not None else None
            ),
            _allowed_worker_cores_json=canonical_json(
                value.get("allowed_worker_cores")
            ),
            _parameters_json=canonical_json(parameters),
        )

    @property
    def allowed_worker_cores(self) -> Any:
        return _decode_json(self._allowed_worker_cores_json, "allowed_worker_cores")

    @property
    def parameters(self) -> dict[str, Any]:
        return _decode_object(self._parameters_json, "matmul parameters")

    def to_runtime_descriptor(self) -> dict[str, Any]:
        result = {
            _RUNTIME_PARAMETER_NAMES.get(key, key): child
            for key, child in self.parameters.items()
        }
        result["kind"] = self.runtime_kind
        if self.compute_grid is not None:
            result["core_grid"] = self.compute_grid.to_list()
        if self.allowed_worker_cores is not None:
            result["allowed_worker_cores"] = self.allowed_worker_cores
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "matmul",
            "program_family": self.program_family,
            "runtime_kind": self.runtime_kind,
            "compute_grid": (
                self.compute_grid.to_list() if self.compute_grid is not None else None
            ),
            "allowed_worker_cores": self.allowed_worker_cores,
            **self.parameters,
        }


@dataclass(frozen=True)
class SDPAProgramConfig:
    grid: CoreGrid
    q_chunk_size: int
    k_chunk_size: int
    max_cores_per_head_batch: int
    exp_approx_mode: bool
    runtime_kind: str = "ttnn_sdpa_program_config"
    _sub_core_grids_json: str = field(default="null", repr=False)
    _extra_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if self.q_chunk_size < 0 or self.k_chunk_size < 0:
            raise SpaceSchemaError("SDPA chunk sizes must be non-negative")
        if self.max_cores_per_head_batch <= 0:
            raise SpaceSchemaError("SDPA max_cores_per_head_batch must be positive")
        _decode_json(self._sub_core_grids_json, "SDPA sub_core_grids")
        _decode_object(self._extra_json, "SDPA extra_fields")

    @classmethod
    def from_runtime_descriptor(cls, value: Mapping[str, Any]) -> "SDPAProgramConfig":
        descriptor = copy.deepcopy(dict(value))
        runtime_kind = str(descriptor.pop("kind", ""))
        if runtime_kind != "ttnn_sdpa_program_config":
            raise SpaceSchemaError(f"unsupported SDPA descriptor kind: {runtime_kind}")
        grid_value = descriptor.pop("core_grid")
        sub_core_grids = descriptor.pop("sub_core_grids", None)
        return cls(
            grid=CoreGrid.from_value(grid_value),
            q_chunk_size=int(descriptor.pop("q_chunk_size", 0)),
            k_chunk_size=int(descriptor.pop("k_chunk_size", 0)),
            max_cores_per_head_batch=int(descriptor.pop("max_cores_per_head_batch")),
            exp_approx_mode=bool(descriptor.pop("exp_approx_mode")),
            runtime_kind=runtime_kind,
            _sub_core_grids_json=canonical_json(sub_core_grids),
            _extra_json=canonical_json(descriptor),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SDPAProgramConfig":
        return cls(
            grid=CoreGrid.from_value(value["grid"]),
            q_chunk_size=int(value["q_chunk_size"]),
            k_chunk_size=int(value["k_chunk_size"]),
            max_cores_per_head_batch=int(value["max_cores_per_head_batch"]),
            exp_approx_mode=bool(value["exp_approx_mode"]),
            runtime_kind=str(value.get("runtime_kind", "ttnn_sdpa_program_config")),
            _sub_core_grids_json=canonical_json(value.get("sub_core_grids")),
            _extra_json=canonical_json(value.get("extra_fields") or {}),
        )

    @property
    def sub_core_grids(self) -> Any:
        return _decode_json(self._sub_core_grids_json, "SDPA sub_core_grids")

    def to_runtime_descriptor(self) -> dict[str, Any]:
        result = _decode_object(self._extra_json, "SDPA extra_fields")
        result.update(
            {
                "kind": self.runtime_kind,
                "core_grid": self.grid.to_list(),
                "q_chunk_size": self.q_chunk_size,
                "k_chunk_size": self.k_chunk_size,
                "exp_approx_mode": self.exp_approx_mode,
                "max_cores_per_head_batch": self.max_cores_per_head_batch,
            }
        )
        if self.sub_core_grids is not None:
            result["sub_core_grids"] = self.sub_core_grids
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sdpa",
            "runtime_kind": self.runtime_kind,
            "grid": self.grid.to_list(),
            "sub_core_grids": self.sub_core_grids,
            "q_chunk_size": self.q_chunk_size,
            "k_chunk_size": self.k_chunk_size,
            "max_cores_per_head_batch": self.max_cores_per_head_batch,
            "exp_approx_mode": self.exp_approx_mode,
            "extra_fields": _decode_object(self._extra_json, "SDPA extra_fields"),
        }


@dataclass(frozen=True)
class EdgeConfig:
    producer_path: str
    consumer_path: str
    producer_output_memory: MemoryConfig
    consumer_input_memory: MemoryConfig
    conversion: str

    def __post_init__(self) -> None:
        if self.conversion not in {
            "none",
            "explicit",
            "producer_compatible",
            "consumer_accepts",
        }:
            raise SpaceSchemaError(f"unsupported edge conversion: {self.conversion}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EdgeConfig":
        return cls(
            producer_path=str(value["producer_path"]),
            consumer_path=str(value["consumer_path"]),
            producer_output_memory=MemoryConfig.from_dict(
                value["producer_output_memory"]
            ),
            consumer_input_memory=MemoryConfig.from_dict(
                value["consumer_input_memory"]
            ),
            conversion=str(value["conversion"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_path": self.producer_path,
            "consumer_path": self.consumer_path,
            "producer_output_memory": self.producer_output_memory.to_dict(),
            "consumer_input_memory": self.consumer_input_memory.to_dict(),
            "conversion": self.conversion,
        }


@dataclass(frozen=True)
class SearchSpaceConfig:
    _templates_json: str = field(repr=False)
    _operators_json: str = field(repr=False)
    _memory_configs_json: str = field(repr=False)
    _core_grids_json: str = field(repr=False)
    _extra_program_configs_json: str = field(repr=False)
    _edges_json: str = field(repr=False)
    schema_version: int = AUTOTUNE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUTOTUNE_SCHEMA_VERSION:
            raise SpaceSchemaError(
                f"search-space schema_version must be {AUTOTUNE_SCHEMA_VERSION}"
            )
        templates = _decode_object(self._templates_json, "templates")
        operators = _decode_object(self._operators_json, "operators")
        memory = _decode_object(self._memory_configs_json, "memory_configs")
        grids = _decode_object(self._core_grids_json, "core_grids")
        extras = _decode_object(
            self._extra_program_configs_json, "extra_program_configs"
        )
        edges = _decode_object(self._edges_json, "edges")
        _validate_templates(templates)
        for name, operator in operators.items():
            _normalize_operator(name, operator)
        for path, descriptor in memory.items():
            if not path:
                raise SpaceSchemaError("memory config path must be non-empty")
            MemoryConfig.from_dict(descriptor)
        for path, grid in grids.items():
            if not path:
                raise SpaceSchemaError("core grid path must be non-empty")
            CoreGrid.from_value(grid)
        for path, descriptor in extras.items():
            if not path or not isinstance(descriptor, dict):
                raise SpaceSchemaError("extra program config must be an object")
        for name, edge in edges.items():
            if not name:
                raise SpaceSchemaError("edge name must be non-empty")
            EdgeConfig.from_dict(edge)

    @classmethod
    def from_runtime_config(
        cls, runtime_config: Mapping[str, Any]
    ) -> "SearchSpaceConfig":
        config = copy.deepcopy(dict(runtime_config))
        config.pop("autotune", None)
        operators = _extract_operators(config)
        memory_configs: dict[str, Any] = {}
        core_grids: dict[str, Any] = {}
        extra_program_configs: dict[str, Any] = {}
        _extract_runtime_descriptors(
            config,
            memory_configs=memory_configs,
            core_grids=core_grids,
            extra_program_configs=extra_program_configs,
        )
        edges = _extract_edges(memory_configs)
        return cls.create(
            templates=_extract_templates(config),
            operators=operators,
            memory_configs=memory_configs,
            core_grids=core_grids,
            extra_program_configs=extra_program_configs,
            edges=edges,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchSpaceConfig":
        if int(value.get("schema_version", 0)) != AUTOTUNE_SCHEMA_VERSION:
            raise SpaceSchemaError(
                f"search-space schema_version must be {AUTOTUNE_SCHEMA_VERSION}"
            )
        return cls.create(
            templates=value.get("templates") or {},
            operators=value.get("operators") or {},
            memory_configs=value.get("memory_configs") or {},
            core_grids=value.get("core_grids") or {},
            extra_program_configs=value.get("extra_program_configs") or {},
            edges=value.get("edges") or {},
        )

    @classmethod
    def create(
        cls,
        *,
        templates: Mapping[str, Any],
        operators: Mapping[str, Any],
        memory_configs: Mapping[str, Any],
        core_grids: Mapping[str, Any],
        extra_program_configs: Mapping[str, Any],
        edges: Mapping[str, Any],
    ) -> "SearchSpaceConfig":
        normalized_operators = {
            str(name): _normalize_operator(str(name), value)
            for name, value in operators.items()
        }
        normalized_memory = {
            str(path): MemoryConfig.from_dict(value).to_dict()
            for path, value in memory_configs.items()
        }
        normalized_grids = {
            str(path): CoreGrid.from_value(value).to_list()
            for path, value in core_grids.items()
        }
        normalized_edges = {
            str(name): EdgeConfig.from_dict(value).to_dict()
            for name, value in edges.items()
        }
        try:
            normalized_templates = normalize_template_selection(templates)
        except TemplateSelectionError as exc:
            raise SpaceSchemaError(str(exc)) from exc
        return cls(
            _templates_json=canonical_json(normalized_templates),
            _operators_json=canonical_json(normalized_operators),
            _memory_configs_json=canonical_json(normalized_memory),
            _core_grids_json=canonical_json(normalized_grids),
            _extra_program_configs_json=canonical_json(
                copy.deepcopy(dict(extra_program_configs))
            ),
            _edges_json=canonical_json(normalized_edges),
        )

    @property
    def templates(self) -> dict[str, Any]:
        return _decode_object(self._templates_json, "templates")

    @property
    def operators(self) -> dict[str, Any]:
        return _decode_object(self._operators_json, "operators")

    @property
    def memory_configs(self) -> dict[str, Any]:
        return _decode_object(self._memory_configs_json, "memory_configs")

    @property
    def core_grids(self) -> dict[str, Any]:
        return _decode_object(self._core_grids_json, "core_grids")

    @property
    def extra_program_configs(self) -> dict[str, Any]:
        return _decode_object(self._extra_program_configs_json, "extra_program_configs")

    @property
    def edges(self) -> dict[str, Any]:
        return _decode_object(self._edges_json, "edges")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "templates": self.templates,
            "operators": self.operators,
            "memory_configs": self.memory_configs,
            "core_grids": self.core_grids,
            "extra_program_configs": self.extra_program_configs,
            "edges": self.edges,
        }

    def tunable_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        sdpa = payload["operators"].get("attention.sdpa")
        if isinstance(sdpa, dict):
            sdpa.pop("exp_approx_mode", None)
        return payload

    def apply_to_runtime_config(
        self, runtime_config: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(runtime_config))
        result.pop("autotune", None)
        _apply_operators(result, self.operators)
        for path, descriptor in self.memory_configs.items():
            _set_path(
                result,
                path,
                MemoryConfig.from_dict(descriptor).to_runtime_descriptor(),
            )
        for path, grid in self.core_grids.items():
            _set_path(result, path, CoreGrid.from_value(grid).to_list())
        for path, descriptor in self.extra_program_configs.items():
            _set_path(result, path, copy.deepcopy(descriptor))
        result = _apply_templates(result, self.templates)
        result["autotune"] = self.to_dict()
        template_config = result.get("template_config")
        if isinstance(template_config, dict):
            template_config["autotune"] = self.to_dict()
        return result

    def with_sdpa_grid(self, grid: CoreGrid) -> "SearchSpaceConfig":
        payload = self.to_dict()
        sdpa = payload["operators"].get("attention.sdpa")
        if not isinstance(sdpa, dict):
            raise SpaceSchemaError("search space has no attention.sdpa operator")
        sdpa["grid"] = grid.to_list()
        program_grid_path = "attention.sdpa_program_config.core_grid"
        if program_grid_path in payload["core_grids"]:
            payload["core_grids"][program_grid_path] = grid.to_list()
        return SearchSpaceConfig.from_dict(payload)

    def with_lm_head_dram_concat(self) -> "SearchSpaceConfig":
        payload = self.to_dict()
        dram = MemoryConfig.named("DRAM_MEMORY_CONFIG").to_dict()
        for path in (
            "lm_head.shard_output_memory_config",
            "lm_head.concat_memory_config",
        ):
            payload["memory_configs"][path] = copy.deepcopy(dram)
        lm_head = payload["operators"].get("lm_head.shards")
        if isinstance(lm_head, dict):
            lm_head["output_memory"] = copy.deepcopy(dram)
        edge = payload["edges"].get("lm_head_shards_to_concat")
        if isinstance(edge, dict):
            edge["producer_output_memory"] = copy.deepcopy(dram)
            edge["consumer_input_memory"] = copy.deepcopy(dram)
            edge["conversion"] = "none"
        return SearchSpaceConfig.from_dict(payload)


def adapt_legacy_presets(
    runtime_config: Mapping[str, Any],
    *,
    memory_layout: str,
    program_config: str,
) -> SearchSpaceConfig:
    if memory_layout not in {OFFICIAL_LINEAR_OUTPUTS, LM_HEAD_DRAM_CONCAT}:
        raise SpaceSchemaError(f"unsupported legacy memory preset: {memory_layout}")
    if program_config not in {
        OFFICIAL_PROGRAM_CONFIG,
        SDPA_GRID_8X4_PROGRAM_CONFIG,
    }:
        raise SpaceSchemaError(f"unsupported legacy program preset: {program_config}")
    space = SearchSpaceConfig.from_runtime_config(runtime_config)
    if memory_layout == LM_HEAD_DRAM_CONCAT:
        space = space.with_lm_head_dram_concat()
    if program_config == SDPA_GRID_8X4_PROGRAM_CONFIG:
        space = space.with_sdpa_grid(CoreGrid(8, 4))
    return space


def _extract_templates(config: Mapping[str, Any]) -> dict[str, str]:
    attention = config.get("attention") or {}
    operations = set(attention.get("op_sequence") or [])
    mlp = config.get("mlp") or {}
    lm_head = config.get("lm_head") or {}
    return {
        KV_UPDATE_AXIS: (
            FUSED_PAGED_UPDATE
            if "paged_fused_update_cache" in operations
            else SEPARATE_PAGED_UPDATE
        ),
        ROPE_AXIS: (
            FUSED_QK_ROPE
            if "rotary_embedding_llama_fused_qk" in operations
            else SEPARATE_QK_ROPE
        ),
        GATE_UP_AXIS: (
            PACKED_GATE_UP
            if mlp.get("packed_gate_up_program_config") is not None
            or mlp.get("template") == PACKED_GATE_UP
            else SEPARATE_GATE_UP
        ),
        ACTIVATION_AXIS: (
            "gate_linear_fused_silu"
            if mlp.get("gate_linear_activation") == "silu"
            else MUL_FUSED_SILU
        ),
        "lm_head": (
            "official_split_force_argmax"
            if lm_head.get("argmax_strategy") == "full_logits_untilize_multicore_argmax"
            else str(lm_head.get("template", "official_split_lm_head"))
        ),
    }


def _validate_templates(templates: Mapping[str, Any]) -> None:
    try:
        normalize_template_selection(templates)
    except TemplateSelectionError as exc:
        raise SpaceSchemaError(str(exc)) from exc
    lm_head = templates.get("lm_head")
    if not isinstance(lm_head, str) or not lm_head:
        raise SpaceSchemaError("template choice 'lm_head' must be a non-empty string")


def _extract_operators(config: Mapping[str, Any]) -> dict[str, Any]:
    operators: dict[str, Any] = {}
    for name, metadata in _MATMUL_PATHS.items():
        section = config.get(metadata["section"]) or {}
        if "program" in metadata:
            descriptor = section.get(metadata["program"])
            if not isinstance(descriptor, Mapping):
                continue
            programs = [MatmulProgramConfig.from_runtime_descriptor(descriptor)]
        else:
            descriptors = section.get(metadata["programs"])
            if not isinstance(descriptors, list) or not descriptors:
                continue
            if not all(isinstance(descriptor, Mapping) for descriptor in descriptors):
                if all(descriptor is None for descriptor in descriptors):
                    continue
                raise SpaceSchemaError(
                    f"operator {name} program list mixes descriptors and null entries"
                )
            programs = [
                MatmulProgramConfig.from_runtime_descriptor(descriptor)
                for descriptor in descriptors
            ]
        output = section.get(metadata["output_memory"])
        operator = {
            "kind": "matmul",
            "programs": [program.to_dict() for program in programs],
            "compute_grid": copy.deepcopy(section.get("core_grid")),
            "output_memory": (
                MemoryConfig.from_runtime_descriptor(output).to_dict()
                if isinstance(output, Mapping)
                else None
            ),
        }
        operators[name] = operator
    attention = config.get("attention") or {}
    sdpa_descriptor = attention.get("sdpa_program_config")
    if isinstance(sdpa_descriptor, Mapping):
        sdpa = SDPAProgramConfig.from_runtime_descriptor(sdpa_descriptor).to_dict()
        kernel_output = attention.get("sdpa_kernel_output_memory_config")
        output = attention.get("sdpa_output_memory_config")
        sdpa["kernel_output_memory"] = (
            MemoryConfig.from_runtime_descriptor(kernel_output).to_dict()
            if isinstance(kernel_output, Mapping)
            else None
        )
        sdpa["output_memory"] = (
            MemoryConfig.from_runtime_descriptor(output).to_dict()
            if isinstance(output, Mapping)
            else None
        )
        operators["attention.sdpa"] = sdpa
    return operators


def _normalize_operator(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpaceSchemaError(f"operator {name} must be an object")
    operator = copy.deepcopy(dict(value))
    kind = operator.get("kind")
    if kind == "matmul":
        programs = operator.get("programs")
        if not isinstance(programs, list) or not programs:
            raise SpaceSchemaError(f"matmul operator {name} has no programs")
        operator["programs"] = [
            MatmulProgramConfig.from_dict(program).to_dict() for program in programs
        ]
        if operator.get("compute_grid") is not None:
            operator["compute_grid"] = CoreGrid.from_value(
                operator["compute_grid"]
            ).to_list()
        if operator.get("output_memory") is not None:
            operator["output_memory"] = MemoryConfig.from_dict(
                operator["output_memory"]
            ).to_dict()
        return operator
    if kind == "sdpa":
        normalized = SDPAProgramConfig.from_dict(operator).to_dict()
        for key in ("kernel_output_memory", "output_memory"):
            if operator.get(key) is not None:
                normalized[key] = MemoryConfig.from_dict(operator[key]).to_dict()
            else:
                normalized[key] = None
        return normalized
    raise SpaceSchemaError(f"operator {name} has unsupported kind {kind!r}")


def _extract_runtime_descriptors(
    value: Any,
    *,
    memory_configs: dict[str, Any],
    core_grids: dict[str, Any],
    extra_program_configs: dict[str, Any],
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key == "autotune":
                continue
            child_path = (*path, key)
            dotted = ".".join(child_path)
            if (
                key.endswith("memory_config")
                and isinstance(child, Mapping)
                and isinstance(child.get("kind"), str)
            ):
                memory_configs[dotted] = MemoryConfig.from_runtime_descriptor(
                    child
                ).to_dict()
            if key == "core_grid" and isinstance(child, (list, tuple)):
                core_grids[dotted] = CoreGrid.from_value(child).to_list()
            if key.endswith("program_config") and isinstance(child, Mapping):
                if dotted not in _TYPED_PROGRAM_PATHS:
                    extra_program_configs[dotted] = copy.deepcopy(dict(child))
            _extract_runtime_descriptors(
                child,
                memory_configs=memory_configs,
                core_grids=core_grids,
                extra_program_configs=extra_program_configs,
                path=child_path,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _extract_runtime_descriptors(
                child,
                memory_configs=memory_configs,
                core_grids=core_grids,
                extra_program_configs=extra_program_configs,
                path=(*path, str(index)),
            )


def _extract_edges(memory_configs: Mapping[str, Any]) -> dict[str, Any]:
    edges: dict[str, Any] = {}
    for name, (producer_path, consumer_path) in _EDGE_PATHS.items():
        producer = memory_configs.get(producer_path)
        consumer = memory_configs.get(consumer_path)
        if not isinstance(producer, Mapping) or not isinstance(consumer, Mapping):
            continue
        producer_config = MemoryConfig.from_dict(producer)
        consumer_config = MemoryConfig.from_dict(consumer)
        conversion = (
            "none"
            if producer_config.to_runtime_descriptor()
            == consumer_config.to_runtime_descriptor()
            else "explicit"
        )
        edges[name] = EdgeConfig(
            producer_path=producer_path,
            consumer_path=consumer_path,
            producer_output_memory=producer_config,
            consumer_input_memory=consumer_config,
            conversion=conversion,
        ).to_dict()
    return edges


def _apply_templates(
    result: dict[str, Any], templates: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_templates(templates)
    try:
        return apply_template_selection(result, templates)
    except TemplateSelectionError as exc:
        raise SpaceSchemaError(str(exc)) from exc


def _apply_operators(result: dict[str, Any], operators: Mapping[str, Any]) -> None:
    for name, raw_operator in operators.items():
        operator = _normalize_operator(name, raw_operator)
        if name in _MATMUL_PATHS:
            metadata = _MATMUL_PATHS[name]
            section = result.setdefault(metadata["section"], {})
            programs = [
                MatmulProgramConfig.from_dict(program).to_runtime_descriptor()
                for program in operator["programs"]
            ]
            if "program" in metadata:
                if len(programs) != 1:
                    raise SpaceSchemaError(
                        f"operator {name} requires exactly one program"
                    )
                section[metadata["program"]] = programs[0]
            else:
                section[metadata["programs"]] = programs
            if operator.get("compute_grid") is not None:
                section["core_grid"] = CoreGrid.from_value(
                    operator["compute_grid"]
                ).to_list()
            if operator.get("output_memory") is not None:
                section[metadata["output_memory"]] = MemoryConfig.from_dict(
                    operator["output_memory"]
                ).to_runtime_descriptor()
            continue
        if name == "attention.sdpa":
            attention = result.setdefault("attention", {})
            baseline = attention.get("sdpa_program_config") or {}
            expected_exp_approx = baseline.get("exp_approx_mode")
            if expected_exp_approx is not None and bool(expected_exp_approx) != bool(
                operator["exp_approx_mode"]
            ):
                raise SpaceSchemaError(
                    "SDPA exp_approx_mode is precision-frozen and cannot change"
                )
            attention["sdpa_program_config"] = SDPAProgramConfig.from_dict(
                operator
            ).to_runtime_descriptor()
            for key, runtime_key in (
                ("kernel_output_memory", "sdpa_kernel_output_memory_config"),
                ("output_memory", "sdpa_output_memory_config"),
            ):
                if operator.get(key) is not None:
                    attention[runtime_key] = MemoryConfig.from_dict(
                        operator[key]
                    ).to_runtime_descriptor()
            continue
        raise SpaceSchemaError(f"unsupported operator: {name}")


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = target
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})
    if isinstance(current, list):
        current[int(parts[-1])] = copy.deepcopy(value)
    else:
        current[parts[-1]] = copy.deepcopy(value)


def _decode_json(value: str, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SpaceSchemaError(f"{label} must be valid JSON") from exc


def _decode_object(value: str, label: str) -> dict[str, Any]:
    payload = _decode_json(value, label)
    if not isinstance(payload, dict):
        raise SpaceSchemaError(f"{label} must be a JSON object")
    return payload
