from __future__ import annotations

import copy
from typing import Any


class TTNNConfigResolutionError(RuntimeError):
    pass


def realize_ttnn_config(value: Any, ttnn: Any) -> Any:
    if isinstance(value, list):
        return [realize_ttnn_config(item, ttnn) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind is None or not str(kind).startswith("ttnn_"):
        return {key: realize_ttnn_config(item, ttnn) for key, item in value.items()}
    resolver = _RESOLVERS.get(str(kind))
    if resolver is None:
        raise TTNNConfigResolutionError(
            f"unsupported TTNN config descriptor kind: {kind}"
        )
    return resolver(copy.deepcopy(value), ttnn)


def _dtype(spec: dict[str, Any], ttnn: Any) -> Any:
    return _required_attr(ttnn, str(spec["name"]))


def _memory_config(spec: dict[str, Any], ttnn: Any) -> Any:
    return _required_attr(ttnn, str(spec["name"]))


def _sharded_memory_config(spec: dict[str, Any], ttnn: Any) -> Any:
    create = _required_attr(ttnn, "create_sharded_memory_config")
    strategy_type = _required_attr(ttnn, "ShardStrategy")
    orientation_type = _required_attr(ttnn, "ShardOrientation")
    strategy = _required_attr(strategy_type, str(spec["strategy"]).upper())
    orientation = _required_attr(
        orientation_type,
        str(spec.get("orientation", "row_major")).upper(),
    )
    core_grid = (
        _core_range_set(spec["core_ranges"], ttnn)
        if spec.get("core_ranges") is not None
        else _core_grid_value(spec["core_grid"], ttnn)
    )
    return create(
        shape=tuple(int(value) for value in spec["shard_shape"]),
        core_grid=core_grid,
        strategy=strategy,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_sharded_memory_config(spec: dict[str, Any], ttnn: Any) -> Any:
    k = int(spec["k"])
    n = int(spec["n"])
    grid_width = int(spec.get("dram_grid_width", 8))
    tile_size = int(spec.get("tile_size", 32))
    padded_n = (
        (n + tile_size * grid_width - 1)
        // (tile_size * grid_width)
        * tile_size
        * grid_width
    )
    core_coord = _required_attr(ttnn, "CoreCoord")
    core_range = _required_attr(ttnn, "CoreRange")
    core_range_set = _required_attr(ttnn, "CoreRangeSet")
    shard_spec_type = _required_attr(ttnn, "ShardSpec")
    orientation = _required_attr(
        _required_attr(ttnn, "ShardOrientation"),
        "ROW_MAJOR",
    )
    memory_layout = _required_attr(
        _required_attr(ttnn, "TensorMemoryLayout"),
        "WIDTH_SHARDED",
    )
    buffer_type = _required_attr(
        _required_attr(ttnn, "BufferType"),
        "DRAM",
    )
    grid = core_range_set(
        {
            core_range(
                core_coord(0, 0),
                core_coord(grid_width - 1, 0),
            )
        }
    )
    shard_spec = shard_spec_type(
        grid,
        (k, padded_n // grid_width),
        orientation,
    )
    memory_config_type = _required_attr(ttnn, "MemoryConfig")
    return memory_config_type(memory_layout, buffer_type, shard_spec)


def _core_grid(spec: dict[str, Any], ttnn: Any) -> Any:
    return _core_grid_value([spec["x"], spec["y"]], ttnn)


def _matmul_dram_sharded_program_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(
        ttnn,
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
    )
    return constructor(
        in0_block_w=int(spec["in0_block_w"]),
        per_core_M=int(spec["per_core_M"]),
        per_core_N=int(spec["per_core_N"]),
        fused_activation=_fused_activation(spec, ttnn),
    )


def _matmul_batched_dram_sharded_program_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(
        ttnn,
        "MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig",
    )
    return constructor(
        in0_block_w=int(spec["in0_block_w"]),
        per_core_M=int(spec["per_core_M"]),
        per_core_N=int(spec["per_core_N"]),
        fused_activation=_fused_activation(spec, ttnn),
    )


def _matmul_multicore_reuse_program_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(ttnn, "MatmulMultiCoreReuseProgramConfig")
    return constructor(
        compute_with_storage_grid_size=_core_coord_value(spec["core_grid"], ttnn),
        in0_block_w=int(spec["in0_block_w"]),
        out_subblock_h=int(spec["out_subblock_h"]),
        out_subblock_w=int(spec["out_subblock_w"]),
        per_core_M=int(spec["per_core_M"]),
        per_core_N=int(spec["per_core_N"]),
    )


def _matmul_multicore_reuse_mcast_program_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(
        ttnn,
        "MatmulMultiCoreReuseMultiCastProgramConfig",
    )
    return constructor(
        compute_with_storage_grid_size=_core_coord_value(spec["core_grid"], ttnn),
        in0_block_w=int(spec["in0_block_w"]),
        out_subblock_h=int(spec["out_subblock_h"]),
        out_subblock_w=int(spec["out_subblock_w"]),
        out_block_h=int(spec.get("out_block_h", spec["per_core_M"])),
        out_block_w=int(spec.get("out_block_w", spec["per_core_N"])),
        per_core_M=int(spec["per_core_M"]),
        per_core_N=int(spec["per_core_N"]),
        transpose_mcast=bool(spec.get("transpose_mcast", False)),
        fused_activation=_fused_activation(spec, ttnn),
        fuse_batch=bool(spec.get("fuse_batch", False)),
    )


def _matmul_multicore_reuse_mcast_1d_program_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(
        ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
    )
    kwargs = {
        "compute_with_storage_grid_size": _core_coord_value(
            spec["core_grid"], ttnn
        ),
        "in0_block_w": int(spec["in0_block_w"]),
        "out_subblock_h": int(spec["out_subblock_h"]),
        "out_subblock_w": int(spec["out_subblock_w"]),
        "out_block_h": int(spec.get("out_block_h", spec["per_core_M"])),
        "out_block_w": int(spec.get("out_block_w", spec["per_core_N"])),
        "per_core_M": int(spec["per_core_M"]),
        "per_core_N": int(spec["per_core_N"]),
        "fuse_batch": bool(spec.get("fuse_batch", True)),
        "fused_activation": _fused_activation(spec, ttnn),
        "mcast_in0": bool(spec.get("mcast_in0", False)),
        "gather_in0": bool(spec.get("gather_in0", False)),
        "num_global_cb_receivers": int(
            spec.get("num_global_cb_receivers", 1)
        ),
        "untilize_out": bool(spec.get("untilize_out", False)),
    }
    hop_cores = spec.get("hop_cores")
    if hop_cores:
        kwargs["hop_cores"] = _core_range_set(hop_cores, ttnn)
    return constructor(**kwargs)


def _sdpa_program_config(spec: dict[str, Any], ttnn: Any) -> Any:
    constructor = _required_attr(ttnn, "SDPAProgramConfig")
    grid = tuple(int(value) for value in spec["core_grid"])
    kwargs = {
        "compute_with_storage_grid_size": grid,
        "q_chunk_size": int(spec["q_chunk_size"]),
        "k_chunk_size": int(spec["k_chunk_size"]),
        "exp_approx_mode": bool(spec["exp_approx_mode"]),
        "max_cores_per_head_batch": int(spec["max_cores_per_head_batch"]),
    }
    if spec.get("sub_core_grids") is not None:
        kwargs["sub_core_grids"] = _core_range_set(
            spec["sub_core_grids"],
            ttnn,
        )
    return constructor(
        **kwargs,
    )


def _layer_norm_program_config(spec: dict[str, Any], ttnn: Any) -> Any:
    constructor = _required_attr(
        ttnn,
        "LayerNormShardedMultiCoreProgramConfig",
    )
    grid = [int(value) for value in spec["core_grid"]]
    return constructor(
        compute_with_storage_grid_size=grid,
        subblock_w=int(spec["subblock_w"]),
        block_h=int(spec["block_h"]),
        block_w=int(spec["block_w"]),
        inplace=bool(spec.get("inplace", False)),
    )


def _wormhole_compute_kernel_config(
    spec: dict[str, Any],
    ttnn: Any,
) -> Any:
    constructor = _required_attr(ttnn, "WormholeComputeKernelConfig")
    math_fidelity = _required_attr(
        _required_attr(ttnn, "MathFidelity"),
        str(spec["math_fidelity"]),
    )
    return constructor(
        math_fidelity=math_fidelity,
        math_approx_mode=bool(spec["math_approx_mode"]),
        fp32_dest_acc_en=bool(spec["fp32_dest_acc_en"]),
        packer_l1_acc=bool(spec["packer_l1_acc"]),
    )


def _core_grid_value(value: Any, ttnn: Any) -> Any:
    x, y = (int(item) for item in value)
    constructor = _required_attr(ttnn, "CoreGrid")
    try:
        return constructor(x=x, y=y)
    except TypeError:
        return constructor(y, x)


def _core_coord_value(value: Any, ttnn: Any) -> Any:
    x, y = (int(item) for item in value)
    constructor = _required_attr(ttnn, "CoreCoord")
    try:
        return constructor(x, y)
    except TypeError:
        return constructor(x=x, y=y)


def _core_range_set(value: Any, ttnn: Any) -> Any:
    if not isinstance(value, list) or not value:
        raise TTNNConfigResolutionError("core_ranges must be a non-empty list")
    core_coord = _required_attr(ttnn, "CoreCoord")
    core_range = _required_attr(ttnn, "CoreRange")
    core_range_set = _required_attr(ttnn, "CoreRangeSet")
    ranges = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise TTNNConfigResolutionError("each core range must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (int(component) for component in item)
        ranges.add(core_range(core_coord(x0, y0), core_coord(x1, y1)))
    return core_range_set(ranges)


def _fused_activation(spec: dict[str, Any], ttnn: Any) -> Any | None:
    value = spec.get("fused_activation")
    if value is None:
        return None
    unary_op_type = _required_attr(ttnn, "UnaryOpType")
    activation = getattr(unary_op_type, str(value).upper(), None)
    if activation is None:
        raise TTNNConfigResolutionError(f"unsupported fused activation: {value!r}")
    return activation


def _required_attr(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    if value is None:
        raise TTNNConfigResolutionError(
            f"TTNN runtime does not expose required config API: {name}"
        )
    return value


_RESOLVERS = {
    "ttnn_dtype": _dtype,
    "ttnn_memory_config": _memory_config,
    "ttnn_sharded_memory_config": _sharded_memory_config,
    "ttnn_dram_sharded_memory_config": _dram_sharded_memory_config,
    "ttnn_core_grid": _core_grid,
    "ttnn_matmul_dram_sharded_program_config": (_matmul_dram_sharded_program_config),
    "ttnn_matmul_multi_core_reuse_multi_cast_dram_sharded_program_config": (
        _matmul_batched_dram_sharded_program_config
    ),
    "ttnn_matmul_multicore_reuse_program_config": (
        _matmul_multicore_reuse_program_config
    ),
    "ttnn_matmul_multicore_reuse_mcast_program_config": (
        _matmul_multicore_reuse_mcast_program_config
    ),
    "ttnn_matmul_multicore_reuse_mcast_1d_program_config": (
        _matmul_multicore_reuse_mcast_1d_program_config
    ),
    "ttnn_sdpa_program_config": _sdpa_program_config,
    "ttnn_layer_norm_program_config": _layer_norm_program_config,
    "ttnn_wormhole_compute_kernel_config": (_wormhole_compute_kernel_config),
}
