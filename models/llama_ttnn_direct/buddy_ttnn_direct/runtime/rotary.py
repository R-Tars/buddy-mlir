from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Sequence


class RotaryConfigurationError(ValueError):
    pass


def attach_prefill_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
) -> SimpleNamespace:
    head_dim = _head_dim(plan)
    seq_len = int(plan["prefill_len"])
    rotary_config = _rotary_config(plan)
    host = build_prefill_rotary_host_tensors(
        torch=torch,
        seq_len=seq_len,
        head_dim=head_dim,
        theta=float(rotary_config["theta"]),
        scaling=rotary_config.get("scaling"),
        dtype_seed=dtype_seed,
    )
    kwargs = _tensor_kwargs(
        ttnn=ttnn,
        device=device,
        dtype_seed=dtype_seed,
    )
    transform_kwargs = dict(kwargs)
    dram_memory = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if dram_memory is not None:
        transform_kwargs["memory_config"] = dram_memory
    shared_rotary = SimpleNamespace(
        cos_matrix=ttnn.from_torch(host.cos, **kwargs),
        sin_matrix=ttnn.from_torch(host.sin, **kwargs),
        transformation_matrix=ttnn.from_torch(
            host.transformation,
            **transform_kwargs,
        ),
    )
    _install_shared_rotary(parameters, shared_rotary, int(plan["layers"]))
    return SimpleNamespace(
        tensor_conversion_count=3,
        rotary_runtime_state={
            "status": "built",
            "source": "hf_rope_config",
            "mode": "prefill",
            "theta": float(rotary_config["theta"]),
            "scaling": rotary_config.get("scaling"),
            "position_start": 0,
            "position_end": seq_len - 1,
            "cos_sin_shape": [1, 1, seq_len, head_dim],
            "transformation_shape": [1, 1, 32, 32],
            "tensor_count": 3,
            "shared_across_layers": True,
        },
    )


def attach_decode_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    plan: dict[str, Any],
    cache_position_value: int | None = None,
    cache_position_values: Sequence[int] | None = None,
) -> SimpleNamespace:
    head_dim = _head_dim(plan)
    batch_size = int(plan["input_shapes"]["cache_position"][0])
    fused_qk = _uses_fused_qk_rope(plan)
    rotary_batch_size = 2 * batch_size if fused_qk else batch_size
    rotary_config = _rotary_config(plan)
    if cache_position_values is None:
        if cache_position_value is None:
            raise ValueError(
                "cache_position_value or cache_position_values is required"
            )
        positions = [int(cache_position_value)] * batch_size
    else:
        positions = [int(value) for value in cache_position_values]
        if len(positions) != batch_size:
            raise ValueError(
                "cache_position_values length must match decode batch size"
            )
    uniform_position = positions[0] if len(set(positions)) == 1 else None
    host = build_decode_rotary_host_tensors(
        torch=torch,
        positions=positions,
        head_dim=head_dim,
        theta=float(rotary_config["theta"]),
        scaling=rotary_config.get("scaling"),
        dtype_seed=dtype_seed,
        fused_qk=fused_qk,
    )
    kwargs = _tensor_kwargs(
        ttnn=ttnn,
        device=device,
        dtype_seed=dtype_seed,
    )
    cos_sin_kwargs = dict(kwargs)
    cos_sin_memory = _rotary_cos_sin_memory_config(
        ttnn,
        device,
        batch_size=rotary_batch_size,
        head_dim=head_dim,
    )
    if cos_sin_memory is not None:
        cos_sin_kwargs["memory_config"] = cos_sin_memory
    transform_kwargs = dict(kwargs)
    transform_memory = _rotary_transform_memory_config(
        ttnn,
        device,
        batch_size=rotary_batch_size,
    )
    if transform_memory is not None:
        transform_kwargs["memory_config"] = transform_memory
    shared_rotary = SimpleNamespace(
        cos_matrix=ttnn.from_torch(host.cos, **cos_sin_kwargs),
        sin_matrix=ttnn.from_torch(host.sin, **cos_sin_kwargs),
        transformation_matrix=ttnn.from_torch(
            host.transformation,
            **transform_kwargs,
        ),
    )
    _install_shared_rotary(parameters, shared_rotary, int(plan["layers"]))
    return SimpleNamespace(
        tensor_conversion_count=3,
        rotary_runtime_state={
            "status": "built",
            "source": "hf_rope_config",
            "mode": "decode",
            "theta": float(rotary_config["theta"]),
            "scaling": rotary_config.get("scaling"),
            "cache_position_value": uniform_position,
            "cache_position_values": positions,
            "positions": positions,
            "template": ("fused_qk_rope" if fused_qk else "separate_qk_rope"),
            "rotary_batch_size": rotary_batch_size,
            "cos_sin_shape": [1, rotary_batch_size, 1, head_dim],
            "transformation_shape": [
                1,
                1,
                rotary_batch_size * 32,
                32,
            ],
            "tensor_count": 3,
            "shared_across_layers": True,
            "memory_config": "height_sharded",
            "ttnn_memory_config": _config_repr(cos_sin_memory),
            "transform_memory_config": "height_sharded",
            "transform_ttnn_memory_config": _config_repr(transform_memory),
        },
    )


def build_prefill_rotary_host_tensors(
    *,
    torch: Any,
    seq_len: int,
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None,
    dtype_seed: str,
) -> SimpleNamespace:
    if seq_len <= 0:
        raise RotaryConfigurationError("prefill sequence length must be positive")
    cos, sin = _cos_sin_values(
        positions=range(seq_len),
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
    )
    return SimpleNamespace(
        cos=_named_tensor(
            torch,
            [[cos]],
            dtype_seed=dtype_seed,
            name="runtime.shared.prefill_rotary_cos",
        ),
        sin=_named_tensor(
            torch,
            [[sin]],
            dtype_seed=dtype_seed,
            name="runtime.shared.prefill_rotary_sin",
        ),
        transformation=_named_tensor(
            torch,
            [[_rotary_transformation_rows(1)]],
            dtype_seed=dtype_seed,
            name="runtime.shared.prefill_rotary_transform",
        ),
    )


def build_decode_rotary_host_tensors(
    *,
    torch: Any,
    positions: list[int],
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None,
    dtype_seed: str,
    fused_qk: bool = False,
) -> SimpleNamespace:
    if not positions:
        raise RotaryConfigurationError("decode positions must be non-empty")
    if any(int(position) < 0 for position in positions):
        raise RotaryConfigurationError("decode positions must be non-negative")
    rotary_positions = list(positions)
    if fused_qk:
        rotary_positions *= 2
    cos, sin = _cos_sin_values(
        positions=rotary_positions,
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
    )
    return SimpleNamespace(
        cos=_named_tensor(
            torch,
            [[[row] for row in cos]],
            dtype_seed=dtype_seed,
            name="runtime.shared.decode_rotary_cos",
        ),
        sin=_named_tensor(
            torch,
            [[[row] for row in sin]],
            dtype_seed=dtype_seed,
            name="runtime.shared.decode_rotary_sin",
        ),
        transformation=_named_tensor(
            torch,
            [[_rotary_transformation_rows(len(rotary_positions))]],
            dtype_seed=dtype_seed,
            name="runtime.shared.decode_rotary_transform",
        ),
    )


def build_decode_rotary_cache_host_tensors(
    *,
    torch: Any,
    cache_len: int,
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None,
    dtype_seed: str,
) -> SimpleNamespace:
    if cache_len <= 0:
        raise RotaryConfigurationError("decode cache length must be positive")
    cos, sin = _cos_sin_values(
        positions=range(cache_len),
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
    )
    return SimpleNamespace(
        cos=_named_tensor(
            torch,
            cos,
            dtype_seed=dtype_seed,
            name="runtime.persistent.decode_rotary_cos_cache",
        ),
        sin=_named_tensor(
            torch,
            sin,
            dtype_seed=dtype_seed,
            name="runtime.persistent.decode_rotary_sin_cache",
        ),
    )


def install_shared_rotary_parameters(
    *,
    parameters: Any,
    cos_matrix: Any,
    sin_matrix: Any,
    transformation_matrix: Any,
    layer_count: int,
) -> None:
    _install_shared_rotary(
        parameters,
        SimpleNamespace(
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            transformation_matrix=transformation_matrix,
        ),
        layer_count,
    )


def decode_rotary_cos_sin_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    return _rotary_cos_sin_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        head_dim=head_dim,
    )


def decode_rotary_transform_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
) -> Any | None:
    return _rotary_transform_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
    )


def _cos_sin_values(
    *,
    positions: Any,
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None,
) -> tuple[list[list[float]], list[list[float]]]:
    inv_freq = _inverse_frequencies(
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
    )
    cos_rows: list[list[float]] = []
    sin_rows: list[list[float]] = []
    for position in positions:
        angles = [float(position) * frequency for frequency in inv_freq]
        cos_rows.append([value for angle in angles for value in (math.cos(angle),) * 2])
        sin_rows.append([value for angle in angles for value in (math.sin(angle),) * 2])
    return cos_rows, sin_rows


def _inverse_frequencies(
    *,
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None,
) -> list[float]:
    if head_dim <= 0 or head_dim % 2:
        raise RotaryConfigurationError("head_dim must be a positive even integer")
    if theta <= 0:
        raise RotaryConfigurationError("rope theta must be positive")
    frequencies = [
        1.0 / (theta ** (dimension / head_dim)) for dimension in range(0, head_dim, 2)
    ]
    if scaling is None:
        return frequencies
    rope_type = str(scaling.get("rope_type", scaling.get("type", "default")))
    if rope_type in {"default", "none"}:
        return frequencies
    if rope_type != "llama3":
        raise RotaryConfigurationError(f"unsupported rope scaling type: {rope_type}")
    factor = float(scaling["factor"])
    low_factor = float(scaling["low_freq_factor"])
    high_factor = float(scaling["high_freq_factor"])
    original_length = float(scaling["original_max_position_embeddings"])
    if factor <= 0 or low_factor <= 0 or high_factor <= low_factor:
        raise RotaryConfigurationError("invalid llama3 rope scaling factors")
    low_wavelength = original_length / low_factor
    high_wavelength = original_length / high_factor
    scaled: list[float] = []
    for frequency in frequencies:
        wavelength = 2.0 * math.pi / frequency
        if wavelength < high_wavelength:
            scaled.append(frequency)
        elif wavelength > low_wavelength:
            scaled.append(frequency / factor)
        else:
            smooth = (original_length / wavelength - low_factor) / (
                high_factor - low_factor
            )
            scaled.append((1.0 - smooth) * frequency / factor + smooth * frequency)
    return scaled


def _rotary_transformation_rows(batch_size: int) -> list[list[float]]:
    tile_size = 32
    matrix = [[0.0] * tile_size for _ in range(tile_size)]
    for index in range(0, tile_size, 2):
        matrix[index][index + 1] = 1.0
        matrix[index + 1][index] = -1.0
    return [list(row) for _ in range(batch_size) for row in matrix]


def _named_tensor(
    torch: Any,
    values: Any,
    *,
    dtype_seed: str,
    name: str,
) -> Any:
    dtype = (
        getattr(torch, "bfloat16", None)
        if dtype_seed == "bf16"
        else getattr(torch, "float32", None)
    )
    try:
        tensor = torch.tensor(values, dtype=dtype)
    except TypeError:
        tensor = torch.tensor(values)
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor


def _head_dim(plan: dict[str, Any]) -> int:
    shape = plan["layer_parameter_shapes"]["rotary_cos_matrix"]
    return int(shape[-1])


def _rotary_config(plan: dict[str, Any]) -> dict[str, Any]:
    value = plan.get("rotary") or {}
    return {
        "theta": float(value.get("theta", 10000.0)),
        "scaling": value.get("scaling"),
    }


def _uses_fused_qk_rope(plan: dict[str, Any]) -> bool:
    templates = plan.get("templates")
    return isinstance(templates, dict) and templates.get("attention.rope") == (
        "fused_qk_rope"
    )


def _tensor_kwargs(
    *,
    ttnn: Any,
    device: Any,
    dtype_seed: str,
) -> dict[str, Any]:
    kwargs = {"device": device}
    dtype_name = "bfloat16" if dtype_seed == "bf16" else "float32"
    dtype = getattr(ttnn, dtype_name, None)
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    return kwargs


def _install_shared_rotary(
    parameters: Any,
    rotary: SimpleNamespace,
    layer_count: int,
) -> None:
    for layer_id in range(layer_count):
        layer = parameters.layers[layer_id]
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = SimpleNamespace()
            layer.attention = attention
        attention.rotary = rotary


def _rotary_cos_sin_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    tile_size = int(getattr(ttnn, "TILE_SIZE", 32))
    return _sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        shard_shape=(tile_size, head_dim),
    ) or getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)


def _rotary_transform_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
) -> Any | None:
    tile_size = int(getattr(ttnn, "TILE_SIZE", 32))
    return _sharded_memory_config(
        ttnn,
        device,
        batch_size=batch_size,
        shard_shape=(tile_size, tile_size),
    ) or getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)


def _sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    shard_shape: tuple[int, int],
) -> Any | None:
    create_sharded = getattr(ttnn, "create_sharded_memory_config", None)
    core_grid_type = getattr(ttnn, "CoreGrid", None)
    strategy = getattr(getattr(ttnn, "ShardStrategy", None), "HEIGHT", None)
    orientation = getattr(
        getattr(ttnn, "ShardOrientation", None),
        "ROW_MAJOR",
        None,
    )
    if not callable(create_sharded) or not callable(core_grid_type):
        return None
    if strategy is None:
        return None
    compute_grid = None
    grid_size = getattr(device, "compute_with_storage_grid_size", None)
    if callable(grid_size):
        try:
            compute_grid = grid_size()
        except Exception:
            compute_grid = None
    physical_x = int(getattr(compute_grid, "x", 8) or 8)
    physical_y = int(getattr(compute_grid, "y", 8) or 8)
    grid_x = max(1, min(batch_size, physical_x))
    while grid_x > 1 and batch_size % grid_x:
        grid_x -= 1
    grid_y = max(1, (batch_size + grid_x - 1) // grid_x)
    if grid_y > physical_y:
        return None
    try:
        core_grid = core_grid_type(y=grid_y, x=grid_x)
    except TypeError:
        core_grid = core_grid_type(grid_y, grid_x)
    try:
        return create_sharded(
            shape=shard_shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:
        return None


def _config_repr(value: Any | None) -> str | None:
    return None if value is None else str(value)
