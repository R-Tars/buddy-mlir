from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

TEMPLATE_REGISTRY_SCHEMA_VERSION = 1

KV_UPDATE_AXIS = "attention.kv_update"
ROPE_AXIS = "attention.rope"
ACTIVATION_AXIS = "mlp.activation_placement"
GATE_UP_AXIS = "mlp.gate_up"

SEPARATE_PAGED_UPDATE = "separate_paged_update"
FUSED_PAGED_UPDATE = "fused_paged_update"
SEPARATE_QK_ROPE = "separate_qk_rope"
FUSED_QK_ROPE = "fused_qk_rope"
MUL_FUSED_SILU = "mul_fused_silu"
GATE_LINEAR_FUSED_SILU = "gate_linear_fused_silu"
SEPARATE_GATE_UP = "separate_gate_up"
PACKED_GATE_UP = "packed_gate_up"

DEFAULT_TEMPLATE_SELECTION = {
    KV_UPDATE_AXIS: SEPARATE_PAGED_UPDATE,
    ROPE_AXIS: SEPARATE_QK_ROPE,
    ACTIVATION_AXIS: MUL_FUSED_SILU,
    GATE_UP_AXIS: SEPARATE_GATE_UP,
}

_ALIASES = {
    KV_UPDATE_AXIS: {
        "paged_fused_update_cache": FUSED_PAGED_UPDATE,
    },
    GATE_UP_AXIS: {
        "separate_projection": SEPARATE_GATE_UP,
        "packed_projection": PACKED_GATE_UP,
    },
}

_FUSED_ACTIVATION_PROGRAM_KINDS = {
    "ttnn_matmul_dram_sharded_program_config",
    "ttnn_matmul_multicore_reuse_mcast_program_config",
    "ttnn_matmul_multicore_reuse_mcast_1d_program_config",
    "ttnn_matmul_multi_core_reuse_multi_cast_dram_sharded_program_config",
}


class TemplateSelectionError(ValueError):
    """Raised when a semantic template selection is malformed or incompatible."""


@dataclass(frozen=True)
class TemplateConstraint:
    code: str
    path: str
    message: str
    error_class: str = "invalid_program_config"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "error_class": self.error_class,
        }


@dataclass(frozen=True)
class TemplateAvailability:
    template: str
    available: bool | None
    resolved_apis: tuple[str, ...]
    missing_api_groups: tuple[tuple[str, ...], ...]

    @property
    def status(self) -> str:
        if self.available is None:
            return "not_probed"
        return "available" if self.available else "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "status": self.status,
            "available": self.available,
            "resolved_apis": list(self.resolved_apis),
            "missing_api_groups": [list(group) for group in self.missing_api_groups],
        }


CodegenHook = Callable[[dict[str, Any]], None]
LegalityPredicate = Callable[[Mapping[str, Any]], tuple[TemplateConstraint, ...]]
ReferenceEvaluator = Callable[..., Any]


@dataclass(frozen=True)
class TemplateDefinition:
    name: str
    axis: str
    is_default: bool
    required_api_groups: tuple[tuple[str, ...], ...]
    operation_sequence: tuple[str, ...]
    reference_semantics: str
    config_fields: tuple[str, ...]
    launch_count: int
    intermediate_count: int
    codegen_hook: CodegenHook
    legality_predicate: LegalityPredicate
    reference_evaluator: ReferenceEvaluator

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TEMPLATE_REGISTRY_SCHEMA_VERSION,
            "name": self.name,
            "axis": self.axis,
            "is_default": self.is_default,
            "availability": {
                "required_api_groups": [
                    list(group) for group in self.required_api_groups
                ]
            },
            "legality_predicate": self.legality_predicate.__name__,
            "codegen_hook": self.codegen_hook.__name__,
            "config_schema": {
                "axis": self.axis,
                "enum_value": self.name,
                "runtime_fields": list(self.config_fields),
            },
            "reference_semantics": self.reference_semantics,
            "operation_sequence": list(self.operation_sequence),
            "cost_metadata": {
                "launch_count": self.launch_count,
                "intermediate_count": self.intermediate_count,
            },
        }


def list_template_definitions() -> tuple[TemplateDefinition, ...]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def get_template_definition(name: str) -> TemplateDefinition:
    canonical = canonical_template_name(name)
    try:
        return _REGISTRY[canonical]
    except KeyError as exc:
        raise TemplateSelectionError(f"unknown template: {name!r}") from exc


def template_registry_schema() -> dict[str, Any]:
    return {
        "schema_version": TEMPLATE_REGISTRY_SCHEMA_VERSION,
        "axes": {
            axis: {
                "default": DEFAULT_TEMPLATE_SELECTION[axis],
                "enum": [
                    definition.name
                    for definition in list_template_definitions()
                    if definition.axis == axis
                ],
            }
            for axis in DEFAULT_TEMPLATE_SELECTION
        },
        "templates": [
            definition.to_dict() for definition in list_template_definitions()
        ],
    }


def canonical_template_name(name: str, axis: str | None = None) -> str:
    value = str(name)
    if axis is not None:
        value = _ALIASES.get(axis, {}).get(value, value)
        definition = _REGISTRY.get(value)
        if definition is None or definition.axis != axis:
            raise TemplateSelectionError(
                f"template axis {axis!r} does not accept {name!r}"
            )
        return value
    for aliases in _ALIASES.values():
        value = aliases.get(value, value)
    if value not in _REGISTRY:
        raise TemplateSelectionError(f"unknown template: {name!r}")
    return value


def normalize_template_selection(
    selection: Mapping[str, Any],
    *,
    require_all: bool = True,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for axis, default in DEFAULT_TEMPLATE_SELECTION.items():
        if axis not in selection:
            if require_all:
                raise TemplateSelectionError(f"template choices are missing: {axis!r}")
            normalized[axis] = default
            continue
        normalized[axis] = canonical_template_name(str(selection[axis]), axis)
    for key, value in selection.items():
        if key in DEFAULT_TEMPLATE_SELECTION:
            continue
        if not isinstance(value, str) or not value:
            raise TemplateSelectionError(
                f"template choice {key!r} must be a non-empty string"
            )
        normalized[str(key)] = str(value)
    return normalized


def template_choice_from_runtime_config(
    runtime_config: Mapping[str, Any],
    axis: str,
) -> str:
    default = DEFAULT_TEMPLATE_SELECTION[axis]
    autotune = runtime_config.get("autotune")
    if not isinstance(autotune, Mapping):
        template_config = runtime_config.get("template_config")
        if isinstance(template_config, Mapping):
            autotune = template_config.get("autotune")
    templates = autotune.get("templates") if isinstance(autotune, Mapping) else None
    if isinstance(templates, Mapping) and axis in templates:
        return canonical_template_name(str(templates[axis]), axis)
    return default


def validate_template_selection(
    selection: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None = None,
) -> tuple[TemplateConstraint, ...]:
    try:
        normalized = normalize_template_selection(selection)
    except TemplateSelectionError as exc:
        return (
            TemplateConstraint(
                code="TEMPLATE_SELECTION_INVALID",
                path="templates",
                message=str(exc),
            ),
        )
    constraints: list[TemplateConstraint] = []
    if (
        normalized[GATE_UP_AXIS] == PACKED_GATE_UP
        and normalized[ACTIVATION_AXIS] == GATE_LINEAR_FUSED_SILU
    ):
        constraints.append(
            TemplateConstraint(
                code="PACKED_GATE_UP_PARTIAL_ACTIVATION",
                path=f"templates.{ACTIVATION_AXIS}",
                message=(
                    "packed gate/up cannot fuse SILU into the shared linear "
                    "because activation must apply only to the gate half"
                ),
            )
        )
    config = runtime_config or {}
    for axis in DEFAULT_TEMPLATE_SELECTION:
        definition = _REGISTRY[normalized[axis]]
        constraints.extend(definition.legality_predicate(config))
    return tuple(constraints)


def apply_template_selection(
    runtime_config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_template_selection(selection)
    constraints = validate_template_selection(normalized, runtime_config)
    if constraints:
        detail = "; ".join(
            f"{constraint.code}: {constraint.message}" for constraint in constraints
        )
        raise TemplateSelectionError(detail)
    result = copy.deepcopy(dict(runtime_config))
    autotune = result.get("autotune")
    if not isinstance(autotune, dict):
        autotune = {}
        result["autotune"] = autotune
    autotune["templates"] = copy.deepcopy(normalized)
    template_config = result.get("template_config")
    if isinstance(template_config, dict):
        template_autotune = template_config.get("autotune")
        if not isinstance(template_autotune, dict):
            template_autotune = {}
            template_config["autotune"] = template_autotune
        template_autotune["templates"] = copy.deepcopy(normalized)
    for axis in DEFAULT_TEMPLATE_SELECTION:
        _REGISTRY[normalized[axis]].codegen_hook(result)
    return result


def apply_template_axis_updates(
    runtime_config: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = set(updates).difference(DEFAULT_TEMPLATE_SELECTION)
    if unknown:
        raise TemplateSelectionError(
            "unknown template axis: " + ", ".join(sorted(unknown))
        )
    selection = {
        axis: template_choice_from_runtime_config(runtime_config, axis)
        for axis in DEFAULT_TEMPLATE_SELECTION
    }
    selection.update(updates)
    normalized = normalize_template_selection(selection)
    constraints = validate_template_selection(normalized, runtime_config)
    if constraints:
        detail = "; ".join(
            f"{constraint.code}: {constraint.message}" for constraint in constraints
        )
        raise TemplateSelectionError(detail)

    result = copy.deepcopy(dict(runtime_config))
    autotune = result.get("autotune")
    if not isinstance(autotune, dict):
        autotune = {}
        result["autotune"] = autotune
    autotune["templates"] = copy.deepcopy(normalized)
    template_config = result.get("template_config")
    if isinstance(template_config, dict):
        template_autotune = template_config.get("autotune")
        if not isinstance(template_autotune, dict):
            template_autotune = {}
            template_config["autotune"] = template_autotune
        template_autotune["templates"] = copy.deepcopy(normalized)
    for axis in DEFAULT_TEMPLATE_SELECTION:
        if axis in updates:
            _REGISTRY[normalized[axis]].codegen_hook(result)
    return result


def probe_template_availability(
    name: str,
    ttnn_module: Any | None,
) -> TemplateAvailability:
    definition = get_template_definition(name)
    if ttnn_module is None:
        return TemplateAvailability(
            template=definition.name,
            available=None,
            resolved_apis=(),
            missing_api_groups=(),
        )
    resolved: list[str] = []
    missing: list[tuple[str, ...]] = []
    for alternatives in definition.required_api_groups:
        selected = next(
            (
                path
                for path in alternatives
                if _resolve_api_path(ttnn_module, path) is not None
            ),
            None,
        )
        if selected is None:
            missing.append(alternatives)
        else:
            resolved.append(selected)
    return TemplateAvailability(
        template=definition.name,
        available=not missing,
        resolved_apis=tuple(resolved),
        missing_api_groups=tuple(missing),
    )


def dry_run_template(
    name: str,
    runtime_config: Mapping[str, Any],
    *,
    ttnn_module: Any | None = None,
) -> dict[str, Any]:
    definition = get_template_definition(name)
    selection = dict(DEFAULT_TEMPLATE_SELECTION)
    selection[definition.axis] = definition.name
    configured = apply_template_selection(runtime_config, selection)
    availability = probe_template_availability(definition.name, ttnn_module)
    constraints = validate_template_selection(selection, configured)
    return {
        "schema_version": TEMPLATE_REGISTRY_SCHEMA_VERSION,
        "status": "dry_run" if not constraints else "rejected",
        "dry_run": True,
        "template": definition.name,
        "axis": definition.axis,
        "selection": selection,
        "availability": availability.to_dict(),
        "legality": {
            "passed": not constraints,
            "constraints": [item.to_dict() for item in constraints],
        },
        "codegen": {
            "hook": definition.codegen_hook.__name__,
            "runtime_fields": list(definition.config_fields),
            "op_sequence": list(definition.operation_sequence),
        },
        "reference_semantics": definition.reference_semantics,
        "configured_runtime": configured,
    }


def evaluate_template_reference(name: str, **inputs: Any) -> Any:
    return get_template_definition(name).reference_evaluator(**inputs)


def _resolve_api_path(owner: Any, path: str) -> Any | None:
    value = owner
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def _always_legal(_: Mapping[str, Any]) -> tuple[TemplateConstraint, ...]:
    return ()


def _packed_gate_up_legal(
    config: Mapping[str, Any],
) -> tuple[TemplateConstraint, ...]:
    if not config:
        return ()
    hidden = int(config.get("hidden_size", 0) or 0)
    intermediate = int(config.get("intermediate_size", 0) or 0)
    constraints: list[TemplateConstraint] = []
    if hidden <= 0 or intermediate <= 0:
        constraints.append(
            TemplateConstraint(
                code="PACKED_GATE_UP_SHAPE_MISSING",
                path="templates.mlp.gate_up",
                message="packed gate/up requires hidden and intermediate sizes",
                error_class="shape_incompatible",
            )
        )
    elif (2 * intermediate) % 32:
        constraints.append(
            TemplateConstraint(
                code="PACKED_GATE_UP_TILE_ALIGNMENT",
                path="templates.mlp.gate_up",
                message="packed gate/up output width must be tile aligned",
                error_class="shape_incompatible",
            )
        )
    mlp = config.get("mlp")
    if isinstance(mlp, Mapping):
        split_strategy = str(
            mlp.get("packed_gate_up_split_strategy", "split")
        )
        if split_strategy not in {"split", "slice"}:
            constraints.append(
                TemplateConstraint(
                    code="PACKED_GATE_UP_SPLIT_STRATEGY",
                    path="mlp.packed_gate_up_split_strategy",
                    message=(
                        "packed gate/up split strategy must be 'split' or 'slice'"
                    ),
                    error_class="invalid_program_config",
                )
            )
        gate_program = mlp.get("gate_program_config")
        up_program = mlp.get("up_program_config")
        if (
            gate_program is not None
            and up_program is not None
            and gate_program != up_program
        ):
            constraints.append(
                TemplateConstraint(
                    code="PACKED_GATE_UP_PROGRAM_MISMATCH",
                    path="templates.mlp.gate_up",
                    message="gate and up program configs must match before packing",
                )
            )
    return tuple(constraints)


def _gate_linear_fused_silu_legal(
    config: Mapping[str, Any],
) -> tuple[TemplateConstraint, ...]:
    if not config:
        return ()
    constraints: list[TemplateConstraint] = []
    for section_name in ("mlp", "prefill"):
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            continue
        program = section.get("gate_program_config")
        if not isinstance(program, Mapping):
            continue
        kind = str(program.get("kind", ""))
        if kind not in _FUSED_ACTIVATION_PROGRAM_KINDS:
            constraints.append(
                TemplateConstraint(
                    code="GATE_LINEAR_FUSED_SILU_PROGRAM_UNSUPPORTED",
                    path=f"{section_name}.gate_program_config",
                    message=(
                        f"program config {kind!r} does not expose " "fused_activation"
                    ),
                )
            )
    return tuple(constraints)


def _normalize_attention_op(
    config: dict[str, Any],
    *,
    old: tuple[str, ...],
    new: str,
) -> None:
    attention = config.setdefault("attention", {})
    operations = list(attention.get("op_sequence") or [])
    replacement_index = next(
        (index for index, operation in enumerate(operations) if operation in old),
        None,
    )
    operations = [operation for operation in operations if operation not in old]
    if replacement_index is not None:
        operations.insert(min(replacement_index, len(operations)), new)
    attention["op_sequence"] = operations


def _apply_separate_paged_update(config: dict[str, Any]) -> None:
    _normalize_attention_op(
        config,
        old=("paged_update_cache", "paged_fused_update_cache"),
        new="paged_update_cache",
    )
    attention = config.setdefault("attention", {})
    attention.pop("fused_cache_key_memory_config", None)
    attention.pop("fused_cache_value_memory_config", None)


def _apply_fused_paged_update(config: dict[str, Any]) -> None:
    _normalize_attention_op(
        config,
        old=("paged_update_cache", "paged_fused_update_cache"),
        new="paged_fused_update_cache",
    )
    attention = config.setdefault("attention", {})
    head_dim = int(config.get("head_dim", 128) or 128)
    attention["fused_cache_key_memory_config"] = _height_sharded_config(
        [[0, 0, 7, 3]], [32, head_dim]
    )
    attention["fused_cache_value_memory_config"] = _height_sharded_config(
        [[0, 4, 7, 7]], [32, head_dim]
    )


def _apply_separate_qk_rope(config: dict[str, Any]) -> None:
    _normalize_attention_op(
        config,
        old=("rotary_embedding_decode", "rotary_embedding_llama_fused_qk"),
        new="rotary_embedding_decode",
    )
    attention = config.setdefault("attention", {})
    for field in (
        "fused_q_memory_config",
        "fused_k_memory_config",
        "fused_rope_cos_sin_memory_config",
        "fused_rope_transform_memory_config",
    ):
        attention.pop(field, None)


def _apply_fused_qk_rope(config: dict[str, Any]) -> None:
    _normalize_attention_op(
        config,
        old=("rotary_embedding_decode", "rotary_embedding_llama_fused_qk"),
        new="rotary_embedding_llama_fused_qk",
    )
    attention = config.setdefault("attention", {})
    head_dim = int(config.get("head_dim", 128) or 128)
    q_memory = _height_sharded_config([[0, 0, 7, 3]], [32, head_dim])
    k_memory = _height_sharded_config([[0, 4, 7, 7]], [32, head_dim])
    combined = [[0, 0, 7, 3], [0, 4, 7, 7]]
    attention["fused_q_memory_config"] = q_memory
    attention["fused_k_memory_config"] = k_memory
    attention["fused_rope_cos_sin_memory_config"] = _height_sharded_config(
        combined, [32, head_dim]
    )
    attention["fused_rope_transform_memory_config"] = _height_sharded_config(
        combined, [32, 32]
    )


def _apply_mul_fused_silu(config: dict[str, Any]) -> None:
    config.setdefault("mlp", {}).pop("gate_linear_activation", None)
    _set_gate_fused_activation(config, None)


def _apply_gate_linear_fused_silu(config: dict[str, Any]) -> None:
    config.setdefault("mlp", {})["gate_linear_activation"] = "silu"
    _set_gate_fused_activation(config, "silu")


def _set_gate_fused_activation(config: dict[str, Any], value: str | None) -> None:
    for section_name in ("mlp", "prefill"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        program = section.get("gate_program_config")
        if not isinstance(program, Mapping):
            continue
        descriptor = copy.deepcopy(dict(program))
        if value is None:
            descriptor.pop("fused_activation", None)
        elif str(descriptor.get("kind", "")) in _FUSED_ACTIVATION_PROGRAM_KINDS:
            descriptor["fused_activation"] = value
        section["gate_program_config"] = descriptor


def _apply_separate_gate_up(config: dict[str, Any]) -> None:
    mlp = config.setdefault("mlp", {})
    for field in (
        "packed_gate_up_program_config",
        "packed_gate_up_output_memory_config",
        "packed_gate_up_split_strategy",
        "packed_gate_up_split_output_memory_config",
        "packed_gate_up_mul_input_memory_config",
        "packed_gate_up_mul_conversion",
    ):
        mlp.pop(field, None)


def _apply_packed_gate_up(config: dict[str, Any]) -> None:
    mlp = config.setdefault("mlp", {})
    gate_program = mlp.get("gate_program_config")
    if mlp.get("packed_gate_up_program_config") is None and isinstance(
        gate_program, Mapping
    ):
        packed_program = copy.deepcopy(dict(gate_program))
        for key in ("per_core_N", "per_core_n", "out_block_w"):
            if key in packed_program:
                packed_program[key] = 2 * int(packed_program[key])
        mlp["packed_gate_up_program_config"] = packed_program
    elif "packed_gate_up_program_config" not in mlp:
        mlp["packed_gate_up_program_config"] = gate_program
    mlp.setdefault(
        "packed_gate_up_output_memory_config",
        copy.deepcopy(mlp.get("gate_output_memory_config")),
    )
    mlp.setdefault("packed_gate_up_split_strategy", "split")
    mlp.setdefault(
        "packed_gate_up_split_output_memory_config",
        {
            "kind": "ttnn_memory_config",
            "name": "L1_MEMORY_CONFIG",
        },
    )
    mlp.setdefault(
        "packed_gate_up_mul_input_memory_config",
        {
            "kind": "ttnn_memory_config",
            "name": "L1_MEMORY_CONFIG",
        },
    )
    mlp.setdefault("packed_gate_up_mul_conversion", False)


def _height_sharded_config(
    core_ranges: list[list[int]],
    shard_shape: list[int],
) -> dict[str, Any]:
    core_count = sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in core_ranges)
    return {
        "kind": "ttnn_sharded_memory_config",
        "strategy": "height",
        "core_grid": [8, core_count // 8],
        "core_ranges": core_ranges,
        "shard_shape": shard_shape,
        "orientation": "row_major",
    }


def _cache_update_reference(
    *,
    key_cache: Any,
    value_cache: Any,
    key: Any,
    value: Any,
    positions: Any,
    **_: Any,
) -> tuple[Any, Any]:
    result_key = key_cache.clone()
    result_value = value_cache.clone()
    for user, position in enumerate(positions):
        result_key[user, :, int(position), :] = key[0, user]
        result_value[user, :, int(position), :] = value[0, user]
    return result_key, result_value


def _rope_reference(
    *,
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    transformation: Any,
    rope: Callable[..., Any],
    **_: Any,
) -> tuple[Any, Any]:
    return (
        rope(q, cos, sin, transformation),
        rope(k, cos, sin, transformation),
    )


def _mlp_activation_reference(
    *,
    hidden: Any,
    gate_weight: Any,
    up_weight: Any,
    silu: Callable[[Any], Any],
    **_: Any,
) -> Any:
    return silu(hidden @ gate_weight) * (hidden @ up_weight)


def _separate_gate_up_reference(
    *, hidden: Any, gate_weight: Any, up_weight: Any, **_: Any
) -> tuple[Any, Any]:
    return hidden @ gate_weight, hidden @ up_weight


def _packed_gate_up_reference(
    *,
    hidden: Any,
    gate_weight: Any,
    up_weight: Any,
    cat: Callable[..., Any],
    **_: Any,
) -> tuple[Any, Any]:
    width = int(gate_weight.shape[-1])
    packed = hidden @ cat((gate_weight, up_weight), dim=-1)
    return packed[..., :width], packed[..., width:]


def _definition(
    *,
    name: str,
    axis: str,
    required_api_groups: tuple[tuple[str, ...], ...],
    operation_sequence: tuple[str, ...],
    reference_semantics: str,
    config_fields: tuple[str, ...],
    launch_count: int,
    intermediate_count: int,
    codegen_hook: CodegenHook,
    legality_predicate: LegalityPredicate = _always_legal,
    reference_evaluator: ReferenceEvaluator,
) -> TemplateDefinition:
    return TemplateDefinition(
        name=name,
        axis=axis,
        is_default=DEFAULT_TEMPLATE_SELECTION[axis] == name,
        required_api_groups=required_api_groups,
        operation_sequence=operation_sequence,
        reference_semantics=reference_semantics,
        config_fields=config_fields,
        launch_count=launch_count,
        intermediate_count=intermediate_count,
        codegen_hook=codegen_hook,
        legality_predicate=legality_predicate,
        reference_evaluator=reference_evaluator,
    )


_REGISTRY = {
    definition.name: definition
    for definition in (
        _definition(
            name=SEPARATE_PAGED_UPDATE,
            axis=KV_UPDATE_AXIS,
            required_api_groups=(("experimental.paged_update_cache",),),
            operation_sequence=("paged_update_cache.k", "paged_update_cache.v"),
            reference_semantics="K and V caches receive identical indexed writes",
            config_fields=(),
            launch_count=2,
            intermediate_count=0,
            codegen_hook=_apply_separate_paged_update,
            reference_evaluator=_cache_update_reference,
        ),
        _definition(
            name=FUSED_PAGED_UPDATE,
            axis=KV_UPDATE_AXIS,
            required_api_groups=(("experimental.paged_fused_update_cache",),),
            operation_sequence=("paged_fused_update_cache.kv",),
            reference_semantics="K and V caches receive identical indexed writes",
            config_fields=(
                "attention.fused_cache_key_memory_config",
                "attention.fused_cache_value_memory_config",
            ),
            launch_count=1,
            intermediate_count=0,
            codegen_hook=_apply_fused_paged_update,
            reference_evaluator=_cache_update_reference,
        ),
        _definition(
            name=SEPARATE_QK_ROPE,
            axis=ROPE_AXIS,
            required_api_groups=(("experimental.rotary_embedding_llama",),),
            operation_sequence=("rotary_embedding.q", "rotary_embedding.k"),
            reference_semantics="apply the same Llama RoPE transform to Q and K",
            config_fields=(),
            launch_count=2,
            intermediate_count=0,
            codegen_hook=_apply_separate_qk_rope,
            reference_evaluator=_rope_reference,
        ),
        _definition(
            name=FUSED_QK_ROPE,
            axis=ROPE_AXIS,
            required_api_groups=(("experimental.rotary_embedding_llama_fused_qk",),),
            operation_sequence=("rotary_embedding_llama_fused_qk",),
            reference_semantics="apply the same Llama RoPE transform to Q and K",
            config_fields=(
                "attention.fused_q_memory_config",
                "attention.fused_k_memory_config",
                "attention.fused_rope_cos_sin_memory_config",
                "attention.fused_rope_transform_memory_config",
            ),
            launch_count=1,
            intermediate_count=0,
            codegen_hook=_apply_fused_qk_rope,
            reference_evaluator=_rope_reference,
        ),
        _definition(
            name=MUL_FUSED_SILU,
            axis=ACTIVATION_AXIS,
            required_api_groups=(
                (("mul", "multiply")),
                ("UnaryWithParam",),
                ("UnaryOpType.SILU",),
            ),
            operation_sequence=("linear.gate", "linear.up", "mul[input_a=SILU]"),
            reference_semantics="silu(hidden @ gate_weight) * (hidden @ up_weight)",
            config_fields=(),
            launch_count=3,
            intermediate_count=2,
            codegen_hook=_apply_mul_fused_silu,
            reference_evaluator=_mlp_activation_reference,
        ),
        _definition(
            name=GATE_LINEAR_FUSED_SILU,
            axis=ACTIVATION_AXIS,
            required_api_groups=(
                ("linear",),
                ("mul", "multiply"),
                ("UnaryOpType.SILU",),
            ),
            operation_sequence=("linear.gate[activation=SILU]", "linear.up", "mul"),
            reference_semantics="silu(hidden @ gate_weight) * (hidden @ up_weight)",
            config_fields=(
                "mlp.gate_linear_activation",
                "mlp.gate_program_config.fused_activation",
                "prefill.gate_program_config.fused_activation",
            ),
            launch_count=3,
            intermediate_count=2,
            codegen_hook=_apply_gate_linear_fused_silu,
            legality_predicate=_gate_linear_fused_silu_legal,
            reference_evaluator=_mlp_activation_reference,
        ),
        _definition(
            name=SEPARATE_GATE_UP,
            axis=GATE_UP_AXIS,
            required_api_groups=(("linear",),),
            operation_sequence=("linear.gate", "linear.up"),
            reference_semantics="return hidden @ gate_weight and hidden @ up_weight",
            config_fields=(),
            launch_count=2,
            intermediate_count=2,
            codegen_hook=_apply_separate_gate_up,
            reference_evaluator=_separate_gate_up_reference,
        ),
        _definition(
            name=PACKED_GATE_UP,
            axis=GATE_UP_AXIS,
            required_api_groups=(("linear",), ("split", "slice")),
            operation_sequence=("linear.gate_up_packed", "split.gate_up"),
            reference_semantics=(
                "concat gate/up weights, run one linear, then split into the "
                "same gate and up projections"
            ),
            config_fields=(
                "mlp.packed_gate_up_program_config",
                "mlp.packed_gate_up_output_memory_config",
                "mlp.packed_gate_up_split_strategy",
                "mlp.packed_gate_up_split_output_memory_config",
                "mlp.packed_gate_up_mul_input_memory_config",
                "mlp.packed_gate_up_mul_conversion",
            ),
            launch_count=2,
            intermediate_count=3,
            codegen_hook=_apply_packed_gate_up,
            legality_predicate=_packed_gate_up_legal,
            reference_evaluator=_packed_gate_up_reference,
        ),
    )
}
