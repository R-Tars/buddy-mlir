from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import sha256_json

TRANSFER_SCHEMA_VERSION = 1
DEFAULT_LAYER_GROUP = "group_default"
OVERRIDE_LAYER_GROUP = "group_override"


class LayerTransferError(ValueError):
    """Raised when representative-layer transfer is incomplete or ambiguous."""


@dataclass(frozen=True)
class LayerGroup:
    name: str
    representative_layer: int
    member_layers: tuple[int, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.name:
            raise LayerTransferError("layer group name must be non-empty")
        if not self.member_layers:
            raise LayerTransferError(f"layer group {self.name!r} must not be empty")
        if len(set(self.member_layers)) != len(self.member_layers):
            raise LayerTransferError(
                f"layer group {self.name!r} contains duplicate layers"
            )
        if tuple(sorted(self.member_layers)) != self.member_layers:
            raise LayerTransferError(
                f"layer group {self.name!r} members must be sorted"
            )
        if self.representative_layer not in self.member_layers:
            raise LayerTransferError(
                f"representative layer {self.representative_layer} is not a "
                f"member of {self.name!r}"
            )
        if not self.reason:
            raise LayerTransferError(
                f"layer group {self.name!r} must explain its transfer boundary"
            )

    @property
    def transfer_enabled(self) -> bool:
        return len(self.member_layers) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "representative_layer": self.representative_layer,
            "member_layers": list(self.member_layers),
            "transfer_enabled": self.transfer_enabled,
            "transfer_count": len(self.member_layers) - 1,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LayerTransferPlan:
    model_family: str
    layer_count: int
    groups: tuple[LayerGroup, ...]
    schema_version: int = TRANSFER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSFER_SCHEMA_VERSION:
            raise LayerTransferError(
                f"transfer schema_version must be {TRANSFER_SCHEMA_VERSION}"
            )
        if not self.model_family:
            raise LayerTransferError("model_family must be non-empty")
        if self.layer_count <= 0:
            raise LayerTransferError("layer_count must be positive")
        if not self.groups:
            raise LayerTransferError("at least one layer group is required")
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise LayerTransferError("layer group names must be unique")
        members = [layer for group in self.groups for layer in group.member_layers]
        expected = list(range(self.layer_count))
        if sorted(members) != expected:
            raise LayerTransferError(
                "layer groups must partition every model layer exactly once; "
                f"expected {expected}, observed {sorted(members)}"
            )

    @property
    def representative_layers(self) -> tuple[int, ...]:
        return tuple(group.representative_layer for group in self.groups)

    def group(self, name: str) -> LayerGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise LayerTransferError(f"unknown layer group: {name!r}")

    def group_for_layer(self, layer: int) -> LayerGroup:
        for group in self.groups:
            if layer in group.member_layers:
                return group
        raise LayerTransferError(
            f"layer {layer} is outside transfer plan [0, {self.layer_count})"
        )

    def to_dict(self) -> dict[str, Any]:
        assignments = [
            {
                "layer": layer,
                "group": self.group_for_layer(layer).name,
                "representative_layer": self.group_for_layer(
                    layer
                ).representative_layer,
                "is_representative": (
                    self.group_for_layer(layer).representative_layer == layer
                ),
            }
            for layer in range(self.layer_count)
        ]
        return {
            "schema_version": self.schema_version,
            "strategy": "representative_layer_transfer",
            "model_family": self.model_family,
            "layer_count": self.layer_count,
            "representative_layers": list(self.representative_layers),
            "groups": [group.to_dict() for group in self.groups],
            "layer_assignments": assignments,
        }


def build_llama31_8b_transfer_plan(*, layer_count: int = 32) -> LayerTransferPlan:
    if layer_count != 32:
        raise LayerTransferError(
            "Llama 3.1 8B representative transfer requires exactly 32 layers"
        )
    return LayerTransferPlan(
        model_family="llama31_8b",
        layer_count=layer_count,
        groups=(
            LayerGroup(
                name=DEFAULT_LAYER_GROUP,
                representative_layer=0,
                member_layers=tuple(range(31)),
                reason="layers 0-30 share shapes and the default precision recipe",
            ),
            LayerGroup(
                name=OVERRIDE_LAYER_GROUP,
                representative_layer=31,
                member_layers=(31,),
                reason="layer 31 has the frozen MLP precision override",
            ),
        ),
    )


def transfer_representative_states(
    plan: LayerTransferPlan,
    representative_states: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    expected = {group.name for group in plan.groups}
    observed = {str(name) for name in representative_states}
    if observed != expected:
        raise LayerTransferError(
            "representative states must match layer groups exactly; "
            f"expected {sorted(expected)}, observed {sorted(observed)}"
        )

    layer_states: dict[int, dict[str, Any]] = {}
    group_reports: list[dict[str, Any]] = []
    for group in plan.groups:
        state = dict(representative_states[group.name])
        state_hash = sha256_json(state)
        for layer in group.member_layers:
            layer_states[layer] = copy.deepcopy(state)
        group_reports.append(
            {
                **group.to_dict(),
                "state_sha256": state_hash,
                "applied_layers": list(group.member_layers),
            }
        )

    report = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "status": "passed",
        "strategy": "representative_layer_transfer",
        "model_family": plan.model_family,
        "layer_count": plan.layer_count,
        "groups": group_reports,
        "layer_assignments": [
            {
                "layer": layer,
                "group": plan.group_for_layer(layer).name,
                "representative_layer": plan.group_for_layer(
                    layer
                ).representative_layer,
                "state_sha256": sha256_json(layer_states[layer]),
            }
            for layer in range(plan.layer_count)
        ],
    }
    return layer_states, report
