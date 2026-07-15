from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..dtype_recipes import PERFORMANCE_RECIPE

AUTOTUNE_SCHEMA_VERSION = 2
CONTRACT_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_FROZEN_EXECUTION = {
    "execution_mode": "trace",
    "runtime_input_mode": "persistent",
    "after_prefill": True,
    "sampling": "force_argmax",
    "page_table": "fixed",
}
_PRECISION_KEY_MARKERS = (
    "dtype",
    "precision",
    "math_fidelity",
    "fp32_dest_acc",
    "math_approx",
    "exp_approx",
    "packer_l1_acc",
)


class ContractViolation(ValueError):
    """Raised before measurement when a candidate breaks a frozen contract."""


@dataclass(frozen=True)
class PrecisionContract:
    recipe: str
    source_profile: str
    source_sha256: str
    _fields_json: str = field(repr=False)
    schema_version: int = CONTRACT_SCHEMA_VERSION
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractViolation(
                f"precision schema_version must be {CONTRACT_SCHEMA_VERSION}"
            )
        if self.recipe != PERFORMANCE_RECIPE:
            raise ContractViolation(
                "autotune precision recipe is frozen to "
                f"{PERFORMANCE_RECIPE!r}; observed {self.recipe!r}"
            )
        if not self.source_profile:
            raise ContractViolation("precision source_profile must be non-empty")
        _validate_sha256("precision source_sha256", self.source_sha256)
        if not self.frozen:
            raise ContractViolation("precision contract must be frozen")
        fields = _decode_mapping(self._fields_json, "precision fields")
        if not fields:
            raise ContractViolation("precision fields must be non-empty")

    @classmethod
    def from_template_config(
        cls, template_config: Mapping[str, Any]
    ) -> "PrecisionContract":
        from ..compiler.official_config import load_official_config_profile

        recipe = str(template_config.get("dtype_recipe", ""))
        profile = str(template_config.get("official_config_profile", ""))
        if recipe != PERFORMANCE_RECIPE:
            raise ContractViolation(
                "semantic autotune requires the production performance recipe; "
                f"observed {recipe!r}"
            )
        if not profile:
            raise ContractViolation(
                "semantic autotune requires an official_config_profile"
            )
        official = load_official_config_profile(profile)
        dtype_recipe = (official.get("parity_config") or {}).get("dtype_recipe")
        if not isinstance(dtype_recipe, dict):
            raise ContractViolation(
                "official profile does not contain parity_config.dtype_recipe"
            )
        fields = {
            "dtype_recipe": dtype_recipe,
            "runtime_precision_fields": _collect_precision_fields(
                official.get("runtime_config") or {}
            ),
        }
        source_sha256 = _sha256_json(official)
        return cls(
            recipe=recipe,
            source_profile=profile,
            source_sha256=source_sha256,
            _fields_json=_canonical_json(fields),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrecisionContract":
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            raise ContractViolation("precision contract fields must be an object")
        contract = cls(
            recipe=str(payload.get("recipe", "")),
            source_profile=str(payload.get("source_profile", "")),
            source_sha256=str(payload.get("source_sha256", "")),
            _fields_json=_canonical_json(fields),
            schema_version=int(payload.get("schema_version", CONTRACT_SCHEMA_VERSION)),
            frozen=bool(payload.get("frozen", False)),
        )
        supplied_hash = payload.get("hash")
        if supplied_hash is not None and supplied_hash != contract.hash:
            raise ContractViolation(
                "precision contract hash does not match its frozen fields"
            )
        return contract

    @property
    def fields(self) -> dict[str, Any]:
        return _decode_mapping(self._fields_json, "precision fields")

    @property
    def hash(self) -> str:
        return _sha256_json(self._identity_payload())

    def assert_hash(self, expected_hash: str) -> None:
        if self.hash != expected_hash:
            raise ContractViolation(
                "candidate precision hash differs from the frozen baseline: "
                f"expected {expected_hash}, observed {self.hash}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "hash": self.hash}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe": self.recipe,
            "source_profile": self.source_profile,
            "source_sha256": self.source_sha256,
            "frozen": self.frozen,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class ExecutionContract:
    prompt_corpus_sha256: str
    execution_mode: str = "trace"
    runtime_input_mode: str = "persistent"
    after_prefill: bool = True
    sampling: str = "force_argmax"
    page_table: str = "fixed"
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractViolation(
                f"execution schema_version must be {CONTRACT_SCHEMA_VERSION}"
            )
        _validate_sha256("prompt_corpus_sha256", self.prompt_corpus_sha256)
        observed = {
            "execution_mode": self.execution_mode,
            "runtime_input_mode": self.runtime_input_mode,
            "after_prefill": self.after_prefill,
            "sampling": self.sampling,
            "page_table": self.page_table,
        }
        mismatches = {
            key: {"expected": expected, "observed": observed[key]}
            for key, expected in _FROZEN_EXECUTION.items()
            if observed[key] != expected
        }
        if mismatches:
            raise ContractViolation(
                "candidate changed the production execution contract: "
                + _canonical_json(mismatches)
            )

    @classmethod
    def from_template_config(
        cls,
        template_config: Mapping[str, Any],
        *,
        prompt_corpus_sha256: str,
    ) -> "ExecutionContract":
        expected = {
            "runtime_input_mode": "persistent",
            "generation_template": "device_argmax_greedy",
            "lm_head_argmax_strategy": ("full_logits_untilize_multicore_argmax"),
        }
        mismatches = {
            key: {
                "expected": value,
                "observed": template_config.get(key),
            }
            for key, value in expected.items()
            if template_config.get(key) != value
        }
        if mismatches:
            raise ContractViolation(
                "baseline does not satisfy the production execution contract: "
                + _canonical_json(mismatches)
            )
        return cls(prompt_corpus_sha256=prompt_corpus_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "runtime_input_mode": self.runtime_input_mode,
            "after_prefill": self.after_prefill,
            "sampling": self.sampling,
            "page_table": self.page_table,
            "prompt_corpus_sha256": self.prompt_corpus_sha256,
        }


@dataclass(frozen=True)
class MeasurementContract:
    warmup: int
    iterations: int
    repetitions: int = 1
    kind: str = "candidate_search"
    scope: str = "post_prefill_steady_decode"
    metric: str = "tokens_per_second_per_user"
    synchronize_device: bool = True
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractViolation(
                f"measurement schema_version must be {CONTRACT_SCHEMA_VERSION}"
            )
        if self.warmup < 0:
            raise ContractViolation("measurement warmup must be non-negative")
        if self.iterations <= 0:
            raise ContractViolation("measurement iterations must be positive")
        if self.repetitions <= 0:
            raise ContractViolation("measurement repetitions must be positive")
        if self.scope != "post_prefill_steady_decode":
            raise ContractViolation(
                "autotune measurement scope must be post_prefill_steady_decode"
            )
        if self.metric != "tokens_per_second_per_user":
            raise ContractViolation(
                "autotune metric must be tokens_per_second_per_user"
            )
        if not self.synchronize_device:
            raise ContractViolation("device synchronization cannot be disabled")

    @classmethod
    def final_confirmation(cls) -> "MeasurementContract":
        return cls(
            warmup=5,
            iterations=100,
            repetitions=3,
            kind="final_confirmation",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "scope": self.scope,
            "metric": self.metric,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "repetitions": self.repetitions,
            "synchronize_device": self.synchronize_device,
        }


@dataclass(frozen=True)
class CandidateConfig:
    precision_contract: PrecisionContract
    expected_precision_hash: str
    execution_contract: ExecutionContract
    measurement_contract: MeasurementContract
    semantic_graph_sha256: str
    model_config_sha256: str
    weights_recipe_sha256: str
    runtime_commit: str
    _device_descriptor_json: str = field(repr=False)
    _target_json: str = field(repr=False)
    _tunable_state_json: str = field(repr=False)
    schema_version: int = AUTOTUNE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUTOTUNE_SCHEMA_VERSION:
            raise ContractViolation(
                f"candidate schema_version must be {AUTOTUNE_SCHEMA_VERSION}"
            )
        self.precision_contract.assert_hash(self.expected_precision_hash)
        for label, value in (
            ("semantic_graph_sha256", self.semantic_graph_sha256),
            ("model_config_sha256", self.model_config_sha256),
            ("weights_recipe_sha256", self.weights_recipe_sha256),
        ):
            _validate_sha256(label, value)
        if not self.runtime_commit.strip():
            raise ContractViolation("runtime_commit must be non-empty")
        device = _decode_mapping(self._device_descriptor_json, "device descriptor")
        target = _decode_mapping(self._target_json, "candidate target")
        state = _decode_mapping(self._tunable_state_json, "tunable state")
        if not device:
            raise ContractViolation("device descriptor must be non-empty")
        for key in ("batch_size", "cache_len", "page_block_size"):
            if key not in target:
                raise ContractViolation(f"candidate target is missing {key}")
        _reject_precision_mutations(state)

    @classmethod
    def create(
        cls,
        *,
        precision_contract: PrecisionContract,
        expected_precision_hash: str,
        execution_contract: ExecutionContract,
        measurement_contract: MeasurementContract,
        semantic_graph_sha256: str,
        model_config_sha256: str,
        weights_recipe_sha256: str,
        runtime_commit: str,
        device_descriptor: Mapping[str, Any],
        target: Mapping[str, Any],
        tunable_state: Mapping[str, Any],
    ) -> "CandidateConfig":
        return cls(
            precision_contract=precision_contract,
            expected_precision_hash=expected_precision_hash,
            execution_contract=execution_contract,
            measurement_contract=measurement_contract,
            semantic_graph_sha256=semantic_graph_sha256,
            model_config_sha256=model_config_sha256,
            weights_recipe_sha256=weights_recipe_sha256,
            runtime_commit=runtime_commit,
            _device_descriptor_json=_canonical_json(device_descriptor),
            _target_json=_canonical_json(target),
            _tunable_state_json=_canonical_json(tunable_state),
        )

    @property
    def device_descriptor(self) -> dict[str, Any]:
        return _decode_mapping(self._device_descriptor_json, "device descriptor")

    @property
    def target(self) -> dict[str, Any]:
        return _decode_mapping(self._target_json, "candidate target")

    @property
    def tunable_state(self) -> dict[str, Any]:
        return _decode_mapping(self._tunable_state_json, "tunable state")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "precision_contract": self.precision_contract.to_dict(),
            "execution_contract": self.execution_contract.to_dict(),
            "measurement_contract": self.measurement_contract.to_dict(),
            "semantic_graph_sha256": self.semantic_graph_sha256,
            "model_config_sha256": self.model_config_sha256,
            "weights_recipe_sha256": self.weights_recipe_sha256,
            "runtime_commit": self.runtime_commit,
            "device_descriptor": self.device_descriptor,
            "target": self.target,
            "tunable_state": self.tunable_state,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()


def canonical_json(value: Any) -> str:
    return _canonical_json(value)


def sha256_json(value: Any) -> str:
    return _sha256_json(value)


def _collect_precision_fields(value: Any, path: tuple[str, ...] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return result
    for raw_key, child in value.items():
        key = str(raw_key)
        child_path = (*path, key)
        normalized = key.lower()
        if any(marker in normalized for marker in _PRECISION_KEY_MARKERS):
            result[".".join(child_path)] = child
            continue
        result.update(_collect_precision_fields(child, child_path))
    return result


def _reject_precision_mutations(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_path = (*path, key)
            if any(marker in normalized for marker in _PRECISION_KEY_MARKERS):
                raise ContractViolation(
                    "candidate tunable state contains frozen precision field: "
                    + ".".join(child_path)
                )
            _reject_precision_mutations(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_precision_mutations(child, (*path, str(index)))


def _validate_sha256(label: str, value: str) -> None:
    if len(value) != _SHA256_LENGTH:
        raise ContractViolation(f"{label} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractViolation(f"{label} must be a SHA256 digest") from exc


def _decode_mapping(value: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractViolation(f"{label} must be a JSON object")
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
