from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .schema import canonical_json, sha256_json

RANKING_SCHEMA_VERSION = 1
RANKING_DEFAULT_TOP_K = 5
RANKING_DEFAULT_RIDGE_STRENGTH = 0.10
RANKING_MINIMUM_MEASUREMENT_REDUCTION = 0.30
RANKING_HARDWARE_MODEL_WEIGHT = 0.50

RANKING_NUMERIC_FEATURES = (
    "analytical_score_log",
    "m_tiles_log",
    "k_tiles_log",
    "n_tiles_log",
    "core_count_log",
    "in0_block_w_log",
    "per_core_m_log",
    "per_core_n_log",
    "block_tiles_log",
    "subblock_tiles_log",
    "weight_bytes_log",
    "activation_bytes_log",
    "estimated_dram_bytes_log",
    "estimated_noc_bytes_log",
    "l1_footprint_log",
    "conversion_cost_log",
    "program_count_log",
    "work_tiles_per_core_log",
    "k_blocks_log",
    "memory_bytes_per_core_log",
    "fpu_util_percent",
    "cb_wait_fraction",
    "noc_stall_fraction",
    "packer_stall_fraction",
    "unpacker_stall_fraction",
    "achieved_bandwidth_gbps_log",
    "is_incumbent",
)
RANKING_CATEGORICAL_FEATURES = (
    "operator_kind",
    "program_family",
    "candidate_kind",
    "program_signature",
)
RANKING_NUMERIC_INTERACTIONS = (
    (
        "operator_kind",
        (
            "analytical_score_log",
            "core_count_log",
            "in0_block_w_log",
            "per_core_m_log",
            "per_core_n_log",
            "block_tiles_log",
            "subblock_tiles_log",
            "l1_footprint_log",
            "conversion_cost_log",
            "program_count_log",
            "work_tiles_per_core_log",
            "memory_bytes_per_core_log",
            "fpu_util_percent",
            "cb_wait_fraction",
            "noc_stall_fraction",
            "achieved_bandwidth_gbps_log",
            "is_incumbent",
        ),
    ),
    (
        "program_family",
        (
            "in0_block_w_log",
            "per_core_m_log",
            "per_core_n_log",
            "block_tiles_log",
            "subblock_tiles_log",
            "l1_footprint_log",
            "work_tiles_per_core_log",
            "memory_bytes_per_core_log",
        ),
    ),
    (
        "candidate_kind",
        (
            "analytical_score_log",
            "core_count_log",
            "in0_block_w_log",
            "per_core_m_log",
            "per_core_n_log",
            "block_tiles_log",
            "subblock_tiles_log",
            "l1_footprint_log",
            "conversion_cost_log",
            "program_count_log",
            "work_tiles_per_core_log",
            "memory_bytes_per_core_log",
            "is_incumbent",
        ),
    ),
)
RANKING_CATEGORICAL_INTERACTIONS = (
    ("operator_kind", "program_family"),
    ("candidate_kind", "program_family"),
)


class RankingModelError(ValueError):
    """Raised when ranking examples, models, or evaluations are malformed."""


@dataclass(frozen=True)
class RankingExample:
    campaign_id: str
    candidate_id: str
    operator_name: str
    latency_ms: float
    _features_json: str = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        campaign_id: str,
        candidate_id: str,
        operator_name: str,
        latency_ms: float,
        features: Mapping[str, Any],
    ) -> "RankingExample":
        latency = float(latency_ms)
        if not campaign_id or not candidate_id or not operator_name:
            raise RankingModelError(
                "ranking campaign, candidate, and operator must be non-empty"
            )
        if not math.isfinite(latency) or latency <= 0:
            raise RankingModelError(
                "ranking target latency must be finite and positive"
            )
        _validate_feature_mapping(features)
        return cls(
            campaign_id=str(campaign_id),
            candidate_id=str(candidate_id),
            operator_name=str(operator_name),
            latency_ms=latency,
            _features_json=canonical_json(features),
        )

    @property
    def features(self) -> dict[str, Any]:
        return json.loads(self._features_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "operator": self.operator_name,
            "latency_ms": self.latency_ms,
            "features": self.features,
        }


@dataclass(frozen=True)
class HardwareRankingModel:
    coefficients: tuple[float, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    category_values: tuple[tuple[str, tuple[str, ...]], ...]
    profiler_features_json: str
    training_example_count: int
    ridge_strength: float
    schema_version: int = RANKING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.numeric_means) != len(RANKING_NUMERIC_FEATURES):
            raise RankingModelError("ranking model numeric mean count mismatch")
        if len(self.numeric_scales) != len(RANKING_NUMERIC_FEATURES):
            raise RankingModelError("ranking model numeric scale count mismatch")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.numeric_scales):
            raise RankingModelError("ranking model scales must be positive")
        if len(self.coefficients) != len(self.encoded_feature_names):
            raise RankingModelError("ranking model coefficient count mismatch")
        if self.training_example_count <= 0:
            raise RankingModelError("ranking model must have training examples")

    @property
    def categories(self) -> dict[str, tuple[str, ...]]:
        return dict(self.category_values)

    @property
    def profiler_features(self) -> dict[str, dict[str, float]]:
        return json.loads(self.profiler_features_json)

    @property
    def encoded_feature_names(self) -> tuple[str, ...]:
        names = ["intercept", *RANKING_NUMERIC_FEATURES]
        for key in RANKING_CATEGORICAL_FEATURES:
            names.extend(f"{key}={value}" for value in self.categories.get(key, ()))
        for category, numeric_names in RANKING_NUMERIC_INTERACTIONS:
            for value in self.categories.get(category, ()):
                names.extend(
                    f"{category}={value}*{numeric_name}"
                    for numeric_name in numeric_names
                )
        for left, right in RANKING_CATEGORICAL_INTERACTIONS:
            for left_value in self.categories.get(left, ()):
                names.extend(
                    f"{left}={left_value}*{right}={right_value}"
                    for right_value in self.categories.get(right, ())
                )
        return tuple(names)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())

    def predict_features(self, features: Mapping[str, Any]) -> float:
        encoded = _encode_features(
            features,
            numeric_means=self.numeric_means,
            numeric_scales=self.numeric_scales,
            categories=self.categories,
        )
        score = sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, encoded)
        )
        return score

    def predict_candidate(self, candidate: Any) -> float:
        return self.predict_features(
            extract_ranking_features(
                candidate,
                profiler_features=self.profiler_features,
            )
        )

    def rank_candidates(self, candidates: Sequence[Any]) -> list[Any]:
        values = list(candidates)
        predicted = sorted(
            values,
            key=lambda candidate: (
                self.predict_candidate(candidate),
                int(_candidate_mapping(candidate).get("l1_bytes", 0)),
                str(_candidate_mapping(candidate).get("candidate_id", "")),
            ),
        )
        analytical = sorted(
            values,
            key=lambda candidate: (
                float(_candidate_mapping(candidate).get("analytical_score", 0.0)),
                int(_candidate_mapping(candidate).get("l1_bytes", 0)),
                str(_candidate_mapping(candidate).get("candidate_id", "")),
            ),
        )
        return _blend_rankings(
            values,
            predicted=predicted,
            analytical=analytical,
            model_weight=RANKING_HARDWARE_MODEL_WEIGHT,
        )

    def scheduler_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_kind": "ridge_linear_regression_with_interactions",
            "target": "within_campaign_hardware_rank_percentile",
            "analytical_prior_weight": 1.0 - RANKING_HARDWARE_MODEL_WEIGHT,
            "hardware_model_weight": RANKING_HARDWARE_MODEL_WEIGHT,
            "fingerprint": self.fingerprint,
            "training_example_count": self.training_example_count,
            "final_hardware_measurement_required": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_kind": "ridge_linear_regression_with_interactions",
            "target": "within_campaign_hardware_rank_percentile",
            "ranking_blend": {
                "analytical_prior_weight": (1.0 - RANKING_HARDWARE_MODEL_WEIGHT),
                "hardware_model_weight": RANKING_HARDWARE_MODEL_WEIGHT,
            },
            "numeric_features": list(RANKING_NUMERIC_FEATURES),
            "categorical_features": list(RANKING_CATEGORICAL_FEATURES),
            "encoded_feature_names": list(self.encoded_feature_names),
            "coefficients": list(self.coefficients),
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "category_values": {
                key: list(values) for key, values in self.category_values
            },
            "profiler_features": self.profiler_features,
            "training_example_count": self.training_example_count,
            "ridge_strength": self.ridge_strength,
            "usage": ["candidate_ranking", "next_batch_selection"],
            "final_hardware_measurement_required": True,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardwareRankingModel":
        categories = value.get("category_values") or {}
        profiler = value.get("profiler_features") or {}
        return cls(
            coefficients=tuple(float(item) for item in value["coefficients"]),
            numeric_means=tuple(float(item) for item in value["numeric_means"]),
            numeric_scales=tuple(float(item) for item in value["numeric_scales"]),
            category_values=tuple(
                (key, tuple(str(item) for item in categories.get(key, ())))
                for key in RANKING_CATEGORICAL_FEATURES
            ),
            profiler_features_json=canonical_json(profiler),
            training_example_count=int(value["training_example_count"]),
            ridge_strength=float(value["ridge_strength"]),
            schema_version=int(value.get("schema_version", RANKING_SCHEMA_VERSION)),
        )


def build_profiler_ranking_features(
    profiler_report: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    regions = profiler_report.get("regions") or ()
    if not isinstance(regions, Sequence):
        raise RankingModelError("profiler regions must be a sequence")
    for raw in regions:
        if not isinstance(raw, Mapping) or not raw.get("region"):
            continue
        hardware = raw.get("hardware_metrics") or {}
        latency = _positive_number(raw.get("device_kernel_latency_ms")) or 0.0
        result[str(raw["region"])] = {
            "fpu_util_percent": _number(hardware.get("fpu_util_percent")),
            "cb_wait_fraction": _duration_fraction(
                hardware.get("cb_wait_front_ms"), latency
            ),
            "noc_stall_fraction": _percent_fraction(
                hardware.get("noc_congestion_impact_percent")
                if hardware.get("noc_congestion_impact_percent") is not None
                else hardware.get("noc_util_percent")
            ),
            "packer_stall_fraction": _duration_fraction(
                hardware.get("packer_stall_ms"), latency
            ),
            "unpacker_stall_fraction": _duration_fraction(
                hardware.get("unpacker_stall_ms"), latency
            ),
            "achieved_bandwidth_gbps": _number(
                hardware.get("achieved_weight_bandwidth_gbps")
            ),
        }
    return result


def extract_ranking_features(
    candidate: Any,
    *,
    profiler_features: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    value = _candidate_mapping(candidate)
    metadata = _mapping(value.get("metadata"))
    state = _mapping(metadata.get("candidate"))
    workloads = metadata.get("workloads") or ()
    if not isinstance(workloads, Sequence):
        workloads = ()
    workload = next(
        (item for item in workloads if isinstance(item, Mapping)),
        {},
    )
    program = _first_program(state)

    operator = str(value.get("operator") or state.get("operator") or "unknown")
    family = str(
        metadata.get("program_family")
        or state.get("program_family")
        or program.get("program_family")
        or program.get("kind")
        or "unknown"
    )
    candidate_kind = str(value.get("candidate_kind") or "unknown")

    m_tiles = _first_positive(
        workload.get("m_tiles"),
        metadata.get("m_tiles"),
        1.0,
    )
    k_tiles = _first_positive(
        workload.get("k_tiles"),
        _mapping(state.get("derivation")).get("k_tiles"),
        metadata.get("active_context_len")
        and (_number(metadata.get("active_context_len")) / 32.0),
        1.0,
    )
    n_tiles = _first_positive(
        workload.get("n_tiles"),
        _mapping(state.get("derivation")).get("n_tiles"),
        _mapping(state.get("derivation")).get("head_dim")
        and (_number(_mapping(state.get("derivation")).get("head_dim")) / 32.0),
        1.0,
    )
    core_count = _first_positive(
        workload.get("active_cores"),
        metadata.get("effective_cores"),
        metadata.get("grid_cores"),
        _first_sequence_number(state.get("worker_core_counts")),
        _grid_cores(program.get("compute_grid")),
        _grid_cores(program.get("grid")),
        1.0,
    )
    in0_block_w = _first_positive(
        workload.get("in0_block_w"),
        program.get("in0_block_w"),
        1.0,
    )
    per_core_m = _first_positive(
        program.get("per_core_m"),
        program.get("per_core_M"),
        1.0,
    )
    per_core_n = _first_positive(
        program.get("per_core_n"),
        program.get("per_core_N"),
        1.0,
    )
    block_tiles = _first_positive(
        _number(program.get("out_block_h")) * _number(program.get("out_block_w")),
        _number(program.get("q_chunk_size"))
        * _number(program.get("k_chunk_size"))
        / (32.0 * 32.0),
        1.0,
    )
    subblock_tiles = _first_positive(
        _number(program.get("out_subblock_h")) * _number(program.get("out_subblock_w")),
        1.0,
    )

    tile_elements = 32.0 * 32.0
    weight_bytes = k_tiles * n_tiles * tile_elements * 2.0
    activation_bytes = (m_tiles * k_tiles + m_tiles * n_tiles) * tile_elements * 2.0
    estimated_dram = weight_bytes + activation_bytes
    estimated_noc = (
        activation_bytes + weight_bytes * max(0.0, core_count - 1.0) / core_count
    )
    l1_bytes = max(0.0, _number(value.get("l1_bytes")))
    conversion_cost = _first_positive(
        state.get("hot_path_conversion_count"),
        metadata.get("added_conversion_count"),
        0.0,
    )
    programs = state.get("programs") or ()
    program_count = len(programs) if isinstance(programs, Sequence) else 1
    program_count = max(1, program_count)
    total_work = m_tiles * k_tiles * n_tiles
    k_blocks = k_tiles / max(1.0, in0_block_w)

    profile = _profile_for_operator(operator, profiler_features or {})
    features: dict[str, Any] = {
        "operator_kind": _operator_kind(operator),
        "program_family": family,
        "candidate_kind": candidate_kind,
        "program_signature": _program_signature(state),
        "analytical_score_log": math.log1p(
            max(0.0, _number(value.get("analytical_score")))
        ),
        "m_tiles_log": math.log1p(m_tiles),
        "k_tiles_log": math.log1p(k_tiles),
        "n_tiles_log": math.log1p(n_tiles),
        "core_count_log": math.log1p(core_count),
        "in0_block_w_log": math.log1p(in0_block_w),
        "per_core_m_log": math.log1p(per_core_m),
        "per_core_n_log": math.log1p(per_core_n),
        "block_tiles_log": math.log1p(block_tiles),
        "subblock_tiles_log": math.log1p(subblock_tiles),
        "weight_bytes_log": math.log1p(weight_bytes),
        "activation_bytes_log": math.log1p(activation_bytes),
        "estimated_dram_bytes_log": math.log1p(estimated_dram),
        "estimated_noc_bytes_log": math.log1p(estimated_noc),
        "l1_footprint_log": math.log1p(l1_bytes),
        "conversion_cost_log": math.log1p(conversion_cost),
        "program_count_log": math.log1p(program_count),
        "work_tiles_per_core_log": math.log1p(total_work / core_count),
        "k_blocks_log": math.log1p(k_blocks),
        "memory_bytes_per_core_log": math.log1p(
            (estimated_dram + l1_bytes) / core_count
        ),
        "fpu_util_percent": _number(profile.get("fpu_util_percent")),
        "cb_wait_fraction": _number(profile.get("cb_wait_fraction")),
        "noc_stall_fraction": _number(profile.get("noc_stall_fraction")),
        "packer_stall_fraction": _number(profile.get("packer_stall_fraction")),
        "unpacker_stall_fraction": _number(profile.get("unpacker_stall_fraction")),
        "achieved_bandwidth_gbps_log": math.log1p(
            max(0.0, _number(profile.get("achieved_bandwidth_gbps")))
        ),
        "is_incumbent": float(bool(value.get("is_incumbent"))),
    }
    _validate_feature_mapping(features)
    return features


def ranking_examples_from_active_report(
    report: Mapping[str, Any],
    *,
    campaign_name: str,
    profiler_features: Mapping[str, Mapping[str, float]] | None = None,
) -> list[RankingExample]:
    operators = report.get("operators") or {}
    if not isinstance(operators, Mapping):
        raise RankingModelError("active report operators must be a mapping")
    examples: list[RankingExample] = []
    for operator_name, operator_report in operators.items():
        if not isinstance(operator_report, Mapping):
            continue
        candidates = {
            str(item.get("candidate_id")): item
            for item in operator_report.get("analytical_ranking", ())
            if isinstance(item, Mapping) and item.get("candidate_id")
        }
        measurements: dict[str, float] = {}
        for round_report in operator_report.get("rounds", ()):
            if not isinstance(round_report, Mapping):
                continue
            for candidate_id, result in _mapping(
                round_report.get("measurements")
            ).items():
                if not isinstance(result, Mapping) or not result.get("passed"):
                    continue
                latency = _measurement_latency(result)
                if latency is not None:
                    measurements[str(candidate_id)] = latency
        campaign_id = f"{campaign_name}:{operator_name}"
        for candidate_id, latency in measurements.items():
            candidate = candidates.get(candidate_id)
            if candidate is None:
                continue
            examples.append(
                RankingExample.create(
                    campaign_id=campaign_id,
                    candidate_id=candidate_id,
                    operator_name=str(operator_name),
                    latency_ms=latency,
                    features=extract_ranking_features(
                        candidate,
                        profiler_features=profiler_features,
                    ),
                )
            )
    return examples


def train_hardware_ranking_model(
    examples: Sequence[RankingExample],
    *,
    ridge_strength: float = RANKING_DEFAULT_RIDGE_STRENGTH,
    profiler_features: Mapping[str, Mapping[str, float]] | None = None,
) -> HardwareRankingModel:
    rows = list(examples)
    if len(rows) < 3:
        raise RankingModelError("ranking model requires at least three examples")
    if not math.isfinite(ridge_strength) or ridge_strength <= 0:
        raise RankingModelError("ridge strength must be finite and positive")

    feature_rows = [row.features for row in rows]
    numeric_columns = [
        [float(features[name]) for features in feature_rows]
        for name in RANKING_NUMERIC_FEATURES
    ]
    means = tuple(statistics.fmean(column) for column in numeric_columns)
    scales = tuple(statistics.pstdev(column) or 1.0 for column in numeric_columns)
    categories = {
        name: tuple(sorted({str(features[name]) for features in feature_rows}))
        for name in RANKING_CATEGORICAL_FEATURES
    }
    matrix = [
        _encode_features(
            features,
            numeric_means=means,
            numeric_scales=scales,
            categories=categories,
        )
        for features in feature_rows
    ]
    targets_by_example: dict[tuple[str, str], float] = {}
    for campaign_id in {row.campaign_id for row in rows}:
        campaign = sorted(
            (row for row in rows if row.campaign_id == campaign_id),
            key=lambda row: (row.latency_ms, row.candidate_id),
        )
        denominator = max(1, len(campaign) - 1)
        for rank, row in enumerate(campaign):
            targets_by_example[(campaign_id, row.candidate_id)] = rank / denominator
    targets = [targets_by_example[(row.campaign_id, row.candidate_id)] for row in rows]
    coefficients = _ridge_solve(matrix, targets, ridge_strength)
    return HardwareRankingModel(
        coefficients=tuple(coefficients),
        numeric_means=means,
        numeric_scales=scales,
        category_values=tuple(
            (name, categories[name]) for name in RANKING_CATEGORICAL_FEATURES
        ),
        profiler_features_json=canonical_json(profiler_features or {}),
        training_example_count=len(rows),
        ridge_strength=float(ridge_strength),
    )


def evaluate_ranking_leave_one_campaign_out(
    examples: Sequence[RankingExample],
    *,
    top_k: int = RANKING_DEFAULT_TOP_K,
    ridge_strength: float = RANKING_DEFAULT_RIDGE_STRENGTH,
    profiler_features: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise RankingModelError("top_k must be positive")
    groups: dict[str, list[RankingExample]] = {}
    for example in examples:
        groups.setdefault(example.campaign_id, []).append(example)
    eligible = {name: rows for name, rows in groups.items() if len(rows) > top_k}
    excluded = {
        name: {
            "candidate_count": len(rows),
            "reason": "candidate_count_not_larger_than_top_k",
        }
        for name, rows in groups.items()
        if name not in eligible
    }
    if len(eligible) < 2:
        raise RankingModelError(
            "leave-one-campaign-out requires two campaigns larger than top_k"
        )

    folds = []
    total_candidates = 0
    total_selected = 0
    recovered = 0
    top_k_recovered = 0
    correlations = []
    for campaign_id, holdout in sorted(eligible.items()):
        training = [
            example
            for other_id, rows in groups.items()
            if other_id != campaign_id
            for example in rows
        ]
        model = train_hardware_ranking_model(
            training,
            ridge_strength=ridge_strength,
            profiler_features=profiler_features,
        )
        model_order = sorted(
            holdout,
            key=lambda example: (
                model.predict_features(example.features),
                example.candidate_id,
            ),
        )
        analytical_order = sorted(
            holdout,
            key=lambda example: (
                float(example.features["analytical_score_log"]),
                example.candidate_id,
            ),
        )
        predicted = _blend_rankings(
            holdout,
            predicted=model_order,
            analytical=analytical_order,
            model_weight=RANKING_HARDWARE_MODEL_WEIGHT,
            identifier=lambda example: example.candidate_id,
        )
        actual = sorted(
            holdout,
            key=lambda example: (example.latency_ms, example.candidate_id),
        )
        winner = actual[0].candidate_id
        top_ids = [example.candidate_id for example in predicted[:top_k]]
        incumbent = next(
            (
                example.candidate_id
                for example in holdout
                if bool(example.features["is_incumbent"])
            ),
            None,
        )
        selected = set(top_ids)
        if incumbent is not None:
            selected.add(incumbent)
        winner_in_top_k = winner in top_ids
        winner_selected = winner in selected
        correlation = _spearman_rank_correlation(
            [example.candidate_id for example in predicted],
            [example.candidate_id for example in actual],
        )
        total_candidates += len(holdout)
        total_selected += len(selected)
        recovered += int(winner_selected)
        top_k_recovered += int(winner_in_top_k)
        correlations.append(correlation)
        folds.append(
            {
                "campaign_id": campaign_id,
                "candidate_count": len(holdout),
                "training_example_count": len(training),
                "actual_winner": winner,
                "predicted_top_k": top_ids,
                "incumbent_candidate_id": incumbent,
                "selected_candidate_count": len(selected),
                "winner_in_top_k": winner_in_top_k,
                "winner_recovered": winner_selected,
                "winner_predicted_rank": (
                    [item.candidate_id for item in predicted].index(winner) + 1
                ),
                "spearman_rank_correlation": correlation,
            }
        )

    measurement_reduction = 1.0 - total_selected / total_candidates
    all_winners_recovered = recovered == len(folds)
    return {
        "schema_version": RANKING_SCHEMA_VERSION,
        "method": "leave_one_operator_campaign_out",
        "top_k": top_k,
        "source_campaign_count": len(groups),
        "campaign_count": len(folds),
        "candidate_count": total_candidates,
        "selected_measurement_count": total_selected,
        "top_k_winner_recall": top_k_recovered / len(folds),
        "winner_recall_with_incumbent": recovered / len(folds),
        "all_winners_recovered": all_winners_recovered,
        "mean_spearman_rank_correlation": statistics.fmean(correlations),
        "measurement_reduction": measurement_reduction,
        "minimum_measurement_reduction": (RANKING_MINIMUM_MEASUREMENT_REDUCTION),
        "passed": (
            all_winners_recovered
            and measurement_reduction >= RANKING_MINIMUM_MEASUREMENT_REDUCTION
        ),
        "excluded_campaigns": excluded,
        "folds": folds,
    }


def _blend_rankings(
    values: Sequence[Any],
    *,
    predicted: Sequence[Any],
    analytical: Sequence[Any],
    model_weight: float,
    identifier: Callable[[Any], str] | None = None,
) -> list[Any]:
    identify = identifier or (
        lambda value: str(_candidate_mapping(value).get("candidate_id", ""))
    )
    denominator = max(1, len(values) - 1)
    predicted_ranks = {
        identify(value): rank / denominator for rank, value in enumerate(predicted)
    }
    analytical_ranks = {
        identify(value): rank / denominator for rank, value in enumerate(analytical)
    }
    if len(predicted_ranks) != len(values) or len(analytical_ranks) != len(values):
        raise RankingModelError("ranking candidate ids must be unique")
    analytical_weight = 1.0 - model_weight
    return sorted(
        values,
        key=lambda value: (
            model_weight * predicted_ranks[identify(value)]
            + analytical_weight * analytical_ranks[identify(value)],
            analytical_ranks[identify(value)],
            predicted_ranks[identify(value)],
            identify(value),
        ),
    )


def _candidate_mapping(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    to_dict = getattr(candidate, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise RankingModelError("ranking candidate must be a mapping or provide to_dict")


def _validate_feature_mapping(features: Mapping[str, Any]) -> None:
    missing = set(RANKING_NUMERIC_FEATURES + RANKING_CATEGORICAL_FEATURES) - set(
        features
    )
    if missing:
        raise RankingModelError(
            "ranking features are missing: " + ", ".join(sorted(missing))
        )
    for name in RANKING_NUMERIC_FEATURES:
        value = float(features[name])
        if not math.isfinite(value):
            raise RankingModelError(f"ranking feature {name} must be finite")
    for name in RANKING_CATEGORICAL_FEATURES:
        if not str(features[name]):
            raise RankingModelError(
                f"ranking categorical feature {name} must not be empty"
            )


def _encode_features(
    features: Mapping[str, Any],
    *,
    numeric_means: Sequence[float],
    numeric_scales: Sequence[float],
    categories: Mapping[str, Sequence[str]],
) -> list[float]:
    _validate_feature_mapping(features)
    result = [1.0]
    normalized = {
        name: (float(features[name]) - mean) / scale
        for name, mean, scale in zip(
            RANKING_NUMERIC_FEATURES,
            numeric_means,
            numeric_scales,
        )
    }
    result.extend(normalized[name] for name in RANKING_NUMERIC_FEATURES)
    for name in RANKING_CATEGORICAL_FEATURES:
        observed = str(features[name])
        result.extend(float(observed == value) for value in categories.get(name, ()))
    for category, numeric_names in RANKING_NUMERIC_INTERACTIONS:
        observed = str(features[category])
        for value in categories.get(category, ()):
            active = float(observed == value)
            result.extend(active * normalized[name] for name in numeric_names)
    for left, right in RANKING_CATEGORICAL_INTERACTIONS:
        observed_left = str(features[left])
        observed_right = str(features[right])
        for left_value in categories.get(left, ()):
            result.extend(
                float(observed_left == left_value and observed_right == right_value)
                for right_value in categories.get(right, ())
            )
    return result


def _ridge_solve(
    matrix: Sequence[Sequence[float]],
    targets: Sequence[float],
    ridge_strength: float,
) -> list[float]:
    width = len(matrix[0])
    normal = [[0.0] * width for _ in range(width)]
    right = [0.0] * width
    for row, target in zip(matrix, targets):
        if len(row) != width:
            raise RankingModelError("ranking matrix rows have inconsistent widths")
        for left_index, left_value in enumerate(row):
            right[left_index] += left_value * target
            for right_index, right_value in enumerate(row):
                normal[left_index][right_index] += left_value * right_value
    for index in range(1, width):
        normal[index][index] += ridge_strength
    normal[0][0] += 1e-9
    return _solve_linear_system(normal, right)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], targets: Sequence[float]
) -> list[float]:
    size = len(targets)
    augmented = [
        [float(value) for value in matrix[row]] + [float(targets[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise RankingModelError("ranking normal equation is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _spearman_rank_correlation(
    predicted_order: Sequence[str], actual_order: Sequence[str]
) -> float:
    if set(predicted_order) != set(actual_order):
        raise RankingModelError("rank correlation orders must contain the same ids")
    if len(predicted_order) < 2:
        return 1.0
    predicted = {value: index + 1 for index, value in enumerate(predicted_order)}
    actual = {value: index + 1 for index, value in enumerate(actual_order)}
    differences = sum((predicted[value] - actual[value]) ** 2 for value in predicted)
    count = len(predicted)
    return 1.0 - 6.0 * differences / (count * (count * count - 1))


def _measurement_latency(value: Mapping[str, Any]) -> float | None:
    direct = _positive_number(value.get("latency_ms"))
    if direct is not None:
        return direct
    statistics_report = _mapping(value.get("statistics"))
    return _positive_number(statistics_report.get("p50"))


def _first_program(state: Mapping[str, Any]) -> dict[str, Any]:
    programs = state.get("programs") or ()
    if isinstance(programs, Sequence):
        for program in programs:
            if isinstance(program, Mapping):
                return dict(program)
    program = state.get("program")
    return dict(program) if isinstance(program, Mapping) else {}


def _program_signature(state: Mapping[str, Any]) -> str:
    programs = state.get("programs") or ()
    if not isinstance(programs, Sequence) or isinstance(programs, (str, bytes)):
        programs = ()
    if not programs and isinstance(state.get("program"), Mapping):
        programs = (state["program"],)
    fields = (
        "program_family",
        "runtime_kind",
        "compute_grid",
        "grid",
        "in0_block_w",
        "per_core_m",
        "per_core_n",
        "out_block_h",
        "out_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "max_cores_per_head_batch",
        "q_chunk_size",
        "k_chunk_size",
        "sub_core_grids",
    )
    signature = [
        {name: program.get(name) for name in fields if name in program}
        for program in programs
        if isinstance(program, Mapping)
    ]
    return canonical_json(signature) if signature else "unknown"


def _profile_for_operator(
    operator: str,
    profiler_features: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    normalized = operator.lower()
    mappings = (
        (("qkv",), "qkv_linear"),
        (("o_proj", "o_projection"), "o_projection"),
        (("gate", "packed"), "gate_linear"),
        (("up",), "up_linear"),
        (("down",), "down_linear"),
        (("sdpa",), "sdpa"),
        (("lm_head",), "lm_head_shards"),
        (("fused_attention",), "qkv_linear"),
    )
    for needles, region in mappings:
        if any(needle in normalized for needle in needles):
            return dict(profiler_features.get(region, {}))
    return {}


def _operator_kind(operator: str) -> str:
    normalized = operator.lower()
    if "sdpa" in normalized:
        return "sdpa"
    if "attention" in normalized or "qkv" in normalized or "o_proj" in normalized:
        return "attention_projection"
    if "mlp" in normalized or any(
        name in normalized for name in ("gate", "up", "down")
    ):
        return "mlp_projection"
    if "lm_head" in normalized:
        return "lm_head"
    return normalized.split(".", 1)[0] or "unknown"


def _duration_fraction(value: Any, latency: float) -> float:
    duration = _number(value)
    return duration / latency if latency > 0 else 0.0


def _percent_fraction(value: Any) -> float:
    return _number(value) / 100.0


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = [_number(item) for item in value]
        return max(numbers, default=0.0)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number > 0 else None


def _first_positive(*values: Any) -> float:
    for value in values:
        number = _positive_number(value)
        if number is not None:
            return number
    return 0.0


def _first_sequence_number(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _number(value[0]) if value else 0.0
    return 0.0


def _grid_cores(value: Any) -> float:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        return _number(value[0]) * _number(value[1])
    return 0.0
