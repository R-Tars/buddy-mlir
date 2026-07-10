from __future__ import annotations

import math
from typing import Any


def compare_snapshots(
    observed: dict[str, Any],
    reference: dict[str, Any],
    *,
    pcc_threshold: float,
    atol: float | None = None,
) -> dict[str, Any]:
    observed_values = _values(observed)
    reference_values = _values(reference)
    if len(observed_values) != len(reference_values):
        return {
            "name": observed.get("name") or reference.get("name"),
            "status": "shape_mismatch",
            "passed": False,
            "observed_sample_count": len(observed_values),
            "reference_sample_count": len(reference_values),
            "pcc_threshold": float(pcc_threshold),
            "atol": atol,
        }
    if not observed_values:
        return {
            "name": observed.get("name") or reference.get("name"),
            "status": "empty_sample",
            "passed": False,
            "sample_count": 0,
            "pcc_threshold": float(pcc_threshold),
            "atol": atol,
        }

    deltas = [
        observed_value - reference_value
        for observed_value, reference_value in zip(
            observed_values,
            reference_values,
            strict=True,
        )
    ]
    max_abs_error = max(abs(delta) for delta in deltas)
    mean_abs_error = sum(abs(delta) for delta in deltas) / len(deltas)
    rmse = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
    pcc = _pearson_correlation(observed_values, reference_values)
    cosine = _cosine_similarity(observed_values, reference_values)
    finite = all(
        math.isfinite(value)
        for value in (*observed_values, *reference_values)
    )
    passed = finite and pcc >= float(pcc_threshold)
    if atol is not None:
        passed = passed and max_abs_error <= float(atol)
    return {
        "name": observed.get("name") or reference.get("name"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "sample_count": len(observed_values),
        "finite": finite,
        "pcc": pcc,
        "pcc_threshold": float(pcc_threshold),
        "cosine_similarity": cosine,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "atol": atol,
        "observed_sha256": observed.get("sha256"),
        "reference_sha256": reference.get("sha256"),
    }


def compare_top_token(observed: int, reference: int) -> dict[str, Any]:
    passed = int(observed) == int(reference)
    return {
        "name": "top_token",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "observed": int(observed),
        "reference": int(reference),
    }


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered,
            right_centered,
            strict=True,
        )
    )
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return 1.0 if left == right else 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return 1.0 if left == right else 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def _values(snapshot: dict[str, Any]) -> list[float]:
    values = snapshot.get("values")
    if not isinstance(values, list):
        raise ValueError("snapshot values must be a list")
    return [float(value) for value in values]
