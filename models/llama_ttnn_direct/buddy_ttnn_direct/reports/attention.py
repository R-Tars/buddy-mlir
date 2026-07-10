from __future__ import annotations

from typing import Any

from ..smoke_attention_layer import ATTENTION_LAYER_OPS
from ..smoke_attention_primitive import ATTENTION_PRIMITIVES
from .runtime import (
    observed_ops_cover_planned as _observed_ops_cover_planned,
    shape_dict_has_int_lists as _shape_dict_has_int_lists,
    ttnn_runtime_identity_available as _ttnn_runtime_identity_available,
)
from .schema import (
    field_keys as _field_keys,
    int_equal as _int_equal,
    int_list as _int_list,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    path_exists as _path_exists,
)


def attention_primitives_dry_run_complete(reports: Any) -> bool:
    if not isinstance(reports, dict):
        return False
    if set(reports) != set(ATTENTION_PRIMITIVES):
        return False
    for primitive in ATTENTION_PRIMITIVES:
        report = reports.get(primitive)
        if not isinstance(report, dict):
            return False
        if report.get("status") != "dry_run" or report.get("dry_run") is not True:
            return False
        if not _path_exists(report.get("report")):
            return False
    return True


def attention_primitives_dry_run_observed(
    reports: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(reports, dict):
        return {}
    observed = {}
    for primitive, report in sorted(reports.items()):
        if not isinstance(report, dict):
            continue
        observed[str(primitive)] = {
            "status": report.get("status"),
            "dry_run": report.get("dry_run"),
            "report_exists": _path_exists(report.get("report")),
        }
    return observed


def attention_primitive_reports_complete(
    reports: Any,
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_heads: Any,
    num_kv_heads: Any,
    head_dim: Any,
) -> bool:
    if not isinstance(reports, list):
        return False
    observed_sequence = [
        report.get("primitive")
        for report in reports
        if isinstance(report, dict)
    ]
    if observed_sequence != list(ATTENTION_PRIMITIVES):
        return False
    if len(reports) != len(ATTENTION_PRIMITIVES):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        if not _non_empty_string(report.get("report")):
            return False
        if report.get("status") != "passed":
            return False
        if report.get("error") is not None:
            return False
        if not _nonnegative_number(report.get("latency_ms")):
            return False
        if not _non_empty_string(report.get("dtype")):
            return False
        if not _non_empty_string(report.get("layout")):
            return False
        if "memory_config" not in report:
            return False
        if not _int_equal(report.get("batch_size"), batch_size):
            return False
        if not _int_equal(report.get("hidden_size"), hidden_size):
            return False
        if not _int_equal(report.get("num_heads"), num_heads):
            return False
        if not _int_equal(report.get("num_kv_heads"), num_kv_heads):
            return False
        if not _int_equal(report.get("head_dim"), head_dim):
            return False
        if not _int_equal(report.get("max_cache_len"), cache_len):
            return False
        if not _shape_dict_has_int_lists(report.get("input_shapes")):
            return False
        expected_shapes = report.get("expected_output_shapes")
        output_shapes = report.get("output_shapes")
        if not _shape_dict_has_int_lists(expected_shapes):
            return False
        if not isinstance(output_shapes, dict):
            return False
        for name, expected_shape in expected_shapes.items():
            if _int_list(output_shapes.get(name)) != _int_list(
                expected_shape
            ):
                return False
        reference = report.get("reference") or {}
        if reference.get("status") != "passed":
            return False
        if not _observed_ops_cover_planned(
            reference.get("planned_ops"),
            reference.get("observed_ops"),
        ):
            return False
        environment = report.get("ttnn_environment") or {}
        if not isinstance(environment, dict):
            return False
        if environment.get("module_available") is not True:
            return False
        if not _ttnn_runtime_identity_available(environment):
            return False
        if not _non_empty_string(environment.get("tt_metal_git_commit")):
            return False
    return True


def attention_primitive_reports_observed(
    reports: Any,
) -> list[dict[str, Any]]:
    if not isinstance(reports, list):
        return []
    observed = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        input_shapes = report.get("input_shapes")
        output_shapes = report.get("output_shapes")
        expected_shapes = report.get("expected_output_shapes")
        reference = report.get("reference") or {}
        environment = report.get("ttnn_environment") or {}
        observed.append(
            {
                "primitive": report.get("primitive"),
                "status": report.get("status"),
                "latency_ms": report.get("latency_ms"),
                "error": report.get("error"),
                "input_shape_keys": _field_keys(input_shapes),
                "expected_output_shape_keys": _field_keys(expected_shapes),
                "output_shape_keys": _field_keys(output_shapes),
                "dtype": report.get("dtype"),
                "layout": report.get("layout"),
                "memory_config": report.get("memory_config"),
                "reference_status": reference.get("status"),
                "planned_ops": reference.get("planned_ops"),
                "observed_ops": reference.get("observed_ops"),
                "ttnn_version": environment.get("version")
                if isinstance(environment, dict)
                else None,
                "ttnn_module_file": environment.get("module_file")
                if isinstance(environment, dict)
                else None,
                "tt_metal_git_commit": environment.get(
                    "tt_metal_git_commit"
                )
                if isinstance(environment, dict)
                else None,
            }
        )
    return observed


def attention_layer_primitive_reports_complete(reports: Any) -> bool:
    if not isinstance(reports, list):
        return False
    observed_sequence = [
        report.get("primitive")
        for report in reports
        if isinstance(report, dict)
    ]
    if observed_sequence != list(ATTENTION_LAYER_OPS):
        return False
    if len(reports) != len(ATTENTION_LAYER_OPS):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        if report.get("status") != "passed":
            return False
        if report.get("error") is not None:
            return False
        if not _nonnegative_number(report.get("latency_ms")):
            return False
        if not _non_empty_string(report.get("dtype")):
            return False
        if not _non_empty_string(report.get("layout")):
            return False
        if "memory_config" not in report:
            return False
        if not _shape_dict_has_int_lists(report.get("input_shapes")):
            return False
        expected_shapes = report.get("expected_output_shapes")
        output_shapes = report.get("output_shapes")
        if not _shape_dict_has_int_lists(expected_shapes):
            return False
        if not isinstance(output_shapes, dict):
            return False
        for name, expected_shape in expected_shapes.items():
            if _int_list(output_shapes.get(name)) != _int_list(
                expected_shape
            ):
                return False
    return True


def attention_layer_primitive_reports_observed(
    reports: Any,
) -> list[dict[str, Any]]:
    if not isinstance(reports, list):
        return []
    observed = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        input_shapes = report.get("input_shapes")
        output_shapes = report.get("output_shapes")
        expected_shapes = report.get("expected_output_shapes")
        observed.append(
            {
                "primitive": report.get("primitive"),
                "status": report.get("status"),
                "latency_ms": report.get("latency_ms"),
                "error": report.get("error"),
                "input_shape_keys": _field_keys(input_shapes),
                "expected_output_shape_keys": _field_keys(expected_shapes),
                "output_shape_keys": _field_keys(output_shapes),
                "dtype": report.get("dtype"),
                "layout": report.get("layout"),
                "memory_config": report.get("memory_config"),
            }
        )
    return observed
