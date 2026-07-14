from __future__ import annotations

from typing import Any

from .tensor_meta import tensor_shape


NUMERIC_REFERENCE_NOT_RUN_REASON = (
    "No torch numeric reference is executed by this path yet; the reference "
    "evidence is limited to generated-path structure, shape, and dtype checks."
)


def generated_observed_op_sequence(model: Any, ttnn: Any) -> list[str] | None:
    ops = getattr(model, "ops", None)
    op_log = getattr(ops, "op_log", None)
    if isinstance(op_log, list):
        return [str(item) for item in op_log]
    calls = getattr(ttnn, "calls", None)
    if not isinstance(calls, list):
        return None
    return [
        str(call["op"])
        if isinstance(call, dict) and "op" in call
        else str(call)
        for call in calls
    ]


def decode_step_reference(
    *,
    plan: dict[str, Any],
    layer_count: int,
    output_shapes: dict[str, Any],
    output: dict[str, Any],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    expected_outputs = plan["expected_output_shapes"]
    output_kind = str(plan.get("output_kind", "token"))
    expected_output = expected_outputs[output_kind]
    accepted_output_shapes = [expected_output]
    if output_kind == "token":
        accepted_output_shapes.append([expected_output[0]])
    checks = [
        value_check("layer_count", layer_count, plan["layers"]),
        value_check("output_kind", output.get("kind"), output_kind),
        shape_check(
            f"output.{output_kind}",
            output_shapes.get(output_kind),
            accepted=accepted_output_shapes,
        ),
        dtype_check(f"output.{output_kind}", output.get("dtype")),
        shape_check(
            "output.key_cache",
            output_shapes.get("key_cache"),
            expected=expected_outputs["key_cache"],
        ),
        shape_check(
            "output.value_cache",
            output_shapes.get("value_cache"),
            expected=expected_outputs["value_cache"],
        ),
    ]
    checks.extend(_kv_shape_checks(output_shapes, expected_outputs))
    checks.append(op_sequence_coverage_check(plan["op_sequence"], observed_ops))
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "structural_shape_dtype_op_sequence",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "planned_ops": list(plan["op_sequence"]),
        "observed_ops": observed_ops,
        "checks": checks,
    }


def prefill_reference(
    *,
    plan: dict[str, Any],
    layer_count: int,
    output_shapes: dict[str, Any],
    output: dict[str, Any],
    observed_ops: list[str] | None,
) -> dict[str, Any]:
    expected_outputs = plan["expected_output_shapes"]
    checks = [
        value_check("layer_count", layer_count, plan["layers"]),
        shape_check(
            "output.token",
            output_shapes.get("token"),
            accepted=[
                expected_outputs["token"],
                [expected_outputs["token"][0], 1],
                [expected_outputs["token"][0]],
            ],
        ),
        dtype_check("output.token", output.get("dtype")),
    ]
    checks.extend(_kv_shape_checks(output_shapes, expected_outputs))
    checks.append(op_sequence_coverage_check(plan["op_sequence"], observed_ops))
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "prefill_structural_shape_dtype_op_sequence",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "planned_ops": plan["op_sequence"],
        "observed_ops": observed_ops,
        "checks": checks,
    }


def observed_cache_population(
    *,
    plan: dict[str, Any],
    cache_reports: list[dict[str, Any]],
    output_shapes: dict[str, Any],
) -> list[dict[str, Any]]:
    by_layer = {
        int(report.get("layer_id", index)): dict(report)
        for index, report in enumerate(cache_reports)
        if isinstance(report, dict)
    }
    population = []
    for layer in output_shapes.get("kv_cache_layers", []):
        layer_id = int(layer["layer_id"])
        generated_report = by_layer.get(layer_id, {})
        population.append(
            {
                "layer_id": layer_id,
                "status": "filled",
                "key_cache_shape": layer.get("key_cache"),
                "value_cache_shape": layer.get("value_cache"),
                "expected_key_cache_shape": plan["expected_output_shapes"]["key_cache"],
                "expected_value_cache_shape": plan["expected_output_shapes"]["value_cache"],
                "write_policy": generated_report.get("write_policy", "fill_cache_per_user"),
                "update_shape_layout": generated_report.get(
                    "update_shape_layout", "batch_heads_seq_head_dim"
                ),
                "planned_user_count": plan["batch_size"],
                "filled_user_count": generated_report.get("filled_user_count"),
                "users": generated_report.get("users", []),
                "generated_report": generated_report,
            }
        )
    return population


def loop_input_shapes(
    *, token_ids: Any, page_table: Any, cache_position: Any, kv_cache: Any
) -> dict[str, Any]:
    return {
        "token_ids": tensor_shape(token_ids),
        "page_table": tensor_shape(page_table),
        "cache_position": tensor_shape(cache_position),
        "key_cache": tensor_shape(kv_cache[0].k),
        "value_cache": tensor_shape(kv_cache[0].v),
    }


def loop_output_shapes(
    *, token: Any, kv_cache: Any, layer_count: int
) -> dict[str, Any]:
    return {
        "token": tensor_shape(token),
        "key_cache": tensor_shape(kv_cache[0].k),
        "value_cache": tensor_shape(kv_cache[0].v),
        "kv_cache_layers": [
            {
                "layer_id": layer_id,
                "key_cache": tensor_shape(layer_cache.k),
                "value_cache": tensor_shape(layer_cache.v),
            }
            for layer_id, layer_cache in enumerate(kv_cache[:layer_count])
        ],
    }


def loop_generated_token_ids(
    *, token: Any, ttnn: Any, batch_size: int
) -> dict[str, Any]:
    materialized = _materialize_token_ids(
        token, ttnn=ttnn, batch_size=batch_size
    )
    if materialized is not None:
        return {
            "status": "materialized",
            "source": materialized["source"],
            "token_ids_by_user": materialized["token_ids_by_user"],
        }
    return {
        "status": "unavailable",
        "source": "placeholder_unmaterialized",
        "token_ids_by_user": [[-1] for _ in range(batch_size)],
        "message": "Token ids could not be materialized from the TTNN output.",
    }


def _materialize_token_ids(
    token: Any, *, ttnn: Any, batch_size: int
) -> dict[str, Any] | None:
    direct = _normalize_token_ids(token, batch_size=batch_size)
    if direct is not None:
        return {"source": "tensor_value", "token_ids_by_user": direct}

    to_torch = getattr(ttnn, "to_torch", None)
    if callable(to_torch):
        try:
            host = to_torch(token)
        except Exception:
            host = None
        normalized = _normalize_token_ids(host, batch_size=batch_size)
        if normalized is not None:
            return {
                "source": "ttnn.to_torch",
                "token_ids_by_user": normalized,
            }
    return None


def _normalize_token_ids(
    value: Any, *, batch_size: int
) -> list[list[int]] | None:
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            return None
    for attr in ("data", "values", "value"):
        if not isinstance(value, (list, tuple)) and hasattr(value, attr):
            value = getattr(value, attr)
            if hasattr(value, "tolist"):
                value = value.tolist()
            break
    if not isinstance(value, (list, tuple)):
        if batch_size == 1:
            try:
                return [[_as_int(value)]]
            except Exception:
                return None
        return None
    if not value:
        return None

    rows: list[list[int]] = []
    if isinstance(value[0], (list, tuple)) and len(value) == batch_size:
        for row in value:
            if not isinstance(row, (list, tuple)) or not row:
                return None
            rows.append([_last_token_value(row)])
    elif len(value) == batch_size:
        rows = [[_as_int(item)] for item in value]
    elif batch_size == 1:
        rows = [[_last_token_value(value)]]
    else:
        flat = _flatten_token_values(value)
        if len(flat) != batch_size:
            return None
        rows = [[item] for item in flat]
    return rows if len(rows) == batch_size else None


def _last_token_value(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty token value")
        return _last_token_value(value[-1])
    return _as_int(value)


def _flatten_token_values(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        values: list[int] = []
        for item in value:
            values.extend(_flatten_token_values(item))
        return values
    return [_as_int(value)]


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def shape_check(
    name: str,
    actual: list[int] | None,
    *,
    expected: list[int] | None = None,
    accepted: list[list[int]] | None = None,
) -> dict[str, Any]:
    accepted_shapes = [
        shape for shape in (accepted if accepted is not None else [expected]) if shape is not None
    ]
    return {
        "name": name,
        "type": "shape",
        "actual": actual,
        "expected": expected,
        "accepted": accepted_shapes,
        "passed": actual in accepted_shapes,
    }


def dtype_check(name: str, actual: str | None) -> dict[str, Any]:
    return {"name": name, "type": "dtype", "actual": actual, "passed": actual is not None}


def value_check(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": "value",
        "actual": actual,
        "expected": expected,
        "passed": actual == expected,
    }


def op_sequence_coverage_check(
    planned_ops: list[str], observed_ops: list[str] | None
) -> dict[str, Any]:
    if not isinstance(observed_ops, list):
        return {
            "name": "observed_op_sequence",
            "type": "sequence_coverage",
            "actual": None,
            "expected": planned_ops,
            "passed": False,
            "reason": "generated op instrumentation was not available",
        }
    planned_index = 0
    for observed in observed_ops:
        if planned_index < len(planned_ops) and observed == planned_ops[planned_index]:
            planned_index += 1
    missing = planned_ops[planned_index:]
    return {
        "name": "observed_op_sequence",
        "type": "sequence_coverage",
        "actual": observed_ops,
        "expected": planned_ops,
        "missing_from_ordered_coverage": missing,
        "passed": not missing,
    }


def _kv_shape_checks(
    output_shapes: dict[str, Any], expected_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = []
    for layer in output_shapes.get("kv_cache_layers", []):
        layer_id = int(layer["layer_id"])
        checks.extend(
            [
                shape_check(
                    f"kv_cache_layers.{layer_id}.key_cache",
                    layer.get("key_cache"),
                    expected=expected_outputs["key_cache"],
                ),
                shape_check(
                    f"kv_cache_layers.{layer_id}.value_cache",
                    layer.get("value_cache"),
                    expected=expected_outputs["value_cache"],
                ),
            ]
        )
    return checks
