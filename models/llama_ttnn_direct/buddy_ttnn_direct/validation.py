from __future__ import annotations

import json
import py_compile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .codegen.artifacts import (
    prepare_offline_artifacts,
    write_json,
)
from .codegen.config_diff import (
    default_official_config_path as default_official_parity_config_path,
    diff_official_config,
    dump_config_diff,
)
from .codegen.config_emit import (
    dump_parameter_config,
    emit_parameter_config,
)
from .codegen.package import package_ttnn_direct_program
from .codegen.parameters import materialize_parameters_from_program
from .codegen.program import write_decode_program_bundle
from .codegen.ttnn_tensorizer import tensorize_parameters_from_program_dry_run
from .search.decode_step_autotune import run_decode_step_autotune
from .search.report import dump_search_report
from .search.runner import run_lm_head_search
from .search.space import load_search_space
from .semantic.dump import dump_graph_json
from .semantic.graph import LlamaModelGraph
from .semantic.importer_hf_llama import import_hf_llama
from .smoke_attention_layer import run_smoke_attention_layer
from .smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
    run_smoke_attention_primitive,
)
from .smoke_decode_shell import run_smoke_decode_shell
from .smoke_single_layer_decode import (
    DECODE_PARAMETER_ROLES,
    profile_decode_step,
    run_smoke_decode_step,
    run_smoke_single_layer_decode,
)
from .templates.diff import (
    diff_plan_against_official,
    dump_plan_diff,
    load_official_template,
)
from .templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_template_config,
)


VALIDATION_STEPS = (
    "import_llama",
    "plan",
    "plan_diff",
    "emit_config",
    "prepare_artifacts",
    "build_program",
    "py_compile",
    "official_config_diff",
    "tensorize_parameters_dry_run",
    "decode_shell_dry_run",
    "attention_primitives_dry_run",
    "attention_layer_dry_run",
    "single_layer_decode_dry_run",
    "decode_step_smoke_dry_run",
    "decode_step_profile_dry_run",
    "search_dry_run",
    "decode_step_autotune_dry_run",
    "package_program",
)

REAL_DECODE_VALIDATION_STEPS = (
    "materialize_parameters",
    "decode_shell",
    "smoke_decode_step",
    "profile_decode_step",
    "decode_step_autotune",
)

PROFILE_SECTION_LATENCY_KEYS = (
    "embedding_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
)

PROFILE_LAYER_LATENCY_KEYS = (
    "rms_norm_attn_ms",
    "attention_ms",
    "residual_add_attn_ms",
    "rms_norm_mlp_ms",
    "mlp_ms",
    "residual_add_mlp_ms",
    "total_ms",
)

PROFILE_BOTTLENECK_SECTION_KEYS = (
    "tensor_conversion_ms",
    "embedding_ms",
    "per_layer_attention_ms",
    "per_layer_mlp_ms",
    "layer_stack_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
    "trace_execute_ms",
)


def default_official_template_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "reference"
        / "official_llama31_decode_template.json"
    )


def default_search_space_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "search"
        / "spaces"
        / "lm_head_minimal.json"
    )


def default_decode_step_search_space_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "search"
        / "spaces"
        / "decode_step_minimal.json"
    )


def validate_direct(
    *,
    model_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    official_template_path: str | Path | None = None,
    official_config_path: str | Path | None = None,
    search_space_path: str | Path | None = None,
    decode_step_search_space_path: str | Path | None = None,
    metric: str = "latency_ms",
) -> dict[str, Any]:
    """Run all device-free TTNN Direct scaffold checks and write a report."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "validation_report.json"
    model_path = Path(model_path)
    config_path = Path(config_path)
    official_template_path = (
        Path(official_template_path)
        if official_template_path is not None
        else default_official_template_path()
    )
    official_config_path = (
        Path(official_config_path)
        if official_config_path is not None
        else default_official_parity_config_path()
    )
    search_space_path = (
        Path(search_space_path)
        if search_space_path is not None
        else default_search_space_path()
    )
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )

    paths = {
        "semantic_json": root / "semantic_graph.json",
        "execution_plan": root / "execution_plan.json",
        "plan_diff": root / "plan_diff.json",
        "official_config_diff": root / "official_config_diff.json",
        "parameter_config": root / "parameter_config.json",
        "artifacts_dir": root / "offline_artifacts",
        "program_dir": root / "program",
        "tensorize_report": root / "tensorize_report.json",
        "decode_shell_report": root / "decode_shell_report.json",
        "attention_primitives_dir": root / "attention_primitives",
        "attention_layer_report": root / "attention_layer_report.json",
        "single_layer_decode_report": root / "single_layer_decode_report.json",
        "decode_step_smoke_report": root / "decode_step_smoke_report.json",
        "decode_step_profile_report": root / "decode_step_profile_report.json",
        "search_report": root / "search_report.json",
        "search_candidates_dir": root / "search_candidates",
        "decode_step_autotune_report": root / "decode_step_autotune_report.json",
        "decode_step_autotune_candidates_dir": (
            root / "decode_step_autotune_candidates"
        ),
        "package_dir": root / "package",
        "report": report_path,
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "validate-direct",
        "status": "running",
        "model_path": str(model_path),
        "config": str(config_path),
        "out_dir": str(root),
        "official_template": str(official_template_path),
        "official_config": str(official_config_path),
        "search_space": str(search_space_path),
        "decode_step_search_space": str(decode_step_search_space_path),
        "metric": metric,
        "results": {step: "pending" for step in VALIDATION_STEPS},
        "steps": {},
        "artifacts": {name: str(path) for name, path in paths.items()},
    }

    template_config: dict[str, Any] = {}
    graph: LlamaModelGraph | None = None
    plan: dict[str, Any] | None = None
    parameter_config: dict[str, Any] | None = None

    def persist() -> None:
        _write_json(report_path, report)

    def run_step(
        name: str,
        action: Callable[[], dict[str, Any] | None],
    ) -> bool:
        try:
            detail = action() or {}
        except Exception as exc:  # pragma: no cover - exercised by CLI users.
            report["results"][name] = "fail"
            report["steps"][name] = {
                "status": "fail",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _mark_remaining_skipped(report, name)
            report["status"] = "fail"
            persist()
            return False

        report["results"][name] = "pass"
        report["steps"][name] = {"status": "pass", **detail}
        persist()
        return True

    def _validation_device() -> str:
        return str(template_config.get("device", "p150a"))

    def _validation_decode_layers(max_layers: int = 2) -> int:
        _require(graph, "import_llama")
        assert graph is not None
        return max(1, min(max_layers, int(graph.num_layers)))

    def import_step() -> dict[str, Any]:
        nonlocal template_config, graph
        template_config = load_template_config(config_path)
        graph = import_hf_llama(
            model_path,
            mode="decode",
            batch_size=int(template_config["batch_size"]),
            seq_len=int(template_config["decode_seq_len"]),
            max_cache_len=int(template_config["max_cache_len"]),
            generation_mode=(
                "greedy"
                if template_config["generation_template"]
                == "device_argmax_greedy"
                else "sampling"
            ),
        )
        dump_graph_json(graph, paths["semantic_json"])
        return {
            "semantic_json": str(paths["semantic_json"]),
            "model_name": graph.model_name,
            "num_layers": graph.num_layers,
            "batch_size": graph.batch_size,
            "seq_len": graph.seq_len,
            "max_cache_len": graph.max_cache_len,
        }

    def plan_step() -> dict[str, Any]:
        nonlocal plan
        _require(graph, "import_llama")
        plan = build_execution_plan(graph, template_config)
        dump_execution_plan(plan, paths["execution_plan"])
        return {
            "execution_plan": str(paths["execution_plan"]),
            "num_layers": len(plan["layers"]),
            "final": list(plan["final"]),
        }

    def diff_step() -> dict[str, Any]:
        _require(plan, "plan")
        official_template = load_official_template(official_template_path)
        diff = diff_plan_against_official(plan, official_template)
        dump_plan_diff(diff, paths["plan_diff"])
        return {
            "plan_diff": str(paths["plan_diff"]),
            "missing_ops": list(diff["missing_ops"]),
            "extra_ops": list(diff["extra_ops"]),
            "order_mismatch": list(diff["order_mismatch"]),
        }

    def emit_config_step() -> dict[str, Any]:
        nonlocal parameter_config
        _require(graph, "import_llama")
        parameter_config = emit_parameter_config(
            graph,
            recipe=template_config["dtype_recipe"],
            lm_head_split_count=int(template_config["lm_head_split_count"]),
        )
        dump_parameter_config(parameter_config, paths["parameter_config"])
        return {
            "parameter_config": str(paths["parameter_config"]),
            "weight_count": len(parameter_config["weights"]),
            "lm_head_split_count": (
                parameter_config["lm_head"]["split_count"]
            ),
        }

    def prepare_artifacts_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        _require(parameter_config, "emit_config")
        artifact_paths = prepare_offline_artifacts(
            model_path,
            graph,
            parameter_config,
            paths["artifacts_dir"],
        )
        return {
            "artifacts_dir": str(paths["artifacts_dir"]),
            "manifests": {
                name: str(path) for name, path in sorted(artifact_paths.items())
            },
        }

    def build_program_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        _require(plan, "plan")
        program_paths = write_decode_program_bundle(
            graph=graph,
            plan=plan,
            template_config=template_config,
            model_path=model_path,
            out_dir=paths["program_dir"],
        )
        return {
            "program_dir": str(paths["program_dir"]),
            "artifacts": {
                name: str(path) for name, path in sorted(program_paths.items())
            },
        }

    def py_compile_step() -> dict[str, Any]:
        compiled = [
            paths["program_dir"] / "model.py",
            paths["program_dir"] / "run_decode.py",
        ]
        for source in compiled:
            py_compile.compile(str(source), doraise=True)
        return {"compiled": [str(source) for source in compiled]}

    def official_config_diff_step() -> dict[str, Any]:
        diff = diff_official_config(
            paths["program_dir"] / "config.json",
            official_config_path,
        )
        dump_config_diff(diff, paths["official_config_diff"])
        return {
            "official_config_diff": str(paths["official_config_diff"]),
            "diff_status": diff["status"],
            "issue_count": diff["summary"]["issue_count"],
            "missing_count": diff["summary"]["missing_count"],
            "mismatch_count": diff["summary"]["mismatch_count"],
            "extra_count": diff["summary"]["extra_count"],
        }

    def tensorize_parameters_dry_run_step() -> dict[str, Any]:
        tensor_report = tensorize_parameters_from_program_dry_run(
            program_dir=paths["program_dir"],
            roles=["embedding", "norm", "attention", "mlp", "lm_head"],
            layers=[0],
            device=_validation_device(),
            out=paths["tensorize_report"],
        )
        return {
            "tensorize_report": str(paths["tensorize_report"]),
            "roles": list(tensor_report["roles"]),
            "tensor_count": tensor_report["tensor_count"],
            "dry_run": tensor_report["dry_run"],
        }

    def decode_shell_dry_run_step() -> dict[str, Any]:
        shell_report = run_smoke_decode_shell(
            out=paths["decode_shell_report"],
            program_dir=paths["program_dir"],
            layers=1,
            disable_attention=True,
            device=_validation_device(),
            dry_run=True,
        )
        return {
            "decode_shell_report": str(paths["decode_shell_report"]),
            "layers": shell_report["layers_requested"],
            "dry_run": shell_report["dry_run"],
            "smoke_status": shell_report["status"],
        }

    def attention_primitives_dry_run_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        assert graph is not None
        reports = {}
        for primitive in ATTENTION_PRIMITIVES:
            report_path = paths["attention_primitives_dir"] / f"{primitive}.json"
            primitive_report = run_smoke_attention_primitive(
                out=report_path,
                primitive=primitive,
                device=_validation_device(),
                batch_size=int(template_config["batch_size"]),
                hidden_size=int(graph.hidden_size),
                num_heads=int(graph.num_attention_heads),
                num_kv_heads=int(graph.num_key_value_heads),
                head_dim=int(graph.head_dim),
                max_cache_len=int(template_config["max_cache_len"]),
                dry_run=True,
            )
            reports[primitive] = {
                "report": str(report_path),
                "status": primitive_report["status"],
                "dry_run": primitive_report["dry_run"],
            }
        return {
            "attention_primitives_dir": str(paths["attention_primitives_dir"]),
            "primitive_count": len(reports),
            "reports": reports,
        }

    def attention_layer_dry_run_step() -> dict[str, Any]:
        layer_report = run_smoke_attention_layer(
            out=paths["attention_layer_report"],
            program_dir=paths["program_dir"],
            layer=0,
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            dry_run=True,
        )
        return {
            "attention_layer_report": str(paths["attention_layer_report"]),
            "layer": layer_report["layer"],
            "primitive_count": len(layer_report["primitive_reports"]),
            "dry_run": layer_report["dry_run"],
            "smoke_status": layer_report["status"],
        }

    def single_layer_decode_dry_run_step() -> dict[str, Any]:
        smoke_report = run_smoke_single_layer_decode(
            out=paths["single_layer_decode_report"],
            program_dir=paths["program_dir"],
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            dry_run=True,
        )
        return {
            "single_layer_decode_report": str(
                paths["single_layer_decode_report"]
            ),
            "layers": smoke_report["layers"],
            "dry_run": smoke_report["dry_run"],
            "smoke_status": smoke_report["status"],
            "op_count": len(smoke_report["op_sequence"]),
        }

    def decode_step_smoke_dry_run_step() -> dict[str, Any]:
        smoke_report = run_smoke_decode_step(
            out=paths["decode_step_smoke_report"],
            program_dir=paths["program_dir"],
            layers=_validation_decode_layers(),
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            trace=True,
            trace_iterations=1,
            dry_run=True,
        )
        return {
            "decode_step_smoke_report": str(paths["decode_step_smoke_report"]),
            "layers": smoke_report["layers"],
            "dry_run": smoke_report["dry_run"],
            "trace_status": smoke_report["trace"]["status"],
            "ttnn_environment": smoke_report.get("ttnn_environment"),
            "smoke_status": smoke_report["status"],
            "op_count": len(smoke_report["op_sequence"]),
            **_reference_summary(smoke_report),
        }

    def decode_step_profile_dry_run_step() -> dict[str, Any]:
        profile_report = profile_decode_step(
            out=paths["decode_step_profile_report"],
            program_dir=paths["program_dir"],
            layers=_validation_decode_layers(),
            device=_validation_device(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            trace=True,
            trace_iterations=1,
            dry_run=True,
        )
        return {
            "decode_step_profile_report": str(
                paths["decode_step_profile_report"]
            ),
            "layers": profile_report["layers"],
            "dry_run": profile_report["dry_run"],
            "trace_status": profile_report["trace"]["status"],
            "ttnn_environment": profile_report.get("ttnn_environment"),
            "profile_status": profile_report["status"],
            "bottleneck": profile_report["bottleneck_summary"]["max_section"],
            **_reference_summary(profile_report),
        }

    def search_step() -> dict[str, Any]:
        _require(graph, "import_llama")
        search_report = run_lm_head_search(
            graph=graph,
            base_config=template_config,
            space=load_search_space(search_space_path),
            metric=metric,
            out=paths["search_report"],
            candidates_dir=paths["search_candidates_dir"],
            dry_run=True,
        )
        dump_search_report(search_report, paths["search_report"])
        return {
            "search_report": str(paths["search_report"]),
            "candidates_dir": str(paths["search_candidates_dir"]),
            "candidate_count": search_report["candidate_count"],
            "dry_run": search_report["dry_run"],
        }

    def decode_step_autotune_dry_run_step() -> dict[str, Any]:
        autotune_report = run_decode_step_autotune(
            program_dir=paths["program_dir"],
            space=load_search_space(decode_step_search_space_path),
            out=paths["decode_step_autotune_report"],
            layers=_validation_decode_layers(),
            batch_size=int(template_config["batch_size"]),
            cache_len=int(template_config["max_cache_len"]),
            metric=metric,
            candidates_dir=paths["decode_step_autotune_candidates_dir"],
            dry_run=True,
            device=_validation_device(),
            trace=True,
            trace_iterations=1,
        )
        dump_search_report(autotune_report, paths["decode_step_autotune_report"])
        return {
            "decode_step_autotune_report": str(
                paths["decode_step_autotune_report"]
            ),
            "candidates_dir": str(
                paths["decode_step_autotune_candidates_dir"]
            ),
            "candidate_count": autotune_report["candidate_count"],
            "metric_direction": autotune_report.get("metric_direction"),
            "status_counts": autotune_report.get("status_counts", {}),
            "reference_status_counts": autotune_report.get(
                "reference_status_counts",
                {},
            ),
            "trace_status_counts": autotune_report.get(
                "trace_status_counts",
                {},
            ),
            "dry_run": autotune_report["dry_run"],
            "trace_enabled": autotune_report["trace_enabled"],
        }

    def package_step() -> dict[str, Any]:
        package_paths = package_ttnn_direct_program(
            paths["program_dir"],
            paths["package_dir"],
        )
        return {
            "package_dir": str(paths["package_dir"]),
            "artifacts": {
                name: str(path) for name, path in sorted(package_paths.items())
            },
        }

    step_actions = {
        "import_llama": import_step,
        "plan": plan_step,
        "plan_diff": diff_step,
        "emit_config": emit_config_step,
        "prepare_artifacts": prepare_artifacts_step,
        "build_program": build_program_step,
        "py_compile": py_compile_step,
        "official_config_diff": official_config_diff_step,
        "tensorize_parameters_dry_run": tensorize_parameters_dry_run_step,
        "decode_shell_dry_run": decode_shell_dry_run_step,
        "attention_primitives_dry_run": attention_primitives_dry_run_step,
        "attention_layer_dry_run": attention_layer_dry_run_step,
        "single_layer_decode_dry_run": single_layer_decode_dry_run_step,
        "decode_step_smoke_dry_run": decode_step_smoke_dry_run_step,
        "decode_step_profile_dry_run": decode_step_profile_dry_run_step,
        "search_dry_run": search_step,
        "decode_step_autotune_dry_run": (
            decode_step_autotune_dry_run_step
        ),
        "package_program": package_step,
    }

    for step in VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            return report

    report["status"] = "pass"
    persist()
    return report


def validate_real_decode(
    *,
    program_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    decode_step_search_space_path: str | Path | None = None,
    layers: int = 1,
    batch_size: int | None = None,
    cache_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    metric: str = "latency_ms",
    dry_run: bool = False,
    skip_autotune: bool = False,
    require_trace: bool = False,
    min_tokens_per_second_per_user: float | None = None,
    decode_shell_pcc_threshold: float = 0.99,
    require_decode_shell_numeric_reference: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run the real-weight generated decode validation gates.

    This is intentionally separate from ``validate_direct``: that command stays
    device-free, while this one proves the materialize/tensorize/runtime path.
    """
    layer_count = int(layers)
    if layer_count <= 0:
        raise ValueError("layers must be positive")
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "real_decode_validation_report.json"
    program_dir = Path(program_dir)
    model_path = Path(model_path)
    program_config = _load_program_config(program_dir)
    program_num_layers = int(program_config["num_layers"])
    if layer_count > program_num_layers:
        raise ValueError(
            "layers must be <= generated config num_layers "
            f"({program_num_layers})"
        )
    resolved_batch_size = _resolve_runtime_dimension(
        "batch_size",
        requested=batch_size,
        fallback=program_config.get("batch_size"),
    )
    resolved_cache_len = _resolve_runtime_dimension(
        "cache_len",
        requested=cache_len,
        fallback=program_config.get("max_cache_len"),
    )
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )
    layers_to_materialize = list(range(layer_count))

    paths = {
        "materialize_report": root / "parameter_materialization_report.json",
        "decode_shell_report": root / "decode_shell_report.json",
        "smoke_report": root / "decode_step_smoke_report.json",
        "profile_report": root / "decode_step_profile_report.json",
        "autotune_report": root / "decode_step_autotune_report.json",
        "autotune_candidates_dir": root / "decode_step_autotune_candidates",
        "evidence_manifest": root / "real_decode_evidence_manifest.json",
        "report": report_path,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "validate-real-decode",
        "status": "running",
        "program_dir": str(program_dir),
        "model_path": str(model_path),
        "out_dir": str(root),
        "decode_step_search_space": str(decode_step_search_space_path),
        "program_num_layers": program_num_layers,
        "layers": layer_count,
        "requested_batch_size": batch_size,
        "requested_cache_len": cache_len,
        "batch_size": resolved_batch_size,
        "cache_len": resolved_cache_len,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations,
        "metric": metric,
        "dry_run": dry_run,
        "skip_autotune": skip_autotune,
        "require_trace": require_trace,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "decode_shell_pcc_threshold": decode_shell_pcc_threshold,
        "require_decode_shell_numeric_reference": (
            require_decode_shell_numeric_reference
        ),
        "results": {
            step: "pending" for step in REAL_DECODE_VALIDATION_STEPS
        },
        "steps": {},
        "artifacts": {name: str(path) for name, path in paths.items()},
    }

    def persist() -> None:
        _write_json(report_path, report)

    def run_step(
        name: str,
        action: Callable[[], dict[str, Any]],
    ) -> bool:
        try:
            detail = action()
        except Exception as exc:  # pragma: no cover - exercised by CLI users.
            report["results"][name] = "fail"
            report["steps"][name] = {
                "status": "fail",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _mark_remaining_skipped(
                report,
                name,
                REAL_DECODE_VALIDATION_STEPS,
            )
            report["status"] = "fail"
            persist()
            return False

        status = str(detail.get("status", "pass"))
        report["results"][name] = status
        report["steps"][name] = detail
        persist()
        if status in {"pass", "dry_run", "skipped"}:
            return True

        _mark_remaining_skipped(
            report,
            name,
            REAL_DECODE_VALIDATION_STEPS,
        )
        report["status"] = status
        persist()
        return False

    def materialize_step() -> dict[str, Any]:
        if dry_run:
            required_tensor_paths = _required_materialized_tensor_paths(
                layer_count=layer_count,
                lm_head_split_count=None,
            )
            materialize_report = {
                "schema_version": 1,
                "status": "dry_run",
                "backend": "torch",
                "model_path": str(model_path),
                "program_dir": str(program_dir),
                "materialized_layer_ids": layers_to_materialize,
                "required_tensor_paths": required_tensor_paths,
                "message": "Dry run only; safetensors payloads were not loaded.",
            }
            _write_json(paths["materialize_report"], materialize_report)
            return {
                "status": "dry_run",
                "materialize_report": str(paths["materialize_report"]),
                "materialized_layer_ids": layers_to_materialize,
                "required_tensor_paths": required_tensor_paths,
            }

        materialize_report = materialize_parameters_from_program(
            model_path=model_path,
            program_dir=program_dir,
            backend="torch",
            layers=layers_to_materialize,
            out=paths["materialize_report"],
        )
        materialized_tensor_paths = sorted(
            (materialize_report.get("tensors") or {}).keys()
        )
        lm_head_split_count = materialize_report["lm_head"]["split_count"]
        required_tensor_paths = _required_materialized_tensor_paths(
            layer_count=layer_count,
            lm_head_split_count=lm_head_split_count,
        )
        return {
            "status": "pass",
            "materialize_report": str(paths["materialize_report"]),
            "materialized_layer_ids": list(
                materialize_report["materialized_layer_ids"]
            ),
            "tensor_count": materialize_report["tensor_count"],
            "lm_head_split_count": lm_head_split_count,
            "materialized_tensor_paths": materialized_tensor_paths,
            "required_tensor_paths": required_tensor_paths,
            "missing_required_tensor_paths": sorted(
                set(required_tensor_paths) - set(materialized_tensor_paths)
            ),
            "key_tensors": _materialization_key_tensors(
                materialize_report.get("tensors") or {},
                required_tensor_paths,
            ),
        }

    def materialized_lm_head_split_count() -> int | None:
        materialize = report.get("steps", {}).get("materialize_parameters", {})
        value = materialize.get("lm_head_split_count")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def tensorization_path_detail(
        runtime_report: dict[str, Any],
    ) -> dict[str, Any]:
        setup = runtime_report.get("parameter_setup") or {}
        tensorization = setup.get("tensorization") or {}
        if not isinstance(tensorization, dict):
            tensorization = {}
        required_tensor_paths = _required_tensorized_tensor_paths(
            layer_count=layer_count,
            lm_head_split_count=materialized_lm_head_split_count(),
        )
        tensorized_tensor_paths = _tensorized_tensor_paths(tensorization)
        return {
            "tensorized_tensor_paths": tensorized_tensor_paths,
            "required_tensorized_tensor_paths": required_tensor_paths,
            "missing_required_tensorized_tensor_paths": sorted(
                set(required_tensor_paths) - set(tensorized_tensor_paths)
            ),
        }

    def decode_shell_step() -> dict[str, Any]:
        shell_report = run_smoke_decode_shell(
            out=paths["decode_shell_report"],
            program_dir=program_dir,
            layers=layer_count,
            disable_attention=True,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            dry_run=dry_run,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
            pcc_threshold=decode_shell_pcc_threshold,
        )
        numeric_reference = (
            shell_report.get("reference", {}).get("numeric_reference", {})
        )
        return {
            "status": _runtime_step_status(shell_report, dry_run=dry_run),
            "decode_shell_report": str(paths["decode_shell_report"]),
            "runtime_status": shell_report["status"],
            "layers": shell_report.get("layers_requested"),
            "parameter_source": shell_report.get("parameter_source"),
            "input_source": shell_report.get("input_source"),
            "runtime_input_tensor_count": shell_report.get(
                "runtime_input_tensor_count"
            ),
            "numeric_reference_status": numeric_reference.get("status"),
            "numeric_reference_kind": numeric_reference.get("kind"),
            "pcc": numeric_reference.get("pcc"),
            "pcc_threshold": numeric_reference.get(
                "pcc_threshold",
                decode_shell_pcc_threshold,
            ),
            **_reference_summary(shell_report),
        }

    def smoke_step() -> dict[str, Any]:
        smoke_report = run_smoke_decode_step(
            out=paths["smoke_report"],
            program_dir=program_dir,
            layers=layer_count,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            dry_run=dry_run,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        return {
            "status": _runtime_step_status(smoke_report, dry_run=dry_run),
            "smoke_report": str(paths["smoke_report"]),
            "runtime_status": smoke_report["status"],
            "layers": smoke_report.get("layers"),
            "batch_size": smoke_report.get("batch_size"),
            "cache_len": smoke_report.get("cache_len"),
            "parameter_source": smoke_report.get("parameter_source"),
            "input_source": smoke_report.get("input_source"),
            "tensor_conversion_count": smoke_report.get(
                "tensor_conversion_count"
            ),
            "trace_status": smoke_report.get("trace", {}).get("status"),
            "trace": _trace_summary(smoke_report.get("trace")),
            "ttnn_environment": smoke_report.get("ttnn_environment"),
            "parameter_setup": smoke_report.get("parameter_setup"),
            **tensorization_path_detail(smoke_report),
            **_reference_summary(smoke_report),
        }

    def profile_step() -> dict[str, Any]:
        profile_report = profile_decode_step(
            out=paths["profile_report"],
            program_dir=program_dir,
            layers=layer_count,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            dry_run=dry_run,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        bottleneck = profile_report.get("bottleneck_summary", {})
        return {
            "status": _runtime_step_status(profile_report, dry_run=dry_run),
            "profile_report": str(paths["profile_report"]),
            "runtime_status": profile_report["status"],
            "layers": profile_report.get("layers"),
            "batch_size": profile_report.get("batch_size"),
            "cache_len": profile_report.get("cache_len"),
            "parameter_source": profile_report.get("parameter_source"),
            "input_source": profile_report.get("input_source"),
            "tensor_conversion_count": profile_report.get(
                "tensor_conversion_count"
            ),
            "tensor_conversion_ms": profile_report.get("tensor_conversion_ms"),
            "latency_ms": profile_report.get("latency_ms"),
            "section_latency_ms": profile_report.get("section_latency_ms"),
            "layer_profiles": profile_report.get("layer_profiles", []),
            "lm_head_profile": profile_report.get("lm_head_profile"),
            "throughput_summary": profile_report.get("throughput_summary"),
            "bottleneck_summary": bottleneck,
            "max_section": bottleneck.get("max_section"),
            "trace_status": profile_report.get("trace", {}).get("status"),
            "trace": _trace_summary(profile_report.get("trace")),
            "ttnn_environment": profile_report.get("ttnn_environment"),
            "parameter_setup": profile_report.get("parameter_setup"),
            **tensorization_path_detail(profile_report),
            **_reference_summary(profile_report),
        }

    def autotune_step() -> dict[str, Any]:
        if skip_autotune:
            return {
                "status": "skipped",
                "reason": "skip_autotune requested",
            }
        autotune_report = run_decode_step_autotune(
            program_dir=program_dir,
            model_path=None if dry_run else model_path,
            space=load_search_space(decode_step_search_space_path),
            out=paths["autotune_report"],
            layers=layer_count,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            metric=metric,
            candidates_dir=paths["autotune_candidates_dir"],
            dry_run=dry_run,
            device=device,
            device_id=device_id,
            dtype_seed=dtype_seed,
            trace=trace,
            trace_iterations=trace_iterations,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        dump_search_report(autotune_report, paths["autotune_report"])
        return {
            "status": (
                "dry_run"
                if dry_run
                else "pass"
                if autotune_report.get("best") is not None
                else "fail"
            ),
            "autotune_report": str(paths["autotune_report"]),
            "candidates_dir": str(paths["autotune_candidates_dir"]),
            "candidate_count": autotune_report["candidate_count"],
            "metric_direction": autotune_report.get("metric_direction"),
            "status_counts": autotune_report.get("status_counts", {}),
            "passed_candidate_count": autotune_report.get(
                "passed_candidate_count",
                0,
            ),
            "failed_candidate_count": autotune_report.get(
                "failed_candidate_count",
                0,
            ),
            "best": autotune_report.get("best", {}).get("id")
            if autotune_report.get("best") is not None
            else None,
            "best_reference_status": (
                autotune_report.get("best", {}).get("reference_status")
                if autotune_report.get("best") is not None
                else None
            ),
            "reference_status_counts": autotune_report.get(
                "reference_status_counts",
                _candidate_reference_status_counts(autotune_report),
            ),
            "trace_status_counts": autotune_report.get(
                "trace_status_counts",
                {},
            ),
            "dry_run": autotune_report["dry_run"],
        }

    step_actions = {
        "materialize_parameters": materialize_step,
        "decode_shell": decode_shell_step,
        "smoke_decode_step": smoke_step,
        "profile_decode_step": profile_step,
        "decode_step_autotune": autotune_step,
    }

    for step in REAL_DECODE_VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            return report

    acceptance = _real_decode_acceptance(
        report,
        require_trace=require_trace,
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        require_decode_shell_numeric_reference=(
            require_decode_shell_numeric_reference
        ),
    )
    report["acceptance"] = acceptance
    report["status"] = (
        "dry_run"
        if dry_run
        else "pass"
        if acceptance["passed"]
        else "acceptance_failed"
    )
    evidence = _real_decode_evidence_manifest(report, paths)
    _write_json(paths["evidence_manifest"], evidence)
    report["evidence"] = {
        "status": evidence["status"],
        "manifest": str(paths["evidence_manifest"]),
        "artifact_count": len(evidence["artifacts"]),
        "failed_acceptance_checks": evidence["acceptance"][
            "failed_checks"
        ],
    }
    persist()
    return report


def _runtime_step_status(
    runtime_report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if runtime_report.get("passed"):
        return "pass"
    return str(runtime_report.get("status", "fail"))


def _load_program_config(program_dir: Path) -> dict[str, Any]:
    config_path = program_dir / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("num_layers", "batch_size", "max_cache_len"):
        if key not in config:
            raise ValueError(f"generated program config missing {key!r}")
    return config


def _resolve_runtime_dimension(
    name: str,
    *,
    requested: int | None,
    fallback: Any,
) -> int:
    value = fallback if requested is None else requested
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _trace_summary(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    execute_samples = trace.get("execute_samples_ms")
    if isinstance(execute_samples, list):
        execute_sample_count = len(execute_samples)
    else:
        execute_sample_count = None
    summary = {
        "requested": trace.get("requested"),
        "status": trace.get("status"),
        "iterations": trace.get("iterations"),
        "capture_latency_ms": trace.get("capture_latency_ms"),
        "execute_latency_ms": trace.get("execute_latency_ms"),
        "execute_sample_count": execute_sample_count,
    }
    if trace.get("error") is not None:
        summary["error"] = trace.get("error")
    return summary


def _required_materialized_tensor_paths(
    *,
    layer_count: int,
    lm_head_split_count: int | None,
) -> list[str]:
    paths = [
        "embedding.weight",
        "final_norm.weight",
        "lm_head.weight",
    ]
    for layer_id in range(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.q_proj.weight",
                f"layers.{layer_id}.attention.k_proj.weight",
                f"layers.{layer_id}.attention.v_proj.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    if lm_head_split_count is not None:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(int(lm_head_split_count))
        )
    return paths


def _required_tensorized_tensor_paths(
    *,
    layer_count: int,
    lm_head_split_count: int | None,
) -> list[str]:
    paths = [
        "embedding.weight",
        "final_norm.weight",
    ]
    for layer_id in range(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    if lm_head_split_count is not None:
        paths.extend(
            f"lm_head.splits.{shard_id}.weight"
            for shard_id in range(int(lm_head_split_count))
        )
    return paths


def _materialization_key_tensors(
    tensors: dict[str, Any],
    required_tensor_paths: list[str],
) -> dict[str, Any]:
    key_paths = []
    for path in (
        "embedding.weight",
        "final_norm.weight",
        "layers.0.attention.wqkv_packed.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "lm_head.splits.0.weight",
    ):
        if path in required_tensor_paths:
            key_paths.append(path)
    return {
        path: tensors[path]
        for path in key_paths
        if path in tensors
    }


def _real_decode_evidence_manifest(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    steps = report.get("steps", {})
    materialize = steps.get("materialize_parameters", {})
    decode_shell = steps.get("decode_shell", {})
    smoke = steps.get("smoke_decode_step", {})
    profile = steps.get("profile_decode_step", {})
    autotune = steps.get("decode_step_autotune", {})
    acceptance = report.get("acceptance", {})
    failed_checks = [
        check
        for check in acceptance.get("checks", [])
        if isinstance(check, dict) and not check.get("passed")
    ]
    if report.get("dry_run"):
        status = "dry_run"
    elif report.get("status") == "pass" and acceptance.get("passed"):
        status = "accepted"
    else:
        status = "incomplete"

    return {
        "schema_version": 1,
        "status": status,
        "validation": {
            "command": report.get("command"),
            "status": report.get("status"),
            "dry_run": report.get("dry_run"),
            "program_dir": report.get("program_dir"),
            "model_path": report.get("model_path"),
            "program_num_layers": report.get("program_num_layers"),
            "layers": report.get("layers"),
            "requested_batch_size": report.get("requested_batch_size"),
            "requested_cache_len": report.get("requested_cache_len"),
            "batch_size": report.get("batch_size"),
            "cache_len": report.get("cache_len"),
            "device": report.get("device"),
            "device_id": report.get("device_id"),
            "dtype_seed": report.get("dtype_seed"),
            "trace_enabled": report.get("trace_enabled"),
            "trace_iterations": report.get("trace_iterations"),
            "metric": report.get("metric"),
            "skip_autotune": report.get("skip_autotune"),
        },
        "requirements": {
            "require_trace": report.get("require_trace"),
            "min_tokens_per_second_per_user": report.get(
                "min_tokens_per_second_per_user"
            ),
            "decode_shell_pcc_threshold": report.get(
                "decode_shell_pcc_threshold"
            ),
            "require_decode_shell_numeric_reference": report.get(
                "require_decode_shell_numeric_reference"
            ),
        },
        "artifacts": [
            _artifact_evidence(name, path)
            for name, path in paths.items()
            if name != "evidence_manifest"
        ],
        "device_evidence": {
            "smoke_ttnn_environment": smoke.get("ttnn_environment"),
            "profile_ttnn_environment": profile.get("ttnn_environment"),
        },
        "weight_evidence": {
            "materialization": {
                "status": materialize.get("status"),
                "materialized_layer_ids": materialize.get(
                    "materialized_layer_ids"
                ),
                "tensor_count": materialize.get("tensor_count"),
                "lm_head_split_count": materialize.get("lm_head_split_count"),
                "required_tensor_paths": materialize.get(
                    "required_tensor_paths",
                    [],
                ),
                "missing_required_tensor_paths": materialize.get(
                    "missing_required_tensor_paths",
                    [],
                ),
                "key_tensors": materialize.get("key_tensors", {}),
            },
            "smoke_tensorization": _tensorization_evidence(smoke),
            "profile_tensorization": _tensorization_evidence(profile),
        },
        "runtime_evidence": {
            "decode_shell": {
                "status": decode_shell.get("status"),
                "runtime_status": decode_shell.get("runtime_status"),
                "layers": decode_shell.get("layers"),
                "parameter_source": decode_shell.get("parameter_source"),
                "input_source": decode_shell.get("input_source"),
                "reference_status": decode_shell.get("reference_status"),
                "numeric_reference_status": decode_shell.get(
                    "numeric_reference_status"
                ),
                "numeric_reference_kind": decode_shell.get(
                    "numeric_reference_kind"
                ),
                "pcc": decode_shell.get("pcc"),
                "pcc_threshold": decode_shell.get("pcc_threshold"),
                "reference_planned_ops": decode_shell.get(
                    "reference_planned_ops"
                ),
                "reference_observed_ops": decode_shell.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": decode_shell.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "smoke_decode_step": {
                "status": smoke.get("status"),
                "runtime_status": smoke.get("runtime_status"),
                "layers": smoke.get("layers"),
                "batch_size": smoke.get("batch_size"),
                "cache_len": smoke.get("cache_len"),
                "parameter_source": smoke.get("parameter_source"),
                "input_source": smoke.get("input_source"),
                "tensor_conversion_count": smoke.get(
                    "tensor_conversion_count"
                ),
                "trace_status": smoke.get("trace_status"),
                "trace": smoke.get("trace"),
                "reference_status": smoke.get("reference_status"),
                "reference_kind": smoke.get("reference_kind"),
                "reference_planned_ops": smoke.get("reference_planned_ops"),
                "reference_observed_ops": smoke.get("reference_observed_ops"),
                "reference_failed_checks": smoke.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "profile_decode_step": {
                "status": profile.get("status"),
                "runtime_status": profile.get("runtime_status"),
                "layers": profile.get("layers"),
                "batch_size": profile.get("batch_size"),
                "cache_len": profile.get("cache_len"),
                "parameter_source": profile.get("parameter_source"),
                "input_source": profile.get("input_source"),
                "tensor_conversion_count": profile.get(
                    "tensor_conversion_count"
                ),
                "tensor_conversion_ms": profile.get("tensor_conversion_ms"),
                "latency_ms": profile.get("latency_ms"),
                "section_latency_ms": profile.get("section_latency_ms"),
                "layer_profiles": profile.get("layer_profiles", []),
                "lm_head_profile": profile.get("lm_head_profile"),
                "trace_status": profile.get("trace_status"),
                "trace": profile.get("trace"),
                "reference_status": profile.get("reference_status"),
                "reference_kind": profile.get("reference_kind"),
                "reference_planned_ops": profile.get("reference_planned_ops"),
                "reference_observed_ops": profile.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": profile.get(
                    "reference_failed_checks",
                    [],
                ),
                "throughput_summary": profile.get("throughput_summary"),
                "bottleneck_summary": profile.get("bottleneck_summary"),
                "max_section": profile.get("max_section"),
            },
            "decode_step_autotune": {
                "status": autotune.get("status"),
                "candidate_count": autotune.get("candidate_count"),
                "passed_candidate_count": autotune.get(
                    "passed_candidate_count"
                ),
                "failed_candidate_count": autotune.get(
                    "failed_candidate_count"
                ),
                "best": autotune.get("best"),
                "best_reference_status": autotune.get(
                    "best_reference_status"
                ),
                "status_counts": autotune.get("status_counts", {}),
                "reference_status_counts": autotune.get(
                    "reference_status_counts",
                    {},
                ),
                "trace_status_counts": autotune.get(
                    "trace_status_counts",
                    {},
                ),
            },
        },
        "acceptance": {
            "status": acceptance.get("status"),
            "passed": acceptance.get("passed"),
            "check_count": len(acceptance.get("checks", [])),
            "failed_checks": [check.get("name") for check in failed_checks],
        },
    }


def _artifact_evidence(name: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "directory"
    else:
        kind = "missing"
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
    }


def _tensorization_evidence(step: dict[str, Any]) -> dict[str, Any]:
    tensorization = _step_tensorization_summary(step)
    return {
        "status": tensorization.get("status"),
        "roles": tensorization.get("roles"),
        "tensor_count": tensorization.get("tensor_count"),
        "target_dtype_counts": tensorization.get("target_dtype_counts", {}),
        "layout_counts": tensorization.get("layout_counts", {}),
        "memory_config_counts": tensorization.get("memory_config_counts", {}),
        "ttnn_dtype_counts": tensorization.get("ttnn_dtype_counts", {}),
        "ttnn_layout_counts": tensorization.get("ttnn_layout_counts", {}),
        "ttnn_memory_config_counts": tensorization.get(
            "ttnn_memory_config_counts",
            {},
        ),
        "tensor_paths": tensorization.get("tensor_paths", []),
        "required_tensorized_tensor_paths": step.get(
            "required_tensorized_tensor_paths",
            [],
        ),
        "missing_required_tensorized_tensor_paths": step.get(
            "missing_required_tensorized_tensor_paths",
            [],
        ),
        "key_paths": tensorization.get("key_paths", []),
        "key_tensors": tensorization.get("key_tensors", {}),
    }


def _real_decode_acceptance(
    report: dict[str, Any],
    *,
    require_trace: bool,
    min_tokens_per_second_per_user: float | None,
    require_decode_shell_numeric_reference: bool,
) -> dict[str, Any]:
    if report.get("dry_run"):
        return {
            "status": "dry_run",
            "passed": True,
            "require_trace": require_trace,
            "min_tokens_per_second_per_user": (
                min_tokens_per_second_per_user
            ),
            "require_decode_shell_numeric_reference": (
                require_decode_shell_numeric_reference
            ),
            "checks": [],
            "message": "Dry run only; runtime acceptance was not evaluated.",
        }

    steps = report.get("steps", {})
    materialize = steps.get("materialize_parameters", {})
    decode_shell = steps.get("decode_shell", {})
    smoke = steps.get("smoke_decode_step", {})
    profile = steps.get("profile_decode_step", {})
    smoke_tensorization = _step_tensorization_summary(smoke)
    profile_tensorization = _step_tensorization_summary(profile)
    smoke_environment = _step_ttnn_environment(smoke)
    profile_environment = _step_ttnn_environment(profile)
    smoke_trace = _step_trace_summary(smoke)
    profile_trace = _step_trace_summary(profile)
    expected_layers = report.get("layers")
    expected_layer_ids = _expected_layer_ids(expected_layers)
    expected_batch_size = report.get("batch_size")
    expected_cache_len = report.get("cache_len")
    expected_trace_iterations = report.get("trace_iterations")
    throughput = profile.get("throughput_summary") or {}
    profile_section_latency = profile.get("section_latency_ms")
    profile_layer_profiles = profile.get("layer_profiles")
    profile_bottleneck = profile.get("bottleneck_summary")
    checks = [
        _acceptance_check(
            "materialize_parameters.tensor_count",
            _positive_number(materialize.get("tensor_count")),
            observed=materialize.get("tensor_count"),
            minimum=1,
        ),
        _acceptance_check(
            "materialize_parameters.layer_ids",
            materialize.get("materialized_layer_ids") == expected_layer_ids,
            observed=materialize.get("materialized_layer_ids"),
            expected=expected_layer_ids,
        ),
        _acceptance_check(
            "materialize_parameters.lm_head_split_count",
            _positive_number(materialize.get("lm_head_split_count")),
            observed=materialize.get("lm_head_split_count"),
            minimum=1,
        ),
        _acceptance_check(
            "materialize_parameters.required_tensor_paths",
            materialize.get("missing_required_tensor_paths") == [],
            observed=materialize.get("missing_required_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "decode_shell.layers",
            _int_equal(decode_shell.get("layers"), expected_layers),
            observed=decode_shell.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "decode_shell.parameter_source",
            decode_shell.get("parameter_source") == "hf_model",
            observed=decode_shell.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "decode_shell.reference_status",
            decode_shell.get("reference_status") == "passed",
            observed=decode_shell.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "decode_shell.observed_op_sequence",
            _observed_ops_cover_planned(
                decode_shell.get("reference_planned_ops"),
                decode_shell.get("reference_observed_ops"),
            ),
            observed=decode_shell.get("reference_observed_ops"),
            expected=decode_shell.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "smoke_decode_step.parameter_source",
            smoke.get("parameter_source") == "hf_model",
            observed=smoke.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "smoke_decode_step.layers",
            _int_equal(smoke.get("layers"), expected_layers),
            observed=smoke.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "smoke_decode_step.batch_size",
            _int_equal(smoke.get("batch_size"), expected_batch_size),
            observed=smoke.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "smoke_decode_step.cache_len",
            _int_equal(smoke.get("cache_len"), expected_cache_len),
            observed=smoke.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "smoke_decode_step.tensor_conversion_count",
            _positive_number(smoke.get("tensor_conversion_count")),
            observed=smoke.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.ttnn_module_available",
            smoke_environment.get("module_available") is True,
            observed=smoke_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "smoke_decode_step.ttnn_version",
            _non_empty_string(smoke_environment.get("version")),
            observed=smoke_environment.get("version"),
            required=True,
        ),
        _acceptance_check(
            "smoke_decode_step.tt_metal_git_commit",
            _non_empty_string(
                smoke_environment.get("tt_metal_git_commit")
            ),
            observed=smoke_environment.get("tt_metal_git_commit"),
            source=smoke_environment.get("tt_metal_git_commit_source"),
            required=True,
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_status",
            smoke_tensorization.get("status") == "pass",
            observed=smoke_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_roles",
            _contains_all(
                smoke_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=smoke_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "smoke_decode_step.required_tensorized_tensor_paths",
            smoke.get("missing_required_tensorized_tensor_paths") == [],
            observed=smoke.get("missing_required_tensorized_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_memory_configs",
            _positive_count(smoke_tensorization.get("memory_config_counts")),
            observed=smoke_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.tensorization_ttnn_memory_configs",
            _positive_count(
                smoke_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=smoke_tensorization.get("ttnn_memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "smoke_decode_step.reference_status",
            smoke.get("reference_status") == "passed",
            observed=smoke.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "smoke_decode_step.observed_op_sequence",
            _observed_ops_cover_planned(
                smoke.get("reference_planned_ops"),
                smoke.get("reference_observed_ops"),
            ),
            observed=smoke.get("reference_observed_ops"),
            expected=smoke.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "profile_decode_step.parameter_source",
            profile.get("parameter_source") == "hf_model",
            observed=profile.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "profile_decode_step.layers",
            _int_equal(profile.get("layers"), expected_layers),
            observed=profile.get("layers"),
            expected=expected_layers,
        ),
        _acceptance_check(
            "profile_decode_step.batch_size",
            _int_equal(profile.get("batch_size"), expected_batch_size),
            observed=profile.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "profile_decode_step.cache_len",
            _int_equal(profile.get("cache_len"), expected_cache_len),
            observed=profile.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "profile_decode_step.tensor_conversion_count",
            _positive_number(profile.get("tensor_conversion_count")),
            observed=profile.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.tensor_conversion_ms",
            _nonnegative_number(profile.get("tensor_conversion_ms")),
            observed=profile.get("tensor_conversion_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.ttnn_module_available",
            profile_environment.get("module_available") is True,
            observed=profile_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "profile_decode_step.ttnn_version",
            _non_empty_string(profile_environment.get("version")),
            observed=profile_environment.get("version"),
            required=True,
        ),
        _acceptance_check(
            "profile_decode_step.tt_metal_git_commit",
            _non_empty_string(
                profile_environment.get("tt_metal_git_commit")
            ),
            observed=profile_environment.get("tt_metal_git_commit"),
            source=profile_environment.get("tt_metal_git_commit_source"),
            required=True,
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_status",
            profile_tensorization.get("status") == "pass",
            observed=profile_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_roles",
            _contains_all(
                profile_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=profile_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "profile_decode_step.required_tensorized_tensor_paths",
            profile.get("missing_required_tensorized_tensor_paths") == [],
            observed=profile.get("missing_required_tensorized_tensor_paths"),
            expected=[],
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_memory_configs",
            _positive_count(profile_tensorization.get("memory_config_counts")),
            observed=profile_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.tensorization_ttnn_memory_configs",
            _positive_count(
                profile_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=profile_tensorization.get("ttnn_memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "profile_decode_step.reference_status",
            profile.get("reference_status") == "passed",
            observed=profile.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "profile_decode_step.observed_op_sequence",
            _observed_ops_cover_planned(
                profile.get("reference_planned_ops"),
                profile.get("reference_observed_ops"),
            ),
            observed=profile.get("reference_observed_ops"),
            expected=profile.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "profile_decode_step.section_latency_ms",
            _has_nonnegative_fields(
                profile_section_latency,
                PROFILE_SECTION_LATENCY_KEYS,
            ),
            observed=_field_keys(profile_section_latency),
            expected=list(PROFILE_SECTION_LATENCY_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.layer_profile_count",
            _layer_profile_ids(profile_layer_profiles) == expected_layer_ids,
            observed=_layer_profile_ids(profile_layer_profiles),
            expected=expected_layer_ids,
        ),
        _acceptance_check(
            "profile_decode_step.layer_profile_sections",
            _layer_profiles_have_nonnegative_fields(
                profile_layer_profiles,
                PROFILE_LAYER_LATENCY_KEYS,
            ),
            observed=_layer_profile_field_keys(profile_layer_profiles),
            expected=list(PROFILE_LAYER_LATENCY_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.bottleneck_summary",
            _bottleneck_summary_complete(profile_bottleneck),
            observed=_bottleneck_summary_observed(profile_bottleneck),
            expected=list(PROFILE_BOTTLENECK_SECTION_KEYS),
        ),
        _acceptance_check(
            "profile_decode_step.throughput_status",
            throughput.get("status") == "measured",
            observed=throughput.get("status"),
            expected="measured",
        ),
        _acceptance_check(
            "profile_decode_step.latency_ms",
            _positive_number(throughput.get("latency_ms")),
            observed=throughput.get("latency_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.tokens_per_second_per_user",
            _positive_number(
                throughput.get("tokens_per_second_per_user")
            ),
            observed=throughput.get("tokens_per_second_per_user"),
            minimum=0,
        ),
        _acceptance_check(
            "profile_decode_step.aggregate_tokens_per_second",
            _positive_number(
                throughput.get("aggregate_tokens_per_second")
            ),
            observed=throughput.get("aggregate_tokens_per_second"),
            minimum=0,
        ),
    ]
    if require_decode_shell_numeric_reference:
        checks.append(
            _acceptance_check(
                "decode_shell.numeric_reference_status",
                decode_shell.get("numeric_reference_status") == "passed",
                observed=decode_shell.get("numeric_reference_status"),
                expected="passed",
            )
        )
    if require_trace:
        checks.extend(
            [
                _acceptance_check(
                    "smoke_decode_step.trace_status",
                    smoke.get("trace_status") == "captured_and_executed",
                    observed=smoke.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "smoke_decode_step.trace_iterations",
                    _int_equal(
                        smoke_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=smoke_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "smoke_decode_step.trace_execute_sample_count",
                    _int_equal(
                        smoke_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=smoke_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "profile_decode_step.trace_status",
                    profile.get("trace_status") == "captured_and_executed",
                    observed=profile.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "profile_decode_step.trace_iterations",
                    _int_equal(
                        profile_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=profile_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "profile_decode_step.trace_execute_sample_count",
                    _int_equal(
                        profile_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=profile_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    (
                        "profile_decode_step."
                        "trace_execute_tokens_per_second_per_user"
                    ),
                    _positive_number(
                        throughput.get(
                            "trace_execute_tokens_per_second_per_user"
                        )
                    ),
                    observed=throughput.get(
                        "trace_execute_tokens_per_second_per_user"
                    ),
                    minimum=0,
                ),
            ]
        )

    if min_tokens_per_second_per_user is not None:
        observed = throughput.get("tokens_per_second_per_user")
        checks.append(
            _acceptance_check(
                "profile_decode_step.min_tokens_per_second_per_user",
                _number_at_least(observed, min_tokens_per_second_per_user),
                observed=observed,
                minimum=min_tokens_per_second_per_user,
            )
        )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "require_trace": require_trace,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "require_decode_shell_numeric_reference": (
            require_decode_shell_numeric_reference
        ),
        "checks": checks,
    }


def _acceptance_check(
    name: str,
    passed: bool,
    **details: Any,
) -> dict[str, Any]:
    check = {
        "name": name,
        "passed": bool(passed),
    }
    check.update(details)
    return check


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _has_nonnegative_fields(value: Any, fields: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        field in value and _nonnegative_number(value.get(field))
        for field in fields
    )


def _field_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)


def _layer_profile_ids(layer_profiles: Any) -> list[int]:
    if not isinstance(layer_profiles, list):
        return []
    layer_ids = []
    for profile in layer_profiles:
        if not isinstance(profile, dict):
            return []
        try:
            layer_ids.append(int(profile["layer_id"]))
        except (KeyError, TypeError, ValueError):
            return []
    return layer_ids


def _layer_profiles_have_nonnegative_fields(
    layer_profiles: Any,
    fields: tuple[str, ...],
) -> bool:
    if not isinstance(layer_profiles, list) or not layer_profiles:
        return False
    return all(
        isinstance(profile, dict)
        and _has_nonnegative_fields(profile, fields)
        for profile in layer_profiles
    )


def _layer_profile_field_keys(layer_profiles: Any) -> list[list[str]]:
    if not isinstance(layer_profiles, list):
        return []
    return [
        _field_keys(profile)
        for profile in layer_profiles
        if isinstance(profile, dict)
    ]


def _bottleneck_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    sections = summary.get("sections_ms")
    max_section = summary.get("max_section")
    return (
        _non_empty_string(max_section)
        and isinstance(sections, dict)
        and str(max_section) in sections
        and _nonnegative_number(summary.get("max_section_ms"))
        and _has_nonnegative_fields(
            sections,
            PROFILE_BOTTLENECK_SECTION_KEYS,
        )
    )


def _bottleneck_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    sections = summary.get("sections_ms")
    return {
        "max_section": summary.get("max_section"),
        "max_section_ms": summary.get("max_section_ms"),
        "sections_ms": _field_keys(sections),
    }


def _observed_ops_cover_planned(planned_ops: Any, observed_ops: Any) -> bool:
    if not isinstance(planned_ops, list) or not isinstance(observed_ops, list):
        return False
    planned_index = 0
    for observed in observed_ops:
        if (
            planned_index < len(planned_ops)
            and str(observed) == str(planned_ops[planned_index])
        ):
            planned_index += 1
    return planned_index == len(planned_ops)


def _number_at_least(value: Any, minimum: Any) -> bool:
    try:
        return float(value) >= float(minimum)
    except (TypeError, ValueError):
        return False


def _int_equal(observed: Any, expected: Any) -> bool:
    try:
        return int(observed) == int(expected)
    except (TypeError, ValueError):
        return False


def _expected_layer_ids(layers: Any) -> list[int]:
    try:
        layer_count = int(layers)
    except (TypeError, ValueError):
        return []
    if layer_count <= 0:
        return []
    return list(range(layer_count))


def _step_tensorization_summary(step: dict[str, Any]) -> dict[str, Any]:
    setup = step.get("parameter_setup") or {}
    tensorization = setup.get("tensorization") or {}
    return tensorization if isinstance(tensorization, dict) else {}


def _tensorized_tensor_paths(tensorization: dict[str, Any]) -> list[str]:
    paths = tensorization.get("tensor_paths")
    if isinstance(paths, list):
        return sorted({str(path) for path in paths if path is not None})

    paths = tensorization.get("key_paths")
    if isinstance(paths, list):
        return sorted({str(path) for path in paths if path is not None})

    key_tensors = tensorization.get("key_tensors")
    if isinstance(key_tensors, dict):
        return sorted(str(path) for path in key_tensors)

    return []


def _step_ttnn_environment(step: dict[str, Any]) -> dict[str, Any]:
    environment = step.get("ttnn_environment") or {}
    return environment if isinstance(environment, dict) else {}


def _step_trace_summary(step: dict[str, Any]) -> dict[str, Any]:
    trace = step.get("trace") or {}
    return trace if isinstance(trace, dict) else {}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _contains_all(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, list):
        return False
    return set(expected).issubset(set(observed))


def _positive_count(counts: Any) -> bool:
    if not isinstance(counts, dict):
        return False
    total = 0
    for count in counts.values():
        try:
            total += int(count)
        except (TypeError, ValueError):
            return False
    return total > 0


def _reference_summary(runtime_report: dict[str, Any]) -> dict[str, Any]:
    reference = runtime_report.get("reference") or {}
    checks = reference.get("checks") or []
    return {
        "reference_status": reference.get("status"),
        "reference_kind": reference.get("kind"),
        "reference_planned_ops": reference.get("planned_ops"),
        "reference_observed_ops": reference.get("observed_ops"),
        "reference_failed_checks": [
            check.get("name")
            for check in checks
            if isinstance(check, dict) and not check.get("passed")
        ],
    }


def _candidate_reference_status_counts(report: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in report.get("candidates", []):
        status = candidate.get("reference_status")
        if status is None:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def dump_validation_report(report: dict[str, Any], out: str | Path) -> None:
    _write_json(Path(out), report)


def _require(value: Any, step: str) -> None:
    if value is None:
        raise RuntimeError(f"validate-direct requires {step} to pass first")


def _mark_remaining_skipped(
    report: dict[str, Any],
    failed_step: str,
    steps: tuple[str, ...] = VALIDATION_STEPS,
) -> None:
    seen_failed = False
    for step in steps:
        if step == failed_step:
            seen_failed = True
            continue
        if not seen_failed:
            continue
        if report["results"].get(step) == "pending":
            report["results"][step] = "skipped"
            report["steps"][step] = {
                "status": "skipped",
                "reason": f"blocked by failed step: {failed_step}",
            }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
