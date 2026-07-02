from __future__ import annotations

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
    "smoke_decode_step",
    "profile_decode_step",
    "decode_step_autotune",
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
            "smoke_status": smoke_report["status"],
            "op_count": len(smoke_report["op_sequence"]),
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
            "profile_status": profile_report["status"],
            "bottleneck": profile_report["bottleneck_summary"]["max_section"],
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
    decode_step_search_space_path = (
        Path(decode_step_search_space_path)
        if decode_step_search_space_path is not None
        else default_decode_step_search_space_path()
    )
    layers_to_materialize = list(range(layer_count))

    paths = {
        "materialize_report": root / "parameter_materialization_report.json",
        "smoke_report": root / "decode_step_smoke_report.json",
        "profile_report": root / "decode_step_profile_report.json",
        "autotune_report": root / "decode_step_autotune_report.json",
        "autotune_candidates_dir": root / "decode_step_autotune_candidates",
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
        "layers": layer_count,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations,
        "metric": metric,
        "dry_run": dry_run,
        "skip_autotune": skip_autotune,
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
            materialize_report = {
                "schema_version": 1,
                "status": "dry_run",
                "backend": "torch",
                "model_path": str(model_path),
                "program_dir": str(program_dir),
                "materialized_layer_ids": layers_to_materialize,
                "message": "Dry run only; safetensors payloads were not loaded.",
            }
            _write_json(paths["materialize_report"], materialize_report)
            return {
                "status": "dry_run",
                "materialize_report": str(paths["materialize_report"]),
                "materialized_layer_ids": layers_to_materialize,
            }

        materialize_report = materialize_parameters_from_program(
            model_path=model_path,
            program_dir=program_dir,
            backend="torch",
            layers=layers_to_materialize,
            out=paths["materialize_report"],
        )
        return {
            "status": "pass",
            "materialize_report": str(paths["materialize_report"]),
            "materialized_layer_ids": list(
                materialize_report["materialized_layer_ids"]
            ),
            "tensor_count": materialize_report["tensor_count"],
            "lm_head_split_count": materialize_report["lm_head"]["split_count"],
        }

    def smoke_step() -> dict[str, Any]:
        smoke_report = run_smoke_decode_step(
            out=paths["smoke_report"],
            program_dir=program_dir,
            layers=layer_count,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
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
            "parameter_source": smoke_report.get("parameter_source"),
            "input_source": smoke_report.get("input_source"),
            "tensor_conversion_count": smoke_report.get(
                "tensor_conversion_count"
            ),
            "trace_status": smoke_report.get("trace", {}).get("status"),
        }

    def profile_step() -> dict[str, Any]:
        profile_report = profile_decode_step(
            out=paths["profile_report"],
            program_dir=program_dir,
            layers=layer_count,
            model_path=None if dry_run else model_path,
            device=device,
            device_id=device_id,
            batch_size=batch_size,
            cache_len=cache_len,
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
            "parameter_source": profile_report.get("parameter_source"),
            "input_source": profile_report.get("input_source"),
            "tensor_conversion_count": profile_report.get(
                "tensor_conversion_count"
            ),
            "tensor_conversion_ms": profile_report.get("tensor_conversion_ms"),
            "max_section": bottleneck.get("max_section"),
            "trace_status": profile_report.get("trace", {}).get("status"),
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
            batch_size=batch_size,
            cache_len=cache_len,
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
            "best": autotune_report.get("best", {}).get("id")
            if autotune_report.get("best") is not None
            else None,
            "dry_run": autotune_report["dry_run"],
        }

    step_actions = {
        "materialize_parameters": materialize_step,
        "smoke_decode_step": smoke_step,
        "profile_decode_step": profile_step,
        "decode_step_autotune": autotune_step,
    }

    for step in REAL_DECODE_VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            return report

    report["status"] = "dry_run" if dry_run else "pass"
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
