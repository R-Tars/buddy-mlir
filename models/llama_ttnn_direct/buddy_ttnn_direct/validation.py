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
    PARITY_SECTIONS,
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
from .search.decode_step_autotune import (
    DECODE_STEP_AUTOTUNE_KNOBS,
    run_decode_step_autotune,
)
from .search.report import dump_search_report
from .search.runner import run_lm_head_search
from .search.space import load_search_space
from .semantic.dump import dump_graph_json
from .semantic.graph import LlamaModelGraph
from .semantic.importer_hf_llama import import_hf_llama
from .smoke_attention_layer import ATTENTION_LAYER_OPS, run_smoke_attention_layer
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
    "official_config_diff",
    "materialize_parameters",
    "decode_shell",
    "attention_primitives",
    "attention_layer",
    "single_layer_decode",
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
            "knob_coverage": autotune_report.get("knob_coverage"),
            "search_space": autotune_report.get("search_space"),
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
    official_config_path: str | Path | None = None,
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
    require_official_config_match: bool = False,
    require_full_depth: bool = False,
    require_program_runtime_shape: bool = False,
    require_batch32_decode_step: bool = False,
    min_tokens_per_second_per_user: float | None = None,
    baseline_tokens_per_second_per_user: float | None = None,
    min_baseline_ratio: float | None = None,
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
    if (
        baseline_tokens_per_second_per_user is not None
        and baseline_tokens_per_second_per_user <= 0.0
    ):
        raise ValueError("baseline_tokens_per_second_per_user must be positive")
    if min_baseline_ratio is not None:
        if min_baseline_ratio < 0.0:
            raise ValueError("min_baseline_ratio must be nonnegative")
        if baseline_tokens_per_second_per_user is None:
            raise ValueError(
                "baseline_tokens_per_second_per_user is required when "
                "min_baseline_ratio is set"
            )

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "real_decode_validation_report.json"
    program_dir = Path(program_dir)
    model_path = Path(model_path)
    program_config = _load_program_config(program_dir)
    program_num_layers = int(program_config["num_layers"])
    program_batch_size = int(program_config["batch_size"])
    program_cache_len = int(program_config["max_cache_len"])
    program_seq_len = int(program_config.get("seq_len", 1))
    program_hidden_size = int(program_config["hidden_size"])
    program_vocab_size = int(program_config["vocab_size"])
    program_num_attention_heads = int(program_config["num_attention_heads"])
    program_num_kv_heads = int(program_config["num_key_value_heads"])
    program_head_dim = int(program_config["head_dim"])
    program_template_config = (
        program_config.get("template_config")
        if isinstance(program_config.get("template_config"), dict)
        else {}
    )
    program_generation = (
        program_config.get("generation")
        if isinstance(program_config.get("generation"), dict)
        else {}
    )
    program_kv_cache = (
        program_config.get("kv_cache")
        if isinstance(program_config.get("kv_cache"), dict)
        else _kv_cache_contract_from_template_config(
            program_template_config,
            cache_len=program_cache_len,
            num_kv_heads=program_num_kv_heads,
            head_dim=program_head_dim,
        )
    )
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
    official_config_path = (
        Path(official_config_path)
        if official_config_path is not None
        else default_official_parity_config_path()
    )
    layers_to_materialize = list(range(layer_count))
    decode_step_contract = _decode_step_contract(
        layer_count=layer_count,
        batch_size=resolved_batch_size,
        seq_len=program_seq_len,
        cache_len=resolved_cache_len,
        num_kv_heads=program_num_kv_heads,
        head_dim=program_head_dim,
        kv_cache=program_kv_cache,
        generation=program_generation,
    )

    paths = {
        "official_config_diff": root / "official_config_diff.json",
        "materialize_report": root / "parameter_materialization_report.json",
        "decode_shell_report": root / "decode_shell_report.json",
        "attention_primitives_dir": root / "attention_primitives",
        "attention_layer_report": root / "attention_layer_report.json",
        "single_layer_decode_report": root / "single_layer_decode_report.json",
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
        "official_config": str(official_config_path),
        "decode_step_search_space": str(decode_step_search_space_path),
        "program_num_layers": program_num_layers,
        "program_batch_size": program_batch_size,
        "program_cache_len": program_cache_len,
        "program_seq_len": program_seq_len,
        "program_hidden_size": program_hidden_size,
        "program_vocab_size": program_vocab_size,
        "program_num_attention_heads": program_num_attention_heads,
        "program_num_key_value_heads": program_num_kv_heads,
        "program_head_dim": program_head_dim,
        "program_generation": program_generation,
        "program_kv_cache": program_kv_cache,
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
        "require_official_config_match": require_official_config_match,
        "require_full_depth": require_full_depth,
        "require_program_runtime_shape": require_program_runtime_shape,
        "require_batch32_decode_step": require_batch32_decode_step,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "baseline_tokens_per_second_per_user": (
            baseline_tokens_per_second_per_user
        ),
        "min_baseline_ratio": min_baseline_ratio,
        "decode_shell_pcc_threshold": decode_shell_pcc_threshold,
        "require_decode_shell_numeric_reference": (
            require_decode_shell_numeric_reference
        ),
        "results": {
            step: "pending" for step in REAL_DECODE_VALIDATION_STEPS
        },
        "decode_step_contract": decode_step_contract,
        "steps": {},
        "artifacts": {name: str(path) for name, path in paths.items()},
    }

    def persist() -> None:
        _write_json(report_path, report)

    def write_evidence_summary() -> dict[str, Any]:
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
        return evidence

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

    def official_config_diff_step() -> dict[str, Any]:
        diff = diff_official_config(
            program_dir / "config.json",
            official_config_path,
        )
        dump_config_diff(diff, paths["official_config_diff"])
        section_statuses = {
            section: summary.get("status")
            for section, summary in sorted(diff["sections"].items())
            if isinstance(summary, dict)
        }
        return {
            "status": "pass",
            "official_config_diff": str(paths["official_config_diff"]),
            "official_config": str(official_config_path),
            "diff_status": diff["status"],
            "issue_count": diff["summary"]["issue_count"],
            "missing_count": diff["summary"]["missing_count"],
            "mismatch_count": diff["summary"]["mismatch_count"],
            "extra_count": diff["summary"]["extra_count"],
            "matching_count": diff["summary"]["matching_count"],
            "gap_summary": diff.get("gap_summary"),
            "sections": sorted(diff["sections"]),
            "section_statuses": section_statuses,
        }

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
        *,
        required_layer_count: int = layer_count,
    ) -> dict[str, Any]:
        setup = runtime_report.get("parameter_setup") or {}
        tensorization = setup.get("tensorization") or {}
        if not isinstance(tensorization, dict):
            tensorization = {}
        required_tensor_paths = _required_tensorized_tensor_paths(
            layer_count=required_layer_count,
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

    def attention_primitives_step() -> dict[str, Any]:
        primitive_reports = []
        for primitive in ATTENTION_PRIMITIVES:
            report_path = paths["attention_primitives_dir"] / f"{primitive}.json"
            primitive_report = run_smoke_attention_primitive(
                out=report_path,
                primitive=primitive,
                device=device,
                device_id=device_id,
                batch_size=resolved_batch_size,
                hidden_size=program_hidden_size,
                num_heads=program_num_attention_heads,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                max_cache_len=resolved_cache_len,
                dtype_seed=dtype_seed,
                dry_run=dry_run,
                ttnn_module=ttnn_module,
                torch_module=torch_module,
            )
            primitive_reports.append(
                {
                    "report": str(report_path),
                    **primitive_report,
                }
            )
        return {
            "status": _attention_primitives_step_status(
                primitive_reports,
                dry_run=dry_run,
            ),
            "attention_primitives_dir": str(paths["attention_primitives_dir"]),
            "runtime_status_counts": _primitive_runtime_status_counts(
                primitive_reports
            ),
            "primitive_count": len(primitive_reports),
            "primitive_sequence": [
                report.get("primitive")
                for report in primitive_reports
                if isinstance(report, dict)
            ],
            "primitive_reports": primitive_reports,
            "ttnn_environment": _first_ttnn_environment(primitive_reports),
        }

    def attention_layer_step() -> dict[str, Any]:
        layer_report = run_smoke_attention_layer(
            out=paths["attention_layer_report"],
            program_dir=program_dir,
            layer=0,
            device=device,
            device_id=device_id,
            batch_size=resolved_batch_size,
            cache_len=resolved_cache_len,
            dtype_seed=dtype_seed,
            dry_run=dry_run,
            ttnn_module=ttnn_module,
            torch_module=torch_module,
        )
        primitive_reports = layer_report.get("primitive_reports") or []
        return {
            "status": _runtime_step_status(layer_report, dry_run=dry_run),
            "attention_layer_report": str(paths["attention_layer_report"]),
            "runtime_status": layer_report["status"],
            "layer": layer_report.get("layer"),
            "batch_size": layer_report.get("batch_size"),
            "cache_len": layer_report.get("cache_len"),
            "hidden_size": layer_report.get("hidden_size"),
            "num_heads": layer_report.get("num_heads"),
            "num_kv_heads": layer_report.get("num_kv_heads"),
            "head_dim": layer_report.get("head_dim"),
            "latency_ms": layer_report.get("latency_ms"),
            "primitive_count": len(primitive_reports),
            "primitive_sequence": [
                primitive.get("primitive")
                for primitive in primitive_reports
                if isinstance(primitive, dict)
            ],
            "primitive_reports": primitive_reports,
            "output_shapes": layer_report.get("output_shapes"),
            "tensor_conversion_count": layer_report.get(
                "tensor_conversion_count"
            ),
            "memory_config_conversion_count": layer_report.get(
                "memory_config_conversion_count"
            ),
            "ttnn_environment": layer_report.get("ttnn_environment"),
            **_reference_summary(layer_report),
        }

    def single_layer_decode_step() -> dict[str, Any]:
        single_layer_report = run_smoke_single_layer_decode(
            out=paths["single_layer_decode_report"],
            program_dir=program_dir,
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
            "status": _runtime_step_status(
                single_layer_report,
                dry_run=dry_run,
            ),
            "single_layer_decode_report": str(
                paths["single_layer_decode_report"]
            ),
            "runtime_status": single_layer_report["status"],
            "layers": single_layer_report.get("layers"),
            "batch_size": single_layer_report.get("batch_size"),
            "cache_len": single_layer_report.get("cache_len"),
            "parameter_source": single_layer_report.get("parameter_source"),
            "input_source": single_layer_report.get("input_source"),
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(single_layer_report)
            ),
            "tensor_conversion_count": single_layer_report.get(
                "tensor_conversion_count"
            ),
            "output_shapes": single_layer_report.get("output_shapes"),
            "trace_status": single_layer_report.get("trace", {}).get("status"),
            "trace": _trace_summary(single_layer_report.get("trace")),
            "ttnn_environment": single_layer_report.get("ttnn_environment"),
            "parameter_setup": single_layer_report.get("parameter_setup"),
            **tensorization_path_detail(
                single_layer_report,
                required_layer_count=1,
            ),
            **_reference_summary(single_layer_report),
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
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(smoke_report)
            ),
            "tensor_conversion_count": smoke_report.get(
                "tensor_conversion_count"
            ),
            "output_shapes": smoke_report.get("output_shapes"),
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
            "synthetic_runtime_input_tensor_count": (
                _step_synthetic_runtime_input_count(profile_report)
            ),
            "tensor_conversion_count": profile_report.get(
                "tensor_conversion_count"
            ),
            "tensor_conversion_ms": profile_report.get("tensor_conversion_ms"),
            "latency_ms": profile_report.get("latency_ms"),
            "section_latency_ms": profile_report.get("section_latency_ms"),
            "layer_profiles": profile_report.get("layer_profiles", []),
            "lm_head_profile": profile_report.get("lm_head_profile"),
            "output_shapes": profile_report.get("output_shapes"),
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
            "best_trace_status": (
                autotune_report.get("best", {}).get("trace_status")
                if autotune_report.get("best") is not None
                else None
            ),
            "best_parameter_source": (
                autotune_report.get("best", {}).get("parameter_source")
                if autotune_report.get("best") is not None
                else None
            ),
            "best_metric": (
                autotune_report.get("best", {}).get("metric")
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
            "output_kind_counts": autotune_report.get(
                "output_kind_counts",
                {},
            ),
            "knob_coverage": autotune_report.get("knob_coverage"),
            "search_space": autotune_report.get("search_space"),
            "best_output_kind": (
                autotune_report.get("best", {}).get("output_kind")
                if autotune_report.get("best") is not None
                else None
            ),
            "dry_run": autotune_report["dry_run"],
        }

    step_actions = {
        "official_config_diff": official_config_diff_step,
        "materialize_parameters": materialize_step,
        "decode_shell": decode_shell_step,
        "attention_primitives": attention_primitives_step,
        "attention_layer": attention_layer_step,
        "single_layer_decode": single_layer_decode_step,
        "smoke_decode_step": smoke_step,
        "profile_decode_step": profile_step,
        "decode_step_autotune": autotune_step,
    }

    for step in REAL_DECODE_VALIDATION_STEPS:
        if not run_step(step, step_actions[step]):
            write_evidence_summary()
            return report

    acceptance = _real_decode_acceptance(
        report,
        require_trace=require_trace,
        require_official_config_match=require_official_config_match,
        require_full_depth=require_full_depth,
        require_program_runtime_shape=require_program_runtime_shape,
        require_batch32_decode_step=require_batch32_decode_step,
        min_tokens_per_second_per_user=min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            baseline_tokens_per_second_per_user
        ),
        min_baseline_ratio=min_baseline_ratio,
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
    write_evidence_summary()
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


def _attention_primitives_step_status(
    primitive_reports: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry_run"
    if all(report.get("passed") is True for report in primitive_reports):
        return "pass"
    for report in primitive_reports:
        if report.get("passed") is not True:
            return str(report.get("status", "fail"))
    return "fail"


def _primitive_runtime_status_counts(
    primitive_reports: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in primitive_reports:
        status = str(report.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _first_ttnn_environment(
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for report in reports:
        environment = report.get("ttnn_environment")
        if isinstance(environment, dict):
            return environment
    return None


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
    if isinstance(execute_samples, list):
        summary["execute_samples_ms"] = execute_samples
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


def _lm_head_source_reference_complete(materialize: Any) -> bool:
    observed = _lm_head_source_reference_observed(materialize)
    lm_head = observed.get("lm_head.weight")
    split0 = observed.get("lm_head.splits.0.weight")
    if not isinstance(lm_head, dict) or not isinstance(split0, dict):
        return False
    return (
        lm_head.get("materialization") == "metadata_reference"
        and lm_head.get("materialized") is False
        and split0.get("source_read") == "sliced_tensor"
    )


def _lm_head_source_reference_observed(materialize: Any) -> dict[str, Any]:
    if not isinstance(materialize, dict):
        return {}
    key_tensors = materialize.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return {}
    observed: dict[str, Any] = {}
    for path in ("lm_head.weight", "lm_head.splits.0.weight"):
        tensor = key_tensors.get(path)
        if not isinstance(tensor, dict):
            continue
        observed[path] = {
            "shape": tensor.get("shape"),
            "materialization": tensor.get("materialization"),
            "materialized": tensor.get("materialized"),
            "source_read": tensor.get("source_read"),
        }
    return observed


def _real_decode_evidence_manifest(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    steps = report.get("steps", {})
    official_config_diff = steps.get("official_config_diff", {})
    materialize = steps.get("materialize_parameters", {})
    decode_shell = steps.get("decode_shell", {})
    attention_primitives = steps.get("attention_primitives", {})
    attention_layer = steps.get("attention_layer", {})
    single_layer = steps.get("single_layer_decode", {})
    smoke = steps.get("smoke_decode_step", {})
    profile = steps.get("profile_decode_step", {})
    autotune = steps.get("decode_step_autotune", {})
    acceptance = report.get("acceptance", {})
    decode_step_contract = report.get("decode_step_contract") or {}
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
    results = report.get("results") or {}

    return {
        "schema_version": 1,
        "status": status,
        "validation": {
            "command": report.get("command"),
            "status": report.get("status"),
            "dry_run": report.get("dry_run"),
            "program_dir": report.get("program_dir"),
            "model_path": report.get("model_path"),
            "official_config": report.get("official_config"),
            "program_num_layers": report.get("program_num_layers"),
            "program_batch_size": report.get("program_batch_size"),
            "program_cache_len": report.get("program_cache_len"),
            "program_seq_len": report.get("program_seq_len"),
            "program_hidden_size": report.get("program_hidden_size"),
            "program_vocab_size": report.get("program_vocab_size"),
            "program_num_attention_heads": report.get(
                "program_num_attention_heads"
            ),
            "program_num_key_value_heads": report.get(
                "program_num_key_value_heads"
            ),
            "program_head_dim": report.get("program_head_dim"),
            "program_generation": report.get("program_generation"),
            "program_kv_cache": report.get("program_kv_cache"),
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
            "baseline_tokens_per_second_per_user": report.get(
                "baseline_tokens_per_second_per_user"
            ),
            "min_baseline_ratio": report.get("min_baseline_ratio"),
            "require_official_config_match": report.get(
                "require_official_config_match"
            ),
            "require_full_depth": report.get("require_full_depth"),
            "require_program_runtime_shape": report.get(
                "require_program_runtime_shape"
            ),
            "require_batch32_decode_step": report.get(
                "require_batch32_decode_step"
            ),
            "results": dict(results),
            "failed_steps": _step_names_with_status(
                results,
                failing=True,
            ),
            "skipped_steps": _step_names_with_status(
                results,
                status="skipped",
            ),
        },
        "requirements": {
            "require_official_config_match": report.get(
                "require_official_config_match"
            ),
            "require_full_depth": report.get("require_full_depth"),
            "require_program_runtime_shape": report.get(
                "require_program_runtime_shape"
            ),
            "require_batch32_decode_step": report.get(
                "require_batch32_decode_step"
            ),
            "require_trace": report.get("require_trace"),
            "min_tokens_per_second_per_user": report.get(
                "min_tokens_per_second_per_user"
            ),
            "baseline_tokens_per_second_per_user": report.get(
                "baseline_tokens_per_second_per_user"
            ),
            "min_baseline_ratio": report.get("min_baseline_ratio"),
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
            "attention_primitives_ttnn_environment": (
                attention_primitives.get("ttnn_environment")
            ),
            "attention_layer_ttnn_environment": attention_layer.get(
                "ttnn_environment"
            ),
            "single_layer_ttnn_environment": single_layer.get(
                "ttnn_environment"
            ),
            "smoke_ttnn_environment": smoke.get("ttnn_environment"),
            "profile_ttnn_environment": profile.get("ttnn_environment"),
        },
        "performance_evidence": {
            "throughput_baseline": _throughput_baseline_summary(
                report,
                profile,
            ),
        },
        "decode_step_contract": decode_step_contract,
        "config_evidence": {
            "official_config_diff": {
                "status": official_config_diff.get("status"),
                "report": official_config_diff.get("official_config_diff"),
                "official_config": official_config_diff.get(
                    "official_config"
                ),
                "diff_status": official_config_diff.get("diff_status"),
                "issue_count": official_config_diff.get("issue_count"),
                "missing_count": official_config_diff.get("missing_count"),
                "mismatch_count": official_config_diff.get("mismatch_count"),
                "extra_count": official_config_diff.get("extra_count"),
                "matching_count": official_config_diff.get("matching_count"),
                "gap_summary": official_config_diff.get("gap_summary"),
                "sections": official_config_diff.get("sections", []),
                "section_statuses": official_config_diff.get(
                    "section_statuses",
                    {},
                ),
            },
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
            "single_layer_tensorization": _tensorization_evidence(
                single_layer
            ),
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
                "runtime_input_tensor_count": decode_shell.get(
                    "runtime_input_tensor_count"
                ),
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
            "attention_primitives": {
                "status": attention_primitives.get("status"),
                "runtime_status_counts": attention_primitives.get(
                    "runtime_status_counts",
                    {},
                ),
                "primitive_count": attention_primitives.get(
                    "primitive_count"
                ),
                "primitive_sequence": attention_primitives.get(
                    "primitive_sequence"
                ),
                "primitive_reports": attention_primitives.get(
                    "primitive_reports",
                    [],
                ),
            },
            "attention_layer": {
                "status": attention_layer.get("status"),
                "runtime_status": attention_layer.get("runtime_status"),
                "layer": attention_layer.get("layer"),
                "batch_size": attention_layer.get("batch_size"),
                "cache_len": attention_layer.get("cache_len"),
                "hidden_size": attention_layer.get("hidden_size"),
                "num_heads": attention_layer.get("num_heads"),
                "num_kv_heads": attention_layer.get("num_kv_heads"),
                "head_dim": attention_layer.get("head_dim"),
                "latency_ms": attention_layer.get("latency_ms"),
                "primitive_count": attention_layer.get("primitive_count"),
                "primitive_sequence": attention_layer.get(
                    "primitive_sequence"
                ),
                "primitive_reports": attention_layer.get(
                    "primitive_reports",
                    [],
                ),
                "output_shapes": attention_layer.get("output_shapes"),
                "tensor_conversion_count": attention_layer.get(
                    "tensor_conversion_count"
                ),
                "memory_config_conversion_count": attention_layer.get(
                    "memory_config_conversion_count"
                ),
                "reference_status": attention_layer.get("reference_status"),
                "reference_kind": attention_layer.get("reference_kind"),
                "reference_planned_ops": attention_layer.get(
                    "reference_planned_ops"
                ),
                "reference_planned_observed_ops": attention_layer.get(
                    "reference_planned_observed_ops"
                ),
                "reference_observed_ops": attention_layer.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": attention_layer.get(
                    "reference_failed_checks",
                    [],
                ),
            },
            "single_layer_decode": {
                "status": single_layer.get("status"),
                "runtime_status": single_layer.get("runtime_status"),
                "layers": single_layer.get("layers"),
                "batch_size": single_layer.get("batch_size"),
                "cache_len": single_layer.get("cache_len"),
                "parameter_source": single_layer.get("parameter_source"),
                "input_source": single_layer.get("input_source"),
                "synthetic_runtime_input_tensor_count": single_layer.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "tensor_conversion_count": single_layer.get(
                    "tensor_conversion_count"
                ),
                "output_shapes": single_layer.get("output_shapes"),
                "trace_status": single_layer.get("trace_status"),
                "trace": single_layer.get("trace"),
                "reference_status": single_layer.get("reference_status"),
                "reference_kind": single_layer.get("reference_kind"),
                "reference_planned_ops": single_layer.get(
                    "reference_planned_ops"
                ),
                "reference_observed_ops": single_layer.get(
                    "reference_observed_ops"
                ),
                "reference_failed_checks": single_layer.get(
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
                "synthetic_runtime_input_tensor_count": smoke.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "tensor_conversion_count": smoke.get(
                    "tensor_conversion_count"
                ),
                "output_shapes": smoke.get("output_shapes"),
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
                "synthetic_runtime_input_tensor_count": profile.get(
                    "synthetic_runtime_input_tensor_count"
                ),
                "tensor_conversion_count": profile.get(
                    "tensor_conversion_count"
                ),
                "tensor_conversion_ms": profile.get("tensor_conversion_ms"),
                "latency_ms": profile.get("latency_ms"),
                "section_latency_ms": profile.get("section_latency_ms"),
                "layer_profiles": profile.get("layer_profiles", []),
                "lm_head_profile": profile.get("lm_head_profile"),
                "output_shapes": profile.get("output_shapes"),
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
                "best_trace_status": autotune.get("best_trace_status"),
                "best_parameter_source": autotune.get(
                    "best_parameter_source"
                ),
                "best_metric": autotune.get("best_metric"),
                "best_output_kind": autotune.get("best_output_kind"),
                "status_counts": autotune.get("status_counts", {}),
                "reference_status_counts": autotune.get(
                    "reference_status_counts",
                    {},
                ),
                "trace_status_counts": autotune.get(
                    "trace_status_counts",
                    {},
                ),
                "output_kind_counts": autotune.get(
                    "output_kind_counts",
                    {},
                ),
                "knob_coverage": autotune.get("knob_coverage"),
                "search_space": autotune.get("search_space"),
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


def _step_names_with_status(
    results: Any,
    *,
    status: str | None = None,
    failing: bool = False,
) -> list[str]:
    if not isinstance(results, dict):
        return []
    names = []
    passing_statuses = {"pass", "dry_run", "skipped", "pending"}
    for name, value in results.items():
        value = str(value)
        if status is not None and value == status:
            names.append(str(name))
        elif failing and value not in passing_statuses:
            names.append(str(name))
    return names


def _tensorization_evidence(step: dict[str, Any]) -> dict[str, Any]:
    tensorization = _step_tensorization_summary(step)
    return {
        "status": tensorization.get("status"),
        "roles": tensorization.get("roles"),
        "tensor_count": tensorization.get("tensor_count"),
        "target_dtype_counts": tensorization.get("target_dtype_counts", {}),
        "layout_counts": tensorization.get("layout_counts", {}),
        "memory_config_counts": tensorization.get("memory_config_counts", {}),
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
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


def _throughput_baseline_summary(
    report: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    throughput = profile.get("throughput_summary") or {}
    observed = throughput.get("tokens_per_second_per_user")
    baseline = report.get("baseline_tokens_per_second_per_user")
    min_ratio = report.get("min_baseline_ratio")
    ratio = None
    if _positive_number(observed) and _positive_number(baseline):
        ratio = float(observed) / float(baseline)
    summary = {
        "metric": "tokens_per_second_per_user",
        "observed": observed,
        "baseline": baseline,
        "ratio": ratio,
        "min_ratio": min_ratio,
    }
    if min_ratio is not None:
        summary["passed"] = _number_at_least(ratio, min_ratio)
    return summary


def _kv_cache_contract_from_template_config(
    template_config: dict[str, Any],
    *,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    template = template_config.get("kv_cache_template")
    return {
        "template": template,
        "policy": "paged" if template == "paged_kv_cache" else None,
        "page_block_size": 32,
        "dtype": "bfloat8_b",
        "max_cache_len": cache_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _decode_step_contract(
    *,
    layer_count: int,
    batch_size: int,
    seq_len: int,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    page_block_size = _safe_int(kv_cache.get("page_block_size")) or 32
    page_count = max(1, (cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    physical_kv_cache_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    logical_kv_cache_shape = [
        batch_size,
        cache_len,
        num_kv_heads,
        head_dim,
    ]
    kv_policy = kv_cache.get("policy")
    kv_template = kv_cache.get("template")
    generation_template = generation.get("template")
    retain_logits = bool(generation.get("retain_logits", False))
    output_kind = "logits" if retain_logits else "token"
    return {
        "schema_version": 1,
        "source": "generated_program_config",
        "layers": layer_count,
        "batch_size": batch_size,
        "decode_seq_len": seq_len,
        "cache_len": cache_len,
        "token_input_shape": [batch_size, seq_len],
        "kv_cache_policy": kv_policy,
        "kv_cache_template": kv_template,
        "uses_paged_kv_cache": (
            kv_policy == "paged" or kv_template == "paged_kv_cache"
        ),
        "kv_page_block_size": page_block_size,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "page_table_shape": [batch_size, page_count],
        "cache_position_shape": [batch_size],
        "kv_cache_shape": physical_kv_cache_shape,
        "kv_cache_physical_shape": physical_kv_cache_shape,
        "kv_cache_logical_shape": logical_kv_cache_shape,
        "kv_cache_layer_ids": list(range(layer_count)),
        "generation_template": generation_template,
        "output_kind": output_kind,
        "accepted_output_kinds": ["token", "logits"],
    }


def _real_decode_acceptance(
    report: dict[str, Any],
    *,
    require_trace: bool,
    require_official_config_match: bool,
    require_full_depth: bool,
    require_program_runtime_shape: bool,
    require_batch32_decode_step: bool,
    min_tokens_per_second_per_user: float | None,
    baseline_tokens_per_second_per_user: float | None,
    min_baseline_ratio: float | None,
    require_decode_shell_numeric_reference: bool,
) -> dict[str, Any]:
    if report.get("dry_run"):
        return {
            "status": "dry_run",
            "passed": True,
            "require_official_config_match": require_official_config_match,
            "require_full_depth": require_full_depth,
            "require_program_runtime_shape": require_program_runtime_shape,
            "require_batch32_decode_step": require_batch32_decode_step,
            "require_trace": require_trace,
            "min_tokens_per_second_per_user": (
                min_tokens_per_second_per_user
            ),
            "baseline_tokens_per_second_per_user": (
                baseline_tokens_per_second_per_user
            ),
            "min_baseline_ratio": min_baseline_ratio,
            "require_decode_shell_numeric_reference": (
                require_decode_shell_numeric_reference
            ),
            "checks": [],
            "message": "Dry run only; runtime acceptance was not evaluated.",
        }

    steps = report.get("steps", {})
    official_config_diff = steps.get("official_config_diff", {})
    materialize = steps.get("materialize_parameters", {})
    decode_shell = steps.get("decode_shell", {})
    attention_primitives = steps.get("attention_primitives", {})
    attention_layer = steps.get("attention_layer", {})
    single_layer = steps.get("single_layer_decode", {})
    smoke = steps.get("smoke_decode_step", {})
    profile = steps.get("profile_decode_step", {})
    autotune = steps.get("decode_step_autotune", {})
    decode_contract = report.get("decode_step_contract") or {}
    single_layer_tensorization = _step_tensorization_summary(single_layer)
    smoke_tensorization = _step_tensorization_summary(smoke)
    profile_tensorization = _step_tensorization_summary(profile)
    attention_primitives_environment = _step_ttnn_environment(
        attention_primitives
    )
    attention_layer_environment = _step_ttnn_environment(attention_layer)
    single_layer_environment = _step_ttnn_environment(single_layer)
    smoke_environment = _step_ttnn_environment(smoke)
    profile_environment = _step_ttnn_environment(profile)
    single_layer_trace = _step_trace_summary(single_layer)
    smoke_trace = _step_trace_summary(smoke)
    profile_trace = _step_trace_summary(profile)
    expected_layers = report.get("layers")
    expected_layer_ids = _expected_layer_ids(expected_layers)
    program_num_layers = report.get("program_num_layers")
    program_batch_size = report.get("program_batch_size")
    program_cache_len = report.get("program_cache_len")
    program_seq_len = report.get("program_seq_len")
    program_hidden_size = report.get("program_hidden_size")
    program_num_attention_heads = report.get("program_num_attention_heads")
    program_num_kv_heads = report.get("program_num_key_value_heads")
    program_head_dim = report.get("program_head_dim")
    expected_batch_size = report.get("batch_size")
    expected_cache_len = report.get("cache_len")
    expected_trace_iterations = report.get("trace_iterations")
    expected_output_kind = decode_contract.get("output_kind")
    lm_head_split_count = _safe_int(materialize.get("lm_head_split_count"))
    expected_token_input_shape = [expected_batch_size, 1]
    expected_page_table_shape = [
        expected_batch_size,
        decode_contract.get("page_count"),
    ]
    expected_cache_position_shape = [expected_batch_size]
    expected_max_num_blocks = (
        expected_batch_size * decode_contract.get("page_count")
        if isinstance(expected_batch_size, int)
        and isinstance(decode_contract.get("page_count"), int)
        else None
    )
    expected_kv_cache_shape = [
        expected_max_num_blocks,
        program_num_kv_heads,
        decode_contract.get("kv_page_block_size"),
        program_head_dim,
    ]
    expected_logical_kv_cache_shape = [
        expected_batch_size,
        expected_cache_len,
        program_num_kv_heads,
        program_head_dim,
    ]
    throughput = profile.get("throughput_summary") or {}
    throughput_baseline = _throughput_baseline_summary(report, profile)
    profile_section_latency = profile.get("section_latency_ms")
    profile_lm_head = profile.get("lm_head_profile")
    profile_layer_profiles = profile.get("layer_profiles")
    profile_bottleneck = profile.get("bottleneck_summary")
    skip_autotune = bool(report.get("skip_autotune"))
    checks = [
        _acceptance_check(
            "official_config_diff.status",
            official_config_diff.get("status") == "pass",
            observed=official_config_diff.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "official_config_diff.diff_status",
            official_config_diff.get("diff_status")
            in {"match", "diff_found"},
            observed=official_config_diff.get("diff_status"),
            expected=["match", "diff_found"],
        ),
        _acceptance_check(
            "official_config_diff.issue_count",
            _nonnegative_number(official_config_diff.get("issue_count")),
            observed=official_config_diff.get("issue_count"),
            minimum=0,
        ),
        _acceptance_check(
            "official_config_diff.sections",
            _contains_all(
                official_config_diff.get("sections"),
                list(PARITY_SECTIONS),
            ),
            observed=official_config_diff.get("sections"),
            expected=list(PARITY_SECTIONS),
        ),
        _acceptance_check(
            "official_config_diff.gap_summary",
            _config_gap_summary_complete(
                official_config_diff.get("gap_summary")
            ),
            observed=_config_gap_summary_observed(
                official_config_diff.get("gap_summary")
            ),
            expected=list(PARITY_SECTIONS),
        ),
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
            "materialize_parameters.lm_head_source_reference",
            _lm_head_source_reference_complete(materialize),
            observed=_lm_head_source_reference_observed(materialize),
            expected={
                "lm_head.weight.materialization": "metadata_reference",
                "lm_head.weight.materialized": False,
                "lm_head.splits.0.weight.source_read": "sliced_tensor",
            },
        ),
        _acceptance_check(
            "decode_step_contract.decode_seq_len",
            _int_equal(decode_contract.get("decode_seq_len"), 1),
            observed=decode_contract.get("decode_seq_len"),
            expected=1,
        ),
        _acceptance_check(
            "decode_step_contract.token_input_shape",
            _int_list(decode_contract.get("token_input_shape"))
            == expected_token_input_shape,
            observed=decode_contract.get("token_input_shape"),
            expected=expected_token_input_shape,
        ),
        _acceptance_check(
            "decode_step_contract.paged_kv_cache",
            decode_contract.get("uses_paged_kv_cache") is True,
            observed={
                "uses_paged_kv_cache": decode_contract.get(
                    "uses_paged_kv_cache"
                ),
                "kv_cache_policy": decode_contract.get("kv_cache_policy"),
                "kv_cache_template": decode_contract.get(
                    "kv_cache_template"
                ),
            },
            expected=True,
        ),
        _acceptance_check(
            "decode_step_contract.kv_page_block_size",
            _positive_number(decode_contract.get("kv_page_block_size")),
            observed=decode_contract.get("kv_page_block_size"),
            minimum=1,
        ),
        _acceptance_check(
            "decode_step_contract.max_num_blocks",
            _int_equal(
                decode_contract.get("max_num_blocks"),
                expected_max_num_blocks,
            ),
            observed=decode_contract.get("max_num_blocks"),
            expected=expected_max_num_blocks,
        ),
        _acceptance_check(
            "decode_step_contract.page_table_shape",
            _int_list(decode_contract.get("page_table_shape"))
            == expected_page_table_shape,
            observed=decode_contract.get("page_table_shape"),
            expected=expected_page_table_shape,
        ),
        _acceptance_check(
            "decode_step_contract.cache_position_shape",
            _int_list(decode_contract.get("cache_position_shape"))
            == expected_cache_position_shape,
            observed=decode_contract.get("cache_position_shape"),
            expected=expected_cache_position_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_shape",
            _int_list(decode_contract.get("kv_cache_shape"))
            == expected_kv_cache_shape,
            observed=decode_contract.get("kv_cache_shape"),
            expected=expected_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_physical_shape",
            _int_list(decode_contract.get("kv_cache_physical_shape"))
            == expected_kv_cache_shape,
            observed=decode_contract.get("kv_cache_physical_shape"),
            expected=expected_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.kv_cache_logical_shape",
            _int_list(decode_contract.get("kv_cache_logical_shape"))
            == expected_logical_kv_cache_shape,
            observed=decode_contract.get("kv_cache_logical_shape"),
            expected=expected_logical_kv_cache_shape,
        ),
        _acceptance_check(
            "decode_step_contract.output_kind",
            decode_contract.get("output_kind") in {"token", "logits"},
            observed=decode_contract.get("output_kind"),
            expected=["token", "logits"],
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
            "decode_shell.input_source",
            decode_shell.get("input_source") == "synthetic",
            observed=decode_shell.get("input_source"),
            expected="synthetic",
        ),
        _acceptance_check(
            "decode_shell.runtime_input_tensor_count",
            _positive_number(
                decode_shell.get("runtime_input_tensor_count")
            ),
            observed=decode_shell.get("runtime_input_tensor_count"),
            minimum=1,
        ),
        _acceptance_check(
            "decode_shell.runtime_status",
            decode_shell.get("runtime_status") == "passed",
            observed=decode_shell.get("runtime_status"),
            expected="passed",
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
            "attention_primitives.primitive_count",
            _int_equal(
                attention_primitives.get("primitive_count"),
                len(ATTENTION_PRIMITIVES),
            ),
            observed=attention_primitives.get("primitive_count"),
            expected=len(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_primitives.primitive_sequence",
            attention_primitives.get("primitive_sequence")
            == list(ATTENTION_PRIMITIVES),
            observed=attention_primitives.get("primitive_sequence"),
            expected=list(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_primitives.status",
            attention_primitives.get("status") == "pass",
            observed=attention_primitives.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "attention_primitives.runtime_status_counts",
            attention_primitives.get("runtime_status_counts")
            == {"passed": len(ATTENTION_PRIMITIVES)},
            observed=attention_primitives.get("runtime_status_counts"),
            expected={"passed": len(ATTENTION_PRIMITIVES)},
        ),
        _acceptance_check(
            "attention_primitives.ttnn_module_available",
            attention_primitives_environment.get("module_available") is True,
            observed=attention_primitives_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "attention_primitives.ttnn_version",
            _non_empty_string(
                attention_primitives_environment.get("version")
            ),
            observed=attention_primitives_environment.get("version"),
            required=True,
        ),
        _acceptance_check(
            "attention_primitives.tt_metal_git_commit",
            _non_empty_string(
                attention_primitives_environment.get("tt_metal_git_commit")
            ),
            observed=attention_primitives_environment.get(
                "tt_metal_git_commit"
            ),
            source=attention_primitives_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "attention_primitives.primitive_reports",
            _attention_primitive_reports_complete(
                attention_primitives.get("primitive_reports"),
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_heads=program_num_attention_heads,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
            ),
            observed=_attention_primitive_reports_observed(
                attention_primitives.get("primitive_reports")
            ),
            expected=list(ATTENTION_PRIMITIVES),
        ),
        _acceptance_check(
            "attention_layer.layer",
            _int_equal(attention_layer.get("layer"), 0),
            observed=attention_layer.get("layer"),
            expected=0,
        ),
        _acceptance_check(
            "attention_layer.batch_size",
            _int_equal(attention_layer.get("batch_size"), expected_batch_size),
            observed=attention_layer.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "attention_layer.cache_len",
            _int_equal(attention_layer.get("cache_len"), expected_cache_len),
            observed=attention_layer.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "attention_layer.hidden_size",
            _int_equal(attention_layer.get("hidden_size"), program_hidden_size),
            observed=attention_layer.get("hidden_size"),
            expected=program_hidden_size,
        ),
        _acceptance_check(
            "attention_layer.num_kv_heads",
            _int_equal(
                attention_layer.get("num_kv_heads"),
                program_num_kv_heads,
            ),
            observed=attention_layer.get("num_kv_heads"),
            expected=program_num_kv_heads,
        ),
        _acceptance_check(
            "attention_layer.head_dim",
            _int_equal(attention_layer.get("head_dim"), program_head_dim),
            observed=attention_layer.get("head_dim"),
            expected=program_head_dim,
        ),
        _acceptance_check(
            "attention_layer.latency_ms",
            _nonnegative_number(attention_layer.get("latency_ms")),
            observed=attention_layer.get("latency_ms"),
            minimum=0,
        ),
        _acceptance_check(
            "attention_layer.tensor_conversion_count",
            _positive_number(attention_layer.get("tensor_conversion_count")),
            observed=attention_layer.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "attention_layer.memory_config_conversion_count",
            _positive_number(
                attention_layer.get("memory_config_conversion_count")
            ),
            observed=attention_layer.get("memory_config_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "attention_layer.runtime_status",
            attention_layer.get("runtime_status") == "passed",
            observed=attention_layer.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "attention_layer.ttnn_module_available",
            attention_layer_environment.get("module_available") is True,
            observed=attention_layer_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "attention_layer.ttnn_version",
            _non_empty_string(attention_layer_environment.get("version")),
            observed=attention_layer_environment.get("version"),
            required=True,
        ),
        _acceptance_check(
            "attention_layer.tt_metal_git_commit",
            _non_empty_string(
                attention_layer_environment.get("tt_metal_git_commit")
            ),
            observed=attention_layer_environment.get(
                "tt_metal_git_commit"
            ),
            source=attention_layer_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "attention_layer.reference_status",
            attention_layer.get("reference_status") == "passed",
            observed=attention_layer.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "attention_layer.primitive_sequence",
            attention_layer.get("primitive_sequence")
            == list(ATTENTION_LAYER_OPS),
            observed=attention_layer.get("primitive_sequence"),
            expected=list(ATTENTION_LAYER_OPS),
        ),
        _acceptance_check(
            "attention_layer.primitive_reports",
            _attention_layer_primitive_reports_complete(
                attention_layer.get("primitive_reports")
            ),
            observed=_attention_layer_primitive_reports_observed(
                attention_layer.get("primitive_reports")
            ),
            expected=list(ATTENTION_LAYER_OPS),
        ),
        _acceptance_check(
            "attention_layer.output_shapes",
            _attention_layer_output_shapes_complete(
                attention_layer.get("output_shapes"),
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_attention_layer_output_shape_observed(
                attention_layer.get("output_shapes")
            ),
            expected=_expected_attention_layer_output_shape_summary(
                batch_size=expected_batch_size,
                cache_len=expected_cache_len,
                hidden_size=program_hidden_size,
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "attention_layer.observed_op_sequence",
            _observed_ops_cover_planned(
                attention_layer.get("reference_planned_observed_ops"),
                attention_layer.get("reference_observed_ops"),
            ),
            observed=attention_layer.get("reference_observed_ops"),
            expected=attention_layer.get("reference_planned_observed_ops"),
        ),
        _acceptance_check(
            "single_layer_decode.parameter_source",
            single_layer.get("parameter_source") == "hf_model",
            observed=single_layer.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "single_layer_decode.input_source",
            single_layer.get("input_source") == "synthetic",
            observed=single_layer.get("input_source"),
            expected="synthetic",
        ),
        _acceptance_check(
            "single_layer_decode.synthetic_runtime_inputs",
            _positive_number(
                single_layer.get("synthetic_runtime_input_tensor_count")
            ),
            observed=single_layer.get(
                "synthetic_runtime_input_tensor_count"
            ),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.layers",
            _int_equal(single_layer.get("layers"), 1),
            observed=single_layer.get("layers"),
            expected=1,
        ),
        _acceptance_check(
            "single_layer_decode.batch_size",
            _int_equal(single_layer.get("batch_size"), expected_batch_size),
            observed=single_layer.get("batch_size"),
            expected=expected_batch_size,
        ),
        _acceptance_check(
            "single_layer_decode.cache_len",
            _int_equal(single_layer.get("cache_len"), expected_cache_len),
            observed=single_layer.get("cache_len"),
            expected=expected_cache_len,
        ),
        _acceptance_check(
            "single_layer_decode.tensor_conversion_count",
            _positive_number(single_layer.get("tensor_conversion_count")),
            observed=single_layer.get("tensor_conversion_count"),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.runtime_status",
            single_layer.get("runtime_status") == "passed",
            observed=single_layer.get("runtime_status"),
            expected="passed",
        ),
        _acceptance_check(
            "single_layer_decode.ttnn_module_available",
            single_layer_environment.get("module_available") is True,
            observed=single_layer_environment.get("module_available"),
            expected=True,
        ),
        _acceptance_check(
            "single_layer_decode.ttnn_version",
            _non_empty_string(single_layer_environment.get("version")),
            observed=single_layer_environment.get("version"),
            required=True,
        ),
        _acceptance_check(
            "single_layer_decode.tt_metal_git_commit",
            _non_empty_string(
                single_layer_environment.get("tt_metal_git_commit")
            ),
            observed=single_layer_environment.get("tt_metal_git_commit"),
            source=single_layer_environment.get(
                "tt_metal_git_commit_source"
            ),
            required=True,
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_status",
            single_layer_tensorization.get("status") == "pass",
            observed=single_layer_tensorization.get("status"),
            expected="pass",
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_roles",
            _contains_all(
                single_layer_tensorization.get("roles"),
                DECODE_PARAMETER_ROLES,
            ),
            observed=single_layer_tensorization.get("roles"),
            expected=list(DECODE_PARAMETER_ROLES),
        ),
        _acceptance_check(
            "single_layer_decode.required_tensorized_tensor_paths",
            single_layer.get("missing_required_tensorized_tensor_paths")
            == [],
            observed=single_layer.get(
                "missing_required_tensorized_tensor_paths"
            ),
            expected=[],
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_memory_configs",
            _positive_count(
                single_layer_tensorization.get("memory_config_counts")
            ),
            observed=single_layer_tensorization.get("memory_config_counts"),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.tensorization_ttnn_memory_configs",
            _positive_count(
                single_layer_tensorization.get("ttnn_memory_config_counts")
            ),
            observed=single_layer_tensorization.get(
                "ttnn_memory_config_counts"
            ),
            minimum=1,
        ),
        _acceptance_check(
            "single_layer_decode.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                single_layer_tensorization,
                layer_count=1,
            ),
            observed=_embedding_norm_weight_transform_observed(
                single_layer_tensorization,
                layer_count=1,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(1),
            },
        ),
        _acceptance_check(
            "single_layer_decode.linear_weight_transforms",
            _linear_weight_transform_complete(
                single_layer_tensorization,
                layer_count=1,
            ),
            observed=_linear_weight_transform_observed(
                single_layer_tensorization,
                layer_count=1,
            ),
            expected={
                "transform": "transpose_2d",
                "paths": _linear_weight_transform_paths(1),
            },
        ),
        _acceptance_check(
            "single_layer_decode.lm_head_transform",
            _lm_head_transform_complete(
                single_layer_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(single_layer_tensorization),
            expected={
                "transform": "transpose_2d",
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "single_layer_decode.reference_status",
            single_layer.get("reference_status") == "passed",
            observed=single_layer.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "single_layer_decode.output_shapes",
            _decode_output_shapes_complete(
                single_layer.get("output_shapes"),
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                single_layer.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=1,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "single_layer_decode.observed_op_sequence",
            _observed_ops_cover_planned(
                single_layer.get("reference_planned_ops"),
                single_layer.get("reference_observed_ops"),
            ),
            observed=single_layer.get("reference_observed_ops"),
            expected=single_layer.get("reference_planned_ops"),
        ),
        _acceptance_check(
            "smoke_decode_step.parameter_source",
            smoke.get("parameter_source") == "hf_model",
            observed=smoke.get("parameter_source"),
            expected="hf_model",
        ),
        _acceptance_check(
            "smoke_decode_step.input_source",
            smoke.get("input_source") == "synthetic",
            observed=smoke.get("input_source"),
            expected="synthetic",
        ),
        _acceptance_check(
            "smoke_decode_step.synthetic_runtime_inputs",
            _positive_number(
                smoke.get("synthetic_runtime_input_tensor_count")
            ),
            observed=smoke.get("synthetic_runtime_input_tensor_count"),
            minimum=1,
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
            "smoke_decode_step.runtime_status",
            smoke.get("runtime_status") == "passed",
            observed=smoke.get("runtime_status"),
            expected="passed",
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
            "smoke_decode_step.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            observed=_embedding_norm_weight_transform_observed(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(
                    expected_layers
                ),
            },
        ),
        _acceptance_check(
            "smoke_decode_step.linear_weight_transforms",
            _linear_weight_transform_complete(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            observed=_linear_weight_transform_observed(
                smoke_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "transform": "transpose_2d",
                "paths": _linear_weight_transform_paths(expected_layers),
            },
        ),
        _acceptance_check(
            "smoke_decode_step.lm_head_transform",
            _lm_head_transform_complete(
                smoke_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(smoke_tensorization),
            expected={
                "transform": "transpose_2d",
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "smoke_decode_step.reference_status",
            smoke.get("reference_status") == "passed",
            observed=smoke.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "smoke_decode_step.output_shapes",
            _decode_output_shapes_complete(
                smoke.get("output_shapes"),
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                smoke.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
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
            "profile_decode_step.input_source",
            profile.get("input_source") == "synthetic",
            observed=profile.get("input_source"),
            expected="synthetic",
        ),
        _acceptance_check(
            "profile_decode_step.synthetic_runtime_inputs",
            _positive_number(
                profile.get("synthetic_runtime_input_tensor_count")
            ),
            observed=profile.get("synthetic_runtime_input_tensor_count"),
            minimum=1,
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
            "profile_decode_step.runtime_status",
            profile.get("runtime_status") == "profiled",
            observed=profile.get("runtime_status"),
            expected="profiled",
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
            "profile_decode_step.embedding_norm_weight_transforms",
            _embedding_norm_weight_transform_complete(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            observed=_embedding_norm_weight_transform_observed(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "embedding_transform": "reshape_embedding_weight_4d",
                "norm_transform": "reshape_norm_weight_4d",
                "paths": _embedding_norm_weight_transform_paths(
                    expected_layers
                ),
            },
        ),
        _acceptance_check(
            "profile_decode_step.linear_weight_transforms",
            _linear_weight_transform_complete(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            observed=_linear_weight_transform_observed(
                profile_tensorization,
                layer_count=expected_layers,
            ),
            expected={
                "transform": "transpose_2d",
                "paths": _linear_weight_transform_paths(expected_layers),
            },
        ),
        _acceptance_check(
            "profile_decode_step.lm_head_transform",
            _lm_head_transform_complete(
                profile_tensorization,
                lm_head_split_count,
            ),
            observed=_lm_head_transform_observed(profile_tensorization),
            expected={
                "transform": "transpose_2d",
                "count": lm_head_split_count,
            },
        ),
        _acceptance_check(
            "profile_decode_step.reference_status",
            profile.get("reference_status") == "passed",
            observed=profile.get("reference_status"),
            expected="passed",
        ),
        _acceptance_check(
            "profile_decode_step.output_shapes",
            _decode_output_shapes_complete(
                profile.get("output_shapes"),
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
            observed=_decode_output_shape_observed(
                profile.get("output_shapes")
            ),
            expected=_expected_decode_output_shape_summary(
                layer_count=expected_layers,
                batch_size=expected_batch_size,
                seq_len=program_seq_len,
                cache_len=expected_cache_len,
                vocab_size=report.get("program_vocab_size"),
                num_kv_heads=program_num_kv_heads,
                head_dim=program_head_dim,
                output_kind=expected_output_kind,
                page_block_size=decode_contract.get("kv_page_block_size"),
            ),
        ),
        _acceptance_check(
            "profile_decode_step.lm_head_profile",
            _lm_head_profile_complete(
                profile_lm_head,
                output_kind=expected_output_kind,
            ),
            observed=_lm_head_profile_observed(profile_lm_head),
            expected={
                "split_count": "positive",
                "lm_head_ms": "nonnegative",
                "argmax_ms": "nonnegative",
                "argmax_status": (
                    "skipped"
                    if expected_output_kind == "logits"
                    else "profiled"
                ),
            },
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
    if require_official_config_match:
        checks.append(
            _acceptance_check(
                "official_config_diff.match",
                official_config_diff.get("diff_status") == "match",
                observed=official_config_diff.get("diff_status"),
                expected="match",
                issue_count=official_config_diff.get("issue_count"),
            )
        )
    if require_full_depth:
        checks.append(
            _acceptance_check(
                "validation.full_depth_layers",
                _int_equal(expected_layers, program_num_layers),
                observed=expected_layers,
                expected=program_num_layers,
            )
        )
    if require_program_runtime_shape:
        checks.extend(
            [
                _acceptance_check(
                    "validation.program_batch_size",
                    _int_equal(expected_batch_size, program_batch_size),
                    observed=expected_batch_size,
                    expected=program_batch_size,
                ),
                _acceptance_check(
                    "validation.program_cache_len",
                    _int_equal(expected_cache_len, program_cache_len),
                    observed=expected_cache_len,
                    expected=program_cache_len,
                ),
            ]
        )
    if require_batch32_decode_step:
        checks.append(
            _acceptance_check(
                "decode_step_contract.batch32",
                _int_equal(decode_contract.get("batch_size"), 32),
                observed=decode_contract.get("batch_size"),
                expected=32,
            )
        )
    if require_trace:
        checks.extend(
            [
                _acceptance_check(
                    "single_layer_decode.trace_status",
                    single_layer.get("trace_status")
                    == "captured_and_executed",
                    observed=single_layer.get("trace_status"),
                    expected="captured_and_executed",
                ),
                _acceptance_check(
                    "single_layer_decode.trace_iterations",
                    _int_equal(
                        single_layer_trace.get("iterations"),
                        expected_trace_iterations,
                    ),
                    observed=single_layer_trace.get("iterations"),
                    expected=expected_trace_iterations,
                ),
                _acceptance_check(
                    "single_layer_decode.trace_execute_sample_count",
                    _int_equal(
                        single_layer_trace.get("execute_sample_count"),
                        expected_trace_iterations,
                    ),
                    observed=single_layer_trace.get("execute_sample_count"),
                    expected=expected_trace_iterations,
                ),
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
                    "profile_decode_step.trace_profile",
                    _trace_profile_complete(
                        profile_trace,
                        throughput,
                        expected_iterations=expected_trace_iterations,
                    ),
                    observed=_trace_profile_observed(
                        profile_trace,
                        throughput,
                    ),
                    expected={
                        "status": "captured_and_executed",
                        "iterations": expected_trace_iterations,
                        "execute_samples_ms": "positive",
                        "capture_latency_ms": "nonnegative",
                        "execute_latency_ms": "positive",
                        "trace_execute_mean_ms": "positive",
                        "trace_execute_tokens_per_second_per_user": (
                            "positive"
                        ),
                        "trace_execute_aggregate_tokens_per_second": (
                            "positive"
                        ),
                    },
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

    if baseline_tokens_per_second_per_user is not None:
        checks.append(
            _acceptance_check(
                "profile_decode_step.baseline_tokens_per_second_per_user",
                _positive_number(baseline_tokens_per_second_per_user),
                observed=baseline_tokens_per_second_per_user,
                minimum=0,
            )
        )
    if min_baseline_ratio is not None:
        checks.append(
            _acceptance_check(
                "profile_decode_step.min_baseline_ratio",
                _number_at_least(
                    throughput_baseline.get("ratio"),
                    min_baseline_ratio,
                ),
                observed=throughput_baseline.get("ratio"),
                minimum=min_baseline_ratio,
                baseline=baseline_tokens_per_second_per_user,
                tokens_per_second_per_user=throughput.get(
                    "tokens_per_second_per_user"
                ),
            )
        )

    if not skip_autotune:
        checks.extend(
            [
                _acceptance_check(
                    "decode_step_autotune.status",
                    autotune.get("status") == "pass",
                    observed=autotune.get("status"),
                    expected="pass",
                ),
                _acceptance_check(
                    "decode_step_autotune.candidate_count",
                    _positive_number(autotune.get("candidate_count")),
                    observed=autotune.get("candidate_count"),
                    minimum=1,
                ),
                _acceptance_check(
                    "decode_step_autotune.knob_coverage",
                    _autotune_knob_coverage_complete(
                        autotune.get("knob_coverage"),
                        candidate_count=autotune.get("candidate_count"),
                    ),
                    observed=_autotune_knob_coverage_observed(
                        autotune.get("knob_coverage")
                    ),
                    expected=list(DECODE_STEP_AUTOTUNE_KNOBS),
                ),
                _acceptance_check(
                    "decode_step_autotune.output_kind_counts",
                    _autotune_output_kind_counts_complete(
                        autotune.get("output_kind_counts"),
                        autotune.get("knob_coverage"),
                    ),
                    observed=_autotune_output_kind_counts_observed(
                        autotune.get("output_kind_counts"),
                        autotune.get("knob_coverage"),
                    ),
                    expected="output kinds implied by generation_template",
                ),
                _acceptance_check(
                    "decode_step_autotune.passed_candidate_count",
                    _positive_number(autotune.get("passed_candidate_count")),
                    observed=autotune.get("passed_candidate_count"),
                    minimum=1,
                ),
                _acceptance_check(
                    "decode_step_autotune.best",
                    _non_empty_string(autotune.get("best")),
                    observed=autotune.get("best"),
                    required=True,
                ),
                _acceptance_check(
                    "decode_step_autotune.best_reference_status",
                    autotune.get("best_reference_status") == "passed",
                    observed=autotune.get("best_reference_status"),
                    expected="passed",
                ),
                _acceptance_check(
                    "decode_step_autotune.best_parameter_source",
                    autotune.get("best_parameter_source") == "hf_model",
                    observed=autotune.get("best_parameter_source"),
                    expected="hf_model",
                ),
                _acceptance_check(
                    "decode_step_autotune.best_metric",
                    _nonnegative_number(autotune.get("best_metric")),
                    observed=autotune.get("best_metric"),
                    minimum=0,
                ),
            ]
        )
        if require_trace:
            checks.append(
                _acceptance_check(
                    "decode_step_autotune.best_trace_status",
                    autotune.get("best_trace_status")
                    == "captured_and_executed",
                    observed=autotune.get("best_trace_status"),
                    expected="captured_and_executed",
                )
            )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "require_official_config_match": require_official_config_match,
        "require_full_depth": require_full_depth,
        "require_program_runtime_shape": require_program_runtime_shape,
        "require_batch32_decode_step": require_batch32_decode_step,
        "require_trace": require_trace,
        "min_tokens_per_second_per_user": min_tokens_per_second_per_user,
        "baseline_tokens_per_second_per_user": (
            baseline_tokens_per_second_per_user
        ),
        "min_baseline_ratio": min_baseline_ratio,
        "throughput_baseline": throughput_baseline,
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


def _config_gap_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    status = summary.get("status")
    if status not in {"match", "diff_found"}:
        return False
    if not _int_equal(summary.get("section_count"), len(PARITY_SECTIONS)):
        return False
    counts = summary.get("issue_counts_by_section")
    if not isinstance(counts, dict):
        return False
    if set(counts) != set(PARITY_SECTIONS):
        return False
    try:
        issue_count = int(summary.get("issue_count"))
        section_counts = {
            section: int(counts[section])
            for section in PARITY_SECTIONS
        }
    except (TypeError, ValueError):
        return False
    if issue_count < 0 or any(count < 0 for count in section_counts.values()):
        return False
    if sum(section_counts.values()) != issue_count:
        return False
    sections_with_issues = summary.get("sections_with_issues")
    if not isinstance(sections_with_issues, list):
        return False
    if set(sections_with_issues) - set(PARITY_SECTIONS):
        return False
    expected_sections = [
        section
        for section in PARITY_SECTIONS
        if section_counts[section] > 0
    ]
    if sections_with_issues != expected_sections:
        return False
    top_issue_paths = summary.get("top_issue_paths")
    if not isinstance(top_issue_paths, list):
        return False
    if issue_count == 0:
        return status == "match" and sections_with_issues == [] and top_issue_paths == []
    return status == "diff_found" and bool(top_issue_paths) and all(
        _config_gap_issue_complete(issue) for issue in top_issue_paths
    )


def _config_gap_issue_complete(issue: Any) -> bool:
    if not isinstance(issue, dict):
        return False
    return (
        issue.get("kind") in {"missing", "mismatch", "extra"}
        and issue.get("section") in PARITY_SECTIONS
        and _non_empty_string(issue.get("path"))
    )


def _config_gap_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    counts = summary.get("issue_counts_by_section")
    top_issue_paths = summary.get("top_issue_paths")
    return {
        "status": summary.get("status"),
        "issue_count": summary.get("issue_count"),
        "section_count": summary.get("section_count"),
        "sections_with_issues": summary.get("sections_with_issues"),
        "issue_count_sections": _field_keys(counts),
        "top_issue_count": len(top_issue_paths)
        if isinstance(top_issue_paths, list)
        else None,
    }


def _lm_head_profile_complete(
    profile: Any,
    *,
    output_kind: Any = "token",
) -> bool:
    if not isinstance(profile, dict):
        return False
    expected_argmax_status = (
        "skipped" if output_kind == "logits" else "profiled"
    )
    return (
        _positive_number(profile.get("split_count"))
        and _nonnegative_number(profile.get("lm_head_ms"))
        and _nonnegative_number(profile.get("argmax_ms"))
        and profile.get("argmax_status") == expected_argmax_status
    )


def _lm_head_profile_observed(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "split_count": profile.get("split_count"),
        "lm_head_ms": profile.get("lm_head_ms"),
        "argmax_ms": profile.get("argmax_ms"),
        "argmax_status": profile.get("argmax_status"),
    }


def _lm_head_transform_complete(
    tensorization: Any,
    split_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_count = _safe_int(split_count)
    if expected_count is None or expected_count <= 0:
        return False
    counts = tensorization.get("transform_counts")
    if not isinstance(counts, dict):
        return False
    observed_count = _safe_int(counts.get("transpose_2d"))
    if observed_count is None or observed_count < expected_count:
        return False
    expected_paths = {
        f"lm_head.splits.{shard_id}.weight"
        for shard_id in range(expected_count)
    }
    transformed_paths = _transformed_tensor_paths(
        tensorization,
        "transpose_2d",
    )
    if transformed_paths and not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    split0 = key_tensors.get("lm_head.splits.0.weight")
    if not isinstance(split0, dict):
        return False
    return split0.get("transform") == "transpose_2d"


def _embedding_norm_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    paths = _embedding_norm_weight_transform_paths(layer_count)
    embedding_paths = set(paths["embedding"])
    norm_paths = set(paths["norm"])
    if not embedding_paths or not norm_paths:
        return False
    embedding_transformed = _transformed_tensor_paths(
        tensorization,
        "reshape_embedding_weight_4d",
    )
    norm_transformed = _transformed_tensor_paths(
        tensorization,
        "reshape_norm_weight_4d",
    )
    if not embedding_paths.issubset(embedding_transformed):
        return False
    if not norm_paths.issubset(norm_transformed):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "embedding.weight",
        "layers.0.input_norm.weight",
        "layers.0.post_attention_norm.weight",
        "final_norm.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict):
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                return False
    return True


def _embedding_norm_weight_transform_paths(layer_count: Any) -> dict[str, list[str]]:
    norm_paths = ["final_norm.weight"]
    for layer_id in _expected_layer_ids(layer_count):
        norm_paths.extend(
            [
                f"layers.{layer_id}.input_norm.weight",
                f"layers.{layer_id}.post_attention_norm.weight",
            ]
        )
    return {
        "embedding": ["embedding.weight"],
        "norm": norm_paths,
    }


def _embedding_norm_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = _embedding_norm_weight_transform_paths(layer_count)
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths["embedding"] + expected_paths["norm"]:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "embedding_transformed_paths": sorted(
            _transformed_tensor_paths(
                tensorization,
                "reshape_embedding_weight_4d",
            )
        ),
        "norm_transformed_paths": sorted(
            _transformed_tensor_paths(
                tensorization,
                "reshape_norm_weight_4d",
            )
        ),
        "key_tensors": key_observed,
    }


def _linear_weight_transform_complete(
    tensorization: Any,
    *,
    layer_count: Any,
) -> bool:
    if not isinstance(tensorization, dict):
        return False
    expected_paths = set(_linear_weight_transform_paths(layer_count))
    if not expected_paths:
        return False
    transformed_paths = _transformed_tensor_paths(
        tensorization,
        "transpose_2d",
    )
    if not expected_paths.issubset(transformed_paths):
        return False
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return False
    for key_path in (
        "layers.0.attention.wqkv_packed.weight",
        "layers.0.attention.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ):
        tensor = key_tensors.get(key_path)
        if isinstance(tensor, dict) and tensor.get("transform") != "transpose_2d":
            return False
    return True


def _linear_weight_transform_paths(layer_count: Any) -> list[str]:
    paths: list[str] = []
    for layer_id in _expected_layer_ids(layer_count):
        paths.extend(
            [
                f"layers.{layer_id}.attention.wqkv_packed.weight",
                f"layers.{layer_id}.attention.o_proj.weight",
                f"layers.{layer_id}.mlp.gate_proj.weight",
                f"layers.{layer_id}.mlp.up_proj.weight",
                f"layers.{layer_id}.mlp.down_proj.weight",
            ]
        )
    return paths


def _transformed_tensor_paths(
    tensorization: dict[str, Any],
    transform: str,
) -> set[str]:
    paths_by_kind = tensorization.get("transform_paths_by_kind")
    if isinstance(paths_by_kind, dict):
        paths = paths_by_kind.get(transform, [])
        if isinstance(paths, list):
            return {str(path) for path in paths}
    key_tensors = tensorization.get("key_tensors")
    if not isinstance(key_tensors, dict):
        return set()
    return {
        str(path)
        for path, tensor in key_tensors.items()
        if isinstance(tensor, dict) and tensor.get("transform") == transform
    }


def _linear_weight_transform_observed(
    tensorization: Any,
    *,
    layer_count: Any,
) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    expected_paths = _linear_weight_transform_paths(layer_count)
    transformed_paths = sorted(
        _transformed_tensor_paths(tensorization, "transpose_2d")
    )
    key_tensors = tensorization.get("key_tensors")
    key_observed = {}
    if isinstance(key_tensors, dict):
        for path in expected_paths:
            tensor = key_tensors.get(path)
            if isinstance(tensor, dict):
                key_observed[path] = {
                    "transform": tensor.get("transform"),
                    "source_shape": tensor.get("source_shape"),
                    "shape": tensor.get("shape"),
                }
    return {
        "expected_paths": expected_paths,
        "transformed_paths": transformed_paths,
        "key_tensors": key_observed,
    }


def _lm_head_transform_observed(tensorization: Any) -> dict[str, Any]:
    if not isinstance(tensorization, dict):
        return {}
    key_tensors = tensorization.get("key_tensors")
    split0 = {}
    if isinstance(key_tensors, dict):
        maybe_split0 = key_tensors.get("lm_head.splits.0.weight")
        if isinstance(maybe_split0, dict):
            split0 = {
                "transform": maybe_split0.get("transform"),
                "source_shape": maybe_split0.get("source_shape"),
                "shape": maybe_split0.get("shape"),
            }
    return {
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
        "lm_head.splits.0.weight": split0,
    }


def _trace_profile_complete(
    trace: Any,
    throughput: Any,
    *,
    expected_iterations: Any,
) -> bool:
    if not isinstance(trace, dict) or not isinstance(throughput, dict):
        return False
    samples = trace.get("execute_samples_ms")
    return (
        trace.get("status") == "captured_and_executed"
        and _int_equal(trace.get("iterations"), expected_iterations)
        and _trace_samples_complete(samples, expected_iterations)
        and _nonnegative_number(trace.get("capture_latency_ms"))
        and _positive_number(trace.get("execute_latency_ms"))
        and _positive_number(throughput.get("trace_execute_mean_ms"))
        and _positive_number(
            throughput.get("trace_execute_tokens_per_second_per_user")
        )
        and _positive_number(
            throughput.get("trace_execute_aggregate_tokens_per_second")
        )
        and _int_equal(throughput.get("trace_iterations"), expected_iterations)
    )


def _trace_samples_complete(samples: Any, expected_iterations: Any) -> bool:
    if not isinstance(samples, list):
        return False
    try:
        expected_count = int(expected_iterations)
    except (TypeError, ValueError):
        return False
    return (
        len(samples) == expected_count
        and bool(samples)
        and all(_positive_number(sample) for sample in samples)
    )


def _trace_profile_observed(
    trace: Any,
    throughput: Any,
) -> dict[str, Any]:
    trace_dict = trace if isinstance(trace, dict) else {}
    throughput_dict = throughput if isinstance(throughput, dict) else {}
    return {
        "status": trace_dict.get("status"),
        "iterations": trace_dict.get("iterations"),
        "execute_sample_count": trace_dict.get("execute_sample_count"),
        "capture_latency_ms": trace_dict.get("capture_latency_ms"),
        "execute_latency_ms": trace_dict.get("execute_latency_ms"),
        "execute_samples_ms": trace_dict.get("execute_samples_ms"),
        "trace_execute_mean_ms": throughput_dict.get(
            "trace_execute_mean_ms"
        ),
        "trace_execute_tokens_per_second_per_user": throughput_dict.get(
            "trace_execute_tokens_per_second_per_user"
        ),
        "trace_execute_aggregate_tokens_per_second": throughput_dict.get(
            "trace_execute_aggregate_tokens_per_second"
        ),
        "trace_iterations": throughput_dict.get("trace_iterations"),
    }


def _autotune_knob_coverage_complete(
    coverage: Any,
    *,
    candidate_count: Any,
) -> bool:
    if not isinstance(coverage, dict):
        return False
    coverage_count = coverage.get("candidate_count")
    if not _positive_number(coverage_count):
        return False
    if candidate_count is not None and not _int_equal(
        coverage_count,
        candidate_count,
    ):
        return False
    if not _contains_all(
        coverage.get("knobs"),
        DECODE_STEP_AUTOTUNE_KNOBS,
    ):
        return False
    values = coverage.get("values")
    value_counts = coverage.get("value_counts")
    if not isinstance(values, dict) or not isinstance(value_counts, dict):
        return False
    for knob in DECODE_STEP_AUTOTUNE_KNOBS:
        knob_values = values.get(knob)
        if not isinstance(knob_values, list) or not knob_values:
            return False
        counts = value_counts.get(knob)
        if not isinstance(counts, dict) or not counts:
            return False
        try:
            count_total = sum(int(count) for count in counts.values())
        except (TypeError, ValueError):
            return False
        if not _int_equal(count_total, coverage_count):
            return False
    return True


def _autotune_knob_coverage_observed(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "knobs": coverage.get("knobs"),
        "candidate_count": coverage.get("candidate_count"),
        "values": coverage.get("values"),
        "varied_knobs": coverage.get("varied_knobs"),
    }


def _autotune_output_kind_counts_complete(
    counts: Any,
    coverage: Any,
) -> bool:
    if not isinstance(counts, dict) or not isinstance(coverage, dict):
        return False
    expected_kinds = _autotune_expected_output_kinds(coverage)
    if not expected_kinds:
        return True
    for kind in expected_kinds:
        if not _positive_number(counts.get(kind)):
            return False
    return True


def _autotune_output_kind_counts_observed(
    counts: Any,
    coverage: Any,
) -> dict[str, Any]:
    return {
        "counts": counts if isinstance(counts, dict) else {},
        "expected_output_kinds": _autotune_expected_output_kinds(coverage),
    }


def _autotune_expected_output_kinds(coverage: Any) -> list[str]:
    if not isinstance(coverage, dict):
        return []
    values = coverage.get("values")
    if not isinstance(values, dict):
        return []
    generation_templates = values.get("generation_template")
    if not isinstance(generation_templates, list):
        return []
    kinds = []
    for template in generation_templates:
        kind = "token" if template == "device_argmax_greedy" else "logits"
        if kind not in kinds:
            kinds.append(kind)
    return kinds


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


def _step_synthetic_runtime_input_count(step: dict[str, Any]) -> Any:
    setup = step.get("parameter_setup") or {}
    if not isinstance(setup, dict):
        return None
    return setup.get("synthetic_runtime_input_tensor_count")


def _decode_output_shapes_complete(
    output_shapes: Any,
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> bool:
    expected = _expected_decode_output_shape_summary(
        layer_count=layer_count,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=cache_len,
        vocab_size=vocab_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        output_kind=output_kind,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    output_kind = expected["output_kind"]
    output_shape = _int_list(output_shapes.get(output_kind))
    if output_kind == "token":
        if output_shape not in expected["accepted_output_shapes"]:
            return False
    elif output_shape != expected["output_shape"]:
        return False
    if _int_list(output_shapes.get("key_cache")) != expected["kv_cache_shape"]:
        return False
    if _int_list(output_shapes.get("value_cache")) != expected["kv_cache_shape"]:
        return False
    layers = output_shapes.get("kv_cache_layers")
    if not isinstance(layers, list):
        return False
    if len(layers) != len(expected["kv_cache_layer_ids"]):
        return False
    observed_layer_ids = []
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        try:
            layer_id = int(layer["layer_id"])
        except (KeyError, TypeError, ValueError):
            return False
        observed_layer_ids.append(layer_id)
        if _int_list(layer.get("key_cache")) != expected["kv_cache_shape"]:
            return False
        if _int_list(layer.get("value_cache")) != expected["kv_cache_shape"]:
            return False
    return observed_layer_ids == expected["kv_cache_layer_ids"]


def _attention_layer_output_shapes_complete(
    output_shapes: Any,
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> bool:
    expected = _expected_attention_layer_output_shape_summary(
        batch_size=batch_size,
        cache_len=cache_len,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    return (
        _int_list(output_shapes.get("attention_output"))
        == expected["attention_output"]
        and _int_list(output_shapes.get("key_cache"))
        == expected["kv_cache_shape"]
        and _int_list(output_shapes.get("value_cache"))
        == expected["kv_cache_shape"]
    )


def _expected_attention_layer_output_shape_summary(
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    hidden = _safe_int(hidden_size)
    attention_output = (
        [1, 1, batch, hidden] if None not in (batch, hidden) else []
    )
    kv_shape = _paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    return {
        "attention_output": attention_output,
        "kv_cache_shape": kv_shape,
    }


def _attention_layer_output_shape_observed(
    output_shapes: Any,
) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    return {
        "attention_output": _int_list(
            output_shapes.get("attention_output")
        ),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
    }


def _paged_kv_cache_shape(
    *,
    batch_size: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> list[int]:
    batch = _safe_int(batch_size)
    cache = _safe_int(cache_len)
    kv_heads = _safe_int(num_kv_heads)
    dim = _safe_int(head_dim)
    block = _safe_int(page_block_size) or 32
    if None in (batch, cache, kv_heads, dim) or cache < 0 or block <= 0:
        return []
    pages_per_user = max(1, (cache + block - 1) // block)
    return [batch * pages_per_user, kv_heads, block, dim]


def _attention_primitive_reports_complete(
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
        if not _non_empty_string(environment.get("version")):
            return False
        if not _non_empty_string(environment.get("tt_metal_git_commit")):
            return False
    return True


def _attention_primitive_reports_observed(
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
                "tt_metal_git_commit": environment.get(
                    "tt_metal_git_commit"
                )
                if isinstance(environment, dict)
                else None,
            }
        )
    return observed


def _attention_layer_primitive_reports_complete(reports: Any) -> bool:
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


def _attention_layer_primitive_reports_observed(
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
                "input_shape_keys": _field_keys(input_shapes),
                "expected_output_shape_keys": _field_keys(expected_shapes),
                "output_shape_keys": _field_keys(output_shapes),
                "dtype": report.get("dtype"),
                "layout": report.get("layout"),
                "memory_config": report.get("memory_config"),
            }
        )
    return observed


def _shape_dict_has_int_lists(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(_int_list(shape) for shape in value.values())


def _expected_decode_output_shape_summary(
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    seq = _safe_int(seq_len)
    vocab = _safe_int(vocab_size)
    layers = _safe_int(layer_count)
    token_shape = [batch, seq] if batch is not None and seq is not None else []
    token_vector = [batch] if batch is not None else []
    normalized_output_kind = (
        "logits" if output_kind == "logits" else "token"
    )
    logits_shape = (
        [batch, seq, vocab]
        if None not in (batch, seq, vocab)
        else []
    )
    output_shape = logits_shape if normalized_output_kind == "logits" else token_shape
    kv_shape = _paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    layer_ids = list(range(layers)) if layers is not None and layers > 0 else []
    return {
        "output_kind": normalized_output_kind,
        "output_shape": output_shape,
        "accepted_output_shapes": [
            shape for shape in (token_shape, token_vector) if shape
        ]
        if normalized_output_kind == "token"
        else [output_shape],
        "kv_cache_shape": kv_shape,
        "kv_cache_layer_ids": layer_ids,
    }


def _decode_output_shape_observed(output_shapes: Any) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    layers = output_shapes.get("kv_cache_layers")
    return {
        "output_kind": "logits" if "logits" in output_shapes else "token",
        "token": _int_list(output_shapes.get("token")),
        "logits": _int_list(output_shapes.get("logits")),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
        "kv_cache_layer_ids": [
            _safe_int(layer.get("layer_id"))
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
        "kv_cache_layer_shapes": [
            {
                "layer_id": _safe_int(layer.get("layer_id")),
                "key_cache": _int_list(layer.get("key_cache")),
                "value_cache": _int_list(layer.get("value_cache")),
            }
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
    }


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        converted = _safe_int(item)
        if converted is None:
            return []
        result.append(converted)
    return result


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        "reference_planned_observed_ops": reference.get(
            "planned_observed_ops"
        ),
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
