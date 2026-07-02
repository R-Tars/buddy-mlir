from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codegen.artifacts import prepare_offline_artifacts
from .codegen.config_emit import (
    SEED_RECIPE,
    dump_parameter_config,
    emit_parameter_config,
    parameter_config_dry_run_report,
)
from .codegen.python_ttnn import dry_run_report, write_python_ttnn_skeleton
from .semantic.dump import dump_graph_json, load_graph_json
from .semantic.importer_hf_llama import import_hf_llama
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, run_smoke_mlp
from .templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_execution_plan,
    load_template_config,
)
from .templates.diff import (
    diff_plan_against_official,
    dump_plan_diff,
    load_official_template,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buddy-ttnn-direct",
        description="Buddy-TTNN Direct model tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_llama = subparsers.add_parser(
        "import-llama",
        help="Import an HF Llama config into a semantic graph JSON.",
    )
    import_llama.add_argument(
        "--model-path",
        required=True,
        help="Local HF model directory or model id.",
    )
    import_llama.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="decode",
        help="Semantic graph mode to emit.",
    )
    import_llama.add_argument("--batch-size", type=int, required=True)
    import_llama.add_argument("--seq-len", type=int, required=True)
    import_llama.add_argument("--max-cache-len", type=int, required=True)
    import_llama.add_argument(
        "--generation-mode",
        choices=("greedy", "sampling"),
        default="greedy",
    )
    import_llama.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit graph JSON from config and metadata only. Phase 1 never "
            "loads full tensor payloads, so this mainly documents intent."
        ),
    )
    import_llama.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output semantic graph JSON path.",
    )
    import_llama.set_defaults(func=_cmd_import_llama)

    plan = subparsers.add_parser(
        "plan",
        help="Map a Llama semantic graph to official-like TTNN templates.",
    )
    plan.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    plan.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Template seed config JSON.",
    )
    plan.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output execution plan JSON path.",
    )
    plan.set_defaults(func=_cmd_plan)

    codegen_python = subparsers.add_parser(
        "codegen-python",
        help="Generate a Python TTNN skeleton from an execution plan.",
    )
    codegen_python.add_argument(
        "--plan-json",
        type=Path,
        required=True,
        help="Input execution plan JSON from plan.",
    )
    codegen_python.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated Python TTNN skeleton artifacts.",
    )
    codegen_python.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report generated artifact paths without writing.",
    )
    codegen_python.set_defaults(func=_cmd_codegen_python)

    emit_config = subparsers.add_parser(
        "emit-config",
        help="Emit parameter dtype/layout/packing metadata for a Llama graph.",
    )
    emit_config.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    emit_config.add_argument(
        "--recipe",
        default=SEED_RECIPE,
        help="Parameter metadata recipe name.",
    )
    emit_config.add_argument(
        "--lm-head-split-count",
        type=int,
        default=8,
        help="Number of vocab shards for split LM-head metadata.",
    )
    emit_config.add_argument(
        "--kv-page-block-size",
        type=int,
        default=32,
        help="Paged KV cache block size in tokens.",
    )
    emit_config.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output parameter metadata JSON path.",
    )
    emit_config.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report metadata summary without writing.",
    )
    emit_config.set_defaults(func=_cmd_emit_config)

    diff_plan = subparsers.add_parser(
        "diff-plan",
        help="Diff a TTNN Direct plan against an official-like decode template.",
    )
    diff_plan.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="Input TTNN Direct execution plan JSON.",
    )
    diff_plan.add_argument(
        "--official-template",
        type=Path,
        required=True,
        help="Official-like reference decode template JSON.",
    )
    diff_plan.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output structural diff JSON path.",
    )
    diff_plan.set_defaults(func=_cmd_diff_plan)

    prepare_artifacts = subparsers.add_parser(
        "prepare-artifacts",
        help="Prepare offline TTNN Direct weight and cache manifests.",
    )
    prepare_artifacts.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF model directory containing safetensors metadata.",
    )
    prepare_artifacts.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    prepare_artifacts.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Parameter metadata JSON from emit-config.",
    )
    prepare_artifacts.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for offline manifest artifacts.",
    )
    prepare_artifacts.set_defaults(func=_cmd_prepare_artifacts)

    smoke_mlp = subparsers.add_parser(
        "smoke-mlp",
        help="Run or dry-run a small TTNN MLP template smoke test.",
    )
    smoke_mlp.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_mlp.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_mlp.add_argument("--batch-size", type=int, required=True)
    smoke_mlp.add_argument("--hidden-size", type=int, required=True)
    smoke_mlp.add_argument("--intermediate-size", type=int, required=True)
    smoke_mlp.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_mlp.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.99,
        help="Minimum PCC required for a hardware smoke pass.",
    )
    smoke_mlp.add_argument("--seed", type=int, default=0)
    smoke_mlp.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the smoke report schema without opening a TTNN device.",
    )
    smoke_mlp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output smoke report JSON path.",
    )
    smoke_mlp.set_defaults(func=_cmd_smoke_mlp)
    return parser


def _cmd_import_llama(args: argparse.Namespace) -> int:
    graph = import_hf_llama(
        args.model_path,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_cache_len=args.max_cache_len,
        generation_mode=args.generation_mode,
    )
    dump_graph_json(graph, args.out)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"wrote Llama semantic graph{suffix}: {args.out}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    config = load_template_config(args.config)
    plan = build_execution_plan(graph, config)
    dump_execution_plan(plan, args.out)
    print(f"wrote TTNN Direct execution plan: {args.out}")
    return 0


def _cmd_codegen_python(args: argparse.Namespace) -> int:
    plan = load_execution_plan(args.plan_json)
    if args.dry_run:
        print(json.dumps(dry_run_report(plan, args.out_dir), indent=2))
        return 0

    paths = write_python_ttnn_skeleton(plan, args.out_dir)
    print(f"wrote Python TTNN skeleton: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_emit_config(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    if args.dry_run:
        print(
            json.dumps(
                parameter_config_dry_run_report(
                    graph,
                    recipe=args.recipe,
                    lm_head_split_count=args.lm_head_split_count,
                    kv_page_block_size=args.kv_page_block_size,
                ),
                indent=2,
            )
        )
        return 0

    config = emit_parameter_config(
        graph,
        recipe=args.recipe,
        lm_head_split_count=args.lm_head_split_count,
        kv_page_block_size=args.kv_page_block_size,
    )
    dump_parameter_config(config, args.out)
    print(f"wrote parameter metadata config: {args.out}")
    return 0


def _cmd_diff_plan(args: argparse.Namespace) -> int:
    plan = load_execution_plan(args.ours)
    official_template = load_official_template(args.official_template)
    diff = diff_plan_against_official(plan, official_template)
    dump_plan_diff(diff, args.out)
    print(f"wrote TTNN Direct plan diff: {args.out}")
    return 0


def _cmd_prepare_artifacts(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    config = json.loads(args.config.read_text())
    paths = prepare_offline_artifacts(
        args.model_path,
        graph,
        config,
        args.out_dir,
    )
    print(f"wrote TTNN Direct artifact manifests: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_smoke_mlp(args: argparse.Namespace) -> int:
    report = run_smoke_mlp(
        out=args.out,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
        pcc_threshold=args.pcc_threshold,
        seed=args.seed,
    )
    print(f"wrote MLP smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
