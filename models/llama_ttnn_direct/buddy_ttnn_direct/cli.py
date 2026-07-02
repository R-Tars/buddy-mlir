from __future__ import annotations

import argparse
from pathlib import Path

from .semantic.dump import dump_graph_json, load_graph_json
from .semantic.importer_hf_llama import import_hf_llama
from .templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_template_config,
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
