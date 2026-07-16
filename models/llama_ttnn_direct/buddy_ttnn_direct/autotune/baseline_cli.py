from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .baseline import build_baseline_artifact, verify_baseline_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a TTNN Direct Phase 0 baseline bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a baseline bundle")
    build.add_argument("--reports", required=True, nargs=3, type=Path)
    build.add_argument("--program-dir", required=True, type=Path)
    build.add_argument("--seed-config", required=True, type=Path)
    build.add_argument("--prompt-corpus", required=True, type=Path)
    build.add_argument("--ttnn-binary", required=True, type=Path)
    build.add_argument("--out-dir", required=True, type=Path)
    build.add_argument("--repo-root", type=Path)
    build.add_argument("--buddy-commit")
    build.add_argument("--tt-metal-root", type=Path)
    verify = subparsers.add_parser("verify", help="verify a baseline bundle")
    verify.add_argument("--artifact-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        report = build_baseline_artifact(
            args.reports,
            program_dir=args.program_dir,
            seed_config=args.seed_config,
            prompt_corpus=args.prompt_corpus,
            ttnn_binary=args.ttnn_binary,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            buddy_commit=args.buddy_commit,
            tt_metal_root=args.tt_metal_root,
        )
    else:
        report = verify_baseline_artifact(args.artifact_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
