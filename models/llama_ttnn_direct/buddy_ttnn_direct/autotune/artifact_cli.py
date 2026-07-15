from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifact import build_paper_artifact, verify_paper_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a TTNN Direct semantic-autotune paper artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build an artifact from a spec")
    build.add_argument("--spec", required=True, type=Path)
    build.add_argument("--out-dir", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="verify an artifact bundle")
    verify.add_argument("--artifact-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        report = build_paper_artifact(args.spec, out_dir=args.out_dir)
    else:
        report = verify_paper_artifact(args.artifact_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
