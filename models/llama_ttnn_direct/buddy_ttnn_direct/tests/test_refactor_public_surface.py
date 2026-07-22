from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import (
    PRODUCT_COMMANDS as CLI_PRODUCT_COMMANDS,
    build_parser,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.cli import (
    DIAGNOSE_STAGES,
)


PRODUCT_COMMANDS = {
    "build",
    "generate",
    "profile",
    "validate",
    "inspect",
    "diagnose",
}
LEGACY_ROOT_COMMANDS = {
    "import-llama",
    "plan",
    "codegen-python",
    "emit-config",
    "smoke-mlp",
    "smoke-prefill",
    "build-program",
    "materialize-parameters",
    "tensorize-parameters",
    "search",
    "autotune-decode-step",
    "validate-direct",
    "validate-real-decode",
}
EXPECTED_DIAGNOSE_STAGES = {
    "mlp",
    "attention-primitive",
    "attention-layer",
    "prefill",
    "decode-shell",
    "decode-step",
    "decode-step-profile",
    "decode-loop-legacy",
    "depth-sweep",
    "generate-depth-sweep",
    "autotune",
    "autotune-profiler-audit",
    "benchmark-parity",
    "execution-graph-diff",
    "performance-correctness",
    "template-profile",
}


class RefactorPublicSurfaceTest(unittest.TestCase):
    def test_product_commands_are_exactly_preserved(self) -> None:
        parser = build_parser()
        actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertEqual(len(actions), 1)
        self.assertEqual(set(actions[0].choices), PRODUCT_COMMANDS)
        self.assertEqual(set(CLI_PRODUCT_COMMANDS), PRODUCT_COMMANDS)

    def test_every_product_command_parses_help(self) -> None:
        for command in sorted(PRODUCT_COMMANDS):
            with self.subTest(command=command):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        build_parser().parse_args([command, "--help"])
                self.assertEqual(raised.exception.code, 0)

    def test_legacy_root_commands_are_rejected(self) -> None:
        for command in sorted(LEGACY_ROOT_COMMANDS):
            with self.subTest(command=command):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        build_parser().parse_args([command])
                self.assertEqual(raised.exception.code, 2)

    def test_diagnose_stages_remain_parseable(self) -> None:
        self.assertEqual(set(DIAGNOSE_STAGES), EXPECTED_DIAGNOSE_STAGES)
        for stage in DIAGNOSE_STAGES:
            with self.subTest(stage=stage):
                args = build_parser().parse_args(
                    [
                        "diagnose",
                        "--stage",
                        stage,
                        "--out",
                        "/tmp/diagnose.json",
                    ]
                )
                self.assertEqual(args.stage, stage)

    def test_hidden_parser_registry_is_absent(self) -> None:
        cli_source = Path(build_parser.__code__.co_filename).read_text()
        self.assertNotIn("ProductCommandRegistry", cli_source)
        self.assertNotIn("legacy_parsers", cli_source)


if __name__ == "__main__":
    unittest.main()
