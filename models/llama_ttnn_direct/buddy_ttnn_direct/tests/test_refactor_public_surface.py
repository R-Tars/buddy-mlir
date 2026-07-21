from __future__ import annotations

import argparse
import contextlib
import io
import unittest

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import build_parser


PRODUCT_COMMANDS = {
    "build",
    "generate",
    "profile",
    "validate",
    "inspect",
    "diagnose",
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

    def test_every_product_command_parses_help(self) -> None:
        for command in sorted(PRODUCT_COMMANDS):
            with self.subTest(command=command):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        build_parser().parse_args([command, "--help"])
                self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
