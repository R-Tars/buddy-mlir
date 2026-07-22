from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_RUNTIME_MODULES = (
    "smoke_",
    "decode_loop",
    "profile_template",
)


class ProductImportBoundaryTest(unittest.TestCase):
    def test_runtime_does_not_import_legacy_diagnostics(self) -> None:
        violations = _import_violations(
            PACKAGE_ROOT / "runtime",
            FORBIDDEN_RUNTIME_MODULES,
        )
        self.assertEqual(violations, [])

    def test_product_reports_do_not_import_diagnostics(self) -> None:
        violations = _import_violations(
            PACKAGE_ROOT / "reports",
            ("diagnostics",),
        )
        self.assertEqual(violations, [])

    def test_cli_top_level_does_not_import_smoke_modules(self) -> None:
        cli_path = PACKAGE_ROOT / "cli.py"
        tree = ast.parse(cli_path.read_text(), filename=str(cli_path))
        imported = [
            module
            for statement in tree.body
            for module in _statement_imports(statement)
        ]
        self.assertFalse(
            [module for module in imported if "smoke_" in module]
        )
        self.assertNotIn("_choices_actions", cli_path.read_text())

    def test_product_parser_exposes_only_product_commands(self) -> None:
        from models.llama_ttnn_direct.buddy_ttnn_direct.cli import (
            PRODUCT_COMMANDS,
            build_parser,
        )

        parser = build_parser()
        command_actions = [
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        ]
        self.assertEqual(len(command_actions), 1)
        self.assertEqual(
            set(command_actions[0].choices),
            set(PRODUCT_COMMANDS),
        )

    def test_building_product_parser_only_loads_diagnostics_registration(self) -> None:
        script = """
import sys
from models.llama_ttnn_direct.buddy_ttnn_direct.cli import build_parser
build_parser()
loaded = [
    name for name in sys.modules
    if '.smoke_' in name
    or '.diagnostics.benchmark_' in name
    or '.diagnostics.execution_graph_' in name
    or '.diagnostics.performance_' in name
    or '.diagnostics.autotune' in name
]
raise SystemExit(1 if loaded else 0)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


def _import_violations(
    root: Path, forbidden: tuple[str, ...]
) -> list[str]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for statement in ast.walk(tree):
            for module in _statement_imports(statement):
                if any(_module_matches(module, name) for name in forbidden):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
    return violations


def _module_matches(module: str, forbidden: str) -> bool:
    if forbidden == "diagnostics":
        return forbidden in module.split(".")
    return forbidden in module


def _statement_imports(statement: ast.AST) -> list[str]:
    if isinstance(statement, ast.Import):
        return [alias.name for alias in statement.names]
    if isinstance(statement, ast.ImportFrom):
        return [statement.module or ""]
    return []


if __name__ == "__main__":
    unittest.main()
