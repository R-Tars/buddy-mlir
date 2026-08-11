from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "models.llama_ttnn_direct.buddy_ttnn_direct"
PRODUCT_DIRS = ("semantic", "compiler", "codegen", "runtime", "ttnn_compat", "reports")
PRODUCT_COMMANDS = {"build", "generate", "profile", "validate", "inspect", "diagnose"}
CLI_MODULE = f"{PACKAGE}.cli"
PRODUCT_AUTOTUNE_DEBT = {
    ("compiler/tuning.py", f"{PACKAGE}.autotune.space"),
    ("codegen/parameters.py", f"{PACKAGE}.autotune.templates"),
    ("codegen/ttnn_tensorizer.py", f"{PACKAGE}.autotune.templates"),
}
AUTOTUNE_SMOKE_DEBT = set()
DIAGNOSTICS_AUTOTUNE_ENTRYPOINT = {
    ("diagnostics/cli.py", f"{PACKAGE}.autotune.campaign"),
}
DIAGNOSTICS_SHARED_AUTOTUNE = {
    f"{PACKAGE}.autotune.microbench",
}
RETIRED_VALIDATION_MODULES = (
    f"{PACKAGE}.diagnostics.validation_workflow",
    f"{PACKAGE}.diagnostics.legacy_validation",
)
RETIRED_REPORT_HELPERS = tuple(
    f"{PACKAGE}.reports.{name}"
    for name in (
        "artifacts",
        "attention",
        "config",
        "depth",
        "evidence",
        "performance",
        "runtime_diagnostics",
        "tensorization",
    )
)
REMOVED = ("generate.py", "runtime_inputs.py", "validation.py", "codegen/python_ttnn.py", "decode_loop.py", "profile_template.py")
class Phase3ImportBoundaryTest(unittest.TestCase):
    def test_product_modules_have_no_new_reverse_dependencies(self) -> None:
        observed, unexpected = set(), []
        for path in _python_files(PRODUCT_DIRS):
            relative = path.relative_to(ROOT).as_posix()
            for imported in _imports(path):
                domain = _domain(imported)
                edge = (relative, imported)
                if domain == "autotune":
                    observed.add(edge)
                    if edge not in PRODUCT_AUTOTUNE_DEBT:
                        unexpected.append(edge)
                elif domain in {"diagnostics", "correctness"} or _legacy(imported):
                    unexpected.append(edge)
        self.assertEqual(unexpected, [])
        self.assertEqual(observed, PRODUCT_AUTOTUNE_DEBT)
    def test_cli_only_imports_diagnostics_registration_and_dispatch(self) -> None:
        imports = _function_imports(ROOT / "cli.py")
        diagnostics = {item for item in imports if item[1].startswith(f"{PACKAGE}.diagnostics")}
        self.assertEqual(
            diagnostics,
            {
                ("build_parser", f"{PACKAGE}.diagnostics.cli", ("add_diagnose_arguments",)),
                ("_cmd_diagnose", f"{PACKAGE}.diagnostics.cli", ("run_stage",)),
            },
        )
        self.assertEqual(
            [item for item in imports if _domain(item[1]) == "autotune" or _legacy(item[1])],
            [],
        )
    def test_autotune_has_no_new_diagnostics_or_smoke_dependencies(self) -> None:
        observed, unexpected = set(), []
        for path in _python_files(("autotune",)):
            relative = path.relative_to(ROOT).as_posix()
            for imported in _imports(path):
                edge = (relative, imported)
                if imported in {
                    f"{PACKAGE}.diagnostics.cli",
                    f"{PACKAGE}.diagnostics.validation_workflow",
                }:
                    unexpected.append(edge)
                elif _smoke(imported):
                    observed.add(edge)
                    if edge not in AUTOTUNE_SMOKE_DEBT:
                        unexpected.append(edge)
        self.assertEqual(unexpected, [])
        self.assertEqual(observed, set())

    def test_semantic_autotune_is_the_only_tuning_owner(self) -> None:
        self.assertFalse((ROOT / "diagnostics" / "autotune").exists())
        reverse_edges = []
        for path in _python_files(("autotune",)):
            for imported in _imports(path):
                if _domain(imported) == "diagnostics":
                    reverse_edges.append((path.relative_to(ROOT).as_posix(), imported))
        self.assertEqual(reverse_edges, [])

        observed, unexpected = set(), []
        for path in _python_files(("diagnostics",)):
            relative = path.relative_to(ROOT).as_posix()
            for imported in _imports(path):
                if _domain(imported) != "autotune":
                    continue
                edge = (relative, imported)
                observed.add(edge)
                if imported not in DIAGNOSTICS_SHARED_AUTOTUNE and edge not in DIAGNOSTICS_AUTOTUNE_ENTRYPOINT:
                    unexpected.append(edge)
        self.assertEqual(unexpected, [])
        self.assertEqual(
            observed,
            DIAGNOSTICS_AUTOTUNE_ENTRYPOINT
            | {
                (path.relative_to(ROOT).as_posix(), imported)
                for path in _python_files(("diagnostics",))
                for imported in _imports(path)
                if imported in DIAGNOSTICS_SHARED_AUTOTUNE
            },
        )

    def test_retired_layered_tuner_symbols_are_absent(self) -> None:
        markers = (
            "AUTO" + "TUNE_LEVELS",
            "run_" + "layered_autotune",
            "_select_" + "winner",
            "_confirmation_" + "promotion_decision",
        )
        violations = []
        for path in sorted(ROOT.rglob("*.py")):
            text = path.read_text()
            if any(marker in text for marker in markers):
                violations.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(violations, [])

    def test_phase_era_validation_orchestration_is_absent(self) -> None:
        self.assertFalse((ROOT / "diagnostics" / "validation_workflow.py").exists())
        self.assertFalse((ROOT / "diagnostics" / "legacy_validation.py").exists())
        violations = []
        for path in sorted(ROOT.rglob("*.py")):
            for imported in _imports(path):
                if any(
                    imported == retired or imported.startswith(f"{retired}.")
                    for retired in RETIRED_VALIDATION_MODULES
                ):
                    violations.append(
                        (path.relative_to(ROOT).as_posix(), imported)
                    )
        self.assertEqual(violations, [])

    def test_phase7_orphan_report_helpers_are_absent(self) -> None:
        for retired in RETIRED_REPORT_HELPERS:
            filename = f"{retired.rsplit('.', 1)[-1]}.py"
            self.assertFalse((ROOT / "reports" / filename).exists())
        violations = []
        for path in sorted(ROOT.rglob("*.py")):
            for imported in _imports(path):
                if any(
                    imported == retired or imported.startswith(f"{retired}.")
                    for retired in RETIRED_REPORT_HELPERS
                ):
                    violations.append(
                        (path.relative_to(ROOT).as_posix(), imported)
                    )
        self.assertEqual(violations, [])

    def test_phase3_facades_are_absent(self) -> None:
        self.assertEqual([name for name in REMOVED if (ROOT / name).exists()], [])

    def test_phase8_smoke_ownership_boundary_is_closed(self) -> None:
        smoke_files = sorted(ROOT.glob("smoke_*.py"))
        self.assertEqual(smoke_files, [])
        forbidden_prefix = f"{PACKAGE}.smoke_"
        violations = []
        for path in sorted(ROOT.rglob("*.py")):
            for imported in _imports(path):
                if imported == forbidden_prefix.rstrip("_") or imported.startswith(
                    forbidden_prefix
                ):
                    violations.append(
                        (path.relative_to(ROOT).as_posix(), imported)
                    )
        self.assertEqual(violations, [])

    def test_retired_search_package_has_no_source_or_test_references(self) -> None:
        forbidden_package = f"{PACKAGE}.future"
        retired_marker = "historical" + "_search"
        violations = []
        self.assertFalse((ROOT / "future").exists())
        for path in sorted(ROOT.rglob("*.py")):
            imports = _imports(path)
            if any(
                imported == forbidden_package
                or imported.startswith(f"{forbidden_package}.")
                for imported in imports
            ):
                violations.append(path.relative_to(ROOT).as_posix())
            if retired_marker in path.read_text():
                violations.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(violations, [])

    def test_internal_cli_subprocesses_use_product_commands(self) -> None:
        violations = []
        for path in ROOT.rglob("*.py"):
            if any(part.startswith("tests") for part in path.parts):
                continue
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if not isinstance(node, ast.List):
                    continue
                values = [item.value if isinstance(item, ast.Constant) else None for item in node.elts]
                if CLI_MODULE in values:
                    index = values.index(CLI_MODULE)
                    command = values[index + 1] if index + 1 < len(values) else None
                    if command not in PRODUCT_COMMANDS:
                        violations.append((path.relative_to(ROOT).as_posix(), command))
        self.assertEqual(violations, [])
def _python_files(directories: tuple[str, ...]):
    for directory in directories:
        yield from sorted((ROOT / directory).rglob("*.py"))
def _imports(path: Path) -> list[str]:
    module = _module(path)
    result = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(module, path.name == "__init__.py", node)
            result.append(base)
            if node.module is None:
                result.extend(f"{base}.{alias.name}" for alias in node.names)
    return result
def _function_imports(path: Path) -> set[tuple[str, str, tuple[str, ...]]]:
    result = set()
    for function in (node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(function):
            if isinstance(node, ast.ImportFrom):
                result.add((function.name, _resolve(_module(path), False, node), tuple(a.name for a in node.names)))
            elif isinstance(node, ast.Import):
                result.update((function.name, alias.name, ()) for alias in node.names)
    return result


def _module(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((PACKAGE, *parts))


def _resolve(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = module.split(".") if is_package else module.split(".")[:-1]
    parts = parts[: len(parts) - node.level + 1]
    return ".".join(parts + (node.module.split(".") if node.module else []))


def _domain(module: str) -> str | None:
    prefix = f"{PACKAGE}."
    return module[len(prefix) :].split(".", 1)[0] if module.startswith(prefix) else None


def _smoke(module: str) -> bool:
    return any(part.startswith("smoke_") for part in module.split("."))


def _legacy(module: str) -> bool:
    return _smoke(module) or module.endswith(
        (".legacy_decode_loop", ".decode_loop", ".profile_template")
    )
