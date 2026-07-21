#!/usr/bin/env python3
"""Write the small, reproducible inventory used by TTNN Direct refactors."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


MODEL_ROOT = Path("models/llama_ttnn_direct")
PRODUCT_COMMANDS = ("build", "generate", "profile", "validate", "inspect", "diagnose")


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def category(path: Path) -> str:
    value = path.as_posix()
    if "/docs/evidence/" in value and path.suffix == ".json":
        return "evidence_json"
    if "/tests_diagnostics/" in value:
        return "diagnostic_tests"
    if "/tests/" in value:
        return "product_tests"
    for marker, name in (
        ("/semantic/", "product_semantic"),
        ("/runtime/", "product_runtime"),
        ("/ttnn_compat/", "product_ttnn_compat"),
        ("/autotune/", "autotune_core"),
        ("/diagnostics/", "diagnostics"),
        ("/correctness/", "correctness"),
        ("/future/", "legacy_future"),
    ):
        if marker in value:
            return name
    if any(marker in value for marker in ("/compiler/", "/codegen/", "/templates/")):
        return "product_compiler_codegen"
    if path.suffix == ".md":
        return "docs"
    if path.suffix == ".py":
        return "product_runtime"
    return "other"


def module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def local_imports(path: Path, known: set[str]) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    current = module_name(path).split(".")
    if path.name != "__init__.py":
        current.pop()
    imports: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = current[: len(current) - max(node.level - 1, 0)] if node.level else []
            prefix = ".".join(base + ([node.module] if node.module else []))
            names = [prefix]
            names.extend(f"{prefix}.{alias.name}" for alias in node.names if prefix)
        for name in names:
            probe = name
            while probe:
                if probe in known:
                    imports.add(probe)
                    break
                probe = probe.rpartition(".")[0]
    return sorted(imports)


def referenced(text: str, path: Path, module: str) -> bool:
    rel = path.relative_to(MODEL_ROOT).as_posix()
    return rel in text or path.name in text or module in text


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(item) for item in git("ls-files", MODEL_ROOT.as_posix()).splitlines()]
    paths = [path for path in paths if path.is_file()]
    records = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    texts: dict[Path, str] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = ""
        texts[path] = text
        record = {"path": path.as_posix(), "category": category(path), "lines": len(text.splitlines())}
        records.append(record)
        grouped[str(record["category"])].append(record)

    loc = {"schema_version": 1, "tracked_file_count": len(records), "tracked_lines": sum(r["lines"] for r in records), "categories": {}}
    for name, items in sorted(grouped.items()):
        suffix_lines = defaultdict(int)
        for item in items:
            suffix_lines[Path(str(item["path"])).suffix] += int(item["lines"])
        loc["categories"][name] = {
            "file_count": len(items),
            "total_lines": sum(int(item["lines"]) for item in items),
            "python_lines": suffix_lines[".py"],
            "json_lines": suffix_lines[".json"],
            "markdown_lines": suffix_lines[".md"],
            "largest_files": sorted(items, key=lambda item: int(item["lines"]), reverse=True)[:10],
        }
    write_json(args.out_dir / "tracked_loc.json", loc)
    write_json(args.out_dir / "tracked_files.json", {"schema_version": 1, "files": records})

    py_paths = [path for path in paths if path.suffix == ".py"]
    modules = {module_name(path): path for path in py_paths}
    imports = {module: local_imports(path, set(modules)) for module, path in modules.items()}
    imported_by: dict[str, list[str]] = defaultdict(list)
    for importer, dependencies in imports.items():
        for dependency in dependencies:
            imported_by[dependency].append(importer)
    cmake_text = "\n".join(text for path, text in texts.items() if path.name == "CMakeLists.txt")
    test_text = "\n".join(text for path, text in texts.items() if "/tests" in path.as_posix())
    docs_text = "\n".join(text for path, text in texts.items() if path.suffix == ".md")
    generated_text = "\n".join(text for path, text in texts.items() if any(part in path.parts for part in ("compiler", "codegen")))
    cli_module = "models.llama_ttnn_direct.buddy_ttnn_direct.cli"
    report = {"schema_version": 1, "python_file_count": len(py_paths), "modules": {}}
    unreferenced = []
    for module, path in sorted(modules.items()):
        init_export = any(
            modules[parent].name == "__init__.py"
            for parent in imported_by.get(module, [])
            if parent in modules
        )
        entry = {
            "path": path.as_posix(),
            "imports": imports[module],
            "imported_by": sorted(imported_by.get(module, [])),
            "cli_entrypoint": (
                module == cli_module
                or module in imports.get(cli_module, [])
                or 'if __name__ == "__main__"' in path.read_text(errors="replace")
            ),
            "cmake_reference": referenced(cmake_text, path, module),
            "all_export": init_export,
            "test_reference": referenced(test_text, path, module),
            "documentation_reference": referenced(docs_text, path, module),
            "generated_source_reference": referenced(generated_text, path, module),
        }
        report["modules"][module] = entry
        guards = [entry[key] for key in ("imported_by", "cli_entrypoint", "cmake_reference", "all_export", "test_reference", "documentation_reference", "generated_source_reference")]
        if path.name != "__init__.py" and "/tests" not in path.as_posix() and not any(guards):
            unreferenced.append(entry)
    write_json(args.out_dir / "dependency_report.json", report)
    write_json(args.out_dir / "unreferenced_files.json", {"schema_version": 1, "files": unreferenced})

    sys.path.insert(0, str(Path.cwd()))
    from models.llama_ttnn_direct.buddy_ttnn_direct.cli import build_parser

    command_choices: set[str] = set()
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            command_choices.update(action.choices)
    public = {"schema_version": 1, "expected_commands": list(PRODUCT_COMMANDS), "commands": sorted(command_choices), "passed": command_choices == set(PRODUCT_COMMANDS)}
    write_json(args.out_dir / "public_surface.json", public)
    source = {"schema_version": 1, "buddy_commit": git("rev-parse", "HEAD"), "buddy_tree": git("rev-parse", "HEAD^{tree}"), "inventory_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    write_json(args.out_dir / "source_identity.json", source)

    lines = ["# TTNN Direct tracked LOC", "", f"Total: {loc['tracked_lines']} lines in {loc['tracked_file_count']} files.", "", "| Category | Files | Lines |", "| --- | ---: | ---: |"]
    lines.extend(f"| {name} | {value['file_count']} | {value['total_lines']} |" for name, value in loc["categories"].items())
    (args.out_dir / "tracked_loc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if public["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
