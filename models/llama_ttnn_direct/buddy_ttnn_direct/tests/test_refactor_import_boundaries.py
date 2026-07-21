from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_EVIDENCE = re.compile(r"[A-Za-z0-9_]+_evidence_\d{8}\.json")


class RefactorImportBoundaryTest(unittest.TestCase):
    def test_runtime_does_not_reference_documentation_evidence(self) -> None:
        violations = []
        for path in sorted((PACKAGE_ROOT / "runtime").rglob("*.py")):
            text = path.read_text()
            if (
                "docs.evidence" in text
                or "docs/evidence" in text
                or HISTORICAL_EVIDENCE.search(text)
            ):
                violations.append(str(path.relative_to(PACKAGE_ROOT)))
        self.assertEqual(violations, [])

    def test_removed_template_is_absent_from_package_graph(self) -> None:
        removed = PACKAGE_ROOT / "templates" / "mlp_prefill.py"
        self.assertFalse(removed.exists())
        references = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module.endswith(".templates.mlp_prefill") for module in modules):
                    references.append(str(path.relative_to(PACKAGE_ROOT)))
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()
