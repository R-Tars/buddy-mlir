from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.package import (
    PACKAGE_BACKEND,
    PACKAGE_PROGRAM_TYPE,
    package_dry_run_report,
    package_ttnn_direct_program,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.program import PROGRAM_ARTIFACTS
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.fakes import _write_fake_model_config, _write_template_config


class PackageProgramTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model, self.template, self.program = self.root / "model", self.root / "template.json", self.root / "program"
        _write_fake_model_config(self.model)
        config = json.loads((self.model / "config.json").read_text()); config["_name_or_path"] = "fake-package-program"
        (self.model / "config.json").write_text(json.dumps(config))
        _write_template_config(self.template)
        self.assertEqual(main(["build", "--model-path", str(self.model), "--config", str(self.template), "--out-dir", str(self.program)]), 0)

    def test_package_program_writes_direct_package_manifest(self) -> None:
        out = self.root / "package"
        package_ttnn_direct_program(self.program, out)
        self.assertEqual({path.name for path in out.iterdir()}, {*PROGRAM_ARTIFACTS, "manifest.json", "PACKAGE_README.md"})
        manifest = json.loads((out / "manifest.json").read_text())
        expected = {"backend": PACKAGE_BACKEND, "program_type": PACKAGE_PROGRAM_TYPE, "entrypoint": "model.py", "semantic_graph": "semantic_graph.json", "execution_plan": "execution_plan.json", "weights_manifest": "weights_manifest.json", "model_name": "fake-package-program", "num_layers": 2}
        self.assertEqual({key: manifest[key] for key in expected}, expected)
        runtime = manifest["runtime"]
        self.assertEqual({key: runtime[key] for key in ("buddy_cli_supported", "python_runner_supported", "python_runner", "runner_modes", "dry_run_supported", "real_weight_validation_supported")}, {
            "buddy_cli_supported": True, "python_runner_supported": True, "python_runner": "run_decode.py",
            "runner_modes": ["build", "generate", "profile", "validate", "inspect", "diagnose"],
            "dry_run_supported": True, "real_weight_validation_supported": True,
        })
        self.assertEqual(runtime["legacy_mode_mappings"]["smoke"], "diagnose --stage decode-step")
        readme = (out / "PACKAGE_README.md").read_text()
        for marker in ("TTNN_DIRECT_PACKAGE_DIR", "TT_METAL_LOGS_PATH", 'run_decode.py" inspect', "--stage decode-step", "--stage prefill", "profile --mode generate", "validate --suite dryrun"):
            self.assertIn(marker, readme)
        self.assertNotIn("/tmp", readme)

    def test_package_program_dry_run_reports_manifest_json(self) -> None:
        out = self.root / "package"
        report = package_dry_run_report(self.program, out)
        self.assertFalse(out.exists())
        self.assertEqual((report["dry_run"], report["backend"], report["program_type"], report["manifest"]["entrypoint"], report["manifest"]["runtime"]["python_runner_supported"]), (True, PACKAGE_BACKEND, PACKAGE_PROGRAM_TYPE, "model.py", True))
        self.assertIn("manifest.json", report["artifacts"])

    def test_cmake_target_is_additive_and_separate_from_llama31_tt_rax(self) -> None:
        repo = Path(__file__).parents[4]
        files = ((repo / "CMakeLists.txt", ("BUDDY_BUILD_LLAMA31_TTNN_DIRECT_MODEL",)), (repo / "models/CMakeLists.txt", ("add_subdirectory(llama_ttnn_direct)",)), (repo / "models/llama_ttnn_direct/CMakeLists.txt", ("llama31_ttnn_direct_program", "llama31_ttnn_direct_package", "runtime_artifacts", "evidence_archive", "build\n")))
        for path, markers in files:
            source = path.read_text()
            for marker in markers:
                self.assertIn(marker, source)
        self.assertNotIn("llama31_tt_rax", files[-1][0].read_text())
