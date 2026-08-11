from __future__ import annotations

import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment import (
    collect_tenstorrent_device_environment,
    collect_tenstorrent_process_environment,
    collect_tenstorrent_setup_environment,
    collect_ttnn_environment,
    collect_ttnn_runtime_health,
)


MODULE = "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment"


def _process_environment(output: str) -> dict[str, object]:
    completed = subprocess.CompletedProcess(["ps"], 0, stdout=output, stderr="")
    with patch(f"{MODULE}.subprocess.run", return_value=completed):
        return collect_tenstorrent_process_environment()


class RuntimeEnvironmentTest(unittest.TestCase):
    def assertFields(self, actual: dict[str, object], **expected: object) -> None:
        self.assertEqual({key: actual[key] for key in expected}, expected)

    def test_collect_ttnn_environment_reads_module_commit(self) -> None:
        module = types.SimpleNamespace(__version__="1.2.3", __file__="/tmp/ttnn/__init__.py", __tt_metal_commit__="abc123")
        environment = collect_ttnn_environment(module)
        self.assertFields(
            environment,
            module_available=True,
            version="1.2.3",
            module_file="/tmp/ttnn/__init__.py",
            tt_metal_git_commit="abc123",
            tt_metal_git_commit_source="module.__tt_metal_commit__",
        )

    def test_collect_ttnn_environment_falls_back_to_env_commit(self) -> None:
        with patch.dict("os.environ", {"TT_METAL_GIT_COMMIT": "env456"}, clear=True):
            environment = collect_ttnn_environment(None)
        self.assertFields(
            environment,
            module_available=False,
            version=None,
            tt_metal_git_commit="env456",
            tt_metal_git_commit_source="env.TT_METAL_GIT_COMMIT",
        )

    def test_collect_tenstorrent_device_environment_records_tt_smi_probe(self) -> None:
        completed = subprocess.CompletedProcess(["/usr/bin/tt-smi"], 0, stdout="device 0 ready\n", stderr="")
        with (
            patch(f"{MODULE}._tenstorrent_device_entries", return_value=["/dev/tenstorrent/0"]),
            patch(f"{MODULE}._is_character_device", return_value=True),
            patch(f"{MODULE}._kernel_module_loaded", return_value=True),
            patch(f"{MODULE}.shutil.which", return_value="/usr/bin/tt-smi"),
            patch(f"{MODULE}.subprocess.run", return_value=completed),
        ):
            environment = collect_tenstorrent_device_environment()
        self.assertFields(
            environment,
            device_available=True,
            device_nodes=["/dev/tenstorrent/0"],
            tt_smi_path="/usr/bin/tt-smi",
        )
        self.assertFields(
            environment["tt_smi"],
            status="pass",
            returncode=0,
            stdout="device 0 ready",
        )

    def test_collect_tenstorrent_device_environment_without_tt_smi(self) -> None:
        with (
            patch(f"{MODULE}._tenstorrent_device_entries", return_value=[]),
            patch(f"{MODULE}._kernel_module_loaded", return_value=False),
            patch(f"{MODULE}.shutil.which", return_value=None),
        ):
            environment = collect_tenstorrent_device_environment()
        self.assertFields(
            environment,
            device_available=False,
            device_nodes=[],
            tt_smi_path=None,
            tt_smi=None,
        )

    def test_collect_tenstorrent_setup_environment_reports_probe_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs = root / "docs" / "TenstorrentEnvironment.md"
            docs.parent.mkdir()
            docs.write_text("env docs\n")
            toolchain = root / "toolchain"
            python = toolchain / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/usr/bin/env python\n")
            environment_values = {
                "BUDDY_REPO_ROOT": str(root),
                "BUDDY_LLVM_BUILD": str(root / "llvm" / "build"),
                "TTMLIR_TOOLCHAIN_DIR": str(toolchain),
                "TTMLIR_ENV_BUILD": str(root / "env-build"),
                "TTMLIR_BUILD": str(root / "ttmlir-build"),
                "BUDDY_BUILD": str(root / "buddy-build"),
            }
            with (
                patch.dict("os.environ", environment_values, clear=True),
                patch(
                    f"{MODULE}.importlib.util.find_spec",
                    side_effect=lambda name: object() if name == "ttrt" else None,
                ),
            ):
                environment = collect_tenstorrent_setup_environment(root)
        self.assertFields(
            environment,
            reference_doc_available=True,
            env_missing=[],
            ttmlir_toolchain_python_exists=True,
            ttrt_module_available=True,
            ttnn_module_available=False,
        )
        self.assertIn("-m ttrt query", environment["recommended_probe_commands"][1])

    def test_collect_tenstorrent_process_environment_detects_reset(self) -> None:
        environment = _process_environment(
            """USER PID PPID STAT ELAPSED CMD
alice 100 1 Ss 00:10 /bin/bash -c tt-smi -r 0
alice 101 100 R 00:09 /opt/tt/bin/tt-smi -r 0
bob 200 1 Sl 00:02 python -m examples.tenstorrent.eltwise_binary.eltwise_binary
"""
        )
        self.assertFields(
            environment, status="busy", conflict_count=3, reset_in_progress=True
        )
        self.assertEqual(environment["conflicts"][0]["kind"], "tt_smi_reset")

    def test_collect_tenstorrent_process_environment_detects_debug_workload(self) -> None:
        environment = _process_environment(
            """USER PID PPID STAT ELAPSED CMD
alice 100 1 Sl 00:20 python /project/.test/softmax_launch_debug/run_ttk.py --package-dir /tmp/pkg --mode blocking
"""
        )
        self.assertFields(
            environment, status="busy", conflict_count=1, reset_in_progress=False
        )
        self.assertEqual(environment["conflicts"][0]["kind"], "tenstorrent_workload")

    def test_collect_ttnn_runtime_health_reports_probe_failure(self) -> None:
        completed = subprocess.CompletedProcess(
            ["python", "-c", "..."], 135, stdout="", stderr="Bus error (core dumped)"
        )
        with patch(f"{MODULE}.subprocess.run", return_value=completed):
            environment = collect_ttnn_runtime_health(device_id=0, timeout=1.0)
        self.assertFields(environment, status="fail", device_id=0, returncode=135)
        self.assertIn("Bus error", environment["stderr"])


if __name__ == "__main__":
    unittest.main()
