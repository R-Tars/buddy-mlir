from __future__ import annotations

import types
import subprocess
import unittest
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment import (
    collect_tenstorrent_device_environment,
    collect_tenstorrent_process_environment,
    collect_ttnn_runtime_health,
    collect_ttnn_environment,
)


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_collect_ttnn_environment_reads_module_commit(self) -> None:
        module = types.SimpleNamespace()
        module.__version__ = "1.2.3"
        module.__file__ = "/tmp/ttnn/__init__.py"
        module.__tt_metal_commit__ = "abc123"

        environment = collect_ttnn_environment(module)

        self.assertTrue(environment["module_available"])
        self.assertEqual(environment["version"], "1.2.3")
        self.assertEqual(environment["module_file"], "/tmp/ttnn/__init__.py")
        self.assertEqual(environment["tt_metal_git_commit"], "abc123")
        self.assertEqual(
            environment["tt_metal_git_commit_source"],
            "module.__tt_metal_commit__",
        )

    def test_collect_ttnn_environment_falls_back_to_env_commit(self) -> None:
        with patch.dict(
            "os.environ",
            {"TT_METAL_GIT_COMMIT": "env456"},
            clear=True,
        ):
            environment = collect_ttnn_environment(None)

        self.assertFalse(environment["module_available"])
        self.assertIsNone(environment["version"])
        self.assertEqual(environment["tt_metal_git_commit"], "env456")
        self.assertEqual(
            environment["tt_metal_git_commit_source"],
            "env.TT_METAL_GIT_COMMIT",
        )

    def test_collect_tenstorrent_device_environment_records_tt_smi_probe(
        self,
    ) -> None:
        with patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "_tenstorrent_device_entries",
            return_value=["/dev/tenstorrent/0"],
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "_is_character_device",
            return_value=True,
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "_kernel_module_loaded",
            return_value=True,
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "shutil.which",
            return_value="/usr/bin/tt-smi",
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["/usr/bin/tt-smi"],
                0,
                stdout="device 0 ready\n",
                stderr="",
            ),
        ):
            environment = collect_tenstorrent_device_environment()

        self.assertTrue(environment["device_available"])
        self.assertEqual(environment["device_nodes"], ["/dev/tenstorrent/0"])
        self.assertEqual(environment["tt_smi_path"], "/usr/bin/tt-smi")
        self.assertEqual(environment["tt_smi"]["status"], "pass")
        self.assertEqual(environment["tt_smi"]["returncode"], 0)
        self.assertEqual(environment["tt_smi"]["stdout"], "device 0 ready")

    def test_collect_tenstorrent_device_environment_without_tt_smi(
        self,
    ) -> None:
        with patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "_tenstorrent_device_entries",
            return_value=[],
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "_kernel_module_loaded",
            return_value=False,
        ), patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "shutil.which",
            return_value=None,
        ):
            environment = collect_tenstorrent_device_environment()

        self.assertFalse(environment["device_available"])
        self.assertEqual(environment["device_nodes"], [])
        self.assertIsNone(environment["tt_smi_path"])
        self.assertIsNone(environment["tt_smi"])

    def test_collect_tenstorrent_process_environment_detects_reset(
        self,
    ) -> None:
        ps_output = """USER PID PPID STAT ELAPSED CMD
alice 100 1 Ss 00:10 /bin/bash -c tt-smi -r 0
alice 101 100 R 00:09 /opt/tt/bin/tt-smi -r 0
bob 200 1 Sl 00:02 python -m examples.tenstorrent.eltwise_binary.eltwise_binary
"""
        with patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["ps"],
                0,
                stdout=ps_output,
                stderr="",
            ),
        ):
            environment = collect_tenstorrent_process_environment()

        self.assertEqual(environment["status"], "busy")
        self.assertEqual(environment["conflict_count"], 3)
        self.assertTrue(environment["reset_in_progress"])
        self.assertEqual(environment["conflicts"][0]["kind"], "tt_smi_reset")

    def test_collect_ttnn_runtime_health_reports_probe_failure(self) -> None:
        with patch(
            "models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment."
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["python", "-c", "..."],
                135,
                stdout="",
                stderr="Bus error (core dumped)",
            ),
        ):
            environment = collect_ttnn_runtime_health(device_id=0, timeout=1.0)

        self.assertEqual(environment["status"], "fail")
        self.assertEqual(environment["device_id"], 0)
        self.assertEqual(environment["returncode"], 135)
        self.assertIn("Bus error", environment["stderr"])


if __name__ == "__main__":
    unittest.main()
