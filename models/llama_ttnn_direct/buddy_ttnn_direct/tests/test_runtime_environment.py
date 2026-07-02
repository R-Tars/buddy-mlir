from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.runtime_environment import (
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


if __name__ == "__main__":
    unittest.main()
