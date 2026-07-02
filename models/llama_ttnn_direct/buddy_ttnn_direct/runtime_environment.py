from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


def collect_ttnn_environment(ttnn_module: Any | None) -> dict[str, Any]:
    module_file = _string_attr(ttnn_module, "__file__")
    tt_metal_home = _env_path("TT_METAL_HOME") or _env_path("TT_METAL_ROOT")
    commit, commit_source = _tt_metal_commit(ttnn_module, module_file, tt_metal_home)
    return {
        "module_available": ttnn_module is not None,
        "version": _string_attr(ttnn_module, "__version__"),
        "module_file": module_file,
        "tt_metal_home": tt_metal_home,
        "tt_metal_git_commit": commit,
        "tt_metal_git_commit_source": commit_source,
    }


def _tt_metal_commit(
    ttnn_module: Any | None,
    module_file: str | None,
    tt_metal_home: str | None,
) -> tuple[str | None, str | None]:
    for attr in ("__tt_metal_commit__", "__git_commit__", "__commit__"):
        value = _string_attr(ttnn_module, attr)
        if value:
            return value, f"module.{attr}"

    env_commit = os.environ.get("TT_METAL_GIT_COMMIT")
    if env_commit:
        return env_commit, "env.TT_METAL_GIT_COMMIT"

    if tt_metal_home:
        commit = _git_commit(Path(tt_metal_home))
        if commit:
            return commit, "env.TT_METAL_HOME"

    if module_file:
        commit = _git_commit(Path(module_file).parent)
        if commit:
            return commit, "ttnn.__file__"

    return None, None


def _string_attr(obj: Any | None, name: str) -> str | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _env_path(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _git_commit(path: Path) -> str | None:
    return _git_commit_cached(str(path))


@lru_cache(maxsize=16)
def _git_commit_cached(path_text: str) -> str | None:
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None
