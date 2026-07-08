from __future__ import annotations

import os
import shutil
import stat
import subprocess
from functools import lru_cache
from glob import glob
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


def collect_tenstorrent_device_environment() -> dict[str, Any]:
    entries = _tenstorrent_device_entries()
    device_nodes = [
        path for path in entries if _is_character_device(Path(path))
    ]
    tt_smi_path = shutil.which("tt-smi")
    return {
        "device_available": bool(device_nodes),
        "device_node_count": len(device_nodes),
        "device_nodes": device_nodes,
        "filesystem_entries": entries,
        "driver_loaded": _kernel_module_loaded("tenstorrent"),
        "tt_smi_path": tt_smi_path,
        "tt_smi": _probe_command([tt_smi_path]) if tt_smi_path else None,
    }


def _tenstorrent_device_entries() -> list[str]:
    paths: set[str] = set()
    for pattern in (
        "/dev/tenstorrent",
        "/dev/tenstorrent/*",
        "/dev/tenstorrent*",
    ):
        for path in glob(pattern):
            paths.add(path)
    return sorted(paths)


def _is_character_device(path: Path) -> bool:
    try:
        return stat.S_ISCHR(path.stat().st_mode)
    except OSError:
        return False


def _kernel_module_loaded(name: str) -> bool:
    try:
        lines = Path("/proc/modules").read_text().splitlines()
    except OSError:
        return False
    prefix = f"{name} "
    return any(line.startswith(prefix) for line in lines)


def _probe_command(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "timeout_seconds": timeout,
            "stdout": _trim_probe_output(exc.stdout),
            "stderr": _trim_probe_output(exc.stderr),
        }
    except OSError as exc:
        return {
            "status": "error",
            "command": command,
            "error": str(exc),
        }
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "command": command,
        "returncode": result.returncode,
        "stdout": _trim_probe_output(result.stdout),
        "stderr": _trim_probe_output(result.stderr),
    }


def _trim_probe_output(value: str | bytes | None, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


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
