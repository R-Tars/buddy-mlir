from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
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


def collect_tenstorrent_process_environment() -> dict[str, Any]:
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["ps", "-eo", "user,pid,ppid,stat,etime,cmd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "conflict_count": 0,
            "conflicts": [],
        }
    if result.returncode != 0:
        return {
            "status": "error",
            "returncode": result.returncode,
            "stderr": _trim_probe_output(result.stderr),
            "conflict_count": 0,
            "conflicts": [],
        }

    conflicts: list[dict[str, Any]] = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.strip().split(None, 5)
        if len(fields) < 6:
            continue
        user, pid_text, ppid_text, stat_text, elapsed_text, command = fields
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
        except ValueError:
            continue
        if pid == current_pid or ppid == current_pid:
            continue
        kind = _tenstorrent_process_conflict_kind(command)
        if kind is None:
            continue
        conflicts.append(
            {
                "kind": kind,
                "user": user,
                "pid": pid,
                "ppid": ppid,
                "stat": stat_text,
                "elapsed": elapsed_text,
                "command": command,
            }
        )

    return {
        "status": "busy" if conflicts else "idle",
        "conflict_count": len(conflicts),
        "reset_in_progress": any(
            conflict["kind"] == "tt_smi_reset"
            for conflict in conflicts
        ),
        "conflicts": conflicts[:16],
    }


def _tenstorrent_process_conflict_kind(command: str) -> str | None:
    normalized = " ".join(command.split())
    if "tt-smi -r" in normalized or "tt_smi -r" in normalized:
        return "tt_smi_reset"
    if "examples.tenstorrent" in normalized:
        return "tenstorrent_example"
    if "phase6_torch_add" in normalized:
        return "tenstorrent_example"
    if "trex" in normalized and "tenstorrent" in normalized:
        return "tenstorrent_example"
    return None


def collect_ttnn_runtime_health(
    *,
    device_id: int = 0,
    timeout: float = 45.0,
) -> dict[str, Any]:
    script = f"""
import torch
import ttnn

device = ttnn.open_device(device_id={int(device_id)})
try:
    x = torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16)
    y = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    z = ttnn.to_torch(y)
    print("ttnn_runtime_health=pass")
    print("shape=" + str(list(z.shape)))
    print("sum=" + str(float(z.sum())))
finally:
    ttnn.close_device(device)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "device_id": device_id,
            "timeout_seconds": timeout,
            "stdout": _trim_probe_output(exc.stdout),
            "stderr": _trim_probe_output(exc.stderr),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "error",
            "device_id": device_id,
            "error": str(exc),
        }
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "device_id": device_id,
        "returncode": result.returncode,
        "stdout": _trim_probe_output(result.stdout),
        "stderr": _trim_probe_output(result.stderr),
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
