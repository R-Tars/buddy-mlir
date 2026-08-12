from __future__ import annotations

import hashlib
import os
import resource
import subprocess
from collections.abc import Sequence
from pathlib import Path

def run_logged_command(
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: float | None,
    address_space_limit_bytes: int | None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def set_limits() -> None:
        if address_space_limit_bytes is not None:
            limit = (address_space_limit_bytes, address_space_limit_bytes)
            resource.setrlimit(resource.RLIMIT_AS, limit)

    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(command), cwd=cwd, env=environment, stdout=log,
            stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False,
            preexec_fn=set_limits,
        )
    return int(result.returncode)

def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=False,
        capture_output=True, text=text,
    )
    if result.returncode:
        message = result.stderr if text else result.stderr.decode(errors="replace")
        raise ValueError(f"cannot read git metadata for {root}: {message.strip()}")
    return result.stdout

def git_value(root: Path, *args: str) -> str:
    output = _git(root, *args)
    return output.decode().strip() if isinstance(output, bytes) else output.strip()

def git_diff_sha256(root: Path) -> str:
    return hashlib.sha256(bytes(_git(root, "diff", "--binary", text=False))).hexdigest()

def git_head(root: Path) -> str | None:
    try:
        return git_value(root, "rev-parse", "HEAD")
    except ValueError:
        return None

def absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value))))

def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
