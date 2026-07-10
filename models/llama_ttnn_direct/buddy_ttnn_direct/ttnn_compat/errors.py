from __future__ import annotations

from typing import Iterable, Sequence


class UnsupportedTTNNOp(RuntimeError):
    """Raised when the installed TTNN module does not expose a required op."""

    def __init__(
        self,
        op_name: str,
        searched_paths: Iterable[Sequence[str]] = (),
    ) -> None:
        searched = ", ".join(
            "ttnn." + ".".join(path) for path in searched_paths
        )
        message = f"TTNN Direct requires TTNN op wrapper implementation: {op_name}"
        if searched:
            message += f". Searched: {searched}"
        super().__init__(message)
        self.op_name = op_name
