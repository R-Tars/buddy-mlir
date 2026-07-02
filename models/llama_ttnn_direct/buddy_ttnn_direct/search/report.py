from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_search_report(report: dict[str, Any], out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
