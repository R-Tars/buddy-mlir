from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import LlamaModelGraph, graph_from_dict, graph_to_dict
from .validate import validate_llama_graph


def graph_json_dict(graph: LlamaModelGraph) -> dict[str, Any]:
    validate_llama_graph(graph)
    return graph_to_dict(graph)


def dump_graph_json(graph: LlamaModelGraph, out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = graph_json_dict(graph)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_graph_json(path: str | Path) -> LlamaModelGraph:
    data = json.loads(Path(path).read_text())
    graph = graph_from_dict(data)
    validate_llama_graph(graph)
    return graph
