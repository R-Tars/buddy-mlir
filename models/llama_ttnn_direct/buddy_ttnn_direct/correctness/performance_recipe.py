from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Sequence

OFFICIAL_PERFORMANCE_TOP1_THRESHOLD = 0.895
OFFICIAL_PERFORMANCE_TOP5_THRESHOLD = 0.975
DEFAULT_GREEDY_AGREEMENT_THRESHOLD = 0.90


def load_official_performance_reference(
    path: str | Path,
    *,
    token_count: int,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not source.is_file():
        raise ValueError(f"official accuracy reference does not exist: {source}")
    torch = torch_module or importlib.import_module("torch")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    reference_tokens = _tensor_rows(payload.get("reference_tokens"))
    top5_tokens = _tensor_rows(payload.get("top5_tokens"))
    if len(reference_tokens) != 1 or len(reference_tokens[0]) < 2:
        raise ValueError("reference_tokens must have shape [1, sequence]")
    if not top5_tokens or any(len(row) != 5 for row in top5_tokens):
        raise ValueError("top5_tokens must have shape [sequence - 1, 5]")

    full_tokens = reference_tokens[0]
    split_point = len(full_tokens) // 2
    prompt_token_ids = full_tokens[:split_point]
    target_token_ids = full_tokens[split_point:]
    aligned_top5 = top5_tokens[split_point - 1 :]
    available = min(len(target_token_ids), len(aligned_top5))
    if token_count > available:
        raise ValueError(
            f"token_count {token_count} exceeds reference capacity {available}"
        )
    return {
        "schema_version": 1,
        "kind": "official_tt_transformers_performance_reference",
        "path": str(source),
        "sha256": _sha256(source),
        "full_sequence_token_count": len(full_tokens),
        "split_point": split_point,
        "prompt_token_ids": prompt_token_ids,
        "target_token_ids": target_token_ids[:token_count],
        "top5_token_ids": aligned_top5[:token_count],
        "token_count": int(token_count),
    }


def token_accuracy(
    predicted_token_ids: Sequence[int],
    top5_token_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    predicted = [int(token_id) for token_id in predicted_token_ids]
    top5 = [[int(token_id) for token_id in row] for row in top5_token_ids]
    if len(predicted) != len(top5):
        raise ValueError(
            "predicted and reference token counts differ: "
            f"{len(predicted)} != {len(top5)}"
        )
    if not predicted:
        raise ValueError("token accuracy requires at least one token")
    top1_matches = [
        prediction == row[0] for prediction, row in zip(predicted, top5, strict=True)
    ]
    top5_matches = [
        prediction in row for prediction, row in zip(predicted, top5, strict=True)
    ]
    mismatch_positions = [
        index for index, matched in enumerate(top1_matches) if not matched
    ]
    return {
        "token_count": len(predicted),
        "top1_matches": sum(top1_matches),
        "top5_matches": sum(top5_matches),
        "top1_accuracy": sum(top1_matches) / len(predicted),
        "top5_accuracy": sum(top5_matches) / len(predicted),
        "first_top1_mismatch_positions": mismatch_positions[:20],
    }


def greedy_agreement(
    left_token_ids: Sequence[int],
    right_token_ids: Sequence[int],
) -> dict[str, Any]:
    left = [int(token_id) for token_id in left_token_ids]
    right = [int(token_id) for token_id in right_token_ids]
    if len(left) != len(right):
        raise ValueError(f"greedy token counts differ: {len(left)} != {len(right)}")
    if not left:
        raise ValueError("greedy agreement requires at least one token")
    matches = [
        left_id == right_id for left_id, right_id in zip(left, right, strict=True)
    ]
    mismatch_positions = [index for index, matched in enumerate(matches) if not matched]
    return {
        "token_count": len(left),
        "matches": sum(matches),
        "agreement": sum(matches) / len(left),
        "first_mismatch_positions": mismatch_positions[:20],
    }


def _tensor_rows(value: Any) -> list[list[int]]:
    if value is None:
        raise ValueError("accuracy reference is missing a required tensor")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise ValueError("accuracy reference tensor must convert to a list")
    if value and not isinstance(value[0], list):
        value = [value]
    return [[int(item) for item in row] for row in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
