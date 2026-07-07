from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PromptTokenizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptTokenization:
    prompt_sha256: str
    prompt_char_count: int
    tokenizer_path: str
    token_count: int
    selected_token_id: int
    batch_size: int
    token_ids: list[list[int]]

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "tokenized",
            "source": "prompt_tokenizer",
            "prompt_sha256": self.prompt_sha256,
            "prompt_char_count": self.prompt_char_count,
            "tokenizer_path": self.tokenizer_path,
            "token_count": self.token_count,
            "selected_token_id": self.selected_token_id,
            "batch_size": self.batch_size,
            "token_input_shape": [self.batch_size, 1],
        }


def tokenize_prompt_for_decode(
    *,
    prompt: str,
    batch_size: int,
    tokenizer_path: str | Path,
    vocab_size: int | None = None,
    tokenizer_module: Any | None = None,
) -> PromptTokenization:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prompt == "":
        raise ValueError("prompt must be non-empty")

    tokenizer_path = Path(tokenizer_path)
    tokenizer = _load_tokenizer(tokenizer_path, tokenizer_module)
    token_ids = _encode_prompt(tokenizer, prompt)
    if not token_ids:
        raise PromptTokenizationError("tokenizer returned no prompt token ids")

    selected = int(token_ids[-1])
    if selected < 0:
        raise PromptTokenizationError(
            f"tokenizer returned negative token id {selected}"
        )
    if vocab_size is not None and selected >= int(vocab_size):
        raise PromptTokenizationError(
            f"token id {selected} is outside vocab size {vocab_size}"
        )

    return PromptTokenization(
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_char_count=len(prompt),
        tokenizer_path=str(tokenizer_path),
        token_count=len(token_ids),
        selected_token_id=selected,
        batch_size=batch_size,
        token_ids=[[selected] for _ in range(batch_size)],
    )


def _load_tokenizer(tokenizer_path: Path, tokenizer_module: Any | None) -> Any:
    module = tokenizer_module
    if module is None:
        try:
            module = importlib.import_module("transformers")
        except ImportError as err:
            raise PromptTokenizationError(
                "prompt tokenization requires transformers.AutoTokenizer; "
                "install transformers or pass pre-tokenized inputs through "
                "the lower-level smoke API"
            ) from err
    auto_tokenizer = getattr(module, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise PromptTokenizationError(
            "tokenizer module must provide AutoTokenizer"
        )
    from_pretrained = getattr(auto_tokenizer, "from_pretrained", None)
    if not callable(from_pretrained):
        raise PromptTokenizationError(
            "AutoTokenizer must provide from_pretrained"
        )
    return from_pretrained(str(tokenizer_path))


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    if callable(tokenizer):
        encoded = tokenizer(prompt, add_special_tokens=True)
        ids = _extract_input_ids(encoded)
        if ids:
            return ids

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        encoded = encode(prompt, add_special_tokens=True)
        ids = _extract_input_ids(encoded)
        if ids:
            return ids

    raise PromptTokenizationError(
        "tokenizer did not return input_ids for the prompt"
    )


def _extract_input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if encoded is None:
        return []
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if not isinstance(encoded, list):
        return []
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids: list[int] = []
    for item in encoded:
        if hasattr(item, "item"):
            item = item.item()
        ids.append(int(item))
    return ids
