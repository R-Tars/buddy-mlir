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


@dataclass(frozen=True)
class PrefillPromptTokenization:
    prompt_sha256: str
    prompt_char_count: int
    tokenizer_path: str
    original_token_count: int
    effective_token_count: int
    prefill_len: int
    selected_token_id: int
    batch_size: int
    token_ids: list[list[int]]
    padding_token_id: int
    truncation: str

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "tokenized",
            "source": "prompt_tokenizer_prefill",
            "prompt_sha256": self.prompt_sha256,
            "prompt_char_count": self.prompt_char_count,
            "tokenizer_path": self.tokenizer_path,
            "original_token_count": self.original_token_count,
            "effective_token_count": self.effective_token_count,
            "prefill_len": self.prefill_len,
            "selected_token_id": self.selected_token_id,
            "batch_size": self.batch_size,
            "token_input_shape": [self.batch_size, self.prefill_len],
            "padding_token_id": self.padding_token_id,
            "truncation": self.truncation,
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


def tokenize_prompt_for_prefill(
    *,
    prompt: str,
    batch_size: int,
    prefill_len: int,
    tokenizer_path: str | Path,
    vocab_size: int | None = None,
    tokenizer_module: Any | None = None,
) -> PrefillPromptTokenization:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")
    if prompt == "":
        raise ValueError("prompt must be non-empty")

    tokenizer_path = Path(tokenizer_path)
    tokenizer = _load_tokenizer(tokenizer_path, tokenizer_module)
    raw_token_ids = _encode_prompt(tokenizer, prompt)
    if not raw_token_ids:
        raise PromptTokenizationError("tokenizer returned no prompt token ids")

    for token_id in raw_token_ids:
        if int(token_id) < 0:
            raise PromptTokenizationError(
                f"tokenizer returned negative token id {token_id}"
            )
        if vocab_size is not None and int(token_id) >= int(vocab_size):
            raise PromptTokenizationError(
                f"token id {token_id} is outside vocab size {vocab_size}"
            )

    if len(raw_token_ids) > prefill_len:
        effective = [int(token_id) for token_id in raw_token_ids[-prefill_len:]]
        truncation = "left_truncated_to_prefill_len"
    else:
        effective = [int(token_id) for token_id in raw_token_ids]
        truncation = "none"

    pad_token_id = _tokenizer_pad_token_id(tokenizer)
    padded = effective + [pad_token_id for _ in range(prefill_len - len(effective))]
    selected = int(effective[-1])
    return PrefillPromptTokenization(
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_char_count=len(prompt),
        tokenizer_path=str(tokenizer_path),
        original_token_count=len(raw_token_ids),
        effective_token_count=len(effective),
        prefill_len=prefill_len,
        selected_token_id=selected,
        batch_size=batch_size,
        token_ids=[list(padded) for _ in range(batch_size)],
        padding_token_id=pad_token_id,
        truncation=truncation,
    )


def detokenize_generated_token_ids(
    *,
    token_ids_by_user: list[list[int]],
    tokenizer_path: str | Path,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    if not token_ids_by_user:
        return {
            "status": "empty",
            "source": "no_generated_tokens",
            "generated_text_by_user": [],
            "generated_text": "",
        }

    if any(token_id < 0 for row in token_ids_by_user for token_id in row):
        text_by_user = [_fallback_decode(row) for row in token_ids_by_user]
        return {
            "status": "placeholder",
            "source": "unmaterialized_token_placeholder",
            "generated_text_by_user": text_by_user,
            "generated_text": text_by_user[0] if text_by_user else "",
        }

    try:
        tokenizer = _load_tokenizer(Path(tokenizer_path), tokenizer_module)
        text_by_user = _decode_with_tokenizer(tokenizer, token_ids_by_user)
    except Exception as err:
        text_by_user = [_fallback_decode(row) for row in token_ids_by_user]
        return {
            "status": "fallback",
            "source": "token_id_fallback",
            "fallback_reason": f"{type(err).__name__}: {err}",
            "generated_text_by_user": text_by_user,
            "generated_text": text_by_user[0] if text_by_user else "",
        }

    return {
        "status": "decoded",
        "source": "tokenizer_decode",
        "generated_text_by_user": text_by_user,
        "generated_text": text_by_user[0] if text_by_user else "",
    }


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


def _tokenizer_pad_token_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


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


def _decode_with_tokenizer(
    tokenizer: Any,
    token_ids_by_user: list[list[int]],
) -> list[str]:
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        try:
            decoded = batch_decode(token_ids_by_user, skip_special_tokens=True)
        except TypeError:
            decoded = batch_decode(token_ids_by_user)
        if isinstance(decoded, list):
            return [str(text) for text in decoded]

    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        texts = []
        for row in token_ids_by_user:
            try:
                texts.append(str(decode(row, skip_special_tokens=True)))
            except TypeError:
                texts.append(str(decode(row)))
        return texts

    raise PromptTokenizationError("tokenizer does not provide decode")


def _fallback_decode(token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return " ".join(
        "<unmaterialized-token>" if token_id < 0 else f"<tok:{token_id}>"
        for token_id in token_ids
    )
