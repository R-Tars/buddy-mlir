from __future__ import annotations

import hashlib
import importlib
import math
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
class DecodeRuntimeState:
    batch_size: int
    cache_len: int
    page_block_size: int
    page_count: int
    max_num_blocks: int
    cache_position_value: int
    page_table: list[list[int]]
    cache_position: list[int]

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "decode_runtime_state",
            "batch_size": self.batch_size,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "page_count": self.page_count,
            "max_num_blocks": self.max_num_blocks,
            "cache_position_value": self.cache_position_value,
            "page_table_shape": [self.batch_size, self.page_count],
            "cache_position_shape": [self.batch_size],
        }


@dataclass(frozen=True)
class DecodeRotaryRuntimeState:
    layer_count: int
    head_dim: int
    cache_position_value: int
    matrix_shape: list[int]
    tensor_count: int

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "rotary_runtime_state",
            "layer_count": self.layer_count,
            "head_dim": self.head_dim,
            "cache_position_value": self.cache_position_value,
            "matrix_shape": list(self.matrix_shape),
            "tensors_per_layer": 3,
            "tensor_count": self.tensor_count,
            "tensor_roles": [
                "cos_matrix",
                "sin_matrix",
                "transformation_matrix",
            ],
        }


@dataclass(frozen=True)
class DecodeKVCacheRuntimeState:
    layer_count: int
    batch_size: int
    cache_len: int
    page_block_size: int
    page_count: int
    max_num_blocks: int
    num_kv_heads: int
    head_dim: int
    physical_shape: list[int]
    logical_shape: list[int]
    tensor_count: int

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "kv_cache_runtime_state",
            "layer_count": self.layer_count,
            "batch_size": self.batch_size,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "page_count": self.page_count,
            "max_num_blocks": self.max_num_blocks,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "physical_shape": list(self.physical_shape),
            "logical_shape": list(self.logical_shape),
            "tensors_per_layer": 2,
            "tensor_count": self.tensor_count,
            "tensor_roles": ["key_cache", "value_cache"],
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


def build_decode_runtime_state(
    *,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    prompt_token_count: int,
) -> DecodeRuntimeState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cache_len <= 0:
        raise ValueError("cache_len must be positive")
    if page_block_size <= 0:
        raise ValueError("page_block_size must be positive")
    if prompt_token_count <= 0:
        raise ValueError("prompt_token_count must be positive")

    page_count = max(1, math.ceil(cache_len / page_block_size))
    max_num_blocks = batch_size * page_count
    cache_position_value = min(max(prompt_token_count - 1, 0), cache_len - 1)
    page_table = [
        [batch_id * page_count + page_id for page_id in range(page_count)]
        for batch_id in range(batch_size)
    ]
    return DecodeRuntimeState(
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        page_count=page_count,
        max_num_blocks=max_num_blocks,
        cache_position_value=cache_position_value,
        page_table=page_table,
        cache_position=[cache_position_value for _ in range(batch_size)],
    )


def build_decode_rotary_runtime_state(
    *,
    layer_count: int,
    head_dim: int,
    cache_position_value: int,
) -> DecodeRotaryRuntimeState:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if cache_position_value < 0:
        raise ValueError("cache_position_value must be non-negative")

    return DecodeRotaryRuntimeState(
        layer_count=layer_count,
        head_dim=head_dim,
        cache_position_value=cache_position_value,
        matrix_shape=[1, 1, head_dim, head_dim],
        tensor_count=3 * layer_count,
    )


def build_decode_kv_cache_runtime_state(
    *,
    layer_count: int,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> DecodeKVCacheRuntimeState:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cache_len <= 0:
        raise ValueError("cache_len must be positive")
    if page_block_size <= 0:
        raise ValueError("page_block_size must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")

    page_count = max(1, math.ceil(cache_len / page_block_size))
    max_num_blocks = batch_size * page_count
    physical_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    logical_shape = [batch_size, cache_len, num_kv_heads, head_dim]
    return DecodeKVCacheRuntimeState(
        layer_count=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        page_count=page_count,
        max_num_blocks=max_num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        physical_shape=physical_shape,
        logical_shape=logical_shape,
        tensor_count=2 * layer_count,
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
