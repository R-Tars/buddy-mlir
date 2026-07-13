from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PromptTokenizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptBatch:
    prompts: list[str]
    source: str
    input_prompts_path: str | None = None
    input_prompts_sha256: str | None = None
    input_prompt_count: int | None = None


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
    prompt_sha256_by_user: list[str]
    prompt_char_count_by_user: list[int]
    original_token_count_by_user: list[int]
    effective_token_count_by_user: list[int]
    selected_token_id_by_user: list[int]
    instruct: bool = False
    prompt_source: str = "inline_prompt"
    input_prompts_path: str | None = None
    input_prompts_sha256: str | None = None
    input_prompt_count: int | None = None

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
            "prompt_sha256_by_user": self.prompt_sha256_by_user,
            "prompt_char_count_by_user": self.prompt_char_count_by_user,
            "original_token_count_by_user": self.original_token_count_by_user,
            "effective_token_count_by_user": self.effective_token_count_by_user,
            "selected_token_id_by_user": self.selected_token_id_by_user,
            "instruct": self.instruct,
            "chat_template": "hf_apply_chat_template" if self.instruct else None,
            "prompt_source": self.prompt_source,
            "input_prompts_path": self.input_prompts_path,
            "input_prompts_sha256": self.input_prompts_sha256,
            "input_prompt_count": self.input_prompt_count,
        }


def load_prompt_batch(
    *,
    prompt: str | None,
    input_prompts: str | Path | None,
    batch_size: int,
) -> PromptBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prompt is not None and input_prompts is not None:
        raise ValueError("prompt and input_prompts are mutually exclusive")
    if input_prompts is None:
        if prompt is None or prompt == "":
            raise ValueError("prompt or input_prompts is required")
        return PromptBatch(
            prompts=[prompt for _ in range(batch_size)],
            source="inline_prompt",
            input_prompt_count=1,
        )

    path = Path(input_prompts)
    try:
        content = path.read_bytes()
    except OSError as err:
        raise PromptTokenizationError(
            f"unable to read input prompts file {path}: {err}"
        ) from err
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise PromptTokenizationError(
            f"input prompts file must contain valid UTF-8 JSON: {path}"
        ) from err
    if not isinstance(payload, list):
        raise PromptTokenizationError(
            "input prompts JSON must be a list of strings or prompt objects"
        )

    prompts: list[str] = []
    for index, item in enumerate(payload):
        value = item.get("prompt") if isinstance(item, dict) else item
        if not isinstance(value, str) or value == "":
            raise PromptTokenizationError(
                f"input prompt {index} must be a non-empty string or an "
                "object with a non-empty 'prompt' string"
            )
        prompts.append(value)
    if len(prompts) < batch_size:
        raise PromptTokenizationError(
            f"input prompts file contains {len(prompts)} prompts, but "
            f"batch_size={batch_size} requires at least {batch_size}"
        )
    return PromptBatch(
        prompts=prompts[:batch_size],
        source="input_prompts_file",
        input_prompts_path=str(path.resolve()),
        input_prompts_sha256=hashlib.sha256(content).hexdigest(),
        input_prompt_count=len(prompts),
    )


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
    batch = PromptBatch(
        prompts=[prompt for _ in range(batch_size)],
        source="inline_prompt",
        input_prompt_count=1,
    )
    return tokenize_prompts_for_prefill(
        prompt_batch=batch,
        prefill_len=prefill_len,
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        tokenizer_module=tokenizer_module,
        instruct=False,
        reject_truncation=False,
    )


def tokenize_prompts_for_prefill(
    *,
    prompt_batch: PromptBatch,
    prefill_len: int,
    tokenizer_path: str | Path,
    vocab_size: int | None = None,
    tokenizer_module: Any | None = None,
    instruct: bool = False,
    reject_truncation: bool = True,
    padding_token_id: int | None = None,
) -> PrefillPromptTokenization:
    prompts = list(prompt_batch.prompts)
    if not prompts:
        raise ValueError("prompt batch must be non-empty")
    if prefill_len <= 0:
        raise ValueError("prefill_len must be positive")
    if any(prompt == "" for prompt in prompts):
        raise ValueError("prompts must be non-empty")

    tokenizer_path = Path(tokenizer_path)
    tokenizer = _load_tokenizer(tokenizer_path, tokenizer_module)
    raw_by_user = [
        _encode_prompt(tokenizer, prompt, instruct=instruct)
        for prompt in prompts
    ]
    if any(not token_ids for token_ids in raw_by_user):
        raise PromptTokenizationError("tokenizer returned no prompt token ids")

    for user_id, token_ids in enumerate(raw_by_user):
        for token_id in token_ids:
            if int(token_id) < 0:
                raise PromptTokenizationError(
                    f"tokenizer returned negative token id {token_id} "
                    f"for user {user_id}"
                )
            if vocab_size is not None and int(token_id) >= int(vocab_size):
                raise PromptTokenizationError(
                    f"token id {token_id} for user {user_id} is outside "
                    f"vocab size {vocab_size}"
                )

    oversized = [
        (user_id, len(token_ids))
        for user_id, token_ids in enumerate(raw_by_user)
        if len(token_ids) > prefill_len
    ]
    if oversized and reject_truncation:
        user_id, token_count = max(oversized, key=lambda item: item[1])
        raise PromptTokenizationError(
            "prompt tokenization would truncate official comparison input: "
            f"user {user_id} has {token_count} tokens but prefill_len="
            f"{prefill_len}; increase --prefill-len to at least "
            f"{max(count for _, count in oversized)}"
        )

    effective_by_user: list[list[int]] = []
    truncations: list[str] = []
    for raw_token_ids in raw_by_user:
        if len(raw_token_ids) > prefill_len:
            effective = [int(token_id) for token_id in raw_token_ids[-prefill_len:]]
            truncations.append("left_truncated_to_prefill_len")
        else:
            effective = [int(token_id) for token_id in raw_token_ids]
            truncations.append("none")
        effective_by_user.append(effective)

    pad_token_id = (
        _tokenizer_pad_token_id(tokenizer)
        if padding_token_id is None
        else int(padding_token_id)
    )
    if pad_token_id < 0:
        raise ValueError("padding_token_id must be non-negative")
    if vocab_size is not None and pad_token_id >= int(vocab_size):
        raise ValueError(
            f"padding_token_id {pad_token_id} is outside vocab size {vocab_size}"
        )
    padded_by_user = [
        effective + [pad_token_id for _ in range(prefill_len - len(effective))]
        for effective in effective_by_user
    ]
    prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
    ]
    aggregate_hash = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    original_counts = [len(token_ids) for token_ids in raw_by_user]
    effective_counts = [len(token_ids) for token_ids in effective_by_user]
    selected_ids = [int(token_ids[-1]) for token_ids in effective_by_user]
    truncation = (
        "left_truncated_to_prefill_len"
        if any(value != "none" for value in truncations)
        else "none"
    )
    return PrefillPromptTokenization(
        prompt_sha256=(prompt_hashes[0] if len(set(prompts)) == 1 else aggregate_hash),
        prompt_char_count=(
            len(prompts[0])
            if len(set(prompts)) == 1
            else sum(len(prompt) for prompt in prompts)
        ),
        tokenizer_path=str(tokenizer_path),
        original_token_count=max(original_counts),
        effective_token_count=max(effective_counts),
        prefill_len=prefill_len,
        selected_token_id=selected_ids[0],
        batch_size=len(prompts),
        token_ids=padded_by_user,
        padding_token_id=pad_token_id,
        truncation=truncation,
        prompt_sha256_by_user=prompt_hashes,
        prompt_char_count_by_user=[len(prompt) for prompt in prompts],
        original_token_count_by_user=original_counts,
        effective_token_count_by_user=effective_counts,
        selected_token_id_by_user=selected_ids,
        instruct=bool(instruct),
        prompt_source=prompt_batch.source,
        input_prompts_path=prompt_batch.input_prompts_path,
        input_prompts_sha256=prompt_batch.input_prompts_sha256,
        input_prompt_count=prompt_batch.input_prompt_count,
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


def _encode_prompt(tokenizer: Any, prompt: str, *, instruct: bool = False) -> list[int]:
    if instruct:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise PromptTokenizationError(
                "--instruct requires a tokenizer with apply_chat_template"
            )
        encoded = apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        ids = _extract_input_ids(encoded)
        if ids:
            return ids
        raise PromptTokenizationError(
            "tokenizer chat template did not return input_ids for the prompt"
        )

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
