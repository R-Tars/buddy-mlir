from __future__ import annotations

import json
from pathlib import Path


OFFICIAL_PAGED_ATTENTION_DECODE_OPS = [
    "linear.qkv_packed",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "linear.o_proj",
]


def official_paged_attention_decode_op_sequence() -> list[str]:
    return list(OFFICIAL_PAGED_ATTENTION_DECODE_OPS)


def dump_attention_decode_op_sequence(out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(official_paged_attention_decode_op_sequence(), indent=2)
        + "\n"
    )
