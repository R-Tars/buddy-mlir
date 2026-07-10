from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.templates.attention_decode import (
    dump_attention_decode_op_sequence,
    official_paged_attention_decode_op_sequence,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.diff import (
    expand_layer_templates,
)


EXPECTED_ATTENTION_OPS = [
    "linear.qkv_packed",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "linear.o_proj",
]


class AttentionDecodeTemplateTest(unittest.TestCase):
    def test_official_attention_decode_op_sequence(self) -> None:
        self.assertEqual(
            official_paged_attention_decode_op_sequence(),
            EXPECTED_ATTENTION_OPS,
        )

    def test_dump_attention_decode_op_sequence_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "attention_ops.json"

            dump_attention_decode_op_sequence(out)

            self.assertEqual(json.loads(out.read_text()), EXPECTED_ATTENTION_OPS)

    def test_plan_expansion_uses_attention_template_ops(self) -> None:
        self.assertEqual(
            expand_layer_templates(["official_paged_attention_decode"]),
            EXPECTED_ATTENTION_OPS,
        )


if __name__ == "__main__":
    unittest.main()
