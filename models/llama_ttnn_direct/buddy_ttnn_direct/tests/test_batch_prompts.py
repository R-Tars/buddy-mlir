from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import build_parser
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.inputs import (
    build_decode_runtime_state,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.tokenizer import (
    PromptBatch,
    PromptTokenizationError,
    load_prompt_batch,
    tokenize_prompts_for_prefill,
)


class BatchPromptTest(unittest.TestCase):
    def test_loads_first_batch_from_official_prompt_object_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            content = json.dumps(
                [
                    {"prompt": "alpha"},
                    {"prompt": "beta"},
                    {"prompt": "unused"},
                ]
            ).encode()
            path.write_bytes(content)

            batch = load_prompt_batch(
                prompt=None,
                input_prompts=path,
                batch_size=2,
            )

            self.assertEqual(batch.prompts, ["alpha", "beta"])
            self.assertEqual(batch.source, "input_prompts_file")
            self.assertEqual(batch.input_prompt_count, 3)
            self.assertEqual(
                batch.input_prompts_sha256,
                hashlib.sha256(content).hexdigest(),
            )

    def test_instruct_tokenizes_each_user_with_official_chat_shape(self) -> None:
        calls: list[tuple[object, bool, bool]] = []
        module = _fake_chat_tokenizer_module(calls)
        batch = PromptBatch(
            prompts=["alpha", "beta"],
            source="input_prompts_file",
            input_prompts_path="/tmp/prompts.json",
            input_prompts_sha256="abc",
            input_prompt_count=2,
        )

        tokenization = tokenize_prompts_for_prefill(
            prompt_batch=batch,
            prefill_len=8,
            tokenizer_path="/tmp/tokenizer",
            tokenizer_module=module,
            vocab_size=256,
            instruct=True,
            padding_token_id=0,
        )

        self.assertEqual(
            calls,
            [
                ([{"role": "user", "content": "alpha"}], True, True),
                ([{"role": "user", "content": "beta"}], True, True),
            ],
        )
        self.assertEqual(tokenization.effective_token_count_by_user, [4, 5])
        self.assertEqual(tokenization.selected_token_id_by_user, [13, 23])
        self.assertEqual(tokenization.token_ids[0], [10, 11, 12, 13, 0, 0, 0, 0])
        self.assertEqual(tokenization.token_ids[1], [20, 21, 22, 24, 23, 0, 0, 0])
        report = tokenization.to_report()
        self.assertTrue(report["instruct"])
        self.assertEqual(report["chat_template"], "hf_apply_chat_template")
        self.assertEqual(report["input_prompts_sha256"], "abc")

    def test_batch_prompt_parity_rejects_silent_truncation(self) -> None:
        batch = PromptBatch(
            prompts=["alpha", "beta"],
            source="input_prompts_file",
            input_prompts_path=None,
            input_prompts_sha256=None,
            input_prompt_count=2,
        )

        with self.assertRaisesRegex(
            PromptTokenizationError,
            "increase --prefill-len to at least 5",
        ):
            tokenize_prompts_for_prefill(
                prompt_batch=batch,
                prefill_len=4,
                tokenizer_path="/tmp/tokenizer",
                tokenizer_module=_fake_chat_tokenizer_module([]),
                vocab_size=256,
                instruct=True,
                padding_token_id=0,
            )

    def test_decode_runtime_tracks_each_users_prompt_position(self) -> None:
        state = build_decode_runtime_state(
            batch_size=3,
            cache_len=32,
            page_block_size=8,
            prompt_token_count=[3, 5, 4],
        )

        self.assertIsNone(state.cache_position_value)
        self.assertEqual(state.cache_position_values, [2, 4, 3])
        self.assertEqual(state.cache_position, [2, 4, 3])
        self.assertEqual(
            state.to_report()["cache_position_values"],
            [2, 4, 3],
        )

    def test_generate_cli_accepts_official_prompt_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--program-dir",
                "/tmp/program",
                "--input-prompts",
                "/tmp/prompts.json",
                "--instruct",
                "--device",
                "p150a",
            ]
        )

        self.assertEqual(args.input_prompts, Path("/tmp/prompts.json"))
        self.assertTrue(args.instruct)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--program-dir",
                    "/tmp/program",
                    "--prompt",
                    "hello",
                    "--input-prompts",
                    "/tmp/prompts.json",
                ]
            )


def _fake_chat_tokenizer_module(calls: list[tuple[object, bool, bool]]) -> object:
    class FakeTokenizer:
        pad_token_id = 99

        def apply_chat_template(
            self,
            messages: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> list[int]:
            calls.append((messages, add_generation_prompt, tokenize))
            prompt = messages[0]["content"]  # type: ignore[index]
            return (
                [10, 11, 12, 13]
                if prompt == "alpha"
                else [20, 21, 22, 24, 23]
            )

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_path: str) -> FakeTokenizer:
            return FakeTokenizer()

    return types.SimpleNamespace(AutoTokenizer=AutoTokenizer)


if __name__ == "__main__":
    unittest.main()
