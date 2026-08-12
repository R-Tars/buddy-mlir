from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.performance_correctness import (
    run_performance_correctness,
)

MODULE = "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.performance_correctness"

def _reference(token_count: int) -> dict[str, object]:
    targets = list(range(100, 100 + token_count))
    return {
        "schema_version": 1,
        "kind": "official_tt_transformers_performance_reference",
        "path": "/tmp/reference.refpt",
        "sha256": "a" * 64,
        "full_sequence_token_count": token_count * 2,
        "split_point": token_count,
        "prompt_token_ids": list(range(token_count)),
        "target_token_ids": targets,
        "top5_token_ids": [
            [target + offset for offset in (0, 1000, 2000, 3000, 4000)] for target in targets
        ],
        "token_count": token_count,
    }

def _run_case(
    reference: dict[str, object],
    *,
    roundtrip_token_ids: list[int],
    generated_rows: list[list[int]],
    prefill_len: int,
    cache_len: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = list(reference["target_token_ids"])
    buddy_calls: list[dict[str, object]] = []

    def official_runner(
        _command: object, _cwd: object, _environment: object, log_path: Path, *_unused: object
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                f"BUDDY_ACCURACY_SAMPLE token_iteration={index} "
                f"predicted_token={token}"
                for index, token in enumerate(predictions)
            )
            + "\n"
        )
        return 0

    def buddy_runner(**kwargs: object) -> dict[str, object]:
        buddy_calls.append(kwargs)
        return {"passed": True, "status": "passed", "generated_token_ids": generated_rows}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        program, official, model = root / "program", root / "official", root / "model"
        for path in (program, official, model):
            path.mkdir()
        python = root / "python"
        python.write_text("")
        with (
            patch(f"{MODULE}.load_official_performance_reference", return_value=reference),
            patch(
                f"{MODULE}._buddy_prompt_from_reference",
                return_value={"text": "fixed prompt", "roundtrip_token_ids": roundtrip_token_ids, "adaptation": "none"},
            ),
            patch(f"{MODULE}._git_value", return_value="commit"),
        ):
            report = run_performance_correctness(
                out=root / "performance_correctness.json",
                buddy_program=program,
                official_tt_metal_root=official,
                model_path=model,
                tokenizer_path=model,
                official_python=python,
                token_count=int(reference["token_count"]),
                layers=1,
                batch_size=len(generated_rows),
                prefill_len=prefill_len,
                cache_len=cache_len,
                official_runner=official_runner,
                buddy_runner=buddy_runner,
            )
    return report, buddy_calls

class PerformanceCorrectnessTest(unittest.TestCase):
    def test_runs_official_then_buddy_and_applies_contract(self) -> None:
        reference = _reference(3)
        predictions = list(reference["target_token_ids"])
        report, buddy_calls = _run_case(
            reference,
            roundtrip_token_ids=[0, 1, 2],
            generated_rows=[predictions, predictions],
            prefill_len=3,
            cache_len=8,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["official_run"]["sample_count"], 3)
        self.assertEqual(report["metrics"]["buddy_aggregate"]["top1_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["official_buddy_min_user_greedy_agreement"], 1.0)
        self.assertEqual(buddy_calls[0]["teacher_forcing_token_ids_by_step"], [[100, 100], [101, 101]])
        self.assertEqual(buddy_calls[0]["execution_mode"], "eager")

    def test_replays_prompt_tail_beyond_static_prefill_limit(self) -> None:
        reference = _reference(3)
        reference["prompt_token_ids"] = list(range(258))
        reference["full_sequence_token_count"] = 261
        row = [900, 901, 100, 101, 102]
        report, buddy_calls = _run_case(
            reference,
            roundtrip_token_ids=list(range(256)),
            generated_rows=[row, row],
            prefill_len=258,
            cache_len=264,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(buddy_calls[0]["prefill_len"], 256)
        self.assertEqual(buddy_calls[0]["max_new_tokens"], 5)
        self.assertEqual(
            buddy_calls[0]["teacher_forcing_token_ids_by_step"],
            [[256, 256], [257, 257], [100, 100], [101, 101]],
        )
        self.assertEqual(report["fixed_corpus"]["buddy_prompt_replay_decode_token_count"], 2)

if __name__ == "__main__":
    unittest.main()
