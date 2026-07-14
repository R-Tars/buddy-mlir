from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.performance_correctness import (
    run_performance_correctness,
)


def _reference(token_count: int) -> dict[str, object]:
    targets = [100 + index for index in range(token_count)]
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
            [target, target + 1000, target + 2000, target + 3000, target + 4000]
            for target in targets
        ],
        "token_count": token_count,
    }


class PerformanceCorrectnessTest(unittest.TestCase):
    def test_runs_official_then_buddy_and_applies_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = root / "program"
            official = root / "official"
            model = root / "model"
            python = root / "python"
            for path in (program, official, model):
                path.mkdir()
            python.write_text("")
            output = root / "performance_correctness.json"
            predictions = [100, 101, 102]
            buddy_calls: list[dict[str, object]] = []

            def official_runner(
                _command: object,
                _cwd: object,
                _environment: object,
                log_path: Path,
                _timeout: object,
                _limit: object,
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
                return {
                    "passed": True,
                    "status": "passed",
                    "generated_token_ids": [predictions, predictions],
                }

            with (
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness.load_official_performance_reference",
                    return_value=_reference(3),
                ),
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness._buddy_prompt_from_reference",
                    return_value={
                        "text": "fixed prompt",
                        "roundtrip_token_ids": [0, 1, 2],
                        "adaptation": "none",
                    },
                ),
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness._git_value",
                    return_value="commit",
                ),
            ):
                report = run_performance_correctness(
                    out=output,
                    buddy_program=program,
                    official_tt_metal_root=official,
                    model_path=model,
                    tokenizer_path=model,
                    official_python=python,
                    token_count=3,
                    layers=1,
                    batch_size=2,
                    prefill_len=3,
                    cache_len=8,
                    official_runner=official_runner,
                    buddy_runner=buddy_runner,
                )

        self.assertTrue(report["passed"])
        self.assertEqual(report["official_run"]["sample_count"], 3)
        self.assertEqual(report["metrics"]["buddy_aggregate"]["top1_accuracy"], 1.0)
        self.assertEqual(
            report["metrics"]["official_buddy_min_user_greedy_agreement"],
            1.0,
        )
        self.assertEqual(
            buddy_calls[0]["teacher_forcing_token_ids_by_step"],
            [[100, 100], [101, 101]],
        )
        self.assertEqual(buddy_calls[0]["execution_mode"], "eager")

    def test_replays_prompt_tail_beyond_static_prefill_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = root / "program"
            official = root / "official"
            model = root / "model"
            python = root / "python"
            for path in (program, official, model):
                path.mkdir()
            python.write_text("")
            output = root / "performance_correctness.json"
            reference = _reference(3)
            reference["prompt_token_ids"] = list(range(258))
            reference["full_sequence_token_count"] = 261
            buddy_calls: list[dict[str, object]] = []

            def official_runner(
                _command: object,
                _cwd: object,
                _environment: object,
                log_path: Path,
                _timeout: object,
                _limit: object,
            ) -> int:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "\n".join(
                        f"BUDDY_ACCURACY_SAMPLE token_iteration={index} "
                        f"predicted_token={token}"
                        for index, token in enumerate([100, 101, 102])
                    )
                    + "\n"
                )
                return 0

            def buddy_runner(**kwargs: object) -> dict[str, object]:
                buddy_calls.append(kwargs)
                row = [900, 901, 100, 101, 102]
                return {
                    "passed": True,
                    "status": "passed",
                    "generated_token_ids": [row, row],
                }

            with (
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness.load_official_performance_reference",
                    return_value=reference,
                ),
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness._buddy_prompt_from_reference",
                    return_value={
                        "text": "fixed prompt prefix",
                        "roundtrip_token_ids": list(range(256)),
                        "adaptation": "none",
                    },
                ),
                patch(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics."
                    "performance_correctness._git_value",
                    return_value="commit",
                ),
            ):
                report = run_performance_correctness(
                    out=output,
                    buddy_program=program,
                    official_tt_metal_root=official,
                    model_path=model,
                    tokenizer_path=model,
                    official_python=python,
                    token_count=3,
                    layers=1,
                    batch_size=2,
                    prefill_len=258,
                    cache_len=264,
                    official_runner=official_runner,
                    buddy_runner=buddy_runner,
                )

        self.assertTrue(report["passed"])
        self.assertEqual(buddy_calls[0]["prefill_len"], 256)
        self.assertEqual(buddy_calls[0]["max_new_tokens"], 5)
        self.assertEqual(
            buddy_calls[0]["teacher_forcing_token_ids_by_step"],
            [[256, 256], [257, 257], [100, 100], [101, 101]],
        )
        self.assertEqual(
            report["fixed_corpus"]["buddy_prompt_replay_decode_token_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
