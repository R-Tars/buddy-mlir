from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.correctness.performance_recipe import (
    greedy_agreement,
    load_official_performance_reference,
    token_accuracy,
)


class _FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class _FakeTorch:
    def load(self, *_args: object, **_kwargs: object) -> dict[str, _FakeTensor]:
        return {
            "reference_tokens": _FakeTensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "top5_tokens": _FakeTensor(
                [
                    [10, 11, 12, 13, 14],
                    [20, 21, 22, 23, 24],
                    [30, 31, 32, 33, 34],
                    [5, 50, 51, 52, 53],
                    [6, 60, 61, 62, 63],
                    [7, 70, 71, 72, 73],
                    [8, 80, 81, 82, 83],
                ]
            ),
        }


class PerformanceRecipeTest(unittest.TestCase):
    def test_reference_aligns_last_prompt_logit_with_first_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference.refpt"
            path.write_bytes(b"fixture")
            reference = load_official_performance_reference(
                path,
                token_count=3,
                torch_module=_FakeTorch(),
            )

        self.assertEqual(reference["prompt_token_ids"], [1, 2, 3, 4])
        self.assertEqual(reference["target_token_ids"], [5, 6, 7])
        self.assertEqual(reference["top5_token_ids"][0], [5, 50, 51, 52, 53])

    def test_token_accuracy_and_greedy_agreement(self) -> None:
        accuracy = token_accuracy(
            [5, 61, 999],
            [
                [5, 50, 51, 52, 53],
                [6, 60, 61, 62, 63],
                [7, 70, 71, 72, 73],
            ],
        )
        self.assertAlmostEqual(accuracy["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(accuracy["top5_accuracy"], 2 / 3)
        self.assertEqual(accuracy["first_top1_mismatch_positions"], [1, 2])

        agreement = greedy_agreement([5, 6, 8], [5, 7, 8])
        self.assertAlmostEqual(agreement["agreement"], 2 / 3)
        self.assertEqual(agreement["first_mismatch_positions"], [1])


if __name__ == "__main__":
    unittest.main()
