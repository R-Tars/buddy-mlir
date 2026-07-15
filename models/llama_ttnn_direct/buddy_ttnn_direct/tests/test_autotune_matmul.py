from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    DEFAULT_LAYER_GROUP,
    MATMUL_OPERATORS,
    BenchmarkTarget,
    CandidateConfig,
    DeviceDescriptor,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    SearchSpaceConfig,
    WorkloadSpec,
    build_llama31_8b_transfer_plan,
    enumerate_all_matmul_programs,
    enumerate_matmul_programs,
    run_microbenchmark,
    select_matmul_microbenchmark_winner,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    prompt_corpus_sha256,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_microbench import (
    WORKER_PATH,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    _official_runtime_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PRECISION_MARKERS = (
    "dtype",
    "precision",
    "math_fidelity",
    "fp32_dest_acc",
    "math_approx",
    "exp_approx",
    "packer_l1_acc",
)


class MatmulProgramEnumeratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = _official_runtime_config()
        cls.space = SearchSpaceConfig.from_runtime_config(cls.runtime)
        cls.workload = WorkloadSpec.from_runtime_config(cls.runtime)
        cls.device = DeviceDescriptor.p150a()
        template = json.loads(
            (PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json").read_text()
        )
        cls.precision = PrecisionContract.from_template_config(template)

    def test_all_required_ops_include_official_and_multiple_legal_candidates(
        self,
    ) -> None:
        report = enumerate_all_matmul_programs(
            base_space=self.space,
            workload=self.workload,
            device=self.device,
            precision_contract=self.precision,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["operators"]), set(MATMUL_OPERATORS))
        self.assertEqual(report["operator_count"], 6)
        self.assertEqual(report["precision_contract_hash"], self.precision.hash)
        for operator_name in MATMUL_OPERATORS:
            with self.subTest(operator=operator_name):
                operator = report["operators"][operator_name]
                self.assertEqual(operator["status"], "passed")
                self.assertEqual(len(operator["official_candidate_ids"]), 1)
                self.assertGreaterEqual(operator["legal_candidate_count"], 2)
                self.assertTrue(
                    all(
                        candidate["legality"]["passed"]
                        for candidate in operator["candidates"]
                    )
                )
                self.assertTrue(
                    any(
                        candidate["source"] == "shape_derived"
                        for candidate in operator["candidates"]
                    )
                )

    def test_candidates_obey_shape_worker_l1_and_precision_contracts(self) -> None:
        baseline_precision = _collect_precision_fields(
            _without_autotune_metadata(self.runtime)
        )
        for operator_name in MATMUL_OPERATORS:
            enumeration = self._enumerate(operator_name)
            for candidate in enumeration.candidates:
                with self.subTest(
                    operator=operator_name, candidate=candidate.candidate_id
                ):
                    self.assertEqual(
                        enumeration.precision_contract_hash, self.precision.hash
                    )
                    self.assertLessEqual(
                        max(candidate.worker_core_counts),
                        self.device.worker_core_count,
                    )
                    for shape, program in zip(
                        enumeration.workloads, candidate.programs
                    ):
                        self.assertEqual(
                            (shape.k // 32) % int(program.parameters["in0_block_w"]),
                            0,
                        )
                        self.assertEqual(int(program.parameters["per_core_m"]), 1)
                    target_estimates = [
                        estimate
                        for estimate in candidate.legality.l1_estimates
                        if estimate.path.startswith(operator_name)
                    ]
                    self.assertEqual(len(target_estimates), len(enumeration.workloads))
                    self.assertTrue(
                        all(
                            estimate.total_bytes <= estimate.limit_bytes
                            for estimate in target_estimates
                        )
                    )
                    generated = candidate.search_space.apply_to_runtime_config(
                        self.runtime
                    )
                    self.assertEqual(
                        _collect_precision_fields(
                            _without_autotune_metadata(generated)
                        ),
                        baseline_precision,
                    )

    def test_official_program_vector_is_exactly_preserved(self) -> None:
        for operator_name in MATMUL_OPERATORS:
            enumeration = self._enumerate(operator_name)
            official = enumeration.official_candidates[0]
            baseline_programs = self.space.operators[operator_name]["programs"]
            self.assertEqual(
                [program.to_dict() for program in official.programs],
                baseline_programs,
            )

    def test_phase4_microbench_reports_select_a_non_promoted_winner(self) -> None:
        enumeration = self._enumerate("attention.qkv")
        candidates = enumeration.candidates[:3]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = {}
            for index, candidate in enumerate(candidates):
                config = CandidateConfig.create(
                    precision_contract=self.precision,
                    expected_precision_hash=self.precision.hash,
                    execution_contract=ExecutionContract(
                        prompt_corpus_sha256=prompt_corpus_sha256(
                            "fixed matmul prompts"
                        )
                    ),
                    measurement_contract=MeasurementContract.short_microbenchmark(),
                    semantic_graph_sha256="a" * 64,
                    model_config_sha256="b" * 64,
                    weights_recipe_sha256="c" * 64,
                    runtime_commit="runtime-matmul",
                    device_descriptor=self.device.to_dict(),
                    target={
                        "batch_size": 32,
                        "cache_len": 1024,
                        "page_block_size": 32,
                    },
                    tunable_state={
                        "operator": candidate.operator_name,
                        "programs": [
                            program.to_dict() for program in candidate.programs
                        ],
                    },
                )
                measurements[candidate.candidate_id] = run_microbenchmark(
                    candidate=config,
                    target=BenchmarkTarget.op(
                        "attention.qkv",
                        layer_group=DEFAULT_LAYER_GROUP,
                        representative_layer=0,
                    ),
                    transfer_plan=build_llama31_8b_transfer_plan(),
                    cache_dir=root / "measurement_cache",
                    worker=WORKER_PATH,
                    payload={
                        "counter_path": str(root / f"counter-{index}.txt"),
                        "base": (3.0, 1.0, 2.0)[index],
                    },
                    timeout_seconds=10.0,
                )

            selection = select_matmul_microbenchmark_winner(
                enumeration,
                measurements,
            )
            self.assertEqual(selection["status"], "selected")
            self.assertEqual(
                selection["winner"]["candidate_id"], candidates[1].candidate_id
            )
            self.assertFalse(selection["promotion_allowed"])
            self.assertEqual(
                selection["selection_scope"], "representative_microbenchmark_only"
            )

    def _enumerate(self, operator_name: str):
        return enumerate_matmul_programs(
            operator_name=operator_name,
            base_space=self.space,
            workload=self.workload,
            device=self.device,
            precision_contract=self.precision,
        )


def _collect_precision_fields(value, path=()):
    result = {}
    if not isinstance(value, dict):
        return result
    for key, child in value.items():
        child_path = (*path, str(key))
        if any(marker in str(key).lower() for marker in _PRECISION_MARKERS):
            result[".".join(child_path)] = child
        else:
            result.update(_collect_precision_fields(child, child_path))
    return result


def _without_autotune_metadata(value):
    result = copy.deepcopy(value)
    result.pop("autotune", None)
    template_config = result.get("template_config")
    if isinstance(template_config, dict):
        template_config.pop("autotune", None)
    return result


if __name__ == "__main__":
    unittest.main()
