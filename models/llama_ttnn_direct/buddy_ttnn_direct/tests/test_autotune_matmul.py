from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.legality import (
    DeviceDescriptor,
    WorkloadSpec,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.matmul import (
    MATMUL_OPERATORS,
    MATMUL_PROGRAM_FAMILIES,
    enumerate_all_matmul_programs,
    enumerate_matmul_programs,
    rank_matmul_measurement_candidates,
    select_matmul_microbenchmark_winner,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.microbench import (
    BenchmarkTarget,
    run_microbenchmark,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema import (
    CandidateConfig,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.sdpa import (
    enumerate_sdpa_programs,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search import (
    build_active_measurement_inputs,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.space import SearchSpaceConfig
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.transfer import (
    DEFAULT_LAYER_GROUP,
    build_llama31_8b_transfer_plan,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.config_runtime import (
    realize_ttnn_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    prompt_corpus_sha256,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.hardware_workers import (
    _constant_matmul_correctness,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_microbench import (
    WORKER_PATH,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    _official_runtime_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_ttnn_compat import (
    _fake_config_ttnn,
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
                self.assertGreaterEqual(len(operator["program_families"]), 2)
                self.assertEqual(
                    set(operator["family_availability"]),
                    set(MATMUL_PROGRAM_FAMILIES),
                )

                self.assertEqual(
                    operator["family_availability"]["dram_sharded"]["status"],
                    "legal",
                )
                self.assertEqual(
                    operator["family_availability"]["reuse_multicast_1d"][
                        "status"
                    ],
                    "legal",
                )
                for family, availability in operator[
                    "family_availability"
                ].items():
                    if availability["status"] != "legal":
                        self.assertTrue(
                            availability["reason"],
                            msg=f"{operator_name}:{family}",
                        )
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

    def test_constant_correctness_rejects_reduction_overflow(self) -> None:
        import torch

        overflowing = torch.full((1024,), torch.finfo(torch.float32).max)
        report = _constant_matmul_correctness(
            torch=torch,
            output=overflowing,
            expected=1.0,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["finite"])
        self.assertEqual(report["observed_mean"], float("inf"))

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
                    for program in candidate.programs:
                        if program.compute_grid is not None:
                            self.assertLessEqual(
                                program.compute_grid.x * program.compute_grid.y,
                                self.device.worker_core_count,
                            )
                            self.assertLessEqual(
                                len(program.allowed_worker_cores),
                                program.compute_grid.x * program.compute_grid.y,
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

    def test_every_legal_program_materializes_through_runtime_api(self) -> None:
        fake_ttnn = _fake_config_ttnn()
        observed_families = set()
        for operator_name in MATMUL_OPERATORS:
            enumeration = self._enumerate(operator_name)
            for candidate in enumeration.candidates:
                observed_families.add(candidate.program_family)
                for program in candidate.programs:
                    with self.subTest(
                        operator=operator_name,
                        candidate=candidate.candidate_id,
                        family=program.program_family,
                    ):
                        resolved = realize_ttnn_config(
                            program.to_runtime_descriptor(),
                            fake_ttnn,
                        )
                        self.assertIn("constructor", resolved)
        self.assertEqual(
            observed_families,
            {"dram_sharded", "reuse_multicast_1d"},
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

    def test_analytical_ranking_covers_each_legal_candidate(self) -> None:
        enumeration = self._enumerate("mlp.down")
        ranked = rank_matmul_measurement_candidates(enumeration)

        self.assertEqual(len(ranked), len(enumeration.candidates))
        self.assertEqual(
            {candidate.candidate_id for candidate in ranked},
            {candidate.candidate_id for candidate in enumeration.candidates},
        )
        self.assertEqual(sum(candidate.is_incumbent for candidate in ranked), 1)
        self.assertEqual(
            list(ranked),
            sorted(
                ranked,
                key=lambda item: (
                    item.analytical_score,
                    item.l1_bytes,
                    not item.is_incumbent,
                    item.candidate_id,
                ),
            ),
        )

    def test_enumerator_outputs_form_complete_active_measurement_inputs(self) -> None:
        matmuls = tuple(self._enumerate(operator) for operator in MATMUL_OPERATORS)
        sdpa = enumerate_sdpa_programs(
            base_space=self.space,
            workload=self.workload,
            device=self.device,
            precision_contract=self.precision,
        )

        groups, rejected = build_active_measurement_inputs(
            matmul_results=matmuls,
            sdpa_result=sdpa,
        )

        self.assertEqual(
            set(groups),
            {*MATMUL_OPERATORS, "attention.sdpa"},
        )
        self.assertEqual(set(rejected), set(groups))
        self.assertTrue(all(groups[operator] for operator in groups))

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
