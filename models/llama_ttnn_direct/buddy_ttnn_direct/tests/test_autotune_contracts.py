from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    CandidateConfig,
    ContractViolation,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
    candidate_fingerprint,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    prompt_corpus_sha256,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class AutotuneContractTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json"
        self.template_config = json.loads(config_path.read_text())
        self.precision = PrecisionContract.from_template_config(self.template_config)

    def test_production_baseline_is_fully_represented(self) -> None:
        fields = self.precision.fields
        recipe = fields["dtype_recipe"]
        self.assertEqual(recipe["attention_qkv"], "bfloat8_b")
        self.assertEqual(recipe["attention_o_proj"], "bfloat8_b")
        self.assertEqual(recipe["mlp_intermediate"]["default"], "bfloat4_b")
        self.assertEqual(recipe["mlp_intermediate"]["layer_31"], "bfloat8_b")
        self.assertEqual(recipe["mlp_output"], "bfloat8_b")
        self.assertEqual(recipe["lm_head"], "bfloat8_b")
        self.assertEqual(recipe["kv_cache"], "bfloat8_b")
        self.assertEqual(recipe["rms_norm"], "bfloat16")

        runtime = fields["runtime_precision_fields"]
        self.assertEqual(
            runtime["attention.qkv_compute_kernel_config.math_fidelity"],
            "HiFi2",
        )
        self.assertTrue(runtime["attention.qkv_compute_kernel_config.fp32_dest_acc_en"])
        self.assertFalse(
            runtime[
                "mlp.layer_overrides.31."
                "gate_up_compute_kernel_config.math_approx_mode"
            ]
        )
        self.assertTrue(self.precision.frozen)
        self.assertEqual(len(self.precision.hash), 64)

    def test_precision_payload_mutation_is_rejected(self) -> None:
        payload = self.precision.to_dict()
        mutated = copy.deepcopy(payload)
        mutated["fields"]["dtype_recipe"]["attention_qkv"] = "bfloat16"
        with self.assertRaisesRegex(ContractViolation, "hash does not match"):
            PrecisionContract.from_dict(mutated)

    def test_candidate_rejects_a_rehashed_precision_contract(self) -> None:
        payload = self.precision.to_dict()
        payload.pop("hash")
        payload["fields"]["dtype_recipe"]["attention_qkv"] = "bfloat16"
        mutated_precision = PrecisionContract.from_dict(payload)
        with self.assertRaisesRegex(
            ContractViolation, "differs from the frozen baseline"
        ):
            self._candidate(precision_contract=mutated_precision)

    def test_candidate_cannot_contain_precision_knobs(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "frozen precision field"):
            self._candidate(
                tunable_state={
                    "templates": {"attention": "official"},
                    "dtype_recipe": "all_bf16_correctness",
                }
            )

    def test_execution_contract_rejects_non_production_mode(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "production execution contract"):
            ExecutionContract(
                prompt_corpus_sha256=prompt_corpus_sha256("prompt"),
                execution_mode="eager",
            )

    def test_baseline_execution_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.template_config)
        mutated["runtime_input_mode"] = "recreate"
        with self.assertRaisesRegex(ContractViolation, "baseline does not satisfy"):
            ExecutionContract.from_template_config(
                mutated,
                prompt_corpus_sha256=prompt_corpus_sha256("prompt"),
            )

    def test_fingerprint_covers_runtime_and_measurement_contract(self) -> None:
        baseline = self._candidate()
        runtime_changed = self._candidate(runtime_commit="runtime-b")
        measurement_changed = self._candidate(
            measurement=MeasurementContract(warmup=5, iterations=11)
        )
        state_changed = self._candidate(
            tunable_state={"templates": {"attention": "fused_qk"}}
        )
        self.assertNotEqual(
            candidate_fingerprint(baseline),
            candidate_fingerprint(runtime_changed),
        )
        self.assertNotEqual(
            candidate_fingerprint(baseline),
            candidate_fingerprint(measurement_changed),
        )
        self.assertNotEqual(
            candidate_fingerprint(baseline),
            candidate_fingerprint(state_changed),
        )

    def test_final_confirmation_contract_is_five_by_one_hundred_by_three(
        self,
    ) -> None:
        contract = MeasurementContract.final_confirmation()
        self.assertEqual(
            (contract.warmup, contract.iterations, contract.repetitions),
            (5, 100, 3),
        )

    def _candidate(
        self,
        *,
        runtime_commit: str = "runtime-a",
        measurement: MeasurementContract | None = None,
        tunable_state: dict | None = None,
        precision_contract: PrecisionContract | None = None,
    ) -> CandidateConfig:
        digest = "a" * 64
        return CandidateConfig.create(
            precision_contract=precision_contract or self.precision,
            expected_precision_hash=self.precision.hash,
            execution_contract=ExecutionContract(
                prompt_corpus_sha256=prompt_corpus_sha256("prompt")
            ),
            measurement_contract=measurement
            or MeasurementContract(warmup=5, iterations=10),
            semantic_graph_sha256=digest,
            model_config_sha256="b" * 64,
            weights_recipe_sha256="c" * 64,
            runtime_commit=runtime_commit,
            device_descriptor={
                "device": "p150a",
                "device_id": 0,
                "architecture": "blackhole",
            },
            target={
                "batch_size": 32,
                "cache_len": 1024,
                "page_block_size": 32,
            },
            tunable_state=tunable_state or {"templates": {"attention": "official"}},
        )


if __name__ == "__main__":
    unittest.main()
