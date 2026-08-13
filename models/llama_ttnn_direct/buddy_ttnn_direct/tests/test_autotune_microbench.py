from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    MeasurementCandidate,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.microbench import (
    ActiveMicrobenchmarkRunner,
    BenchmarkTarget,
    MicrobenchmarkError,
    build_microbench_report,
    make_worker_response,
    run_microbenchmark,
    write_microbench_report,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema import (
    CandidateConfig,
    ExecutionContract,
    MeasurementContract,
    PrecisionContract,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.transfer import (
    DEFAULT_LAYER_GROUP,
    OVERRIDE_LAYER_GROUP,
    build_llama31_8b_transfer_plan,
    transfer_representative_states,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune.measurement import (
    prompt_corpus_sha256,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = (
    "models.llama_ttnn_direct.buddy_ttnn_direct.tests."
    "test_autotune_microbench:deterministic_worker"
)


def deterministic_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = request["payload"]
    counter_path = Path(payload["counter_path"])
    count = int(counter_path.read_text()) if counter_path.is_file() else 0
    counter_path.write_text(str(count + 1))
    repetition = int(request["repetition"])
    iterations = int(request["candidate"]["measurement_contract"]["iterations"])
    base = float(payload.get("base", 1.0)) + repetition
    samples = [base + 0.25 * index for index in range(iterations)]
    return make_worker_response(
        request,
        samples,
        program_cache_count=7 + repetition,
        trace_capture_count=1,
        new_tensor_allocations=0,
        metadata={"worker_kind": "deterministic", "repetition": repetition},
    )


def invalid_worker(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "passed", "samples": [1.0]}


def failing_worker(_: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("intentional isolated failure")


class RepresentativeTransferTest(unittest.TestCase):
    def test_llama_transfer_groups_are_explicit_and_complete(self) -> None:
        plan = build_llama31_8b_transfer_plan()
        default = plan.group(DEFAULT_LAYER_GROUP)
        override = plan.group(OVERRIDE_LAYER_GROUP)

        self.assertEqual(default.representative_layer, 0)
        self.assertEqual(default.member_layers, tuple(range(31)))
        self.assertTrue(default.transfer_enabled)
        self.assertEqual(override.representative_layer, 31)
        self.assertEqual(override.member_layers, (31,))
        self.assertFalse(override.transfer_enabled)

        report = plan.to_dict()
        self.assertEqual(report["representative_layers"], [0, 31])
        self.assertEqual(len(report["layer_assignments"]), 32)
        self.assertEqual(report["layer_assignments"][30]["representative_layer"], 0)
        self.assertEqual(report["layer_assignments"][31]["representative_layer"], 31)

    def test_representative_states_expand_with_auditable_hashes(self) -> None:
        plan = build_llama31_8b_transfer_plan()
        layer_states, report = transfer_representative_states(
            plan,
            {
                DEFAULT_LAYER_GROUP: {"program": "default"},
                OVERRIDE_LAYER_GROUP: {"program": "layer31"},
            },
        )

        self.assertEqual(len(layer_states), 32)
        self.assertEqual(layer_states[0], {"program": "default"})
        self.assertEqual(layer_states[30], {"program": "default"})
        self.assertEqual(layer_states[31], {"program": "layer31"})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(
            report["layer_assignments"][0]["state_sha256"],
            report["layer_assignments"][30]["state_sha256"],
        )
        self.assertNotEqual(
            report["layer_assignments"][30]["state_sha256"],
            report["layer_assignments"][31]["state_sha256"],
        )


class RepresentativeMicrobenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json").read_text()
        )
        cls.precision = PrecisionContract.from_template_config(config)

    def test_cache_cv_runtime_invalidation_and_process_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            counter = root / "worker_count.txt"
            cache = root / "measurement_cache"
            plan = build_llama31_8b_transfer_plan()
            target = BenchmarkTarget.op(
                "qkv_linear",
                layer_group=DEFAULT_LAYER_GROUP,
                representative_layer=0,
            )
            candidate = self._candidate(runtime_commit="runtime-a")

            first = run_microbenchmark(
                candidate=candidate,
                target=target,
                transfer_plan=plan,
                cache_dir=cache,
                worker=WORKER_PATH,
                payload={"counter_path": str(counter), "base": 1.0},
                timeout_seconds=10.0,
            )

            self.assertEqual(first["status"], "passed")
            self.assertFalse(first["cache"]["hit"])
            self.assertEqual(counter.read_text(), "2")
            self.assertEqual(len(first["repetitions"]), 2)
            self.assertEqual(first["statistics"]["sample_count"], 8)
            self.assertGreater(first["statistics"]["coefficient_of_variation"], 0.0)
            self.assertGreater(
                first["statistics"]["coefficient_of_variation_percent"], 0.0
            )
            child_pids = {
                repetition["process"]["pid"] for repetition in first["repetitions"]
            }
            self.assertEqual(len(child_pids), 2)
            self.assertNotIn(os.getpid(), child_pids)
            self.assertTrue(
                all(
                    repetition["process"]["exit_code"] == 0
                    for repetition in first["repetitions"]
                )
            )
            self.assertEqual(first["instrumentation"]["program_cache_count"], 8)
            self.assertEqual(first["instrumentation"]["trace_capture_count"], 2)
            self.assertEqual(first["instrumentation"]["new_tensor_allocations"], 0)

            cached = run_microbenchmark(
                candidate=candidate,
                target=target,
                transfer_plan=plan,
                cache_dir=cache,
                worker=WORKER_PATH,
                payload={"counter_path": str(counter), "base": 1.0},
                timeout_seconds=10.0,
            )
            self.assertTrue(cached["cache"]["hit"])
            self.assertTrue(cached["cache"]["measurement_reused"])
            self.assertEqual(counter.read_text(), "2")
            self.assertEqual(
                cached["candidate_fingerprint"], first["candidate_fingerprint"]
            )

            runtime_changed = run_microbenchmark(
                candidate=self._candidate(runtime_commit="runtime-b"),
                target=target,
                transfer_plan=plan,
                cache_dir=cache,
                worker=WORKER_PATH,
                payload={"counter_path": str(counter), "base": 1.0},
                timeout_seconds=10.0,
            )
            self.assertFalse(runtime_changed["cache"]["hit"])
            self.assertEqual(counter.read_text(), "4")
            self.assertNotEqual(runtime_changed["cache"]["key"], first["cache"]["key"])

    def test_op_and_region_reports_retain_transfer_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            counter = root / "worker_count.txt"
            plan = build_llama31_8b_transfer_plan()
            candidate = self._candidate(runtime_commit="runtime-op-region")
            common = {
                "candidate": candidate,
                "transfer_plan": plan,
                "cache_dir": root / "measurement_cache",
                "worker": WORKER_PATH,
                "payload": {"counter_path": str(counter), "base": 2.0},
                "timeout_seconds": 10.0,
            }
            op_result = run_microbenchmark(
                target=BenchmarkTarget.op(
                    "mlp_gate",
                    layer_group=DEFAULT_LAYER_GROUP,
                    representative_layer=0,
                ),
                **common,
            )
            region_result = run_microbenchmark(
                target=BenchmarkTarget.region(
                    "mlp_region",
                    ("mlp_gate", "mlp_up", "mul_silu", "mlp_down"),
                    layer_group=OVERRIDE_LAYER_GROUP,
                    representative_layer=31,
                ),
                **common,
            )

            report = build_microbench_report(
                [op_result, region_result], transfer_plan=plan
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["granularity_counts"], {"op": 1, "region": 1})
            self.assertEqual(
                op_result["transfer"]["group"]["member_layers"],
                list(range(31)),
            )
            self.assertEqual(region_result["transfer"]["group"]["member_layers"], [31])
            out = root / "microbench_report.json"
            written = write_microbench_report(
                out, [op_result, region_result], transfer_plan=plan
            )
            self.assertEqual(json.loads(out.read_text()), written)

    def test_invalid_worker_result_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = run_microbenchmark(
                candidate=self._candidate(runtime_commit="runtime-invalid"),
                target=BenchmarkTarget.op(
                    "qkv_linear",
                    layer_group=DEFAULT_LAYER_GROUP,
                    representative_layer=0,
                ),
                transfer_plan=build_llama31_8b_transfer_plan(),
                cache_dir=root / "measurement_cache",
                worker=(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.tests."
                    "test_autotune_microbench:invalid_worker"
                ),
                timeout_seconds=10.0,
            )
            self.assertEqual(result["status"], "failed")
            self.assertFalse(Path(result["cache"]["path"]).exists())

    def test_nonzero_worker_exit_preserves_reported_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_microbenchmark(
                candidate=self._candidate(runtime_commit="runtime-failing"),
                target=BenchmarkTarget.op(
                    "qkv_linear",
                    layer_group=DEFAULT_LAYER_GROUP,
                    representative_layer=0,
                ),
                transfer_plan=build_llama31_8b_transfer_plan(),
                cache_dir=Path(tmpdir) / "measurement_cache",
                worker=(
                    "models.llama_ttnn_direct.buddy_ttnn_direct.tests."
                    "test_autotune_microbench:failing_worker"
                ),
                timeout_seconds=10.0,
            )

            repetition = result["repetitions"][0]
            self.assertEqual(result["status"], "failed")
            self.assertIn("intentional isolated failure", repetition["error"])
            self.assertEqual(
                repetition["process"]["worker_response"]["error"],
                "RuntimeError: intentional isolated failure",
            )

    def test_active_runner_applies_each_round_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            counter = root / "worker_count.txt"
            descriptor = MeasurementCandidate.create(
                candidate_id="qkv-active",
                operator_name="attention.qkv",
                candidate_kind="matmul",
                analytical_score=1.0,
                l1_bytes=1024,
                source="fixture",
                is_incumbent=True,
            )
            runner = ActiveMicrobenchmarkRunner(
                candidate_configs={
                    descriptor.candidate_id: self._candidate(
                        runtime_commit="runtime-active"
                    )
                },
                targets={
                    descriptor.operator_name: BenchmarkTarget.op(
                        "attention.qkv",
                        layer_group=DEFAULT_LAYER_GROUP,
                        representative_layer=0,
                    )
                },
                transfer_plan=build_llama31_8b_transfer_plan(),
                cache_dir=root / "measurement_cache",
                worker=WORKER_PATH,
                payloads={
                    descriptor.candidate_id: {
                        "counter_path": str(counter),
                        "base": 1.0,
                    }
                },
                timeout_seconds=10.0,
            )
            contract = MeasurementContract(
                warmup=3,
                iterations=10,
                repetitions=1,
                kind="successive_halving_round_1",
            )

            report = runner(descriptor, contract, "round_1_short")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["measurement_contract"], contract.to_dict())
            self.assertEqual(report["statistics"]["sample_count"], 10)
            self.assertEqual(counter.read_text(), "1")
            self.assertEqual(report["active_scheduler"]["round"], "round_1_short")

    def test_target_must_use_the_group_representative(self) -> None:
        with self.assertRaisesRegex(MicrobenchmarkError, "uses 0"):
            BenchmarkTarget.op(
                "qkv_linear",
                layer_group=DEFAULT_LAYER_GROUP,
                representative_layer=1,
            ).validate_transfer_plan(build_llama31_8b_transfer_plan())

    def _candidate(self, *, runtime_commit: str) -> CandidateConfig:
        return CandidateConfig.create(
            precision_contract=self.precision,
            expected_precision_hash=self.precision.hash,
            execution_contract=ExecutionContract(
                prompt_corpus_sha256=prompt_corpus_sha256("fixed prompts")
            ),
            measurement_contract=MeasurementContract(
                warmup=3,
                iterations=4,
                repetitions=2,
                kind="representative_microbench",
            ),
            semantic_graph_sha256="a" * 64,
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
            tunable_state={"templates": {"attention.rope": "separate_qk_rope"}},
        )


if __name__ == "__main__":
    unittest.main()
