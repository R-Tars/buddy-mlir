from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    CompileValidationRequest,
    CoreGrid,
    DeviceDescriptor,
    FusedQKRoPEWorkload,
    MemoryConfig,
    PagedFusedUpdateWorkload,
    SearchSpaceConfig,
    TensorSpec,
    WorkloadSpec,
    classify_validation_error,
    run_compile_validation,
    validate_candidate,
    write_legality_report,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.config import (
    build_codegen_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    load_template_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class AutotuneLegalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = _official_runtime_config()
        cls.space = SearchSpaceConfig.from_runtime_config(cls.runtime)
        cls.workload = WorkloadSpec.from_runtime_config(cls.runtime)
        cls.device = DeviceDescriptor.p150a()

    def test_current_official_config_is_statically_legal(self) -> None:
        report = validate_candidate(
            self.space,
            self.workload,
            self.device,
            candidate_id="official",
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.status, "static_legal")
        self.assertEqual(report.issues, ())
        self.assertEqual(len(report.l1_estimates), 14)
        self.assertTrue(
            all(
                estimate.total_bytes <= estimate.limit_bytes
                for estimate in report.l1_estimates
            )
        )

    def test_grid_shape_program_and_shard_errors_are_stable(self) -> None:
        payload = self.space.to_dict()
        payload["core_grids"]["attention.core_grid"] = [12, 10]
        qkv = payload["operators"]["attention.qkv"]["programs"][0]
        qkv["in0_block_w"] = 3
        input_path = "rms_norm.attention.output_memory_config"
        payload["memory_configs"][input_path]["shard_shape"] = [32, 64]

        report = validate_candidate(
            SearchSpaceConfig.from_dict(payload),
            self.workload,
            self.device,
        )
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.passed)
        self.assertEqual(report.status, "static_rejected")
        self.assertIn("GRID_OUT_OF_BOUNDS", codes)
        self.assertIn("MATMUL_K_BLOCK_DIVISIBILITY", codes)
        self.assertIn("SHARD_SHAPE_MISMATCH", codes)
        self.assertEqual(report.error_class, "invalid_core_grid")

    def test_l1_overflow_is_rejected_without_spawning_compile(self) -> None:
        payload = self.space.to_dict()
        qkv = payload["operators"]["attention.qkv"]["programs"][0]
        qkv["in0_block_w"] = 128
        qkv["per_core_n"] = 512
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "compile-ran"
            request = CompileValidationRequest(
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                )
            )
            report = validate_candidate(
                SearchSpaceConfig.from_dict(payload),
                self.workload,
                self.device,
                compile_request=request,
            )

            self.assertFalse(report.static_passed)
            self.assertIsNone(report.compile_validation)
            self.assertFalse(marker.exists())
            self.assertIn(
                "L1_SAFETY_LIMIT_EXCEEDED",
                {issue.code for issue in report.issues},
            )
            self.assertIn(
                "CB_PAGE_COUNT_EXCEEDED",
                {issue.code for issue in report.issues},
            )
            self.assertIn(
                "l1_overflow",
                {issue.error_class for issue in report.issues},
            )

    def test_sdpa_chunk_core_and_l1_constraints(self) -> None:
        payload = self.space.to_dict()
        sdpa = payload["operators"]["attention.sdpa"]
        sdpa["grid"] = [8, 4]
        sdpa["max_cores_per_head_batch"] = 64
        sdpa["q_chunk_size"] = 1024
        sdpa["k_chunk_size"] = 4096

        report = validate_candidate(
            SearchSpaceConfig.from_dict(payload),
            self.workload,
            self.device,
        )
        codes = {issue.code for issue in report.issues}

        self.assertIn("SDPA_CORE_ALLOCATION_EXCEEDED", codes)
        self.assertIn("L1_SAFETY_LIMIT_EXCEEDED", codes)
        self.assertIn("CB_PAGE_COUNT_EXCEEDED", codes)

    def test_sdpa_sub_core_grids_must_be_disjoint_and_bounded(self) -> None:
        payload = self.space.to_dict()
        sdpa = payload["operators"]["attention.sdpa"]
        sdpa["sub_core_grids"] = [[0, 0, 4, 3], [4, 3, 8, 7]]

        report = validate_candidate(
            SearchSpaceConfig.from_dict(payload),
            self.workload,
            self.device,
        )
        codes = {issue.code for issue in report.issues}

        self.assertIn("SDPA_SUB_CORE_OVERLAP", codes)
        self.assertIn("SDPA_SUB_CORE_OUT_OF_BOUNDS", codes)

    def test_paged_fused_update_legal_and_overlap_rejected(self) -> None:
        fused = _fused_cache_workload()
        space = _with_template(
            self.space,
            "attention.kv_update",
            "paged_fused_update_cache",
        )
        legal = validate_candidate(
            space,
            replace(self.workload, paged_fused_update=fused),
            self.device,
        )
        self.assertTrue(legal.passed, [issue.to_dict() for issue in legal.issues])

        overlapping_value = replace(
            fused.value_input,
            cores=fused.key_input.cores,
        )
        invalid = validate_candidate(
            space,
            replace(
                self.workload,
                paged_fused_update=replace(
                    fused,
                    value_input=overlapping_value,
                ),
            ),
            self.device,
        )
        self.assertIn(
            "FUSED_CACHE_CORE_OVERLAP",
            {issue.code for issue in invalid.issues},
        )
        self.assertEqual(invalid.error_class, "invalid_core_grid")

    def test_fused_qk_rope_legal_and_dtype_rejected(self) -> None:
        fused = _fused_qk_rope_workload()
        space = _with_template(self.space, "attention.rope", "fused_qk_rope")
        legal = validate_candidate(
            space,
            replace(self.workload, fused_qk_rope=fused),
            self.device,
        )
        self.assertTrue(legal.passed, [issue.to_dict() for issue in legal.issues])

        invalid = validate_candidate(
            space,
            replace(
                self.workload,
                fused_qk_rope=replace(
                    fused,
                    q=replace(fused.q, dtype="bfloat8_b"),
                ),
            ),
            self.device,
        )
        self.assertIn(
            "FUSED_QK_ROPE_DTYPE",
            {issue.code for issue in invalid.issues},
        )
        self.assertEqual(invalid.error_class, "unsupported_layout")

    def test_compile_validation_is_isolated_and_classified(self) -> None:
        success = run_compile_validation(
            CompileValidationRequest(
                command=(sys.executable, "-c", "print('compiled')")
            )
        )
        self.assertTrue(success.passed)
        self.assertEqual(success.stdout_tail.strip(), "compiled")

        failure = run_compile_validation(
            CompileValidationRequest(
                command=(
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('L1 buffer overflow'); sys.exit(3)",
                )
            )
        )
        self.assertFalse(failure.passed)
        self.assertEqual(failure.return_code, 3)
        self.assertEqual(failure.error_class, "l1_overflow")

        timeout = run_compile_validation(
            CompileValidationRequest(
                command=(sys.executable, "-c", "import time; time.sleep(2)"),
                timeout_seconds=0.05,
            )
        )
        self.assertEqual(timeout.status, "timeout")
        self.assertEqual(timeout.error_class, "runtime_error")

        unavailable = run_compile_validation(
            CompileValidationRequest(command=("definitely-not-a-real-command",))
        )
        self.assertEqual(unavailable.status, "launch_failed")
        self.assertEqual(unavailable.error_class, "api_unavailable")

    def test_all_documented_error_classes_have_stable_patterns(self) -> None:
        cases = {
            "shape incompatible": "shape_incompatible",
            "L1 overflow": "l1_overflow",
            "unsupported layout": "unsupported_layout",
            "invalid program config": "invalid_program_config",
            "invalid core grid": "invalid_core_grid",
            "API unavailable": "api_unavailable",
            "compilation failed": "compile_error",
            "runtime error": "runtime_error",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(classify_validation_error(message)[0], expected)

    def test_legality_report_is_atomic_and_compact(self) -> None:
        accepted = validate_candidate(
            self.space,
            self.workload,
            self.device,
            candidate_id="official",
        )
        payload = self.space.to_dict()
        payload["core_grids"]["attention.core_grid"] = [12, 10]
        rejected = validate_candidate(
            SearchSpaceConfig.from_dict(payload),
            self.workload,
            self.device,
            candidate_id="invalid-grid",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legality_report.json"
            payload = write_legality_report(path, [accepted, rejected])
            loaded = json.loads(path.read_text())

        self.assertEqual(payload, loaded)
        self.assertTrue(loaded["passed"])
        self.assertEqual(loaded["status"], "completed")
        self.assertTrue(loaded["all_candidates_processed"])
        self.assertFalse(loaded["all_candidates_accepted"])
        self.assertEqual(loaded["accepted_count"], 1)
        self.assertEqual(loaded["rejected_count"], 1)
        self.assertEqual(loaded["candidates"][0]["status"], "static_legal")
        self.assertEqual(loaded["candidates"][1]["status"], "static_rejected")


def _with_template(
    space: SearchSpaceConfig, name: str, value: str
) -> SearchSpaceConfig:
    payload = space.to_dict()
    payload["templates"][name] = value
    return SearchSpaceConfig.from_dict(payload)


def _sharded_memory(
    *, grid: tuple[int, int], shard_shape: tuple[int, int]
) -> MemoryConfig:
    return MemoryConfig.from_dict(
        {
            "buffer": "l1",
            "layout": "height_sharded",
            "runtime_kind": "ttnn_sharded_memory_config",
            "runtime_name": None,
            "grid": list(grid),
            "shard_shape": list(shard_shape),
            "orientation": "row_major",
            "extra_fields": {},
        }
    )


def _core_rectangle(
    x_start: int, y_start: int, width: int, height: int
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (x, y)
        for y in range(y_start, y_start + height)
        for x in range(x_start, x_start + width)
    )


def _tensor(
    name: str,
    logical_shape: tuple[int, ...],
    padded_shape: tuple[int, ...],
    *,
    dtype: str,
    layout: str,
    memory: MemoryConfig,
    cores: tuple[tuple[int, int], ...] = (),
) -> TensorSpec:
    return TensorSpec(
        name=name,
        logical_shape=logical_shape,
        padded_shape=padded_shape,
        dtype=dtype,
        layout=layout,
        memory=memory,
        cores=cores,
    )


def _fused_cache_workload() -> PagedFusedUpdateWorkload:
    key_cores = _core_rectangle(0, 0, 8, 4)
    value_cores = _core_rectangle(0, 4, 8, 4)
    input_memory = _sharded_memory(grid=(8, 4), shard_shape=(8, 128))
    cache_memory = MemoryConfig.named("DRAM_MEMORY_CONFIG")
    key_input = _tensor(
        "key_input",
        (1, 32, 8, 128),
        (1, 32, 8, 128),
        dtype="bf16",
        layout="tile",
        memory=input_memory,
        cores=key_cores,
    )
    value_input = replace(key_input, name="value_input", cores=value_cores)
    key_cache = _tensor(
        "key_cache",
        (1024, 8, 32, 128),
        (1024, 8, 32, 128),
        dtype="bfloat8_b",
        layout="tile",
        memory=cache_memory,
    )
    value_cache = replace(key_cache, name="value_cache")
    page_table = _tensor(
        "page_table",
        (32, 32),
        (32, 32),
        dtype="int32",
        layout="row_major",
        memory=cache_memory,
    )
    update_indices = _tensor(
        "update_indices",
        (32,),
        (32,),
        dtype="int32",
        layout="row_major",
        memory=cache_memory,
    )
    return PagedFusedUpdateWorkload(
        key_input=key_input,
        value_input=value_input,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        update_indices=update_indices,
    )


def _fused_qk_rope_workload() -> FusedQKRoPEWorkload:
    q_cores = _core_rectangle(0, 0, 8, 4)
    k_cores = _core_rectangle(0, 4, 8, 4)
    all_cores = q_cores + k_cores
    qk_memory = _sharded_memory(grid=(8, 4), shard_shape=(32, 128))
    cos_sin_memory = _sharded_memory(grid=(8, 8), shard_shape=(32, 128))
    transform_memory = _sharded_memory(grid=(8, 8), shard_shape=(32, 32))
    q = _tensor(
        "q",
        (1, 32, 32, 128),
        (1, 32, 32, 128),
        dtype="bf16",
        layout="tile",
        memory=qk_memory,
        cores=q_cores,
    )
    k = replace(q, name="k", cores=k_cores)
    cos = _tensor(
        "cos",
        (1, 64, 1, 128),
        (1, 64, 32, 128),
        dtype="bf16",
        layout="tile",
        memory=cos_sin_memory,
        cores=all_cores,
    )
    sin = replace(cos, name="sin")
    transformation = _tensor(
        "transformation",
        (1, 64, 32, 32),
        (1, 64, 32, 32),
        dtype="bf16",
        layout="tile",
        memory=transform_memory,
        cores=all_cores,
    )
    return FusedQKRoPEWorkload(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        transformation=transformation,
        q_output_memory=qk_memory,
        k_output_memory=qk_memory,
    )


def _official_runtime_config() -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_root = Path(tmpdir)
        (model_root / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": "fake-llama-legality",
                    "model_type": "llama",
                    "num_hidden_layers": 2,
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "vocab_size": 128256,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 500000.0,
                    "max_position_embeddings": 131072,
                    "tie_word_embeddings": False,
                }
            )
        )
        seed = load_template_config(
            PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json"
        )
        graph = import_hf_llama(
            model_root,
            mode="decode",
            batch_size=32,
            seq_len=1,
            max_cache_len=1024,
            generation_mode="greedy",
        )
        return build_codegen_config(build_execution_plan(graph, seed))


if __name__ == "__main__":
    unittest.main()
