from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    DeviceDescriptor,
    PrecisionContract,
    SearchSpaceConfig,
    WorkloadSpec,
    enumerate_sdpa_programs,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.config_runtime import (
    realize_ttnn_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_layer import (
    run_smoke_attention_layer,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.smoke_attention_primitive import (
    run_smoke_attention_primitive,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    load_template_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    PACKAGE_ROOT,
    _official_runtime_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_ttnn_compat import (
    _fake_config_ttnn,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests_diagnostics.test_smoke_attention_primitive import (
    _fake_torch,
    _fake_ttnn,
)


class SDPAEnumeratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = _official_runtime_config()
        cls.space = SearchSpaceConfig.from_runtime_config(cls.runtime)
        cls.device = DeviceDescriptor.p150a()
        cls.precision = PrecisionContract.from_template_config(
            load_template_config(PACKAGE_ROOT / "configs" / "p150a_llama31_8b_b32.json")
        )

    def test_official_and_all_required_dimensions_are_enumerated(self) -> None:
        enumeration = self._enumerate(self.runtime)
        report = enumeration.to_dict()

        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["official_candidate_ids"]), 1)
        self.assertGreaterEqual(report["legal_candidate_count"], 2)
        self.assertEqual(report["precision_contract_hash"], self.precision.hash)
        self.assertEqual(
            report["searched_dimensions"]["grid"],
            [[8, 4], [8, 8], [8, 10], [10, 8], [10, 10]],
        )
        self.assertEqual(
            report["searched_dimensions"]["q_chunk_size"],
            [0, 32, 64],
        )
        self.assertEqual(
            report["searched_dimensions"]["k_chunk_size"],
            [0, 32, 64, 128, 256, 512],
        )
        self.assertEqual(
            report["searched_dimensions"]["max_cores_per_head_batch"],
            [4, 8, 16, 32],
        )
        self.assertGreater(
            len(report["searched_dimensions"]["sub_core_grids"]),
            1,
        )
        self.assertEqual(
            {
                item["buffer"]
                for item in report["searched_dimensions"]["kernel_output_memory"]
            },
            {"dram", "l1"},
        )

    def test_exp_approximation_is_frozen_for_every_candidate(self) -> None:
        enumeration = self._enumerate(self.runtime)
        official = enumeration.official_candidates[0]

        self.assertFalse(official.program.exp_approx_mode)
        self.assertTrue(
            all(
                candidate.program.exp_approx_mode == official.program.exp_approx_mode
                for candidate in enumeration.candidates
            )
        )
        self.assertEqual(
            enumeration.to_dict()["frozen_dimensions"],
            {"exp_approx_mode": False},
        )

    def test_gqa_sharded_kernel_outputs_are_rejected(self) -> None:
        enumeration = self._enumerate(self.runtime)
        self.assertTrue(
            all(
                candidate.kernel_output_memory.layout == "interleaved"
                for candidate in enumeration.candidates
            )
        )
        issue_codes = {
            issue["code"]
            for candidate in enumeration.rejected
            for issue in candidate["legality"]["issues"]
        }
        self.assertIn("SDPA_GQA_SHARDED_OUTPUT_UNSUPPORTED", issue_codes)

    def test_official_runtime_config_is_preserved_exactly(self) -> None:
        official = self._enumerate(self.runtime).official_candidates[0]
        generated = official.search_space.apply_to_runtime_config(self.runtime)

        self.assertEqual(
            generated["attention"]["sdpa_program_config"],
            self.runtime["attention"]["sdpa_program_config"],
        )
        self.assertEqual(
            generated["attention"]["sdpa_kernel_output_memory_config"],
            self.runtime["attention"]["sdpa_kernel_output_memory_config"],
        )
        self.assertEqual(
            generated["attention"]["sdpa_output_memory_config"],
            self.runtime["attention"]["sdpa_output_memory_config"],
        )

    def test_candidate_runtime_writeback_keeps_grid_and_memory_choice(self) -> None:
        enumeration = self._enumerate(self.runtime)
        candidate = next(
            item
            for item in enumeration.candidates
            if item.program.grid.to_list() == [10, 10]
            and item.program.sub_core_grids is None
            and item.kernel_output_memory.runtime_name == "L1_MEMORY_CONFIG"
        )
        generated = candidate.search_space.apply_to_runtime_config(self.runtime)

        self.assertEqual(
            generated["attention"]["sdpa_program_config"]["core_grid"],
            [10, 10],
        )
        self.assertNotIn(
            "sub_core_grids",
            generated["attention"]["sdpa_program_config"],
        )
        self.assertEqual(
            generated["attention"]["sdpa_kernel_output_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )
        edge = candidate.search_space.edges["sdpa_to_concat_heads"]
        self.assertEqual(
            edge["producer_output_memory"]["runtime_name"],
            "L1_MEMORY_CONFIG",
        )
        self.assertEqual(edge["conversion"], "explicit")
        sub_core_candidate = next(
            item
            for item in enumeration.candidates
            if item.program.sub_core_grids is not None
        )
        sub_core_generated = sub_core_candidate.search_space.apply_to_runtime_config(
            self.runtime
        )
        self.assertEqual(
            sub_core_generated["attention"]["sdpa_program_config"]["sub_core_grids"],
            sub_core_candidate.program.sub_core_grids,
        )

    def test_cache_lengths_generate_distinct_chunk_sets_and_identities(self) -> None:
        observed = {}
        for cache_len in (128, 512, 1024):
            runtime = copy.deepcopy(self.runtime)
            runtime["max_cache_len"] = cache_len
            enumeration = self._enumerate(runtime)
            observed[cache_len] = {
                "k_chunks": {
                    candidate.program.k_chunk_size
                    for candidate in enumeration.candidates
                },
                "candidate_ids": {
                    candidate.candidate_id for candidate in enumeration.candidates
                },
            }

        self.assertEqual(observed[128]["k_chunks"], {0, 32, 64, 128})
        self.assertEqual(observed[512]["k_chunks"], {0, 32, 64, 128, 256})
        self.assertEqual(
            observed[1024]["k_chunks"],
            {0, 32, 64, 128, 256, 512},
        )
        self.assertTrue(
            observed[128]["candidate_ids"].isdisjoint(observed[512]["candidate_ids"])
        )
        self.assertTrue(
            observed[512]["candidate_ids"].isdisjoint(observed[1024]["candidate_ids"])
        )

    def test_every_legal_program_materializes_through_runtime_api(self) -> None:
        fake_ttnn = _fake_config_ttnn()
        enumeration = self._enumerate(self.runtime)

        for candidate in enumeration.candidates:
            with self.subTest(candidate=candidate.candidate_id):
                resolved = realize_ttnn_config(
                    candidate.program.to_runtime_descriptor(),
                    fake_ttnn,
                )
                self.assertEqual(resolved["constructor"], "sdpa")
                if candidate.program.sub_core_grids is None:
                    self.assertNotIn("sub_core_grids", resolved)
                else:
                    self.assertEqual(
                        resolved["sub_core_grids"][0],
                        "core_range_set",
                    )

    def test_candidate_runs_single_op_and_full_attention_smokes(self) -> None:
        candidate = next(
            item
            for item in self._enumerate(self.runtime).candidates
            if item.program.sub_core_grids is not None
            and item.kernel_output_memory.runtime_name == "L1_MEMORY_CONFIG"
        )
        runtime_config = candidate.runtime_config()
        fake_ttnn = _fake_runtime_ttnn()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primitive_report = run_smoke_attention_primitive(
                out=root / "single_op.json",
                primitive="paged_scaled_dot_product_attention_decode",
                device="p150a",
                batch_size=2,
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                max_cache_len=32,
                sdpa_runtime_config=runtime_config,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )
            self.assertTrue(primitive_report["passed"])

            program_dir = root / "program"
            program_dir.mkdir()
            (program_dir / "config.json").write_text(
                json.dumps(
                    {
                        "num_layers": 2,
                        "batch_size": 2,
                        "max_cache_len": 32,
                        "hidden_size": 16,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 4,
                    }
                )
            )
            layer_report = run_smoke_attention_layer(
                out=root / "full_attention.json",
                program_dir=program_dir,
                layer=0,
                device="p150a",
                sdpa_runtime_config=runtime_config,
                ttnn_module=fake_ttnn,
                torch_module=_fake_torch(),
            )
            self.assertTrue(layer_report["passed"])

        sdpa_calls = [
            call
            for call in fake_ttnn.calls
            if call["op"] == "paged_scaled_dot_product_attention_decode"
        ]
        self.assertEqual(len(sdpa_calls), 2)
        for call in sdpa_calls:
            with self.subTest(call=call):
                program = call["kwargs"]["program_config"]
                self.assertEqual(program["constructor"], "sdpa")
                self.assertEqual(program["sub_core_grids"][0], "core_range_set")
                self.assertEqual(
                    call["kwargs"]["memory_config"],
                    "ttnn.L1_MEMORY_CONFIG",
                )
        conversion = next(
            call
            for call in reversed(fake_ttnn.calls)
            if call["op"] == "to_memory_config"
        )
        self.assertEqual(
            conversion["kwargs"]["memory_config"],
            "sharded:(32, 128)",
        )

    def test_report_is_json_serializable(self) -> None:
        report = self._enumerate(self.runtime).to_dict()
        self.assertEqual(json.loads(json.dumps(report)), report)

    def _enumerate(self, runtime):
        return enumerate_sdpa_programs(
            base_space=self.space,
            workload=WorkloadSpec.from_runtime_config(runtime),
            device=self.device,
            precision_contract=self.precision,
        )


def _fake_runtime_ttnn():
    module = _fake_ttnn(with_create_sharded=True)
    module.CoreCoord = lambda x, y: (x, y)
    module.CoreRange = lambda start, end: ("core_range", start, end)
    module.CoreRangeSet = lambda ranges: (
        "core_range_set",
        tuple(sorted(ranges)),
    )

    def sdpa_program_config(**kwargs):
        return {"constructor": "sdpa", **kwargs}

    module.SDPAProgramConfig = sdpa_program_config
    return module


if __name__ == "__main__":
    unittest.main()
