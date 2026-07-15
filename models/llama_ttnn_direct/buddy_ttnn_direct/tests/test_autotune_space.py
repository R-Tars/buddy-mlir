from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    CoreGrid,
    MatmulProgramConfig,
    MemoryConfig,
    SDPAProgramConfig,
    SearchSpaceConfig,
    SpaceSchemaError,
    adapt_legacy_presets,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.config import (
    build_codegen_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.tuning import (
    apply_runtime_tuning,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
    load_template_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class StructuredAutotuneSpaceTest(unittest.TestCase):
    def test_official_runtime_round_trip_is_lossless(self) -> None:
        runtime = _official_runtime_config()
        space = SearchSpaceConfig.from_runtime_config(runtime)
        serialized = json.loads(json.dumps(space.to_dict()))
        restored = SearchSpaceConfig.from_dict(serialized)
        generated = restored.apply_to_runtime_config(runtime)

        self.assertEqual(_without_autotune(generated), runtime)
        self.assertEqual(generated["autotune"], serialized)
        self.assertEqual(generated["template_config"]["autotune"], serialized)
        self.assertEqual(
            set(serialized["operators"]),
            {
                "attention.qkv",
                "attention.o_proj",
                "attention.sdpa",
                "mlp.gate",
                "mlp.up",
                "mlp.down",
                "lm_head.shards",
            },
        )
        self.assertGreaterEqual(len(serialized["memory_configs"]), 20)
        self.assertGreaterEqual(len(serialized["core_grids"]), 20)
        self.assertIn(
            "prefill.qkv_program_config",
            serialized["extra_program_configs"],
        )
        self.assertEqual(
            set(serialized["edges"]),
            {
                "sdpa_to_concat_heads",
                "final_norm_to_lm_head",
                "lm_head_shards_to_concat",
            },
        )

    def test_schema_v2_directly_generates_the_current_runtime_config(self) -> None:
        runtime = _official_runtime_config()
        space = SearchSpaceConfig.from_runtime_config(runtime)
        generated = apply_runtime_tuning(runtime, space.to_dict())
        self.assertEqual(_without_autotune(generated), runtime)
        self.assertNotIn("memory_layout", generated["autotune"])
        self.assertNotIn("program_config", generated["autotune"])

    def test_legacy_presets_adapt_to_structured_fields(self) -> None:
        runtime = _official_runtime_config()
        space = adapt_legacy_presets(
            runtime,
            memory_layout="lm_head_dram_concat",
            program_config="sdpa_grid_8x4",
        )
        generated = space.apply_to_runtime_config(runtime)
        self.assertEqual(
            generated["attention"]["sdpa_program_config"]["core_grid"],
            [8, 4],
        )
        self.assertEqual(
            generated["lm_head"]["shard_output_memory_config"]["name"],
            "DRAM_MEMORY_CONFIG",
        )
        self.assertEqual(
            generated["lm_head"]["concat_memory_config"]["name"],
            "DRAM_MEMORY_CONFIG",
        )
        self.assertEqual(generated["autotune"]["schema_version"], 2)

    def test_runtime_descriptors_round_trip_unknown_fields(self) -> None:
        memory_descriptor = {
            "kind": "ttnn_sharded_memory_config",
            "strategy": "height",
            "core_grid": [8, 4],
            "shard_shape": [32, 128],
            "orientation": "row_major",
            "vendor_extension": 7,
        }
        self.assertEqual(
            MemoryConfig.from_runtime_descriptor(
                memory_descriptor
            ).to_runtime_descriptor(),
            memory_descriptor,
        )

        matmul_descriptor = {
            "kind": "ttnn_matmul_multicore_reuse_mcast_program_config",
            "core_grid": [8, 8],
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 4,
            "per_core_M": 1,
            "per_core_N": 16,
            "transpose_mcast": False,
            "fuse_batch": False,
            "vendor_extension": "kept",
        }
        self.assertEqual(
            MatmulProgramConfig.from_runtime_descriptor(
                matmul_descriptor
            ).to_runtime_descriptor(),
            matmul_descriptor,
        )

        sdpa_descriptor = {
            "kind": "ttnn_sdpa_program_config",
            "core_grid": [8, 8],
            "sub_core_grids": [[0, 0, 7, 3]],
            "q_chunk_size": 0,
            "k_chunk_size": 32,
            "exp_approx_mode": False,
            "max_cores_per_head_batch": 16,
            "vendor_extension": True,
        }
        self.assertEqual(
            SDPAProgramConfig.from_runtime_descriptor(
                sdpa_descriptor
            ).to_runtime_descriptor(),
            sdpa_descriptor,
        )

    def test_tunable_identity_excludes_frozen_sdpa_approximation(self) -> None:
        runtime = _official_runtime_config()
        tunable = SearchSpaceConfig.from_runtime_config(runtime).tunable_dict()
        self.assertNotIn("exp_approx_mode", tunable["operators"]["attention.sdpa"])
        self.assertEqual(
            runtime["attention"]["sdpa_program_config"]["exp_approx_mode"],
            False,
        )

    def test_invalid_grid_and_program_family_are_rejected(self) -> None:
        with self.assertRaises(SpaceSchemaError):
            CoreGrid(0, 8)
        with self.assertRaisesRegex(SpaceSchemaError, "unsupported matmul"):
            MatmulProgramConfig.from_dict(
                {
                    "kind": "matmul",
                    "program_family": "imaginary",
                    "runtime_kind": "imaginary",
                    "compute_grid": [8, 8],
                }
            )


def _official_runtime_config() -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_root = Path(tmpdir)
        (model_root / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": "fake-llama-space",
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
        plan = build_execution_plan(graph, seed)
        return build_codegen_config(plan)


def _without_autotune(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("autotune", None)
    template_config = result.get("template_config")
    if isinstance(template_config, dict):
        template_config.pop("autotune", None)
    return result


if __name__ == "__main__":
    unittest.main()
