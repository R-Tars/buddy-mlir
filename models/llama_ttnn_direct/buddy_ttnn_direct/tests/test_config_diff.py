from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.llama_ttnn_direct.buddy_ttnn_direct.cli import main
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.config_diff import (
    PARITY_SECTIONS,
    build_config_parity_view,
    default_official_config_path,
    diff_official_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.codegen.python_ttnn import (
    build_codegen_config,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.semantic.importer_hf_llama import (
    import_hf_llama,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.templates.registry import (
    build_execution_plan,
)


class ConfigDiffTest(unittest.TestCase):
    def test_normalized_parity_config_matches_itself(self) -> None:
        official = json.loads(default_official_config_path().read_text())

        diff = diff_official_config(official, official)

        self.assertEqual(diff["status"], "match")
        self.assertEqual(diff["summary"]["issue_count"], 0)
        self.assertGreater(diff["summary"]["matching_count"], 0)
        self.assertEqual(set(diff["sections"]), set(PARITY_SECTIONS))

    def test_generated_config_reports_known_official_gaps(self) -> None:
        ours = _fake_generated_config()
        official = json.loads(default_official_config_path().read_text())

        diff = diff_official_config(ours, official)

        self.assertEqual(diff["status"], "diff_found")
        missing_paths = {field["path"] for field in diff["missing_fields"]}
        mismatch_paths = {field["path"] for field in diff["mismatched_fields"]}
        self.assertIn("memory_config.attention_qkv", missing_paths)
        self.assertIn("program_config.attention_sdpa", missing_paths)
        self.assertIn("compute_fidelity.mlp", missing_paths)
        self.assertIn("core_grid.attention", missing_paths)
        self.assertIn("lm_head.argmax_strategy", mismatch_paths)
        self.assertIn("paged_attention.scale", mismatch_paths)
        self.assertEqual(
            diff["ours"]["parity_config"]["lm_head"]["split_count"],
            8,
        )

    def test_cli_diff_official_config_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ours = root / "config.json"
            out = root / "config_diff.json"
            ours.write_text(json.dumps(_fake_generated_config()))

            exit_code = main(
                [
                    "diff-official-config",
                    "--ours",
                    str(ours),
                    "--official",
                    str(default_official_config_path()),
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["status"], "diff_found")
            self.assertGreater(report["summary"]["issue_count"], 0)

    def test_build_config_parity_view_accepts_generated_config(self) -> None:
        view = build_config_parity_view(_fake_generated_config())

        self.assertEqual(view["source_format"], "generated_ttnn_direct_config")
        self.assertEqual(
            view["parity_config"]["dtype_recipe"]["recipe"],
            "official_like_performance_seed",
        )
        self.assertEqual(view["parity_config"]["lm_head"]["shard_count"], 8)


def _fake_generated_config() -> dict[str, object]:
    graph = import_hf_llama(
        "/tmp/fake-config-diff",
        config={
            "_name_or_path": "fake-config-diff",
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        },
        state_dict_metadata=[],
        mode="decode",
        batch_size=32,
        seq_len=1,
        max_cache_len=1024,
    )
    plan = build_execution_plan(
        graph,
        {
            "device": "p150a",
            "model": "llama3.1-8b",
            "batch_size": 32,
            "decode_seq_len": 1,
            "prefill_seq_len": 128,
            "max_cache_len": 1024,
            "attention_template": "official_paged_attention_decode",
            "mlp_template": "official_gated_mlp_decode",
            "lm_head_template": "official_split_lm_head",
            "kv_cache_template": "paged_kv_cache",
            "generation_template": "device_argmax_greedy",
            "lm_head_split_count": 8,
            "dtype_recipe": "official_like_performance_seed",
        },
    )
    return build_codegen_config(plan)


if __name__ == "__main__":
    unittest.main()
