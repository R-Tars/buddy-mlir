from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import torch

from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    ACTIVATION_AXIS,
    DEFAULT_TEMPLATE_SELECTION,
    FUSED_PAGED_UPDATE,
    FUSED_QK_ROPE,
    GATE_LINEAR_FUSED_SILU,
    GATE_UP_AXIS,
    KV_UPDATE_AXIS,
    MUL_FUSED_SILU,
    PACKED_GATE_UP,
    ROPE_AXIS,
    SEPARATE_GATE_UP,
    SEPARATE_PAGED_UPDATE,
    SEPARATE_QK_ROPE,
    DeviceDescriptor,
    SearchSpaceConfig,
    TemplateSelectionError,
    WorkloadSpec,
    apply_template_selection,
    dry_run_template,
    evaluate_template_reference,
    list_template_definitions,
    normalize_template_selection,
    probe_template_availability,
    template_registry_schema,
    validate_candidate,
    validate_template_selection,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.templates.attention import (
    render_attention,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.compiler.templates.mlp import (
    render_mlp,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.tests.test_autotune_space import (
    _official_runtime_config,
)

EXPECTED_TEMPLATES = {
    SEPARATE_PAGED_UPDATE,
    FUSED_PAGED_UPDATE,
    SEPARATE_QK_ROPE,
    FUSED_QK_ROPE,
    MUL_FUSED_SILU,
    GATE_LINEAR_FUSED_SILU,
    SEPARATE_GATE_UP,
    PACKED_GATE_UP,
}


class ExistingAPITemplateRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = _official_runtime_config()

    def test_registry_has_all_eight_existing_api_templates(self) -> None:
        definitions = list_template_definitions()
        schema = template_registry_schema()

        self.assertEqual({item.name for item in definitions}, EXPECTED_TEMPLATES)
        self.assertEqual(schema["schema_version"], 1)
        self.assertEqual(
            set(schema["axes"]),
            {KV_UPDATE_AXIS, ROPE_AXIS, ACTIVATION_AXIS, GATE_UP_AXIS},
        )
        self.assertEqual(
            schema["axes"][GATE_UP_AXIS]["default"],
            SEPARATE_GATE_UP,
        )
        self.assertNotEqual(schema["axes"][GATE_UP_AXIS]["default"], PACKED_GATE_UP)
        json.dumps(schema)
        for definition in definitions:
            with self.subTest(template=definition.name):
                self.assertTrue(definition.required_api_groups)
                self.assertTrue(definition.reference_semantics)
                self.assertTrue(callable(definition.codegen_hook))
                self.assertTrue(callable(definition.legality_predicate))
                self.assertTrue(callable(definition.reference_evaluator))

    def test_every_template_has_a_device_free_dry_run(self) -> None:
        for name in sorted(EXPECTED_TEMPLATES):
            with self.subTest(template=name):
                report = dry_run_template(name, self.runtime)
                self.assertEqual(report["status"], "dry_run")
                self.assertTrue(report["dry_run"])
                self.assertTrue(report["legality"]["passed"])
                self.assertEqual(report["availability"]["status"], "not_probed")
                self.assertTrue(report["codegen"]["hook"])
                self.assertTrue(report["reference_semantics"])

    def test_availability_probe_resolves_only_public_ttnn_apis(self) -> None:
        ttnn = _fake_ttnn_api()
        for name in sorted(EXPECTED_TEMPLATES):
            with self.subTest(template=name):
                availability = probe_template_availability(name, ttnn)
                self.assertTrue(availability.available)
                self.assertFalse(availability.missing_api_groups)

        del ttnn.experimental.paged_fused_update_cache
        unavailable = probe_template_availability(FUSED_PAGED_UPDATE, ttnn)
        self.assertFalse(unavailable.available)
        self.assertEqual(
            unavailable.missing_api_groups,
            (("experimental.paged_fused_update_cache",),),
        )

    def test_legacy_aliases_normalize_to_phase3_names(self) -> None:
        normalized = normalize_template_selection(
            {
                KV_UPDATE_AXIS: "paged_fused_update_cache",
                ROPE_AXIS: SEPARATE_QK_ROPE,
                ACTIVATION_AXIS: MUL_FUSED_SILU,
                GATE_UP_AXIS: "separate_projection",
                "lm_head": "official_split_force_argmax",
            }
        )
        self.assertEqual(normalized[KV_UPDATE_AXIS], FUSED_PAGED_UPDATE)
        self.assertEqual(normalized[GATE_UP_AXIS], SEPARATE_GATE_UP)

    def test_incompatible_packed_partial_activation_is_rejected(self) -> None:
        selection = {
            **DEFAULT_TEMPLATE_SELECTION,
            GATE_UP_AXIS: PACKED_GATE_UP,
            ACTIVATION_AXIS: GATE_LINEAR_FUSED_SILU,
        }
        constraints = validate_template_selection(selection, self.runtime)
        self.assertEqual(
            [item.code for item in constraints],
            ["PACKED_GATE_UP_PARTIAL_ACTIVATION"],
        )
        with self.assertRaisesRegex(
            TemplateSelectionError, "PACKED_GATE_UP_PARTIAL_ACTIVATION"
        ):
            apply_template_selection(self.runtime, selection)

    def test_fused_attention_hooks_feed_phase2_legality(self) -> None:
        baseline = SearchSpaceConfig.from_runtime_config(self.runtime)
        payload = baseline.to_dict()
        payload["templates"][KV_UPDATE_AXIS] = FUSED_PAGED_UPDATE
        payload["templates"][ROPE_AXIS] = FUSED_QK_ROPE
        candidate = SearchSpaceConfig.from_dict(payload)
        configured = candidate.apply_to_runtime_config(self.runtime)
        workload = WorkloadSpec.from_runtime_config(configured)
        report = validate_candidate(
            candidate,
            workload,
            DeviceDescriptor.p150a(),
            candidate_id="fused-attention",
        )

        self.assertIsNotNone(workload.paged_fused_update)
        self.assertIsNotNone(workload.fused_qk_rope)
        self.assertTrue(report.passed)
        self.assertEqual(report.status, "static_legal")
        self.assertEqual(report.issues, ())
        self.assertEqual(
            configured["attention"]["op_sequence"][2:4],
            [
                "rotary_embedding_llama_fused_qk",
                "paged_fused_update_cache",
            ],
        )

    def test_packed_codegen_hook_doubles_gate_program_width(self) -> None:
        baseline = SearchSpaceConfig.from_runtime_config(self.runtime)
        payload = baseline.to_dict()
        payload["templates"][GATE_UP_AXIS] = PACKED_GATE_UP
        configured = SearchSpaceConfig.from_dict(payload).apply_to_runtime_config(
            self.runtime
        )
        mlp = configured["mlp"]
        self.assertEqual(
            mlp["packed_gate_up_program_config"]["per_core_N"],
            2 * mlp["gate_program_config"]["per_core_N"],
        )
        self.assertEqual(
            mlp["packed_gate_up_output_memory_config"],
            mlp["gate_output_memory_config"],
        )
        self.assertEqual(mlp["packed_gate_up_split_strategy"], "split")
        self.assertEqual(
            mlp["packed_gate_up_split_output_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )
        self.assertEqual(
            mlp["packed_gate_up_mul_input_memory_config"]["name"],
            "L1_MEMORY_CONFIG",
        )
        self.assertFalse(mlp["packed_gate_up_mul_conversion"])

    def test_gate_linear_silu_updates_final_sharded_program_configs(self) -> None:
        baseline = SearchSpaceConfig.from_runtime_config(self.runtime)
        payload = baseline.to_dict()
        payload["templates"][ACTIVATION_AXIS] = GATE_LINEAR_FUSED_SILU
        configured = SearchSpaceConfig.from_dict(payload).apply_to_runtime_config(
            self.runtime
        )

        self.assertEqual(
            configured["mlp"]["gate_program_config"]["fused_activation"],
            "silu",
        )
        self.assertEqual(
            configured["prefill"]["gate_program_config"]["fused_activation"],
            "silu",
        )
        self.assertEqual(
            configured["mlp"]["gate_linear_activation"],
            "silu",
        )

    def test_generated_source_contains_all_existing_api_hooks(self) -> None:
        source = render_attention() + render_mlp()
        for marker in (
            "rotary_embedding_fused_qk",
            "paged_fused_update_cache",
            "gate_linear_fused_silu",
            "gate_up_proj.weight",
            "split_last_dim",
            "mul_silu",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

    def test_reference_semantics_match_for_each_template_pair(self) -> None:
        key_cache = torch.zeros(2, 1, 4, 2)
        value_cache = torch.zeros_like(key_cache)
        key = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        value = key + 10
        cache_inputs = {
            "key_cache": key_cache,
            "value_cache": value_cache,
            "key": key,
            "value": value,
            "positions": [1, 2],
        }
        separate_cache = evaluate_template_reference(
            SEPARATE_PAGED_UPDATE, **cache_inputs
        )
        fused_cache = evaluate_template_reference(FUSED_PAGED_UPDATE, **cache_inputs)
        self.assertTrue(torch.equal(separate_cache[0], fused_cache[0]))
        self.assertTrue(torch.equal(separate_cache[1], fused_cache[1]))

        q = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        k = q + 1
        rope_inputs = {
            "q": q,
            "k": k,
            "cos": torch.tensor(2.0),
            "sin": torch.tensor(3.0),
            "transformation": torch.tensor(0.0),
            "rope": lambda tensor, cos, sin, _transform: tensor * cos + sin,
        }
        separate_rope = evaluate_template_reference(SEPARATE_QK_ROPE, **rope_inputs)
        fused_rope = evaluate_template_reference(FUSED_QK_ROPE, **rope_inputs)
        self.assertTrue(torch.equal(separate_rope[0], fused_rope[0]))
        self.assertTrue(torch.equal(separate_rope[1], fused_rope[1]))

        torch.manual_seed(7)
        hidden = torch.randn(2, 3)
        gate_weight = torch.randn(3, 4)
        up_weight = torch.randn(3, 4)
        activation_inputs = {
            "hidden": hidden,
            "gate_weight": gate_weight,
            "up_weight": up_weight,
            "silu": torch.nn.functional.silu,
        }
        mul_fused = evaluate_template_reference(MUL_FUSED_SILU, **activation_inputs)
        gate_fused = evaluate_template_reference(
            GATE_LINEAR_FUSED_SILU, **activation_inputs
        )
        torch.testing.assert_close(mul_fused, gate_fused)

        projection_inputs = {
            "hidden": hidden,
            "gate_weight": gate_weight,
            "up_weight": up_weight,
            "cat": torch.cat,
        }
        separate_gate_up = evaluate_template_reference(
            SEPARATE_GATE_UP, **projection_inputs
        )
        packed_gate_up = evaluate_template_reference(
            PACKED_GATE_UP, **projection_inputs
        )
        torch.testing.assert_close(separate_gate_up[0], packed_gate_up[0])
        torch.testing.assert_close(separate_gate_up[1], packed_gate_up[1])


def _fake_ttnn_api() -> SimpleNamespace:
    operation = lambda *args, **kwargs: (args, kwargs)
    return SimpleNamespace(
        experimental=SimpleNamespace(
            paged_update_cache=operation,
            paged_fused_update_cache=operation,
            rotary_embedding_llama=operation,
            rotary_embedding_llama_fused_qk=operation,
        ),
        linear=operation,
        mul=operation,
        split=operation,
        UnaryWithParam=operation,
        UnaryOpType=SimpleNamespace(SILU="silu"),
    )


if __name__ == "__main__":
    unittest.main()
