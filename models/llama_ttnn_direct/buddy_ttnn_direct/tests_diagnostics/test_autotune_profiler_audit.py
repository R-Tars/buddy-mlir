from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.autotune_profiler_audit import (
    BOTTLENECK_CLASSES,
    REGION_ORDER,
    ProfilerAuditError,
    assign_decode_regions,
    build_autotune_profiler_audit,
    normalize_trace_timestamps,
    select_trace_replay,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.steady_profile import (
    _install_segmented_prefill_profiler,
)

OFFSET_NS = 429_754_328_000.0


def _op_codes() -> list[str]:
    return [
        "UntilizeDeviceOperation",
        "EmbeddingsDeviceOperation",
        "InterleavedToShardedDeviceOperation",
        "LayerNormDeviceOperation",
        "MatmulDeviceOperation",
        "NLPCreateQKVHeadsDecodeDeviceOperation",
        "RotaryEmbeddingLlamaDeviceOperation",
        "RotaryEmbeddingLlamaDeviceOperation",
        "PagedUpdateCacheDeviceOperation",
        "PagedUpdateCacheDeviceOperation",
        "SdpaDecodeDeviceOperation",
        "InterleavedToShardedDeviceOperation",
        "NLPConcatHeadsDecodeDeviceOperation",
        "MatmulDeviceOperation",
        "TilizeDeviceOperation",
        "BinaryNgDeviceOperation",
        "ReshardDeviceOperation",
        "LayerNormDeviceOperation",
        "MatmulDeviceOperation",
        "MatmulDeviceOperation",
        "BinaryNgDeviceOperation",
        "MatmulDeviceOperation",
        "BinaryNgDeviceOperation",
        "ReshardDeviceOperation",
        "LayerNormDeviceOperation",
        "MatmulDeviceOperation",
        "ShardedToInterleavedDeviceOperation",
        "MatmulDeviceOperation",
        "ShardedToInterleavedDeviceOperation",
        "ConcatDeviceOperation",
        "UntilizeDeviceOperation",
        "ArgMaxDeviceOperation",
        "CopyDeviceOperation",
    ]


def _rows(*, trace: bool = True) -> list[dict[str, str]]:
    rows = []
    for index, code in enumerate(_op_codes()):
        duration_ns = 10_000.0 + index
        is_matmul = code == "MatmulDeviceOperation"
        rows.append(
            {
                "OP CODE": code,
                "DEVICE KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns),
                "OP TO OP LATENCY [ns]": str(-OFFSET_NS + (index % 3)),
                "METAL TRACE ID": "7" if trace else "",
                "METAL TRACE REPLAY SESSION ID": "1" if trace else "",
                "CORE COUNT": "80" if is_matmul else "32",
                "AVAILABLE WORKER CORE COUNT": "110",
                "INPUT_0_MEMORY": "DEV_0_L1_WIDTH_SHARDED",
                "INPUT_1_MEMORY": ("DEV_0_DRAM_INTERLEAVED" if is_matmul else ""),
                "OUTPUT_0_MEMORY": "DEV_0_L1_WIDTH_SHARDED",
                "PM COMPUTE [ns]": "10" if is_matmul else "20",
                "PM BANDWIDTH [ns]": "100" if is_matmul else "20",
                "PM IDEAL [ns]": "100",
                "PM FPU UTIL (%)": "50",
                "DEVICE BRISC KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns / 2),
                "DEVICE NCRISC KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns / 3),
                "DEVICE TRISC0 KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns / 4),
                "DEVICE TRISC1 KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns / 4),
                "DEVICE TRISC2 KERNEL DURATION [ns]": str(OFFSET_NS + duration_ns / 4),
                "PM REQ I BW": "[100.0]",
                "PM REQ O BW": "[50.0]",
            }
        )
    return rows


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, float]:
    rows = _rows()
    csv_path = tmp_path / "ops.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total_ms = sum(10_000.0 + index for index in range(len(rows))) / 1_000_000.0
    profile = {
        "status": "profiled",
        "layers": 1,
        "batch_size": 32,
        "prefill_len": 256,
        "cache_len": 1024,
        "after_prefill": True,
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "persistent_input_count": 7,
        "trace_capture_count": 1,
        "trace_execute_count": 1,
        "program_compile_count_after_capture": 0,
        "decode_step_ms_p50": total_ms,
        "trace_key": {"layer_count": 1, "batch_size": 32},
        "runtime_inputs": {
            "new_device_tensors_per_decode_step": 0,
            "host_to_device_updates_per_decode_step": 0,
            "page_table_reused": True,
            "token_update": "captured_device_to_device_copy",
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    program_dir = tmp_path / "program"
    program_dir.mkdir()
    config = {
        "num_layers": 1,
        "generation": {
            "template": "device_argmax_greedy",
            "mode": "greedy",
        },
        "autotune": {
            "operators": {
                "attention.qkv": {"programs": [{"program_family": "dram_sharded"}]},
                "attention.o_proj": {"programs": [{"program_family": "dram_sharded"}]},
                "attention.sdpa": {"runtime_kind": "ttnn_sdpa_program_config"},
                "mlp.gate": {"programs": [{"program_family": "dram_sharded"}]},
                "mlp.up": {"programs": [{"program_family": "dram_sharded"}]},
                "mlp.down": {"programs": [{"program_family": "dram_sharded"}]},
                "lm_head.shards": {
                    "programs": [
                        {"program_family": "dram_sharded"},
                        {"program_family": "dram_sharded"},
                    ]
                },
            }
        },
    }
    (program_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return csv_path, profile_path, program_dir, total_ms


def test_select_normalize_and_assign_full_trace() -> None:
    rows, selection = select_trace_replay(_rows())
    total_ms = sum(10_000.0 + index for index in range(len(rows))) / 1_000_000.0
    normalized, clock = normalize_trace_timestamps(
        rows,
        e2e_latency_ms=total_ms,
    )
    assignments = assign_decode_regions(
        normalized,
        num_layers=1,
        lm_head_shards=2,
    )

    assert selection["selected_op_count"] == len(rows)
    assert clock["validation"] == "pass"
    assert clock["offset_inlier_ratio"] == 1.0
    assert clock["corrected_kernel_duration_count"] == len(rows)
    assert {item["region"] for item in assignments} >= set(REGION_ORDER)
    assert assignments[0]["region"] == "runtime_inputs"
    assert assignments[-1]["region"] == "runtime_inputs"


def test_build_audit_writes_json_csv_and_budget_answers(tmp_path: Path) -> None:
    csv_path, profile_path, program_dir, total_ms = _write_fixture(tmp_path)
    output = tmp_path / "audit" / "autotune_profiler_audit.json"

    report = build_autotune_profiler_audit(
        profiler_csv=csv_path,
        profile_report=profile_path,
        program_dir=program_dir,
        out=output,
    )

    assert report["passed"] is True
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert report["latency"]["device_kernel_total_ms"] == pytest.approx(
        total_ms, abs=0.0001
    )
    assert report["clock_normalization"]["validation"] == "pass"
    assert set(report["budget_answers"]) == {
        "mlp",
        "attention_matmul",
        "sdpa",
        "lm_head",
        "kv_update_plus_rope",
    }
    assert all(
        answer["percent_of_decode"] > 0.0
        for answer in report["budget_answers"].values()
    )
    assert all(
        region["classification"] in BOTTLENECK_CLASSES for region in report["regions"]
    )
    qkv = next(
        region for region in report["regions"] if region["region"] == "qkv_linear"
    )
    assert qkv["classification"] == "dram_bound"
    assert qkv["program_family"] == ["dram_sharded"]


def test_missing_trace_replay_is_a_classified_failure(tmp_path: Path) -> None:
    csv_path, profile_path, program_dir, _ = _write_fixture(tmp_path)
    rows = _rows(trace=False)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = build_autotune_profiler_audit(
        profiler_csv=csv_path,
        profile_report=profile_path,
        program_dir=program_dir,
        out=tmp_path / "failed.json",
    )

    assert report["passed"] is False
    assert "no complete Metal trace replay" in report["error"]


def test_assign_rejects_truncated_model_trace() -> None:
    rows = _rows()[:-4]
    normalized = [dict(row, _duration_ns=10_000.0) for row in rows]

    with pytest.raises(ProfilerAuditError, match="expected"):
        assign_decode_regions(normalized, num_layers=1, lm_head_shards=2)


def test_segmented_prefill_flushes_after_each_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BUDDY_TTNN_PROFILER_AUDIT", "1")
    calls: list[object] = []

    class Model:
        def prefill_layer(self, value: object) -> object:
            calls.append(("layer", value))
            return value

    class TTNN:
        def ReadDeviceProfiler(self, device: object) -> None:
            calls.append(("flush", device))

    model = Model()
    device = object()
    _install_segmented_prefill_profiler(model, TTNN(), device)

    value = object()
    assert model.prefill_layer(value) is value
    assert calls == [("layer", value), ("flush", device)]
