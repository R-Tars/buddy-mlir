from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.autotune_profiler_audit import (
    BOTTLENECK_CLASSES,
    REGION_ORDER,
    ProfilerAuditError,
    _PROFILER_MEASUREMENT,
    _profiler_command,
    assign_decode_regions,
    build_autotune_profiler_audit,
    normalize_trace_timestamps,
    select_trace_replay,
)
from models.llama_ttnn_direct.buddy_ttnn_direct.runtime.steady_profile import (
    _install_segmented_prefill_profiler,
)


OFFSET_NS = 429_754_328_000.0
OP_CODES = """
UntilizeDeviceOperation EmbeddingsDeviceOperation InterleavedToShardedDeviceOperation
LayerNormDeviceOperation MatmulDeviceOperation NLPCreateQKVHeadsDecodeDeviceOperation
RotaryEmbeddingLlamaDeviceOperation RotaryEmbeddingLlamaDeviceOperation
PagedUpdateCacheDeviceOperation PagedUpdateCacheDeviceOperation SdpaDecodeDeviceOperation
InterleavedToShardedDeviceOperation NLPConcatHeadsDecodeDeviceOperation MatmulDeviceOperation
TilizeDeviceOperation BinaryNgDeviceOperation ReshardDeviceOperation LayerNormDeviceOperation
MatmulDeviceOperation MatmulDeviceOperation BinaryNgDeviceOperation MatmulDeviceOperation
BinaryNgDeviceOperation ReshardDeviceOperation LayerNormDeviceOperation MatmulDeviceOperation
ShardedToInterleavedDeviceOperation MatmulDeviceOperation ShardedToInterleavedDeviceOperation
ConcatDeviceOperation UntilizeDeviceOperation ArgMaxDeviceOperation CopyDeviceOperation
""".split()


def _row(index: int, code: str, trace: bool) -> dict[str, str]:
    duration = 10_000.0 + index
    matmul = code == "MatmulDeviceOperation"
    return {
        "OP CODE": code,
        "DEVICE KERNEL DURATION [ns]": str(OFFSET_NS + duration),
        "OP TO OP LATENCY [ns]": str(-OFFSET_NS + (index % 3)),
        "METAL TRACE ID": "7" if trace else "",
        "METAL TRACE REPLAY SESSION ID": "1" if trace else "",
        "CORE COUNT": "80" if matmul else "32",
        "AVAILABLE WORKER CORE COUNT": "110",
        "INPUT_0_MEMORY": "DEV_0_L1_WIDTH_SHARDED",
        "INPUT_1_MEMORY": "DEV_0_DRAM_INTERLEAVED" if matmul else "",
        "OUTPUT_0_MEMORY": "DEV_0_L1_WIDTH_SHARDED",
        "PM COMPUTE [ns]": "10" if matmul else "20",
        "PM BANDWIDTH [ns]": "100" if matmul else "20",
        "PM IDEAL [ns]": "100",
        "PM FPU UTIL (%)": "50",
        "DEVICE BRISC KERNEL DURATION [ns]": str(OFFSET_NS + duration / 2),
        "DEVICE NCRISC KERNEL DURATION [ns]": str(OFFSET_NS + duration / 3),
        "DEVICE TRISC0 KERNEL DURATION [ns]": str(OFFSET_NS + duration / 4),
        "DEVICE TRISC1 KERNEL DURATION [ns]": str(OFFSET_NS + duration / 4),
        "DEVICE TRISC2 KERNEL DURATION [ns]": str(OFFSET_NS + duration / 4),
        "PM REQ I BW": "[100.0]",
        "PM REQ O BW": "[50.0]",
    }


def _rows(trace: bool = True) -> list[dict[str, str]]:
    return [_row(index, code, trace) for index, code in enumerate(OP_CODES)]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, float]:
    rows = _rows()
    csv_path = tmp_path / "ops.csv"
    _write_csv(csv_path, rows)
    total_ms = sum(10_000.0 + index for index in range(len(rows))) / 1_000_000.0
    profile = {
        "status": "profiled",
        "passed": True,
        "mode": "decode-steady",
        "layers": 1,
        "batch_size": 32,
        "prefill_len": 256,
        "cache_len": 1024,
        "after_prefill": True,
        "warmup": 0,
        "iterations": 1,
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "prefill_execution_mode": "eager",
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
    program = tmp_path / "program"
    program.mkdir()
    dram = lambda count=1: {  # noqa: E731
        "programs": [{"program_family": "dram_sharded"} for _ in range(count)]
    }
    operators = {
        name: dram()
        for name in ("attention.qkv", "attention.o_proj", "mlp.gate", "mlp.up", "mlp.down")
    }
    operators.update(
        {
            "attention.sdpa": {"runtime_kind": "ttnn_sdpa_program_config"},
            "lm_head.shards": dram(2),
        }
    )
    config = {
        "num_layers": 1,
        "generation": {"template": "device_argmax_greedy", "mode": "greedy"},
        "autotune": {"operators": operators},
    }
    (program / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return csv_path, profile_path, program, total_ms


def test_select_normalize_and_assign_full_trace() -> None:
    rows, selection = select_trace_replay(_rows())
    total_ms = sum(10_000.0 + index for index in range(len(rows))) / 1_000_000.0
    normalized, clock = normalize_trace_timestamps(rows, e2e_latency_ms=total_ms)
    assignments = assign_decode_regions(normalized, num_layers=1, lm_head_shards=2)

    assert selection["selected_op_count"] == len(rows)
    assert clock["validation"] == "pass"
    assert clock["offset_inlier_ratio"] == 1.0
    assert clock["corrected_kernel_duration_count"] == len(rows)
    assert {item["region"] for item in assignments} >= set(REGION_ORDER)
    assert assignments[0]["region"] == assignments[-1]["region"] == "runtime_inputs"


def test_build_audit_writes_json_csv_and_budget_answers(tmp_path: Path) -> None:
    csv_path, profile_path, program, total_ms = _write_fixture(tmp_path)
    output = tmp_path / "audit" / "autotune_profiler_audit.json"
    report = build_autotune_profiler_audit(
        profiler_csv=csv_path,
        profile_report=profile_path,
        program_dir=program,
        out=output,
    )

    assert report["passed"] is True
    assert output.is_file() and output.with_suffix(".csv").is_file()
    assert report["latency"]["device_kernel_total_ms"] == pytest.approx(total_ms, abs=0.0001)
    assert report["clock_normalization"]["validation"] == "pass"
    assert set(report["budget_answers"]) == {
        "mlp", "attention_matmul", "sdpa", "lm_head", "kv_update_plus_rope"
    }
    assert all(value["percent_of_decode"] > 0 for value in report["budget_answers"].values())
    assert all(region["classification"] in BOTTLENECK_CLASSES for region in report["regions"])
    qkv = next(region for region in report["regions"] if region["region"] == "qkv_linear")
    assert (qkv["classification"], qkv["program_family"]) == (
        "dram_bound", ["dram_sharded"]
    )


def test_missing_trace_replay_is_a_classified_failure(tmp_path: Path) -> None:
    csv_path, profile_path, program, _ = _write_fixture(tmp_path)
    _write_csv(csv_path, _rows(trace=False))
    report = build_autotune_profiler_audit(
        profiler_csv=csv_path,
        profile_report=profile_path,
        program_dir=program,
        out=tmp_path / "failed.json",
    )
    assert report["passed"] is False
    assert "no complete Metal trace replay" in report["error"]


@pytest.mark.parametrize(
    ("scope", "field", "value", "failure"),
    [
        ("profile", "execution_mode", "eager", "execution_mode"),
        ("profile", "runtime_input_mode", "recreate", "runtime_input_mode"),
        ("profile", "after_prefill", False, "after_prefill"),
        ("runtime_inputs", "new_device_tensors_per_decode_step", 1, "new_device_tensors"),
        ("runtime_inputs", "host_to_device_updates_per_decode_step", 1, "host_to_device"),
        ("runtime_inputs", "page_table_reused", False, "page_table_reused"),
        ("runtime_inputs", "token_update", "host_copy", "token_update"),
        ("profile", "program_compile_count_after_capture", 1, "program_compile"),
        ("profile", "trace_execute_count", None, "trace_execute_count"),
        ("generation", "mode", "sampling", "force_argmax"),
    ],
)
def test_decode_profile_contract_fails_closed(
    tmp_path: Path,
    scope: str,
    field: str,
    value: object,
    failure: str,
) -> None:
    csv_path, profile_path, program, _ = _write_fixture(tmp_path)
    target_path = program / "config.json" if scope == "generation" else profile_path
    payload = json.loads(target_path.read_text())
    target = payload[scope] if scope in {"runtime_inputs", "generation"} else payload
    if value is None:
        target.pop(field)
    else:
        target[field] = value
    target_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_autotune_profiler_audit(
        profiler_csv=csv_path,
        profile_report=profile_path,
        program_dir=program,
        out=tmp_path / f"{failure}.json",
    )

    assert report["passed"] is False
    assert failure in report["error"]


def test_profiler_capture_uses_canonical_zero_by_one_by_one_contract() -> None:
    command = _profiler_command(
        program_dir=Path("/program"),
        model_path=Path("/model"),
        input_prompts=Path("/prompts.json"),
        tokenizer_path=None,
        instruct=False,
        device="p150a",
        device_id=0,
        profile_report=Path("/profile.json"),
        profiler_dir=Path("/profiler"),
    )
    options = dict(zip(command, command[1:]))

    assert _PROFILER_MEASUREMENT.to_dict() == {
        "schema_version": 1,
        "kind": "profiler_audit_capture",
        "scope": "post_prefill_steady_decode",
        "metric": "tokens_per_second_per_user",
        "warmup": 0,
        "iterations": 1,
        "repetitions": 1,
        "synchronize_device": True,
    }
    assert options["--warmup"] == "0"
    assert options["--iterations"] == "1"
    for option in ("--after-prefill", "--runtime-input-mode", "--execution-mode"):
        assert option in command


def test_assign_rejects_truncated_model_trace() -> None:
    normalized = [dict(row, _duration_ns=10_000.0) for row in _rows()[:-4]]
    with pytest.raises(ProfilerAuditError, match="expected"):
        assign_decode_regions(normalized, num_layers=1, lm_head_shards=2)


def test_segmented_prefill_flushes_after_each_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUDDY_TTNN_PROFILER_AUDIT", "1")
    calls: list[object] = []

    class Model:
        def prefill_layer(self, value: object) -> object:
            calls.append(("layer", value))
            return value

    class TTNN:
        def ReadDeviceProfiler(self, device: object) -> None:
            calls.append(("flush", device))

    model, device = Model(), object()
    _install_segmented_prefill_profiler(model, TTNN(), device)
    value = object()
    assert model.prefill_layer(value) is value
    assert calls == [("layer", value), ("flush", device)]
