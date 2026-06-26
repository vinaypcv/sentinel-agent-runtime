"""Tests for guardrail benchmark generation and result export."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_benchmark_module() -> ModuleType:
    """Load the benchmark script as a module from its file path."""
    module_path = REPOSITORY_ROOT / "benchmarks" / "benchmark_guardrails.py"
    spec = importlib.util.spec_from_file_location("benchmark_guardrails", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_dataset_has_expected_categories() -> None:
    """The synthetic benchmark should contain the requested 80 claims."""
    benchmark = load_benchmark_module()
    dataset = benchmark.build_dataset()
    counts: dict[str, int] = {}
    for claim in dataset:
        counts[claim.category] = counts.get(claim.category, 0) + 1

    assert len(dataset) == 80
    assert counts == {
        "safe": 20,
        "unsafe_dosage": 20,
        "contradiction": 20,
        "irrelevant": 20,
    }
    assert sum(1 for claim in dataset if claim.should_block) == 60


def test_run_benchmark_returns_explainable_json_shape() -> None:
    """Benchmark output should include all modes, metrics, and predictions."""
    benchmark = load_benchmark_module()
    report = benchmark.run_benchmark()

    json.dumps(report)
    assert report["dataset"]["total_claims"] == 80
    assert set(report["modes"]) == {
        "llm_only_mock_baseline",
        "vector_similarity_only",
        "symbolic_rules_only",
        "sams_peg_brahman_os",
    }
    for mode_name, payload in report["modes"].items():
        assert payload["label"] == benchmark.MODE_LABELS[mode_name]
        assert len(payload["predictions"]) == 80
        assert set(payload["metrics"]) == {
            "block_rate",
            "false_positive_rate",
            "false_negative_rate",
            "average_viveka_score",
            "latency_ms",
        }


def test_write_outputs_creates_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark export should write parseable JSON and CSV files."""
    benchmark = load_benchmark_module()
    json_output = tmp_path / "guardrail_results.json"
    csv_output = tmp_path / "guardrail_results.csv"
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(benchmark, "JSON_OUTPUT", json_output)
    monkeypatch.setattr(benchmark, "CSV_OUTPUT", csv_output)

    report = benchmark.run_benchmark()
    benchmark.write_outputs(report)

    parsed = json.loads(json_output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_output.open(encoding="utf-8")))

    assert parsed["benchmark"] == "guardrail_modes"
    assert len(rows) == 4
    assert rows[0]["mode"]
    assert rows[0]["label"]


def test_write_outputs_rejects_malformed_report() -> None:
    """Malformed benchmark reports should fail clearly instead of writing junk."""
    benchmark = load_benchmark_module()
    with pytest.raises(TypeError, match="modes must be a dictionary"):
        benchmark.write_outputs({"modes": []})
