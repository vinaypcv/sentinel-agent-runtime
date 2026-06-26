"""Tests for the Gemini/mock Dharma repair benchmark suite."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPOSITORY_ROOT / "benchmarks" / "benchmark_repair.py"


def load_repair_benchmark() -> ModuleType:
    """Load the repair benchmark script as a module."""
    spec = importlib.util.spec_from_file_location("benchmark_repair", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load repair benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_case_loading() -> None:
    """All ten controlled repair cases should load successfully."""
    benchmark = load_repair_benchmark()

    cases = benchmark.load_cases()

    assert len(cases) == 10
    assert {case.case_id for case in cases} == {
        "bounded_stack_capacity_bug",
        "off_by_one_bug",
        "missing_none_check",
        "wrong_sort_order",
        "wrong_exception_type",
        "mutation_while_iterating",
        "unsafe_eval_usage",
        "path_traversal_vulnerability",
        "missing_input_validation",
        "incorrect_aggregation_logic",
    }
    assert all(case.mock_patch.startswith("diff --git ") for case in cases)


def test_metadata_validation_failure(tmp_path: Path) -> None:
    """Malformed metadata should fail with a helpful validation message."""
    benchmark = load_repair_benchmark()
    case_dir = tmp_path / "broken_case"
    case_dir.mkdir()
    (case_dir / "metadata.json").write_text(
        json.dumps({"case_id": "broken_case"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing fields"):
        benchmark.load_cases(tmp_path)


def test_result_serialization_and_outputs(tmp_path: Path) -> None:
    """Repair benchmark reports should serialize to JSON and CSV."""
    benchmark = load_repair_benchmark()
    report = {
        "benchmark": "repair_reliability",
        "provider": "mock",
        "skipped": False,
        "results": [
            {
                "case_id": "case-one",
                "provider": "mock",
                "patch_generated": True,
                "valid_diff": True,
                "peg_passed": True,
                "static_checks_passed": True,
                "tests_passed": True,
                "rollback_triggered": False,
                "latency_seconds": 0.1,
                "error_message": None,
                "decision_json": {"decision": "pass", "score": 1.0},
            }
        ],
        "summary": {},
    }

    json_path, csv_path = benchmark.write_outputs(report, tmp_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert parsed["results"][0]["decision_json"]["decision"] == "pass"
    assert rows[0]["case_id"] == "case-one"
    assert json.loads(rows[0]["decision_json"])["score"] == 1.0


def test_mock_benchmark_execution(tmp_path: Path) -> None:
    """Mock provider should repair a small subset without live provider calls."""
    benchmark = load_repair_benchmark()

    report = benchmark.run_benchmark(provider="mock", max_cases=2)
    json_path, csv_path = benchmark.write_outputs(report, tmp_path)

    assert report["skipped"] is False
    assert report["summary"]["case_count"] == 2
    assert all(result["provider"] == "mock" for result in report["results"])
    assert all(result["patch_generated"] for result in report["results"])
    assert all(result["valid_diff"] for result in report["results"])
    assert all(result["peg_passed"] for result in report["results"])
    assert all(result["static_checks_passed"] for result in report["results"])
    assert all(result["tests_passed"] for result in report["results"])
    assert json_path.is_file()
    assert csv_path.is_file()


def test_gemini_benchmark_skipped_unless_live_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal tests must not call Gemini without explicit live-test env vars."""
    benchmark = load_repair_benchmark()
    monkeypatch.delenv("RUN_LIVE_API_TESTS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    report = benchmark.run_benchmark(provider="gemini", max_cases=1)

    assert report["skipped"] is True
    assert report["results"] == []
    assert "RUN_LIVE_API_TESTS=1" in report["skip_reason"]
