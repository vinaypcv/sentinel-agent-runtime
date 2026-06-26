"""Benchmark Dharma Loop repair reliability across controlled Python bug cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail  # noqa: E402
from brahman_os.ledger.karma_ledger import KarmaLedger  # noqa: E402
from brahman_os.repair.llm_patch_generator import (  # noqa: E402
    GeminiPatchGenerator,
    MockPatchGenerator,
    PatchGenerationError,
    PatchGenerator,
)
from brahman_os.repair.patcher import Patcher  # noqa: E402
from brahman_os.repair.test_runner import TestRunner  # noqa: E402
from brahman_os.schemas import PolicyRule, SafetyDecision, VivekaDecision  # noqa: E402

CASES_DIR = REPO_ROOT / "benchmarks" / "repair_cases"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results"
JSON_OUTPUT_NAME = "repair_results.json"
CSV_OUTPUT_NAME = "repair_results.csv"


@dataclass(frozen=True, slots=True)
class RepairCase:
    """One controlled benchmark repair case."""

    case_id: str
    title: str
    case_dir: Path
    source_file: str
    test_file: str
    expected_behavior_file: str
    expected_files_changed: tuple[str, ...]
    mock_patch: str


@dataclass(frozen=True, slots=True)
class RepairBenchmarkResult:
    """Serializable result for one provider and repair case."""

    case_id: str
    provider: str
    patch_generated: bool
    valid_diff: bool
    peg_passed: bool
    static_checks_passed: bool
    tests_passed: bool
    rollback_triggered: bool
    latency_seconds: float
    error_message: str | None
    decision_json: dict[str, Any]


def gemini_live_enabled() -> bool:
    """Return whether live Gemini repair generation is explicitly enabled."""
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return os.environ.get("RUN_LIVE_API_TESTS") == "1" and has_key


def load_cases(cases_dir: Path = CASES_DIR) -> list[RepairCase]:
    """Load and validate repair case metadata and required fixture files."""
    cases: list[RepairCase] = []
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"missing metadata.json for case: {case_dir.name}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed metadata for case {case_dir.name}: {error}") from error
        cases.append(validate_case_metadata(case_dir, metadata))
    if not cases:
        raise ValueError(f"no repair cases found in {cases_dir}")
    return cases


def validate_case_metadata(case_dir: Path, metadata: dict[str, Any]) -> RepairCase:
    """Validate one metadata document and return a typed case."""
    required = {
        "case_id",
        "title",
        "source_file",
        "test_file",
        "expected_behavior_file",
        "expected_files_changed",
        "mock_patch",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"{case_dir.name} metadata missing fields: {sorted(missing)}")

    case_id = require_string(metadata, "case_id", case_dir)
    if case_id != case_dir.name:
        raise ValueError(f"{case_dir.name} metadata case_id must match directory name")

    expected_files_changed = metadata["expected_files_changed"]
    if not isinstance(expected_files_changed, list) or not all(
        isinstance(item, str) and item for item in expected_files_changed
    ):
        raise ValueError(f"{case_id} expected_files_changed must be a list of strings")

    case = RepairCase(
        case_id=case_id,
        title=require_string(metadata, "title", case_dir),
        case_dir=case_dir,
        source_file=require_string(metadata, "source_file", case_dir),
        test_file=require_string(metadata, "test_file", case_dir),
        expected_behavior_file=require_string(metadata, "expected_behavior_file", case_dir),
        expected_files_changed=tuple(expected_files_changed),
        mock_patch=require_string(metadata, "mock_patch", case_dir),
    )
    for relative_path in (
        case.source_file,
        case.test_file,
        case.expected_behavior_file,
    ):
        if not (case_dir / relative_path).is_file():
            raise ValueError(f"{case_id} missing required file: {relative_path}")
    return case


def require_string(metadata: dict[str, Any], field: str, case_dir: Path) -> str:
    """Read a required non-empty string metadata field."""
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_dir.name} metadata field {field} must be non-empty")
    return value


def select_cases(
    cases: list[RepairCase],
    *,
    case_selector: str,
    max_cases: int | None,
) -> list[RepairCase]:
    """Select requested benchmark cases."""
    selected = cases if case_selector == "all" else [
        case for case in cases if case.case_id == case_selector
    ]
    if not selected:
        raise ValueError(f"unknown repair case: {case_selector}")
    return selected[:max_cases] if max_cases is not None else selected


def build_generator(case: RepairCase, provider: str) -> PatchGenerator:
    """Create a patch generator for one case and provider."""
    if provider == "mock":
        return MockPatchGenerator(case.mock_patch)
    if provider == "gemini":
        if not gemini_live_enabled():
            raise PatchGenerationError(
                "Gemini benchmark requires RUN_LIVE_API_TESTS=1 and GEMINI_API_KEY "
                "or GOOGLE_API_KEY"
            )
        return GeminiPatchGenerator()
    raise ValueError("provider must be mock or gemini")


def ensure_git_apply_header(patch: str, *, file_path: str) -> str:
    """Add a git diff header when a provider returns plain unified diff output."""
    if patch.startswith("diff --git "):
        return patch
    if patch.startswith("--- "):
        return f"diff --git a/{file_path} b/{file_path}\n{patch}"
    return patch


def prepare_workspace(case: RepairCase, workspace: Path) -> None:
    """Copy a case into an isolated temporary repair workspace."""
    shutil.copy2(case.case_dir / case.source_file, workspace / case.source_file)
    shutil.copy2(case.case_dir / case.test_file, workspace / case.test_file)
    for relative_path in (case.source_file, case.test_file):
        target = workspace / relative_path
        normalized = target.read_text(encoding="utf-8").replace("\r\n", "\n")
        target.write_text(normalized, encoding="utf-8", newline=os.linesep)
    (workspace / "pyproject.toml").write_text(
        (
            "[tool.ruff]\n"
            'target-version = "py311"\n\n'
            "[tool.mypy]\n"
            'python_version = "3.11"\n'
            "follow_imports = \"skip\"\n"
            "strict = true\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(("git", "init"), cwd=workspace, capture_output=True, text=True, check=False)
    subprocess.run(
        ("git", "config", "core.autocrlf", "false"),
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ("git", "add", case.source_file, case.test_file, "pyproject.toml"),
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )


def evaluate_with_peg(
    *,
    tests_passed: bool,
    static_checks_passed: bool,
    valid_diff: bool,
) -> VivekaDecision:
    """Produce the final PEG approval decision for one repair attempt."""
    rule = PolicyRule(
        rule_id="repair-benchmark-verification",
        description="Patch, static checks, and regression tests must all pass.",
        condition="valid diff, static checks, and tests pass",
        decision=SafetyDecision.PASS,
        confidence=1.0,
    )
    evidence_vector = torch.tensor(
        [float(valid_diff), float(static_checks_passed), float(tests_passed)],
        dtype=torch.float32,
    )
    return PramanaEpistemicGuardrail(threshold=0.7).evaluate(
        evidence_vector,
        torch.ones(3, dtype=torch.float32),
        rules=(rule,),
        rule_results={
            rule.rule_id: valid_diff and static_checks_passed and tests_passed,
        },
    )


def run_case(case: RepairCase, provider: str) -> RepairBenchmarkResult:
    """Run one repair case through generation, patching, checks, tests, and PEG."""
    started = time.perf_counter()
    patch_generated = False
    valid_diff = False
    static_checks_passed = False
    tests_passed = False
    rollback_triggered = False
    error_message: str | None = None
    decision = evaluate_with_peg(
        tests_passed=False,
        static_checks_passed=False,
        valid_diff=False,
    )

    with tempfile.TemporaryDirectory(prefix=f"brahman-repair-{case.case_id}-") as directory:
        workspace = Path(directory)
        try:
            prepare_workspace(case, workspace)
            runner = TestRunner(workspace)
            initial_test = runner.run((case.test_file,))
            generator = build_generator(case, provider)
            patch = generator.generate(
                failing_file_content=(workspace / case.source_file).read_text(
                    encoding="utf-8"
                ),
                pytest_traceback="\n".join(initial_test.traceback_summary),
                test_file_content=(workspace / case.test_file).read_text(encoding="utf-8"),
                constraints=(
                    f"Modify only: {', '.join(case.expected_files_changed)}.",
                    (case.case_dir / case.expected_behavior_file).read_text(
                        encoding="utf-8"
                    ),
                    "Return a unified diff only.",
                ),
            )
            patch = ensure_git_apply_header(patch, file_path=case.source_file)
            patch_generated = True

            ledger = KarmaLedger(workspace / "karma-ledger.jsonl")
            patcher = Patcher(workspace, ledger)
            patcher.validate_patch(patch)
            valid_diff = True
            patch_result = patcher.apply(
                patch,
                goal_id="repair-benchmark",
                issue_id=case.case_id,
            )
            rollback_triggered = patch_result.rolled_back
            static_checks_passed = all(check.passed for check in patch_result.quality_checks)
            final_test = runner.run((case.test_file,))
            tests_passed = final_test.passed
        except Exception as error:
            error_message = str(error)
        finally:
            decision = evaluate_with_peg(
                tests_passed=tests_passed,
                static_checks_passed=static_checks_passed,
                valid_diff=valid_diff,
            )

    return RepairBenchmarkResult(
        case_id=case.case_id,
        provider=provider,
        patch_generated=patch_generated,
        valid_diff=valid_diff,
        peg_passed=decision.decision is SafetyDecision.PASS,
        static_checks_passed=static_checks_passed,
        tests_passed=tests_passed,
        rollback_triggered=rollback_triggered,
        latency_seconds=time.perf_counter() - started,
        error_message=error_message,
        decision_json=decision.model_dump(mode="json"),
    )


def run_benchmark(
    *,
    provider: str,
    case_selector: str = "all",
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Run the repair benchmark and return an explainable JSON report."""
    if provider == "gemini" and not gemini_live_enabled():
        return {
            "benchmark": "repair_reliability",
            "provider": provider,
            "skipped": True,
            "skip_reason": (
                "Gemini benchmark requires RUN_LIVE_API_TESTS=1 and GEMINI_API_KEY "
                "or GOOGLE_API_KEY"
            ),
            "results": [],
            "summary": summarize_results([]),
        }

    cases = select_cases(
        load_cases(),
        case_selector=case_selector,
        max_cases=max_cases,
    )
    results = [run_case(case, provider) for case in cases]
    return {
        "benchmark": "repair_reliability",
        "provider": provider,
        "skipped": False,
        "case_count": len(results),
        "results": [asdict(result) for result in results],
        "summary": summarize_results(results),
    }


def summarize_results(results: list[RepairBenchmarkResult]) -> dict[str, Any]:
    """Summarize repair benchmark outcomes."""
    if not results:
        return {
            "case_count": 0,
            "patch_generated_rate": 0.0,
            "valid_diff_rate": 0.0,
            "peg_pass_rate": 0.0,
            "static_checks_pass_rate": 0.0,
            "tests_pass_rate": 0.0,
            "rollback_rate": 0.0,
            "average_latency_seconds": 0.0,
        }
    total = len(results)
    return {
        "case_count": total,
        "patch_generated_rate": sum(result.patch_generated for result in results) / total,
        "valid_diff_rate": sum(result.valid_diff for result in results) / total,
        "peg_pass_rate": sum(result.peg_passed for result in results) / total,
        "static_checks_pass_rate": (
            sum(result.static_checks_passed for result in results) / total
        ),
        "tests_pass_rate": sum(result.tests_passed for result in results) / total,
        "rollback_rate": sum(result.rollback_triggered for result in results) / total,
        "average_latency_seconds": (
            sum(result.latency_seconds for result in results) / total
        ),
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write repair benchmark JSON and CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_OUTPUT_NAME
    csv_path = output_dir / CSV_OUTPUT_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    rows = report.get("results", [])
    if not isinstance(rows, list):
        raise TypeError("repair benchmark report results must be a list")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "provider",
                "patch_generated",
                "valid_diff",
                "peg_passed",
                "static_checks_passed",
                "tests_passed",
                "rollback_triggered",
                "latency_seconds",
                "error_message",
                "decision_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow(
                {
                    **row,
                    "decision_json": json.dumps(row.get("decision_json", {}), sort_keys=True),
                }
            )
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--case", default="all", help="Case ID or 'all'.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and print output paths as JSON."""
    args = parse_args()
    report = run_benchmark(
        provider=args.provider,
        case_selector=args.case,
        max_cases=args.max_cases,
    )
    json_path, csv_path = write_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "skipped": report.get("skipped", False),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
