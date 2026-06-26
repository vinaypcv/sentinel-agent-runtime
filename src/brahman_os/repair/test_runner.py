"""Pytest execution and JSON report parsing for the Dharma repair loop."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from brahman_os.schemas import SafetyDecision, TestRunResult


class TestRunner:
    """Run pytest with pytest-json-report and return structured results."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        python_executable: str | Path | None = None,
        report_directory: str | Path | None = None,
    ) -> None:
        """Initialize a runner scoped to one repository root."""
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"repo_root does not exist: {self.repo_root}")
        self.python_executable = str(python_executable or sys.executable)
        self.report_directory = (
            Path(report_directory).resolve()
            if report_directory is not None
            else self.repo_root / ".brahman" / "test-reports"
        )

    def run(self, test_paths: tuple[str, ...] = ()) -> TestRunResult:
        """Run pytest and parse its JSON report, including failure tracebacks."""
        self.report_directory.mkdir(parents=True, exist_ok=True)
        report_path = self.report_directory / f"pytest-{uuid4()}.json"
        command = (
            self.python_executable,
            "-m",
            "pytest",
            *test_paths,
            "--json-report",
            f"--json-report-file={report_path}",
        )
        started_at = datetime.now(UTC)
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        completed_at = datetime.now(UTC)

        report = self._load_report(report_path)
        summary = report.get("summary", {})
        passed_tests = self._summary_count(summary, "passed")
        failed_tests = sum(
            self._summary_count(summary, key)
            for key in ("failed", "error", "xpassed")
        )
        total_tests = passed_tests + failed_tests
        tracebacks = self._traceback_summary(report)
        passed = completed.returncode == 0 and failed_tests == 0
        duration = max((completed_at - started_at).total_seconds(), 0.0)

        return TestRunResult(
            command=command,
            decision=SafetyDecision.PASS if passed else SafetyDecision.BLOCK,
            passed=passed,
            exit_code=completed.returncode,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            confidence=1.0,
            provenance=("pytest", "pytest-json-report"),
            evidence=(f"report_path={report_path}",),
            violated_rules=() if passed else ("tests_must_pass",),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            traceback_summary=tracebacks,
            report_path=str(report_path),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_pytest(self, test_paths: tuple[str, ...] = ()) -> TestRunResult:
        """Compatibility alias for explicitly named pytest execution."""
        return self.run(test_paths)

    @staticmethod
    def _load_report(report_path: Path) -> dict[str, object]:
        """Load a pytest JSON report or return an empty report safely."""
        if not report_path.is_file():
            return {}
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _summary_count(summary: object, key: str) -> int:
        """Read a non-negative integer summary count."""
        if not isinstance(summary, dict):
            return 0
        value = summary.get(key, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @staticmethod
    def _traceback_summary(report: dict[str, object]) -> tuple[str, ...]:
        """Extract concise failure messages from pytest report entries."""
        raw_tests = report.get("tests", [])
        if not isinstance(raw_tests, list):
            return ()

        summaries: list[str] = []
        for test in raw_tests:
            if not isinstance(test, dict) or test.get("outcome") == "passed":
                continue
            node_id = str(test.get("nodeid", "unknown test"))
            call = test.get("call")
            longrepr = call.get("longrepr") if isinstance(call, dict) else None
            if longrepr:
                summaries.append(f"{node_id}: {longrepr}")
            else:
                summaries.append(f"{node_id}: {test.get('outcome', 'failed')}")
        return tuple(summaries)
