"""Tests for the autonomous software repair loop."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.repair.patcher import (
    Patcher,
    PatchRejectedError,
    QualityCheckResult,
)
from brahman_os.repair.rollback import RollbackManager
from brahman_os.repair.test_runner import TestRunner as RepairTestRunner
from brahman_os.schemas import RepairStatus


def passing_quality_checks() -> tuple[QualityCheckResult, ...]:
    """Return successful results for all required post-patch tools."""
    return tuple(
        QualityCheckResult(
            name=name,
            command=(name,),
            passed=True,
            exit_code=0,
            stdout="",
            stderr="",
        )
        for name in ("ruff", "mypy", "bandit")
    )


def test_failing_test_is_captured(tmp_path: Path) -> None:
    """TestRunner should capture failure counts, tracebacks, and report paths."""
    test_file = tmp_path / "test_failure.py"
    test_file.write_text(
        "def test_failure():\n    assert 1 == 2\n",
        encoding="utf-8",
    )

    result = RepairTestRunner(tmp_path).run(("test_failure.py",))

    assert result.passed is False
    assert result.failed_tests == 1
    assert result.traceback_summary
    assert "test_failure.py::test_failure" in result.traceback_summary[0]
    assert result.duration_seconds >= 0.0
    assert result.report_path is not None
    assert Path(result.report_path).is_file()


def test_passing_test_returns_explainable_json(tmp_path: Path) -> None:
    """Successful pytest runs should produce a complete JSON result."""
    test_file = tmp_path / "test_success.py"
    test_file.write_text(
        "def test_success():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    result = RepairTestRunner(tmp_path).run_pytest(("test_success.py",))
    payload = json.loads(result.model_dump_json())

    assert result.passed is True
    assert result.passed_tests == 1
    assert result.failed_tests == 0
    assert result.traceback_summary == ()
    assert result.duration == result.duration_seconds
    assert payload["decision"] == "pass"
    assert payload["report_path"] == result.report_path
    assert payload["provenance"] == ["pytest", "pytest-json-report"]


def test_patch_applies_cleanly_and_logs_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A safe unified diff should apply and create a successful ledger event."""
    target = tmp_path / "example.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    patcher = Patcher(tmp_path, ledger)
    monkeypatch.setattr(patcher, "_run_quality_checks", passing_quality_checks)
    diff = (
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    result = patcher.apply(diff, goal_id="repair-goal", issue_id="issue-1")
    payload = json.loads(result.to_json())

    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert result.attempt.status is RepairStatus.SUCCEEDED
    assert result.rolled_back is False
    assert payload["files_changed"] == ["example.py"]
    assert payload["attempt"]["decision"] == "pass"
    assert payload["rolled_back"] is False
    assert len(ledger.get_events()) == 1


def test_bad_patch_is_rejected_and_logged(tmp_path: Path) -> None:
    """A malformed patch should fail without modifying repository files."""
    target = tmp_path / "example.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    patcher = Patcher(tmp_path, ledger)
    bad_diff = (
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 999\n"
        "+VALUE = 2\n"
    )

    with pytest.raises(PatchRejectedError, match="does not apply cleanly"):
        patcher.apply(bad_diff, goal_id="repair-goal", issue_id="issue-2")

    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert len(ledger.get_events()) == 1


def test_rollback_restores_previous_file(tmp_path: Path) -> None:
    """RollbackManager should restore byte-for-byte pre-patch content."""
    target = tmp_path / "example.py"
    target.write_text("before\n", encoding="utf-8")
    rollback = RollbackManager(tmp_path)
    snapshot_id = rollback.snapshot(("example.py",))

    target.write_text("after\n", encoding="utf-8")
    rollback.rollback(snapshot_id)

    assert target.read_text(encoding="utf-8") == "before\n"


def test_rollback_removes_file_created_after_snapshot(tmp_path: Path) -> None:
    """A path absent at snapshot time should be removed during rollback."""
    rollback = RollbackManager(tmp_path)
    snapshot_id = rollback.create_snapshot(("created_later.py",))
    created_later = tmp_path / "created_later.py"
    created_later.write_text("temporary\n", encoding="utf-8")

    rollback.rollback(snapshot_id)

    assert not created_later.exists()


def test_rollback_snapshot_explanation_is_json_serializable(tmp_path: Path) -> None:
    """Snapshot explanations should expose metadata without file contents."""
    target = tmp_path / "example.py"
    target.write_text("before\n", encoding="utf-8")
    rollback = RollbackManager(tmp_path)
    snapshot_id = rollback.snapshot(("example.py", "missing.py"))

    explanation = rollback.explain_snapshot(snapshot_id)
    expected_size = len(target.read_bytes())

    json.dumps(explanation)
    assert explanation["file_count"] == 2
    assert explanation["files"] == [
        {"path": "example.py", "existed": True, "byte_size": expected_size},
        {"path": "missing.py", "existed": False, "byte_size": 0},
    ]


def test_rollback_rejects_unknown_snapshot_and_external_paths(tmp_path: Path) -> None:
    """Unknown snapshots and paths outside the repository should fail safely."""
    rollback = RollbackManager(tmp_path)

    with pytest.raises(KeyError, match="unknown snapshot_id"):
        rollback.rollback(uuid4())
    with pytest.raises(ValueError, match="outside repository"):
        rollback.snapshot(("../outside.py",))
    with pytest.raises(ValueError, match="at least one path"):
        rollback.snapshot(())


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "config/.env.production",
        "secrets.yaml",
        "config/client_secret.json",
        "../outside.py",
    ],
)
def test_security_sensitive_or_external_patch_is_blocked(
    tmp_path: Path,
    path: str,
) -> None:
    """Patches must not escape the repo or alter secrets and environment files."""
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    patcher = Patcher(tmp_path, ledger)
    diff = (
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        "+SECRET=value\n"
    )

    with pytest.raises(PatchRejectedError):
        patcher.apply(diff, goal_id="repair-goal", issue_id=str(uuid4()))

    assert len(ledger.get_events()) == 1


def test_patch_adding_secret_like_assignment_is_blocked(tmp_path: Path) -> None:
    """Patches should not introduce hardcoded credential-like assignments."""
    target = tmp_path / "config.py"
    target.write_text("DEBUG = False\n", encoding="utf-8")
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    patcher = Patcher(tmp_path, ledger)
    diff = (
        "--- a/config.py\n"
        "+++ b/config.py\n"
        "@@ -1 +1,2 @@\n"
        " DEBUG = False\n"
        '+ API_KEY = "placeholder"\n'
    )

    with pytest.raises(PatchRejectedError, match="hardcoded secret-like assignment"):
        patcher.apply(diff, goal_id="repair-goal", issue_id="issue-secret")

    assert target.read_text(encoding="utf-8") == "DEBUG = False\n"
    assert len(ledger.get_events()) == 1
