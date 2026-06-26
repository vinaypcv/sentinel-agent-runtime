"""Secure unified-diff application for the Dharma repair loop."""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.repair.rollback import RollbackManager
from brahman_os.schemas import KarmaEvent, RepairAttempt, RepairStatus, SafetyDecision


class PatchRejectedError(ValueError):
    """Raised when a patch violates repository or security constraints."""


@dataclass(frozen=True, slots=True)
class QualityCheckResult:
    """Result of one post-patch quality command."""

    name: str
    command: tuple[str, ...]
    passed: bool
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable quality-check explanation."""
        return {
            "name": self.name,
            "command": list(self.command),
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class PatchResult:
    """Explainable result of a complete patch attempt."""

    attempt: RepairAttempt
    snapshot_id: UUID | None
    files_changed: tuple[str, ...]
    quality_checks: tuple[QualityCheckResult, ...]
    rolled_back: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable explanation of the patch attempt."""
        return {
            "attempt": self.attempt.model_dump(mode="json"),
            "snapshot_id": str(self.snapshot_id) if self.snapshot_id else None,
            "files_changed": list(self.files_changed),
            "quality_checks": [check.to_dict() for check in self.quality_checks],
            "rolled_back": self.rolled_back,
        }

    def to_json(self) -> str:
        """Serialize the complete patch result as explainable JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)


class Patcher:
    """Validate, snapshot, apply, verify, audit, and roll back unified diffs."""

    _SENSITIVE_NAMES = {
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        "secrets",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
    _SECRET_ASSIGNMENT_MARKERS = (
        "api_key",
        "apikey",
        "auth_token",
        "client_secret",
        "password",
        "private_key",
        "secret",
        "token",
    )

    def __init__(
        self,
        repo_root: str | Path,
        ledger: KarmaLedger | None = None,
        *,
        rollback_manager: RollbackManager | None = None,
    ) -> None:
        """Initialize a secure patcher for one repository."""
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"repo_root does not exist: {self.repo_root}")
        self.ledger = ledger or KarmaLedger(
            self.repo_root / ".brahman" / "karma-ledger.jsonl"
        )
        self.rollback_manager = rollback_manager or RollbackManager(self.repo_root)

    def apply(
        self,
        diff: str,
        *,
        goal_id: str,
        issue_id: str,
    ) -> PatchResult:
        """Apply and verify a unified diff, rolling back failed quality checks."""
        attempt_id = uuid4()
        try:
            files_changed = self.validate_patch(diff)
        except (PatchRejectedError, ValueError) as error:
            attempt = RepairAttempt(
                attempt_id=attempt_id,
                issue_id=issue_id,
                status=RepairStatus.FAILED,
                decision=SafetyDecision.BLOCK,
                confidence=1.0,
                provenance=("patcher",),
                evidence=(str(error),),
                violated_rules=("patch_safety",),
            )
            self._log_attempt(
                attempt,
                goal_id=goal_id,
                diff=diff,
                snapshot_id=None,
                result=str(error),
            )
            raise PatchRejectedError(str(error)) from error

        try:
            snapshot_id = self.rollback_manager.snapshot(files_changed)
        except (OSError, TypeError, ValueError) as error:
            attempt = RepairAttempt(
                attempt_id=attempt_id,
                issue_id=issue_id,
                status=RepairStatus.FAILED,
                decision=SafetyDecision.BLOCK,
                confidence=1.0,
                provenance=("patcher", "rollback"),
                evidence=(str(error),),
                violated_rules=("snapshot_required",),
            )
            self._log_attempt(
                attempt,
                goal_id=goal_id,
                diff=diff,
                snapshot_id=None,
                result=str(error),
            )
            raise PatchRejectedError(f"unable to create rollback snapshot: {error}") from error

        try:
            self._apply_with_git(diff)
        except (OSError, PatchRejectedError) as error:
            self.rollback_manager.rollback(snapshot_id)
            attempt = RepairAttempt(
                attempt_id=attempt_id,
                issue_id=issue_id,
                status=RepairStatus.ROLLED_BACK,
                decision=SafetyDecision.BLOCK,
                confidence=1.0,
                provenance=("patcher", "git-apply"),
                evidence=(str(error),),
                violated_rules=("patch_must_apply_cleanly",),
            )
            self._log_attempt(
                attempt,
                goal_id=goal_id,
                diff=diff,
                snapshot_id=snapshot_id,
                result=str(error),
            )
            raise PatchRejectedError(str(error)) from error

        quality_checks = self._run_quality_checks()
        checks_passed = all(check.passed for check in quality_checks)
        rolled_back = not checks_passed
        if rolled_back:
            self.rollback_manager.rollback(snapshot_id)

        attempt = RepairAttempt(
            attempt_id=attempt_id,
            issue_id=issue_id,
            status=RepairStatus.SUCCEEDED if checks_passed else RepairStatus.ROLLED_BACK,
            decision=SafetyDecision.PASS if checks_passed else SafetyDecision.BLOCK,
            confidence=1.0,
            provenance=("patcher", "ruff", "mypy", "bandit"),
            evidence=tuple(
                f"{check.name}: exit_code={check.exit_code}" for check in quality_checks
            ),
            violated_rules=()
            if checks_passed
            else tuple(check.name for check in quality_checks if not check.passed),
        )
        self._log_attempt(
            attempt,
            goal_id=goal_id,
            diff=diff,
            snapshot_id=snapshot_id,
            result=json.dumps(
                {
                    "files_changed": files_changed,
                    "quality_checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "exit_code": check.exit_code,
                        }
                        for check in quality_checks
                    ],
                    "rolled_back": rolled_back,
                },
                sort_keys=True,
            ),
        )
        return PatchResult(
            attempt=attempt,
            snapshot_id=snapshot_id,
            files_changed=files_changed,
            quality_checks=quality_checks,
            rolled_back=rolled_back,
        )

    def apply_patch(
        self,
        diff: str,
        *,
        goal_id: str = "software-repair",
        issue_id: str = "unspecified-issue",
    ) -> PatchResult:
        """Compatibility alias for applying a unified diff."""
        return self.apply(diff, goal_id=goal_id, issue_id=issue_id)

    def validate_patch(self, diff: str) -> tuple[str, ...]:
        """Return safe repository-relative paths from a unified diff."""
        if not diff.strip():
            raise PatchRejectedError("patch must not be empty")

        paths: list[str] = []
        for line in diff.splitlines():
            if self._adds_secret_like_assignment(line):
                raise PatchRejectedError(
                    "patch appears to add a hardcoded secret-like assignment"
                )
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw_path = line[4:].split("\t", maxsplit=1)[0].strip()
            if raw_path == "/dev/null":
                continue
            normalized = self._normalize_diff_path(raw_path)
            if normalized not in paths:
                paths.append(normalized)

        if not paths:
            raise PatchRejectedError("patch does not contain file headers")
        return tuple(paths)

    def _normalize_diff_path(self, raw_path: str) -> str:
        """Normalize one diff path and reject escapes or sensitive files."""
        path = PurePosixPath(raw_path)
        if path.parts and path.parts[0] in {"a", "b"}:
            path = PurePosixPath(*path.parts[1:])
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PatchRejectedError(f"patch path is outside repository: {raw_path}")

        lowered_parts = tuple(part.lower() for part in path.parts)
        if any(
            part in self._SENSITIVE_NAMES
            or part.startswith(".env.")
            or "secret" in part
            for part in lowered_parts
        ):
            raise PatchRejectedError(f"patch targets a security-sensitive file: {path}")

        resolved = (self.repo_root / Path(*path.parts)).resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as error:
            raise PatchRejectedError(
                f"patch path is outside repository: {raw_path}"
            ) from error
        return path.as_posix()

    def _adds_secret_like_assignment(self, line: str) -> bool:
        """Return True when an added diff line appears to hardcode a secret."""
        if not line.startswith("+") or line.startswith("+++"):
            return False

        normalized = line[1:].strip().lower()
        if not normalized or normalized.startswith(("#", '"""', "'''")):
            return False
        if "=" not in normalized and ":" not in normalized:
            return False

        key = normalized.split("=", maxsplit=1)[0].split(":", maxsplit=1)[0].strip()
        return any(marker in key for marker in self._SECRET_ASSIGNMENT_MARKERS)

    def _apply_with_git(self, diff: str) -> None:
        """Validate and apply a patch using git's unified-diff engine."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".diff",
            delete=False,
        ) as patch_file:
            patch_file.write(diff)
            patch_path = Path(patch_file.name)

        try:
            check = subprocess.run(
                ("git", "apply", "--check", "--whitespace=nowarn", str(patch_path)),
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if check.returncode != 0:
                raise PatchRejectedError(
                    f"patch does not apply cleanly: {check.stderr.strip()}"
                )
            applied = subprocess.run(
                ("git", "apply", "--whitespace=nowarn", str(patch_path)),
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if applied.returncode != 0:
                raise PatchRejectedError(
                    f"patch application failed: {applied.stderr.strip()}"
                )
        finally:
            patch_path.unlink(missing_ok=True)

    def _run_quality_checks(self) -> tuple[QualityCheckResult, ...]:
        """Run Ruff, mypy, and Bandit after a patch."""
        commands = (
            ("ruff", (sys.executable, "-m", "ruff", "check", ".")),
            ("mypy", (sys.executable, "-m", "mypy", ".")),
            (
                "bandit",
                (
                    sys.executable,
                    "-m",
                    "bandit",
                    "-r",
                    ".",
                    "-x",
                    ".venv,docs/raw",
                ),
            ),
        )
        results = []
        for name, command in commands:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                results.append(
                    QualityCheckResult(
                        name=name,
                        command=command,
                        passed=completed.returncode == 0,
                        exit_code=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                    )
                )
            except OSError as error:
                results.append(
                    QualityCheckResult(
                        name=name,
                        command=command,
                        passed=False,
                        exit_code=-1,
                        stdout="",
                        stderr=str(error),
                    )
                )
        return tuple(results)

    def _log_attempt(
        self,
        attempt: RepairAttempt,
        *,
        goal_id: str,
        diff: str,
        snapshot_id: UUID | None,
        result: str,
    ) -> None:
        """Write one explainable repair attempt to KarmaLedger."""
        self.ledger.log_event(
            KarmaEvent(
                action_id=str(attempt.attempt_id),
                goal_id=goal_id,
                action_type="software_repair",
                input_summary=f"issue_id={attempt.issue_id}",
                proposed_action=diff[:1000],
                viveka_decision=attempt.decision,
                evidence=attempt.evidence,
                rollback_snapshot_id=snapshot_id,
                result=result,
                status=attempt.status.value,
            )
        )
