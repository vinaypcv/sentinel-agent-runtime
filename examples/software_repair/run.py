"""Run an end-to-end autonomous software repair demonstration."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def ensure_project_runtime() -> None:
    """Re-execute with the project virtual environment when dependencies are absent."""
    if importlib.util.find_spec("torch") is not None:
        return

    virtualenv_python = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
    if not virtualenv_python.is_file():
        raise RuntimeError(
            "Project dependencies are unavailable. Activate the virtual environment."
        )
    completed = subprocess.run(
        [str(virtualenv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(completed.returncode)


ensure_project_runtime()
torch = importlib.import_module("torch")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail  # noqa: E402
from brahman_os.ledger.karma_ledger import KarmaLedger  # noqa: E402
from brahman_os.repair.llm_patch_generator import (  # noqa: E402
    GeminiPatchGenerator,
    MockPatchGenerator,
    PatchGenerator,
)
from brahman_os.repair.patcher import Patcher  # noqa: E402
from brahman_os.repair.test_runner import TestRunner  # noqa: E402
from brahman_os.schemas import KarmaEvent, PolicyRule, SafetyDecision  # noqa: E402

GOAL_ID = "repair-bounded-memory-stack"
ISSUE_ID = "stack-capacity-overflow"

BUGGY_STACK = '''"""A deliberately buggy bounded memory stack."""


class BoundedMemoryStack:
    """Keep at most ``capacity`` integer observations."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items: list[int] = []

    def push(self, item: int) -> None:
        """Push an item onto the stack."""
        self.items.append(item)

    def pop(self) -> int:
        """Pop the newest item."""
        return self.items.pop()
'''

STACK_TEST = '''from bounded_stack import BoundedMemoryStack


def test_stack_discards_oldest_item_at_capacity() -> None:
    stack = BoundedMemoryStack(capacity=2)

    stack.push(1)
    stack.push(2)
    stack.push(3)

    assert stack.items == [2, 3]  # nosec B101
'''

STACK_PATCH = (
    "diff --git a/bounded_stack.py b/bounded_stack.py\n"
    "--- a/bounded_stack.py\n"
    "+++ b/bounded_stack.py\n"
    "@@ -12,6 +12,8 @@ class BoundedMemoryStack:\n"
    " \n"
    "     def push(self, item: int) -> None:\n"
    '         """Push an item onto the stack."""\n'
    "+        if len(self.items) >= self.capacity:\n"
    "+            self.items.pop(0)\n"
    "         self.items.append(item)\n"
    " \n"
    "     def pop(self) -> int:\n"
)


def create_sandbox(workspace: Path) -> None:
    """Create the intentionally failing repair target."""
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("git", "init"),
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    (workspace / "bounded_stack.py").write_text(BUGGY_STACK, encoding="utf-8")
    (workspace / "test_bounded_stack.py").write_text(STACK_TEST, encoding="utf-8")
    (workspace / "pyproject.toml").write_text(
        (
            "[tool.ruff]\n"
            'target-version = "py311"\n\n'
            "[tool.mypy]\n"
            'python_version = "3.11"\n'
            "strict = true\n"
        ),
        encoding="utf-8",
    )


def log_step(
    ledger: KarmaLedger,
    *,
    action_type: str,
    input_summary: str,
    proposed_action: str,
    decision: SafetyDecision,
    evidence: tuple[str, ...],
    result: str,
    status: str,
    rollback_snapshot_id: UUID | None = None,
) -> None:
    """Record one explainable repair-loop stage."""
    ledger.log_event(
        KarmaEvent(
            action_id=str(uuid4()),
            goal_id=GOAL_ID,
            action_type=action_type,
            input_summary=input_summary,
            proposed_action=proposed_action,
            viveka_decision=decision,
            evidence=evidence,
            rollback_snapshot_id=rollback_snapshot_id,
            result=result,
            status=status,
        )
    )


def build_patch_generator(provider: str) -> PatchGenerator:
    """Build the selected patch generator without making provider calls."""
    if provider == "mock":
        return MockPatchGenerator(STACK_PATCH)
    if provider == "gemini":
        return GeminiPatchGenerator()
    raise ValueError("provider must be 'mock' or 'gemini'")


def empty_summary(workspace: Path, *, provider: str) -> dict[str, object]:
    """Create the stable JSON result shape used for success and failure paths."""
    return {
        "goal_id": GOAL_ID,
        "workspace": str(workspace),
        "provider": provider,
        "finalized": False,
        "initial_test": None,
        "patch": {
            "generator": None,
            "provider": provider,
            "generated": False,
            "files_changed": [],
            "snapshot_id": None,
            "rolled_back": None,
        },
        "patch_validation": {
            "passed": False,
            "files_changed": [],
            "error": None,
        },
        "static_checks": [],
        "quality_checks": [],
        "final_test": None,
        "peg_approval": None,
        "rollback": {
            "snapshot_id": None,
            "rolled_back": False,
        },
        "ledger": {
            "path": None,
            "event_count": 0,
            "summary": {},
        },
        "error": None,
    }


def error_payload(stage: str, error: Exception) -> dict[str, str]:
    """Render an exception as JSON-safe structured error data."""
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
    }


def run_demo(workspace: str | Path, *, provider: str = "mock") -> dict[str, object]:
    """Execute the failing-test, patch, verification, and PEG approval loop."""
    sandbox = Path(workspace).resolve()
    summary = empty_summary(sandbox, provider=provider)
    ledger: KarmaLedger | None = None

    try:
        create_sandbox(sandbox)
        ledger = KarmaLedger(sandbox / "karma-ledger.jsonl")
        runner = TestRunner(sandbox)

        initial_test = runner.run(("test_bounded_stack.py",))
        summary["initial_test"] = initial_test.model_dump(mode="json")
        log_step(
            ledger,
            action_type="initial_test_run",
            input_summary="Run the bounded stack regression test.",
            proposed_action="Capture the failing pytest report.",
            decision=initial_test.decision,
            evidence=initial_test.traceback_summary or initial_test.evidence,
            result=initial_test.model_dump_json(),
            status="failed" if not initial_test.passed else "passed",
        )
        if initial_test.passed:
            raise RuntimeError("The demonstration target did not fail before repair")

        failing_content = (sandbox / "bounded_stack.py").read_text(encoding="utf-8")
        test_content = (sandbox / "test_bounded_stack.py").read_text(encoding="utf-8")
        generator = build_patch_generator(provider)
        generator_name = type(generator).__name__
        summary["patch"] = {
            **dict(summary["patch"]),
            "generator": generator_name,
        }
        patch = generator.generate(
            failing_file_content=failing_content,
            pytest_traceback="\n".join(initial_test.traceback_summary),
            test_file_content=test_content,
            constraints=(
                "Modify only bounded_stack.py.",
                "Preserve the public class and method signatures.",
                "Return a unified diff only.",
            ),
        )
        summary["patch"] = {
            **dict(summary["patch"]),
            "generated": True,
        }
        log_step(
            ledger,
            action_type="patch_generation",
            input_summary="Generate a repair from the failing file, traceback, and test.",
            proposed_action=patch,
            decision=SafetyDecision.PASS,
            evidence=(f"{generator_name} returned a statically valid unified diff.",),
            result=json.dumps(
                {
                    "generator": generator_name,
                    "provider": provider,
                    "request_count": len(generator.requests)
                    if isinstance(generator, MockPatchGenerator)
                    else None,
                },
                sort_keys=True,
            ),
            status="completed",
        )

        patcher = Patcher(sandbox, ledger)
        files_changed = patcher.validate_patch(patch)
        summary["patch_validation"] = {
            "passed": True,
            "files_changed": list(files_changed),
            "error": None,
        }
        patch_result = patcher.apply(
            patch,
            goal_id=GOAL_ID,
            issue_id=ISSUE_ID,
        )
        static_checks = [check.to_dict() for check in patch_result.quality_checks]
        summary["static_checks"] = static_checks
        summary["quality_checks"] = static_checks
        summary["patch"] = {
            **dict(summary["patch"]),
            "files_changed": list(patch_result.files_changed),
            "snapshot_id": str(patch_result.snapshot_id),
            "rolled_back": patch_result.rolled_back,
        }
        summary["rollback"] = {
            "snapshot_id": str(patch_result.snapshot_id),
            "rolled_back": patch_result.rolled_back,
        }
        if patch_result.rolled_back:
            raise RuntimeError("Static quality checks rejected the generated patch")

        final_test = runner.run(("test_bounded_stack.py",))
        summary["final_test"] = final_test.model_dump(mode="json")
        log_step(
            ledger,
            action_type="post_patch_test_run",
            input_summary="Run pytest after applying the repair.",
            proposed_action="Verify the bounded stack regression.",
            decision=final_test.decision,
            evidence=final_test.evidence,
            result=final_test.model_dump_json(),
            status="passed" if final_test.passed else "failed",
            rollback_snapshot_id=patch_result.snapshot_id,
        )

        quality_passed = all(check.passed for check in patch_result.quality_checks)
        approval_rule = PolicyRule(
            rule_id="repair-verification-complete",
            description="All static checks and regression tests must pass.",
            condition="ruff, mypy, bandit, and pytest pass",
            decision=SafetyDecision.PASS,
            confidence=1.0,
        )
        generated_vector = torch.tensor(
            [float(final_test.passed), float(quality_passed)],
            dtype=torch.float32,
        )
        context_vector = torch.ones(2, dtype=torch.float32)
        approval = PramanaEpistemicGuardrail(threshold=0.7).evaluate(
            generated_vector,
            context_vector,
            rules=(approval_rule,),
            rule_results={
                approval_rule.rule_id: final_test.passed and quality_passed,
            },
        )
        summary["peg_approval"] = approval.model_dump(mode="json")
        log_step(
            ledger,
            action_type="peg_final_approval",
            input_summary="Evaluate repaired code against test and static-check evidence.",
            proposed_action="Finalize the repair.",
            decision=approval.decision,
            evidence=approval.evidence,
            result=approval.model_dump_json(),
            status="approved"
            if approval.decision is SafetyDecision.PASS
            else "blocked",
            rollback_snapshot_id=patch_result.snapshot_id,
        )

        summary["finalized"] = (
            final_test.passed
            and quality_passed
            and approval.decision is SafetyDecision.PASS
        )
        if not summary["finalized"]:
            summary["error"] = {
                "stage": "finalization",
                "type": "RepairNotFinalized",
                "message": "Repair did not pass final verification.",
            }
    except Exception as error:
        patch_validation = dict(summary["patch_validation"])
        if (
            not patch_validation["passed"]
            and patch_validation["error"] is None
        ):
            summary["patch_validation"] = {
                **patch_validation,
                "error": str(error),
            }
        summary["error"] = error_payload("software_repair_demo", error)
    finally:
        if ledger is not None:
            events = ledger.get_events(goal_id=GOAL_ID)
            summary["ledger"] = {
                "path": str(ledger.path),
                "event_count": len(events),
                "summary": ledger.summarize(),
            }

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional sandbox directory. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="Patch generator provider. Defaults to offline mock.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the demo and print one explainable JSON summary."""
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        if args.workspace is not None:
            summary = run_demo(args.workspace, provider=args.provider)
        else:
            with tempfile.TemporaryDirectory(prefix="brahman-repair-demo-") as directory:
                summary = run_demo(directory, provider=args.provider)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["finalized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
