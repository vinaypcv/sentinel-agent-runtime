"""Tests for shared Brahman-OS schemas."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from brahman_os.schemas import (
    KarmaEvent,
    MemoryEvent,
    MemoryEventType,
    MemoryReadResult,
    PatchProposal,
    PolicyDocument,
    PolicyRule,
    RepairAttempt,
    RepairStatus,
    SafetyDecision,
    VivekaDecision,
)
from brahman_os.schemas import (
    TestRunResult as RunResultSchema,
)

NOW = datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


def policy_rule() -> PolicyRule:
    """Return a valid policy rule fixture."""
    return PolicyRule(
        rule_id="safe-write",
        description="Memory writes require supporting evidence.",
        condition="evidence_count > 0",
        decision=SafetyDecision.PASS,
        confidence=0.95,
        evidence=("policy-review",),
    )


def test_memory_schemas_validate_success_and_failure_cases() -> None:
    """Memory schemas should enforce explainable record consistency."""
    memory_id = uuid4()
    event = MemoryEvent(
        memory_id=memory_id,
        event_type=MemoryEventType.WRITE,
        content="Verified observation",
        confidence=0.9,
        provenance=("sensor-a",),
        evidence=("reading-42",),
        timestamp=NOW,
    )
    result = MemoryReadResult(
        memory_id=memory_id,
        found=True,
        content=event.content,
        confidence=event.confidence,
        provenance=event.provenance,
        evidence=event.evidence,
        retrieved_at=NOW,
    )

    assert json.loads(result.model_dump_json())["content"] == "Verified observation"
    with pytest.raises(ValidationError, match="content must be present"):
        MemoryReadResult(
            memory_id=memory_id,
            found=True,
            content=None,
            confidence=0.5,
        )


def test_policy_schemas_require_rules_and_ordered_timestamps() -> None:
    """Policy documents should contain rules and valid timestamps."""
    document = PolicyDocument(
        name="baseline",
        version="1.0.0",
        rules=(policy_rule(),),
        provenance=("security-team",),
        created_at=NOW,
        updated_at=NOW,
    )

    assert document.rules[0].decision is SafetyDecision.PASS
    with pytest.raises(ValidationError):
        PolicyDocument(
            name="empty",
            version="1.0.0",
            rules=(),
            created_at=NOW,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="updated_at must not precede"):
        PolicyDocument(
            name="time-travel",
            version="1.0.0",
            rules=(policy_rule(),),
            created_at=NOW,
            updated_at=NOW - timedelta(seconds=1),
        )


def test_viveka_decision_is_explainable_json() -> None:
    """Viveka decisions should expose every safety explanation field."""
    decision = VivekaDecision(
        decision=SafetyDecision.BLOCK,
        score=0.2,
        confidence=0.98,
        provenance=("peg", "policy:baseline"),
        evidence=("similarity below threshold",),
        violated_rules=("safe-write",),
        reasons=("Insufficient corroboration.",),
        timestamp=NOW,
    )

    payload = json.loads(decision.model_dump_json())

    assert payload["decision"] == "block"
    assert payload["confidence"] == 0.98
    assert payload["evidence"] == ["similarity below threshold"]
    assert payload["violated_rules"] == ["safe-write"]


def test_karma_event_records_an_explainable_audit_entry() -> None:
    """Karma events should serialize decisions and audit context."""
    event = KarmaEvent(
        action_id="action-1",
        goal_id="goal-1",
        action_type="guardrail_evaluation",
        input_summary="Evaluate a proposed memory write.",
        proposed_action="Write verified observation.",
        viveka_decision=SafetyDecision.PASS,
        evidence=("rule safe-write satisfied",),
        rollback_snapshot_id=None,
        result="Approved.",
        status="completed",
        timestamp=NOW,
    )

    payload = json.loads(event.model_dump_json())

    assert payload["viveka_decision"] == "pass"
    assert payload["goal_id"] == "goal-1"


def test_test_run_result_accepts_consistent_success() -> None:
    """A successful test run should have internally consistent totals."""
    result = RunResultSchema(
        command=("pytest", "tests/test_schemas.py"),
        decision=SafetyDecision.PASS,
        passed=True,
        exit_code=0,
        total_tests=2,
        passed_tests=2,
        failed_tests=0,
        confidence=1.0,
        provenance=("pytest",),
        evidence=("2 passed",),
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        duration_seconds=1.0,
    )

    assert result.passed is True
    assert result.decision is SafetyDecision.PASS


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"passed_tests": 1}, "must equal total_tests"),
        ({"passed": False}, "passed must agree"),
        ({"completed_at": NOW - timedelta(seconds=1)}, "must not precede"),
    ],
)
def test_test_run_result_rejects_inconsistent_results(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Inconsistent test reports should fail schema validation."""
    values: dict[str, object] = {
        "command": ("pytest",),
        "decision": SafetyDecision.PASS,
        "passed": True,
        "exit_code": 0,
        "total_tests": 2,
        "passed_tests": 2,
        "failed_tests": 0,
        "confidence": 1.0,
        "started_at": NOW,
        "completed_at": NOW,
        "duration_seconds": 0.0,
    }
    values.update(overrides)

    with pytest.raises(ValidationError, match=message):
        RunResultSchema.model_validate(values)


def test_patch_proposal_and_repair_attempt_are_traceable() -> None:
    """Patch and repair records should connect through stable identifiers."""
    attempt_id = uuid4()
    proposal = PatchProposal(
        repair_attempt_id=attempt_id,
        summary="Add bounds validation.",
        diff="+ raise ValueError",
        files_changed=("src/example.py",),
        decision=SafetyDecision.PASS,
        confidence=0.8,
        provenance=("repair-agent",),
        evidence=("failing regression test",),
        created_at=NOW,
    )
    attempt = RepairAttempt(
        attempt_id=attempt_id,
        issue_id="issue-1",
        status=RepairStatus.SUCCEEDED,
        decision=SafetyDecision.PASS,
        confidence=0.9,
        provenance=("dharma-loop",),
        evidence=("tests passed",),
        patch_proposal_id=proposal.proposal_id,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=2),
    )

    assert attempt.patch_proposal_id == proposal.proposal_id
    assert json.loads(proposal.model_dump_json())["decision"] == "pass"


@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_normalized_scores_reject_out_of_range_values(field: str, value: float) -> None:
    """All normalized Viveka scores should stay within zero and one."""
    values: dict[str, object] = {
        "decision": SafetyDecision.BLOCK,
        "score": 0.5,
        "confidence": 0.5,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        VivekaDecision.model_validate(values)


def test_models_reject_type_coercion_and_extra_fields() -> None:
    """Strict schemas should reject coercion and undeclared input."""
    with pytest.raises(ValidationError):
        PolicyRule.model_validate(
            {
                "rule_id": "rule-1",
                "description": "Reject coercion.",
                "condition": "always",
                "decision": "pass",
                "confidence": "0.9",
                "unexpected": True,
            }
        )
