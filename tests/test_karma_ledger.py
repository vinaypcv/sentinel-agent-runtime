"""Tests for the append-only KarmaLedger."""

import json
from pathlib import Path
from uuid import uuid4

from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.schemas import KarmaEvent, SafetyDecision


def karma_event(
    *,
    action_id: str = "action-1",
    goal_id: str = "goal-1",
    decision: SafetyDecision = SafetyDecision.PASS,
) -> KarmaEvent:
    """Build a complete audit event."""
    return KarmaEvent(
        action_id=action_id,
        goal_id=goal_id,
        action_type="tool_call",
        input_summary="Summarized input without sensitive payloads.",
        proposed_action="Execute a verified tool call.",
        viveka_decision=decision,
        evidence=("PEG evaluation completed.",),
        rollback_snapshot_id=uuid4(),
        result="Tool call completed.",
        status="completed",
    )


def test_event_is_written_to_jsonl(tmp_path: Path) -> None:
    """Logging an event should append one complete JSON object per line."""
    path = tmp_path / "karma.jsonl"
    ledger = KarmaLedger(path)
    event = karma_event()

    ledger.log_event(event)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"] == str(event.event_id)


def test_event_can_be_read_back_and_latest_returns_it(tmp_path: Path) -> None:
    """Persisted events should round-trip through strict schema validation."""
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    event = karma_event()
    ledger.log_event(event)

    assert ledger.get_events() == (event,)
    assert ledger.latest() == event


def test_filter_by_goal_id_and_decision(tmp_path: Path) -> None:
    """Goal and decision filters should select only matching events."""
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    first = karma_event(goal_id="goal-a")
    second = karma_event(
        action_id="action-2",
        goal_id="goal-b",
        decision=SafetyDecision.BLOCK,
    )
    ledger.log_event(first)
    ledger.log_event(second)

    assert ledger.get_events(goal_id="goal-b") == (second,)
    assert ledger.get_events(decision=SafetyDecision.PASS) == (first,)


def test_malformed_line_is_skipped_safely(tmp_path: Path) -> None:
    """Malformed JSONL records should not prevent valid records from loading."""
    path = tmp_path / "karma.jsonl"
    ledger = KarmaLedger(path)
    event = karma_event()
    ledger.log_event(event)
    with path.open("a", encoding="utf-8") as ledger_file:
        ledger_file.write("{not valid json}\n")

    assert ledger.get_events() == (event,)
    assert ledger.summarize()["malformed_lines"] == 1


def test_summary_and_export_are_explainable_json(tmp_path: Path) -> None:
    """Ledger summaries and exports should be JSON-serializable and auditable."""
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    event = karma_event()
    ledger.log_event(event)
    export_path = tmp_path / "karma-export.json"

    summary = ledger.summarize()
    ledger.export_json(export_path)
    exported = json.loads(export_path.read_text(encoding="utf-8"))

    json.dumps(summary)
    assert summary["total_events"] == 1
    assert summary["by_decision"] == {"pass": 1}
    assert exported["events"][0]["event_id"] == str(event.event_id)
