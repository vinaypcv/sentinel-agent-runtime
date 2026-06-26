"""Tests for the Brahman-OS runtime orchestrator."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail
from brahman_os.guardrails.policy_loader import PolicyLoader
from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.memory.akasha_store import AkashaStore
from brahman_os.runtime.orchestrator import ActionRequest, BrahmanRuntime


class RecordingToolAdapter:
    """Record approved calls and optionally raise an execution error."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def execute(
        self,
        action_type: str,
        arguments: Mapping[str, object],
    ) -> object:
        """Record a call and return a JSON-safe result."""
        self.calls.append((action_type, arguments))
        if self.fail:
            raise RuntimeError("tool execution failed")
        return {"accepted": True, "arguments": dict(arguments)}


def runtime(
    tmp_path: Path,
    adapter: RecordingToolAdapter,
) -> tuple[BrahmanRuntime, KarmaLedger]:
    """Build a grounded runtime with one memory vector."""
    memory = AkashaStore(d_model=2)
    memory.write(
        torch.tensor([1.0, 0.0]),
        {
            "confidence": 1.0,
            "provenance": ("test-context",),
            "evidence": ("approved context",),
        },
    )
    ledger = KarmaLedger(tmp_path / "karma.jsonl")
    return (
        BrahmanRuntime(
            memory=memory,
            verifier=PramanaEpistemicGuardrail(threshold=0.7),
            ledger=ledger,
            policy_loader=PolicyLoader(),
            tool_adapter=adapter,
        ),
        ledger,
    )


def action_request(generated_vector: torch.Tensor) -> ActionRequest:
    """Build a runtime action against the known memory query."""
    return ActionRequest(
        action_id="action-1",
        goal_id="goal-1",
        action_type="record_observation",
        input_summary="Record a verified observation.",
        proposed_action="Persist the observation.",
        generated_vector=generated_vector,
        query_vector=torch.tensor([1.0, 0.0]),
        tool_arguments={"observation": "verified"},
    )


def test_blocked_action_is_not_executed(tmp_path: Path) -> None:
    """An ungrounded action should be logged and blocked before execution."""
    adapter = RecordingToolAdapter()
    orchestrator, ledger = runtime(tmp_path, adapter)

    response = orchestrator.execute_action(
        action_request(torch.tensor([-1.0, 0.0]))
    )

    assert response["status"] == "blocked"
    assert response["executed"] is False
    assert response["decision"]["decision"] == "block"
    assert adapter.calls == []
    assert len(ledger.get_events()) == 1


def test_passed_action_executes_and_logs_result(tmp_path: Path) -> None:
    """A grounded action should execute once and write decision and result events."""
    adapter = RecordingToolAdapter()
    orchestrator, ledger = runtime(tmp_path, adapter)

    response = orchestrator.execute_action(
        action_request(torch.tensor([1.0, 0.0]))
    )

    assert response["status"] == "completed"
    assert response["executed"] is True
    assert response["decision"]["decision"] == "pass"
    assert response["result"] == {
        "accepted": True,
        "arguments": {"observation": "verified"},
    }
    assert len(adapter.calls) == 1
    assert len(ledger.get_events()) == 2
    json.dumps(response)


def test_every_action_writes_to_ledger(tmp_path: Path) -> None:
    """Blocked and passed actions should both leave an audit trail."""
    adapter = RecordingToolAdapter()
    orchestrator, ledger = runtime(tmp_path, adapter)

    orchestrator.execute_action(action_request(torch.tensor([-1.0, 0.0])))
    passed_request = ActionRequest(
        action_id="action-2",
        goal_id="goal-1",
        action_type="record_observation",
        input_summary="Record another verified observation.",
        proposed_action="Persist the second observation.",
        generated_vector=torch.tensor([1.0, 0.0]),
        query_vector=torch.tensor([1.0, 0.0]),
    )
    orchestrator.execute_action(passed_request)

    assert len(ledger.get_events(action_id="action-1")) == 1
    assert len(ledger.get_events(action_id="action-2")) == 2


def test_tool_exceptions_are_logged_and_returned(tmp_path: Path) -> None:
    """Execution exceptions should produce an error response and ledger event."""
    adapter = RecordingToolAdapter(fail=True)
    orchestrator, ledger = runtime(tmp_path, adapter)

    response = orchestrator.execute_action(
        action_request(torch.tensor([1.0, 0.0]))
    )
    events = ledger.get_events()

    assert response["status"] == "error"
    assert response["executed"] is True
    assert response["error"] == {
        "type": "RuntimeError",
        "message": "tool execution failed",
    }
    assert len(events) == 2
    assert events[-1].status == "error"
    assert "RuntimeError: tool execution failed" in events[-1].evidence
