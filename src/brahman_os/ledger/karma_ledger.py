"""Append-only JSONL audit ledger for Brahman-OS actions."""

import json
import os
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import ValidationError

from brahman_os.schemas import KarmaEvent, SafetyDecision


class KarmaLedger:
    """Persist immutable KarmaEvent records in an append-only JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Initialize a ledger and create its file if it does not exist."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = RLock()

    def log_event(self, event: KarmaEvent) -> None:
        """Append one validated event and durably flush it to disk."""
        if not isinstance(event, KarmaEvent):
            raise TypeError("event must be a KarmaEvent")

        serialized = event.model_dump_json()
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(serialized)
            ledger.write("\n")
            ledger.flush()
            os.fsync(ledger.fileno())

    def get_events(
        self,
        *,
        action_id: str | None = None,
        goal_id: str | None = None,
        decision: SafetyDecision | str | None = None,
    ) -> tuple[KarmaEvent, ...]:
        """Read valid events, optionally filtering by action, goal, or decision."""
        normalized_decision = self._normalize_decision(decision)
        events, _ = self._read_events()
        return tuple(
            event
            for event in events
            if (action_id is None or event.action_id == action_id)
            and (goal_id is None or event.goal_id == goal_id)
            and (
                normalized_decision is None
                or event.viveka_decision is normalized_decision
            )
        )

    def latest(self) -> KarmaEvent | None:
        """Return the latest valid event, or ``None`` for an empty ledger."""
        events, _ = self._read_events()
        return events[-1] if events else None

    def summarize(self) -> dict[str, object]:
        """Return a JSON-serializable summary of valid and malformed records."""
        events, malformed_lines = self._read_events()
        return {
            "ledger_path": str(self.path),
            "total_events": len(events),
            "malformed_lines": malformed_lines,
            "by_decision": dict(
                Counter(event.viveka_decision.value for event in events)
            ),
            "by_status": dict(Counter(event.status for event in events)),
            "by_action_type": dict(Counter(event.action_type for event in events)),
            "latest_event_id": str(events[-1].event_id) if events else None,
        }

    def export_json(self, path: str | Path) -> None:
        """Export all valid ledger events as a formatted JSON array."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        events, malformed_lines = self._read_events()
        payload: dict[str, Any] = {
            "source_ledger": str(self.path),
            "malformed_lines_skipped": malformed_lines,
            "events": [event.model_dump(mode="json") for event in events],
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_events(self) -> tuple[list[KarmaEvent], int]:
        """Read valid events while safely skipping malformed ledger lines."""
        events: list[KarmaEvent] = []
        malformed_lines = 0
        with self._lock, self.path.open("r", encoding="utf-8") as ledger:
            for line in ledger:
                if not line.strip():
                    continue
                try:
                    events.append(KarmaEvent.model_validate_json(line))
                except (ValidationError, ValueError):
                    malformed_lines += 1
        return events, malformed_lines

    @staticmethod
    def _normalize_decision(
        decision: SafetyDecision | str | None,
    ) -> SafetyDecision | None:
        """Normalize a decision filter and reject unsupported values."""
        if decision is None or isinstance(decision, SafetyDecision):
            return decision
        try:
            return SafetyDecision(decision)
        except ValueError as error:
            raise ValueError("decision must be pass or block") from error
