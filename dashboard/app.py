"""Streamlit dashboard for local Brahman-OS KarmaLedger JSONL files."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest-runs",
    ".pytest-tmp",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
SCORE_PATTERN = re.compile(r"\b(?:viveka_)?score=([01](?:\.\d+)?)\b", re.IGNORECASE)


def discover_ledger_files(root: Path) -> list[Path]:
    """Return local JSONL files likely to contain KarmaLedger records."""
    ledgers: list[Path] = []
    for path in root.rglob("*.jsonl"):
        if any(part in DEFAULT_EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        ledgers.append(path)
    return sorted(ledgers)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read valid JSON objects from one JSONL file and count malformed lines."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        source = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    value["_source"] = source
                    value["_line"] = line_number
                    rows.append(value)
                else:
                    malformed += 1
    except OSError:
        malformed += 1
    return rows, malformed


def load_events(paths: list[Path]) -> tuple[pd.DataFrame, int]:
    """Load all selected ledger files into a display-friendly dataframe."""
    events: list[dict[str, Any]] = []
    malformed = 0
    for path in paths:
        rows, bad_lines = read_jsonl(path)
        events.extend(rows)
        malformed += bad_lines

    if not events:
        return pd.DataFrame(), malformed

    frame = pd.DataFrame(events)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp", ascending=True, na_position="last")
    frame["viveka_score"] = frame.apply(extract_viveka_score, axis=1)
    return frame.reset_index(drop=True), malformed


def parse_json_object(value: object) -> dict[str, Any]:
    """Parse JSON object strings safely, returning an empty dict otherwise."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_viveka_score(row: pd.Series) -> float | None:
    """Extract a Viveka score from result JSON or explainability evidence."""
    result = parse_json_object(row.get("result"))
    score = result.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool):
        return float(score)

    decision = result.get("decision")
    if isinstance(decision, dict):
        nested_score = decision.get("score")
        if isinstance(nested_score, int | float) and not isinstance(nested_score, bool):
            return float(nested_score)

    evidence = row.get("evidence")
    evidence_items = evidence if isinstance(evidence, list) else []
    for item in evidence_items:
        if not isinstance(item, str):
            continue
        match = SCORE_PATTERN.search(item)
        if match is not None:
            return float(match.group(1))
    return None


def latest_activity(frame: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Return recent events whose action type or status matches a regex."""
    if frame.empty:
        return pd.DataFrame()
    mask = (
        frame.get("action_type", pd.Series(dtype=str)).astype(str).str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
        | frame.get("status", pd.Series(dtype=str)).astype(str).str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    )
    return frame.loc[mask].tail(20)


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return only available columns in a stable order."""
    available = [column for column in columns if column in frame.columns]
    return frame[available] if available else frame


def render_overview(frame: pd.DataFrame, malformed_lines: int) -> None:
    """Render global KarmaLedger summary metrics."""
    total_actions = len(frame)
    decisions = Counter(frame.get("viveka_decision", pd.Series(dtype=str)).dropna())
    scores = frame.get("viveka_score", pd.Series(dtype=float)).dropna()
    average_score = float(scores.mean()) if not scores.empty else 0.0

    cols = st.columns(5)
    cols[0].metric("Total actions", total_actions)
    cols[1].metric("Passed actions", decisions.get("pass", 0))
    cols[2].metric("Blocked actions", decisions.get("block", 0))
    cols[3].metric("Average Viveka score", f"{average_score:.3f}")
    cols[4].metric("Malformed lines skipped", malformed_lines)


def render_karma_ledger(frame: pd.DataFrame) -> None:
    """Render a table of ledger events."""
    if frame.empty:
        st.info("No KarmaLedger events found in selected local JSONL files.")
        return

    display = select_columns(
        frame,
        [
            "timestamp",
            "goal_id",
            "action_id",
            "action_type",
            "viveka_decision",
            "viveka_score",
            "status",
            "rollback_snapshot_id",
            "_source",
            "_line",
        ],
    )
    st.dataframe(display, use_container_width=True)


def render_viveka_timeline(frame: pd.DataFrame) -> None:
    """Render Viveka score history as a line chart."""
    if frame.empty or "timestamp" not in frame or "viveka_score" not in frame:
        st.info("No timestamped Viveka scores are available yet.")
        return

    timeline = frame.dropna(subset=["timestamp", "viveka_score"]).copy()
    if timeline.empty:
        st.info("No Viveka score values were found in ledger results or evidence.")
        return

    timeline = timeline.set_index("timestamp")
    st.line_chart(timeline[["viveka_score"]])


def render_memory_state(frame: pd.DataFrame) -> None:
    """Render memory-related ledger activity."""
    memory_events = latest_activity(frame, r"memory|akasha|read|write")
    st.metric("Memory events", len(memory_events))

    if memory_events.empty:
        st.info("No memory read/write events found in local ledgers yet.")
        return

    display = select_columns(
        memory_events,
        [
            "timestamp",
            "action_type",
            "status",
            "input_summary",
            "viveka_decision",
            "_source",
        ],
    )
    st.dataframe(display, use_container_width=True)


def render_repair_runs(frame: pd.DataFrame) -> None:
    """Render repair-loop test, patch, and rollback activity."""
    repair_events = latest_activity(frame, r"repair|patch|test|rollback")
    failed_tests = count_failed_tests(frame)
    patch_attempts = len(latest_activity(frame, r"software_repair|patch_generation|patch"))
    rollback_events = latest_activity(frame, r"rollback|rolled_back")

    cols = st.columns(3)
    cols[0].metric("Failed tests", failed_tests)
    cols[1].metric("Patch attempts", patch_attempts)
    cols[2].metric("Rollback events", len(rollback_events))

    if repair_events.empty:
        st.info("No repair-loop events found in local ledgers yet.")
        return

    display = select_columns(
        repair_events,
        [
            "timestamp",
            "action_type",
            "status",
            "viveka_decision",
            "rollback_snapshot_id",
            "result",
            "_source",
        ],
    )
    st.dataframe(display, use_container_width=True)


def count_failed_tests(frame: pd.DataFrame) -> int:
    """Count failed tests reported in TestRunResult JSON payloads."""
    if frame.empty or "result" not in frame:
        return 0

    failed = 0
    for result in frame["result"]:
        payload = parse_json_object(result)
        value = payload.get("failed_tests")
        if isinstance(value, int) and not isinstance(value, bool):
            failed += value
    return failed


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Brahman-OS Dashboard", layout="wide")
    st.title("Brahman-OS Dashboard")
    st.caption("Local-only dashboard backed by KarmaLedger JSONL files.")

    ledger_files = discover_ledger_files(REPO_ROOT)
    selected = st.sidebar.multiselect(
        "Ledger JSONL files",
        options=ledger_files,
        default=ledger_files,
        format_func=lambda path: str(path.relative_to(REPO_ROOT)),
    )

    events, malformed_lines = load_events(selected)

    tabs = st.tabs(
        [
            "Overview",
            "KarmaLedger",
            "Viveka Timeline",
            "Memory State",
            "Repair Runs",
        ]
    )
    with tabs[0]:
        render_overview(events, malformed_lines)
    with tabs[1]:
        render_karma_ledger(events)
    with tabs[2]:
        render_viveka_timeline(events)
    with tabs[3]:
        render_memory_state(events)
    with tabs[4]:
        render_repair_runs(events)


if __name__ == "__main__":
    main()
