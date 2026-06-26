"""Tests for the local Streamlit dashboard data helpers."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_dashboard_module() -> ModuleType:
    """Load the Streamlit dashboard module from its script path."""
    module_path = REPOSITORY_ROOT / "dashboard" / "app.py"
    spec = importlib.util.spec_from_file_location("dashboard_app", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load dashboard app")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_jsonl_loads_valid_rows_and_counts_malformed(tmp_path: Path) -> None:
    """Dashboard JSONL loading should keep valid rows and skip bad lines."""
    app = load_dashboard_module()
    ledger = tmp_path / "karma.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-06-26T00:00:00Z",
                        "action_type": "peg_decision",
                        "viveka_decision": "pass",
                        "result": json.dumps({"score": 0.91}),
                    }
                ),
                "not-json",
                json.dumps(["not", "an", "object"]),
            ]
        ),
        encoding="utf-8",
    )

    rows, malformed = app.read_jsonl(ledger)

    assert malformed == 2
    assert len(rows) == 1
    assert rows[0]["_line"] == 1
    assert rows[0]["_source"].endswith("karma.jsonl")


def test_load_events_extracts_scores_and_sorts_by_timestamp(tmp_path: Path) -> None:
    """Loaded events should include extracted Viveka scores and stable ordering."""
    app = load_dashboard_module()
    ledger = tmp_path / "karma.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-06-26T00:00:02Z",
                        "action_type": "later",
                        "viveka_decision": "block",
                        "evidence": ["viveka_score=0.250000"],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-06-26T00:00:01Z",
                        "action_type": "earlier",
                        "viveka_decision": "pass",
                        "result": json.dumps({"score": 0.85}),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    frame, malformed = app.load_events([ledger])

    assert malformed == 0
    assert frame["action_type"].tolist() == ["earlier", "later"]
    assert frame["viveka_score"].tolist() == [0.85, 0.25]


def test_dashboard_helpers_handle_failure_cases() -> None:
    """Helper functions should degrade safely for malformed or missing data."""
    app = load_dashboard_module()
    frame = pd.DataFrame(
        [
            {"action_type": "post_patch_test_run", "status": "failed", "result": "{bad"},
            {
                "action_type": "post_patch_test_run",
                "status": "failed",
                "result": json.dumps({"failed_tests": 3}),
            },
            {"action_type": "memory_write", "status": "completed", "result": None},
        ]
    )

    assert app.parse_json_object("{bad") == {}
    assert app.extract_viveka_score(pd.Series({"evidence": ["no score here"]})) is None
    assert app.count_failed_tests(frame) == 3
    assert len(app.latest_activity(frame, r"memory|write")) == 1
    assert app.select_columns(frame, ["status", "missing"]).columns.tolist() == ["status"]
