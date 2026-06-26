"""Integration tests for the autonomous software repair example."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from brahman_os.ledger.karma_ledger import KarmaLedger

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPOSITORY_ROOT / "examples" / "software_repair" / "run.py"


def load_demo_module() -> ModuleType:
    """Load the repair example as a module."""
    spec = importlib.util.spec_from_file_location("software_repair_demo", DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load software repair demo")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_one_json_object(text: str) -> dict[str, object]:
    """Parse stdout only when it contains exactly one JSON object."""
    decoder = json.JSONDecoder()
    payload, end_index = decoder.raw_decode(text)
    assert text[end_index:].strip() == ""
    assert isinstance(payload, dict)
    return payload


def test_repair_demo_completes_and_audits_every_stage(tmp_path: Path) -> None:
    """The demo should fail, repair, verify, approve, and log each stage."""
    module = load_demo_module()

    summary = module.run_demo(tmp_path, provider="mock")
    events = KarmaLedger(tmp_path / "karma-ledger.jsonl").get_events()

    assert summary["initial_test"]["passed"] is False
    assert summary["final_test"]["passed"] is True
    assert summary["peg_approval"]["decision"] == "pass"
    assert summary["finalized"] is True
    assert summary["patch"]["provider"] == "mock"
    assert summary["patch"]["generator"] == "MockPatchGenerator"
    assert summary["patch_validation"]["passed"] is True
    assert summary["rollback"]["rolled_back"] is False
    assert summary["error"] is None
    assert {check["name"] for check in summary["static_checks"]} == {
        "ruff",
        "mypy",
        "bandit",
    }
    assert all(check["passed"] for check in summary["static_checks"])
    assert len(events) == 5


def test_documented_repair_command_prints_json(tmp_path: Path) -> None:
    """The documented command should produce a successful JSON summary."""
    workspace = tmp_path / "repair-workspace"
    completed = subprocess.run(
        [
            sys.executable,
            str(DEMO_PATH),
            "--provider",
            "mock",
            "--workspace",
            str(workspace),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    summary = load_one_json_object(completed.stdout)

    assert summary["finalized"] is True
    assert summary["final_test"]["passed"] is True
    assert summary["patch"]["provider"] == "mock"
    assert summary["patch"]["rolled_back"] is False
    assert summary["patch_validation"]["passed"] is True
    assert summary["rollback"]["rolled_back"] is False
    assert summary["error"] is None
    assert summary["ledger"]["event_count"] == 5
