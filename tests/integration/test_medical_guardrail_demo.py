"""Integration tests for the medical policy guardrail example."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.schemas import SafetyDecision

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPOSITORY_ROOT / "examples" / "medical_policy_guardrail" / "run.py"


def load_demo_module() -> ModuleType:
    """Load the runnable example as a Python module."""
    spec = importlib.util.spec_from_file_location("medical_policy_guardrail_demo", DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load medical policy guardrail demo")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_evaluates_claims_and_logs_every_decision(tmp_path: Path) -> None:
    """The complete demo should pass one claim and block both unsafe claims."""
    module = load_demo_module()
    ledger_path = tmp_path / "karma.jsonl"

    explanations = module.run_demo(ledger_path)
    decisions = {
        explanation["claim_id"]: explanation["decision"]["decision"]
        for explanation in explanations
    }
    events = KarmaLedger(ledger_path).get_events()

    assert decisions == {
        "safe_grounded_claim": "pass",
        "unsafe_dosage_claim": "block",
        "unsafe_aspirin_claim": "block",
    }
    assert len(events) == 3
    assert events[0].viveka_decision is SafetyDecision.PASS
    assert all(json.dumps(explanation) for explanation in explanations)


def test_demo_command_prints_three_json_records(tmp_path: Path) -> None:
    """The documented command should execute and print explainable JSON."""
    ledger_path = tmp_path / "command-karma.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(DEMO_PATH),
            "--ledger-path",
            str(ledger_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    assert len(records) == 3
    assert {record["decision"]["decision"] for record in records} == {"pass", "block"}
    assert len(KarmaLedger(ledger_path).get_events()) == 3
