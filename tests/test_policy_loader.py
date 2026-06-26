"""Tests for YAML policy loading and validation."""

import json
from pathlib import Path

import pytest

from brahman_os.guardrails.policy_loader import PolicyLoader, PolicyLoadError
from brahman_os.schemas import PolicyRuleType


def test_valid_medical_policy_loads() -> None:
    """The repository medical policy should load into strict Pydantic models."""
    document = PolicyLoader().load("policies/medical_safety.yaml")

    assert document.source_policy_id == "medical_safety_v1"
    assert document.name == "medical_safety_v1"
    assert document.version == "medical_safety_v1"
    assert document.domain == "clinical"
    assert document.provenance == ("policy:medical_safety_v1",)
    assert len(document.rules) == 3
    assert document.rules[0].rule_type is PolicyRuleType.NUMERIC_BOUND
    assert document.rules[0].parameters["value"] == 50
    assert document.rules[0].parameters["unit"] == "mg"
    assert document.rules[0].parameters["severity"] == "critical"
    assert document.rules[1].rule_type is PolicyRuleType.EXCLUSION

    payload = json.loads(document.model_dump_json())
    assert payload["source_policy_id"] == "medical_safety_v1"
    assert payload["domain"] == "clinical"
    assert payload["rules"][0]["parameters"]["operator"] == "<="


def test_malformed_yaml_fails_with_file_and_location(tmp_path: Path) -> None:
    """Malformed YAML should report the file and parser location clearly."""
    path = tmp_path / "malformed.yaml"
    path.write_text("name: broken\nrules: [\n", encoding="utf-8")

    with pytest.raises(PolicyLoadError) as error:
        PolicyLoader().load(path)

    message = str(error.value)
    assert str(path) in message
    assert "Malformed YAML" in message
    assert "line" in message


def test_missing_rule_fields_fail_validation(tmp_path: Path) -> None:
    """Missing required rule fields should include the indexed field path."""
    path = tmp_path / "missing.yaml"
    path.write_text(
        """
name: Incomplete Policy
version: "1.0"
rules:
  - rule_id: missing-description
    rule_type: exclusion
    decision: block
    confidence: 1.0
    parameters:
      condition: prohibited diagnosis
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PolicyLoadError) as error:
        PolicyLoader().load(path)

    message = str(error.value)
    assert "rules.0" in message
    assert "description" in message
    assert "condition" in message
