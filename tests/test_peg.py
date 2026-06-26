"""Tests for the Pramana Epistemic Guardrail."""

import json

import pytest
import torch

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail
from brahman_os.guardrails.viveka import VivekaScorer
from brahman_os.schemas import PolicyRule, SafetyDecision


def critical_rule(rule_id: str) -> PolicyRule:
    """Build an enabled critical policy rule."""
    return PolicyRule(
        rule_id=rule_id,
        description=f"Critical rule {rule_id}",
        condition=f"validate:{rule_id}",
        decision=SafetyDecision.PASS,
        confidence=1.0,
    )


def test_grounded_claim_passes() -> None:
    """Aligned vectors with passing rules should pass PEG."""
    peg = PramanaEpistemicGuardrail(threshold=0.7)
    rule = critical_rule("grounded")

    result = peg.evaluate(
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([1.0, 0.0, 1.0]),
        rules=(rule,),
        rule_results={"grounded": True},
    )

    assert result.decision is SafetyDecision.PASS
    assert result.score == 1.0
    assert "anumana_score=1.000000" in result.evidence


def test_contradiction_blocks_even_with_grounded_vector() -> None:
    """A critical contradiction should strongly suppress vector evidence."""
    peg = PramanaEpistemicGuardrail(threshold=0.7)
    rule = critical_rule("no-contradiction")

    result = peg.evaluate(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        rules=(rule,),
        rule_results={"no-contradiction": False},
    )

    assert result.decision is SafetyDecision.BLOCK
    assert result.score == 0.2


def test_neutral_claim_uses_vector_score() -> None:
    """No symbolic match should leave the normalized vector score unchanged."""
    peg = PramanaEpistemicGuardrail(threshold=0.7)

    result = peg.evaluate(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )

    assert result.score == 0.5
    assert result.decision is SafetyDecision.BLOCK
    assert "no critical rule matched" in result.evidence


def test_violated_rules_appear_in_output() -> None:
    """Contradicted rule identifiers should be preserved in the decision."""
    peg = PramanaEpistemicGuardrail()
    rules = (critical_rule("rule-a"), critical_rule("rule-b"))

    result = peg.evaluate(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        rules=rules,
        rule_results={"rule-a": True, "rule-b": False},
    )

    assert result.violated_rules == ("rule-b",)
    assert "rule rule-b contradicted" in result.evidence


def test_decision_serializes_to_explainable_json() -> None:
    """PEG output should serialize with scores, evidence, and explanation."""
    result = PramanaEpistemicGuardrail().evaluate(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    )

    payload = json.loads(result.model_dump_json())

    assert payload["decision"] == "pass"
    assert payload["score"] == 1.0
    assert payload["evidence"]
    assert payload["reasons"]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_viveka_rejects_invalid_thresholds(threshold: float) -> None:
    """Decision thresholds outside the normalized range should fail."""
    with pytest.raises(ValueError, match="threshold must be between"):
        VivekaScorer(threshold=threshold)


@pytest.mark.parametrize(
    ("generated", "context", "error_type", "message"),
    [
        (
            torch.ones(1, 2),
            torch.ones(1, 2),
            ValueError,
            "one-dimensional",
        ),
        (
            torch.ones(2),
            torch.ones(3),
            ValueError,
            "identical shapes",
        ),
        (
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            TypeError,
            "floating-point dtype",
        ),
        (
            torch.tensor([1.0, float("nan")]),
            torch.ones(2),
            ValueError,
            "finite values",
        ),
    ],
)
def test_peg_rejects_invalid_vectors(
    generated: torch.Tensor,
    context: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    """Malformed grounding vectors should fail before scoring."""
    with pytest.raises(error_type, match=message):
        PramanaEpistemicGuardrail().evaluate(generated, context)


def test_peg_rejects_unknown_rule_results() -> None:
    """Rule results must reference enabled rules supplied to the verifier."""
    with pytest.raises(ValueError, match="unknown or disabled rules"):
        PramanaEpistemicGuardrail().evaluate(
            torch.ones(2),
            torch.ones(2),
            rules=(critical_rule("known"),),
            rule_results={"unknown": False},
        )
