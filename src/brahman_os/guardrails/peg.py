"""Pramana Epistemic Guardrail vector and symbolic validation."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from brahman_os.guardrails.viveka import VivekaScorer
from brahman_os.schemas import PolicyRule, VivekaDecision


class PramanaEpistemicGuardrail:
    """Evaluate grounding and symbolic rules before permitting a claim."""

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize PEG with the Viveka pass threshold."""
        self.viveka = VivekaScorer(threshold=threshold)

    def evaluate(
        self,
        generated_vector: Tensor,
        context_vector: Tensor,
        *,
        rules: Sequence[PolicyRule] = (),
        rule_results: Mapping[str, bool | None] | None = None,
    ) -> VivekaDecision:
        """Return an explainable safety decision for vector and rule evidence."""
        pratyaksha_score = self.pratyaksha_score(generated_vector, context_vector)
        anumana_score, violated_rules, symbolic_evidence = self.anumana_score(
            rules=rules,
            rule_results=rule_results or {},
        )
        viveka = self.viveka.score(pratyaksha_score, anumana_score)
        evidence = (
            f"pratyaksha_score={pratyaksha_score:.6f}",
            f"anumana_score={anumana_score:.6f}",
            f"viveka_score={viveka.score:.6f}",
            *symbolic_evidence,
        )
        confidence = (pratyaksha_score + anumana_score) / 2.0

        return VivekaDecision(
            decision=viveka.decision,
            score=viveka.score,
            confidence=confidence,
            provenance=("peg:cosine_similarity", "peg:symbolic_rules", "viveka"),
            evidence=evidence,
            violated_rules=violated_rules,
            reasons=(viveka.explanation,),
        )

    @staticmethod
    def pratyaksha_score(generated_vector: Tensor, context_vector: Tensor) -> float:
        """Return cosine similarity normalized from ``[-1, 1]`` to ``[0, 1]``."""
        generated = PramanaEpistemicGuardrail._validate_vector(
            generated_vector,
            name="generated_vector",
        )
        context = PramanaEpistemicGuardrail._validate_vector(
            context_vector,
            name="context_vector",
        )
        if generated.shape != context.shape:
            raise ValueError("generated_vector and context_vector must have identical shapes")

        if torch.linalg.vector_norm(generated) == 0 or torch.linalg.vector_norm(context) == 0:
            return 0.5

        similarity = F.cosine_similarity(generated, context, dim=0)
        normalized = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
        return float(normalized.detach().cpu())

    @staticmethod
    def anumana_score(
        *,
        rules: Sequence[PolicyRule],
        rule_results: Mapping[str, bool | None],
    ) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
        """Validate enabled critical rules and report score, violations, and evidence."""
        enabled_rules = {rule.rule_id: rule for rule in rules if rule.enabled}
        unknown_rules = set(rule_results) - set(enabled_rules)
        if unknown_rules:
            unknown = ", ".join(sorted(unknown_rules))
            raise ValueError(f"rule_results contains unknown or disabled rules: {unknown}")

        matched_results = {
            rule_id: result
            for rule_id, result in rule_results.items()
            if result is not None
        }
        violated = tuple(
            rule_id for rule_id, result in matched_results.items() if result is False
        )
        if violated:
            evidence = tuple(f"rule {rule_id} contradicted" for rule_id in violated)
            return 0.0, violated, evidence

        if matched_results:
            passed = tuple(rule_id for rule_id, result in matched_results.items() if result)
            evidence = tuple(f"rule {rule_id} passed" for rule_id in passed)
            return 1.0, (), evidence

        return 0.5, (), ("no critical rule matched",)

    @staticmethod
    def _validate_vector(vector: Tensor, *, name: str) -> Tensor:
        """Validate a finite, one-dimensional floating-point vector."""
        if vector.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not vector.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")
        return vector
