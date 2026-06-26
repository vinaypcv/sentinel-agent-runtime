"""Viveka composite safety scoring."""

from dataclasses import dataclass

from brahman_os.schemas import SafetyDecision


@dataclass(frozen=True, slots=True)
class VivekaScore:
    """Explainable result of composing epistemic and symbolic scores."""

    score: float
    decision: SafetyDecision
    explanation: str


class VivekaScorer:
    """Compose pratyaksha and anumana evidence into a pass/block decision."""

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize the scorer with a normalized decision threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = float(threshold)

    def score(self, pratyaksha_score: float, anumana_score: float) -> VivekaScore:
        """Return the composite score, decision, and scoring explanation."""
        self._validate_score(pratyaksha_score, name="pratyaksha_score")
        self._validate_score(anumana_score, name="anumana_score")

        if anumana_score == 0.0:
            score = pratyaksha_score * 0.2
            rule_effect = "critical contradiction reduced vector evidence to twenty percent"
        elif anumana_score == 1.0:
            score = max(pratyaksha_score, 0.72)
            rule_effect = "all matched critical rules passed and established a 0.72 floor"
        else:
            score = pratyaksha_score
            rule_effect = "no critical rule matched, so vector evidence was used unchanged"

        decision = (
            SafetyDecision.PASS if score >= self.threshold else SafetyDecision.BLOCK
        )
        threshold_relation = (
            "at or above" if decision is SafetyDecision.PASS else "below"
        )
        explanation = (
            f"Viveka score {score:.6f} is {threshold_relation} "
            f"the {self.threshold:.6f} threshold; {rule_effect}."
        )
        return VivekaScore(score=score, decision=decision, explanation=explanation)

    @staticmethod
    def _validate_score(value: float, *, name: str) -> None:
        """Reject values outside the normalized score range."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
