"""Run an end-to-end Brahman-OS medical policy guardrail demonstration."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import operator
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def ensure_project_runtime() -> None:
    """Re-execute with the project virtual environment when dependencies are absent."""
    if importlib.util.find_spec("torch") is not None:
        return

    virtualenv_python = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
    if not virtualenv_python.is_file():
        raise RuntimeError(
            "PyTorch is unavailable. Activate the project virtual environment first."
        )
    completed = subprocess.run(
        [str(virtualenv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(completed.returncode)


ensure_project_runtime()
torch = importlib.import_module("torch")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail  # noqa: E402
from brahman_os.guardrails.policy_loader import PolicyLoader  # noqa: E402
from brahman_os.ledger.karma_ledger import KarmaLedger  # noqa: E402
from brahman_os.memory.akasha_store import AkashaStore  # noqa: E402
from brahman_os.memory.sams import SAMSEmbeddingSubstrate  # noqa: E402
from brahman_os.schemas import (  # noqa: E402
    KarmaEvent,
    PolicyDocument,
    PolicyRule,
    PolicyRuleType,
)

PROTOCOL_SENTENCES = (
    "Compound-X dosage must not exceed 50 mg.",
    "Patients with chronic kidney disease are excluded from Compound-X treatment.",
    "Aspirin is prohibited when Compound-X is administered.",
)

CLAIMS = (
    (
        "safe_grounded_claim",
        "Compound-X dosage is 40 mg for a patient without chronic kidney disease "
        "and with no aspirin.",
    ),
    (
        "unsafe_dosage_claim",
        "Administer Compound-X at a dosage of 75 mg.",
    ),
    (
        "unsafe_aspirin_claim",
        "Administer aspirin together with Compound-X.",
    ),
)

NUMERIC_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    ">=": operator.ge,
    ">": operator.gt,
}


def build_sentence_encoder(texts: Sequence[str]) -> Any:
    """Build an offline SentenceTransformer bag-of-words encoder."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import BoW

    vocabulary = sorted(
        {
            token
            for text in texts
            for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
        }
    )
    return SentenceTransformer(
        modules=[
            BoW(
                vocabulary,
                unknown_word_weight=1.0,
                cumulative_term_frequency=False,
            )
        ],
        device="cpu",
    )


def configure_sams(embedding_dim: int) -> SAMSEmbeddingSubstrate:
    """Initialize a deterministic SAMS contraction for the demonstration."""
    sams = SAMSEmbeddingSubstrate(embedding_dim=embedding_dim, tau=1.0)
    with torch.no_grad():
        sams.sequence_projection.weight.copy_(torch.eye(embedding_dim))
        sams.sequence_projection.bias.zero_()
        sams.gate_projection.weight.zero_()
        sams.gate_projection.bias.zero_()
    sams.eval()
    return sams


def evaluate_symbolic_rules(
    claim: str,
    policy: PolicyDocument,
) -> dict[str, bool | None]:
    """Evaluate the demo's numeric and exclusion policy rules."""
    return {rule.rule_id: evaluate_rule(claim, rule) for rule in policy.rules}


def evaluate_rule(claim: str, rule: PolicyRule) -> bool | None:
    """Evaluate one supported policy rule against a claim."""
    normalized_claim = claim.lower()
    if rule.rule_type is PolicyRuleType.NUMERIC_BOUND:
        return evaluate_numeric_bound(normalized_claim, rule)
    if rule.rule_type is PolicyRuleType.EXCLUSION:
        entity = str(rule.parameters["condition"]).lower()
        if entity not in normalized_claim:
            return None
        return entity_is_negated(normalized_claim, entity)
    return None


def evaluate_numeric_bound(claim: str, rule: PolicyRule) -> bool | None:
    """Evaluate a numeric policy bound when the claim supplies a value."""
    field = str(rule.parameters.get("field", ""))
    unit = str(rule.parameters["unit"])
    field_name = field.removesuffix(f"_{unit}").replace("_", " ")
    pattern = rf"(?:{re.escape(field_name)}\D*)?(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b"
    match = re.search(pattern, claim)
    if match is None:
        return None

    observed = float(match.group(1))
    expected = float(rule.parameters["value"])
    comparison = NUMERIC_OPERATORS[str(rule.parameters["operator"])]
    return comparison(observed, expected)


def entity_is_negated(claim: str, entity: str) -> bool:
    """Return whether a mentioned exclusion entity is explicitly absent."""
    negations = (
        f"no {entity}",
        f"without {entity}",
        f"not taking {entity}",
        f"{entity} absent",
    )
    return any(phrase in claim for phrase in negations)


def run_demo(ledger_path: str | Path) -> list[dict[str, object]]:
    """Execute the complete medical guardrail flow and return explanations."""
    torch.manual_seed(17)
    all_texts = (*PROTOCOL_SENTENCES, *(claim for _, claim in CLAIMS))
    encoder = build_sentence_encoder(all_texts)
    protocol_embeddings = encoder.encode(
        [sentence.lower() for sentence in PROTOCOL_SENTENCES],
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).clone().to(dtype=torch.float32)

    embedding_dim = int(protocol_embeddings.shape[1])
    sams = configure_sams(embedding_dim)
    protocol_output = sams(protocol_embeddings.unsqueeze(0))
    protocol_memory = protocol_output.psi.squeeze(0)

    akasha = AkashaStore(d_model=embedding_dim)
    akasha.write(
        protocol_memory,
        {
            "confidence": 1.0,
            "provenance": ("clinical-protocol", "sentence-transformers:bow"),
            "evidence": PROTOCOL_SENTENCES,
        },
    )
    rollback_snapshot_id = akasha.snapshot()

    policy = PolicyLoader().load(REPOSITORY_ROOT / "policies" / "medical_safety.yaml")
    peg = PramanaEpistemicGuardrail(threshold=0.7)
    ledger = KarmaLedger(ledger_path)
    explanations: list[dict[str, object]] = []

    for label, claim in CLAIMS:
        claim_embedding = encoder.encode(
            claim.lower(),
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).clone().to(dtype=torch.float32)
        claim_psi = sams(claim_embedding.reshape(1, 1, -1)).psi.squeeze(0)
        memory_read = akasha.read(claim_psi)
        if memory_read.embedding is None:
            raise RuntimeError("AkashaStore did not return protocol memory")
        context_vector = torch.tensor(memory_read.embedding, dtype=torch.float32)

        rule_results = evaluate_symbolic_rules(claim, policy)
        decision = peg.evaluate(
            claim_psi,
            context_vector,
            rules=policy.rules,
            rule_results=rule_results,
        )
        action_id = f"medical-claim-{uuid4()}"
        ledger.log_event(
            KarmaEvent(
                action_id=action_id,
                goal_id="medical-policy-guardrail-demo",
                action_type="evaluate_clinical_claim",
                input_summary=claim,
                proposed_action="Accept the clinical claim for downstream use.",
                viveka_decision=decision.decision,
                evidence=decision.evidence,
                rollback_snapshot_id=rollback_snapshot_id,
                result=decision.model_dump_json(),
                status="completed",
            )
        )

        explanation = {
            "claim_id": label,
            "claim": claim,
            "policy_id": policy.source_policy_id,
            "rule_results": rule_results,
            "memory": {
                "memory_id": str(memory_read.memory_id),
                "access_count": memory_read.access_count,
                "confidence": memory_read.confidence,
                "decay_factor": memory_read.decay_factor,
            },
            "decision": decision.model_dump(mode="json"),
            "ledger_action_id": action_id,
        }
        explanations.append(explanation)

    return explanations


def default_ledger_path() -> Path:
    """Return a writable default ledger path outside the source tree."""
    return Path(tempfile.gettempdir()) / "brahman-os-medical-guardrail.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=default_ledger_path(),
        help="JSONL path for KarmaLedger events.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the demonstration and print one explainable JSON object per claim."""
    args = parse_args()
    for explanation in run_demo(args.ledger_path):
        print(json.dumps(explanation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
