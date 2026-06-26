"""Top-level Brahman-OS action orchestration."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import torch
from torch import Tensor

from brahman_os.guardrails.peg import PramanaEpistemicGuardrail
from brahman_os.guardrails.policy_loader import PolicyLoader
from brahman_os.ledger.karma_ledger import KarmaLedger
from brahman_os.memory.akasha_store import AkashaStore
from brahman_os.schemas import (
    KarmaEvent,
    PolicyDocument,
    SafetyDecision,
    VivekaDecision,
)


class ToolAdapter(Protocol):
    """Execution boundary for actions approved by PEG."""

    def execute(
        self,
        action_type: str,
        arguments: Mapping[str, object],
    ) -> object:
        """Execute one approved action and return its result."""


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """Validated input for one runtime action decision."""

    goal_id: str
    action_type: str
    input_summary: str
    proposed_action: str
    generated_vector: Tensor
    tool_arguments: Mapping[str, object] = field(default_factory=dict)
    query_vector: Tensor | None = None
    rule_results: Mapping[str, bool | None] = field(default_factory=dict)
    policy_path: str | Path | None = None
    action_id: str = field(default_factory=lambda: str(uuid4()))
    delta_time: float = 1.0
    rollback_snapshot_id: UUID | None = None

    def __post_init__(self) -> None:
        """Validate request metadata and vector contracts."""
        text_fields = {
            "action_id": self.action_id,
            "goal_id": self.goal_id,
            "action_type": self.action_type,
            "input_summary": self.input_summary,
            "proposed_action": self.proposed_action,
        }
        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        self._validate_vector(self.generated_vector, name="generated_vector")
        if self.query_vector is not None:
            self._validate_vector(self.query_vector, name="query_vector")
            if self.query_vector.shape != self.generated_vector.shape:
                raise ValueError(
                    "query_vector and generated_vector must have identical shapes"
                )
        if not math.isfinite(self.delta_time) or self.delta_time < 0.0:
            raise ValueError("delta_time must be finite and non-negative")
        if not all(isinstance(key, str) and key for key in self.tool_arguments):
            raise ValueError("tool argument keys must be non-empty strings")
        if not all(
            isinstance(key, str)
            and key
            and (value is None or isinstance(value, bool))
            for key, value in self.rule_results.items()
        ):
            raise ValueError("rule_results must map rule IDs to bool or None")

    @staticmethod
    def _validate_vector(vector: Tensor, *, name: str) -> None:
        """Require finite one-dimensional floating-point vectors."""
        if not isinstance(vector, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if vector.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not vector.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")


class BrahmanRuntime:
    """Coordinate memory retrieval, PEG verification, execution, and audit."""

    def __init__(
        self,
        *,
        memory: AkashaStore,
        verifier: PramanaEpistemicGuardrail,
        ledger: KarmaLedger,
        policy_loader: PolicyLoader,
        tool_adapter: ToolAdapter,
        policy_path: str | Path | None = None,
    ) -> None:
        """Initialize the runtime with explicit production dependencies."""
        self.memory = memory
        self.verifier = verifier
        self.ledger = ledger
        self.policy_loader = policy_loader
        self.tool_adapter = tool_adapter
        self.policy_path = Path(policy_path) if policy_path is not None else None

    def execute_action(self, action_request: ActionRequest) -> dict[str, object]:
        """Verify, audit, and conditionally execute one proposed action."""
        if not isinstance(action_request, ActionRequest):
            raise TypeError("action_request must be an ActionRequest")

        ledger_event_ids: list[str] = []
        try:
            query_vector = (
                action_request.query_vector
                if action_request.query_vector is not None
                else action_request.generated_vector
            )
            memory_result = self.memory.read(
                query_vector,
                delta_time=action_request.delta_time,
            )
            context_vector = self._context_vector(
                memory_result.embedding,
                action_request.generated_vector,
            )
            policy = self._load_policy(action_request)
            decision = self.verifier.evaluate(
                action_request.generated_vector,
                context_vector,
                rules=policy.rules if policy is not None else (),
                rule_results=action_request.rule_results,
            )
        except Exception as error:
            event = self._log_exception(
                action_request,
                error,
                stage="verification",
            )
            ledger_event_ids.append(str(event.event_id))
            return self._response(
                action_request=action_request,
                status="error",
                executed=False,
                decision=None,
                memory=None,
                policy=None,
                result=None,
                error=error,
                ledger_event_ids=ledger_event_ids,
            )

        decision_event = self._log_decision(action_request, decision)
        ledger_event_ids.append(str(decision_event.event_id))
        if decision.decision is SafetyDecision.BLOCK:
            return self._response(
                action_request=action_request,
                status="blocked",
                executed=False,
                decision=decision,
                memory=memory_result.model_dump(mode="json"),
                policy=policy,
                result=None,
                error=None,
                ledger_event_ids=ledger_event_ids,
            )

        try:
            raw_result = self.tool_adapter.execute(
                action_request.action_type,
                action_request.tool_arguments,
            )
            result = self._json_safe(raw_result)
        except Exception as error:
            event = self._log_exception(
                action_request,
                error,
                stage="execution",
                rollback_snapshot_id=action_request.rollback_snapshot_id,
            )
            ledger_event_ids.append(str(event.event_id))
            return self._response(
                action_request=action_request,
                status="error",
                executed=True,
                decision=decision,
                memory=memory_result.model_dump(mode="json"),
                policy=policy,
                result=None,
                error=error,
                ledger_event_ids=ledger_event_ids,
            )

        result_event = self._log_result(action_request, decision, result)
        ledger_event_ids.append(str(result_event.event_id))
        return self._response(
            action_request=action_request,
            status="completed",
            executed=True,
            decision=decision,
            memory=memory_result.model_dump(mode="json"),
            policy=policy,
            result=result,
            error=None,
            ledger_event_ids=ledger_event_ids,
        )

    def _load_policy(self, request: ActionRequest) -> PolicyDocument | None:
        """Load the request policy, runtime default policy, or no rules."""
        selected_path = (
            Path(request.policy_path)
            if request.policy_path is not None
            else self.policy_path
        )
        return self.policy_loader.load(selected_path) if selected_path is not None else None

    @staticmethod
    def _context_vector(
        embedding: tuple[float, ...] | None,
        generated_vector: Tensor,
    ) -> Tensor:
        """Return retrieved context or a fail-closed neutral zero vector."""
        if embedding is None:
            return torch.zeros_like(generated_vector)
        context = torch.tensor(
            embedding,
            dtype=generated_vector.dtype,
            device=generated_vector.device,
        )
        if context.shape != generated_vector.shape:
            raise ValueError("retrieved memory context has incompatible dimensions")
        return context

    def _log_decision(
        self,
        request: ActionRequest,
        decision: VivekaDecision,
    ) -> KarmaEvent:
        """Log the PEG decision before any tool execution."""
        event = KarmaEvent(
            action_id=request.action_id,
            goal_id=request.goal_id,
            action_type=f"{request.action_type}:decision",
            input_summary=request.input_summary,
            proposed_action=request.proposed_action,
            viveka_decision=decision.decision,
            evidence=decision.evidence,
            rollback_snapshot_id=request.rollback_snapshot_id,
            result=decision.model_dump_json(),
            status="approved"
            if decision.decision is SafetyDecision.PASS
            else "blocked",
        )
        self.ledger.log_event(event)
        return event

    def _log_result(
        self,
        request: ActionRequest,
        decision: VivekaDecision,
        result: object,
    ) -> KarmaEvent:
        """Log a successful tool execution result."""
        event = KarmaEvent(
            action_id=request.action_id,
            goal_id=request.goal_id,
            action_type=f"{request.action_type}:result",
            input_summary=request.input_summary,
            proposed_action=request.proposed_action,
            viveka_decision=decision.decision,
            evidence=("Tool adapter execution completed.",),
            rollback_snapshot_id=request.rollback_snapshot_id,
            result=json.dumps(result, sort_keys=True),
            status="completed",
        )
        self.ledger.log_event(event)
        return event

    def _log_exception(
        self,
        request: ActionRequest,
        error: Exception,
        *,
        stage: str,
        rollback_snapshot_id: UUID | None = None,
    ) -> KarmaEvent:
        """Log an exception without exposing a Python traceback to callers."""
        error_text = f"{type(error).__name__}: {error}"
        event = KarmaEvent(
            action_id=request.action_id,
            goal_id=request.goal_id,
            action_type=f"{request.action_type}:{stage}_error",
            input_summary=request.input_summary,
            proposed_action=request.proposed_action,
            viveka_decision=SafetyDecision.BLOCK,
            evidence=(error_text,),
            rollback_snapshot_id=rollback_snapshot_id,
            result=json.dumps(
                {"stage": stage, "error": error_text},
                sort_keys=True,
            ),
            status="error",
        )
        self.ledger.log_event(event)
        return event

    @staticmethod
    def _json_safe(value: object) -> object:
        """Convert arbitrary adapter output into JSON-safe data."""
        return json.loads(json.dumps(value, default=str))

    @staticmethod
    def _response(
        *,
        action_request: ActionRequest,
        status: str,
        executed: bool,
        decision: VivekaDecision | None,
        memory: dict[str, object] | None,
        policy: PolicyDocument | None,
        result: object,
        error: Exception | None,
        ledger_event_ids: list[str],
    ) -> dict[str, object]:
        """Build the explainable JSON response for every runtime outcome."""
        return {
            "action_id": action_request.action_id,
            "goal_id": action_request.goal_id,
            "action_type": action_request.action_type,
            "status": status,
            "executed": executed,
            "decision": decision.model_dump(mode="json") if decision else None,
            "memory": memory,
            "policy_id": policy.source_policy_id if policy else None,
            "result": result,
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
            "ledger_event_ids": ledger_event_ids,
        }


Orchestrator = BrahmanRuntime
