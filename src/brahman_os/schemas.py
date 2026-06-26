"""Strict, serializable data contracts shared by Brahman-OS components."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

NormalizedScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class StrictSchema(BaseModel):
    """Base configuration for immutable, strict Brahman-OS schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SafetyDecision(StrEnum):
    """Possible outcomes produced by a safety evaluation."""

    PASS = "pass"
    BLOCK = "block"


class MemoryEventType(StrEnum):
    """Supported persistent-memory mutations."""

    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"


class PolicyRuleType(StrEnum):
    """Supported symbolic policy rule categories."""

    NUMERIC_BOUND = "numeric_bound"
    EXCLUSION = "exclusion"
    REQUIRED_EVIDENCE = "required_evidence"
    FORBIDDEN_TOOL = "forbidden_tool"


class RepairStatus(StrEnum):
    """Lifecycle states for a repair attempt."""

    PROPOSED = "proposed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MemoryEvent(StrictSchema):
    """A durable mutation submitted to the memory substrate."""

    event_id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    event_type: MemoryEventType
    content: NonEmptyString
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    timestamp: AwareDatetime = Field(default_factory=utc_now)


class MemoryReadResult(StrictSchema):
    """The explainable result of reading a memory record."""

    memory_id: UUID
    found: bool
    content: str | None
    embedding: tuple[float, ...] | None = None
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    access_count: NonNegativeInt = 0
    decay_factor: NormalizedScore = 1.0
    retrieved_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_content_presence(self) -> "MemoryReadResult":
        """Require retrievable data exactly when a memory record was found."""
        if self.found and not self.content and self.embedding is None:
            raise ValueError(
                "content must be present when found is true unless embedding is provided"
            )
        if not self.found and (self.content is not None or self.embedding is not None):
            raise ValueError("content and embedding must be null when found is false")
        return self


class PolicyRule(StrictSchema):
    """A single symbolic policy rule and its required decision."""

    rule_id: NonEmptyString
    description: NonEmptyString
    rule_type: PolicyRuleType | None = None
    condition: NonEmptyString
    decision: SafetyDecision
    confidence: NormalizedScore
    parameters: dict[NonEmptyString, str | int | float | bool] = Field(
        default_factory=dict
    )
    evidence: tuple[NonEmptyString, ...] = ()
    enabled: bool = True

    @model_validator(mode="after")
    def validate_rule_parameters(self) -> "PolicyRule":
        """Require parameters appropriate to the selected policy rule type."""
        if self.rule_type is None:
            return self

        required_parameters = {
            PolicyRuleType.NUMERIC_BOUND: {"subject", "operator", "value", "unit"},
            PolicyRuleType.EXCLUSION: {"condition"},
            PolicyRuleType.REQUIRED_EVIDENCE: {"evidence_type"},
            PolicyRuleType.FORBIDDEN_TOOL: {"tool"},
        }
        missing = required_parameters[self.rule_type] - self.parameters.keys()
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"{self.rule_type.value} rule requires parameters: {missing_names}"
            )

        if self.rule_type is PolicyRuleType.NUMERIC_BOUND:
            operator = self.parameters["operator"]
            value = self.parameters["value"]
            if operator not in {"<", "<=", "==", ">=", ">"}:
                raise ValueError("numeric_bound operator must be one of <, <=, ==, >=, >")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("numeric_bound value must be numeric")
        return self


class PolicyDocument(StrictSchema):
    """A versioned collection of symbolic policy rules."""

    policy_id: UUID = Field(default_factory=uuid4)
    source_policy_id: NonEmptyString | None = None
    name: NonEmptyString
    version: NonEmptyString
    domain: NonEmptyString | None = None
    rules: tuple[PolicyRule, ...] = Field(min_length=1)
    provenance: tuple[NonEmptyString, ...] = ()
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "PolicyDocument":
        """Ensure document updates do not predate creation."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class VivekaDecision(StrictSchema):
    """An explainable composite safety decision."""

    decision_id: UUID = Field(default_factory=uuid4)
    decision: SafetyDecision
    score: NormalizedScore
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    violated_rules: tuple[NonEmptyString, ...] = ()
    reasons: tuple[NonEmptyString, ...] = ()
    timestamp: AwareDatetime = Field(default_factory=utc_now)


class KarmaEvent(StrictSchema):
    """An immutable audit event for an agent or repair action."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    action_id: NonEmptyString
    goal_id: NonEmptyString
    action_type: NonEmptyString
    input_summary: NonEmptyString
    proposed_action: NonEmptyString
    viveka_decision: SafetyDecision
    evidence: tuple[NonEmptyString, ...] = ()
    rollback_snapshot_id: UUID | None
    result: str | None
    status: NonEmptyString


class TestRunResult(StrictSchema):
    """Structured output from one verification test command."""

    run_id: UUID = Field(default_factory=uuid4)
    command: tuple[NonEmptyString, ...] = Field(min_length=1)
    decision: SafetyDecision
    passed: bool
    exit_code: int = Field(strict=True)
    total_tests: NonNegativeInt
    passed_tests: NonNegativeInt
    failed_tests: NonNegativeInt
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    violated_rules: tuple[NonEmptyString, ...] = ()
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_seconds: NonNegativeFloat
    traceback_summary: tuple[str, ...] = ()
    report_path: str | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def duration(self) -> float:
        """Return the test duration in seconds."""
        return self.duration_seconds

    @model_validator(mode="after")
    def validate_test_totals(self) -> "TestRunResult":
        """Ensure timing and reported test counts are internally consistent."""
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.passed_tests + self.failed_tests != self.total_tests:
            raise ValueError("passed_tests and failed_tests must equal total_tests")
        if self.passed != (self.exit_code == 0 and self.failed_tests == 0):
            raise ValueError("passed must agree with exit_code and failed_tests")
        return self


class PatchProposal(StrictSchema):
    """An auditable source-code patch proposed for verification."""

    proposal_id: UUID = Field(default_factory=uuid4)
    repair_attempt_id: UUID
    summary: NonEmptyString
    diff: NonEmptyString
    files_changed: tuple[NonEmptyString, ...] = Field(min_length=1)
    decision: SafetyDecision
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    violated_rules: tuple[NonEmptyString, ...] = ()
    created_at: AwareDatetime = Field(default_factory=utc_now)


class RepairAttempt(StrictSchema):
    """A single traceable attempt to repair a diagnosed issue."""

    attempt_id: UUID = Field(default_factory=uuid4)
    issue_id: NonEmptyString
    status: RepairStatus
    decision: SafetyDecision
    confidence: NormalizedScore
    provenance: tuple[NonEmptyString, ...] = ()
    evidence: tuple[NonEmptyString, ...] = ()
    violated_rules: tuple[NonEmptyString, ...] = ()
    patch_proposal_id: UUID | None = None
    test_run_id: UUID | None = None
    started_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> "RepairAttempt":
        """Ensure a completed repair does not finish before it starts."""
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class ExplainableDecision(StrictSchema):
    """Backward-compatible generic safety evaluation result."""

    decision: SafetyDecision
    score: NormalizedScore
    reasons: tuple[NonEmptyString, ...] = ()
    metadata: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)


LedgerEvent = KarmaEvent
