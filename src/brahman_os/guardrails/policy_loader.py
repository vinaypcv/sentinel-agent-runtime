"""Load and validate symbolic guardrail policies from YAML."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from brahman_os.schemas import (
    PolicyDocument,
    PolicyRule,
    PolicyRuleType,
    SafetyDecision,
)


class PolicyLoadError(ValueError):
    """Raised when a policy file cannot be parsed or validated."""


class PolicyLoader:
    """Load strict PolicyDocument objects from YAML files."""

    def __init__(self, policy_directory: str | Path = "policies") -> None:
        """Initialize the loader with the default policy directory."""
        self.policy_directory = Path(policy_directory)

    def load(self, path: str | Path) -> PolicyDocument:
        """Load one YAML file and return its validated policy document."""
        policy_path = Path(path)
        if not policy_path.is_file():
            raise PolicyLoadError(f"Policy file does not exist: {policy_path}")

        try:
            raw_document = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            location = ""
            if error.problem_mark is not None:
                location = (
                    f" at line {error.problem_mark.line + 1}, "
                    f"column {error.problem_mark.column + 1}"
                )
            problem = getattr(error, "problem", None) or str(error)
            raise PolicyLoadError(
                f"Malformed YAML in policy file {policy_path}{location}: {problem}"
            ) from error
        except OSError as error:
            raise PolicyLoadError(f"Unable to read policy file {policy_path}: {error}") from error

        if not isinstance(raw_document, Mapping):
            raise PolicyLoadError(
                f"Policy file {policy_path} must contain a YAML mapping at its root"
            )

        try:
            return self._build_document(raw_document)
        except (ValidationError, TypeError, ValueError) as error:
            raise PolicyLoadError(
                f"Invalid policy file {policy_path}: {self._format_error(error)}"
            ) from error

    def load_all(self, directory: str | Path | None = None) -> tuple[PolicyDocument, ...]:
        """Load every ``.yaml`` and ``.yml`` policy in a directory."""
        policy_directory = Path(directory) if directory is not None else self.policy_directory
        if not policy_directory.is_dir():
            raise PolicyLoadError(f"Policy directory does not exist: {policy_directory}")

        paths = sorted(
            (*policy_directory.glob("*.yaml"), *policy_directory.glob("*.yml")),
            key=lambda path: path.name,
        )
        if not paths:
            raise PolicyLoadError(f"No YAML policy files found in {policy_directory}")
        return tuple(self.load(path) for path in paths)

    def _build_document(self, raw_document: Mapping[str, Any]) -> PolicyDocument:
        """Normalize YAML containers and validate a policy document."""
        raw_rules = raw_document.get("rules")
        if not isinstance(raw_rules, list):
            raise TypeError("rules must be a YAML list")

        rules = tuple(
            self._build_rule(raw_rule, index=index)
            for index, raw_rule in enumerate(raw_rules)
        )
        document_data = self._document_data(raw_document)
        document_data["rules"] = rules
        document_data["provenance"] = self._string_tuple(
            raw_document.get(
                "provenance",
                (f"policy:{raw_document['policy_id']}",)
                if "policy_id" in raw_document
                else (),
            ),
            field_name="provenance",
        )
        return PolicyDocument.model_validate(document_data)

    def _build_rule(self, raw_rule: object, *, index: int) -> PolicyRule:
        """Normalize and validate one YAML rule."""
        if not isinstance(raw_rule, Mapping):
            raise TypeError(f"rules.{index} must be a YAML mapping")

        rule_data = self._rule_data(raw_rule, index=index)
        try:
            rule_data["rule_type"] = PolicyRuleType(rule_data["rule_type"])
        except KeyError as error:
            raise ValueError(f"rules.{index}.rule_type is required") from error
        except ValueError as error:
            supported = ", ".join(rule_type.value for rule_type in PolicyRuleType)
            raise ValueError(
                f"rules.{index}.rule_type must be one of: {supported}"
            ) from error

        try:
            rule_data["decision"] = SafetyDecision(rule_data["decision"])
        except KeyError as error:
            raise ValueError(f"rules.{index}.decision is required") from error
        except ValueError as error:
            raise ValueError(
                f"rules.{index}.decision must be pass or block"
            ) from error

        rule_data["evidence"] = self._string_tuple(
            rule_data.get("evidence", ()),
            field_name=f"rules.{index}.evidence",
        )
        try:
            return PolicyRule.model_validate(rule_data)
        except ValidationError as error:
            raise ValueError(
                f"rules.{index}: {self._format_validation_error(error)}"
            ) from error

    @staticmethod
    def _document_data(raw_document: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize compact and expanded document fields."""
        if "policy_id" not in raw_document:
            return {
                key: value
                for key, value in raw_document.items()
                if key not in {"rules", "provenance"}
            }

        policy_id = raw_document["policy_id"]
        domain = raw_document.get("domain")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain must be a non-empty string")
        return {
            "source_policy_id": policy_id,
            "name": policy_id,
            "version": policy_id,
            "domain": domain,
        }

    @staticmethod
    def _rule_data(raw_rule: Mapping[str, Any], *, index: int) -> dict[str, Any]:
        """Normalize compact and expanded rule fields."""
        if "type" not in raw_rule:
            return dict(raw_rule)

        rule_id = raw_rule.get("id")
        rule_type = raw_rule.get("type")
        entity = raw_rule.get("entity")
        severity = raw_rule.get("severity")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(f"rules.{index}.id is required")
        if not isinstance(entity, str) or not entity:
            raise ValueError(f"rules.{index}.entity is required")
        if not isinstance(severity, str) or not severity:
            raise ValueError(f"rules.{index}.severity is required")

        parameters: dict[str, str | int | float | bool]
        if rule_type == PolicyRuleType.NUMERIC_BOUND.value:
            parameters = {
                "subject": entity,
                "operator": raw_rule.get("operator"),
                "value": raw_rule.get("value"),
                "unit": PolicyLoader._unit_from_field(raw_rule.get("field")),
                "field": raw_rule.get("field"),
                "severity": severity,
            }
            condition = (
                f"{entity} {raw_rule.get('field')} "
                f"{raw_rule.get('operator')} {raw_rule.get('value')}"
            )
        elif rule_type == PolicyRuleType.EXCLUSION.value:
            parameters = {
                "condition": entity,
                "severity": severity,
            }
            condition = f"Exclude {entity}"
        elif rule_type == PolicyRuleType.REQUIRED_EVIDENCE.value:
            parameters = {
                "evidence_type": entity,
                "severity": severity,
            }
            condition = f"Require evidence of {entity}"
        elif rule_type == PolicyRuleType.FORBIDDEN_TOOL.value:
            parameters = {
                "tool": entity,
                "severity": severity,
            }
            condition = f"Forbid tool {entity}"
        else:
            parameters = {"severity": severity}
            condition = f"Validate {entity}"

        return {
            "rule_id": rule_id,
            "description": condition,
            "rule_type": rule_type,
            "condition": condition,
            "decision": SafetyDecision.BLOCK,
            "confidence": 1.0,
            "parameters": parameters,
            "evidence": (),
        }

    @staticmethod
    def _unit_from_field(field: object) -> str:
        """Infer a unit suffix from a compact numeric field name."""
        if not isinstance(field, str) or not field:
            raise ValueError("numeric_bound field must be a non-empty string")
        _, separator, unit = field.rpartition("_")
        return unit if separator and unit else "unspecified"

    @staticmethod
    def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
        """Normalize a YAML string sequence while preserving strict item typing."""
        if isinstance(value, tuple):
            values = value
        elif isinstance(value, list):
            values = tuple(value)
        else:
            raise TypeError(f"{field_name} must be a YAML list of strings")
        if not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{field_name} entries must be non-empty strings")
        return values

    @classmethod
    def _format_error(cls, error: Exception) -> str:
        """Render concise Pydantic and loader errors."""
        if isinstance(error, ValidationError):
            return cls._format_validation_error(error)
        return str(error)

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        """Render Pydantic errors with dotted field paths."""
        messages = []
        for item in error.errors(include_url=False):
            location = ".".join(str(part) for part in item["loc"]) or "document"
            messages.append(f"{location}: {item['msg']}")
        return "; ".join(messages)
