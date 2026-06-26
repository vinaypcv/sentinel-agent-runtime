"""LLM-backed unified-diff generation without code execution."""

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class PatchGenerationError(ValueError):
    """Raised when patch generation or static diff validation fails."""


@dataclass(frozen=True, slots=True)
class PatchGenerationRequest:
    """Inputs supplied to a patch generator."""

    failing_file_content: str
    pytest_traceback: str
    test_file_content: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate all repair inputs without executing any supplied content."""
        required_fields = {
            "failing_file_content": self.failing_file_content,
            "pytest_traceback": self.pytest_traceback,
            "test_file_content": self.test_file_content,
        }
        for field_name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.constraints, tuple):
            raise TypeError("constraints must be a tuple of strings")
        if not all(
            isinstance(constraint, str) and constraint.strip()
            for constraint in self.constraints
        ):
            raise ValueError("constraints must contain non-empty strings")

    @classmethod
    def create(
        cls,
        *,
        failing_file_content: str,
        pytest_traceback: str,
        test_file_content: str,
        constraints: Sequence[str] = (),
    ) -> "PatchGenerationRequest":
        """Create a request while normalizing constraints to an immutable tuple."""
        return cls(
            failing_file_content=failing_file_content,
            pytest_traceback=pytest_traceback,
            test_file_content=test_file_content,
            constraints=tuple(constraints),
        )


class PatchGenerator(ABC):
    """Abstract interface for generating statically validated unified diffs."""

    _SENSITIVE_PATH_MARKERS = (
        ".env",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "secret",
        "token",
    )
    _SENSITIVE_ASSIGNMENT_MARKERS = (
        "api_key",
        "apikey",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    )

    def generate(
        self,
        *,
        failing_file_content: str,
        pytest_traceback: str,
        test_file_content: str,
        constraints: Sequence[str] = (),
    ) -> str:
        """Generate a validated patch from the four repair inputs."""
        request = PatchGenerationRequest.create(
            failing_file_content=failing_file_content,
            pytest_traceback=pytest_traceback,
            test_file_content=test_file_content,
            constraints=constraints,
        )
        return self.generate_patch(request)

    @abstractmethod
    def generate_patch(self, request: PatchGenerationRequest) -> str:
        """Generate and statically validate a unified diff without executing it."""

    @classmethod
    def validate_unified_diff(cls, output: str) -> str:
        """Validate that output is only a syntactically plausible unified diff."""
        if not isinstance(output, str) or not output.strip():
            raise PatchGenerationError("patch output must be a non-empty string")

        patch = output.strip()
        if "```" in patch:
            raise PatchGenerationError("patch output must not use Markdown code fences")
        if "--- " not in patch or "+++ " not in patch or "@@" not in patch:
            raise PatchGenerationError(
                "patch output must contain '--- ', '+++ ', and '@@' diff markers"
            )

        lines = patch.splitlines()
        if not lines[0].startswith(("diff --git ", "--- ")):
            raise PatchGenerationError(
                "patch output must start with 'diff --git' or '---' headers"
            )

        index = 0
        file_sections = 0
        while index < len(lines):
            index = cls._validate_file_section(lines, index)
            file_sections += 1

        if file_sections == 0:
            raise PatchGenerationError("patch output does not contain a file diff")
        cls._validate_patch_safety(lines)
        return f"{patch}\n"

    @classmethod
    def _validate_file_section(cls, lines: list[str], index: int) -> int:
        """Validate one git-style or plain unified-diff file section."""
        if lines[index].startswith("diff --git "):
            if not re.fullmatch(r"diff --git a/\S+ b/\S+", lines[index]):
                raise PatchGenerationError("invalid git diff header")
            index += 1
            while index < len(lines) and not lines[index].startswith("--- "):
                if not cls._is_git_metadata(lines[index]):
                    raise PatchGenerationError(
                        f"non-diff content found before file headers: {lines[index]}"
                    )
                index += 1

        if index + 1 >= len(lines):
            raise PatchGenerationError("patch is missing unified diff file headers")
        if not re.fullmatch(r"--- \S+(?:\t.*)?", lines[index]):
            raise PatchGenerationError("patch is missing a valid original-file header")
        if not re.fullmatch(r"\+\+\+ \S+(?:\t.*)?", lines[index + 1]):
            raise PatchGenerationError("patch is missing a valid updated-file header")
        index += 2

        hunk_count = 0
        while index < len(lines) and not lines[index].startswith(("diff --git ", "--- ")):
            if not lines[index].startswith("@@ "):
                raise PatchGenerationError(
                    f"expected a unified diff hunk header, found: {lines[index]}"
                )
            match = re.fullmatch(
                r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: .*)?",
                lines[index],
            )
            if match is None:
                raise PatchGenerationError("invalid unified diff hunk header")
            old_count = int(match.group(1)) if match.group(1) is not None else 1
            new_count = int(match.group(2)) if match.group(2) is not None else 1
            hunk_count += 1
            index += 1

            old_seen = 0
            new_seen = 0
            while old_seen < old_count or new_seen < new_count:
                if index >= len(lines):
                    raise PatchGenerationError(
                        "unified diff hunk ended before its declared line counts"
                    )
                line = lines[index]
                if line.startswith(" "):
                    old_seen += 1
                    new_seen += 1
                elif line.startswith("+"):
                    new_seen += 1
                elif line.startswith("-"):
                    old_seen += 1
                elif line.startswith("\\"):
                    index += 1
                    continue
                else:
                    raise PatchGenerationError(
                        f"non-diff content found in hunk: {line}"
                    )
                if old_seen > old_count or new_seen > new_count:
                    raise PatchGenerationError(
                        "unified diff hunk exceeds its declared line counts"
                    )
                index += 1

            if index < len(lines) and lines[index].startswith("\\"):
                index += 1

        if hunk_count == 0:
            raise PatchGenerationError("patch is missing a unified diff hunk header")
        return index

    @staticmethod
    def _is_git_metadata(line: str) -> bool:
        """Return whether a line is permitted git diff metadata."""
        prefixes = (
            "index ",
            "new file mode ",
            "deleted file mode ",
            "old mode ",
            "new mode ",
            "similarity index ",
            "dissimilarity index ",
            "rename from ",
            "rename to ",
            "copy from ",
            "copy to ",
        )
        return line.startswith(prefixes)

    @classmethod
    def _validate_patch_safety(cls, lines: list[str]) -> None:
        """Reject generated diffs that target sensitive or out-of-repo paths."""
        for line in lines:
            if line.startswith(("diff --git ", "--- ", "+++ ")):
                cls._validate_safe_diff_header(line)
            if cls._adds_secret_like_assignment(line):
                raise PatchGenerationError(
                    "patch output appears to add a hardcoded secret-like assignment"
                )

    @classmethod
    def _validate_safe_diff_header(cls, line: str) -> None:
        """Validate one diff header path without requiring a repository object."""
        paths = (
            line.removeprefix("diff --git ").split()
            if line.startswith("diff --git ")
            else (line[4:].split("\t", maxsplit=1)[0].strip(),)
        )
        for raw_path in paths:
            if raw_path == "/dev/null":
                continue
            normalized = cls._normalize_diff_path(raw_path)
            lowered = normalized.lower()
            if any(marker in lowered for marker in cls._SENSITIVE_PATH_MARKERS):
                raise PatchGenerationError(
                    f"patch output targets a sensitive path: {normalized}"
                )

    @staticmethod
    def _normalize_diff_path(raw_path: str) -> str:
        """Normalize a generated diff path and reject repository escapes."""
        path = raw_path
        if path.startswith(("a/", "b/")):
            path = path[2:]
        if not path or path.startswith("/") or "\\" in path:
            raise PatchGenerationError(f"patch output path is outside repo: {raw_path}")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise PatchGenerationError(f"patch output path is outside repo: {raw_path}")
        return "/".join(parts)

    @classmethod
    def _adds_secret_like_assignment(cls, line: str) -> bool:
        """Return True when an added line appears to introduce a credential."""
        if not line.startswith("+") or line.startswith("+++"):
            return False

        normalized = line[1:].strip().lower()
        if not normalized or normalized.startswith(("#", '"""', "'''")):
            return False
        if "=" not in normalized and ":" not in normalized:
            return False

        key = normalized.split("=", maxsplit=1)[0].split(":", maxsplit=1)[0].strip()
        return any(marker in key for marker in cls._SENSITIVE_ASSIGNMENT_MARKERS)


class GeminiPatchGenerator(PatchGenerator):
    """Generate unified diffs with the Gemini API through google-genai."""

    _INSTRUCTIONS = (
        "You are a secure software repair patch generator. Return only one unified "
        "diff. Do not use Markdown fences, explanations, shell commands, or prose. "
        "Do not execute code. Make the smallest change needed to address the pytest "
        "failure while respecting every supplied constraint. Do not modify .env, "
        "secrets, credentials, API keys, tokens, or files outside the repository."
    )

    def __init__(self, *, model: str | None = None) -> None:
        """Initialize the generator without importing the SDK or making a request."""
        selected_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = selected_model

    def generate_patch(self, request: PatchGenerationRequest) -> str:
        """Request a diff from Gemini and statically validate it before returning."""
        if not isinstance(request, PatchGenerationRequest):
            raise TypeError("request must be a PatchGenerationRequest")

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise PatchGenerationError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

        try:
            from google import genai
        except ImportError as error:
            raise PatchGenerationError(
                "The 'google-genai' package is required to use GeminiPatchGenerator"
            ) from error

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=self._build_prompt(request),
            )
        except Exception as error:
            raise self._classify_provider_error(error) from error

        self._raise_for_provider_refusal(response)
        output_text = self._extract_response_text(response)
        if not output_text:
            raise PatchGenerationError("Gemini response was empty")
        return self.validate_unified_diff(output_text)

    @classmethod
    def _build_prompt(cls, request: PatchGenerationRequest) -> str:
        """Render a delimited repair request without executing supplied content."""
        constraints = (
            "\n".join(f"- {constraint}" for constraint in request.constraints)
            or "- No additional constraints supplied."
        )
        return (
            f"{cls._INSTRUCTIONS}\n\n"
            "FAILING FILE CONTENT\n"
            "<failing_file>\n"
            f"{request.failing_file_content}\n"
            "</failing_file>\n\n"
            "PYTEST TRACEBACK\n"
            "<pytest_traceback>\n"
            f"{request.pytest_traceback}\n"
            "</pytest_traceback>\n\n"
            "TEST FILE CONTENT\n"
            "<test_file>\n"
            f"{request.test_file_content}\n"
            "</test_file>\n\n"
            "CONSTRAINTS\n"
            f"{constraints}"
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract Gemini response text across SDK response shapes."""
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text.strip()
        return ""

    @staticmethod
    def _classify_provider_error(error: Exception) -> PatchGenerationError:
        """Convert provider exceptions into clear repair-loop errors."""
        text = f"{type(error).__name__}: {error}"
        lowered = text.lower()
        if any(marker in lowered for marker in ("quota", "rate limit", "429")):
            return PatchGenerationError(f"Gemini quota/rate limit error: {text}")
        if "resource_exhausted" in lowered:
            return PatchGenerationError(f"Gemini quota/rate limit error: {text}")
        if any(marker in lowered for marker in ("permission", "api key", "unauth")):
            return PatchGenerationError(f"Gemini authentication error: {text}")
        return PatchGenerationError(f"Gemini provider error: {text}")

    @staticmethod
    def _raise_for_provider_refusal(response: Any) -> None:
        """Raise a clear error when Gemini reports safety refusal metadata."""
        candidates = getattr(response, "candidates", None)
        if candidates is None:
            return
        for candidate in candidates:
            finish_reason = str(getattr(candidate, "finish_reason", "")).lower()
            if any(marker in finish_reason for marker in ("safety", "blocked", "prohibit")):
                raise PatchGenerationError(
                    f"Gemini provider refusal: finish_reason={finish_reason}"
                )


class MockPatchGenerator(PatchGenerator):
    """Deterministic patch generator for tests and offline workflows."""

    def __init__(self, patch: str) -> None:
        """Initialize the mock with one fixed candidate patch."""
        self.patch = patch
        self.requests: list[PatchGenerationRequest] = []

    def generate_patch(self, request: PatchGenerationRequest) -> str:
        """Record the request and return its statically validated fixed patch."""
        if not isinstance(request, PatchGenerationRequest):
            raise TypeError("request must be a PatchGenerationRequest")
        self.requests.append(request)
        return self.validate_unified_diff(self.patch)
