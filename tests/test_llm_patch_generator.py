"""Tests for statically validated LLM patch generation."""

from pathlib import Path

import pytest

from brahman_os.repair.llm_patch_generator import (
    GeminiPatchGenerator,
    MockPatchGenerator,
    PatchGenerationError,
    PatchGenerationRequest,
    PatchGenerator,
)

VALID_PATCH = (
    "diff --git a/src/example.py b/src/example.py\n"
    "index 1234567..89abcde 100644\n"
    "--- a/src/example.py\n"
    "+++ b/src/example.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
)


def test_mock_generator_returns_unified_diff_and_records_input() -> None:
    """The mock should return only a validated diff and retain its request."""
    generator = MockPatchGenerator(VALID_PATCH)

    patch = generator.generate(
        failing_file_content="VALUE = 1\n",
        pytest_traceback="AssertionError: expected 2",
        test_file_content="def test_value():\n    assert VALUE == 2\n",
        constraints=("Modify only src/example.py.",),
    )

    assert patch == VALID_PATCH
    assert len(generator.requests) == 1
    assert generator.requests[0].constraints == ("Modify only src/example.py.",)


def test_patch_generator_is_abstract() -> None:
    """The base generator should not be directly instantiable."""
    with pytest.raises(TypeError):
        PatchGenerator()


@pytest.mark.parametrize(
    "patch",
    [
        "Here is the patch:\n" + VALID_PATCH,
        "```diff\n" + VALID_PATCH + "```\n",
        "--- a/src/example.py\n+++ b/src/example.py\n+VALUE = 2\n",
        "--- a/src/example.py\n+++ b/src/example.py\n@@ invalid @@\n+VALUE = 2\n",
        (
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
            "explanation after patch\n"
        ),
    ],
)
def test_mock_generator_rejects_non_diff_or_malformed_output(patch: str) -> None:
    """Static validation should reject prose, fences, and malformed headers."""
    request = PatchGenerationRequest.create(
        failing_file_content="VALUE = 1\n",
        pytest_traceback="AssertionError",
        test_file_content="assert VALUE == 2\n",
    )

    with pytest.raises(PatchGenerationError):
        MockPatchGenerator(patch).generate_patch(request)


def test_request_rejects_missing_inputs() -> None:
    """Every repair input should be present before generation begins."""
    with pytest.raises(ValueError, match="pytest_traceback"):
        PatchGenerationRequest.create(
            failing_file_content="VALUE = 1\n",
            pytest_traceback="",
            test_file_content="assert VALUE == 2\n",
        )


def test_gemini_generator_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini generation should fail clearly without requiring real test secrets."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    request = PatchGenerationRequest.create(
        failing_file_content="VALUE = 1\n",
        pytest_traceback="AssertionError",
        test_file_content="assert VALUE == 2\n",
    )

    with pytest.raises(PatchGenerationError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        GeminiPatchGenerator().generate_patch(request)


def test_gemini_generator_reads_model_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live provider should default from GEMINI_MODEL without an API call."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    assert GeminiPatchGenerator().model == "gemini-test-model"
    assert GeminiPatchGenerator(model="explicit-model").model == "explicit-model"


@pytest.mark.parametrize(
    "patch",
    [
        (
            "--- a/.env\n"
            "+++ b/.env\n"
            "@@ -0,0 +1 @@\n"
            "+DEBUG=true\n"
        ),
        (
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -0,0 +1 @@\n"
            '+API_KEY = "placeholder"\n'
        ),
        (
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -0,0 +1 @@\n"
            "+VALUE = 2\n"
        ),
    ],
)
def test_diff_validation_rejects_sensitive_or_external_patches(patch: str) -> None:
    """Generated diffs should not touch secrets or paths outside the repo."""
    with pytest.raises(PatchGenerationError):
        MockPatchGenerator(patch).generate(
            failing_file_content="VALUE = 1\n",
            pytest_traceback="AssertionError",
            test_file_content="assert VALUE == 2\n",
        )


def test_gemini_provider_errors_are_clear() -> None:
    """Provider exception classification should distinguish quota failures."""
    error = GeminiPatchGenerator._classify_provider_error(RuntimeError("429 quota"))

    assert isinstance(error, PatchGenerationError)
    assert "quota/rate limit" in str(error)


def test_validation_does_not_execute_generated_code(tmp_path: Path) -> None:
    """Patch text should remain inert during static validation."""
    marker = tmp_path / "executed.txt"
    inert_patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1 +1,2 @@\n"
        " VALUE = 1\n"
        f"+open({str(marker)!r}, 'w').write('executed')\n"
    )

    result = MockPatchGenerator(inert_patch).generate(
        failing_file_content="VALUE = 1\n",
        pytest_traceback="AssertionError",
        test_file_content="assert VALUE == 2\n",
    )

    assert result == inert_patch
    assert not marker.exists()
