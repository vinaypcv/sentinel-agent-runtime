"""Live Gemini patch-generation smoke test.

This test is intentionally opt-in because it calls an external provider.
"""

import os

import pytest

from brahman_os.repair.llm_patch_generator import GeminiPatchGenerator

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1"
    or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="set RUN_LIVE_API_TESTS=1 and GEMINI_API_KEY or GOOGLE_API_KEY to run",
)


def test_live_gemini_patch_generator_returns_valid_diff() -> None:
    """Gemini should return a statically validated unified diff when enabled."""
    pytest.importorskip("google.genai")

    patch = GeminiPatchGenerator().generate(
        failing_file_content="VALUE = 1\n",
        pytest_traceback="AssertionError: assert 1 == 2",
        test_file_content=(
            "from example import VALUE\n\n\n"
            "def test_value() -> None:\n"
            "    assert VALUE == 2\n"
        ),
        constraints=(
            "Modify only example.py.",
            "Change VALUE from 1 to 2.",
            "Return one unified diff only.",
        ),
    )

    assert "--- " in patch
    assert "+++ " in patch
    assert "@@" in patch
    assert "```" not in patch
