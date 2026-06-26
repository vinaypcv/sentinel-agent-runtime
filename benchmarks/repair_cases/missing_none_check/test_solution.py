"""Regression tests for optional name normalization."""

from solution import normalize_name


def test_normalize_name_handles_none() -> None:
    """None should normalize to an empty string."""
    assert normalize_name(None) == ""  # nosec B101


def test_normalize_name_formats_text() -> None:
    """Text should still be trimmed and title-cased."""
    assert normalize_name("  ada lovelace ") == "Ada Lovelace"  # nosec B101
