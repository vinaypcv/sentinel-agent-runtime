"""Regression tests for safe expression calculation."""

import pytest
from solution import calculate


def test_calculate_supports_basic_arithmetic() -> None:
    """Simple arithmetic should still work."""
    assert calculate("2 + 3 * 4") == 14  # nosec B101


def test_calculate_rejects_unsafe_expression() -> None:
    """Non-arithmetic expressions should be rejected."""
    with pytest.raises(ValueError):
        calculate("__import__('os').getcwd()")
