"""Regression tests for exception type."""

import pytest
from solution import parse_positive_int


def test_parse_positive_int_raises_value_error_for_non_positive() -> None:
    """Invalid user input should raise ValueError."""
    with pytest.raises(ValueError):
        parse_positive_int("0")
