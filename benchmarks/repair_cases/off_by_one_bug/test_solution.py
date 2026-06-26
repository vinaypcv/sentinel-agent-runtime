"""Regression test for inclusive range behavior."""

from solution import inclusive_range


def test_inclusive_range_includes_end_value() -> None:
    """The end value should be included."""
    assert inclusive_range(2, 5) == [2, 3, 4, 5]  # nosec B101
