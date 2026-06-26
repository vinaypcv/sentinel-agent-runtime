"""Regression test for mutation while iterating."""

from solution import remove_even


def test_remove_even_does_not_skip_adjacent_values() -> None:
    """Adjacent even values should all be removed."""
    assert remove_even([2, 4, 6, 7]) == [7]  # nosec B101
