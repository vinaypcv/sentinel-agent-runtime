"""Regression test for score ordering."""

from solution import top_scores


def test_top_scores_returns_highest_first() -> None:
    """The highest scores should be returned in descending order."""
    assert top_scores([10, 50, 20, 40]) == [50, 40, 20]  # nosec B101
