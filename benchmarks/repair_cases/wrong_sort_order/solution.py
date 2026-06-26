"""Buggy score sorting."""


def top_scores(scores: list[int]) -> list[int]:
    """Return the three highest scores."""
    return sorted(scores)[:3]
