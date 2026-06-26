"""Buggy range helper."""


def inclusive_range(start: int, end: int) -> list[int]:
    """Return a list that should include both endpoints."""
    return list(range(start, end))
