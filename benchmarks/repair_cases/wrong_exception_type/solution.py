"""Buggy integer parsing."""


def parse_positive_int(value: str) -> int:
    """Parse a positive integer from text."""
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError("value must be positive")
    return parsed
