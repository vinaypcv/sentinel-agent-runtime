"""Buggy expression calculator."""


def calculate(expression: str) -> int:
    """Calculate a simple arithmetic expression."""
    return int(eval(expression))
