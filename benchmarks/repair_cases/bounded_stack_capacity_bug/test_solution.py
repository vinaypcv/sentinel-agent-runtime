"""Regression test for bounded stack capacity."""

from solution import BoundedStack


def test_stack_discards_oldest_item_at_capacity() -> None:
    """The stack should retain only the newest items."""
    stack = BoundedStack(capacity=2)

    stack.push(1)
    stack.push(2)
    stack.push(3)

    assert stack.items == [2, 3]  # nosec B101
