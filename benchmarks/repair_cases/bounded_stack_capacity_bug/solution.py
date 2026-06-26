"""Buggy bounded stack implementation."""


class BoundedStack:
    """A stack that should retain only the newest items up to capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[int] = []

    def push(self, item: int) -> None:
        """Push an item onto the stack."""
        self.items.append(item)

    def pop(self) -> int:
        """Pop the newest item."""
        return self.items.pop()
