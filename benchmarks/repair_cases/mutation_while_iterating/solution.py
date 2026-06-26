"""Buggy list filtering."""


def remove_even(numbers: list[int]) -> list[int]:
    """Remove even numbers from the list."""
    for number in numbers:
        if number % 2 == 0:
            numbers.remove(number)
    return numbers
