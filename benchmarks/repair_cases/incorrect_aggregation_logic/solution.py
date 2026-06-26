"""Buggy sales aggregation."""

from typing import TypedDict


class Sale(TypedDict):
    """One sale record."""

    category: str
    amount: int


def total_by_category(sales: list[Sale]) -> dict[str, int]:
    """Return total amount by category."""
    totals: dict[str, int] = {}
    for sale in sales:
        totals[sale["category"]] = sale["amount"]
    return totals
