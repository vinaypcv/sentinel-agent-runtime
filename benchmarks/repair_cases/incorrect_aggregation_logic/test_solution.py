"""Regression test for aggregation totals."""

from solution import Sale, total_by_category


def test_total_by_category_accumulates_duplicate_categories() -> None:
    """Amounts in the same category should be summed."""
    sales: list[Sale] = [
        {"category": "books", "amount": 10},
        {"category": "books", "amount": 5},
        {"category": "music", "amount": 7},
    ]

    assert total_by_category(sales) == {"books": 15, "music": 7}  # nosec B101
