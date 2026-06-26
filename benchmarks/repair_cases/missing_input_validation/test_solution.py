"""Regression tests for account input validation."""

import pytest
from solution import create_account


def test_create_account_rejects_blank_username() -> None:
    """Blank usernames should be rejected."""
    with pytest.raises(ValueError):
        create_account("   ")


def test_create_account_trims_username() -> None:
    """Accepted usernames should be normalized."""
    assert create_account("  ada ") == {"username": "ada"}  # nosec B101
