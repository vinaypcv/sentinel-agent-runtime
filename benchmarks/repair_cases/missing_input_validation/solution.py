"""Buggy account creation."""


def create_account(username: str) -> dict[str, str]:
    """Create a minimal account payload."""
    return {"username": username}
