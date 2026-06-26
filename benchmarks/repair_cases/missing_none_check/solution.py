"""Buggy name normalization."""


def normalize_name(name: str | None) -> str:
    """Return a display-ready name."""
    return name.strip().title()
